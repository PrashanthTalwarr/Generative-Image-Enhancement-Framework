#!/usr/bin/env python3
"""
Medical Image Enhancement GAN Evaluation Script
Comprehensive evaluation for 42% improvement validation and clinical assessment
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from src.models.medical_gan import MedicalGAN
from src.data.data_loader import MedicalImageDataLoader
from src.evaluation.metrics import MedicalImageMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalGANEvaluator:
    """
    Comprehensive evaluation system for Medical Image Enhancement GAN
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.data_loader = None
        self.metrics_evaluator = None
        
        # Results storage
        self.evaluation_results = {
            'model_info': {},
            'dataset_info': {},
            'individual_results': [],
            'aggregate_metrics': {},
            'clinical_assessment': {},
            'target_achievement': {}
        }
        
        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"evaluation_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📊 MedicalGANEvaluator initialized")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained GAN model"""
        try:
            logger.info(f"🧠 Loading model from: {model_path}")
            
            # Initialize model
            self.model = MedicalGAN(
                input_shape=tuple(self.config['model']['input_shape']),
                learning_rate=self.config['model']['learning_rate']
            )
            
            # Load weights based on path type
            if os.path.isdir(model_path):
                # Directory with checkpoints
                best_model_path = os.path.join(model_path, 'best_generator.h5')
                final_model_path = os.path.join(model_path, 'final_generator.h5')
                
                if os.path.exists(best_model_path):
                    self.model.generator.load_weights(best_model_path)
                    model_type = 'best_model'
                    loaded_path = best_model_path
                elif os.path.exists(final_model_path):
                    self.model.generator.load_weights(final_model_path)
                    model_type = 'final_model'
                    loaded_path = final_model_path
                else:
                    # Look for latest checkpoint
                    checkpoints = [f for f in os.listdir(model_path) if f.startswith('generator_epoch_')]
                    if checkpoints:
                        latest = sorted(checkpoints)[-1]
                        epoch = latest.split('_')[-1].split('.')[0]
                        self.model.load_models(model_path, int(epoch))
                        model_type = 'checkpoint'
                        loaded_path = os.path.join(model_path, latest)
                    else:
                        raise FileNotFoundError("No valid model found in directory")
            
            elif model_path.endswith('.h5'):
                # Single model file
                self.model.generator.load_weights(model_path)
                model_type = 'generator_weights'
                loaded_path = model_path
            else:
                raise ValueError(f"Unsupported model path format: {model_path}")
            
            # Store model info
            self.evaluation_results['model_info'] = {
                'model_path': loaded_path,
                'model_type': model_type,
                'generator_parameters': int(self.model.generator.count_params()),
                'discriminator_parameters': int(self.model.discriminator.count_params()),
                'input_shape': self.model.input_shape,
                'load_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Model type: {model_type}")
            logger.info(f"   Generator params: {self.evaluation_results['model_info']['generator_parameters']:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def setup_data_loader(self):
        """Setup data loader for evaluation"""
        try:
            logger.info("📊 Setting up data loader for evaluation...")
            
            self.data_loader = MedicalImageDataLoader(
                data_dir=self.config['data']['data_dir'],
                image_size=tuple(self.config['data']['image_size']),
                batch_size=self.config['evaluation']['batch_size'],
                validation_split=0.0,  # Use all data for evaluation
                seed=self.config['training']['seed']
            )
            
            logger.info("✅ Data loader setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup data loader: {str(e)}")
            return False
    
    def setup_metrics(self):
        """Setup metrics evaluator"""
        try:
            logger.info("📈 Setting up metrics evaluator...")
            self.metrics_evaluator = MedicalImageMetrics()
            logger.info("✅ Metrics evaluator setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup metrics: {str(e)}")
            return False
    
    def evaluate_single_image(self, low_quality: np.ndarray, 
                             high_quality: np.ndarray,
                             image_path: str = None) -> Dict:
        """Evaluate enhancement on a single image"""
        try:
            # Enhance the image
            enhanced = self.model.enhance_image(low_quality)
            
            # Convert images to [0,1] range for metrics
            low_eval = (low_quality.squeeze() + 1) / 2
            high_eval = (high_quality.squeeze() + 1) / 2
            enh_eval = (enhanced.squeeze() + 1) / 2
            
            # Calculate comprehensive metrics
            metrics = self.metrics_evaluator.calculate_diagnostic_quality_index(
                high_eval, low_eval, enh_eval
            )
            
            # Add image path info
            if image_path:
                metrics['image_path'] = image_path
                metrics['image_name'] = os.path.basename(image_path)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating single image: {str(e)}")
            return {}
    
    def evaluate_dataset(self, max_images: Optional[int] = None) -> Dict:
        """Evaluate model on entire dataset"""
        logger.info("🔍 Starting dataset evaluation...")
        
        try:
            # Get all image paths
            all_paths = self.data_loader.discover_images()
            
            if max_images:
                all_paths = all_paths[:max_images]
            
            logger.info(f"📊 Evaluating {len(all_paths)} images...")
            
            # Process images
            individual_results = []
            failed_images = 0
            
            for i, image_path in enumerate(tqdm(all_paths, desc="Evaluating images")):
                try:
                    # Load original image
                    original_image = self.data_loader.load_image(image_path)
                    if original_image is None:
                        failed_images += 1
                        continue
                    
                    # Preprocess
                    high_quality = self.data_loader.preprocess_image(
                        original_image, normalize=True, augment=False
                    )
                    
                    # Create degraded version
                    degraded_image = self.data_loader.create_quality_degradation(
                        (original_image + 1) / 2 if np.min(original_image) < 0 else original_image,
                        degradation_type='mixed'
                    )
                    
                    low_quality = self.data_loader.preprocess_image(
                        degraded_image, normalize=True, augment=False
                    )
                    
                    # Add batch dimension
                    low_quality = np.expand_dims(low_quality, axis=0)
                    high_quality = np.expand_dims(high_quality, axis=0)
                    
                    # Evaluate
                    result = self.evaluate_single_image(low_quality, high_quality, image_path)
                    
                    if result:
                        result['image_index'] = i
                        individual_results.append(result)
                    
                    # Save sample enhanced images (first 10)
                    if i < 10 and result:
                        self.save_sample_enhancement(
                            low_quality, high_quality, result, 
                            f"sample_{i:03d}.png"
                        )
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate {image_path}: {str(e)}")
                    failed_images += 1
                    continue
            
            self.evaluation_results['individual_results'] = individual_results
            
            # Calculate aggregate metrics
            self.calculate_aggregate_metrics()
            
            # Perform clinical assessment
            self.perform_clinical_assessment()
            
            # Check 42% target achievement
            self.check_target_achievement()
            
            # Store dataset info
            self.evaluation_results['dataset_info'] = {
                'total_images': len(all_paths),
                'successfully_evaluated': len(individual_results),
                'failed_images': failed_images,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Dataset evaluation completed")
            logger.info(f"   Successfully evaluated: {len(individual_results)}/{len(all_paths)}")
            logger.info(f"   Failed images: {failed_images}")
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Dataset evaluation failed: {str(e)}")
            return {}
    
    def calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all evaluated images"""
        if not self.evaluation_results['individual_results']:
            return
        
        logger.info("📊 Calculating aggregate metrics...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.evaluation_results['individual_results'])
        
        # Key metrics to aggregate
        key_metrics = [
            'clinical_quality_score', 'improvement_percentage',
            'psnr', 'ssim', 'sharpness_improvement', 'contrast_improvement',
            'edge_preservation', 'noise_reduction'
        ]
        
        aggregate_metrics = {}
        
        for metric in key_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    aggregate_metrics[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': int(len(values))
                    }
        
        # Calculate percentiles for key metrics
        for metric in ['clinical_quality_score', 'improvement_percentage']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    aggregate_metrics[f'{metric}_percentiles'] = {
                        '25th': float(values.quantile(0.25)),
                        '75th': float(values.quantile(0.75)),
                        '90th': float(values.quantile(0.90)),
                        '95th': float(values.quantile(0.95))
                    }
        
        self.evaluation_results['aggregate_metrics'] = aggregate_metrics
        
        logger.info("✅ Aggregate metrics calculated")
    
    def perform_clinical_assessment(self):
        """Perform clinical assessment based on evaluation results"""
        if not self.evaluation_results['individual_results']:
            return
        
        logger.info("🏥 Performing clinical assessment...")
        
        df = pd.DataFrame(self.evaluation_results['individual_results'])
        
        # Clinical thresholds
        thresholds = self.config.get('clinical', {
            'excellent_psnr': 30.0,
            'good_psnr': 25.0,
            'excellent_ssim': 0.90,
            'good_ssim': 0.80,
            'diagnostic_sharpness': 1.2,
            'diagnostic_contrast': 1.15
        })
        
        total_images = len(df)
        
        # Calculate clinical quality flags
        clinical_assessment = {
            'total_evaluated_images': total_images,
            'quality_distribution': {},
            'clinical_suitability': {},
            'improvement_analysis': {},
            'thresholds_used': thresholds
        }
        
        # Quality distribution
        if 'psnr' in df.columns:
            clinical_assessment['quality_distribution']['excellent_psnr'] = {
                'count': int((df['psnr'] >= thresholds['excellent_psnr']).sum()),
                'percentage': float((df['psnr'] >= thresholds['excellent_psnr']).mean() * 100)
            }
            clinical_assessment['quality_distribution']['good_psnr'] = {
                'count': int((df['psnr'] >= thresholds['good_psnr']).sum()),
                'percentage': float((df['psnr'] >= thresholds['good_psnr']).mean() * 100)
            }
        
        if 'ssim' in df.columns:
            clinical_assessment['quality_distribution']['excellent_ssim'] = {
                'count': int((df['ssim'] >= thresholds['excellent_ssim']).sum()),
                'percentage': float((df['ssim'] >= thresholds['excellent_ssim']).mean() * 100)
            }
            clinical_assessment['quality_distribution']['good_ssim'] = {
                'count': int((df['ssim'] >= thresholds['good_ssim']).sum()),
                'percentage': float((df['ssim'] >= thresholds['good_ssim']).mean() * 100)
            }
        
        # Clinical suitability
        if 'clinical_quality_score' in df.columns:
            scores = df['clinical_quality_score']
            clinical_assessment['clinical_suitability'] = {
                'excellent_quality': {
                    'count': int((scores >= 1.5).sum()),
                    'percentage': float((scores >= 1.5).mean() * 100)
                },
                'good_quality': {
                    'count': int((scores >= 1.2).sum()),
                    'percentage': float((scores >= 1.2).mean() * 100)
                },
                'acceptable_quality': {
                    'count': int((scores >= 1.0).sum()),
                    'percentage': float((scores >= 1.0).mean() * 100)
                }
            }
        
        # Improvement analysis
        if 'improvement_percentage' in df.columns:
            improvements = df['improvement_percentage']
            clinical_assessment['improvement_analysis'] = {
                'mean_improvement': float(improvements.mean()),
                'median_improvement': float(improvements.median()),
                'images_above_42_percent': {
                    'count': int((improvements >= 42.0).sum()),
                    'percentage': float((improvements >= 42.0).mean() * 100)
                },
                'images_above_30_percent': {
                    'count': int((improvements >= 30.0).sum()),
                    'percentage': float((improvements >= 30.0).mean() * 100)
                },
                'images_above_20_percent': {
                    'count': int((improvements >= 20.0).sum()),
                    'percentage': float((improvements >= 20.0).mean() * 100)
                }
            }
        
        self.evaluation_results['clinical_assessment'] = clinical_assessment
        
        logger.info("✅ Clinical assessment completed")
    
    def check_target_achievement(self):
        """Check if 42% improvement target is achieved"""
        if not self.evaluation_results['individual_results']:
            return
        
        logger.info("🎯 Checking 42% target achievement...")
        
        df = pd.DataFrame(self.evaluation_results['individual_results'])
        
        target_achievement = {
            'target_percentage': 42.0,
            'target_clinical_score': 1.42,
            'achievement_status': 'not_achieved'
        }
        
        if 'improvement_percentage' in df.columns:
            improvements = df['improvement_percentage']
            
            target_achievement.update({
                'mean_improvement': float(improvements.mean()),
                'median_improvement': float(improvements.median()),
                'max_improvement': float(improvements.max()),
                'images_achieving_target': int((improvements >= 42.0).sum()),
                'percentage_achieving_target': float((improvements >= 42.0).mean() * 100)
            })
            
            # Determine achievement status
            mean_improvement = target_achievement['mean_improvement']
            percentage_achieving = target_achievement['percentage_achieving_target']
            
            if mean_improvement >= 42.0:
                target_achievement['achievement_status'] = 'fully_achieved'
            elif mean_improvement >= 35.0 or percentage_achieving >= 50.0:
                target_achievement['achievement_status'] = 'largely_achieved'
            elif mean_improvement >= 25.0 or percentage_achieving >= 25.0:
                target_achievement['achievement_status'] = 'partially_achieved'
            else:
                target_achievement['achievement_status'] = 'not_achieved'
        
        if 'clinical_quality_score' in df.columns:
            scores = df['clinical_quality_score']
            target_achievement.update({
                'mean_clinical_score': float(scores.mean()),
                'images_above_target_score': int((scores >= 1.42).sum()),
                'percentage_above_target_score': float((scores >= 1.42).mean() * 100)
            })
        
        self.evaluation_results['target_achievement'] = target_achievement
        
        # Log results
        status = target_achievement['achievement_status']
        mean_imp = target_achievement.get('mean_improvement', 0)
        
        if status == 'fully_achieved':
            logger.info(f"🎉 42% TARGET FULLY ACHIEVED! Average improvement: {mean_imp:.1f}%")
        elif status == 'largely_achieved':
            logger.info(f"🎯 42% TARGET LARGELY ACHIEVED! Average improvement: {mean_imp:.1f}%")
        elif status == 'partially_achieved':
            logger.info(f"📈 42% TARGET PARTIALLY ACHIEVED. Average improvement: {mean_imp:.1f}%")
        else:
            logger.info(f"📊 42% TARGET NOT YET ACHIEVED. Average improvement: {mean_imp:.1f}%")
    
    def save_sample_enhancement(self, low_quality: np.ndarray, high_quality: np.ndarray,
                               metrics: Dict, filename: str):
        """Save sample enhancement visualization"""
        try:
            # Generate enhanced image
            enhanced = self.model.enhance_image(low_quality)
            
            # Convert to displayable format [0, 1]
            low_display = (low_quality[0].squeeze() + 1) / 2
            high_display = (high_quality[0].squeeze() + 1) / 2
            enh_display = (enhanced.squeeze() + 1) / 2
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(low_display, cmap='gray')
            axes[0].set_title('Low Quality Input')
            axes[0].axis('off')
            
            axes[1].imshow(enh_display, cmap='gray')
            axes[1].set_title(f'Enhanced\n(+{metrics.get("improvement_percentage", 0):.1f}%)')
            axes[1].axis('off')
            
            axes[2].imshow(high_display, cmap='gray')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            # Add metrics text
            metrics_text = (
                f"PSNR: {metrics.get('psnr', 0):.2f} dB\n"
                f"SSIM: {metrics.get('ssim', 0):.4f}\n"
                f"Clinical Score: {metrics.get('clinical_quality_score', 0):.4f}"
            )
            
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            sample_path = os.path.join(self.output_dir, filename)
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to save sample enhancement: {str(e)}")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("📋 Generating evaluation report...")
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "="*80,
            "MEDICAL IMAGE ENHANCEMENT GAN - EVALUATION REPORT",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.evaluation_results['model_info'].get('model_path', 'Unknown')}",
            ""
        ])
        
        # Dataset Summary
        dataset_info = self.evaluation_results['dataset_info']
        report_lines.extend([
            "📊 DATASET SUMMARY:",
            f"   Total images processed: {dataset_info.get('total_images', 0)}",
            f"   Successfully evaluated: {dataset_info.get('successfully_evaluated', 0)}",
            f"   Failed evaluations: {dataset_info.get('failed_images', 0)}",
            ""
        ])
        
        # Target Achievement
        target_info = self.evaluation_results['target_achievement']
        status = target_info.get('achievement_status', 'unknown')
        mean_imp = target_info.get('mean_improvement', 0)
        
        report_lines.extend([
            "🎯 42% IMPROVEMENT TARGET ASSESSMENT:",
            f"   Achievement Status: {status.upper().replace('_', ' ')}",
            f"   Mean Improvement: {mean_imp:.1f}%",
            f"   Images Achieving Target: {target_info.get('images_achieving_target', 0)} "
            f"({target_info.get('percentage_achieving_target', 0):.1f}%)",
            f"   Maximum Improvement: {target_info.get('max_improvement', 0):.1f}%",
            ""
        ])
        
        # Aggregate Metrics
        if 'aggregate_metrics' in self.evaluation_results:
            agg_metrics = self.evaluation_results['aggregate_metrics']
            
            report_lines.extend([
                "📈 AGGREGATE PERFORMANCE METRICS:",
                ""
            ])
            
            for metric_name, stats in agg_metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    report_lines.append(
                        f"   {metric_name.replace('_', ' ').title()}:"
                    )
                    report_lines.extend([
                        f"      Mean: {stats['mean']:.4f} ± {stats['std']:.4f}",
                        f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]",
                        f"      Median: {stats['median']:.4f}",
                        ""
                    ])
        
        # Clinical Assessment
        if 'clinical_assessment' in self.evaluation_results:
            clinical = self.evaluation_results['clinical_assessment']
            
            report_lines.extend([
                "🏥 CLINICAL ASSESSMENT:",
                ""
            ])
            
            # Quality distribution
            if 'quality_distribution' in clinical:
                report_lines.append("   Quality Distribution:")
                quality_dist = clinical['quality_distribution']
                
                for quality_level, data in quality_dist.items():
                    report_lines.append(
                        f"      {quality_level.replace('_', ' ').title()}: "
                        f"{data['count']} images ({data['percentage']:.1f}%)"
                    )
                report_lines.append("")
            
            # Clinical suitability
            if 'clinical_suitability' in clinical:
                report_lines.append("   Clinical Suitability:")
                suitability = clinical['clinical_suitability']
                
                for level, data in suitability.items():
                    report_lines.append(
                        f"      {level.replace('_', ' ').title()}: "
                        f"{data['count']} images ({data['percentage']:.1f}%)"
                    )
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "💡 RECOMMENDATIONS:",
            ""
        ])
        
        if status == 'fully_achieved':
            report_lines.extend([
                "   🎉 Excellent performance! 42% target fully achieved.",
                "   ✅ Model ready for clinical validation studies.",
                "   📋 Proceed with radiologist assessment trials.",
                ""
            ])
        elif status == 'largely_achieved':
            report_lines.extend([
                "   🎯 Good performance! Close to 42% target.",
                "   🔧 Minor fine-tuning may achieve full target.",
                "   📊 Consider extending training or adjusting hyperparameters.",
                ""
            ])
        elif status == 'partially_achieved':
            report_lines.extend([
                "   📈 Moderate performance. Significant improvement achieved.",
                "   🔄 Continue training with current architecture.",
                "   🎛️  Consider adjusting loss function weights.",
                ""
            ])
        else:
            report_lines.extend([
                "   ⚠️  Target not achieved. Review model architecture.",
                "   🔍 Analyze failed cases for improvement opportunities.",
                "   🎯 Consider different training strategies or data augmentation.",
                ""
            ])
        
        report_lines.extend([
            "="*80,
            f"Report saved to: {self.output_dir}/evaluation_report.txt",
            "="*80
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Also save detailed JSON results
        json_path = os.path.join(self.output_dir, 'detailed_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"📋 Evaluation report saved: {report_path}")
        logger.info(f"📊 Detailed results saved: {json_path}")
        
        # Print summary to console
        print("\n" + report_text)
        
        return report_path

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load evaluation configuration"""
    
    default_config = {
        'model': {
            'input_shape': [256, 256, 1],
            'learning_rate': 0.0002
        },
        'data': {
            'data_dir': 'data/synthetic',
            'image_size': [256, 256]
        },
        'training': {
            'seed': 42
        },
        'evaluation': {
            'batch_size': 1,
            'max_images': None
        },
        'clinical': {
            'excellent_psnr': 30.0,
            'good_psnr': 25.0,
            'excellent_ssim': 0.90,
            'good_ssim': 0.80,
            'diagnostic_sharpness': 1.2,
            'diagnostic_contrast': 1.15
        }
    }
    
    # Load custom config if available
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Merge configurations
        def merge_dicts(default, custom):
            for key, value in custom.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
        
        merge_dicts(default_config, custom_config)
    
    return default_config

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate Medical Image Enhancement GAN for 42% improvement validation'
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (directory or .h5 file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply overrides
        if args.data_dir:
            config['data']['data_dir'] = args.data_dir
        if args.max_images:
            config['evaluation']['max_images'] = args.max_images
        
        # Initialize evaluator
        evaluator = MedicalGANEvaluator(config)
        
        if args.output_dir:
            evaluator.output_dir = args.output_dir
            os.makedirs(evaluator.output_dir, exist_ok=True)
        
        # Load model
        if not evaluator.load_model(args.model_path):
            logger.error("❌ Failed to load model. Exiting.")
            return 1
        
        # Setup components
        if not evaluator.setup_data_loader():
            logger.error("❌ Failed to setup data loader. Exiting.")
            return 1
        
        if not evaluator.setup_metrics():
            logger.error("❌ Failed to setup metrics. Exiting.")
            return 1
        
        # Run evaluation
        logger.info("🚀 Starting comprehensive model evaluation...")
        
        results = evaluator.evaluate_dataset(max_images=config['evaluation']['max_images'])
        
        if not results:
            logger.error("❌ Evaluation failed. Exiting.")
            return 1
        
        # Generate report
        report_path = evaluator.generate_evaluation_report()
        
        # Print final summary
        target_info = results.get('target_achievement', {})
        status = target_info.get('achievement_status', 'unknown')
        mean_improvement = target_info.get('mean_improvement', 0)
        
        print(f"\n🎯 EVALUATION COMPLETED!")
        print(f"📊 Results saved to: {evaluator.output_dir}")
        
        if status == 'fully_achieved':
            print(f"🎉 SUCCESS! 42% improvement target achieved ({mean_improvement:.1f}%)")
            return 0
        else:
            print(f"📈 Current performance: {mean_improvement:.1f}% improvement")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)