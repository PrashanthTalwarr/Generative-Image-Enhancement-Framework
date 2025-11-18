#!/usr/bin/env python3
"""
Medical Image Enhancement GAN Training Script
Target: 42% image quality improvement with clinical validation
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from src.training.trainer import MedicalGANTrainer, load_config
from src.data.synthetic_generator import MedicalSyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(config: dict):
    """Generate synthetic medical data if needed"""
    data_dir = config['data']['data_dir']
    
    logger.info(f"📊 Checking data availability in: {data_dir}")
    
    # Check if data already exists
    if os.path.exists(data_dir):
        image_count = 0
        for root, dirs, files in os.walk(data_dir):
            image_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if image_count >= 100:  # Minimum threshold
            logger.info(f"✅ Found {image_count} existing images. Skipping data generation.")
            return
        else:
            logger.info(f"⚠️  Only {image_count} images found. Generating more...")
    
    # Generate synthetic data
    logger.info("🏭 Generating synthetic medical images...")
    
    generator = MedicalSyntheticDataGenerator(
        image_size=tuple(config['data']['image_size']),
        seed=config['training']['seed']
    )
    
    # Generate dataset
    generator.generate_dataset(
        output_dir=data_dir,
        num_samples_per_type=500,  # 500 samples per type (brain, chest, ct)
        image_types=['brain', 'chest', 'ct'],
        quality_variants=True  # Generate quality degradation pairs
    )
    
    logger.info("✅ Synthetic data generation completed!")

def validate_config(config: dict) -> bool:
    """Validate configuration parameters"""
    logger.info("🔍 Validating configuration...")
    
    required_keys = {
        'model': ['input_shape', 'learning_rate'],
        'data': ['data_dir', 'image_size'],
        'training': ['epochs', 'batch_size'],
        'paths': ['checkpoints', 'results', 'logs']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            logger.error(f"❌ Missing configuration section: {section}")
            return False
        
        for key in keys:
            if key not in config[section]:
                logger.error(f"❌ Missing configuration key: {section}.{key}")
                return False
    
    # Validate data directory
    data_dir = config['data']['data_dir']
    if not os.path.exists(data_dir):
        logger.warning(f"⚠️  Data directory does not exist: {data_dir}")
        logger.info("📝 Will attempt to generate synthetic data...")
    
    # Create required directories
    for path_key in ['checkpoints', 'results', 'logs']:
        path = config['paths'][path_key]
        os.makedirs(path, exist_ok=True)
        logger.info(f"📁 Created directory: {path}")
    
    logger.info("✅ Configuration validation completed!")
    return True

def print_training_info(config: dict):
    """Print training information"""
    print("\n" + "="*80)
    print("🧠 MEDICAL IMAGE ENHANCEMENT GAN TRAINING")
    print("="*80)
    print(f"🎯 TARGET: 42% Image Quality Improvement")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Model info
    print("🔧 MODEL CONFIGURATION:")
    print(f"   Input Shape: {config['model']['input_shape']}")
    print(f"   Learning Rate: {config['model']['learning_rate']}")
    print(f"   Architecture: U-Net Generator + PatchGAN Discriminator")
    print()
    
    # Data info
    print("📊 DATA CONFIGURATION:")
    print(f"   Data Directory: {config['data']['data_dir']}")
    print(f"   Image Size: {config['data']['image_size']}")
    print(f"   Validation Split: {config['data']['validation_split']*100}%")
    print()
    
    # Training info
    print("🏋️  TRAINING CONFIGURATION:")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Target Clinical Score: {config['training']['target_clinical_score']}")
    print(f"   Early Stopping Patience: {config['training']['patience']}")
    print()
    
    # Paths info
    print("📁 OUTPUT PATHS:")
    print(f"   Checkpoints: {config['paths']['checkpoints']}")
    print(f"   Results: {config['paths']['results']}")  
    print(f"   Logs: {config['paths']['logs']}")
    print()
    
    # Clinical validation
    if 'clinical' in config:
        print("🏥 CLINICAL VALIDATION TARGETS:")
        print(f"   Excellent PSNR: ≥{config['clinical']['excellent_psnr']} dB")
        print(f"   Excellent SSIM: ≥{config['clinical']['excellent_ssim']}")
        print(f"   Diagnostic Sharpness: ≥{config['clinical']['diagnostic_sharpness']}x")
        print(f"   Diagnostic Contrast: ≥{config['clinical']['diagnostic_contrast']}x")
        print()
    
    print("="*80)
    print()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train Medical Image Enhancement GAN for 42% quality improvement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Training parameters (override config)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to training data directory')
    parser.add_argument('--generate-data', action='store_true',
                       help='Force generation of synthetic data')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    
    # Quick training options
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test (5 epochs, small dataset)')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    # GPU options
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        logger.info(f"🖥️  Using GPU device: {args.gpu}")
    
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.learning_rate:
            config['model']['learning_rate'] = args.learning_rate
        if args.data_dir:
            config['data']['data_dir'] = args.data_dir
        if args.use_wandb:
            config['use_wandb'] = True
        
        # Quick test configuration
        if args.quick_test:
            logger.info("🏃 Quick test mode enabled")
            config['training']['epochs'] = 5
            config['training']['batch_size'] = 4
            config['training']['save_interval'] = 2
            config['training']['sample_interval'] = 2
            config['training']['validation_interval'] = 2
        
        # Mixed precision
        if args.mixed_precision:
            logger.info("⚡ Mixed precision training enabled")
            # Enable mixed precision
            import tensorflow as tf
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("❌ Configuration validation failed!")
            return 1
        
        # Generate synthetic data if needed or requested
        if args.generate_data or not os.path.exists(config['data']['data_dir']):
            generate_synthetic_data(config)
        
        # Print training information
        print_training_info(config)
        
        # Initialize trainer
        logger.info("🚀 Initializing Medical GAN Trainer...")
        trainer = MedicalGANTrainer(config)
        
        # Setup training components
        logger.info("🔧 Setting up training components...")
        trainer.setup()
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"🔄 Resuming training from: {args.resume_from}")
            # Implementation for resuming would go here
        
        # Start training
        logger.info("🎯 Starting training to achieve 42% improvement...")
        print("\n🚀 TRAINING STARTED - Monitor progress in TensorBoard!")
        print(f"   Run: tensorboard --logdir {config['paths']['logs']}")
        print(f"   URL: http://localhost:6006")
        print()
        
        # Run training
        trainer.train()
        
        # Training completed
        print("\n🎉 TRAINING COMPLETED!")
        
        if trainer.best_val_score > config['training']['target_clinical_score']:
            improvement = (trainer.best_val_score - 1.0) * 100
            print(f"✅ TARGET ACHIEVED! {improvement:.1f}% improvement (Target: 42%)")
            print("🏥 Model ready for clinical validation!")
        else:
            improvement = (trainer.best_val_score - 1.0) * 100
            print(f"📈 Progress: {improvement:.1f}% improvement (Target: 42%)")
            print("🔄 Consider additional training or hyperparameter tuning")
        
        print(f"\n📁 Results saved in: {trainer.results_dir}")
        print(f"💾 Best model: {os.path.join(trainer.checkpoint_dir, 'best_generator.h5')}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)