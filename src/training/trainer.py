import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import yaml

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import our modules
from src.models.medical_gan import MedicalGAN
from src.data.data_loader import MedicalImageDataLoader

class MedicalGANTrainer:
    """
    Advanced Medical GAN Training Pipeline
    Target: 42% image quality improvement with clinical validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize components
        self.gan = None
        self.data_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_info = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_losses': {'d_loss': [], 'g_loss': [], 'g_loss_pixel': [], 'g_loss_perceptual': []},
            'val_metrics': {'psnr': [], 'ssim': [], 'clinical_score': []},
            'epoch_times': []
        }
        
        # Paths
        self.experiment_name = f"medical_gan_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(config['paths']['checkpoints'], self.experiment_name)
        self.results_dir = os.path.join(config['paths']['results'], self.experiment_name)
        self.logs_dir = os.path.join(config['paths']['logs'], self.experiment_name)
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # TensorBoard
        self.train_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, 'train'))
        self.val_writer = tf.summary.create_file_writer(os.path.join(self.logs_dir, 'validation'))
        
        # Initialize Weights & Biases if available and enabled
        if WANDB_AVAILABLE and config.get('use_wandb', False):
            wandb.init(
                project="medical-image-enhancement",
                name=self.experiment_name,
                config=config,
                dir=self.results_dir
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        print(f"🏋️ MedicalGANTrainer initialized!")
        print(f"   🔬 Experiment: {self.experiment_name}")
        print(f"   📁 Checkpoints: {self.checkpoint_dir}")
        print(f"   📊 Results: {self.results_dir}")
        print(f"   📈 Logs: {self.logs_dir}")
        print(f"   🔍 Wandb: {self.use_wandb}")
    
    def setup(self):
        """Initialize GAN model and data pipeline"""
        print("\n🔧 Setting up training components...")
        
        # Initialize GAN
        print("🧠 Initializing Medical GAN...")
        self.gan = MedicalGAN(
            input_shape=tuple(self.config['model']['input_shape']),
            learning_rate=self.config['model']['learning_rate'],
            beta_1=self.config['model']['beta_1']
        )
        
        # Initialize data loader
        print("📊 Setting up data pipeline...")
        self.data_loader = MedicalImageDataLoader(
            data_dir=self.config['data']['data_dir'],
            image_size=tuple(self.config['data']['image_size']),
            batch_size=self.config['training']['batch_size'],
            validation_split=self.config['data']['validation_split'],
            seed=self.config['training']['seed']
        )
        
        # Get datasets
        self.train_dataset, self.val_dataset, self.dataset_info = self.data_loader.get_datasets()
        
        # Save config
        config_path = os.path.join(self.results_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print("✅ Setup completed!")
        print(f"   📊 Training batches: ~{self.dataset_info['train_images'] // self.config['training']['batch_size']}")
        if self.val_dataset:
            print(f"   ✅ Validation batches: ~{self.dataset_info['val_images'] // self.config['training']['batch_size']}")
    
    def calculate_clinical_metrics(self, low_quality: np.ndarray, 
                                 high_quality: np.ndarray,
                                 enhanced: np.ndarray) -> Dict[str, float]:
        """
        Calculate clinical-relevant metrics for medical image quality assessment
        """
        metrics = {}
        
        # Convert from [-1, 1] to [0, 1] for metric calculations
        high_quality = (high_quality + 1) / 2
        enhanced = (enhanced + 1) / 2
        low_quality = (low_quality + 1) / 2
        
        batch_size = high_quality.shape[0]
        
        psnr_scores = []
        ssim_scores = []
        enhancement_scores = []
        
        for i in range(min(batch_size, 8)):  # Limit to 8 samples for speed
            try:
                # Extract single images
                orig = high_quality[i].squeeze()
                enh = enhanced[i].squeeze()  
                low = low_quality[i].squeeze()
                
                # PSNR (Peak Signal-to-Noise Ratio)
                mse = np.mean((orig - enh) ** 2)
                if mse > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                    psnr_scores.append(psnr)
                
                # SSIM (Structural Similarity Index)
                # Simplified SSIM calculation
                mu1 = np.mean(orig)
                mu2 = np.mean(enh)
                sigma1 = np.var(orig)
                sigma2 = np.var(enh)
                sigma12 = np.mean((orig - mu1) * (enh - mu2))
                
                c1 = (0.01) ** 2
                c2 = (0.03) ** 2
                
                ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
                
                ssim_scores.append(ssim)
                
                # Enhancement effectiveness (vs low quality)
                psnr_low = 20 * np.log10(1.0 / np.sqrt(np.mean((orig - low) ** 2) + 1e-8))
                psnr_enh = 20 * np.log10(1.0 / np.sqrt(np.mean((orig - enh) ** 2) + 1e-8))
                
                enhancement_ratio = psnr_enh / (psnr_low + 1e-8)
                enhancement_scores.append(enhancement_ratio)
                
            except Exception as e:
                print(f"⚠️  Metric calculation error: {e}")
                continue
        
        # Calculate averages
        if psnr_scores:
            metrics['psnr'] = np.mean(psnr_scores)
            metrics['psnr_std'] = np.std(psnr_scores)
        else:
            metrics['psnr'] = 0.0
            metrics['psnr_std'] = 0.0
        
        if ssim_scores:
            metrics['ssim'] = np.mean(ssim_scores)
            metrics['ssim_std'] = np.std(ssim_scores)
        else:
            metrics['ssim'] = 0.0  
            metrics['ssim_std'] = 0.0
        
        if enhancement_scores:
            metrics['enhancement_ratio'] = np.mean(enhancement_scores)
            metrics['enhancement_std'] = np.std(enhancement_scores)
        else:
            metrics['enhancement_ratio'] = 1.0
            metrics['enhancement_std'] = 0.0
        
        # Clinical quality score (weighted combination)
        # Target: >1.42 for 42% improvement
        clinical_score = (
            0.4 * min(metrics['psnr'] / 30.0, 1.0) +  # Normalize PSNR
            0.4 * metrics['ssim'] +                    # SSIM already 0-1
            0.2 * min(metrics['enhancement_ratio'] / 2.0, 1.0)  # Enhancement ratio
        )
        
        metrics['clinical_score'] = clinical_score
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_start = time.time()
        
        # Training metrics
        d_losses = []
        g_losses = []
        g_pixel_losses = []
        g_perceptual_losses = []
        
        # Progress bar
        pbar = tqdm(self.train_dataset, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (low_quality_batch, high_quality_batch) in enumerate(pbar):
            # Training step
            losses = self.gan.train_step(low_quality_batch, high_quality_batch)
            
            # Collect losses
            d_losses.append(losses['d_loss'])
            g_losses.append(losses['g_loss'])
            g_pixel_losses.append(losses['g_loss_pixel'])
            g_perceptual_losses.append(losses['g_loss_perceptual'])
            
            # Update progress bar
            pbar.set_postfix({
                'D_Loss': f"{losses['d_loss']:.4f}",
                'G_Loss': f"{losses['g_loss']:.4f}",
                'PSNR_Est': f"{min(30.0, 10 + losses['g_loss_pixel'] * -10):.1f}"
            })
            
            # Log to TensorBoard (every 100 steps)
            if batch_idx % 100 == 0:
                step = self.current_epoch * len(self.train_dataset) + batch_idx
                with self.train_writer.as_default():
                    tf.summary.scalar('batch_d_loss', losses['d_loss'], step=step)
                    tf.summary.scalar('batch_g_loss', losses['g_loss'], step=step)
                    tf.summary.scalar('batch_g_pixel_loss', losses['g_loss_pixel'], step=step)
                    tf.summary.scalar('batch_g_perceptual_loss', losses['g_loss_perceptual'], step=step)
        
        # Calculate epoch averages
        epoch_losses = {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses), 
            'g_loss_pixel': np.mean(g_pixel_losses),
            'g_loss_perceptual': np.mean(g_perceptual_losses),
            'epoch_time': time.time() - epoch_start
        }
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_dataset is None:
            return {}
        
        print("🔍 Running validation...")
        
        val_metrics = {
            'psnr': [],
            'ssim': [], 
            'enhancement_ratio': [],
            'clinical_score': []
        }
        
        # Process validation batches
        for batch_idx, (low_quality_batch, high_quality_batch) in enumerate(self.val_dataset):
            if batch_idx >= 10:  # Limit validation batches for speed
                break
            
            # Generate enhanced images
            enhanced_batch = self.gan.generator(low_quality_batch, training=False)
            
            # Calculate metrics
            batch_metrics = self.calculate_clinical_metrics(
                low_quality_batch.numpy(),
                high_quality_batch.numpy(), 
                enhanced_batch.numpy()
            )
            
            # Collect metrics
            for key in val_metrics.keys():
                if key in batch_metrics:
                    val_metrics[key].append(batch_metrics[key])
        
        # Average metrics
        averaged_metrics = {}
        for key, values in val_metrics.items():
            if values:
                averaged_metrics[key] = np.mean(values)
                averaged_metrics[f'{key}_std'] = np.std(values)
            else:
                averaged_metrics[key] = 0.0
                averaged_metrics[f'{key}_std'] = 0.0
        
        return averaged_metrics
    
    def generate_sample_images(self, epoch: int, num_samples: int = 4):
        """Generate and save sample enhanced images"""
        try:
            # Get sample batch
            for low_batch, high_batch in self.train_dataset.take(1):
                low_sample = low_batch[:num_samples]
                high_sample = high_batch[:num_samples]
                break
            else:
                return
            
            # Generate enhanced images
            enhanced_sample = self.gan.generator(low_sample, training=False)
            
            # Convert to displayable format [0, 1]
            low_display = (low_sample.numpy() + 1) / 2
            high_display = (high_sample.numpy() + 1) / 2
            enhanced_display = (enhanced_sample.numpy() + 1) / 2
            
            # Create comparison plot
            fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
            if num_samples == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(num_samples):
                # Low quality
                axes[0, i].imshow(low_display[i].squeeze(), cmap='gray')
                axes[0, i].set_title('Low Quality Input')
                axes[0, i].axis('off')
                
                # Enhanced
                axes[1, i].imshow(enhanced_display[i].squeeze(), cmap='gray')
                axes[1, i].set_title('GAN Enhanced')
                axes[1, i].axis('off')
                
                # Original high quality
                axes[2, i].imshow(high_display[i].squeeze(), cmap='gray')
                axes[2, i].set_title('Ground Truth')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Save image
            sample_path = os.path.join(self.results_dir, f'samples_epoch_{epoch:03d}.png')
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to Wandb if available
            if self.use_wandb:
                wandb.log({
                    'samples': wandb.Image(sample_path),
                    'epoch': epoch
                })
            
            print(f"💾 Sample images saved: {sample_path}")
            
        except Exception as e:
            print(f"⚠️  Error generating samples: {e}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        try:
            # Save GAN models
            self.gan.save_models(self.checkpoint_dir, epoch)
            
            # Save training state
            state = {
                'epoch': epoch,
                'config': self.config,
                'training_history': self.training_history,
                'dataset_info': self.dataset_info,
                'best_val_score': self.best_val_score
            }
            
            state_path = os.path.join(self.checkpoint_dir, f'training_state_{epoch:03d}.json')
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Save best model
            if is_best:
                best_gen_path = os.path.join(self.checkpoint_dir, 'best_generator.h5')
                best_disc_path = os.path.join(self.checkpoint_dir, 'best_discriminator.h5')
                
                self.gan.generator.save_weights(best_gen_path)
                self.gan.discriminator.save_weights(best_disc_path)
                
                print(f"🏆 New best model saved! Clinical Score: {self.best_val_score:.4f}")
            
        except Exception as e:
            print(f"⚠️  Error saving checkpoint: {e}")
    
    def log_metrics(self, epoch: int, train_losses: Dict, val_metrics: Dict = None):
        """Log metrics to TensorBoard and Wandb"""
        # TensorBoard logging
        with self.train_writer.as_default():
            for key, value in train_losses.items():
                tf.summary.scalar(f'epoch_{key}', value, step=epoch)
        
        if val_metrics:
            with self.val_writer.as_default():
                for key, value in val_metrics.items():
                    if not key.endswith('_std'):  # Don't log std separately
                        tf.summary.scalar(key, value, step=epoch)
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {'epoch': epoch}
            
            # Add training losses
            for key, value in train_losses.items():
                log_dict[f'train/{key}'] = value
            
            # Add validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
        
        # Update training history
        for key, value in train_losses.items():
            if key in self.training_history['train_losses']:
                self.training_history['train_losses'][key].append(value)
        
        if val_metrics:
            for key in ['psnr', 'ssim', 'clinical_score']:
                if key in val_metrics:
                    self.training_history['val_metrics'][key].append(val_metrics[key])
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training for {self.config['training']['epochs']} epochs...")
        print(f"🎯 Target: 42% improvement (Clinical Score > 1.42)")
        
        # Training parameters
        epochs = self.config['training']['epochs']
        save_interval = self.config['training']['save_interval']
        sample_interval = self.config['training']['sample_interval']
        validation_interval = self.config['training']['validation_interval']
        
        # Early stopping
        patience = self.config['training'].get('patience', 10)
        best_epoch = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\n📊 Epoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if epoch % validation_interval == 0 or epoch == epochs - 1:
                val_metrics = self.validate_epoch()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            # Log metrics
            self.log_metrics(epoch, train_losses, val_metrics)
            
            # Print progress
            print(f"🏋️  Training Losses:")
            print(f"   D_Loss: {train_losses['d_loss']:.6f}")
            print(f"   G_Loss: {train_losses['g_loss']:.6f}")
            print(f"   G_Pixel: {train_losses['g_loss_pixel']:.6f}")
            print(f"   G_Perceptual: {train_losses['g_loss_perceptual']:.6f}")
            
            if val_metrics:
                print(f"✅ Validation Metrics:")
                print(f"   PSNR: {val_metrics.get('psnr', 0):.2f} ± {val_metrics.get('psnr_std', 0):.2f}")
                print(f"   SSIM: {val_metrics.get('ssim', 0):.4f} ± {val_metrics.get('ssim_std', 0):.4f}")
                print(f"   Enhancement Ratio: {val_metrics.get('enhancement_ratio', 1):.3f}x")
                print(f"   🎯 Clinical Score: {val_metrics.get('clinical_score', 0):.4f}")
                
                # Check for improvement (42% target)
                current_score = val_metrics.get('clinical_score', 0)
                if current_score > 1.42:
                    print(f"🎉 TARGET ACHIEVED! 42% improvement reached!")
                
                # Update best score
                if current_score > self.best_val_score:
                    self.best_val_score = current_score
                    best_epoch = epoch
                    is_best = True
                else:
                    is_best = False
                
                # Early stopping check
                if epoch - best_epoch > patience:
                    print(f"🛑 Early stopping triggered. No improvement for {patience} epochs.")
                    break
            else:
                is_best = False
            
            print(f"⏱️  Epoch time: {epoch_time:.2f}s")
            
            # Generate samples
            if epoch % sample_interval == 0 or epoch == epochs - 1:
                self.generate_sample_images(epoch)
            
            # Save checkpoint
            if epoch % save_interval == 0 or epoch == epochs - 1 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Training completed
        total_time = time.time() - time.mktime(self.start_time.timetuple())
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Total time: {total_time / 3600:.2f} hours")
        print(f"🏆 Best Clinical Score: {self.best_val_score:.4f}")
        
        if self.best_val_score > 1.42:
            improvement = (self.best_val_score - 1.0) * 100
            print(f"🎯 SUCCESS! Achieved {improvement:.1f}% improvement (Target: 42%)")
        else:
            print(f"📈 Progress: {(self.best_val_score - 1.0) * 100:.1f}% (Target: 42%)")
        
        # Final model save
        final_path = os.path.join(self.checkpoint_dir, 'final_generator.h5')
        self.gan.generator.save_weights(final_path)
        
        print(f"💾 Final model saved: {final_path}")
        
        # Save training summary
        self.save_training_summary()
    
    def save_training_summary(self):
        """Save comprehensive training summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'dataset_info': self.dataset_info,
            'training_history': self.training_history,
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'target_achieved': self.best_val_score > 1.42,
            'improvement_percentage': (self.best_val_score - 1.0) * 100,
            'model_paths': {
                'best_generator': os.path.join(self.checkpoint_dir, 'best_generator.h5'),
                'final_generator': os.path.join(self.checkpoint_dir, 'final_generator.h5')
            }
        }
        
        summary_path = os.path.join(self.results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 Training summary saved: {summary_path}")


def load_config(config_path: str = None) -> Dict:
    """Load training configuration"""
    
    default_config = {
        'model': {
            'input_shape': [256, 256, 1],
            'learning_rate': 0.0002,
            'beta_1': 0.5
        },
        'data': {
            'data_dir': 'data/synthetic',
            'image_size': [256, 256],
            'validation_split': 0.2
        },
        'training': {
            'epochs': 100,
            'batch_size': 8,
            'seed': 42,
            'save_interval': 10,
            'sample_interval': 5,
            'validation_interval': 5,
            'patience': 15
        },
        'paths': {
            'checkpoints': 'models/checkpoints',
            'results': 'results',
            'logs': 'logs'
        },
        'use_wandb': False
    }
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
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


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing MedicalGANTrainer...")
    
    # Load configuration
    config = load_config()
    
    # Override for quick test
    config['training']['epochs'] = 5
    config['training']['batch_size'] = 4
    config['training']['sample_interval'] = 2
    
    print(f"📋 Configuration:")
    for section, params in config.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    # Initialize trainer
    trainer = MedicalGANTrainer(config)
    
    try:
        # Setup
        trainer.setup()
        
        # Quick training test (just setup, no full training)
        print("\n🧪 Setup test completed successfully!")
        print("🚀 Ready for full training!")
        
        # Uncomment to run full training:
        # trainer.train()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure synthetic data is generated first!")