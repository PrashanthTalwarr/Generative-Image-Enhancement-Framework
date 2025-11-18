import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm
import random

# Optional medical imaging libraries (install if available)
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    
try:
    import nibabel as nib
    NIFTI_AVAILABLE = True
except ImportError:
    NIFTI_AVAILABLE = False

class MedicalImageDataLoader:
    """
    Advanced Medical Image Data Loader for GAN training
    Supports: DICOM, NIFTI, PNG, JPG, TIFF
    Optimized for 42% quality improvement target
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 8,
                 validation_split: float = 0.2,
                 seed: int = 42):
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        # Supported file formats
        self.supported_formats = [
            '.dcm', '.dicom',           # DICOM files
            '.nii', '.nii.gz',          # NIFTI files  
            '.png', '.jpg', '.jpeg',     # Standard images
            '.tiff', '.tif', '.bmp'      # Additional formats
        ]
        
        # Data statistics for normalization
        self.data_stats = {
            'mean': 0.0,
            'std': 1.0,
            'min': 0.0,
            'max': 1.0
        }
        
        print("📊 MedicalImageDataLoader initialized!")
        print(f"   📂 Data directory: {data_dir}")
        print(f"   🖼️  Image size: {image_size}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   ✅ DICOM support: {DICOM_AVAILABLE}")
        print(f"   ✅ NIFTI support: {NIFTI_AVAILABLE}")
    
    def load_dicom_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load DICOM medical image"""
        if not DICOM_AVAILABLE:
            print("⚠️  pydicom not available. Install: pip install pydicom")
            return None
            
        try:
            dicom_data = pydicom.dcmread(filepath)
            
            # Handle different DICOM formats
            if hasattr(dicom_data, 'pixel_array'):
                image = dicom_data.pixel_array.astype(np.float32)
                
                # Handle multi-frame DICOM
                if len(image.shape) > 2:
                    image = image[0] if image.shape[0] == 1 else image[image.shape[0]//2]
                
                # Apply DICOM windowing if available
                if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                    center = float(dicom_data.WindowCenter)
                    width = float(dicom_data.WindowWidth)
                    
                    # Apply window/level
                    img_min = center - width // 2
                    img_max = center + width // 2
                    image = np.clip(image, img_min, img_max)
                
                # Normalize to [0, 1]
                if image.max() > image.min():
                    image = (image - image.min()) / (image.max() - image.min())
                
                return image
            else:
                print(f"⚠️  No pixel data in DICOM: {filepath}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading DICOM {filepath}: {e}")
            return None
    
    def load_nifti_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load NIFTI medical image"""
        if not NIFTI_AVAILABLE:
            print("⚠️  nibabel not available. Install: pip install nibabel")
            return None
            
        try:
            nifti_data = nib.load(filepath)
            image_data = nifti_data.get_fdata()
            
            # Handle 3D volumes - extract middle slice
            if len(image_data.shape) == 3:
                if image_data.shape[2] > 1:  # Multiple slices
                    middle_slice = image_data.shape[2] // 2
                    image = image_data[:, :, middle_slice]
                else:
                    image = image_data.squeeze()
            elif len(image_data.shape) == 4:  # 4D volume - extract middle slice of middle time
                mid_t = image_data.shape[3] // 2
                mid_z = image_data.shape[2] // 2
                image = image_data[:, :, mid_z, mid_t]
            else:
                image = image_data
            
            # Normalize to [0, 1]
            image = image.astype(np.float32)
            if image.max() > image.min():
                image = (image - image.min()) / (image.max() - image.min())
            
            return image
            
        except Exception as e:
            print(f"❌ Error loading NIFTI {filepath}: {e}")
            return None
    
    def load_standard_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load standard image formats (PNG, JPG, etc.)"""
        try:
            # Try OpenCV first (faster for medical images)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Fallback to PIL
                with Image.open(filepath) as pil_image:
                    if pil_image.mode != 'L':
                        pil_image = pil_image.convert('L')  # Convert to grayscale
                    image = np.array(pil_image)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            return image
            
        except Exception as e:
            print(f"❌ Error loading image {filepath}: {e}")
            return None
    
    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """Universal image loader - detects format and loads appropriately"""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        
        ext = Path(filepath).suffix.lower()
        
        # Route to appropriate loader
        if ext in ['.dcm', '.dicom']:
            return self.load_dicom_image(filepath)
        elif ext in ['.nii', '.nii.gz']:  
            return self.load_nifti_image(filepath)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            return self.load_standard_image(filepath)
        else:
            print(f"⚠️  Unsupported format: {ext}")
            return None
    
    def preprocess_image(self, image: np.ndarray, 
                        normalize: bool = True,
                        augment: bool = False) -> np.ndarray:
        """
        Preprocess medical image for GAN training
        """
        # Resize to target size
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Ensure single channel
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Add channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Apply augmentation if requested
        if augment:
            image = self.apply_augmentation(image)
        
        # Normalize for GAN training (to [-1, 1] range)
        if normalize:
            image = (image * 2.0) - 1.0
        
        return image.astype(np.float32)
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply medical-safe augmentations that preserve diagnostic features
        """
        # Random horizontal flip (safe for most medical images)
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Small rotation (±5 degrees to preserve anatomy)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, 
                                 (image.shape[1], image.shape[0]), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_REFLECT)
        
        # Small translation (±10 pixels)
        if np.random.random() > 0.8:
            dx = np.random.randint(-10, 10)
            dy = np.random.randint(-10, 10)
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            image = cv2.warpAffine(image, translation_matrix,
                                 (image.shape[1], image.shape[0]),
                                 borderMode=cv2.BORDER_REFLECT)
        
        # Brightness adjustment (±10%)
        if np.random.random() > 0.6:
            brightness_factor = np.random.uniform(0.9, 1.1)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Contrast adjustment (±15%)
        if np.random.random() > 0.7:
            contrast_factor = np.random.uniform(0.85, 1.15)
            image = np.clip((image - 0.5) * contrast_factor + 0.5, 0, 1)
        
        return image
    
    def create_quality_degradation(self, image: np.ndarray, 
                                 degradation_type: str = 'random') -> np.ndarray:
        """
        Create degraded version of high-quality image for training pairs
        """
        degraded = image.copy()
        
        if degradation_type == 'random':
            # Randomly choose degradation type
            degradation_type = np.random.choice([
                'downsample', 'blur', 'noise', 'compression', 'mixed'
            ])
        
        if degradation_type == 'downsample':
            # Downsample and upsample (most common for super-resolution)
            scale = np.random.uniform(0.25, 0.5)
            h, w = image.shape[:2]
            small_h, small_w = int(h * scale), int(w * scale)
            
            # Downsample
            small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            # Upsample back
            degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            
        elif degradation_type == 'blur':
            # Gaussian blur
            sigma = np.random.uniform(1.0, 3.0)
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            degraded = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
        elif degradation_type == 'noise':
            # Add Gaussian noise
            noise_level = np.random.uniform(0.05, 0.15)
            noise = np.random.normal(0, noise_level, image.shape)
            degraded = np.clip(image + noise, 0, 1)
            
        elif degradation_type == 'compression':
            # Simulate JPEG compression artifacts
            if len(image.shape) == 3:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = (image.squeeze() * 255).astype(np.uint8)
            
            quality = np.random.randint(20, 50)
            
            # Encode and decode to simulate compression
            _, encoded = cv2.imencode('.jpg', img_uint8, 
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            degraded = decoded.astype(np.float32) / 255.0
            
            if len(image.shape) == 3:
                degraded = np.expand_dims(degraded, axis=-1)
                
        elif degradation_type == 'mixed':
            # Apply multiple degradations
            # Start with downsampling
            scale = np.random.uniform(0.3, 0.6)
            h, w = image.shape[:2]
            small_h, small_w = int(h * scale), int(w * scale)
            small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Add slight blur
            if np.random.random() > 0.5:
                sigma = np.random.uniform(0.5, 1.5)
                kernel_size = int(2 * np.ceil(2 * sigma) + 1)
                degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), sigma)
            
            # Add noise
            if np.random.random() > 0.3:
                noise_level = np.random.uniform(0.02, 0.08)
                noise = np.random.normal(0, noise_level, degraded.shape)
                degraded = np.clip(degraded + noise, 0, 1)
        
        return degraded.astype(np.float32)
    
    def discover_images(self) -> List[str]:
        """
        Discover all supported medical images in data directory
        """
        image_paths = []
        
        print(f"🔍 Discovering images in: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory not found: {self.data_dir}")
            return []
        
        # Walk through directory structure
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.supported_formats:
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
        
        print(f"✅ Found {len(image_paths)} medical images")
        
        # Log format distribution
        format_counts = {}
        for path in image_paths:
            ext = Path(path).suffix.lower()
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        print("📊 Format distribution:")
        for fmt, count in sorted(format_counts.items()):
            print(f"   {fmt}: {count}")
        
        return image_paths
    
    def create_train_val_split(self, image_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Create training and validation splits
        """
        if not image_paths:
            return [], []
        
        # Stratified split if possible (based on subdirectory)
        try:
            # Group by parent directory for stratified split
            path_groups = {}
            for path in image_paths:
                parent = os.path.basename(os.path.dirname(path))
                if parent not in path_groups:
                    path_groups[parent] = []
                path_groups[parent].append(path)
            
            train_paths = []
            val_paths = []
            
            for group_name, group_paths in path_groups.items():
                if len(group_paths) > 1:
                    train_group, val_group = train_test_split(
                        group_paths, 
                        test_size=self.validation_split,
                        random_state=self.seed
                    )
                    train_paths.extend(train_group)
                    val_paths.extend(val_group)
                else:
                    # Single file goes to training
                    train_paths.extend(group_paths)
            
        except Exception:
            # Fallback to simple random split
            train_paths, val_paths = train_test_split(
                image_paths,
                test_size=self.validation_split,
                random_state=self.seed
            )
        
        print(f"📊 Data split:")
        print(f"   🏋️  Training: {len(train_paths)} images")
        print(f"   ✅ Validation: {len(val_paths)} images")
        
        return train_paths, val_paths
    
    def create_tf_dataset(self, image_paths: List[str], 
                         is_training: bool = True,
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from image paths
        """
        def load_and_process_pair(path_tensor):
            """Load image and create high/low quality pair"""
            path = path_tensor.numpy().decode('utf-8')
            
            # Load original image
            image = self.load_image(path)
            
            if image is None:
                # Return dummy data if loading fails
                dummy = np.zeros((*self.image_size, 1), dtype=np.float32)
                return dummy, dummy
            
            # Preprocess original (high quality)
            high_quality = self.preprocess_image(
                image, 
                normalize=True, 
                augment=is_training
            )
            
            # Create degraded version (low quality) 
            degraded_image = self.create_quality_degradation(
                (image + 1) / 2,  # Convert back to [0,1] for degradation
                degradation_type='random'
            )
            
            # Preprocess degraded image
            low_quality = self.preprocess_image(
                degraded_image,
                normalize=True,
                augment=False  # Don't augment the degraded version
            )
            
            return low_quality.astype(np.float32), high_quality.astype(np.float32)
        
        def tf_load_and_process(path):
            """TensorFlow wrapper for image processing"""
            low_quality, high_quality = tf.py_function(
                load_and_process_pair,
                [path],
                [tf.float32, tf.float32]
            )
            
            # Set shapes explicitly
            low_quality.set_shape((*self.image_size, 1))
            high_quality.set_shape((*self.image_size, 1))
            
            return low_quality, high_quality
        
        # Create dataset from paths
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(image_paths)), 
                                    reshuffle_each_iteration=True)
        
        # Map to image pairs
        dataset = dataset.map(
            tf_load_and_process,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # Filter out failed loads
        dataset = dataset.filter(
            lambda low, high: tf.reduce_sum(tf.abs(high)) > 0
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        """
        Get training and validation datasets
        Returns: (train_dataset, val_dataset, dataset_info)
        """
        # Discover all images
        all_paths = self.discover_images()
        
        if not all_paths:
            raise ValueError("❌ No supported images found in data directory!")
        
        # Create train/val split
        train_paths, val_paths = self.create_train_val_split(all_paths)
        
        if not train_paths:
            raise ValueError("❌ No training images available!")
        
        # Create datasets
        train_dataset = self.create_tf_dataset(train_paths, is_training=True)
        val_dataset = self.create_tf_dataset(val_paths, is_training=False) if val_paths else None
        
        # Dataset info
        dataset_info = {
            'total_images': len(all_paths),
            'train_images': len(train_paths),
            'val_images': len(val_paths),
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split
        }
        
        print("\n✅ Datasets created successfully!")
        print(f"   🏋️  Training batches: ~{len(train_paths) // self.batch_size}")
        if val_dataset:
            print(f"   ✅ Validation batches: ~{len(val_paths) // self.batch_size}")
        
        return train_dataset, val_dataset, dataset_info
    
    def get_sample_batch(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample batch for testing/visualization
        """
        for low_quality, high_quality in dataset.take(1):
            return low_quality.numpy(), high_quality.numpy()
        return None, None


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing MedicalImageDataLoader...")
    
    # Initialize data loader
    data_loader = MedicalImageDataLoader(
        data_dir="data/synthetic",  # Using our synthetic data
        image_size=(256, 256),
        batch_size=4,
        validation_split=0.2
    )
    
    try:
        # Get datasets
        train_ds, val_ds, info = data_loader.get_datasets()
        
        print(f"\n📊 Dataset Info: {info}")
        
        # Test sample batch
        low_batch, high_batch = data_loader.get_sample_batch(train_ds)
        
        if low_batch is not None:
            print(f"\n🧪 Sample batch shapes:")
            print(f"   Low quality: {low_batch.shape}")
            print(f"   High quality: {high_batch.shape}")
            print(f"   Low quality range: [{low_batch.min():.3f}, {low_batch.max():.3f}]")
            print(f"   High quality range: [{high_batch.min():.3f}, {high_batch.max():.3f}]")
        
        print("\n✅ MedicalImageDataLoader test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure to run synthetic data generation first!")