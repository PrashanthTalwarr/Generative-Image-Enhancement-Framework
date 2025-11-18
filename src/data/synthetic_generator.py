import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, List
import os
from tqdm import tqdm
import json

class MedicalSyntheticDataGenerator:
    def __init__(self, image_size: Tuple[int, int] = (256, 256), seed: int = 42):
        self.image_size = image_size
        self.seed = seed
        np.random.seed(seed)
        print(f" MedicalSyntheticDataGenerator initialized - Image size: {image_size}")
    
    def generate_brain_mri(self, noise_level: float = 0.05) -> np.ndarray:
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)
        
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        
        # Brain outline
        brain_mask = ((x - center[1]) ** 2 / (w * 0.35) ** 2 + 
                     (y - center[0]) ** 2 / (h * 0.4) ** 2) <= 1
        image[brain_mask] = 0.6
        
        # Inner structures
        inner_mask = ((x - center[1]) ** 2 / (w * 0.25) ** 2 + 
                     (y - center[0]) ** 2 / (h * 0.3) ** 2) <= 1
        image[inner_mask] = 0.8
        
        # Ventricles
        ventricle_left = ((x - center[1] + 25) ** 2 / 12 ** 2 + 
                         (y - center[0]) ** 2 / 20 ** 2) <= 1
        ventricle_right = ((x - center[1] - 25) ** 2 / 12 ** 2 + 
                          (y - center[0]) ** 2 / 20 ** 2) <= 1
        image[ventricle_left | ventricle_right] = 0.2
        
        # Add noise
        noise = np.random.normal(0, noise_level, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def generate_chest_xray(self, noise_level: float = 0.08) -> np.ndarray:
        h, w = self.image_size
        image = np.ones((h, w), dtype=np.float32) * 0.3
        
        # Lung regions
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Left lung
        lung_left = ((x - w * 0.25) ** 2 / (w * 0.15) ** 2 + 
                    (y - h * 0.45) ** 2 / (h * 0.25) ** 2) <= 1
        image[lung_left] = 0.1
        
        # Right lung
        lung_right = ((x - w * 0.75) ** 2 / (w * 0.15) ** 2 + 
                     (y - h * 0.45) ** 2 / (h * 0.25) ** 2) <= 1
        image[lung_right] = 0.1
        
        # Heart
        heart_mask = ((x - w * 0.4) ** 2 / (w * 0.1) ** 2 + 
                     (y - h * 0.55) ** 2 / (h * 0.15) ** 2) <= 1
        image[heart_mask] = 0.4
        
        # Add noise
        noise = np.random.normal(0, noise_level, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def generate_ct_slice(self, noise_level: float = 0.06) -> np.ndarray:
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)
        
        center = (h // 2, w // 2)
        y, x = np.ogrid[:h, :w]
        
        # Body outline
        body_mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) <= (min(h, w) * 0.4) ** 2
        image[body_mask] = 0.3
        
        # Organs
        liver_mask = ((x - center[1] - 60) ** 2 / 50 ** 2 + 
                     (y - center[0] - 10) ** 2 / 70 ** 2) <= 1
        liver_mask = liver_mask & body_mask
        image[liver_mask] = 0.5
        
        # Spine
        spine_mask = ((x - center[1]) ** 2 / 15 ** 2 + 
                     (y - center[0]) ** 2 / 100 ** 2) <= 1
        image[spine_mask & body_mask] = 0.9
        
        # Add noise
        noise = np.random.normal(0, noise_level, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def create_quality_variants(self, image: np.ndarray) -> dict:
        variants = {'original': image.copy()}
        
        # Low resolution
        h, w = image.shape
        low_res = cv2.resize(image, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        variants['low_res'] = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Noisy
        noise = np.random.normal(0, 0.1, image.shape)
        variants['noisy'] = np.clip(image + noise, 0, 1)
        
        # Blurred
        variants['blurred'] = cv2.GaussianBlur(image, (5, 5), 2.0)
        
        # Motion blur
        kernel = np.zeros((15, 15))
        kernel[7, :] = 1 / 15
        variants['motion_blur'] = cv2.filter2D(image, -1, kernel)
        
        # Compressed (simplified)
        variants['compressed'] = image * 0.8 + 0.1
        
        return variants
    
    def generate_dataset(self, output_dir: str, num_samples_per_type: int = 500,
                        image_types: List[str] = ['brain', 'chest', 'ct'],
                        quality_variants: bool = True) -> None:
        
        os.makedirs(output_dir, exist_ok=True)
        
        generators = {
            'brain': self.generate_brain_mri,
            'chest': self.generate_chest_xray,
            'ct': self.generate_ct_slice
        }
        
        metadata = {
            'total_samples': num_samples_per_type * len(image_types),
            'image_size': self.image_size,
            'image_types': image_types,
            'samples': []
        }
        
        for img_type in image_types:
            print(f"Generating {num_samples_per_type} {img_type} images...")
            
            type_dir = os.path.join(output_dir, img_type)
            os.makedirs(type_dir, exist_ok=True)
            
            if quality_variants:
                for variant in ['original', 'low_res', 'noisy', 'blurred', 'motion_blur', 'compressed']:
                    os.makedirs(os.path.join(type_dir, variant), exist_ok=True)
            
            for i in tqdm(range(num_samples_per_type), desc=f"Generating {img_type}"):
                noise_level = np.random.uniform(0.03, 0.1)
                base_image = generators[img_type](noise_level=noise_level)
                
                if quality_variants:
                    variants = self.create_quality_variants(base_image)
                    
                    for variant_name, variant_image in variants.items():
                        filename = f"{img_type}_{i:04d}.png"
                        filepath = os.path.join(type_dir, variant_name, filename)
                        
                        image_8bit = (variant_image * 255).astype(np.uint8)
                        cv2.imwrite(filepath, image_8bit)
                        
                        metadata['samples'].append({
                            'filename': filename,
                            'type': img_type,
                            'variant': variant_name,
                            'filepath': filepath
                        })
                else:
                    filename = f"{img_type}_{i:04d}.png"
                    filepath = os.path.join(type_dir, filename)
                    
                    image_8bit = (base_image * 255).astype(np.uint8)
                    cv2.imwrite(filepath, image_8bit)
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Dataset generation complete!")
        print(f"Total images generated: {len(metadata['samples'])}")
        print(f"Dataset saved to: {output_dir}")

if __name__ == "__main__":
    generator = MedicalSyntheticDataGenerator()
    generator.generate_dataset('data/synthetic', 50, ['brain', 'chest', 'ct'], True)
