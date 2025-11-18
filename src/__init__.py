"""
Medical Image Enhancement Framework

A state-of-the-art deep learning framework for enhancing medical images 
using Generative Adversarial Networks (GANs).
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Import main modules
try:
    from .models.medical_gan import MedicalGAN
    from .data.data_loader import MedicalImageDataLoader
    from .evaluation.metrics import MedicalImageMetrics
    
    __all__ = [
        'MedicalGAN',
        'MedicalImageDataLoader', 
        'MedicalImageMetrics',
    ]
except ImportError:
    # Handle case where dependencies are not installed
    __all__ = []