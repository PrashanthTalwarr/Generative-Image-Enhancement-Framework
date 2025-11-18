from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os
import sys
import tensorflow as tf
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
from werkzeug.utils import secure_filename
from functools import wraps

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.models.medical_gan import MedicalGAN
from src.evaluation.metrics import MedicalImageMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'medical-gan-secret-key')

# Global variables
model: Optional[MedicalGAN] = None
metrics_evaluator: Optional[MedicalImageMetrics] = None
model_info: Dict = {}
startup_time = datetime.now()

# API Statistics
api_stats = {
    'total_requests': 0,
    'successful_enhancements': 0,
    'failed_requests': 0,
    'average_processing_time': 0.0,
    'startup_time': startup_time.isoformat()
}

def track_api_call(func):
    """Decorator to track API call statistics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api_stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            api_stats['successful_enhancements'] += 1
            
            # Update average processing time
            processing_time = time.time() - start_time
            current_avg = api_stats['average_processing_time']
            total_successful = api_stats['successful_enhancements']
            
            api_stats['average_processing_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
            
            return result
            
        except Exception as e:
            api_stats['failed_requests'] += 1
            logger.error(f"API call failed: {str(e)}")
            raise
            
    return wrapper

def load_model():
    """Load the trained Medical GAN model"""
    global model, model_info
    
    try:
        logger.info("🧠 Loading Medical GAN model...")
        
        # Initialize model with default configuration
        model = MedicalGAN(
            input_shape=(256, 256, 1),
            learning_rate=0.0002,
            beta_1=0.5
        )
        
        # Try to load pre-trained weights
        checkpoint_dirs = [
            'models/checkpoints',
            'models/final',
            os.path.join(os.path.dirname(__file__), '../../models/checkpoints'),
            os.path.join(os.path.dirname(__file__), '../../models/final')
        ]
        
        model_loaded = False
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                try:
                    # Look for best model first
                    best_gen_path = os.path.join(checkpoint_dir, 'best_generator.h5')
                    if os.path.exists(best_gen_path):
                        model.generator.load_weights(best_gen_path)
                        model_info['model_path'] = best_gen_path
                        model_info['model_type'] = 'best_model'
                        model_loaded = True
                        logger.info(f"✅ Loaded best model: {best_gen_path}")
                        break
                    
                    # Look for final model
                    final_gen_path = os.path.join(checkpoint_dir, 'final_generator.h5')
                    if os.path.exists(final_gen_path):
                        model.generator.load_weights(final_gen_path)
                        model_info['model_path'] = final_gen_path
                        model_info['model_type'] = 'final_model'
                        model_loaded = True
                        logger.info(f"✅ Loaded final model: {final_gen_path}")
                        break
                    
                    # Look for latest checkpoint
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('generator_epoch_')]
                    if checkpoints:
                        latest_checkpoint = sorted(checkpoints)[-1]
                        epoch = latest_checkpoint.split('_')[-1].split('.')[0]
                        
                        gen_path = os.path.join(checkpoint_dir, latest_checkpoint)
                        disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.h5')
                        
                        if os.path.exists(gen_path) and os.path.exists(disc_path):
                            model.generator.load_weights(gen_path)
                            model.discriminator.load_weights(disc_path)
                            model_info['model_path'] = gen_path
                            model_info['model_type'] = 'checkpoint'
                            model_info['epoch'] = epoch
                            model_loaded = True
                            logger.info(f"✅ Loaded checkpoint epoch {epoch}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to load from {checkpoint_dir}: {e}")
                    continue
        
        if not model_loaded:
            logger.warning("⚠️  No pre-trained model found. Using randomly initialized weights.")
            model_info['model_type'] = 'random_initialization'
        
        # Collect model information
        model_info.update({
            'generator_parameters': int(model.generator.count_params()),
            'discriminator_parameters': int(model.discriminator.count_params()),
            'input_shape': model.input_shape,
            'learning_rate': model.learning_rate,
            'status': 'loaded'
        })
        
        logger.info(f"🎯 Model loaded successfully!")
        logger.info(f"   Generator parameters: {model_info['generator_parameters']:,}")
        logger.info(f"   Model type: {model_info['model_type']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        model = None
        model_info = {'status': 'failed', 'error': str(e)}

def initialize_metrics():
    """Initialize the metrics evaluator"""
    global metrics_evaluator
    
    try:
        logger.info("📈 Initializing metrics evaluator...")
        metrics_evaluator = MedicalImageMetrics()
        logger.info("✅ Metrics evaluator initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize metrics: {str(e)}")
        metrics_evaluator = None

def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            if image_data.shape[2] == 3:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            elif image_data.shape[2] == 4:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
        
        # Resize to model input size (256x256)
        if image_data.shape[:2] != (256, 256):
            image_data = cv2.resize(image_data, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        if image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float32) / 255.0
        
        # Normalize to [-1, 1] for GAN
        image_data = (image_data * 2.0) - 1.0
        
        # Add channel and batch dimensions
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=-1)
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        return image_data.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def postprocess_image(image_data: np.ndarray) -> np.ndarray:
    """Postprocess image from model output"""
    try:
        # Remove batch dimension
        if len(image_data.shape) == 4:
            image_data = image_data[0]
        
        # Remove channel dimension if single channel
        if len(image_data.shape) == 3 and image_data.shape[-1] == 1:
            image_data = image_data.squeeze(axis=-1)
        
        # Denormalize from [-1, 1] to [0, 1]
        image_data = (image_data + 1.0) / 2.0
        
        # Convert to 8-bit
        image_data = np.clip(image_data * 255.0, 0, 255).astype(np.uint8)
        
        return image_data
        
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        raise ValueError(f"Image postprocessing failed: {str(e)}")

# API Routes

@app.route('/')
def index():
    """API information endpoint"""
    return jsonify({
        'service': 'Medical Image Enhancement API',
        'version': '1.0.0',
        'status': 'online',
        'model_status': 'loaded' if model else 'not_loaded',
        'metrics_available': metrics_evaluator is not None,
        'startup_time': startup_time.isoformat(),
        'uptime_seconds': int((datetime.now() - startup_time).total_seconds()),
        'endpoints': {
            'enhance': '/enhance (POST) - Enhance medical image',
            'enhance_base64': '/enhance_base64 (POST) - Enhance base64 image',
            'batch_enhance': '/batch_enhance (POST) - Enhance multiple images',
            'metrics': '/metrics (POST) - Calculate quality metrics',
            'model_info': '/model_info (GET) - Model information',
            'health': '/health (GET) - Health check',
            'stats': '/stats (GET) - API statistics'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'metrics_available': metrics_evaluator is not None,
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'memory_usage': 'available'  # Could add actual memory monitoring
    }
    
    # Determine overall health
    if model is None:
        health_status['status'] = 'degraded'
        health_status['issues'] = ['Model not loaded']
    
    return jsonify(health_status)

@app.route('/model_info')
def get_model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(model_info)

@app.route('/stats')
def get_api_stats():
    """Get API usage statistics"""
    return jsonify(api_stats)

@app.route('/enhance', methods=['POST'])
@track_api_call
def enhance_image():
    """Enhance uploaded medical image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dcm'}
        file_ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Read and process image
        start_time = time.time()
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        logger.info(f"📥 Processing image: {file.filename}, shape: {image_array.shape}")
        
        # Preprocess
        input_image = preprocess_image(image_array)
        
        # Enhance using model
        enhanced = model.enhance_image(input_image)
        
        # Postprocess
        output_image = postprocess_image(enhanced)
        
        processing_time = time.time() - start_time
        
        # Convert to base64 for response
        _, buffer = cv2.imencode('.png', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate metrics if evaluator is available
        metrics = {}
        if metrics_evaluator:
            try:
                # Prepare images for metrics (normalize to [0,1])
                orig_norm = image_array.astype(np.float32)
                if orig_norm.max() > 1:
                    orig_norm = orig_norm / 255.0
                
                # Resize original to match output
                if orig_norm.shape[:2] != (256, 256):
                    orig_norm = cv2.resize(orig_norm, (256, 256))
                
                if len(orig_norm.shape) == 3:
                    orig_norm = cv2.cvtColor(orig_norm, cv2.COLOR_RGB2GRAY)
                
                enh_norm = output_image.astype(np.float32) / 255.0
                
                # Create synthetic low-quality version for metrics
                low_quality = orig_norm + np.random.normal(0, 0.1, orig_norm.shape)
                low_quality = np.clip(low_quality, 0, 1)
                
                metrics = metrics_evaluator.calculate_diagnostic_quality_index(
                    orig_norm, low_quality, enh_norm
                )
                
                logger.info(f"📊 Metrics calculated - Clinical Score: {metrics.get('clinical_quality_score', 0):.4f}")
                
            except Exception as e:
                logger.warning(f"Metrics calculation failed: {str(e)}")
                metrics = {'error': 'Metrics calculation failed'}
        
        response = {
            'status': 'success',
            'enhanced_image': img_base64,
            'original_filename': secure_filename(file.filename),
            'processing_time_seconds': processing_time,
            'image_dimensions': {
                'input': list(image_array.shape),
                'output': list(output_image.shape)
            },
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Enhancement completed in {processing_time:.3f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}")
        return jsonify({
            'error': f'Enhancement failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/enhance_base64', methods=['POST'])
@track_api_call
def enhance_base64_image():
    """Enhance base64 encoded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        start_time = time.time()
        
        # Preprocess
        input_image = preprocess_image(image_array)
        
        # Enhance
        enhanced = model.enhance_image(input_image)
        
        # Postprocess
        output_image = postprocess_image(enhanced)
        
        processing_time = time.time() - start_time
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', output_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            'status': 'success',
            'enhanced_image': img_base64,
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Base64 enhancement completed in {processing_time:.3f}s")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Base64 enhancement failed: {str(e)}")
        return jsonify({
            'error': f'Enhancement failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_enhance', methods=['POST'])
@track_api_call
def batch_enhance():
    """Enhance multiple images"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        files = request.files.getlist('images')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No images provided'}), 400
        
        max_batch_size = int(os.environ.get('MAX_BATCH_SIZE', 10))
        if len(files) > max_batch_size:
            return jsonify({'error': f'Batch size too large. Maximum: {max_batch_size}'}), 400
        
        results = []
        start_time = time.time()
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
            
            try:
                # Process individual image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # Preprocess
                input_image = preprocess_image(image_array)
                
                # Enhance
                enhanced = model.enhance_image(input_image)
                
                # Postprocess
                output_image = postprocess_image(enhanced)
                
                # Convert to base64
                _, buffer = cv2.imencode('.png', output_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                results.append({
                    'filename': secure_filename(file.filename),
                    'enhanced_image': img_base64,
                    'status': 'success'
                })
                
                logger.info(f"✅ Batch item {i+1}/{len(files)} completed: {file.filename}")
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                results.append({
                    'filename': secure_filename(file.filename),
                    'status': 'error',
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r['status'] == 'success'])
        
        response = {
            'status': 'completed',
            'results': results,
            'summary': {
                'total_images': len(files),
                'successful': successful,
                'failed': len(results) - successful,
                'processing_time_seconds': total_time,
                'avg_time_per_image': total_time / len(files) if files else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Batch enhancement completed: {successful}/{len(files)} successful")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch enhancement failed: {str(e)}")
        return jsonify({
            'error': f'Batch enhancement failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/calculate_metrics', methods=['POST'])
def calculate_metrics():
    """Calculate quality metrics between original and enhanced images"""
    if metrics_evaluator is None:
        return jsonify({'error': 'Metrics evaluator not available'}), 500
    
    try:
        files = request.files
        
        if 'original' not in files or 'enhanced' not in files:
            return jsonify({'error': 'Both original and enhanced images required'}), 400
        
        # Load images
        orig_bytes = files['original'].read()
        enh_bytes = files['enhanced'].read()
        
        orig_image = np.array(Image.open(io.BytesIO(orig_bytes)))
        enh_image = np.array(Image.open(io.BytesIO(enh_bytes)))
        
        # Preprocess for metrics
        def prepare_for_metrics(img):
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            if img.shape[:2] != (256, 256):
                img = cv2.resize(img, (256, 256))
            
            return img.astype(np.float32) / 255.0
        
        orig_processed = prepare_for_metrics(orig_image)
        enh_processed = prepare_for_metrics(enh_image)
        
        # Create synthetic low-quality for comprehensive metrics
        low_quality = orig_processed + np.random.normal(0, 0.1, orig_processed.shape)
        low_quality = np.clip(low_quality, 0, 1)
        
        # Calculate comprehensive metrics
        metrics = metrics_evaluator.calculate_diagnostic_quality_index(
            orig_processed, low_quality, enh_processed
        )
        
        response = {
            'status': 'success',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📊 Metrics calculated - Clinical Score: {metrics.get('clinical_quality_score', 0):.4f}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {str(e)}")
        return jsonify({
            'error': f'Metrics calculation failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size': '32MB',
        'timestamp': datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/', '/health', '/model_info', '/stats',
            '/enhance', '/enhance_base64', '/batch_enhance', '/calculate_metrics'
        ],
        'timestamp': datetime.now().isoformat()
    }), 404

# Application startup

def initialize_app():
    """Initialize the application"""
    logger.info("🚀 Initializing Medical Image Enhancement API...")
    
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    
    # Load model and initialize components
    load_model()
    initialize_metrics()
    
    logger.info("✅ API initialization completed!")
    logger.info(f"🌐 Ready to serve requests on all endpoints")

if __name__ == '__main__':
    # Initialize application
    initialize_app()
    
    # Configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌐 Starting Medical GAN API server...")
    logger.info(f"   Host: {host}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Debug: {debug}")
    logger.info(f"   Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    logger.info(f"   Metrics: {'✅ Available' if metrics_evaluator else '❌ Not Available'}")
    
    # Start the server
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {str(e)}")