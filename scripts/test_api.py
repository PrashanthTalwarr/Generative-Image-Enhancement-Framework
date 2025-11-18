#!/usr/bin/env python3
"""
Medical Image Enhancement GAN API Testing Suite
Comprehensive testing for production readiness and 42% improvement validation
"""

import requests
import json
import base64
import numpy as np
from PIL import Image
import io
import time
import os
import argparse
from typing import Dict, List, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalGANAPITester:
    """
    Comprehensive API testing suite for Medical Image Enhancement GAN
    """
    
    def __init__(self, api_url: str = "http://localhost:5000", timeout: int = 60):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.test_results = []
        
        print(f"🧪 Medical GAN API Tester initialized")
        print(f"   🌐 API URL: {self.api_url}")
        print(f"   ⏰ Timeout: {self.timeout}s")
        
    def create_test_medical_image(self, size: tuple = (256, 256)) -> np.ndarray:
        """Create synthetic medical-like test image"""
        h, w = size
        
        # Create brain-like structure
        image = np.zeros((h, w), dtype=np.float32)
        
        # Brain outline (elliptical)
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # Main brain structure
        brain_mask = ((x - center_x) ** 2 / (w * 0.35) ** 2 + 
                     (y - center_y) ** 2 / (h * 0.4) ** 2) <= 1
        image[brain_mask] = 0.6
        
        # Inner structures
        inner_mask = ((x - center_x) ** 2 / (w * 0.25) ** 2 + 
                     (y - center_y) ** 2 / (h * 0.3) ** 2) <= 1
        image[inner_mask] = 0.8
        
        # Ventricles
        ventricle_left = ((x - center_x + 20) ** 2 / 12 ** 2 + 
                         (y - center_y) ** 2 / 15 ** 2) <= 1
        ventricle_right = ((x - center_x - 20) ** 2 / 12 ** 2 + 
                          (y - center_y) ** 2 / 15 ** 2) <= 1
        
        image[ventricle_left | ventricle_right] = 0.2
        
        # Add realistic noise
        noise = np.random.normal(0, 0.05, (h, w))
        image = np.clip(image + noise, 0, 1)
        
        return (image * 255).astype(np.uint8)
    
    def test_health_check(self) -> Dict:
        """Test API health endpoint"""
        print("\n🏥 Testing health check endpoint...")
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            result = {
                'test_name': 'health_check',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if result['success']:
                health_data = result['response_data']
                print(f"   ✅ Health check passed")
                print(f"   📊 Status: {health_data.get('status', 'unknown')}")
                print(f"   🧠 Model loaded: {health_data.get('model_loaded', 'unknown')}")
                print(f"   📈 Metrics available: {health_data.get('metrics_available', 'unknown')}")
                print(f"   ⚡ GPU available: {health_data.get('gpu_available', 'unknown')}")
            else:
                print(f"   ❌ Health check failed: HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'test_name': 'health_check',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ Health check error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_model_info(self) -> Dict:
        """Test model info endpoint"""
        print("\n🧠 Testing model info endpoint...")
        
        try:
            response = self.session.get(f"{self.api_url}/model_info", timeout=10)
            
            result = {
                'test_name': 'model_info',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': response.status_code == 200,
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if result['success']:
                model_data = result['response_data']
                print(f"   ✅ Model info retrieved")
                print(f"   🏗️  Generator params: {model_data.get('generator_parameters', 0):,}")
                print(f"   🔍 Discriminator params: {model_data.get('discriminator_parameters', 0):,}")
                print(f"   📏 Input shape: {model_data.get('input_shape', 'unknown')}")
                print(f"   📈 Model type: {model_data.get('model_type', 'unknown')}")
            else:
                print(f"   ❌ Model info failed: HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'test_name': 'model_info',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ Model info error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_single_enhancement(self, save_result: bool = True) -> Dict:
        """Test single image enhancement"""
        print("\n🖼️  Testing single image enhancement...")
        
        try:
            # Create test image
            test_image = self.create_test_medical_image()
            
            # Convert to PIL and then to bytes
            pil_image = Image.fromarray(test_image)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            # Prepare request
            files = {'image': ('test_medical.png', img_data, 'image/png')}
            
            start_time = time.time()
            response = self.session.post(
                f"{self.api_url}/enhance",
                files=files,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            result = {
                'test_name': 'single_enhancement',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'error': None
            }
            
            if result['success']:
                response_data = response.json()
                
                # Extract metrics
                metrics = response_data.get('metrics', {})
                clinical_score = metrics.get('clinical_quality_score', 0)
                improvement_pct = metrics.get('improvement_percentage', 0)
                
                result['response_data'] = {
                    'processing_time': response_data.get('processing_time_seconds', 0),
                    'clinical_score': clinical_score,
                    'improvement_percentage': improvement_pct,
                    'psnr': metrics.get('psnr', 0),
                    'ssim': metrics.get('ssim', 0),
                    'target_42_achieved': improvement_pct >= 42.0
                }
                
                print(f"   ✅ Enhancement successful")
                print(f"   ⏱️  Processing time: {response_data.get('processing_time_seconds', 0):.3f}s")
                print(f"   🎯 Clinical Score: {clinical_score:.4f}")
                print(f"   📈 Improvement: {improvement_pct:.1f}%")
                print(f"   📊 PSNR: {metrics.get('psnr', 0):.2f} dB")
                print(f"   📊 SSIM: {metrics.get('ssim', 0):.4f}")
                
                if improvement_pct >= 42.0:
                    print(f"   🎉 42% TARGET ACHIEVED!")
                else:
                    print(f"   📊 Progress toward 42% target: {improvement_pct/42*100:.1f}%")
                
                # Save enhanced image if requested
                if save_result and 'enhanced_image' in response_data:
                    enhanced_data = base64.b64decode(response_data['enhanced_image'])
                    with open('test_enhanced_result.png', 'wb') as f:
                        f.write(enhanced_data)
                    print(f"   💾 Enhanced image saved: test_enhanced_result.png")
                    
            else:
                result['error'] = response.text
                print(f"   ❌ Enhancement failed: HTTP {response.status_code}")
                print(f"   📝 Error: {result['error'][:200]}")
                
        except Exception as e:
            result = {
                'test_name': 'single_enhancement',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ Enhancement error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_base64_enhancement(self) -> Dict:
        """Test base64 image enhancement"""
        print("\n🔗 Testing base64 image enhancement...")
        
        try:
            # Create test image
            test_image = self.create_test_medical_image()
            pil_image = Image.fromarray(test_image)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Prepare request
            payload = {'image': img_base64}
            
            start_time = time.time()
            response = self.session.post(
                f"{self.api_url}/enhance_base64",
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            result = {
                'test_name': 'base64_enhancement',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'error': None
            }
            
            if result['success']:
                response_data = response.json()
                result['response_data'] = {
                    'processing_time': response_data.get('processing_time_seconds', 0),
                    'has_enhanced_image': 'enhanced_image' in response_data
                }
                
                print(f"   ✅ Base64 enhancement successful")
                print(f"   ⏱️  Processing time: {response_data.get('processing_time_seconds', 0):.3f}s")
            else:
                result['error'] = response.text
                print(f"   ❌ Base64 enhancement failed: HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'test_name': 'base64_enhancement',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ Base64 enhancement error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_batch_enhancement(self, num_images: int = 3) -> Dict:
        """Test batch image enhancement"""
        print(f"\n📦 Testing batch enhancement ({num_images} images)...")
        
        try:
            # Create multiple test images
            files = []
            for i in range(num_images):
                test_image = self.create_test_medical_image()
                # Add slight variation
                noise = np.random.normal(0, 10, test_image.shape)
                test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
                
                pil_image = Image.fromarray(test_image)
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='PNG')
                
                files.append(('images', (f'test_batch_{i}.png', img_buffer.getvalue(), 'image/png')))
            
            start_time = time.time()
            response = self.session.post(
                f"{self.api_url}/batch_enhance",
                files=files,
                timeout=self.timeout * 2  # Longer timeout for batch
            )
            response_time = time.time() - start_time
            
            result = {
                'test_name': 'batch_enhancement',
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200,
                'error': None
            }
            
            if result['success']:
                response_data = response.json()
                summary = response_data.get('summary', {})
                
                result['response_data'] = {
                    'total_images': summary.get('total_images', 0),
                    'successful': summary.get('successful', 0),
                    'failed': summary.get('failed', 0),
                    'processing_time': summary.get('processing_time_seconds', 0),
                    'avg_time_per_image': summary.get('avg_time_per_image', 0)
                }
                
                print(f"   ✅ Batch enhancement successful")
                print(f"   📊 Total images: {summary.get('total_images', 0)}")
                print(f"   ✅ Successful: {summary.get('successful', 0)}")
                print(f"   ❌ Failed: {summary.get('failed', 0)}")
                print(f"   ⏱️  Total time: {summary.get('processing_time_seconds', 0):.3f}s")
                print(f"   ⚡ Avg per image: {summary.get('avg_time_per_image', 0):.3f}s")
            else:
                result['error'] = response.text
                print(f"   ❌ Batch enhancement failed: HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'test_name': 'batch_enhancement',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ Batch enhancement error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_performance_load(self, num_requests: int = 10) -> Dict:
        """Test API performance under load"""
        print(f"\n⚡ Testing performance load ({num_requests} requests)...")
        
        try:
            # Create single test image for reuse
            test_image = self.create_test_medical_image()
            pil_image = Image.fromarray(test_image)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
            start_time = time.time()
            
            for i in range(num_requests):
                try:
                    files = {'image': ('test_load.png', img_data, 'image/png')}
                    
                    req_start = time.time()
                    response = self.session.post(
                        f"{self.api_url}/enhance",
                        files=files,
                        timeout=self.timeout
                    )
                    req_time = time.time() - req_start
                    
                    if response.status_code == 200:
                        successful_requests += 1
                        response_times.append(req_time)
                    else:
                        failed_requests += 1
                    
                    if (i + 1) % 3 == 0:
                        print(f"   📊 Completed {i + 1}/{num_requests} requests...")
                        
                except Exception:
                    failed_requests += 1
            
            total_time = time.time() - start_time
            
            result = {
                'test_name': 'performance_load',
                'success': successful_requests > 0,
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'total_time': total_time,
                'avg_response_time': np.mean(response_times) if response_times else 0,
                'min_response_time': np.min(response_times) if response_times else 0,
                'max_response_time': np.max(response_times) if response_times else 0,
                'success_rate': (successful_requests / num_requests) * 100,
                'throughput': successful_requests / total_time if total_time > 0 else 0
            }
            
            print(f"   ✅ Load test completed")
            print(f"   📊 Success rate: {result['success_rate']:.1f}%")
            print(f"   ⏱️  Avg response time: {result['avg_response_time']:.3f}s")
            print(f"   🚀 Throughput: {result['throughput']:.2f} req/s")
            print(f"   📈 Response time range: {result['min_response_time']:.3f}s - {result['max_response_time']:.3f}s")
            
        except Exception as e:
            result = {
                'test_name': 'performance_load',
                'success': False,
                'error': str(e),
                'total_requests': num_requests
            }
            print(f"   ❌ Load test error: {e}")
        
        self.test_results.append(result)
        return result
    
    def test_api_stats(self) -> Dict:
        """Test API statistics endpoint"""
        print("\n📊 Testing API statistics...")
        
        try:
            response = self.session.get(f"{self.api_url}/stats", timeout=10)
            
            result = {
                'test_name': 'api_stats',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': response.status_code == 200,
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if result['success']:
                stats = result['response_data']
                print(f"   ✅ API stats retrieved")
                print(f"   📈 Total requests: {stats.get('total_requests', 0)}")
                print(f"   ✅ Successful enhancements: {stats.get('successful_enhancements', 0)}")
                print(f"   ❌ Failed requests: {stats.get('failed_requests', 0)}")
                print(f"   ⏱️  Avg processing time: {stats.get('average_processing_time', 0):.3f}s")
            else:
                print(f"   ❌ API stats failed: HTTP {response.status_code}")
                
        except Exception as e:
            result = {
                'test_name': 'api_stats',
                'success': False,
                'error': str(e),
                'status_code': None,
                'response_time': None
            }
            print(f"   ❌ API stats error: {e}")
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_tests(self, save_results: bool = True, load_test: bool = True) -> Dict:
        """Run all comprehensive tests"""
        print("🚀 Starting comprehensive Medical GAN API testing...")
        print("="*80)
        
        # Run all tests
        self.test_health_check()
        time.sleep(1)
        
        self.test_model_info()
        time.sleep(1)
        
        self.test_single_enhancement(save_result=save_results)
        time.sleep(2)
        
        self.test_base64_enhancement()
        time.sleep(2)
        
        self.test_batch_enhancement(num_images=3)
        time.sleep(2)
        
        if load_test:
            self.test_performance_load(num_requests=10)
            time.sleep(1)
        
        self.test_api_stats()
        
        # Generate summary
        summary = self._generate_test_summary()
        
        print("\n" + "="*80)
        print("🎯 TEST SUMMARY")
        print("="*80)
        
        print(f"📊 Total tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        
        if summary['avg_response_time'] > 0:
            print(f"⏱️  Average response time: {summary['avg_response_time']:.3f}s")
        
        # Check 42% target achievement
        enhancement_results = [r for r in self.test_results 
                             if r['test_name'] == 'single_enhancement' and r['success']]
        
        if enhancement_results:
            result = enhancement_results[0]
            if 'response_data' in result:
                improvement = result['response_data'].get('improvement_percentage', 0)
                if improvement >= 42.0:
                    print(f"🎉 42% IMPROVEMENT TARGET ACHIEVED! ({improvement:.1f}%)")
                else:
                    print(f"📊 Current improvement: {improvement:.1f}% (Target: 42%)")
        
        # Save results
        if save_results:
            self._save_test_results()
        
        return summary
    
    def _generate_test_summary(self) -> Dict:
        """Generate comprehensive test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.get('success', False))
        failed_tests = total_tests - passed_tests
        
        response_times = [
            result['response_time'] for result in self.test_results 
            if result.get('response_time') is not None
        ]
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'test_timestamp': datetime.now().isoformat(),
            'api_url': self.api_url,
            'detailed_results': self.test_results
        }
        
        return summary
    
    def _save_test_results(self):
        """Save detailed test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"medical_gan_api_test_results_{timestamp}.json"
        
        summary = self._generate_test_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved: {filename}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(
        description='Test Medical Image Enhancement GAN API'
    )
    
    parser.add_argument('--url', default='http://localhost:5000',
                       help='API URL to test')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Request timeout in seconds')
    parser.add_argument('--no-load-test', action='store_true',
                       help='Skip performance load testing')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving test results')
    parser.add_argument('--quick', action='store_true',
                       help='Run only basic tests')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MedicalGANAPITester(api_url=args.url, timeout=args.timeout)
    
    # Run tests
    if args.quick:
        print("🏃 Running quick tests...")
        tester.test_health_check()
        tester.test_single_enhancement(save_result=not args.no_save)
    else:
        tester.run_comprehensive_tests(
            save_results=not args.no_save,
            load_test=not args.no_load_test
        )
    
    # Determine exit code
    summary = tester._generate_test_summary()
    exit_code = 0 if summary['success_rate'] == 100 else 1
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)