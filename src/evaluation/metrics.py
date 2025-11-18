import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

class MedicalImageMetrics:
    """
    Comprehensive Medical Image Quality Assessment
    Target: 42% improvement validation + Clinical assessment for radiologists
    """
    
    def __init__(self):
        self.clinical_thresholds = {
            'excellent_psnr': 30.0,      # dB
            'good_psnr': 25.0,           # dB  
            'excellent_ssim': 0.90,      # 0-1
            'good_ssim': 0.80,           # 0-1
            'diagnostic_sharpness': 1.2,  # ratio
            'diagnostic_contrast': 1.15   # ratio
        }
        
        print("📈 MedicalImageMetrics initialized!")
        print(f"🎯 Clinical Thresholds:")
        for metric, threshold in self.clinical_thresholds.items():
            print(f"   {metric}: {threshold}")
    
    def psnr(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio (technical quality)"""
        try:
            # Ensure same shape
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Ensure 2D for PSNR calculation
            if len(original.shape) > 2:
                original = original.squeeze()
            if len(enhanced.shape) > 2:
                enhanced = enhanced.squeeze()
            
            return peak_signal_noise_ratio(original, enhanced, data_range=1.0)
        except Exception as e:
            print(f"⚠️  PSNR calculation error: {e}")
            return 0.0
    
    def ssim(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate Structural Similarity Index (perceptual quality)"""
        try:
            # Ensure same shape
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Ensure 2D for SSIM calculation
            if len(original.shape) > 2:
                original = original.squeeze()
            if len(enhanced.shape) > 2:
                enhanced = enhanced.squeeze()
            
            return structural_similarity(original, enhanced, data_range=1.0)
        except Exception as e:
            print(f"⚠️  SSIM calculation error: {e}")
            return 0.0
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            if len(image.shape) > 2:
                image = image.squeeze()
            
            # Convert to uint8 for OpenCV
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
            return np.var(laplacian)
        except Exception as e:
            print(f"⚠️  Sharpness calculation error: {e}")
            return 0.0
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using standard deviation"""
        try:
            if len(image.shape) > 2:
                image = image.squeeze()
            
            return np.std(image)
        except Exception as e:
            print(f"⚠️  Contrast calculation error: {e}")
            return 0.0
    
    def calculate_edge_preservation(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate edge preservation index (critical for medical diagnosis)"""
        try:
            # Ensure same shape
            if original.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            
            # Ensure 2D
            if len(original.shape) > 2:
                original = original.squeeze()
            if len(enhanced.shape) > 2:
                enhanced = enhanced.squeeze()
            
            # Calculate gradients using Sobel operator
            orig_grad_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
            orig_grad_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
            orig_gradient = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
            
            enh_grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            enh_grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            enh_gradient = np.sqrt(enh_grad_x**2 + enh_grad_y**2)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(orig_gradient.flatten(), enh_gradient.flatten())[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"⚠️  Edge preservation calculation error: {e}")
            return 0.0
    
    def calculate_noise_reduction(self, low_quality: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate noise reduction effectiveness"""
        try:
            # Ensure same shape
            if low_quality.shape != enhanced.shape:
                enhanced = cv2.resize(enhanced, (low_quality.shape[1], low_quality.shape[0]))
            
            # Ensure 2D
            if len(low_quality.shape) > 2:
                low_quality = low_quality.squeeze()
            if len(enhanced.shape) > 2:
                enhanced = enhanced.squeeze()
            
            # Estimate noise using high-pass filtering
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            
            noise_low = cv2.filter2D(low_quality, -1, kernel)
            noise_enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            noise_power_low = np.mean(noise_low**2)
            noise_power_enhanced = np.mean(noise_enhanced**2)
            
            if noise_power_low > 0:
                return noise_power_low / (noise_power_enhanced + 1e-8)
            else:
                return 1.0
                
        except Exception as e:
            print(f"⚠️  Noise reduction calculation error: {e}")
            return 1.0
    
    def calculate_diagnostic_quality_index(self, original: np.ndarray, 
                                         low_quality: np.ndarray,
                                         enhanced: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive diagnostic quality metrics
        Target: Clinical Score > 1.42 for 42% improvement
        """
        metrics = {}
        
        # Basic quality metrics
        metrics['psnr'] = self.psnr(original, enhanced)
        metrics['ssim'] = self.ssim(original, enhanced)
        
        # Medical-specific metrics
        metrics['sharpness_original'] = self.calculate_sharpness(original)
        metrics['sharpness_enhanced'] = self.calculate_sharpness(enhanced)
        metrics['sharpness_improvement'] = (
            metrics['sharpness_enhanced'] / (metrics['sharpness_original'] + 1e-8)
        )
        
        metrics['contrast_original'] = self.calculate_contrast(original)
        metrics['contrast_enhanced'] = self.calculate_contrast(enhanced)
        metrics['contrast_improvement'] = (
            metrics['contrast_enhanced'] / (metrics['contrast_original'] + 1e-8)
        )
        
        metrics['edge_preservation'] = self.calculate_edge_preservation(original, enhanced)
        metrics['noise_reduction'] = self.calculate_noise_reduction(low_quality, enhanced)
        
        # Enhancement effectiveness (vs low quality input)
        metrics['enhancement_psnr'] = self.psnr(low_quality, enhanced)
        metrics['enhancement_ssim'] = self.ssim(low_quality, enhanced)
        
        # Calculate improvement ratios
        psnr_low_vs_orig = self.psnr(low_quality, original)
        psnr_enh_vs_orig = metrics['psnr']
        
        metrics['psnr_improvement_ratio'] = psnr_enh_vs_orig / (psnr_low_vs_orig + 1e-8)
        
        ssim_low_vs_orig = self.ssim(low_quality, original)
        ssim_enh_vs_orig = metrics['ssim']
        
        metrics['ssim_improvement_ratio'] = ssim_enh_vs_orig / (ssim_low_vs_orig + 1e-8)
        
        # Clinical Quality Score (weighted combination)
        # Target: > 1.42 for 42% improvement
        clinical_weights = {
            'psnr_normalized': 0.25,        # Technical quality
            'ssim': 0.25,                   # Perceptual quality  
            'sharpness_improvement': 0.20,  # Diagnostic detail
            'contrast_improvement': 0.15,   # Tissue differentiation
            'edge_preservation': 0.15       # Anatomical structure
        }
        
        # Normalize PSNR to 0-1 range
        psnr_normalized = min(metrics['psnr'] / 35.0, 1.0)
        
        clinical_score = (
            clinical_weights['psnr_normalized'] * psnr_normalized +
            clinical_weights['ssim'] * metrics['ssim'] +
            clinical_weights['sharpness_improvement'] * min(metrics['sharpness_improvement'] / 2.0, 1.0) +
            clinical_weights['contrast_improvement'] * min(metrics['contrast_improvement'] / 2.0, 1.0) +
            clinical_weights['edge_preservation'] * metrics['edge_preservation']
        )
        
        metrics['clinical_quality_score'] = clinical_score
        
        # Calculate overall improvement percentage
        improvement_percentage = (clinical_score - 1.0) * 100
        metrics['improvement_percentage'] = max(0.0, improvement_percentage)
        
        # Clinical assessment flags
        metrics['clinical_assessment'] = {
            'excellent_psnr': metrics['psnr'] >= self.clinical_thresholds['excellent_psnr'],
            'good_psnr': metrics['psnr'] >= self.clinical_thresholds['good_psnr'],
            'excellent_ssim': metrics['ssim'] >= self.clinical_thresholds['excellent_ssim'],
            'good_ssim': metrics['ssim'] >= self.clinical_thresholds['good_ssim'],
            'diagnostic_sharpness': metrics['sharpness_improvement'] >= self.clinical_thresholds['diagnostic_sharpness'],
            'diagnostic_contrast': metrics['contrast_improvement'] >= self.clinical_thresholds['diagnostic_contrast'],
            'target_42_achieved': improvement_percentage >= 42.0
        }
        
        return metrics
    
    def batch_evaluation(self, original_batch: np.ndarray,
                        low_quality_batch: np.ndarray,
                        enhanced_batch: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a batch of images and return averaged metrics
        """
        batch_metrics = []
        
        batch_size = min(original_batch.shape[0], 8)  # Limit for performance
        
        for i in range(batch_size):
            try:
                # Extract individual images
                orig = original_batch[i]
                low = low_quality_batch[i] 
                enh = enhanced_batch[i]
                
                # Ensure 2D images
                if len(orig.shape) > 2:
                    orig = orig.squeeze()
                if len(low.shape) > 2:
                    low = low.squeeze()
                if len(enh.shape) > 2:
                    enh = enh.squeeze()
                
                # Calculate metrics for this image
                image_metrics = self.calculate_diagnostic_quality_index(orig, low, enh)
                batch_metrics.append(image_metrics)
                
            except Exception as e:
                print(f"⚠️  Error evaluating image {i}: {e}")
                continue
        
        if not batch_metrics:
            return {}
        
        # Average all metrics
        averaged_metrics = {}
        
        # Get all metric keys (excluding nested dicts)
        metric_keys = [k for k in batch_metrics[0].keys() if k != 'clinical_assessment']
        
        for key in metric_keys:
            values = [m[key] for m in batch_metrics if key in m]
            if values:
                averaged_metrics[key] = np.mean(values)
                averaged_metrics[f'{key}_std'] = np.std(values)
        
        # Average clinical assessment flags
        clinical_flags = {}
        if 'clinical_assessment' in batch_metrics[0]:
            for flag in batch_metrics[0]['clinical_assessment'].keys():
                flag_values = [m['clinical_assessment'][flag] for m in batch_metrics]
                clinical_flags[flag] = np.mean(flag_values)  # Percentage of images meeting criteria
        
        averaged_metrics['clinical_assessment'] = clinical_flags
        
        return averaged_metrics
    
    def generate_clinical_report(self, metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> str:
        """
        Generate human-readable clinical assessment report
        """
        report = []
        report.append("=" * 60)
        report.append("MEDICAL IMAGE ENHANCEMENT - CLINICAL ASSESSMENT REPORT")
        report.append("=" * 60)
        
        # Overall Assessment
        clinical_score = metrics.get('clinical_quality_score', 0)
        improvement_pct = metrics.get('improvement_percentage', 0)
        
        report.append(f"\n🎯 OVERALL ASSESSMENT:")
        report.append(f"   Clinical Quality Score: {clinical_score:.4f}")
        report.append(f"   Image Improvement: {improvement_pct:.1f}%")
        
        if improvement_pct >= 42.0:
            report.append(f"   ✅ TARGET ACHIEVED! (≥42% improvement)")
        else:
            report.append(f"   📈 Progress toward 42% target: {improvement_pct/42*100:.1f}%")
        
        # Technical Metrics
        report.append(f"\n📊 TECHNICAL METRICS:")
        report.append(f"   PSNR: {metrics.get('psnr', 0):.2f} dB")
        report.append(f"   SSIM: {metrics.get('ssim', 0):.4f}")
        
        # Clinical Metrics
        report.append(f"\n🏥 CLINICAL METRICS:")
        report.append(f"   Sharpness Improvement: {metrics.get('sharpness_improvement', 1):.3f}x")
        report.append(f"   Contrast Improvement: {metrics.get('contrast_improvement', 1):.3f}x")
        report.append(f"   Edge Preservation: {metrics.get('edge_preservation', 0):.4f}")
        report.append(f"   Noise Reduction: {metrics.get('noise_reduction', 1):.3f}x")
        
        # Clinical Assessment
        if 'clinical_assessment' in metrics:
            clinical = metrics['clinical_assessment']
            report.append(f"\n✅ CLINICAL QUALITY ASSESSMENT:")
            
            quality_flags = [
                ('Excellent PSNR (≥30dB)', clinical.get('excellent_psnr', False)),
                ('Good PSNR (≥25dB)', clinical.get('good_psnr', False)),
                ('Excellent SSIM (≥0.9)', clinical.get('excellent_ssim', False)),
                ('Good SSIM (≥0.8)', clinical.get('good_ssim', False)),
                ('Diagnostic Sharpness', clinical.get('diagnostic_sharpness', False)),
                ('Diagnostic Contrast', clinical.get('diagnostic_contrast', False))
            ]
            
            for description, passed in quality_flags:
                status = "✅ PASS" if passed else "❌ FAIL"
                if isinstance(passed, float):
                    status = f"{passed*100:.1f}% of images"
                report.append(f"   {description}: {status}")
        
        # Recommendations
        report.append(f"\n💡 CLINICAL RECOMMENDATIONS:")
        
        if improvement_pct >= 42:
            report.append("   🎉 Enhancement quality suitable for clinical use")
            report.append("   📋 Ready for radiologist validation study")
        elif improvement_pct >= 30:
            report.append("   📈 Significant improvement achieved")
            report.append("   🔧 Minor optimization may reach 42% target")
        elif improvement_pct >= 20:
            report.append("   ⚠️  Moderate improvement, continue training")
            report.append("   🎯 Focus on perceptual loss optimization")
        else:
            report.append("   ❌ Insufficient improvement for clinical use")
            report.append("   🔄 Review model architecture and training parameters")
        
        # Diagnostic Impact Assessment
        report.append(f"\n🔬 DIAGNOSTIC IMPACT ASSESSMENT:")
        
        psnr = metrics.get('psnr', 0)
        ssim = metrics.get('ssim', 0)
        
        if psnr >= 30 and ssim >= 0.9:
            report.append("   🏆 EXCELLENT - Likely to improve diagnostic confidence")
        elif psnr >= 25 and ssim >= 0.8:
            report.append("   ✅ GOOD - Should maintain diagnostic accuracy")
        elif psnr >= 20 and ssim >= 0.7:
            report.append("   ⚠️  MODERATE - May provide some diagnostic benefit")
        else:
            report.append("   ❌ POOR - Not recommended for clinical use")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📋 Clinical report saved: {save_path}")
        
        return report_text
    
    def visualize_metrics(self, metrics: Dict[str, float], 
                         save_path: Optional[str] = None):
        """
        Create visualization of key metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Overall Quality Radar Chart
        categories = ['PSNR\n(Technical)', 'SSIM\n(Perceptual)', 
                     'Sharpness\n(Detail)', 'Contrast\n(Tissue)',
                     'Edge Preservation\n(Structure)']
        
        values = [
            min(metrics.get('psnr', 0) / 35.0, 1.0),  # Normalize PSNR
            metrics.get('ssim', 0),
            min(metrics.get('sharpness_improvement', 1) / 2.0, 1.0),
            min(metrics.get('contrast_improvement', 1) / 2.0, 1.0),
            metrics.get('edge_preservation', 0)
        ]
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax = axes[0, 0]
        ax = plt.subplot(2, 2, 1, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Clinical Quality Assessment', size=12, weight='bold', pad=20)
        
        # Plot 2: Improvement Progress
        ax = axes[0, 1]
        improvement_pct = metrics.get('improvement_percentage', 0)
        
        bars = ax.bar(['Current\nImprovement', '42% Target'], 
                     [improvement_pct, 42], 
                     color=['green' if improvement_pct >= 42 else 'orange', 'lightblue'])
        
        ax.set_ylabel('Improvement Percentage (%)')
        ax.set_title('42% Target Progress', weight='bold')
        ax.set_ylim(0, max(50, improvement_pct + 5))
        
        # Add value labels on bars
        for bar, value in zip(bars, [improvement_pct, 42]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', weight='bold')
        
        # Plot 3: Clinical Thresholds
        ax = axes[1, 0]
        
        metrics_names = ['PSNR\n(≥30 Excellent)', 'SSIM\n(≥0.9 Excellent)', 
                        'Sharpness\n(≥1.2x Good)', 'Contrast\n(≥1.15x Good)']
        current_values = [
            metrics.get('psnr', 0),
            metrics.get('ssim', 0),
            metrics.get('sharpness_improvement', 0),
            metrics.get('contrast_improvement', 0)
        ]
        thresholds = [30, 0.9, 1.2, 1.15]
        
        x_pos = np.arange(len(metrics_names))
        bars = ax.bar(x_pos, current_values, 
                     color=['green' if current_values[i] >= thresholds[i] else 'orange' 
                           for i in range(len(current_values))])
        
        # Add threshold lines
        for i, threshold in enumerate(thresholds):
            ax.axhline(y=threshold, xmin=i/len(thresholds), xmax=(i+1)/len(thresholds), 
                      color='red', linestyle='--', alpha=0.7)
        
        ax.set_ylabel('Metric Values')
        ax.set_title('Clinical Threshold Achievement', weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names, fontsize=9)
        
        # Plot 4: Enhancement Comparison
        ax = axes[1, 1]
        
        comparison_metrics = ['PSNR\nImprovement', 'SSIM\nImprovement', 'Overall\nClinical Score']
        comparison_values = [
            metrics.get('psnr_improvement_ratio', 1),
            metrics.get('ssim_improvement_ratio', 1),
            metrics.get('clinical_quality_score', 0)
        ]
        
        bars = ax.bar(comparison_metrics, comparison_values,
                     color=['darkgreen', 'darkblue', 'purple'])
        
        ax.set_ylabel('Ratio / Score')
        ax.set_title('Enhancement Effectiveness', weight='bold')
        ax.axhline(y=1.42, color='red', linestyle='--', alpha=0.7, label='42% Target')
        ax.legend()
        
        # Add value labels
        for bar, value in zip(bars, comparison_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Metrics visualization saved: {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing MedicalImageMetrics...")
    
    # Initialize metrics calculator
    metrics_calc = MedicalImageMetrics()
    
    # Create test images
    original = np.random.rand(256, 256)
    low_quality = original + np.random.normal(0, 0.1, original.shape)
    low_quality = np.clip(low_quality, 0, 1)
    
    # Simulate enhancement (better than input, not perfect)
    enhanced = original + np.random.normal(0, 0.05, original.shape)
    enhanced = np.clip(enhanced, 0, 1)
    
    print("🧪 Calculating test metrics...")
    
    # Calculate comprehensive metrics
    test_metrics = metrics_calc.calculate_diagnostic_quality_index(
        original, low_quality, enhanced
    )
    
    print(f"📊 Test Results:")
    for key, value in test_metrics.items():
        if key != 'clinical_assessment':
            print(f"   {key}: {value:.4f}")
    
    # Generate clinical report
    print("\n📋 Generating clinical report...")
    report = metrics_calc.generate_clinical_report(test_metrics)
    print(report)
    
    # Test batch evaluation
    print("\n🔬 Testing batch evaluation...")
    batch_orig = np.random.rand(4, 256, 256, 1)
    batch_low = batch_orig + np.random.normal(0, 0.1, batch_orig.shape)
    batch_enh = batch_orig + np.random.normal(0, 0.05, batch_orig.shape)
    
    batch_metrics = metrics_calc.batch_evaluation(batch_orig, batch_low, batch_enh)
    
    print(f"📊 Batch Results:")
    for key, value in batch_metrics.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue:.4f}")
        else:
            print(f"   {key}: {value:.4f}")
    
    print("\n✅ MedicalImageMetrics test completed!")