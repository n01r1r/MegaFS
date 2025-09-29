import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("WARNING: lpips not available. LPIPS metric will be disabled.")
    LPIPS_AVAILABLE = False
    lpips = None


class PerceptualLoss(nn.Module):
    def __init__(self, net='alex', use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.use_gpu = use_gpu
        self.lpips_available = LPIPS_AVAILABLE
        
        if self.lpips_available:
            if use_gpu and torch.cuda.is_available():
                self.loss_fn = lpips.LPIPS(net=net).cuda()
            else:
                self.loss_fn = lpips.LPIPS(net=net)
        else:
            self.loss_fn = None
    
    def forward(self, img0, img1):
        if self.lpips_available and self.loss_fn is not None:
            return self.loss_fn(img0, img1)
        else:
            return torch.tensor(float('inf'), device=img0.device)


class ImageMetrics:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        try:
            self.lpips_loss = PerceptualLoss(net='alex', use_gpu=self.use_gpu)
            self.lpips_available = True
        except Exception as e:
            print(f"WARNING: LPIPS not available: {e}")
            self.lpips_available = False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = image_tensor / 255.0 * 2.0 - 1.0
        else:
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor / 255.0 * 2.0 - 1.0
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        if not self.lpips_available:
            return float('inf')
        
        try:
            img1_tensor = self.preprocess_image(img1)
            img2_tensor = self.preprocess_image(img2)
            
            with torch.no_grad():
                lpips_score = self.lpips_loss(img1_tensor, img2_tensor)
                return lpips_score.item()
        except Exception as e:
            print(f"ERROR: LPIPS calculation failed: {e}")
            return float('inf')
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            img1_float = img1.astype(np.float64) / 255.0
            img2_float = img2.astype(np.float64) / 255.0
            
            return psnr(img1_float, img2_float, data_range=1.0)
        except Exception as e:
            print(f"ERROR: PSNR calculation failed: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            return ssim(img1_gray, img2_gray, data_range=255)
        except Exception as e:
            print(f"ERROR: SSIM calculation failed: {e}")
            return 0.0
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            img1_float = img1.astype(np.float64)
            img2_float = img2.astype(np.float64)
            
            mse = np.mean((img1_float - img2_float) ** 2)
            return mse
        except Exception as e:
            print(f"ERROR: MSE calculation failed: {e}")
            return float('inf')
    
    def calculate_all_metrics(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['lpips'] = self.calculate_lpips(img1, img2)
        metrics['psnr'] = self.calculate_psnr(img1, img2)
        metrics['ssim'] = self.calculate_ssim(img1, img2)
        metrics['mse'] = self.calculate_mse(img1, img2)
        
        return metrics


class FaceSwapEvaluator:
    def __init__(self, use_gpu=True):
        self.metrics_calculator = ImageMetrics(use_gpu=use_gpu)
        self.results = []
    
    def evaluate_pair(self, source_img: np.ndarray, target_img: np.ndarray, 
                     swapped_img: np.ndarray, refined_img: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        results = {}
        
        results['swapped_vs_target'] = self.metrics_calculator.calculate_all_metrics(target_img, swapped_img)
        results['swapped_vs_source'] = self.metrics_calculator.calculate_all_metrics(source_img, swapped_img)
        
        if refined_img is not None:
            results['refined_vs_target'] = self.metrics_calculator.calculate_all_metrics(target_img, refined_img)
            results['refined_vs_source'] = self.metrics_calculator.calculate_all_metrics(source_img, refined_img)
        
        return results
    
    def evaluate_batch(self, image_pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]) -> List[Dict[str, Dict[str, float]]]:
        batch_results = []
        
        for i, (source_img, target_img, swapped_img, refined_img) in enumerate(image_pairs):
            print(f"Evaluating pair {i+1}/{len(image_pairs)}")
            try:
                pair_results = self.evaluate_pair(source_img, target_img, swapped_img, refined_img)
                batch_results.append(pair_results)
            except Exception as e:
                print(f"ERROR: Failed to evaluate pair {i+1}: {e}")
                batch_results.append({})
        
        return batch_results
    
    def calculate_statistics(self, results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        stats = {}
        
        metric_values = {}
        for result in results:
            for comparison_type, metrics in result.items():
                if comparison_type not in metric_values:
                    metric_values[comparison_type] = {}
                for metric_name, value in metrics.items():
                    if metric_name not in metric_values[comparison_type]:
                        metric_values[comparison_type][metric_name] = []
                    # Safely handle non-numeric values
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric_value != float('inf') and not np.isnan(numeric_value):
                        metric_values[comparison_type][metric_name].append(numeric_value)
        
        for comparison_type, metrics in metric_values.items():
            stats[comparison_type] = {}
            for metric_name, values in metrics.items():
                if values:
                    stats[comparison_type][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'count': len(values)
                    }
                else:
                    stats[comparison_type][metric_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'count': 0
                    }
        
        return stats


def save_evaluation_results(results: Dict, save_path: str):
    import json
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy(results)
    
    try:
        with open(save_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        print(f"SUCCESS: Evaluation results saved to {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}")


def load_evaluation_results(load_path: str) -> Dict:
    import json
    
    try:
        with open(load_path, 'r') as f:
            results = json.load(f)
        print(f"SUCCESS: Evaluation results loaded from {load_path}")
        return results
    except Exception as e:
        print(f"ERROR: Failed to load results: {e}")
        return {}
