"""
Image processing utilities for MegaFS
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union
import os


class ImageProcessor:
    """Image processing utilities for face swapping"""
    
    @staticmethod
    def load_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
        """Load and preprocess an image"""
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"ERROR: Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if target_size:
                image = cv2.resize(image, target_size)
            
            return image
            
        except Exception as e:
            print(f"ERROR: Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def preprocess_for_model(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to tensor and normalize
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0
        
        if normalize:
            tensor = (tensor - 0.5) / 0.5
        
        return tensor
    
    @staticmethod
    def postprocess_from_model(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
        """Postprocess model output to image"""
        # Denormalize if needed
        if denormalize:
            tensor = tensor * 0.5 + 0.5
        
        # Clamp to valid range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy array
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> bool:
        """Save image to file"""
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            cv2.imwrite(save_path, image_bgr)
            print(f"SUCCESS: Image saved: {save_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Error saving image {save_path}: {e}")
            return False


class ImageLoader:
    """Image loading and caching utilities"""
    
    def __init__(self, cache_size: int = 100):
        self.cache = {}
        self.cache_size = cache_size
    
    def load_image_pair(self, src_path: str, tgt_path: str, 
                       target_size: Tuple[int, int] = (256, 256)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load source and target image pair"""
        src_image = ImageProcessor.load_image(src_path, target_size)
        tgt_image = ImageProcessor.load_image(tgt_path, target_size)
        
        return src_image, tgt_image
    
    def get_cached_image(self, image_path: str) -> Optional[np.ndarray]:
        """Get image from cache"""
        return self.cache.get(image_path)
    
    def cache_image(self, image_path: str, image: np.ndarray):
        """Cache an image"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[image_path] = image
