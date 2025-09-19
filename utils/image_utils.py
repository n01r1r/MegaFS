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
    def align_and_resize(image: np.ndarray, output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Align face roughly using OpenCV Haar cascades, then resize.
        Fallbacks to center-crop + resize if detection fails.
        Expects RGB image, returns RGB image.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
            if len(faces) > 0:
                # pick largest face
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

                # simple crop around face with margin
                box_size = int(max(w, h) * 1.6)
                cx, cy = x + w // 2, y + h // 2
                x1 = max(0, cx - box_size // 2)
                y1 = max(0, cy - box_size // 2)
                x2 = min(image.shape[1], x1 + box_size)
                y2 = min(image.shape[0], y1 + box_size)

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = image
                aligned = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
                return aligned

            # fallback center-crop
            h, w = image.shape[:2]
            side = min(h, w)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            crop = image[y1:y1+side, x1:x1+side]
            return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)

        except Exception:
            # last resort: simple resize
            return cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    
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
