"""
Face detection abstraction layer for MegaFS
Supports Haar Cascade face detection with unified interface
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class FaceDetector(ABC):
    """Base class for face detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array [H, W, 3] in RGB format, range [0, 255]
            
        Returns:
            List of bounding boxes as (x, y, w, h) tuples in image coordinates
        """
        pass


class HaarCascadeDetector(FaceDetector):
    """Haar-Cascade face detector using OpenCV"""
    
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: Tuple[int, int] = (50, 50)
    ):
        """
        Initialize Haar-Cascade detector.
        
        Args:
            scale_factor: Scale factor for image pyramid
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size (width, height)
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load Haar cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar-Cascade"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # Convert to list of tuples (x, y, w, h)
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append((int(x), int(y), int(w), int(h)))
        
        return bboxes


def validate_detection(
    bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    min_bbox_area_ratio: float = 0.01,
    max_bbox_area_ratio: float = 0.95,
    min_bbox_size: int = 20
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate face detection bounding box.
    
    Args:
        bbox: Bounding box as (x, y, w, h) or None if no detection
        image_shape: Image shape as (height, width)
        min_bbox_area_ratio: Minimum face area as ratio of image (default 0.01 = 1%)
        max_bbox_area_ratio: Maximum face area as ratio of image (default 0.95 = 95%)
        min_bbox_size: Minimum bbox width/height in pixels (default 20)
        
    Returns:
        Tuple of (is_valid: bool, reason: str, metrics: dict)
    """
    h, w = image_shape
    image_area = h * w
    metrics = {}
    
    # Check if bbox is None
    if bbox is None:
        return False, "No face detected", metrics
    
    x, y, bbox_w, bbox_h = bbox
    metrics['bbox'] = {'x': x, 'y': y, 'w': bbox_w, 'h': bbox_h}
    
    # Check bbox size
    if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
        metrics['min_size'] = min_bbox_size
        metrics['actual_size'] = {'w': bbox_w, 'h': bbox_h}
        return False, f"Bbox too small: {bbox_w}x{bbox_h} < {min_bbox_size}x{min_bbox_size}", metrics
    
    # Check bbox coordinates are within image bounds
    if x < 0 or y < 0 or x + bbox_w > w or y + bbox_h > h:
        metrics['image_size'] = {'w': w, 'h': h}
        return False, f"Bbox out of bounds: ({x}, {y}, {x+bbox_w}, {y+bbox_h}) vs image ({w}, {h})", metrics
    
    # Calculate area ratio
    bbox_area = bbox_w * bbox_h
    area_ratio = bbox_area / image_area
    metrics['area_ratio'] = area_ratio
    metrics['bbox_area'] = bbox_area
    metrics['image_area'] = image_area
    
    # Check area ratio
    if area_ratio < min_bbox_area_ratio:
        metrics['min_area_ratio'] = min_bbox_area_ratio
        return False, f"Face area too small: {area_ratio:.4f} < {min_bbox_area_ratio:.4f} ({area_ratio*100:.2f}% < {min_bbox_area_ratio*100:.2f}%)", metrics
    
    if area_ratio > max_bbox_area_ratio:
        metrics['max_area_ratio'] = max_bbox_area_ratio
        return False, f"Face area too large: {area_ratio:.4f} > {max_bbox_area_ratio:.4f} ({area_ratio*100:.2f}% > {max_bbox_area_ratio*100:.2f}%)", metrics
    
    # All checks passed
    return True, "Valid detection", metrics


def get_face_detector(
    method: str = "haar",
    device: str = 'cuda',
    checkpoint_dir: str = "weights",
    **kwargs
) -> FaceDetector:
    """
    Factory function to get face detector instance.
    
    Args:
        method: Detector method ("haar")
        device: Device for detectors ('cuda' or 'cpu') - not used for Haar
        checkpoint_dir: Directory containing detector weights - not used for Haar
        **kwargs: Additional detector-specific parameters
        
    Returns:
        FaceDetector instance
        
    Raises:
        ValueError: If method is not supported
    """
    if method == "haar":
        scale_factor = kwargs.get('scale_factor', 1.1)
        min_neighbors = kwargs.get('min_neighbors', 3)
        min_size = kwargs.get('min_size', 50)
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        return HaarCascadeDetector(
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size
        )
    
    else:
        raise ValueError(f"Unsupported detector method: {method}. Supported: 'haar'")

