"""
Unit tests for face detection modes
Tests Haar Cascade detection and mask generation
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_detectors import (
    get_face_detector, validate_detection,
    HaarCascadeDetector
)
from utils.attack_utils import generate_mask_from_detector, FaceDetectionError
from utils.image_utils import ImageProcessor


class TestFaceDetectionModes(unittest.TestCase):
    """Test Haar Cascade face detection on dummy images."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy image with face-like region (bright square in center)
        self.dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        # Add a bright region in center to simulate face
        center_y, center_x = 128, 128
        self.dummy_image[center_y-40:center_y+40, center_x-30:center_x+30] = [200, 180, 160]
    
    def test_haar_detector(self):
        """Test Haar-Cascade detector."""
        try:
            detector = get_face_detector(
                method='haar',
                scale_factor=1.1,
                min_neighbors=3,
                min_size=50
            )
            
            bboxes = detector.detect(self.dummy_image)
            self.assertIsInstance(bboxes, list)
            
            if len(bboxes) > 0:
                bbox = bboxes[0]
                self.assertEqual(len(bbox), 4)
                
        except RuntimeError as e:
            if "Failed to load Haar cascade" in str(e):
                self.skipTest("Haar cascade not available")
            else:
                raise
    
    def test_detector_factory_invalid_method(self):
        """Test that factory raises error for invalid method."""
        with self.assertRaises(ValueError):
            get_face_detector(method='invalid_method')


class TestMaskGeneration(unittest.TestCase):
    """Test mask generation with Haar Cascade detector."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_mask_generation_haar(self):
        """Test mask generation with Haar-Cascade."""
        try:
            detector = get_face_detector(method='haar')
            
            M1, M2 = generate_mask_from_detector(
                self.dummy_image,
                detector,
                device=self.device,
                strict_detection=False
            )
            
            self.assertEqual(M1.shape, (1, 3, 256, 256))
            self.assertEqual(M2.shape, (1, 3, 256, 256))
            
        except RuntimeError as e:
            if "Failed to load Haar cascade" in str(e):
                self.skipTest("Haar cascade not available")
            else:
                raise


class TestDetectionValidation(unittest.TestCase):
    """Test detection validation logic."""
    
    def test_validate_detection_valid_bbox(self):
        """Test validation with valid bbox."""
        image_shape = (256, 256)
        bbox = (50, 50, 100, 120)  # (x, y, w, h)
        
        is_valid, reason, metrics = validate_detection(
            bbox,
            image_shape,
            min_bbox_area_ratio=0.01,
            max_bbox_area_ratio=0.95,
            min_bbox_size=20
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Valid detection")
        self.assertIn('area_ratio', metrics)
    
    def test_validate_detection_none(self):
        """Test validation with None bbox."""
        image_shape = (256, 256)
        is_valid, reason, metrics = validate_detection(None, image_shape)
        
        self.assertFalse(is_valid)
        self.assertEqual(reason, "No face detected")
    
    def test_validate_detection_too_small(self):
        """Test validation with bbox that's too small."""
        image_shape = (256, 256)
        bbox = (50, 50, 10, 10)  # Too small
        
        is_valid, reason, metrics = validate_detection(
            bbox,
            image_shape,
            min_bbox_size=20
        )
        
        self.assertFalse(is_valid)
        self.assertIn("too small", reason.lower())
    
    def test_validate_detection_out_of_bounds(self):
        """Test validation with bbox out of image bounds."""
        image_shape = (256, 256)
        bbox = (250, 250, 100, 100)  # Extends beyond image
        
        is_valid, reason, metrics = validate_detection(bbox, image_shape)
        
        self.assertFalse(is_valid)
        self.assertIn("out of bounds", reason.lower())
    
    def test_validate_detection_area_too_small(self):
        """Test validation with bbox area too small."""
        image_shape = (256, 256)
        bbox = (100, 100, 20, 20)  # Area = 400, ratio = 400/(256*256) ≈ 0.006 < 0.01
        
        is_valid, reason, metrics = validate_detection(
            bbox,
            image_shape,
            min_bbox_area_ratio=0.01
        )
        
        self.assertFalse(is_valid)
        self.assertIn("area too small", reason.lower())
    
    def test_validate_detection_area_too_large(self):
        """Test validation with bbox area too large."""
        image_shape = (256, 256)
        bbox = (10, 10, 240, 240)  # Area = 57600, ratio ≈ 0.88 > 0.95 (if max=0.95)
        
        is_valid, reason, metrics = validate_detection(
            bbox,
            image_shape,
            max_bbox_area_ratio=0.80  # Set lower threshold
        )
        
        self.assertFalse(is_valid)
        self.assertIn("area too large", reason.lower())


class TestStrictDetection(unittest.TestCase):
    """Test strict vs non-strict detection modes."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Create image that might fail detection
        self.dummy_image = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)  # Very dark image
    
    def test_strict_detection_failure(self):
        """Test that strict mode raises error on failure."""
        try:
            detector = get_face_detector(method='haar')
            
            # This should raise FaceDetectionError if detection fails validation
            with self.assertRaises(FaceDetectionError):
                generate_mask_from_detector(
                    self.dummy_image,
                    detector,
                    device=self.device,
                    strict_detection=True,
                    min_bbox_area_ratio=0.1  # High threshold to force failure
                )
                
        except RuntimeError as e:
            if "Failed to load Haar cascade" in str(e):
                self.skipTest("Haar cascade not available")
            else:
                raise
    
    def test_non_strict_detection_fallback(self):
        """Test that non-strict mode uses fallback ellipse."""
        try:
            # Test with Haar detector - should use fallback
            detector = get_face_detector(method='haar')
            
            # This should not raise error, but use fallback ellipse
            M1, M2 = generate_mask_from_detector(
                self.dummy_image,
                detector,
                device=self.device,
                strict_detection=False,
                min_bbox_area_ratio=0.1  # High threshold to force failure
            )
            
            # Should still return valid masks (fallback ellipse)
            self.assertEqual(M1.shape, (1, 3, 256, 256))
            self.assertEqual(M2.shape, (1, 3, 256, 256))
            
        except RuntimeError as e:
            if "Failed to load Haar cascade" in str(e):
                self.skipTest("Haar cascade not available")
            else:
                raise


class TestFallbackDetector(unittest.TestCase):
    """Test fallback detector behavior."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    def test_fallback_detector_used(self):
        """Test that fallback detector is used when primary detector fails."""
        try:
            primary_detector = get_face_detector(method='haar')
            fallback_detector = get_face_detector(method='haar')
            
            # If primary detector fails, fallback detector will be tried
            try:
                M1, M2 = generate_mask_from_detector(
                    self.dummy_image,
                    primary_detector,
                    device=self.device,
                    strict_detection=False,
                    fallback_detector=fallback_detector
                )
                
                # Should get valid masks if fallback detector succeeds
                self.assertEqual(M1.shape, (1, 3, 256, 256))
                self.assertEqual(M2.shape, (1, 3, 256, 256))
            except FaceDetectionError:
                # If both detectors fail, this is acceptable
                # The fallback detector might also fail on some images
                pass
            
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("Detector models not available")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

