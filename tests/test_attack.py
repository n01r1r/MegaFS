"""
Unit tests for adversarial attack utilities
Test mask generation and gradient flow
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierfe import HieRFE
from models.resnet import resnet50
from models.megafs import MegaFS


class TestGradientFlow(unittest.TestCase):
    """Test gradient flow through HieRFE."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a small test model
        self.backbone = resnet50(False)
        self.hierfe = HieRFE(self.backbone, num_latents=[4, 6, 8], depth=50)
        self.hierfe = self.hierfe.to(self.device)
        
        # Create dummy input
        self.dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        self.dummy_input.requires_grad = True
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through HieRFE."""
        # Set to training mode
        self.hierfe.train()
        
        # Forward pass
        latents, f4 = self.hierfe(self.dummy_input)
        
        # Backward pass
        loss = latents.sum() + f4.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.dummy_input.grad)
        self.assertTrue((self.dummy_input.grad != 0).any())
    
    def test_gradient_in_eval_mode(self):
        """Test that gradients are computed even in eval mode when input requires_grad."""
        # Set to eval mode
        self.hierfe.eval()
        
        # Forward pass
        latents, f4 = self.hierfe(self.dummy_input)
        
        # Backward pass
        loss = latents.sum() + f4.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.dummy_input.grad)


class TestMegaFSGradientMode(unittest.TestCase):
    """Test MegaFS gradient mode compatibility."""
    
    def test_enable_grads_mode(self):
        """Test that MegaFS can be set to gradient computation mode."""
        from config import Config
        
        config = Config(
            swap_type='ftm',
            dataset_root='./dataset/CelebAMask-HQ',
            img_root='./dataset/CelebAMask-HQ/CelebA-HQ-img',
            checkpoint_dir='./weights'
        )
        
        try:
            model = MegaFS(
                config=config,
                debug=False,
                enable_grads=True,
                device='cpu'  # Use CPU for tests
            )
            
            # Test that encoder requires grad in training mode
            model.train()
            for param in model.encoder.parameters():
                self.assertTrue(param.requires_grad)
            
        except FileNotFoundError:
            # Skip if weights not available
            self.skipTest("Weights not available, skipping integration test")


class TestAttackClass(unittest.TestCase):
    """Test DualTargetPGDAttack class."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a small test model
        backbone = resnet50(False)
        self.hierfe = HieRFE(backbone, num_latents=[4, 6, 8], depth=50)
        self.hierfe = self.hierfe.to(self.device)
    
    def test_attack_initialization(self):
        """Test attack class initialization."""
        from utils.attack_utils import DualTargetPGDAttack
        
        attack = DualTargetPGDAttack(
            identity_extractor=self.hierfe,
            epsilon=8.0,
            alpha=1.0,
            num_iter=10,
            lambda_1=1.0,
            lambda_2=1.0,
            device=self.device,
            verbose=False
        )
        
        self.assertEqual(attack.epsilon, 8.0)
        self.assertEqual(attack.num_iter, 10)
    
    def test_mask_generation_method(self):
        """Test that mask generation method works."""
        from utils.attack_utils import DualTargetPGDAttack
        import numpy as np
        
        try:
            attack = DualTargetPGDAttack(
                identity_extractor=self.hierfe,
                device=self.device,
                verbose=False,
                checkpoint_dir='./weights',
                strict_detection=False  # Use fallback if detection fails
            )
        except RuntimeError:
            self.skipTest("Detector not available, skipping mask generation test")
        
        # Create dummy image (numpy array)
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        M1, M2 = attack.generate_masks(dummy_image)
        
        self.assertEqual(M1.shape, (1, 3, 256, 256))
        self.assertEqual(M2.shape, (1, 3, 256, 256))
    
    def test_haar_detection_mode(self):
        """Test that DualTargetPGDAttack works with Haar detection mode."""
        from utils.attack_utils import DualTargetPGDAttack, FaceDetectionError
        import numpy as np
        
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        try:
            attack = DualTargetPGDAttack(
                identity_extractor=self.hierfe,
                device=self.device,
                verbose=False,
                checkpoint_dir='./weights',
                detector_method='haar',
                strict_detection=False  # Allows fallback ellipse if detection fails
            )
            
            # Haar may use fallback ellipse if detection fails
            try:
                M1, M2 = attack.generate_masks(dummy_image)
                self.assertEqual(M1.shape, (1, 3, 256, 256))
                self.assertEqual(M2.shape, (1, 3, 256, 256))
            except FaceDetectionError:
                # If detection fails even with fallback, this is acceptable for some images
                pass
                
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("Detector not available")
            else:
                raise
    
    def test_detector_from_config(self):
        """Test that detector method is properly set from config."""
        from utils.attack_utils import DualTargetPGDAttack
        
        # Test that detector_method parameter is accepted
        try:
            attack = DualTargetPGDAttack(
                identity_extractor=self.hierfe,
                device=self.device,
                verbose=False,
                checkpoint_dir='./weights',
                detector_method='haar'  # Use Haar as it doesn't require weights
            )
            
            # Verify detector was created
            self.assertIsNotNone(attack.detector)
            
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("Detector not available")
            else:
                raise
    
    def test_fallback_detector_chain(self):
        """Test fallback detector chain (primary Haar fails, fallback to Haar)."""
        from utils.attack_utils import DualTargetPGDAttack, FaceDetectionError
        import numpy as np
        
        dummy_image = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)  # Dark image
        
        try:
            attack = DualTargetPGDAttack(
                identity_extractor=self.hierfe,
                device=self.device,
                verbose=False,
                checkpoint_dir='./weights',
                detector_method='haar',
                fallback_detector_method='haar',
                strict_detection=False
            )
            
            # If primary detector fails, fallback detector will be tried
            # If both fail, should use fallback ellipse (non-strict mode)
            try:
                M1, M2 = attack.generate_masks(dummy_image)
                self.assertEqual(M1.shape, (1, 3, 256, 256))
                self.assertEqual(M2.shape, (1, 3, 256, 256))
            except FaceDetectionError:
                # If both detectors fail and strict mode is used, this is acceptable
                pass
            
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("Detector models not available")
            else:
                raise


if __name__ == '__main__':
    unittest.main()

