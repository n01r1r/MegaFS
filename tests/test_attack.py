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

from utils.attack_utils import generate_mask_from_blazeface
from models.hierfe import HieRFE
from models.resnet import resnet50
from models.megafs import MegaFS
from models.blazeface import get_blazeface_model


class TestMaskGeneration(unittest.TestCase):
    """Test BlazeFace-based mask generation."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy image (numpy array)
        self.dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Try to load BlazeFace model
        try:
            self.blazeface_model = get_blazeface_model(device=self.device, checkpoint_dir='./weights')
        except Exception:
            self.blazeface_model = None
            self.skipTest("BlazeFace model not available, skipping mask generation tests")
        
    def test_mask_shape(self):
        """Test that generated masks have correct shape."""
        if self.blazeface_model is None:
            self.skipTest("BlazeFace model not available")
            
        M1, M2 = generate_mask_from_blazeface(
            self.dummy_image,
            self.blazeface_model,
            device=self.device
        )
        
        # Check shapes
        self.assertEqual(M1.shape, (1, 3, 256, 256))
        self.assertEqual(M2.shape, (1, 3, 256, 256))
        
        # Check values in [0, 1]
        self.assertTrue((M1 >= 0).all())
        self.assertTrue((M1 <= 1).all())
        self.assertTrue((M2 >= 0).all())
        self.assertTrue((M2 <= 1).all())
        
        # Check complementary (approximately)
        mask_diff = torch.abs(M1 + M2 - torch.ones_like(M1))
        self.assertTrue((mask_diff < 1e-4).all())
    
    def test_mask_detached(self):
        """Test that masks are detached from computation graph."""
        if self.blazeface_model is None:
            self.skipTest("BlazeFace model not available")
            
        M1, M2 = generate_mask_from_blazeface(
            self.dummy_image,
            self.blazeface_model,
            device=self.device
        )
        
        # Check requires_grad is False
        self.assertFalse(M1.requires_grad)
        self.assertFalse(M2.requires_grad)


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
                checkpoint_dir='./weights'
            )
        except RuntimeError:
            self.skipTest("BlazeFace model not available, skipping mask generation test")
        
        # Create dummy image (numpy array)
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        M1, M2 = attack.generate_masks(dummy_image)
        
        self.assertEqual(M1.shape, (1, 3, 256, 256))
        self.assertEqual(M2.shape, (1, 3, 256, 256))


if __name__ == '__main__':
    unittest.main()

