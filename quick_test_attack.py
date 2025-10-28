"""
Quick test script for adversarial attack framework
Run this to verify everything is working
"""

import os
import sys
import torch
from models.hierfe import HieRFE

def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from models.resnet import resnet50
        from utils.attack_utils import DualTargetPGDAttack, generate_mask_from_fpn
        from config import Config
        from models.megafs import MegaFS
        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_mask_generation():
    """Test mask generation from FPN features."""
    print("\n" + "=" * 60)
    print("Testing mask generation...")
    print("=" * 60)
    
    try:
        from models.resnet import resnet50
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create model
        backbone = resnet50(False)
        hierfe = HieRFE(backbone, num_latents=[2, 6, 6], depth=50).to(device)
        hierfe.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        dummy_norm = (dummy_input - dummy_input.min()) / (dummy_input.max() - dummy_input.min()) * 2 - 1
        
        # Generate masks
        from utils.attack_utils import generate_mask_from_fpn
        M1, M2 = generate_mask_from_fpn(
            hierfe, 
            dummy_norm, 
            output_size=(256, 256),
            feature_layers=['f8'],
            device=device
        )
        
        # Verify shapes
        assert M1.shape == (1, 3, 256, 256), f"M1 shape incorrect: {M1.shape}"
        assert M2.shape == (1, 3, 256, 256), f"M2 shape incorrect: {M2.shape}"
        
        # Verify values
        assert (M1 >= 0).all() and (M1 <= 1).all(), "M1 not in [0, 1]"
        assert (M2 >= 0).all() and (M2 <= 1).all(), "M2 not in [0, 1]"
        
        # Verify complementary
        mask_diff = torch.abs(M1 + M2 - torch.ones_like(M1))
        assert (mask_diff < 1e-4).all(), "M1 and M2 not complementary"
        
        print("[OK] Mask generation successful")
        print(f"  M1 shape: {M1.shape}")
        print(f"  M1 (face): {M1.sum()/M1.numel():.2%}")
        print(f"  M2 (background): {M2.sum()/M2.numel():.2%}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Mask generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow through HieRFE."""
    print("\n" + "=" * 60)
    print("Testing gradient flow...")
    print("=" * 60)
    
    try:
        from models.resnet import resnet50
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        backbone = resnet50(False)
        hierfe = HieRFE(backbone, num_latents=[2, 6, 6], depth=50).to(device)
        hierfe.train()
        
        # Create input with gradients
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        dummy_input.requires_grad = True
        
        # Forward pass
        latents, f4 = hierfe(dummy_input)
        
        # Backward pass
        loss = latents.sum() + f4.sum()
        loss.backward()
        
        # Check gradients
        assert dummy_input.grad is not None, "Gradients not computed"
        assert (dummy_input.grad != 0).any(), "Gradients are all zeros"
        
        print("[OK] Gradient flow successful")
        print(f"  Latents shape: {latents.shape}")
        print(f"  F4 shape: {f4.shape}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Gradient flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attack_initialization():
    """Test DualTargetPGDAttack initialization."""
    print("\n" + "=" * 60)
    print("Testing attack initialization...")
    print("=" * 60)
    
    try:
        from models.resnet import resnet50
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create model
        backbone = resnet50(False)
        hierfe = HieRFE(backbone, num_latents=[4, 6, 8], depth=50).to(device)
        
        # Initialize attack
        from utils.attack_utils import DualTargetPGDAttack
        attack = DualTargetPGDAttack(
            identity_extractor=hierfe,
            epsilon=8.0,
            alpha=1.0,
            num_iter=10,
            lambda_1=1.0,
            lambda_2=1.0,
            device=device,
            verbose=False
        )
        
        print("[OK] Attack initialization successful")
        print(f"  Epsilon: {attack.epsilon}")
        print(f"  Alpha: {attack.alpha}")
        print(f"  Iterations: {attack.num_iter}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Attack initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration file loading."""
    print("\n" + "=" * 60)
    print("Testing configuration loading...")
    print("=" * 60)
    
    try:
        import yaml
        
        config_path = 'configs/attack_config.yaml'
        
        if not os.path.exists(config_path):
            print(f"[WARN] Config file not found: {config_path}")
            print("  This is okay if you haven't created it yet")
            return True
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        assert 'attack' in config, "Missing 'attack' section"
        assert 'mask_generation' in config, "Missing 'mask_generation' section"
        assert 'paths' in config, "Missing 'paths' section"
        
        print("[OK] Configuration loaded successfully")
        print(f"  Attack type: {config['attack'].get('type', 'N/A')}")
        print(f"  Epsilon: {config['attack'].get('epsilon', 'N/A')}")
        print(f"  Num iter: {config['attack'].get('num_iter', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all quick tests."""
    print("\n" + "=" * 60)
    print("HieRFE Dual-Target Adversarial Attack - Quick Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Mask Generation", test_mask_generation()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Attack Init", test_attack_initialization()))
    results.append(("Config Loading", test_config_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! You're ready to run attacks.")
        print("\nNext steps:")
        print("  1. python experiments/run_attack.py --image-id 2332")
        print("  2. python experiments/evaluate_attack.py --image-id 2332")
    else:
        print("\n[WARN] Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

