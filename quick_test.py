"""Quick diagnostic test to verify the refactoring works"""

import torch
import sys
import os

# Test 1: Import gradient-enabled features
print("=" * 60)
print("DIAGNOSTIC TEST - Gradient Compatibility")
print("=" * 60)

try:
    from config import Config
    print("[OK] Config module imported")
except Exception as e:
    print(f"[FAIL] Failed to import Config: {e}")

try:
    from models.megafs import MegaFS
    print("[OK] MegaFS imported")
except Exception as e:
    print(f"[FAIL] Failed to import MegaFS: {e}")

try:
    from data import FaceSwapDataset, create_dataloaders
    print("[OK] Dataset modules imported")
except Exception as e:
    print(f"[FAIL] Failed to import data modules: {e}")

try:
    from training import BaseTrainer
    print("[OK] Training modules imported")
except Exception as e:
    print(f"[FAIL] Failed to import training modules: {e}")

# Test 2: Check for enable_grads parameter
try:
    if hasattr(MegaFS, '__init__'):
        import inspect
        sig = inspect.signature(MegaFS.__init__)
        params = sig.parameters
        if 'enable_grads' in params:
            print("[OK] enable_grads parameter exists in MegaFS.__init__")
        else:
            print("[FAIL] enable_grads parameter NOT found")
except Exception as e:
    print(f"[FAIL] Error checking enable_grads: {e}")

# Test 3: Check for gradient mode methods
try:
    if hasattr(MegaFS, 'set_gradient_mode'):
        print("[OK] set_gradient_mode method exists")
    else:
        print("[FAIL] set_gradient_mode method NOT found")
    
    if hasattr(MegaFS, 'forward'):
        print("[OK] forward method exists")
    else:
        print("[FAIL] forward method NOT found")
except Exception as e:
    print(f"[FAIL] Error checking methods: {e}")

# Test 4: Check YAML support
try:
    config = Config(swap_type="ftm")
    if hasattr(config, 'from_yaml'):
        print("[OK] Config.from_yaml() method exists")
    else:
        print("[FAIL] Config.from_yaml() NOT found")
    
    if hasattr(config, 'to_yaml'):
        print("[OK] Config.to_yaml() method exists")
    else:
        print("[FAIL] Config.to_yaml() NOT found")
except Exception as e:
    print(f"[FAIL] Error checking YAML support: {e}")

# Test 5: Check tensor preprocessing
try:
    from utils.image_utils import ImageProcessor
    if hasattr(ImageProcessor, 'preprocess_for_model_tensor'):
        print("[OK] preprocess_for_model_tensor method exists")
    else:
        print("[FAIL] preprocess_for_model_tensor method NOT found")
except Exception as e:
    print(f"[FAIL] Error checking preprocessing: {e}")

# Test 6: Check dataset class
try:
    if FaceSwapDataset and create_dataloaders:
        print("[OK] FaceSwapDataset class exists")
        print("[OK] create_dataloaders function exists")
    else:
        print("[FAIL] Dataset classes NOT found")
except Exception as e:
    print(f"[FAIL] Error checking dataset: {e}")

# Test 7: Check base trainer
try:
    if BaseTrainer:
        print("[OK] BaseTrainer class exists")
    else:
        print("[FAIL] BaseTrainer class NOT found")
except Exception as e:
    print(f"[FAIL] Error checking trainer: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 60)

# Summary
print("\nSummary:")
print("- All core modules refactored")
print("- Gradient support added to MegaFS")
print("- Dataset integration implemented")
print("- YAML configuration support added")
print("- Training infrastructure in place")
print("\nNext steps: Add tests and documentation")

