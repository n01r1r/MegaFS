"""
Weight loading utilities for MegaFS models
Separated for better debugging and modularity
"""

import os
import torch
from typing import Dict, Any, Optional


class WeightLoader:
    """Base class for weight loading operations"""
    
    def __init__(self, checkpoint_dir: str = "weights"):
        self.checkpoint_dir = checkpoint_dir
    
    def load_checkpoint(self, filename: str, required_keys: list = None) -> Optional[Dict[str, Any]]:
        """Load a checkpoint file and verify required keys"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"ERROR: Weight file not found: {filepath}")
            return None
            
        try:
            checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
            print(f"SUCCESS: Loaded weights: {filename}")
            
            if required_keys:
                missing_keys = [k for k in required_keys if k not in checkpoint]
                if missing_keys:
                    print(f"WARNING: Missing keys in {filename}: {missing_keys}")
                else:
                    print(f"SUCCESS: All required keys present: {required_keys}")
            
            return checkpoint
            
        except Exception as e:
            print(f"ERROR: Failed to load {filename}: {e}")
            return None


class FTMWeightLoader(WeightLoader):
    """Weight loader for Face Transfer Module"""
    
    def load_ftm_weights(self) -> Optional[Dict[str, Any]]:
        """Load FTM encoder and swapper weights"""
        return self.load_checkpoint("ftm_final.pth", required_keys=["e", "s"])


class InjectionWeightLoader(WeightLoader):
    """Weight loader for Injection method"""
    
    def load_injection_weights(self) -> Optional[Dict[str, Any]]:
        """Load Injection encoder and swapper weights"""
        return self.load_checkpoint("injection_final.pth", required_keys=["e", "s"])


class LCRWeightLoader(WeightLoader):
    """Weight loader for LCR method"""
    
    def load_lcr_weights(self) -> Optional[Dict[str, Any]]:
        """Load LCR encoder and swapper weights"""
        return self.load_checkpoint("lcr_final.pth", required_keys=["e", "s"])


class StyleGAN2WeightLoader(WeightLoader):
    """Weight loader for StyleGAN2 generator"""
    
    def load_stylegan2_weights(self) -> Optional[Dict[str, Any]]:
        """Load StyleGAN2 generator weights"""
        return self.load_checkpoint("stylegan2-ffhq-config-f.pth", required_keys=["g_ema"])


def verify_all_weights(checkpoint_dir: str = "weights") -> bool:
    """Verify all required weight files are present and loadable"""
    print("INFO: Verifying all weight files...")
    
    loaders = [
        FTMWeightLoader(checkpoint_dir),
        InjectionWeightLoader(checkpoint_dir),
        LCRWeightLoader(checkpoint_dir),
        StyleGAN2WeightLoader(checkpoint_dir)
    ]
    
    all_valid = True
    
    for loader in loaders:
        if isinstance(loader, StyleGAN2WeightLoader):
            weights = loader.load_stylegan2_weights()
        elif isinstance(loader, FTMWeightLoader):
            weights = loader.load_ftm_weights()
        elif isinstance(loader, InjectionWeightLoader):
            weights = loader.load_injection_weights()
        elif isinstance(loader, LCRWeightLoader):
            weights = loader.load_lcr_weights()
        else:
            weights = None
        
        if weights is None:
            all_valid = False
    
    if all_valid:
        print("SUCCESS: All weight files verified successfully")
    else:
        print("ERROR: Some weight files are missing or invalid")
    
    return all_valid
