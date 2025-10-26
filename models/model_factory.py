"""
Model factory for creating MegaFS components
Centralized model creation for better debugging
"""

import torch
from typing import Optional, Dict, Any
from .hierfe import HieRFE
from .face_transfer import FaceTransferModule
from .generator import Generator
from .resnet import resnet50
from .weight_loaders import FTMWeightLoader, InjectionWeightLoader, LCRWeightLoader, StyleGAN2WeightLoader


class ModelFactory:
    """Factory class for creating and loading MegaFS model components"""
    
    def __init__(self, checkpoint_dir: str = "weights", device: str = "cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.weight_loaders = {
            "ftm": FTMWeightLoader(checkpoint_dir),
            "injection": InjectionWeightLoader(checkpoint_dir),
            "lcr": LCRWeightLoader(checkpoint_dir),
            "stylegan2": StyleGAN2WeightLoader(checkpoint_dir)
        }
        # Weight loading method mapping for better maintainability
        self.load_methods = {
            "ftm": "load_ftm_weights",
            "injection": "load_injection_weights",
            "lcr": "load_lcr_weights"
        }
    
    def _load_weights_for_type(self, loader, swap_type: str) -> Optional[Dict]:
        """Load weights for a given swap type using the appropriate method"""
        method_name = self.load_methods.get(swap_type)
        if method_name and hasattr(loader, method_name):
            return getattr(loader, method_name)()
        return None
    
    def create_encoder(self, swap_type: str) -> HieRFE:
        """Create and load encoder model"""
        print(f"INFO: Creating encoder for {swap_type}...")
        
        # Encoder configuration
        latent_split = [4, 6, 8]
        encoder = HieRFE(resnet50(False), num_latents=latent_split, depth=50)
        
        # Move to device
        device = torch.device(self.device)
        encoder = encoder.to(device)
        
        # Load weights using centralized method
        loader = self.weight_loaders[swap_type]
        weights = self._load_weights_for_type(loader, swap_type)
        
        if weights and "e" in weights:
            # Use strict=True like original
            encoder.load_state_dict(weights["e"], strict=True)
            print(f"SUCCESS: Encoder weights loaded for {swap_type}")
        else:
            print(f"WARNING: No encoder weights found for {swap_type}")
        
        encoder.eval()
        return encoder
    
    def create_swapper(self, swap_type: str) -> FaceTransferModule:
        """Create and load swapper model"""
        print(f"INFO: Creating swapper for {swap_type}...")
        
        # Swapper configuration
        num_blocks = 3 if swap_type == "ftm" else 1
        num_latents = 18
        swap_indice = 4
        
        swapper = FaceTransferModule(
            num_blocks=num_blocks,
            swap_indice=swap_indice,
            num_latents=num_latents,
            typ=swap_type
        )
        
        # Move to device
        device = torch.device(self.device)
        swapper = swapper.to(device)
        
        # Load weights using centralized method
        loader = self.weight_loaders[swap_type]
        weights = self._load_weights_for_type(loader, swap_type)
        
        if weights and "s" in weights:
            state_dict = weights["s"]
            if swap_type == "injection":
                import re
                remapped = {}
                pattern = re.compile(r"(blocks\.\d+\.(att_path[12])\.(\d+)\.)0\.")
                for k, v in state_dict.items():
                    # Collapse only the extra '.0.' immediately after att_pathX.<idx>
                    new_key = pattern.sub(r"\\1", k)
                    remapped[new_key] = v
                # Load non-strict to tolerate any remaining mismatches
                missing = swapper.load_state_dict(remapped, strict=False)
            else:
                missing = swapper.load_state_dict(state_dict, strict=True)
            print(f"SUCCESS: Swapper weights loaded for {swap_type}")
        else:
            print(f"WARNING: No swapper weights found for {swap_type}")
        
        swapper.eval()
        return swapper
    
    def create_generator(self) -> Generator:
        """Create and load StyleGAN2 generator"""
        print("INFO: Creating StyleGAN2 generator...")
        
        # Generator configuration
        size = 1024
        generator = Generator(size, 512, 8, channel_multiplier=2)
        
        # Move to device
        device = torch.device(self.device)
        generator = generator.to(device)
        
        # Load weights
        loader = self.weight_loaders["stylegan2"]
        weights = loader.load_stylegan2_weights()
        
        if weights and "g_ema" in weights:
            generator.load_state_dict(weights["g_ema"], strict=False)
            print("SUCCESS: StyleGAN2 generator weights loaded")
        else:
            print("WARNING: No StyleGAN2 generator weights found")
        
        generator.eval()
        return generator
    
    def create_all_models(self, swap_type: str) -> Dict[str, torch.nn.Module]:
        """Create all model components for a given swap type"""
        print(f"INFO: Creating all models for {swap_type}...")
        
        models = {
            "encoder": self.create_encoder(swap_type),
            "swapper": self.create_swapper(swap_type),
            "generator": self.create_generator()
        }
        
        print(f"SUCCESS: All models created for {swap_type}")
        return models
