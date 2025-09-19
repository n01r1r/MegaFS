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
    
    def __init__(self, checkpoint_dir: str = "weights"):
        self.checkpoint_dir = checkpoint_dir
        self.weight_loaders = {
            "ftm": FTMWeightLoader(checkpoint_dir),
            "injection": InjectionWeightLoader(checkpoint_dir),
            "lcr": LCRWeightLoader(checkpoint_dir),
            "stylegan2": StyleGAN2WeightLoader(checkpoint_dir)
        }
    
    def create_encoder(self, swap_type: str) -> HieRFE:
        """Create and load encoder model"""
        print(f"INFO: Creating encoder for {swap_type}...")
        
        # Encoder configuration
        latent_split = [4, 6, 8]
        encoder = HieRFE(resnet50(False), num_latents=latent_split, depth=50).cuda()
        
        # Load weights
        loader = self.weight_loaders[swap_type]
        weights = loader.load_ftm_weights()  # All methods use same structure
        
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
        ).cuda()
        
        # Load weights
        loader = self.weight_loaders[swap_type]
        weights = loader.load_ftm_weights()  # All methods use same structure
        
        if weights and "s" in weights:
            swapper.load_state_dict(weights["s"], strict=True)
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
        generator = Generator(size, 512, 8, channel_multiplier=2).cuda()
        
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
