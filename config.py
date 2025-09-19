"""
Configuration management for MegaFS
Centralized configuration for easier debugging and analysis
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    size: int = 1024
    latent_split: list = None
    num_latents: int = 18
    swap_indice: int = 4
    num_blocks: int = 1
    
    def __post_init__(self):
        if self.latent_split is None:
            self.latent_split = [4, 6, 8]


@dataclass
class PathConfig:
    """Path configuration for datasets and weights"""
    dataset_root: str = ""
    img_root: str = ""
    mask_root: str = ""
    checkpoint_dir: str = "weights"
    data_map_path: str = "data_map.json"
    
    def __post_init__(self):
        if not self.dataset_root and self.img_root:
            self.dataset_root = os.path.dirname(self.img_root)


@dataclass
class SwapConfig:
    """Swap method configuration"""
    swap_type: str = "ftm"  # ftm, injection, lcr
    refine: bool = True
    
    @property
    def is_valid_swap_type(self) -> bool:
        return self.swap_type in ["ftm", "injection", "lcr"]


class Config:
    """Main configuration class"""
    
    def __init__(self, 
                 swap_type: str = "ftm",
                 dataset_root: str = "",
                 img_root: str = "",
                 mask_root: str = "",
                 checkpoint_dir: str = "weights"):
        
        self.swap = SwapConfig(swap_type=swap_type)
        self.paths = PathConfig(
            dataset_root=dataset_root,
            img_root=img_root,
            mask_root=mask_root,
            checkpoint_dir=checkpoint_dir
        )
        self.model = ModelConfig()
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        if not self.swap.is_valid_swap_type:
            raise ValueError(f"Invalid swap_type: {self.swap.swap_type}. Must be one of: ftm, injection, lcr")
        
        if not os.path.exists(self.paths.checkpoint_dir):
            print(f"WARNING: Checkpoint directory not found: {self.paths.checkpoint_dir}")
    
    def get_weight_filename(self) -> str:
        """Get the weight filename for the current swap type"""
        return f"{self.swap.swap_type}_final.pth"
    
    def get_stylegan2_filename(self) -> str:
        """Get the StyleGAN2 weight filename"""
        return "stylegan2-ffhq-config-f.pth"
    
    def print_config(self):
        """Print current configuration"""
        print("INFO: MegaFS Configuration:")
        print(f"  Swap Type: {self.swap.swap_type}")
        print(f"  Dataset Root: {self.paths.dataset_root}")
        print(f"  Image Root: {self.paths.img_root}")
        print(f"  Mask Root: {self.paths.mask_root}")
        print(f"  Checkpoint Dir: {self.paths.checkpoint_dir}")
        print(f"  Weight File: {self.get_weight_filename()}")
        print(f"  StyleGAN2 File: {self.get_stylegan2_filename()}")


# Default configurations for different environments
DEFAULT_CONFIGS = {
    "colab": Config(
        swap_type="ftm",
        dataset_root="/content/CelebAMask-HQ",
        img_root="/content/CelebAMask-HQ/CelebA-HQ-img",
        mask_root="/content/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
        checkpoint_dir="/content/drive/MyDrive/Datasets/weights"
    ),
    "local": Config(
        swap_type="ftm",
        dataset_root="./CelebAMask-HQ",
        img_root="./CelebAMask-HQ/CelebA-HQ-img",
        mask_root="./CelebAMask-HQ/CelebAMask-HQ-mask-anno",
        checkpoint_dir="./weights"
    )
}
