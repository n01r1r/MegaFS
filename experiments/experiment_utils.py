"""
Utility functions for experiments.
Extracted from run_attack.py to improve modularity.
"""

import os
import sys
import yaml
import json
from typing import Dict, Any, List, Optional
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.megafs import MegaFS


def load_config(config_path: str) -> Dict[str, Any]:
    """Load attack configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict[str, Any]) -> MegaFS:
    """Setup MegaFS model with gradients enabled."""
    # Create config object
    cfg = Config(
        swap_type=config['model']['swap_type'],
        dataset_root=config['paths']['dataset_root'],
        img_root=config['paths']['img_root'],
        mask_root=config['paths']['mask_root'],
        checkpoint_dir=config['paths']['checkpoint_dir']
    )
    
    # Initialize model with gradients enabled
    model = MegaFS(
        config=cfg,
        debug=True,
        enable_grads=config['model']['enable_grads'],
        device=config['device']
    )
    
    return model


def get_image_path(image_id: int, img_root: str) -> str:
    """Get full path to image file."""
    # Prefer non-padded filename (e.g., 2332.jpg). Fallback to 5-digit padded if needed.
    non_padded = os.path.join(img_root, f"{image_id}.jpg")
    if os.path.exists(non_padded):
        return non_padded
    padded = os.path.join(img_root, f"{image_id:05d}.jpg")
    return padded if os.path.exists(padded) else non_padded


def _format_val(val: float) -> str:
    """Format float value for directory names."""
    as_float = float(val)
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float).replace('.', 'p')


def _format_attack_dirname(cfg: Dict[str, Any], prefix: str = "") -> str:
    """Generate a directory name that captures all relevant attack hyperparameters."""
    attack_cfg = cfg['attack']
    parts = []
    if prefix:
        parts.append(prefix)
    parts.extend([
        f"l1_{_format_val(attack_cfg.get('lambda_1', 0.0))}",
        f"l2_{_format_val(attack_cfg.get('lambda_2', 0.0))}",
        f"lsim_{_format_val(attack_cfg.get('lambda_sim', 0.0))}",
        f"ltv_{_format_val(attack_cfg.get('lambda_tv', 0.0))}",
        f"e_{_format_val(attack_cfg.get('epsilon', 0.0))}",
        f"iter_{int(attack_cfg.get('num_iter', 0))}",
    ])
    alpha = attack_cfg.get('alpha')
    if alpha is not None:
        parts.append(f"a_{_format_val(alpha)}")
    sem_variant = attack_cfg.get('sem_variant')
    if sem_variant:
        parts.append(f"sem_{sem_variant}")
    target_type = attack_cfg.get('target_type')
    if target_type and target_type != 'image':
        parts.append(f"target_{target_type}")
    if attack_cfg.get('maximize_similarity'):
        parts.append("simmax")
    if attack_cfg.get('random_init'):
        parts.append("randinit")
    # Include mask blur if available
    edge_blur = cfg.get('mask_generation', {}).get('edge_blur')
    if edge_blur:
        parts.append(f"blur_{int(edge_blur)}")
    return "_".join(parts)


def make_exp_output_dir(base_dir: str, cfg: Dict[str, Any], image_id: int) -> str:
    """Create and return a flat output directory for a given config and image id."""
    exp_name = _format_attack_dirname(cfg, prefix=f"{int(image_id):05d}")
    odir = os.path.join(base_dir, exp_name)
    os.makedirs(odir, exist_ok=True)
    return odir


def make_exp_dir(base_dir: str, cfg: Dict[str, Any]) -> str:
    """Create and return experiment directory without image subfolder."""
    exp_name = f"exp_{_format_attack_dirname(cfg)}"
    edir = os.path.join(base_dir, exp_name)
    os.makedirs(edir, exist_ok=True)
    return edir
