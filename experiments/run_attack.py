"""
Adversarial attack runner for HieRFE dual-target strategy
Execute attacks on CelebA-HQ images with configurable parameters
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import Config
from models.megafs import MegaFS
from utils.attack_utils import DualTargetPGDAttack, compute_metrics


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


def run_single_attack(
    config: Dict[str, Any],
    model: MegaFS,
    image_id: int,
    output_dir: str
) -> Dict[str, Any]:
    """Run attack on a single image."""
    print(f"\n{'='*60}")
    print(f"Attacking image ID: {image_id}")
    print(f"{'='*60}")
    
    # Get image path
    img_root = config['paths']['img_root']
    image_path = get_image_path(image_id, img_root)
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Create attack instance
    attack = DualTargetPGDAttack(
        identity_extractor=model.encoder,
        epsilon=config['attack']['epsilon'],
        alpha=config['attack']['alpha'],
        num_iter=config['attack']['num_iter'],
        lambda_1=config['attack']['lambda_1'],
        lambda_2=config['attack']['lambda_2'],
        feature_layers=config['mask_generation']['feature_layers'],
        mask_threshold=config['mask_generation']['threshold'],
        mask_type=config['mask_generation']['mask_type'],
        device=config['device'],
        verbose=config['experiment']['verbose']
    )
    
    # Execute attack
    # Use non-padded naming for outputs to match dataset convention
    output_path = os.path.join(output_dir, f"image_{image_id}")
    os.makedirs(output_path, exist_ok=True)
    
    try:
        adversarial = attack.attack(image_path, output_path)
        
        # Load original for metrics
        from utils.image_utils import ImageProcessor
        original = ImageProcessor.load_image(image_path, target_size=(256, 256))
        
        # Compute metrics
        metrics = compute_metrics(original, adversarial)
        print(f"\nAttack completed!")
        print(f"L2 norm: {metrics.get('L2_norm', 'N/A'):.2f}")
        print(f"L-inf norm: {metrics.get('Linf_norm', 'N/A'):.2f}")
        if 'SSIM' in metrics:
            print(f"SSIM: {metrics['SSIM']:.4f}")
        
        # Save metrics (ensure JSON-serializable types)
        import json
        serializable_metrics = {k: (float(v) if hasattr(v, 'item') else float(v) if isinstance(v, (np.floating,)) else v)
                                for k, v in metrics.items()}
        with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        # After attack: perform swap to observe success/failure
        try:
            img_root = config['paths']['img_root']
            # pick random source different from current image
            candidates = [f for f in os.listdir(img_root) if f.lower().endswith('.jpg')]
            current_name = os.path.basename(image_path)
            candidates = [f for f in candidates if f != current_name]
            if candidates:
                src_file = random.choice(candidates)
                src_path = os.path.join(img_root, src_file)
                from utils.image_utils import ImageProcessor
                src_img = ImageProcessor.load_image(src_path, target_size=(256, 256))
                # Prepare tensors
                src_tensor = ImageProcessor.preprocess_for_model(src_img, normalize=True).unsqueeze(0).to(model.device)
                orig_tgt_tensor = ImageProcessor.preprocess_for_model(original, normalize=True).unsqueeze(0).to(model.device)
                adv_tgt_tensor = ImageProcessor.preprocess_for_model(adversarial, normalize=True).unsqueeze(0).to(model.device)
                # Swap on original and adversarial targets
                swap_orig = model.swap(src_tensor, orig_tgt_tensor, return_tensor=False)
                swap_adv = model.swap(src_tensor, adv_tgt_tensor, return_tensor=False)
                # Save results
                import cv2
                cv2.imwrite(os.path.join(output_path, 'swap_on_original.jpg'), swap_orig[:, :, ::-1])
                cv2.imwrite(os.path.join(output_path, 'swap_on_adversarial.jpg'), swap_adv[:, :, ::-1])
                # Save comparison grid
                comp = np.hstack([src_img, original, adversarial, swap_orig, swap_adv])
                cv2.imwrite(os.path.join(output_path, 'swap_comparison.jpg'), comp[:, :, ::-1])
        except Exception as e:
            print(f"Warning: swap post-check failed: {e}")
        
        return {
            'image_id': image_id,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR during attack: {e}")
        import traceback
        traceback.print_exc()
        return {
            'image_id': image_id,
            'success': False,
            'error': str(e)
        }


def run_batch_attack(
    config: Dict[str, Any],
    model: MegaFS,
    image_ids: list,
    output_dir: str
) -> None:
    """Run attacks on multiple images."""
    results = []
    
    for image_id in image_ids:
        result = run_single_attack(config, model, image_id, output_dir)
        if result:
            results.append(result)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Batch attack summary")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Successful attacks: {sum(1 for r in results if r.get('success', False))}")
    
    if any('metrics' in r for r in results):
        avg_l2 = np.mean([r['metrics']['L2_norm'] for r in results if 'metrics' in r])
        print(f"Average L2 norm: {avg_l2:.2f}")


def main():
    parser = argparse.ArgumentParser(description='HieRFE Dual-Target Adversarial Attack')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml',
                        help='Path to attack configuration file')
    parser.add_argument('--image-id', type=int, default=2332,
                        help='Single image ID to attack')
    parser.add_argument('--batch', type=str, default=None,
                        help='Comma-separated list of image IDs (e.g., "2332,2107")')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    output_dir = config['experiment']['output_dir']
    
    # Override max samples if specified
    if args.max_samples:
        config['experiment']['max_samples'] = args.max_samples
    
    # Setup model
    print("Setting up MegaFS model...")
    model = setup_model(config)
    print(f"✓ Model loaded on {config['device']}")
    
    # Determine which images to attack
    if args.batch:
        image_ids = [int(x.strip()) for x in args.batch.split(',')]
    else:
        image_ids = [args.image_id]
    
    # Limit samples if specified
    if config['experiment'].get('max_samples'):
        image_ids = image_ids[:config['experiment']['max_samples']]
    
    # Run attack
    if len(image_ids) == 1:
        run_single_attack(config, model, image_ids[0], output_dir)
    else:
        run_batch_attack(config, model, image_ids, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Attack completed! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

