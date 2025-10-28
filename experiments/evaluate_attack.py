"""
Evaluation and visualization for adversarial attacks
Compare clean vs adversarial face swapping results
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from models.megafs import MegaFS
from utils.attack_utils import DualTargetPGDAttack, compute_metrics
from utils.image_utils import ImageProcessor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_face_swap(model: MegaFS, image_tensor: torch.Tensor) -> np.ndarray:
    """Run face swap on a tensor."""
    # For evaluation, we need source and target
    # Using same image as both (self-swap for simplicity)
    with torch.no_grad():
        swapped = model.forward(image_tensor, image_tensor)
    
    # Convert to numpy
    swapped_np = swapped[0].permute(1, 2, 0).cpu().numpy()
    
    # Denormalize if needed
    if swapped_np.min() < 0:
        swapped_np = (swapped_np + 1) / 2 * 255
    else:
        swapped_np = swapped_np * 255
    
    swapped_np = np.clip(swapped_np, 0, 255).astype(np.uint8)
    return swapped_np


def create_evaluation_grid(
    original: np.ndarray,
    adversarial: np.ndarray,
    clean_swap: np.ndarray,
    adv_swap: np.ndarray,
    metrics: Dict[str, float]
) -> np.ndarray:
    """Create a grid showing original, adversarial, and swaps."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Adversarial
    axes[0, 1].imshow(adversarial)
    axes[0, 1].set_title(f"Adversarial (L-inf: {metrics.get('Linf_norm', 0):.1f})")
    axes[0, 1].axis('off')
    
    # Clean swap
    axes[1, 0].imshow(clean_swap)
    axes[1, 0].set_title('Clean Face Swap')
    axes[1, 0].axis('off')
    
    # Adversarial swap
    axes[1, 1].imshow(adv_swap)
    axes[1, 1].set_title('Adversarial Face Swap')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return buf


def plot_comparison(
    clean_metrics: Dict[str, float],
    adv_metrics: Dict[str, float],
    save_path: str
):
    """Plot comparison between clean and adversarial swaps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # SSIM comparison
    metrics_to_plot = ['SSIM', 'L2_norm', 'Linf_norm']
    
    clean_vals = [clean_metrics.get(m, 0) for m in metrics_to_plot]
    adv_vals = [adv_metrics.get(m, 0) for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    axes[0].bar(x - width/2, clean_vals, width, label='Clean', alpha=0.7)
    axes[0].bar(x + width/2, adv_vals, width, label='Adversarial', alpha=0.7)
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Comparison of Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_to_plot)
    axes[0].legend()
    
    # Metrics table
    axes[1].axis('off')
    table_data = [
        ['Metric', 'Clean', 'Adversarial'],
        ['SSIM', f"{clean_metrics.get('SSIM', 0):.4f}", f"{adv_metrics.get('SSIM', 0):.4f}"],
        ['L2 norm', f"{clean_metrics.get('L2_norm', 0):.2f}", f"{adv_metrics.get('L2_norm', 0):.2f}"],
        ['L-inf norm', f"{clean_metrics.get('Linf_norm', 0):.2f}", f"{adv_metrics.get('Linf_norm', 0):.2f}"],
    ]
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_image_pair(
    config: Dict[str, Any],
    model: MegaFS,
    image_id: int,
    output_dir: str
):
    """Evaluate clean vs adversarial swap on a single image."""
    print(f"\nEvaluating image ID: {image_id}")
    
    from utils.attack_utils import DualTargetPGDAttack
    
    # Get image path
    img_root = config['paths']['img_root']
    image_path = get_image_path(image_id, img_root)
    
    # Load images
    original = ImageProcessor.load_image(image_path, target_size=(256, 256))
    
    # Generate adversarial image
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
        verbose=False
    )
    
    adversarial = attack.attack(image_path, output_dir=None)
    
    # Convert to tensors for face swap
    original_tensor = ImageProcessor.preprocess_for_model(original).unsqueeze(0).to(config['device'])
    adv_tensor = ImageProcessor.preprocess_for_model(adversarial).unsqueeze(0).to(config['device'])
    
    # Run face swaps
    print("Running face swaps...")
    clean_swap = evaluate_face_swap(model, original_tensor)
    adv_swap = evaluate_face_swap(model, adv_tensor)
    
    # Compute metrics
    clean_metrics = compute_metrics(original, original)
    adv_metrics = compute_metrics(original, adversarial)
    swap_metrics = compute_metrics(clean_swap, adv_swap)
    
    print(f"Clean SSIM: {clean_metrics.get('SSIM', 'N/A')}")
    print(f"Adversarial SSIM: {adv_metrics.get('SSIM', 'N/A')}")
    print(f"Swap difference: {swap_metrics.get('Linf_norm', 'N/A'):.2f}")
    
    # Create visualizations
    output_path = Path(output_dir) / f"eval_{image_id:05d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comparison grid
    grid = create_evaluation_grid(original, adversarial, clean_swap, adv_swap, adv_metrics)
    plt.imsave(str(output_path / 'comparison_grid.png'), grid / 255.)
    
    # Plot comparison
    plot_comparison(clean_metrics, adv_metrics, str(output_path / 'metrics_comparison.png'))
    
    # Save individual images
    import cv2
    cv2.imwrite(str(output_path / 'clean_swap.jpg'), clean_swap[:, :, ::-1])
    cv2.imwrite(str(output_path / 'adv_swap.jpg'), adv_swap[:, :, ::-1])
    
    # Save metrics
    import json
    all_metrics = {
        'clean': clean_metrics,
        'adversarial': adv_metrics,
        'swap_difference': swap_metrics
    }
    with open(str(output_path / 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Evaluation saved to: {output_path}")


def get_image_path(image_id: int, img_root: str) -> str:
    """Get full path to image."""
    image_file = f"{image_id:05d}.jpg"
    return os.path.join(img_root, image_file)


def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial attack effectiveness')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml',
                        help='Configuration file')
    parser.add_argument('--image-id', type=int, required=True,
                        help='Image ID to evaluate')
    parser.add_argument('--output-dir', type=str, default='experiments/evaluation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup model
    cfg = Config(
        swap_type=config['model']['swap_type'],
        dataset_root=config['paths']['dataset_root'],
        img_root=config['paths']['img_root'],
        mask_root=config['paths']['mask_root'],
        checkpoint_dir=config['paths']['checkpoint_dir']
    )
    
    model = MegaFS(
        config=cfg,
        debug=True,
        enable_grads=False,  # No gradients needed for evaluation
        device=config['device']
    )
    
    # Evaluate
    evaluate_image_pair(config, model, args.image_id, args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

