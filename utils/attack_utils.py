"""
Adversarial attack utilities for MegaFS using HieRFE's dual-target strategy
Based on self-supervised mask generation from FPN feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from .image_utils import ImageProcessor
from models.hierfe import HieRFE


def generate_mask_from_fpn(
    identity_extractor: HieRFE,
    image_tensor_normalized: torch.Tensor,
    output_size: Tuple[int, int] = (256, 256),
    feature_layers: List[str] = ['f8', 'f16'],
    threshold: float = 0.3,
    mask_type: str = 'hard',
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate face mask M1 using HieRFE's FPN feature maps (XAI-inspired approach).
    
    Args:
        identity_extractor: HieRFE model
        image_tensor_normalized: Input tensor in range [-1, 1]
        output_size: Target mask size (height, width)
        feature_layers: Which FPN layers to use ('f8', 'f16', 'f32')
        threshold: Threshold for hard masking (0-1)
        mask_type: 'hard' or 'soft'
        device: Device for computation
        
    Returns:
        M1 (face mask), M2 (background mask), both detached
    """
    identity_extractor.eval()
    
    with torch.no_grad():
        # Extract FPN features
        f4, f8, f16, f32 = identity_extractor.fpn(image_tensor_normalized)
        
        # Compute attention maps for each layer
        attention_maps = []
        
        if 'f8' in feature_layers:
            # f8: [B, 512, 8, 8]
            attn_f8 = f8.pow(2).mean(dim=1, keepdim=True)
            attn_f8 = F.interpolate(attn_f8, size=output_size, mode='bilinear', align_corners=False)
            attention_maps.append(attn_f8)
        
        if 'f16' in feature_layers:
            # f16: [B, 512, 16, 16]
            attn_f16 = f16.pow(2).mean(dim=1, keepdim=True)
            attn_f16 = F.interpolate(attn_f16, size=output_size, mode='bilinear', align_corners=False)
            attention_maps.append(attn_f16)
        
        if 'f32' in feature_layers:
            # f32: [B, 512, 32, 32]
            attn_f32 = f32.pow(2).mean(dim=1, keepdim=True)
            attn_f32 = F.interpolate(attn_f32, size=output_size, mode='bilinear', align_corners=False)
            attention_maps.append(attn_f32)
        
        # Average across selected layers
        if len(attention_maps) > 1:
            attention_map = torch.stack(attention_maps).mean(dim=0)
        else:
            attention_map = attention_maps[0]
        
        # Min-max normalization
        min_val = attention_map.min()
        max_val = attention_map.max()
        attention_map = (attention_map - min_val) / (max_val - min_val + 1e-6)
        
        # Apply threshold to create mask
        if mask_type == 'hard':
            M1_soft = (attention_map > threshold).float()
        else:
            M1_soft = attention_map
        
        # Expand to 3 channels if needed [B, 1, H, W] -> [B, 3, H, W]
        if M1_soft.shape[1] == 1:
            M1_soft = M1_soft.repeat(1, 3, 1, 1)
    
    # Detach to prevent gradient flow during PGD
    M1 = M1_soft.detach()
    M2 = (1.0 - M1).detach()
    
    return M1, M2


def visualize_mask(mask: torch.Tensor, save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize mask as RGB image.
    
    Args:
        mask: Mask tensor [B, 3, H, W] or [B, 1, H, W]
        save_path: Optional path to save visualization
        
    Returns:
        RGB numpy array
    """
    # Handle single channel or multi-channel
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    
    # Convert to numpy
    if len(mask.shape) == 4:
        mask_np = mask[0].permute(1, 2, 0).cpu().numpy()
    else:
        mask_np = mask.permute(1, 2, 0).cpu().numpy()
    
    # Convert to uint8
    mask_np = (mask_np * 255).astype(np.uint8)
    
    if save_path:
        import cv2
        cv2.imwrite(save_path, mask_np[:, :, ::-1])  # RGB to BGR for OpenCV
    
    return mask_np


class DualTargetPGDAttack:
    """
    Dual-target PGD attack on HieRFE with self-supervised mask generation.
    
    Implements:
    - L_ID: Identity destruction on face region (A1)
    - L_SEM: Semantic collapse via background (A2)
    """
    
    def __init__(
        self,
        identity_extractor: HieRFE,
        epsilon: float = 8.0,
        alpha: float = 1.0,
        num_iter: int = 100,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        feature_layers: List[str] = ['f8', 'f16'],
        mask_threshold: float = 0.3,
        mask_type: str = 'hard',
        device: str = 'cuda',
        verbose: bool = True
    ):
        self.identity_extractor = identity_extractor
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.feature_layers = feature_layers
        self.mask_threshold = mask_threshold
        self.mask_type = mask_type
        self.device = device
        self.verbose = verbose
        
        # History for logging
        self.loss_history = {'total': [], 'L_ID': [], 'L_SEM': []}
    
    def generate_masks(
        self,
        image_tensor_normalized: torch.Tensor,
        output_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate masks using FPN features."""
        return generate_mask_from_fpn(
            self.identity_extractor,
            image_tensor_normalized,
            output_size=output_size,
            feature_layers=self.feature_layers,
            threshold=self.mask_threshold,
            mask_type=self.mask_type,
            device=self.device
        )
    
    def attack(self, image_path: str, output_dir: Optional[str] = None) -> np.ndarray:
        """
        Execute dual-target PGD attack.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            Adversarial image as numpy array
        """
        # 1. Load and preprocess image
        image_np = ImageProcessor.load_image(image_path, target_size=(256, 256))
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        image_tensor_normalized = ImageProcessor.preprocess_for_model_tensor(image_tensor)
        
        # 2. Generate masks using FPN
        H, W = image_tensor.shape[2:]
        M1, M2 = self.generate_masks(image_tensor_normalized, output_size=(H, W))
        
        if self.verbose:
            print(f"Generated masks - M1: {M1.sum()/M1.numel():.2%} face, M2: {M2.sum()/M2.numel():.2%} background")
        
        # 3. Extract target features (clean face)
        with torch.no_grad():
            preprocessed_face = ImageProcessor.preprocess_for_model_tensor(image_tensor * M1)
            target_latents, target_f4_from_face = self.identity_extractor(preprocessed_face)
        
        # 4. Initialize perturbation
        delta = torch.zeros_like(image_tensor, requires_grad=True).to(self.device)
        
        # 5. PGD loop
        for i in range(self.num_iter):
            # Adversarial image
            adv_image = image_tensor + delta
            adv_image_clipped = torch.clamp(adv_image, 0, 255)
            
            # Preprocess with masks
            adv_face_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_clipped * M1)
            adv_bg_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_clipped * M2)
            
            # Forward through HieRFE
            adv_latents, adv_f4_from_face = self.identity_extractor(adv_face_preprocessed)
            _, adv_f4_from_bg = self.identity_extractor(adv_bg_preprocessed)
            
            # Loss computation
            # L_ID: Maximize negative cosine similarity (destroy identity)
            L_ID_loss = -F.cosine_similarity(adv_latents, target_latents).mean()
            
            # L_SEM: Minimize MSE to inject face structure into background
            L_SEM_loss = F.mse_loss(adv_f4_from_bg, target_f4_from_face)
            
            # Combined loss
            total_loss = (self.lambda_1 * L_ID_loss) + (self.lambda_2 * L_SEM_loss)
            
            # Backward
            total_loss.backward()
            
            # Gradient step (PGD)
            grad = delta.grad.detach()
            delta.data = delta.data - self.alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.grad.zero_()
            
            # Logging
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['L_ID'].append(L_ID_loss.item())
            self.loss_history['L_SEM'].append(L_SEM_loss.item())
            
            if self.verbose and (i % 20 == 0 or i == self.num_iter - 1):
                print(f"Iter {i:3d}: Total={total_loss.item():.4f}, "
                      f"L_ID={L_ID_loss.item():.4f}, L_SEM={L_SEM_loss.item():.4f}")
        
        # 6. Generate final adversarial image
        final_adv = torch.clamp(image_tensor + delta.detach(), 0, 255)
        final_adv_np = final_adv.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 7. Save results if requested
        if output_dir:
            self._save_results(image_np, final_adv_np, M1, M2, output_dir)
        
        return final_adv_np
    
    def _save_results(
        self,
        original: np.ndarray,
        adversarial: np.ndarray,
        M1: torch.Tensor,
        M2: torch.Tensor,
        output_dir: str
    ):
        """Save attack results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import cv2
        
        # Save images
        cv2.imwrite(str(output_path / 'original.jpg'), original[:, :, ::-1])
        cv2.imwrite(str(output_path / 'adversarial.jpg'), adversarial[:, :, ::-1])
        
        # Save masks
        M1_np = visualize_mask(M1)
        M2_np = visualize_mask(M2)
        cv2.imwrite(str(output_path / 'mask_face.jpg'), M1_np[:, :, ::-1])
        cv2.imwrite(str(output_path / 'mask_bg.jpg'), M2_np[:, :, ::-1])
        
        # Save perturbation
        perturbation = (adversarial.astype(np.float32) - original.astype(np.float32)).astype(np.int16)
        perturbation_vis = (perturbation + 127).astype(np.uint8)
        cv2.imwrite(str(output_path / 'perturbation.jpg'), perturbation_vis[:, :, ::-1])
        
        # Save comparison grid
        comparison = np.hstack([original, adversarial, perturbation_vis])
        cv2.imwrite(str(output_path / 'comparison.jpg'), comparison[:, :, ::-1])
        
        # Save loss curves
        self._plot_loss_curves(str(output_path / 'loss_curves.jpg'))
    
    def _plot_loss_curves(self, save_path: str):
        """Plot and save loss curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].plot(self.loss_history['total'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            
            axes[1].plot(self.loss_history['L_ID'])
            axes[1].set_title('L_ID (Identity Destruction)')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Loss')
            
            axes[2].plot(self.loss_history['L_SEM'])
            axes[2].set_title('L_SEM (Semantic Collapse)')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()
        except ImportError:
            print("Warning: matplotlib not available, skipping loss curve plot")


def compute_metrics(original: np.ndarray, adversarial: np.ndarray) -> Dict[str, float]:
    """
    Compute attack effectiveness metrics.
    
    Args:
        original: Original image [H, W, 3]
        adversarial: Adversarial image [H, W, 3]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # L2 perturbation
    l2_norm = np.linalg.norm(adversarial.astype(np.float32) - original.astype(np.float32))
    metrics['L2_norm'] = l2_norm
    
    # L-inf perturbation
    linf_norm = np.abs(adversarial.astype(np.float32) - original.astype(np.float32)).max()
    metrics['Linf_norm'] = linf_norm
    
    # Perceptual metrics (if available)
    try:
        import lpips
        from skimage.metrics import structural_similarity as ssim
        
        # SSIM
        ssim_val = ssim(original, adversarial, multichannel=True, channel_axis=2)
        metrics['SSIM'] = ssim_val
        
        # LPIPS (requires model initialization)
        # Note: This is commented out to avoid requiring pretrained model on import
        # lpips_model = lpips.LPIPS(net='alex')
        # original_tensor = lpips.im2tensor(original)
        # adv_tensor = lpips.im2tensor(adversarial)
        # lpips_val = lpips_model.forward(original_tensor, adv_tensor).item()
        # metrics['LPIPS'] = lpips_val
        
    except ImportError:
        pass
    
    return metrics

