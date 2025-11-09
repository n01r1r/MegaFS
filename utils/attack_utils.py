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
    feature_layers: List[str] = ['f4', 'f8', 'f16'],
    threshold: Any = 0.2,  # LOWER default for better mask coverage
    mask_type: str = 'hard',
    device: str = 'cuda',
    temperature: float = 0.15,
    blur_ks: int = 0,
    blur_sigma: float = 0.0
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
        
        if 'f4' in feature_layers:
            # f4: higher spatial resolution, strong on contours
            attn_f4 = f4.pow(2).mean(dim=1, keepdim=True)
            attn_f4 = F.interpolate(attn_f4, size=output_size, mode='bilinear', align_corners=False)
            # normalize per-map
            attn_f4 = (attn_f4 - attn_f4.min()) / (attn_f4.max() - attn_f4.min() + 1e-6)
            attention_maps.append(attn_f4)

        if 'f8' in feature_layers:
            # f8: [B, 512, 8, 8]
            attn_f8 = f8.pow(2).mean(dim=1, keepdim=True)
            attn_f8 = F.interpolate(attn_f8, size=output_size, mode='bilinear', align_corners=False)
            attn_f8 = (attn_f8 - attn_f8.min()) / (attn_f8.max() - attn_f8.min() + 1e-6)
            attention_maps.append(attn_f8)
        
        if 'f16' in feature_layers:
            # f16: [B, 512, 16, 16]
            attn_f16 = f16.pow(2).mean(dim=1, keepdim=True)
            attn_f16 = F.interpolate(attn_f16, size=output_size, mode='bilinear', align_corners=False)
            attn_f16 = (attn_f16 - attn_f16.min()) / (attn_f16.max() - attn_f16.min() + 1e-6)
            attention_maps.append(attn_f16)
        
        if 'f32' in feature_layers:
            # f32: [B, 512, 32, 32]
            attn_f32 = f32.pow(2).mean(dim=1, keepdim=True)
            attn_f32 = F.interpolate(attn_f32, size=output_size, mode='bilinear', align_corners=False)
            attn_f32 = (attn_f32 - attn_f32.min()) / (attn_f32.max() - attn_f32.min() + 1e-6)
            attention_maps.append(attn_f32)
        
        # Multi-scale fusion via element-wise product (AND gate)
        if len(attention_maps) > 1:
            attention_map = attention_maps[0]
            for k in range(1, len(attention_maps)):
                attention_map = (attention_map * attention_maps[k]).clamp(0, 1)
        else:
            attention_map = attention_maps[0]
        
        # Min-max normalization
        min_val = attention_map.min()
        max_val = attention_map.max()
        attention_map = (attention_map - min_val) / (max_val - min_val + 1e-6)
        
        # Adaptive threshold (if requested)
        thr_val: float
        if isinstance(threshold, str) and threshold == 'auto':
            # Otsu on CPU
            attn_np = attention_map.detach().cpu().numpy().astype(np.float32)
            attn_u8 = (attn_np * 255.0).astype(np.uint8)
            import cv2
            _, thr = cv2.threshold(attn_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr_val = float(thr) / 255.0
        else:
            thr_val = float(threshold)

        # Soft mask via temperature-controlled sigmoid around threshold
        M1_soft = torch.sigmoid((attention_map - thr_val) / max(1e-6, temperature))
        
        # Expand to 3 channels if needed [B, 1, H, W] -> [B, 3, H, W]
        if M1_soft.shape[1] == 1:
            M1_soft = M1_soft.repeat(1, 3, 1, 1)

    # Optional blur (approximate with average pooling)
    if blur_ks and blur_ks > 1:
        k = int(blur_ks)
        pad = k // 2
        M1_soft = F.avg_pool2d(M1_soft, kernel_size=k, stride=1, padding=pad)

    # Confidence weighting: multiply by attention magnitude
    with torch.no_grad():
        if attention_map.shape[1] == 1:
            attn3 = attention_map.repeat(1, 3, 1, 1)
        else:
            attn3 = attention_map
    M1_soft = (M1_soft * attn3).clamp(0, 1)

    # Derive background and renormalize so M1+M2≈1
    M1 = M1_soft.detach()
    M2 = (1.0 - M1).detach()
    S = (M1 + M2).clamp(min=1e-6)
    M1 = (M1 / S).detach()
    M2 = (M2 / S).detach()
    
    mask_face_mean = float(M1.sum().item())/M1.numel()
    if mask_face_mean < 0.15:
        print(f"[MASK WARN] FPN mask face area very small: {mask_face_mean:.3f}. Try threshold=0.15 or change fusion.")
    else:
        print(f"[MASK DEBUG] FPN mask face area: {mask_face_mean:.3f}")
    
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
        feature_layers: List[str] = ['f4', 'f8', 'f16'],
        mask_threshold: Any = 0.3,
        mask_type: str = 'hard',
        device: str = 'cuda',
        verbose: bool = True,
        sem_variant: str = 'mse_f4',  # 'mse_f4' (default), 'l1_f4', 'self_collapse', 'self_collapse_mid', 'contrastive_bg'
        preproc: str = 'none',
        mask_mode: str = 'fpn',  # 'fpn' | 'ellipse' | 'anno' (future)
        mask_blur_ks: int = 0,
        mask_blur_sigma: float = 0.0,
        mask_temperature: float = 0.15,
        loss_schedule: bool = False,
        clip_grad: float = 0.0
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
        self.sem_variant = sem_variant
        self.preproc_mode = preproc
        self.mask_mode = mask_mode
        self.mask_blur_ks = int(mask_blur_ks)
        self.mask_blur_sigma = float(mask_blur_sigma)
        self.mask_temperature = float(mask_temperature)
        self.loss_schedule = bool(loss_schedule)
        self.clip_grad = float(clip_grad)
        
        # History for logging
        self.loss_history = {
            'total': [],
            'L_ID': [],
            'L_SEM': [],
            'cos_sim': [],
            'eps_sat_pct': [],
            'grad_norm': []
        }
    
    def generate_masks(
        self,
        image_tensor_normalized: torch.Tensor,
        output_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate masks using FPN features with soft/blur options."""
        M1, M2 = generate_mask_from_fpn(
            self.identity_extractor,
            image_tensor_normalized,
            output_size=output_size,
            feature_layers=self.feature_layers,
            threshold=self.mask_threshold,
            mask_type=self.mask_type,
            device=self.device,
            temperature=self.mask_temperature,
            blur_ks=self.mask_blur_ks,
            blur_sigma=self.mask_blur_sigma
        )
        return M1, M2
    
    def attack(self, image_path: str, output_dir: Optional[str] = None, output_prefix: str = "target") -> np.ndarray:
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
        image_np = ImageProcessor.apply_preprocessing(image_np, mode=self.preproc_mode)
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        image_tensor_normalized = ImageProcessor.preprocess_for_model_tensor(image_tensor)
        
        # 2. Generate masks
        H, W = image_tensor.shape[2:]
        if self.mask_mode == 'ellipse':
            # Build ellipse mask on preprocessed RGB
            bbox = ImageProcessor.detect_face_bbox(image_np)
            m1_np = ImageProcessor.make_ellipse_mask(image_np, bbox, edge_blur_ks=self.mask_blur_ks)
            # Convert to tensor shape [1,3,H,W]
            m1_t = torch.from_numpy(m1_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            M1 = m1_t
            M2 = (1.0 - M1).clamp(0, 1)
        else:
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
            
            # Optional loss schedule (simple ramp for lambda weights)
            if self.loss_schedule:
                t = (i + 1) / float(max(1, self.num_iter))
                # emphasize semantic later (example schedule)
                lambda_1 = self.lambda_1 * (1.0 - 0.3 * t)
                lambda_2 = self.lambda_2 * (0.7 + 0.3 * t)
            else:
                lambda_1 = self.lambda_1
                lambda_2 = self.lambda_2

            # Forward through HieRFE
            adv_latents, adv_f4_from_face = self.identity_extractor(adv_face_preprocessed)
            _, adv_f4_from_bg = self.identity_extractor(adv_bg_preprocessed)
            
            # Loss computation
            # L_ID: Maximize negative cosine similarity (destroy identity)
            cos_sim = F.cosine_similarity(adv_latents, target_latents).mean()
            L_ID_loss = -cos_sim
            
            # L_SEM variants
            if self.sem_variant == 'l1_f4':
                L_SEM_loss = F.l1_loss(adv_f4_from_bg, target_f4_from_face)
            elif self.sem_variant == 'self_collapse':
                # Suppress background features rather than matching face f4
                L_SEM_loss = -torch.mean(adv_f4_from_bg.pow(2))
            elif self.sem_variant == 'self_collapse_mid':
                # Use mid-level proxy by average of f8/f16 via FPN forward on preprocessed bg
                with torch.no_grad():
                    # compute FPN features from normalized bg (reuse encoder.fpn)
                    f4_mid, f8_mid, f16_mid, f32_mid = self.identity_extractor.fpn(adv_bg_preprocessed)
                L_SEM_loss = -torch.mean(f8_mid.pow(2)) - 0.5 * torch.mean(f16_mid.pow(2))
            elif self.sem_variant == 'contrastive_bg':
                # Push bg features away from face features using margin on cosine
                margin = 0.2
                cos_bg_face = F.cosine_similarity(adv_f4_from_bg.flatten(1), target_f4_from_face.flatten(1)).mean()
                L_SEM_loss = F.relu(margin - (1.0 - cos_bg_face))
            else:
                # Default: MSE between background features and face features
                L_SEM_loss = F.mse_loss(adv_f4_from_bg, target_f4_from_face)
            
            # Combined loss
            total_loss = (lambda_1 * L_ID_loss) + (lambda_2 * L_SEM_loss)
            
            # Backward
            total_loss.backward()
            
            # Gradient step (PGD)
            grad = delta.grad.detach()
            if self.clip_grad and self.clip_grad > 0:
                grad = torch.clamp(grad, -self.clip_grad, self.clip_grad)
            delta.data = delta.data - self.alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.grad.zero_()
            
            # Logging
            with torch.no_grad():
                eps_sat = (delta.abs() >= (self.epsilon - 1e-6)).float().mean().item() * 100.0
                grad_norm = grad.abs().mean().item()
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['L_ID'].append(L_ID_loss.item())
            self.loss_history['L_SEM'].append(L_SEM_loss.item())
            self.loss_history['cos_sim'].append(float(cos_sim.item()))
            self.loss_history['eps_sat_pct'].append(eps_sat)
            self.loss_history['grad_norm'].append(grad_norm)
            
            if self.verbose and (i % 20 == 0 or i == self.num_iter - 1):
                print(
                    f"Iter {i:5d}: Total={total_loss.item():.4f}, "
                    f"L_ID={L_ID_loss.item():.4f} (cos={float(cos_sim.item()):.4f}), "
                    f"L_SEM={L_SEM_loss.item():.4f}, eps_sat={eps_sat:.1f}%"
                )
        
        # 6. Generate final adversarial image
        final_adv = torch.clamp(image_tensor + delta.detach(), 0, 255)
        final_adv_np = final_adv.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # 7. Save results if requested
        if output_dir:
            self._save_results(image_np, final_adv_np, M1, M2, output_dir, output_prefix=output_prefix)
        
        return final_adv_np
    
    def _save_results(
        self,
        original: np.ndarray,
        adversarial: np.ndarray,
        M1: torch.Tensor,
        M2: torch.Tensor,
        output_dir: str,
        output_prefix: str = "target"
    ):
        """Save attack results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import cv2
        
        # Save images
        cv2.imwrite(str(output_path / f'original_{output_prefix}.jpg'), original[:, :, ::-1])
        cv2.imwrite(str(output_path / f'adversarial_{output_prefix}.jpg'), adversarial[:, :, ::-1])
        
        # Save masks
        M1_np = visualize_mask(M1)
        M2_np = visualize_mask(M2)
        cv2.imwrite(str(output_path / f'mask_face_{output_prefix}.jpg'), M1_np[:, :, ::-1])
        cv2.imwrite(str(output_path / f'mask_bg_{output_prefix}.jpg'), M2_np[:, :, ::-1])
        
        # Save perturbation
        perturbation = (adversarial.astype(np.float32) - original.astype(np.float32)).astype(np.int16)
        perturbation_vis = (perturbation + 127).astype(np.uint8)
        cv2.imwrite(str(output_path / f'perturbation_{output_prefix}.jpg'), perturbation_vis[:, :, ::-1])
        
        # Save comparison grid
        comparison = np.hstack([original, adversarial, perturbation_vis])
        cv2.imwrite(str(output_path / f'comparison_{output_prefix}.jpg'), comparison[:, :, ::-1])
        
        # Save loss curves
        self._plot_loss_curves(str(output_path / f'loss_curves_{output_prefix}.jpg'))

        # Save mask stats for later aggregation
        try:
            m1_mean = float(M1.mean().item())
            m2_mean = float(M2.mean().item())
            stats = {
                'mask_face_mean': m1_mean,
                'mask_bg_mean': m2_mean,
                'sem_variant': self.sem_variant
            }
            with open(str(output_path / f'mask_stats_{output_prefix}.json'), 'w') as f:
                import json
                json.dump(stats, f, indent=2)
        except Exception:
            pass
    
    def _plot_loss_curves(self, save_path: str):
        """Plot and save loss curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            
            axes[0,0].plot(self.loss_history['total'])
            axes[0,0].set_title('Total Loss')
            axes[0,0].set_xlabel('Iteration')
            axes[0,0].set_ylabel('Loss')
            
            axes[0,1].plot(self.loss_history['L_ID'])
            axes[0,1].set_title('L_ID (Identity Destruction)')
            axes[0,1].set_xlabel('Iteration')
            axes[0,1].set_ylabel('Loss')
            
            axes[0,2].plot(self.loss_history['L_SEM'])
            axes[0,2].set_title('L_SEM (Semantic Collapse)')
            axes[0,2].set_xlabel('Iteration')
            axes[0,2].set_ylabel('Loss')

            axes[1,0].plot(self.loss_history.get('cos_sim', []))
            axes[1,0].set_title('Cosine Similarity (face latents)')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('cos')

            axes[1,1].plot(self.loss_history.get('eps_sat_pct', []))
            axes[1,1].set_title('Epsilon Saturation (%)')
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('% pixels at |delta|=epsilon')

            axes[1,2].plot(self.loss_history.get('grad_norm', []))
            axes[1,2].set_title('Grad Norm (mean |grad|)')
            axes[1,2].set_xlabel('Iteration')
            axes[1,2].set_ylabel('value')
            
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

