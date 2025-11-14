"""
Adversarial attack utilities for MegaFS using HieRFE's dual-target strategy
Based on BlazeFace face detection for mask generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from .image_utils import ImageProcessor
from models.hierfe import HieRFE
from models.blazeface import get_blazeface_model, detect_faces
from .face_detectors import (
    FaceDetector, get_face_detector, validate_detection,
    BlazeFaceDetector, HaarCascadeDetector
)


class FaceDetectionError(Exception):
    """Exception raised when face detection fails or doesn't meet validation criteria"""
    
    def __init__(self, message: str, reason: str = "", metrics: Optional[Dict[str, Any]] = None):
        """
        Initialize FaceDetectionError.
        
        Args:
            message: Error message
            reason: Detailed reason for failure
            metrics: Validation metrics dictionary
        """
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.metrics = metrics or {}
    
    def __str__(self):
        msg = self.message
        if self.reason:
            msg += f"\nReason: {self.reason}"
        if self.metrics:
            msg += f"\nMetrics: {self.metrics}"
        return msg


def generate_mask_from_detector(
    image_np: np.ndarray,
    detector: FaceDetector,
    device: str = 'cuda',
    edge_blur_ks: int = 0,
    strict_detection: bool = True,
    min_bbox_area_ratio: float = 0.01,
    max_bbox_area_ratio: float = 0.95,
    min_bbox_size: int = 20,
    fallback_detector: Optional[FaceDetector] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate face mask M1 using face detector with validation.
    
    Args:
        image_np: Input image as numpy array [H, W, 3] in RGB format, range [0, 255]
        detector: FaceDetector instance
        device: Device for computation
        edge_blur_ks: Kernel size for edge blur (0 = no blur)
        strict_detection: If True, raise FaceDetectionError on failure; if False, use fallback ellipse
        min_bbox_area_ratio: Minimum face area as ratio of image
        max_bbox_area_ratio: Maximum face area as ratio of image
        min_bbox_size: Minimum bbox width/height in pixels
        fallback_detector: Optional fallback detector to try if primary fails
        
    Returns:
        M1 (face mask), M2 (background mask), both detached tensors [1, 3, H, W]
        
    Raises:
        FaceDetectionError: If detection fails and strict_detection=True
    """
    H, W = image_np.shape[:2]
    
    # Try primary detector
    bboxes = detector.detect(image_np)
    
    # If no detection, try fallback detector
    if len(bboxes) == 0 and fallback_detector is not None:
        print(f"[MASK INFO] Primary detector failed, trying fallback detector...")
        bboxes = fallback_detector.detect(image_np)
    
    # Select largest face if multiple detected
    if len(bboxes) > 0:
        # Sort by area (w * h) and take largest
        bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
        bbox = bboxes[0]  # (x, y, w, h)
        print(f"[MASK DEBUG] Detected face: bbox={bbox}")
    else:
        bbox = None
    
    # Validate detection
    is_valid, reason, metrics = validate_detection(
        bbox,
        (H, W),
        min_bbox_area_ratio=min_bbox_area_ratio,
        max_bbox_area_ratio=max_bbox_area_ratio,
        min_bbox_size=min_bbox_size
    )
    
    # Handle validation failure
    if not is_valid:
        error_msg = f"Face detection failed validation: {reason}"
        if strict_detection:
            raise FaceDetectionError(error_msg, reason=reason, metrics=metrics)
        else:
            print(f"[MASK WARN] {error_msg}, using center ellipse as fallback")
            bbox = None
    
    # Create ellipse mask from bounding box
    m1_np = ImageProcessor.make_ellipse_mask(image_np, bbox, edge_blur_ks=edge_blur_ks)
    
    # Convert to tensor [1, 3, H, W]
    m1_t = torch.from_numpy(m1_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    M1 = m1_t
    M2 = (1.0 - M1).clamp(0, 1)
    
    mask_face_mean = float(M1.sum().item()) / M1.numel()
    print(f"[MASK DEBUG] Mask face area: {mask_face_mean:.3f}")
    if metrics:
        print(f"[MASK DEBUG] Detection metrics: {metrics}")
    
    return M1.detach(), M2.detach()


def generate_mask_from_blazeface(
    image_np: np.ndarray,
    blazeface_model,
    device: str = 'cuda',
    edge_blur_ks: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate face mask M1 using BlazeFace face detection (backward compatibility).
    
    Args:
        image_np: Input image as numpy array [H, W, 3] in RGB format, range [0, 255]
        blazeface_model: BlazeFace model instance
        device: Device for computation
        edge_blur_ks: Kernel size for edge blur (0 = no blur)
        
    Returns:
        M1 (face mask), M2 (background mask), both detached tensors [1, 3, H, W]
    """
    # Create detector wrapper for backward compatibility
    detector = BlazeFaceDetector(model=blazeface_model, device=device)
    
    # Use non-strict mode for backward compatibility
    return generate_mask_from_detector(
        image_np,
        detector,
        device=device,
        edge_blur_ks=edge_blur_ks,
        strict_detection=False  # Backward compatible: use fallback ellipse
    )


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
    Dual-target PGD attack on HieRFE with BlazeFace mask generation.
    
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
        device: str = 'cuda',
        verbose: bool = True,
        sem_variant: str = 'mse_f4',  # 'mse_f4' (default), 'l1_f4', 'self_collapse', 'self_collapse_mid', 'contrastive_bg'
        preproc: str = 'none',
        mask_blur_ks: int = 0,
        loss_schedule: bool = False,
        clip_grad: float = 0.0,
        checkpoint_dir: str = "weights",
        detector_method: str = "blazeface_padded",
        strict_detection: bool = True,
        fallback_detector_method: Optional[str] = None,
        min_bbox_area_ratio: float = 0.01,
        max_bbox_area_ratio: float = 0.95,
        min_bbox_size: int = 20,
        detector_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.identity_extractor = identity_extractor
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device
        self.verbose = verbose
        self.sem_variant = sem_variant
        self.preproc_mode = preproc
        self.mask_blur_ks = int(mask_blur_ks)
        self.loss_schedule = bool(loss_schedule)
        self.clip_grad = float(clip_grad)
        self.strict_detection = strict_detection
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.min_bbox_size = min_bbox_size
        
        # Initialize face detector
        detector_kwargs = detector_kwargs or {}
        self.detector = get_face_detector(
            method=detector_method,
            device=device,
            checkpoint_dir=checkpoint_dir,
            **detector_kwargs
        )
        
        # Initialize fallback detector if specified
        self.fallback_detector = None
        if fallback_detector_method:
            fallback_kwargs = detector_kwargs.copy()
            self.fallback_detector = get_face_detector(
                method=fallback_detector_method,
                device=device,
                checkpoint_dir=checkpoint_dir,
                **fallback_kwargs
            )
        
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
        image_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks using configured face detector with validation.
        
        Raises:
            FaceDetectionError: If detection fails and strict_detection=True
        """
        M1, M2 = generate_mask_from_detector(
            image_np,
            self.detector,
            device=self.device,
            edge_blur_ks=self.mask_blur_ks,
            strict_detection=self.strict_detection,
            min_bbox_area_ratio=self.min_bbox_area_ratio,
            max_bbox_area_ratio=self.max_bbox_area_ratio,
            min_bbox_size=self.min_bbox_size,
            fallback_detector=self.fallback_detector
        )
        return M1, M2
    
    def attack(self, image_path: str, output_dir: Optional[str] = None, output_prefix: str = "target") -> np.ndarray:
        """
        Execute dual-target PGD attack.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            output_prefix: Prefix for output files
            
        Returns:
            Adversarial image as numpy array
            
        Raises:
            FaceDetectionError: If face detection fails validation and strict_detection=True
        """
        # 1. Load and preprocess image
        image_np = ImageProcessor.load_image(image_path, target_size=(256, 256))
        image_np = ImageProcessor.apply_preprocessing(image_np, mode=self.preproc_mode)
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        # 2. Generate masks using face detector - THIS WILL STOP ATTACK IF DETECTION FAILS
        # generate_masks() will raise FaceDetectionError if strict_detection=True and validation fails
        try:
            M1, M2 = self.generate_masks(image_np)
        except FaceDetectionError as e:
            if self.verbose:
                print(f"[ATTACK ERROR] Face detection failed. Attack stopped.")
                print(f"Error: {e}")
            raise  # Re-raise to stop attack
        
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

