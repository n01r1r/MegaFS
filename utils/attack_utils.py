"""
Adversarial attack utilities for MegaFS using HieRFE's dual-target strategy
Uses Haar Cascade face detection for mask generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    _SSIM_AVAILABLE = True
except ImportError:
    skimage_ssim = None
    _SSIM_AVAILABLE = False

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    _LPIPS_AVAILABLE = False

_LPIPS_MODEL = None

from .image_utils import ImageProcessor
from models.hierfe import HieRFE
from .face_detectors import (
    FaceDetector, get_face_detector, validate_detection,
    HaarCascadeDetector
)

try:
    from .metrics import PerceptualLoss
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    PerceptualLoss = None


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Compute Total Variation loss to encourage smooth perturbations.
    
    Args:
        img: Image tensor [B, C, H, W]
        
    Returns:
        TV loss scalar
    """
    b, c, h, w = img.shape
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (b * c * h * w)


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
    Dual-target PGD attack on HieRFE with Haar Cascade face detection for mask generation.
    
    Implements:
    - L_ID: Minimize cosine similarity to destroy identity in face region (A1)
    - L_SIM: Similarity preservation loss (LPIPS/MSE/L1) to maintain visual similarity
    - L_TV: Total Variation loss to encourage smooth perturbations
    
    Note: L_SEM has been removed to improve visual quality while maintaining attack effectiveness.
    """
    
    def __init__(
        self,
        identity_extractor: HieRFE,
        epsilon: float = 8.0,
        alpha: float = 1.0,
        num_iter: int = 100,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        lambda_sim: float = 0.0,  # Similarity preservation weight (0 = disabled)
        lambda_tv: float = 0.01,  # Total Variation weight
        device: str = 'cuda',
        verbose: bool = True,
        sem_variant: str = 'self_collapse',  # 'mse_f4', 'l1_f4', 'self_collapse' (default), 'self_collapse_mid', 'contrastive_bg'
        preproc: str = 'none',
        mask_blur_ks: int = 0,
        loss_schedule: bool = False,
        clip_grad: float = 0.0,
        checkpoint_dir: str = "weights",
        detector_method: str = "haar",
        strict_detection: bool = True,
        fallback_detector_method: Optional[str] = None,
        min_bbox_area_ratio: float = 0.01,
        max_bbox_area_ratio: float = 0.95,
        min_bbox_size: int = 20,
        detector_kwargs: Optional[Dict[str, Any]] = None,
        sim_loss_type: str = 'mse',  # 'mse', 'l1', 'perceptual', 'lpips'
        structure_weakening_factor: float = 0.7,  # Structure weakening factor for 'structure_weakening' variant (0.0-1.0)
        early_stop_threshold: Optional[float] = 0.2,  # Early stopping threshold (L_ID < threshold). Set to None to disable.
        convergence_window: int = 1000,  # Number of recent iterations to check for convergence
        convergence_tolerance: float = 1e-6,  # Loss change tolerance for convergence detection
        min_iter_for_convergence: int = 1000,  # Minimum iterations before checking convergence
        maximize_similarity: bool = False,  # If True, maximize similarity (L_ID = 1 - cos_sim) instead of minimizing
        random_init: bool = False,  # If True, initialize delta with random noise instead of zeros
        target_type: str = 'image'  # Target type: 'image', 'noise', 'zero', 'uniform', 'entropy'
    ):
        self.identity_extractor = identity_extractor
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_sim = float(lambda_sim)
        self.lambda_tv = float(lambda_tv)
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
        self.sim_loss_type = sim_loss_type
        self.structure_weakening_factor = float(structure_weakening_factor)
        self.early_stop_threshold = early_stop_threshold
        self.convergence_window = int(convergence_window)
        self.convergence_tolerance = float(convergence_tolerance)
        self.min_iter_for_convergence = int(min_iter_for_convergence)
        self.maximize_similarity = bool(maximize_similarity)
        self.random_init = bool(random_init)
        self.target_type = str(target_type)  # 'image', 'noise', 'zero', 'uniform', 'entropy'
        
        # Initialize LPIPS model if needed
        self.lpips_model = None
        if self.sim_loss_type == 'lpips' and LPIPS_AVAILABLE and PerceptualLoss is not None:
            use_gpu = (device == 'cuda' and torch.cuda.is_available())
            self.lpips_model = PerceptualLoss(net='alex', use_gpu=use_gpu)
            if use_gpu:
                self.lpips_model = self.lpips_model.to(device)
        
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
            'L_sim': [],
            'L_TV': [],
            'cos_sim': [],
            'eps_sat_pct': [],
            'grad_norm': [],
            'early_stopped': False
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
    
    def attack(self, image_path: str, output_dir: Optional[str] = None, output_prefix: str = "target", target_image_path: Optional[str] = None) -> np.ndarray:
        """
        Execute dual-target PGD attack.
        
        Args:
            image_path: Path to input image to attack
            output_dir: Directory to save results
            output_prefix: Prefix for output files
            target_image_path: Optional path to target identity image. If None, uses image_path's identity (self-attack).
                              For dual-target attack: source attack should use target image, target attack should use source image.
            
        Returns:
            Adversarial image as numpy array
            
        Raises:
            FaceDetectionError: If face detection fails validation and strict_detection=True
        """
        # Reset history per attack
        self.loss_history = {
            'total': [],
            'L_ID': [],
            'L_sim': [],
            'L_TV': [],
            'cos_sim': [],
            'eps_sat_pct': [],
            'grad_norm': [],
            'early_stopped': False
        }
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
        
        # 2.5. Extract full image f4 features for structure_weakening (from original image, not target)
        # This is needed for semantic loss regardless of target_type
        if self.sem_variant == 'structure_weakening':
            with torch.no_grad():
                original_full_preprocessed = ImageProcessor.preprocess_for_model_tensor(image_tensor)
                _, self._target_f4_full = self.identity_extractor(original_full_preprocessed)
        
        # 3. Extract target features based on target_type
        if self.target_type == 'entropy':
            # No target needed for entropy maximization
            target_latents = None
            target_f4_from_face = None
            if self.verbose:
                print(f"Target type: entropy (no target extraction needed)")
        elif self.target_type in ['zero', 'zero-l1', 'zero-huber']:
            # Zero target: will be created after we know latent shape
            target_latents = None
            target_f4_from_face = None
            if self.verbose:
                loss_type = self.target_type.split('-')[1] if '-' in self.target_type else 'mse'
                print(f"Target type: {self.target_type} (will create zero vector after getting latent shape, loss={loss_type})")
        elif self.target_type == 'uniform':
            # Uniform target: will be created after we know latent shape
            target_latents = None
            target_f4_from_face = None
            if self.verbose:
                print(f"Target type: uniform (will create uniform vector after getting latent shape)")
        elif self.target_type == 'noise':
            # Noise target: generate noise image and extract latents
            # Use full noise image (no mask) - noise has no identity information regardless of region
            if self.verbose:
                print(f"Target type: noise (generating Gaussian noise image, using full image without mask)")
            # Generate Gaussian noise in [0, 255] range
            noise_tensor = torch.randn_like(image_tensor) * 127.5 + 127.5  # Mean 127.5, std 127.5
            noise_tensor = torch.clamp(noise_tensor, 0, 255)
            
            # Extract latents from full noise image (no mask needed - noise has no identity)
            with torch.no_grad():
                noise_preprocessed = ImageProcessor.preprocess_for_model_tensor(noise_tensor)
                target_latents, target_f4_from_face = self.identity_extractor(noise_preprocessed)
                if self.verbose:
                    print(f"Extracted noise target latents: shape {target_latents.shape}")
        else:
            # Default: 'image' - use target_image_path or image_path
            if self.verbose:
                print(f"Target type: image (using {'target' if target_image_path else 'source'} identity)")
            target_identity_path = target_image_path if target_image_path is not None else image_path
            target_image_np = ImageProcessor.load_image(target_identity_path, target_size=(256, 256))
            target_image_np = ImageProcessor.apply_preprocessing(target_image_np, mode=self.preproc_mode)
            if target_image_np is None:
                raise ValueError(f"Failed to load target identity image: {target_identity_path}")
            
            target_image_tensor = torch.from_numpy(target_image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            
            # Generate masks for target identity image (needed for face region extraction)
            target_M1, _ = self.generate_masks(target_image_np)
            
            with torch.no_grad():
                target_preprocessed_face = ImageProcessor.preprocess_for_model_tensor(target_image_tensor * M1)
                target_latents, target_f4_from_face = self.identity_extractor(target_preprocessed_face)
                
                # Note: _target_f4_full is already extracted from original image above (for structure_weakening)
                # It's not extracted from target image, as structure_weakening compares against original structure
        
        # 4. Initialize perturbation
        if self.random_init:
            # Random initialization: uniform random in [-epsilon, epsilon]
            delta = torch.empty_like(image_tensor, requires_grad=True).to(self.device)
            delta.data.uniform_(-self.epsilon, self.epsilon)
        else:
            # Zero initialization (default)
            delta = torch.zeros_like(image_tensor, requires_grad=True).to(self.device)
        
        # 5. PGD loop
        for i in range(self.num_iter):
            if delta.grad is not None:
                delta.grad.zero_()
            
            # Optional loss schedule (simple ramp for lambda weights)
            if self.loss_schedule:
                t = (i + 1) / float(max(1, self.num_iter))
                # emphasize semantic later (example schedule)
                lambda_1 = self.lambda_1 * (1.0 - 0.3 * t)
                lambda_2 = self.lambda_2 * (0.7 + 0.3 * t)
            else:
                lambda_1 = self.lambda_1
                lambda_2 = self.lambda_2

            L_ID_loss = torch.zeros(1, device=self.device)
            L_SEM_loss = torch.zeros(1, device=self.device)
            L_sim_loss = torch.zeros(1, device=self.device)
            L_TV_loss = torch.zeros(1, device=self.device)
            cos_sim = torch.zeros(1, device=self.device)
            last_grad_norm = 0.0
            
            # Compute all losses first (without backward)
            # Identity loss (eval mode)
            if lambda_1 > 0:
                self.identity_extractor.eval()
                adv_image_face = torch.clamp(image_tensor + delta, 0, 255)
                adv_face_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_face * M1)
                adv_latents, _ = self.identity_extractor(adv_face_preprocessed)
                
                # Compute target latents if needed (for zero/uniform, create after getting adv_latents shape)
                if self.target_type in ['zero', 'zero-l1', 'zero-huber']:
                    target_latents = torch.zeros_like(adv_latents)
                elif self.target_type == 'uniform':
                    # Create uniform vector: all elements = 1/dim, then normalize
                    target_latents = torch.ones_like(adv_latents) / adv_latents.shape[1]
                    target_latents = F.normalize(target_latents, p=2, dim=1)
                
                # Compute L_ID_loss based on target_type
                if self.target_type == 'entropy':
                    # Entropy maximization: maximize entropy of adv_latents
                    probs = F.softmax(adv_latents, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    L_ID_loss = -entropy  # Minimize negative entropy = maximize entropy
                elif self.target_type == 'zero':
                    # Zero target: use MSE (default)
                    L_ID_loss = F.mse_loss(adv_latents, target_latents)
                elif self.target_type == 'zero-l1':
                    # Zero target: use L1 loss (more robust to outliers)
                    L_ID_loss = F.l1_loss(adv_latents, target_latents)
                elif self.target_type == 'zero-huber':
                    # Zero target: use Huber loss (smooth L1, robust to outliers)
                    delta_huber = adv_latents - target_latents
                    huber_delta = 1.0  # Huber loss threshold
                    abs_delta = torch.abs(delta_huber)
                    quadratic = torch.clamp(abs_delta, max=huber_delta)
                    linear = abs_delta - quadratic
                    L_ID_loss = (0.5 * quadratic.pow(2) + huber_delta * linear).mean()
                elif self.target_type == 'uniform':
                    # Uniform target: use MSE
                    L_ID_loss = F.mse_loss(adv_latents, target_latents)
                else:
                    # Default: cosine similarity (for 'image' and 'noise')
                    if target_latents is None:
                        raise ValueError(f"target_latents is None for target_type={self.target_type}")
                    cos_sim = F.cosine_similarity(adv_latents, target_latents).mean()
                    # If maximize_similarity=True, use (1 - cos_sim) to maximize similarity
                    # If False, use cos_sim to minimize similarity (identity destruction)
                    if self.maximize_similarity:
                        L_ID_loss = 1.0 - cos_sim
                    else:
                        L_ID_loss = cos_sim
            else:
                self.identity_extractor.eval()
                with torch.no_grad():
                    adv_image_face = torch.clamp(image_tensor + delta, 0, 255)
                    adv_face_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_face * M1)
                    adv_latents, _ = self.identity_extractor(adv_face_preprocessed)
                    
                    # Compute target latents if needed (for zero/uniform)
                    if self.target_type in ['zero', 'zero-l1', 'zero-huber']:
                        target_latents = torch.zeros_like(adv_latents)
                    elif self.target_type == 'uniform':
                        target_latents = torch.ones_like(adv_latents) / adv_latents.shape[1]
                        target_latents = F.normalize(target_latents, p=2, dim=1)
                    
                    # Compute L_ID_loss based on target_type
                    if self.target_type == 'entropy':
                        probs = F.softmax(adv_latents, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                        L_ID_loss = -entropy
                    elif self.target_type == 'zero':
                        # Zero target: use MSE (default)
                        L_ID_loss = F.mse_loss(adv_latents, target_latents)
                    elif self.target_type == 'zero-l1':
                        # Zero target: use L1 loss (more robust to outliers)
                        L_ID_loss = F.l1_loss(adv_latents, target_latents)
                    elif self.target_type == 'zero-huber':
                        # Zero target: use Huber loss (smooth L1, robust to outliers)
                        delta_huber = adv_latents - target_latents
                        huber_delta = 1.0  # Huber loss threshold
                        abs_delta = torch.abs(delta_huber)
                        quadratic = torch.clamp(abs_delta, max=huber_delta)
                        linear = abs_delta - quadratic
                        L_ID_loss = (0.5 * quadratic.pow(2) + huber_delta * linear).mean()
                    elif self.target_type == 'uniform':
                        # Uniform target: use MSE
                        L_ID_loss = F.mse_loss(adv_latents, target_latents)
                    else:
                        if target_latents is None:
                            raise ValueError(f"target_latents is None for target_type={self.target_type}")
                        cos_sim = F.cosine_similarity(adv_latents, target_latents).mean()
                        if self.maximize_similarity:
                            L_ID_loss = 1.0 - cos_sim
                        else:
                            L_ID_loss = cos_sim
            
            # Semantic loss (train mode, but compute without backward)
            if lambda_2 > 0:
                self.identity_extractor.train()
                
                if self.sem_variant == 'structure_weakening':
                    adv_image_full = torch.clamp(image_tensor + delta, 0, 255)
                    adv_full_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_full)
                    _, adv_f4_full = self.identity_extractor(adv_full_preprocessed)
                    L_SEM_loss = F.mse_loss(adv_f4_full, self._target_f4_full * self.structure_weakening_factor)
                else:
                    adv_image_bg = torch.clamp(image_tensor + delta, 0, 255)
                    adv_bg_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_bg * M2)
                    _, adv_f4_from_bg = self.identity_extractor(adv_bg_preprocessed)
                    
                    if self.sem_variant == 'l1_f4':
                        L_SEM_loss = F.l1_loss(adv_f4_from_bg, target_f4_from_face)
                    elif self.sem_variant == 'self_collapse':
                        L_SEM_loss = torch.mean(adv_f4_from_bg.pow(2))
                    elif self.sem_variant == 'self_collapse_mid':
                        f4_mid, f8_mid, f16_mid, f32_mid = self.identity_extractor.fpn(adv_bg_preprocessed)
                        L_SEM_loss = torch.mean(f8_mid.pow(2)) + 0.5 * torch.mean(f16_mid.pow(2))
                    elif self.sem_variant == 'contrastive_bg':
                        margin = 0.2
                        cos_bg_face = F.cosine_similarity(adv_f4_from_bg.flatten(1), target_f4_from_face.flatten(1)).mean()
                        L_SEM_loss = F.relu(margin - (1.0 - cos_bg_face))
                    else:
                        L_SEM_loss = F.mse_loss(adv_f4_from_bg, target_f4_from_face)
            else:
                self.identity_extractor.train()
                with torch.no_grad():
                    if self.sem_variant == 'structure_weakening':
                        adv_image_full = torch.clamp(image_tensor + delta, 0, 255)
                        adv_full_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_full)
                        _, adv_f4_full = self.identity_extractor(adv_full_preprocessed)
                        L_SEM_loss = F.mse_loss(adv_f4_full, self._target_f4_full * self.structure_weakening_factor)
                    else:
                        adv_image_bg = torch.clamp(image_tensor + delta, 0, 255)
                        adv_bg_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_bg * M2)
                        _, adv_f4_from_bg = self.identity_extractor(adv_bg_preprocessed)
                        if self.sem_variant == 'l1_f4':
                            L_SEM_loss = F.l1_loss(adv_f4_from_bg, target_f4_from_face)
                        elif self.sem_variant == 'self_collapse':
                            L_SEM_loss = torch.mean(adv_f4_from_bg.pow(2))
                        elif self.sem_variant == 'self_collapse_mid':
                            f4_mid, f8_mid, f16_mid, f32_mid = self.identity_extractor.fpn(adv_bg_preprocessed)
                            L_SEM_loss = torch.mean(f8_mid.pow(2)) + 0.5 * torch.mean(f16_mid.pow(2))
                        elif self.sem_variant == 'contrastive_bg':
                            margin = 0.2
                            cos_bg_face = F.cosine_similarity(adv_f4_from_bg.flatten(1), target_f4_from_face.flatten(1)).mean()
                            L_SEM_loss = F.relu(margin - (1.0 - cos_bg_face))
                        else:
                            L_SEM_loss = F.mse_loss(adv_f4_from_bg, target_f4_from_face)
            
            # Similarity preservation loss
            if self.lambda_sim > 0:
                adv_image_full = torch.clamp(image_tensor + delta, 0, 255)
                
                if self.sim_loss_type == 'mse':
                    L_sim_loss = F.mse_loss(adv_image_full, image_tensor)
                elif self.sim_loss_type == 'l1':
                    L_sim_loss = F.l1_loss(adv_image_full, image_tensor)
                elif self.sim_loss_type == 'lpips':
                    if self.lpips_model is not None:
                        adv_lpips = (adv_image_full / 127.5) - 1.0
                        orig_lpips = (image_tensor / 127.5) - 1.0
                        L_sim_loss = self.lpips_model(adv_lpips, orig_lpips).mean()
                    else:
                        L_sim_loss = F.mse_loss(adv_image_full, image_tensor)
                elif self.sim_loss_type == 'perceptual':
                    adv_preprocessed = ImageProcessor.preprocess_for_model_tensor(adv_image_full)
                    clean_preprocessed = ImageProcessor.preprocess_for_model_tensor(image_tensor)
                    with torch.no_grad():
                        _, clean_f4 = self.identity_extractor(clean_preprocessed)
                    _, adv_f4 = self.identity_extractor(adv_preprocessed)
                    L_sim_loss = F.mse_loss(adv_f4, clean_f4)
                else:
                    L_sim_loss = torch.zeros(1, device=self.device)
            else:
                L_sim_loss = torch.zeros(1, device=self.device)
            
            # Total Variation loss
            if self.lambda_tv > 0:
                adv_image_full = torch.clamp(image_tensor + delta, 0, 255)
                L_TV_loss = total_variation_loss(delta)
            else:
                L_TV_loss = torch.zeros(1, device=self.device)
            
            # Compute total loss and backward once
            total_loss = (lambda_1 * L_ID_loss) + (lambda_2 * L_SEM_loss) + (self.lambda_sim * L_sim_loss) + (self.lambda_tv * L_TV_loss)
            
            # Backward on total loss (all gradients accumulated properly)
            total_loss.backward()
            
            # Apply gradients with L2 normalization (preserves lambda direction, ensures consistent step size)
            if delta.grad is not None:
                grad = delta.grad.detach()
                if self.clip_grad and self.clip_grad > 0:
                    grad = torch.clamp(grad, -self.clip_grad, self.clip_grad)
                # L2 normalization: preserves direction (lambda ratio) while ensuring alpha = actual step size
                # This prevents gradient vanishing/exploding and makes alpha a true distance metric
                normalized_grad = F.normalize(grad, p=2, dim=[1, 2, 3], eps=1e-8)
                delta.data = delta.data - self.alpha * normalized_grad
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                delta.grad.zero_()
                last_grad_norm = normalized_grad.abs().mean().item()
            
            # Early stopping: check convergence based on loss change (only)
            early_stop = False
            if i >= self.min_iter_for_convergence:
                # Check convergence based on loss history
                if len(self.loss_history['L_ID']) >= self.convergence_window:
                    # Get recent L_ID losses
                    recent_losses = self.loss_history['L_ID'][-self.convergence_window:]
                    # Calculate loss change over the window
                    loss_change = abs(recent_losses[-1] - recent_losses[0])
                    # Check if loss has converged (change is below tolerance)
                    if loss_change < self.convergence_tolerance:
                        early_stop = True
                        if self.verbose:
                            print(f"Early stopping at iter {i}: Convergence detected (L_ID change={loss_change:.6f} < {self.convergence_tolerance} over {self.convergence_window} iterations)")
                        self.loss_history['early_stopped'] = True
            
            # Logging
            with torch.no_grad():
                eps_sat = (delta.abs() >= (self.epsilon - 1e-6)).float().mean().item() * 100.0
            self.loss_history['total'].append(float(total_loss.item()))
            self.loss_history['L_ID'].append(float(L_ID_loss.item()))
            self.loss_history['L_sim'].append(float(L_sim_loss.item()))
            self.loss_history['L_TV'].append(float(L_TV_loss.item()))
            self.loss_history['cos_sim'].append(float(cos_sim.item()))
            self.loss_history['eps_sat_pct'].append(eps_sat)
            self.loss_history['grad_norm'].append(last_grad_norm)
            
            if self.verbose and (i % 20 == 0 or i == self.num_iter - 1 or early_stop):
                sim_str = f", L_sim={L_sim_loss.item():.4f}" if self.lambda_sim > 0 else ""
                tv_str = f", L_TV={L_TV_loss.item():.4f}" if self.lambda_tv > 0 else ""
                print(
                    f"Iter {i:5d}: Total={total_loss.item():.4f}, "
                    f"L_ID={L_ID_loss.item():.4f} (cos={float(cos_sim.item()):.4f})"
                    f"{sim_str}{tv_str}, eps_sat={eps_sat:.1f}%"
                )
            
            # Break if early stopping
            if early_stop:
                break
        
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
            
            axes[0,2].plot(self.loss_history.get('L_sim', []))
            axes[0,2].set_title('L_SIM (Similarity Preservation)')
            axes[0,2].set_xlabel('Iteration')
            axes[0,2].set_ylabel('Loss')

            axes[1,0].plot(self.loss_history.get('L_TV', []))
            axes[1,0].set_title('L_TV (Total Variation)')
            axes[1,0].set_xlabel('Iteration')
            axes[1,0].set_ylabel('Loss')

            axes[1,1].plot(self.loss_history.get('eps_sat_pct', []))
            axes[1,1].set_title('Epsilon Saturation (%)')
            axes[1,1].set_xlabel('Iteration')
            axes[1,1].set_ylabel('% pixels at |delta|=epsilon')

            axes[1,2].plot(self.loss_history.get('grad_norm', []))
            axes[1,2].set_title('Grad Norm (mean |grad|)')
            axes[1,2].set_xlabel('Iteration')
            axes[1,2].set_ylabel('value')
            
            # Add early stopping indicator if applicable
            if self.loss_history.get('early_stopped', False):
                fig.suptitle('Loss Curves (Early Stopped)', fontsize=14, fontweight='bold')
            
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
    
    # SSIM (if available)
    if _SSIM_AVAILABLE and skimage_ssim is not None:
        try:
            ssim_val = skimage_ssim(original, adversarial, channel_axis=2)
            metrics['SSIM'] = float(ssim_val)
        except Exception:
            pass
    
    # LPIPS (if available)
    if _LPIPS_AVAILABLE:
        global _LPIPS_MODEL
        try:
            if _LPIPS_MODEL is None:
                _LPIPS_MODEL = lpips.LPIPS(net='alex')
            # Convert numpy images [H,W,3] in 0-255 to torch [-1,1]
            def _to_lpips_tensor(img: np.ndarray) -> torch.Tensor:
                tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
                tensor = tensor / 127.5 - 1.0
                return tensor
            orig_tensor = _to_lpips_tensor(original)
            adv_tensor = _to_lpips_tensor(adversarial)
            with torch.no_grad():
                lpips_val = _LPIPS_MODEL(orig_tensor, adv_tensor).item()
            metrics['LPIPS'] = float(lpips_val)
        except Exception:
            pass
    
    return metrics

