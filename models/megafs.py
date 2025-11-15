"""
MegaFS main class implementation
Based on One-Shot-Face-Swapping-on-Megapixels repository
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tF
from .resnet import resnet50
from .hierfe import HieRFE
from .face_transfer import FaceTransferModule
from .stylegan2 import Generator
from .soft_erosion import SoftErosion


def encode_segmentation_rgb(segmentation, no_neck=True):
    """Encode segmentation mask to RGB format"""
    parse = segmentation[:,:,0]

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    valid_index = np.where(parse==hair_id)
    hair_map[valid_index] = 255

    return np.stack([face_map, mouth_map, hair_map], axis=2)


from typing import Any, Dict, Optional, Tuple
import sys
import os
import cv2
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.model_factory import ModelFactory
from models.weight_loaders import verify_all_weights
from utils.image_utils import ImageProcessor, ImageLoader, encode_segmentation_rgb
from utils.mask_converter import build_labeled_png
from utils.data_utils import DataMapManager
from utils.debug_utils import DebugLogger, PerformanceProfiler, check_system_requirements


class MegaFS(nn.Module):
    """MegaFS class for face swapping - Modular version with nn.Module support"""
    
    def __init__(self, 
                 swap_type: str = "ftm",
                 img_root: str = "",
                 mask_root: str = "",
                 checkpoint_dir: str = "weights",
                 data_map: Optional[Dict[int, Dict[str, Any]]] = None,
                 config: Optional[Config] = None,
                 debug: bool = True,
                 enable_grads: bool = False,
                 device: str = "cuda"):
        
        # Initialize nn.Module
        super(MegaFS, self).__init__()
        
        # Initialize configuration
        if config is None:
            self.config = Config(
                swap_type=swap_type,
                img_root=img_root,
                mask_root=mask_root,
                checkpoint_dir=checkpoint_dir
            )
        else:
            self.config = config
        
        # Store device
        self.device_str = device
        self.device = torch.device(device)
        
        # Initialize utilities
        self.debug_logger = DebugLogger(enabled=debug)
        self.profiler = PerformanceProfiler()
        self.image_loader = ImageLoader()
        
        # Initialize data manager
        self.data_manager = DataMapManager()
        if data_map:
            self.data_manager.data_map = data_map
        
        # Gradient computation mode
        self.enable_grads = enable_grads
        
        # Print configuration
        if debug:
            self.config.print_config()
            check_system_requirements()
        
        # Verify weights before loading
        if not verify_all_weights(self.config.paths.checkpoint_dir):
            raise RuntimeError("Required weight files are missing or invalid")
        
        # Initialize model factory and create models
        self.model_factory = ModelFactory(self.config.paths.checkpoint_dir, device=device)
        self.models = self.model_factory.create_all_models(self.config.swap.swap_type)
        
        # Register models as submodules using nn.Module's register_module
        self.encoder = self.models["encoder"]
        self.swapper = self.models["swapper"]
        self.generator = self.models["generator"]
        
        # Register as submodules
        self.add_module('encoder', self.encoder)
        self.add_module('swapper', self.swapper)
        self.add_module('generator', self.generator)
        
        # Set model mode based on gradient configuration
        self.set_gradient_mode(self.enable_grads)
        
        # Initialize smooth mask
        from .soft_erosion import SoftErosion
        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7)
        self.smooth_mask.to(self.device)
        self.smooth_mask.eval()
        self.add_module('smooth_mask', self.smooth_mask)
        
        if self.enable_grads:
            self.smooth_mask.train()
        
        try:
            dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                _ = self.encoder(dummy_input)
        except Exception as e:
            pass
        
        try:
            dummy_struct = torch.randn(1, 512, 4, 4).to(self.device)
            dummy_lats = torch.randn(1, 18, 512).to(self.device)
            with torch.no_grad():
                _ = self.generator(dummy_struct, [dummy_lats, None], randomize_noise=False)
        except Exception as e:
            pass
    
    def set_gradient_mode(self, enabled: bool):
        """Toggle between eval (inference) and train (gradients) mode"""
        self.enable_grads = enabled
        if enabled:
            self.train()
        else:
            self.eval()
        
        if self.debug_logger.enabled:
            status = "ENABLED" if enabled else "DISABLED"
            self.debug_logger.log(f"Gradient computation {status}", "INFO")
    
    def train(self, mode: bool = True):
        """Override train() to set model to training mode"""
        super().train(mode)
        # Set gradient mode based on training mode
        self.enable_grads = mode
        # All submodules will be set to train/eval by super().train()
        return self
    
    def eval(self):
        """Override eval() to set model to evaluation mode"""
        super().eval()
        # Disable gradients in eval mode
        self.enable_grads = False
        # All submodules will be set to eval by super().eval()
        return self
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Clean forward pass that preserves gradients.
        
        Args:
            source: Source image tensor [B, 3, 256, 256]
            target: Target image tensor [B, 3, 256, 256]
            
        Returns:
            Swapped face tensor [B, 3, 1024, 1024]
        """
        # Ensure on correct device
        source = source.to(self.device)
        target = target.to(self.device)
        
        # Concatenate for encoding
        ts = torch.cat([target, source], dim=0)
        
        # Encode
        lats, struct = self.encoder(ts)
        
        # Extract latents
        idd_lats = lats[1:]  # Source latents
        att_lats = lats[0].unsqueeze(0)  # Target latents
        att_struct = struct[0].unsqueeze(0)  # Target structure
        
        # Swap
        swapped_lats = self.swapper(idd_lats, att_lats)
        
        # Generate
        fake_swap, _ = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)
        
        return fake_swap

    def read_pair(self, src_idx: int, tgt_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Read source and target image pair using data manager."""
        # Use data manager to resolve paths
        src_img_path, _ = self.data_manager.resolve_paths_for_id(src_idx, self.config.paths.dataset_root)
        tgt_img_path, tgt_mask_path = self.data_manager.resolve_paths_for_id(tgt_idx, self.config.paths.dataset_root)
        
        # Load images using image processor
        target_size = (self.config.model.size, self.config.model.size)
        src_image = ImageProcessor.load_image(src_img_path, target_size=None) if src_img_path else None
        tgt_image = ImageProcessor.load_image(tgt_img_path, target_size=None) if tgt_img_path else None
        tgt_mask = ImageProcessor.load_image(tgt_mask_path, target_size=None) if tgt_mask_path else None
        # If mask PNG not present, attempt to build it from mask-anno on the fly
        if tgt_mask is None and self.config.paths.mask_root:
            try:
                anno_root = os.path.join(self.config.paths.dataset_root, "CelebAMask-HQ-mask-anno")
                out_root = os.path.join(self.config.paths.dataset_root, "CelebAMaskHQ-mask")
                # id string from filename path resolver if available
                gid = int(tgt_idx)
                out_png = build_labeled_png(anno_root, gid, out_root)
                tgt_mask = ImageProcessor.load_image(out_png, target_size=None)
            except Exception as _:
                pass

        # Convert labeled PNG to 3-channel (face/mouth/hair) mask as original
        if tgt_mask is not None:
            tgt_mask = encode_segmentation_rgb(tgt_mask)
        
        if src_image is None:
            raise FileNotFoundError(f"Source image not found for ID {src_idx}")
        if tgt_image is None:
            raise FileNotFoundError(f"Target image not found for ID {tgt_idx}")
        
        # For strict parity with original, do not align here; return raw RGB
        return src_image, tgt_image, tgt_mask

    def preprocess(self, src: np.ndarray, tgt: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess images for model input using ImageProcessor"""
        # Resize images
        src_resized = cv2.resize(src.copy(), (256, 256))
        tgt_resized = cv2.resize(tgt.copy(), (256, 256))
        
        # Convert to tensors and normalize
        src_tensor = ImageProcessor.preprocess_for_model(src_resized, normalize=True)
        tgt_tensor = ImageProcessor.preprocess_for_model(tgt_resized, normalize=True)
        
        return src_tensor.unsqueeze_(0), tgt_tensor.unsqueeze_(0)

    def run(self, src_idx: int, tgt_idx: int, refine: bool = True, save_path: Optional[str] = None):
        try:
            src_face_rgb, tgt_face_rgb, tgt_mask = self.read_pair(src_idx, tgt_idx)
            source, target = self.preprocess(src_face_rgb, tgt_face_rgb)
            
            swapped_face = self.swap(source, target)
            swapped_face = self.postprocess(swapped_face, tgt_face_rgb, tgt_mask)

            result = np.hstack((src_face_rgb, tgt_face_rgb, swapped_face))

            if refine:
                swapped_tensor, _ = self.preprocess(swapped_face, swapped_face)
                refined_face = self.refine(swapped_tensor)
                refined_face = self.postprocess(refined_face, tgt_face_rgb, tgt_mask)
                result = np.hstack((result, refined_face))

            if save_path:
                if ImageProcessor.save_image(result, save_path):
                    return save_path, result
                else:
                    return None, result
            else:
                return None, result
                
        except Exception as e:
            self.debug_logger.log(f"Error in face swap: {e}", "ERROR")
            raise

    def swap(self, source: torch.Tensor, target: torch.Tensor, return_tensor: bool = False) -> torch.Tensor:
        """
        Swap faces from source to target.
        
        Args:
            source: Source image tensor
            target: Target image tensor
            return_tensor: If True, return tensor. If False, return numpy array (original behavior)
            
        Returns:
            Swapped face as tensor or numpy array
        """
        print(f'[DEBUG][swap] source: {type(source)}, shape={getattr(source, "shape", None)}')
        print(f'[DEBUG][swap] target: {type(target)}, shape={getattr(target, "shape", None)}')
        assert isinstance(source, torch.Tensor), f'source is {type(source)}!'
        assert isinstance(target, torch.Tensor), f'target is {type(target)}!'
        # Choose context based on gradient mode
        context = torch.enable_grad() if self.enable_grads else torch.no_grad()
        
        with context:
            try:
                # Ensure inputs are tensors
                if not isinstance(target, torch.Tensor):
                    if isinstance(target, tuple):
                        target = target[0]
                    target = torch.tensor(target) if isinstance(target, np.ndarray) else target
                if not isinstance(source, torch.Tensor):
                    if isinstance(source, tuple):
                        source = source[0]
                    source = torch.tensor(source) if isinstance(source, np.ndarray) else source
                
                ts = torch.cat([target, source], dim=0).to(self.device)
                encoder_output = self.encoder(ts)
                
                # Handle encoder output - it should return (latents, f4)
                if isinstance(encoder_output, tuple):
                    lats, struct = encoder_output
                else:
                    # If encoder returns single value, this is an error
                    raise ValueError(f"Encoder returned unexpected type: {type(encoder_output)}, expected tuple")

                idd_lats = lats[1:]
                att_lats = lats[0].unsqueeze_(0)
                att_struct = struct[0].unsqueeze_(0)

                swapped_lats = self.swapper(idd_lats, att_lats)
                generator_output = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)
                
                # Handle generator output - it should return (image, noise) tuple
                if isinstance(generator_output, tuple):
                    fake_swap, _ = generator_output
                else:
                    fake_swap = generator_output
                
            except Exception as e:
                self.debug_logger.log(f"Error in swap method: {e}", "ERROR")
                import traceback
                self.debug_logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
                raise

            # If returning tensor, return raw output
            if return_tensor:
                return fake_swap
            
            # Original behavior: normalize and convert to numpy
            # Detach if gradients are enabled to allow numpy conversion
            fake_swap_detached = fake_swap.detach() if fake_swap.requires_grad else fake_swap
            fake_swap_max = torch.max(fake_swap_detached)
            fake_swap_min = torch.min(fake_swap_detached)
            denormed_fake_swap = (fake_swap_detached[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0
            fake_swap_numpy = denormed_fake_swap.permute((1, 2, 0)).cpu().numpy()
            return fake_swap_numpy

    def refine(self, swapped_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            try:
                lats, struct = self.encoder(swapped_tensor.to(self.device))
                fake_refine, _ = self.generator(struct, [lats, None], randomize_noise=False)
                
            except Exception as e:
                self.debug_logger.log(f"Error in refine method: {e}", "ERROR")
                raise

            fake_refine_max = torch.max(fake_refine)
            fake_refine_min = torch.min(fake_refine)
            denormed_fake_refine = (fake_refine[0] - fake_refine_min) / (fake_refine_max - fake_refine_min) * 255.0
            fake_refine_numpy = denormed_fake_refine.permute((1, 2, 0)).cpu().numpy()
        return fake_refine_numpy

    def postprocess(self, swapped_face: np.ndarray, target: np.ndarray, target_mask: Optional[np.ndarray]) -> np.ndarray:
        """Postprocess swapped face with optional mask blending using ImageProcessor"""
        if target_mask is None:
            # Keep RGB and return as uint8
            return swapped_face.astype(np.uint8)

        # Resize mask to target size (nearest to preserve labels)
        target_mask = cv2.resize(
            target_mask, (self.config.model.size, self.config.model.size), interpolation=cv2.INTER_NEAREST
        )

        # Convert mask to tensor and process like original
        mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).to(self.device)
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]  # face + mouth channels like original

        # Apply smooth mask
        soft_face_mask_tensor, _ = self.smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()

        soft_face_mask = soft_face_mask_tensor.cpu().numpy()[:, :, np.newaxis]
        result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        return result.astype(np.uint8)  # Keep RGB
