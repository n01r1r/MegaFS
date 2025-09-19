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
from utils.image_utils import ImageProcessor, ImageLoader
from utils.mask_converter import build_labeled_png
from utils.data_utils import DataMapManager
from utils.debug_utils import DebugLogger, PerformanceProfiler, check_system_requirements


class MegaFS(object):
    """MegaFS class for face swapping - Modular version"""
    
    def __init__(self, 
                 swap_type: str = "ftm",
                 img_root: str = "",
                 mask_root: str = "",
                 checkpoint_dir: str = "weights",
                 data_map: Optional[Dict[int, Dict[str, Any]]] = None,
                 config: Optional[Config] = None,
                 debug: bool = True):
        
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
        
        # Initialize utilities
        self.debug_logger = DebugLogger(enabled=debug)
        self.profiler = PerformanceProfiler()
        self.image_loader = ImageLoader()
        
        # Initialize data manager
        self.data_manager = DataMapManager()
        if data_map:
            self.data_manager.data_map = data_map
        
        # Print configuration
        if debug:
            self.config.print_config()
            check_system_requirements()
        
        # Verify weights before loading
        if not verify_all_weights(self.config.paths.checkpoint_dir):
            raise RuntimeError("Required weight files are missing or invalid")
        
        # Initialize model factory and create models
        self.model_factory = ModelFactory(self.config.paths.checkpoint_dir)
        self.models = self.model_factory.create_all_models(self.config.swap.swap_type)
        
        # Assign models for backward compatibility
        self.encoder = self.models["encoder"]
        self.swapper = self.models["swapper"]
        self.generator = self.models["generator"]
        # Force eval for safety
        self.encoder.eval()
        self.swapper.eval()
        self.generator.eval()
        
        # Initialize smooth mask
        from .soft_erosion import SoftErosion
        self.smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        self.smooth_mask.eval()
        
        # Initialize LazyModules with dummy forward pass
        if debug:
            self.debug_logger.log("Initializing LazyModules with dummy forward pass...")
        
        try:
            # Create dummy input for encoder initialization
            dummy_input = torch.randn(1, 3, 256, 256).cuda()
            with torch.no_grad():
                _ = self.encoder(dummy_input)
            if debug:
                self.debug_logger.log("Encoder LazyModule initialized successfully")
        except Exception as e:
            if debug:
                self.debug_logger.log(f"Encoder initialization failed: {e}", "WARNING")
        
        # Initialize generator with dummy forward pass
        try:
            dummy_struct = torch.randn(1, 512, 4, 4).cuda()
            dummy_lats = torch.randn(1, 18, 512).cuda()
            with torch.no_grad():
                _ = self.generator(dummy_struct, [dummy_lats, None], randomize_noise=False)
            if debug:
                self.debug_logger.log("Generator LazyModule initialized successfully")
        except Exception as e:
            if debug:
                self.debug_logger.log(f"Generator initialization failed: {e}", "WARNING")
        
        # Log model information
        if debug:
            try:
                self.debug_logger.log_model_info(self.encoder, "Encoder")
            except Exception as e:
                self.debug_logger.log(f"Encoder info logging failed: {e}", "WARNING")
            
            try:
                self.debug_logger.log_model_info(self.swapper, "Swapper")
            except Exception as e:
                self.debug_logger.log(f"Swapper info logging failed: {e}", "WARNING")
            
            try:
                self.debug_logger.log_model_info(self.generator, "Generator")
            except Exception as e:
                self.debug_logger.log(f"Generator info logging failed: {e}", "WARNING")
            
            self.debug_logger.log("MegaFS initialization completed successfully")

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
                self.debug_logger.log(f"Built mask on-the-fly: {out_png}")
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
        """Run face swapping with improved error handling and logging"""
        try:
            self.debug_logger.log(f"Starting face swap: {src_idx} -> {tgt_idx}")
            
            # Load and preprocess images
            src_face_rgb, tgt_face_rgb, tgt_mask = self.read_pair(src_idx, tgt_idx)
            source, target = self.preprocess(src_face_rgb, tgt_face_rgb)
            
            # Perform face swapping
            self.profiler.start_timer("face_swap")
            swapped_face = self.swap(source, target)
            swap_time = self.profiler.end_timer("face_swap")
            self.debug_logger.log_timing("Face Swap", swap_time)
            
            # Postprocess result
            swapped_face = self.postprocess(swapped_face, tgt_face_rgb, tgt_mask)

            # Create result image
            # All images are RGB here; keep RGB until save
            result = np.hstack((src_face_rgb, tgt_face_rgb, swapped_face))

            # Refine if requested
            if refine:
                self.profiler.start_timer("refinement")
                swapped_tensor, _ = self.preprocess(swapped_face, swapped_face)
                refined_face = self.refine(swapped_tensor)
                refined_face = self.postprocess(refined_face, tgt_face_rgb, tgt_mask)
                result = np.hstack((result, refined_face))
                refine_time = self.profiler.end_timer("refinement")
                self.debug_logger.log_timing("Refinement", refine_time)

            # Save if path provided
            if save_path:
                if ImageProcessor.save_image(result, save_path):
                    self.debug_logger.log(f"Result saved: {save_path}")
                    return save_path, result
                else:
                    self.debug_logger.log("Failed to save result", "ERROR")
                    return None, result
            else:
                return None, result
                
        except Exception as e:
            self.debug_logger.log(f"Error in face swap: {e}", "ERROR")
            raise

    def swap(self, source: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """Perform face swapping"""
        with torch.no_grad():
            try:
                self.debug_logger.log(f"Input shapes - source: {source.shape}, target: {target.shape}")
                
                ts = torch.cat([target, source], dim=0).cuda()
                self.debug_logger.log(f"Concatenated tensor shape: {ts.shape}")
                
                lats, struct = self.encoder(ts)
                self.debug_logger.log(f"Encoder output - lats: {lats.shape}, struct: {struct.shape}")

                # lats는 [2, num_latents, 512] 형태의 텐서
                # struct는 [2, C, H, W] 형태의 텐서
                idd_lats = lats[1:]  # 소스 이미지의 latent [1, num_latents, 512]
                att_lats = lats[0].unsqueeze_(0)  # 타겟 이미지의 latent [1, num_latents, 512]
                att_struct = struct[0].unsqueeze_(0)  # 타겟 이미지의 구조 [1, C, H, W]
                
                self.debug_logger.log(f"Latent shapes - idd_lats: {idd_lats.shape}, att_lats: {att_lats.shape}")
                self.debug_logger.log(f"Structure shape - att_struct: {att_struct.shape}")

                swapped_lats = self.swapper(idd_lats, att_lats)
                self.debug_logger.log(f"Swapper output shape: {swapped_lats.shape}")

                # 원본 코드와 동일한 방식으로 generator 호출
                # [swapped_lats, None] 형태로 직접 전달
                
                # 디버깅: generator 호출 전 텐서 상태 확인
                self.debug_logger.log(f"Generator input - att_struct: {att_struct.shape}, dtype: {att_struct.dtype}")
                self.debug_logger.log(f"Generator input - swapped_lats: {swapped_lats.shape}, dtype: {swapped_lats.dtype}")
                self.debug_logger.log(f"Generator input - styles: {[swapped_lats.shape, None]}")
                
                # Generator의 파라미터 상태 확인
                for name, param in self.generator.named_parameters():
                    if 'weight' in name and param.dim() < 3:
                        self.debug_logger.log(f"WARNING: Parameter {name} has {param.dim()} dimensions: {param.shape}")
                    if 'weight' in name and param.numel() == 0:
                        self.debug_logger.log(f"WARNING: Parameter {name} is empty: {param.shape}")
                
                # Generator의 첫 번째 conv layer 확인
                if hasattr(self.generator, 'conv1'):
                    conv1_weight = self.generator.conv1.conv.weight
                    self.debug_logger.log(f"Generator conv1 weight shape: {conv1_weight.shape}, dims: {conv1_weight.dim()}")
                
                fake_swap, _ = self.generator(att_struct, [swapped_lats, None], randomize_noise=False)
                self.debug_logger.log(f"Generator output shape: {fake_swap.shape}")
                
            except Exception as e:
                self.debug_logger.log(f"Error in swap method: {e}", "ERROR")
                raise

            fake_swap_max = torch.max(fake_swap)
            fake_swap_min = torch.min(fake_swap)
            denormed_fake_swap = (fake_swap[0] - fake_swap_min) / (fake_swap_max - fake_swap_min) * 255.0
            fake_swap_numpy = denormed_fake_swap.permute((1, 2, 0)).cpu().numpy()
        return fake_swap_numpy

    def refine(self, swapped_tensor: torch.Tensor) -> np.ndarray:
        """Refine swapped face by re-encoding and generating."""
        with torch.no_grad():
            try:
                self.debug_logger.log(f"Refine input shape: {swapped_tensor.shape}")
                
                # 스왑 결과를 재인코딩하여 latent만 사용해 재생성합니다.
                lats, struct = self.encoder(swapped_tensor.cuda())
                self.debug_logger.log(f"Refine encoder output - lats: {lats.shape}, struct: {struct.shape}")

                # 원본 코드와 동일한 방식으로 generator 호출
                # [lats, None] 형태로 직접 전달
                fake_refine, _ = self.generator(struct, [lats, None], randomize_noise=False)
                self.debug_logger.log(f"Refine generator output shape: {fake_refine.shape}")
                
            except Exception as e:
                self.debug_logger.log(f"Error in refine method: {e}", "ERROR")
                raise

            # Denormalization process remains the same.
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

        # Convert mask to single-channel [H,W] in 0..1
        if target_mask.ndim == 3:
            mask_gray = np.max(target_mask, axis=2)
        else:
            mask_gray = target_mask
        mask_gray = (mask_gray.astype(np.float32) / 255.0)

        # To tensor
        face_mask_tensor = torch.from_numpy(mask_gray).float().cuda()

        # Apply smooth mask
        soft_face_mask_tensor, _ = self.smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()

        soft_face_mask = soft_face_mask_tensor.cpu().numpy()[:, :, np.newaxis]
        result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        return result.astype(np.uint8)
