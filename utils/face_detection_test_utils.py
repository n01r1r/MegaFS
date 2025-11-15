"""
Visualization utilities for face detection testing
Provides functions to visualize bboxes, masks, and compare detection modes
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import torch


def draw_bbox_on_image(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Input image as numpy array [H, W, 3] in RGB format
        bbox: Bounding box as (x, y, w, h) or None
        color: RGB color tuple for bbox (default: green)
        thickness: Line thickness
        label: Optional text label to display above bbox
        
    Returns:
        Image with bbox drawn (RGB format)
    """
    result = image.copy()
    
    if bbox is None:
        return result
    
    x, y, w, h = bbox
    
    # Draw rectangle (OpenCV uses BGR, so convert color)
    bgr_color = (color[2], color[1], color[0])
    cv2.rectangle(result, (x, y), (x + w, y + h), bgr_color, thickness)
    
    # Draw label if provided
    if label:
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(
            result,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            bgr_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            label,
            (x, y - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            text_thickness,
            cv2.LINE_AA
        )
    
    return result


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay mask on image with transparency.
    
    Args:
        image: Input image as numpy array [H, W, 3] in RGB format, range [0, 255]
        mask: Mask as numpy array [H, W] or [H, W, 3] in range [0, 1]
        color: RGB color tuple for mask overlay (default: red)
        alpha: Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        
    Returns:
        Image with mask overlay (RGB format)
    """
    # Ensure mask is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel
    
    # Normalize mask to [0, 1] if needed
    if mask.max() > 1.0:
        mask = mask.astype(np.float32) / 255.0
    
    # Create colored mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    colored_mask[:, :, 0] = color[0] * mask
    colored_mask[:, :, 1] = color[1] * mask
    colored_mask[:, :, 2] = color[2] * mask
    
    # Blend with original image
    result = image.astype(np.float32) * (1.0 - alpha * mask[:, :, np.newaxis]) + \
             colored_mask * (alpha * mask[:, :, np.newaxis])
    
    return result.astype(np.uint8)


def create_mode_comparison_grid(
    image: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    modes: List[str],
    show_bbox: bool = True,
    show_mask: bool = True
) -> np.ndarray:
    """
    Create side-by-side comparison grid showing all detection modes.
    
    Args:
        image: Original image [H, W, 3] in RGB format
        results: Dictionary mapping mode names to result dicts with keys:
                 'bbox', 'mask', 'success', 'label'
        modes: List of mode names to display
        show_bbox: Whether to draw bbox on images
        show_mask: Whether to overlay mask on images
        
    Returns:
        Comparison grid image (RGB format)
    """
    h, w = image.shape[:2]
    n_modes = len(modes)
    
    # Create grid: one row with original + one per mode
    grid_width = w * (n_modes + 1)
    grid_height = h
    
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Draw original image
    grid[:, :w] = image
    
    # Draw each mode result
    for idx, mode in enumerate(modes):
        if mode not in results:
            continue
        
        result = results[mode]
        x_offset = w * (idx + 1)
        
        # Start with original image
        mode_image = image.copy()
        
        # Overlay mask if available and requested
        if show_mask and 'mask' in result and result['mask'] is not None:
            mask = result['mask']
            if isinstance(mask, torch.Tensor):
                mask_np = mask[0].permute(1, 2, 0).cpu().numpy()
            else:
                mask_np = mask
            mode_image = overlay_mask_on_image(mode_image, mask_np, color=(255, 0, 0), alpha=0.4)
        
        # Draw bbox if available and requested
        if show_bbox and 'bbox' in result and result['bbox'] is not None:
            label = result.get('label', mode)
            mode_image = draw_bbox_on_image(mode_image, result['bbox'], color=(0, 255, 0), label=label)
        
        # Add success/failure indicator
        success = result.get('success', False)
        status_color = (0, 255, 0) if success else (255, 0, 0)
        status_text = "OK" if success else "FAIL"
        cv2.putText(
            mode_image,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (status_color[2], status_color[1], status_color[0]),  # BGR
            2,
            cv2.LINE_AA
        )
        
        # Place in grid
        grid[:, x_offset:x_offset + w] = mode_image
    
    return grid


def save_detection_visualization(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    mask: Optional[np.ndarray],
    save_path: str,
    mode_name: str = "detection"
) -> None:
    """
    Save annotated image with bbox and mask overlay.
    
    Args:
        image: Original image [H, W, 3] in RGB format
        bbox: Bounding box as (x, y, w, h) or None
        mask: Mask as numpy array or torch.Tensor [1, 3, H, W] or [H, W, 3]
        save_path: Path to save visualization
        mode_name: Name of detection mode for labeling
    """
    result = image.copy()
    
    # Overlay mask if provided
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            # Convert tensor to numpy
            if len(mask.shape) == 4:
                mask_np = mask[0].permute(1, 2, 0).cpu().numpy()
            else:
                mask_np = mask.permute(1, 2, 0).cpu().numpy()
        else:
            mask_np = mask
        
        result = overlay_mask_on_image(result, mask_np, color=(255, 0, 0), alpha=0.4)
    
    # Draw bbox if provided
    if bbox is not None:
        result = draw_bbox_on_image(result, bbox, color=(0, 255, 0), label=mode_name)
    
    # Save image (convert RGB to BGR for OpenCV)
    cv2.imwrite(save_path, result[:, :, ::-1])


def create_summary_visualization(
    image: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a comprehensive summary visualization showing all modes.
    
    Args:
        image: Original image [H, W, 3] in RGB format
        results: Dictionary mapping mode names to result dicts
        save_path: Optional path to save visualization
        
    Returns:
        Summary visualization image
    """
    modes = list(results.keys())
    grid = create_mode_comparison_grid(image, results, modes, show_bbox=True, show_mask=True)
    
    if save_path:
        cv2.imwrite(save_path, grid[:, :, ::-1])
    
    return grid

