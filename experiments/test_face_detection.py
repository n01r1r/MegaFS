"""
Standalone test runner for face detection modes
Tests face detection and mask generation on a single image using Haar Cascade detector
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from utils.image_utils import ImageProcessor
from utils.face_detectors import get_face_detector, validate_detection
from utils.attack_utils import generate_mask_from_detector, FaceDetectionError
from utils.face_detection_test_utils import (
    create_mode_comparison_grid,
    save_detection_visualization,
    create_summary_visualization
)


def get_image_path(image_id: int, img_root: str) -> str:
    """Get full path to image file."""
    non_padded = os.path.join(img_root, f"{image_id}.jpg")
    if os.path.exists(non_padded):
        return non_padded
    padded = os.path.join(img_root, f"{image_id:05d}.jpg")
    return padded if os.path.exists(padded) else non_padded


def test_detection_mode(
    image_np: np.ndarray,
    mode: str,
    device: str = 'cuda',
    checkpoint_dir: str = "weights",
    strict_detection: bool = False,
    detector_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Test a single detection mode on an image.
    
    Args:
        image_np: Input image as numpy array [H, W, 3] in RGB format
        mode: Detection mode name ('haar')
        device: Device for detectors ('cuda' or 'cpu')
        checkpoint_dir: Directory containing detector weights
        strict_detection: Whether to use strict detection mode
        detector_kwargs: Optional detector-specific parameters
        
    Returns:
        Dictionary with keys: 'success', 'bbox', 'mask', 'metrics', 'error'
    """
    result = {
        'mode': mode,
        'success': False,
        'bbox': None,
        'mask': None,
        'metrics': {},
        'error': None
    }
    
    try:
        # Create detector
        detector_kwargs = detector_kwargs or {}
        detector = get_face_detector(
            method=mode,
            device=device,
            checkpoint_dir=checkpoint_dir,
            **detector_kwargs
        )
        
        # Run detection
        bboxes = detector.detect(image_np)
        
        # Select largest face if multiple detected
        if len(bboxes) > 0:
            bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
            bbox = bboxes[0]
        else:
            bbox = None
        
        # Validate detection
        H, W = image_np.shape[:2]
        is_valid, reason, metrics = validate_detection(
            bbox,
            (H, W),
            min_bbox_area_ratio=0.01,
            max_bbox_area_ratio=0.95,
            min_bbox_size=20
        )
        
        result['bbox'] = bbox
        result['metrics'] = metrics
        
        # Generate mask
        try:
            M1, M2 = generate_mask_from_detector(
                image_np,
                detector,
                device=device,
                edge_blur_ks=0,
                strict_detection=strict_detection,
                min_bbox_area_ratio=0.01,
                max_bbox_area_ratio=0.95,
                min_bbox_size=20
            )
            result['mask'] = M1
            result['success'] = True
            
            # Add mask statistics
            mask_area_ratio = float(M1.sum().item()) / M1.numel()
            result['metrics']['mask_area_ratio'] = mask_area_ratio
            
        except FaceDetectionError as e:
            result['error'] = str(e)
            result['success'] = False
            # For non-strict mode, fallback ellipse may be used
            if not strict_detection:
                # Non-strict mode might use fallback ellipse
                pass
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test face detection modes on an image')
    parser.add_argument('--image-id', type=int, default=None,
                        help='Image ID from dataset')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Direct path to image file')
    parser.add_argument('--img-root', type=str, default='./dataset/CelebAMask-HQ/CelebA-HQ-img',
                        help='Root directory for images (if using image-id)')
    parser.add_argument('--modes', type=str, default='haar',
                        help='Comma-separated list of modes to test (default: haar)')
    parser.add_argument('--output-dir', type=str, default='experiments/results/face_detection_tests',
                        help='Directory to save visualizations and results')
    parser.add_argument('--strict', action='store_true',
                        help='Use strict detection mode (fail if detection invalid)')
    parser.add_argument('--save-viz', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for detectors')
    parser.add_argument('--checkpoint-dir', type=str, default='weights',
                        help='Directory containing detector weights')
    
    args = parser.parse_args()
    
    # Determine image path
    if args.image_path:
        image_path = args.image_path
    elif args.image_id:
        image_path = get_image_path(args.image_id, args.img_root)
    else:
        print("ERROR: Must provide either --image-id or --image-path")
        return 1
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return 1
    
    # Load image
    print(f"Loading image: {image_path}")
    image_np = ImageProcessor.load_image(image_path, target_size=None)
    if image_np is None:
        print(f"ERROR: Failed to load image")
        return 1
    
    print(f"Image shape: {image_np.shape}")
    
    # Parse modes
    modes = [m.strip() for m in args.modes.split(',')]
    valid_modes = ['haar']  # BlazeFace removed - only Haar supported
    for mode in modes:
        if mode not in valid_modes:
            print(f"ERROR: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}")
            return 1
    
    print(f"\nTesting modes: {', '.join(modes)}")
    print(f"Strict detection: {args.strict}")
    
    # Test each mode
    results = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode}")
        print(f"{'='*60}")
        
        # Get detector-specific kwargs
        detector_kwargs = {}
        if mode == 'haar':
            detector_kwargs['scale_factor'] = 1.1
            detector_kwargs['min_neighbors'] = 3
            detector_kwargs['min_size'] = 50
        
        result = test_detection_mode(
            image_np,
            mode,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            strict_detection=args.strict,
            detector_kwargs=detector_kwargs
        )
        
        results[mode] = result
        
        # Print result
        if result['success']:
            print(f"[SUCCESS] Detection successful")
            if result['bbox']:
                x, y, w, h = result['bbox']
                print(f"  Bbox: ({x}, {y}, {w}, {h})")
                if 'area_ratio' in result['metrics']:
                    print(f"  Area ratio: {result['metrics']['area_ratio']:.4f}")
                if 'mask_area_ratio' in result['metrics']:
                    print(f"  Mask area ratio: {result['metrics']['mask_area_ratio']:.4f}")
            else:
                print(f"  No bbox (using fallback ellipse)")
        else:
            print(f"[FAILED] Detection failed")
            if result['error']:
                print(f"  Error: {result['error']}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'Success':<10} {'Bbox':<30} {'Mask Area':<15}")
    print("-" * 75)
    for mode in modes:
        result = results[mode]
        success_str = "[OK]" if result['success'] else "[FAIL]"
        if result['bbox']:
            x, y, w, h = result['bbox']
            bbox_str = f"({x},{y},{w},{h})"
        else:
            bbox_str = "None (fallback)"
        mask_area = result['metrics'].get('mask_area_ratio', 0.0)
        print(f"{mode:<20} {success_str:<10} {bbox_str:<30} {mask_area:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    report = {
        'image_path': image_path,
        'image_shape': list(image_np.shape),
        'modes_tested': modes,
        'strict_detection': args.strict,
        'results': {}
    }
    
    for mode, result in results.items():
        report['results'][mode] = {
            'success': result['success'],
            'bbox': result['bbox'],
            'metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                       for k, v in result['metrics'].items()},
            'error': result['error']
        }
    
    report_path = output_dir / 'detection_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to: {report_path}")
    
    # Save visualizations
    if args.save_viz:
        # Create comparison grid
        comparison_grid = create_summary_visualization(
            image_np,
            results,
            save_path=str(output_dir / 'comparison_grid.jpg')
        )
        print(f"Saved comparison grid to: {output_dir / 'comparison_grid.jpg'}")
        
        # Save individual visualizations
        for mode in modes:
            result = results[mode]
            viz_path = output_dir / f'detection_{mode}.jpg'
            save_detection_visualization(
                image_np,
                result['bbox'],
                result['mask'],
                str(viz_path),
                mode_name=mode
            )
        print(f"Saved individual visualizations to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

