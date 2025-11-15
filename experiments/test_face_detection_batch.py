"""
Batch test script for face detection modes
Tests all 3 modes on multiple images from dataset and collects statistics
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experiments.test_face_detection import (
    get_image_path,
    test_detection_mode
)
from utils.image_utils import ImageProcessor


def get_available_image_ids(img_root: str, max_samples: int = None) -> List[int]:
    """
    Get list of available image IDs from dataset.
    
    Args:
        img_root: Root directory containing images
        max_samples: Maximum number of samples to return
        
    Returns:
        List of image IDs
    """
    if not os.path.exists(img_root):
        return []
    
    image_ids = []
    for filename in os.listdir(img_root):
        if filename.lower().endswith('.jpg'):
            # Try to extract ID from filename
            try:
                # Remove extension
                name = os.path.splitext(filename)[0]
                # Try as integer
                img_id = int(name)
                image_ids.append(img_id)
            except ValueError:
                continue
    
    image_ids.sort()
    
    if max_samples:
        image_ids = image_ids[:max_samples]
    
    return image_ids


def test_batch(
    image_ids: List[int],
    img_root: str,
    modes: List[str],
    device: str = 'cuda',
    checkpoint_dir: str = "weights",
    strict_detection: bool = False,
    output_dir: str = "experiments/results/face_detection_batch_tests"
) -> Dict[str, Any]:
    """
    Test all modes on a batch of images.
    
    Args:
        image_ids: List of image IDs to test
        img_root: Root directory for images
        modes: List of detection modes to test
        device: Device for detectors
        checkpoint_dir: Directory containing detector weights
        strict_detection: Whether to use strict detection mode
        output_dir: Directory to save results
        
    Returns:
        Dictionary with aggregate statistics
    """
    all_results = []
    mode_stats = {mode: {'success': 0, 'failure': 0, 'bbox_sizes': [], 'mask_areas': []} for mode in modes}
    
    print(f"Testing {len(image_ids)} images with {len(modes)} modes...")
    print(f"{'='*60}")
    
    for idx, image_id in enumerate(image_ids):
        print(f"\n[{idx+1}/{len(image_ids)}] Testing image ID: {image_id}")
        
        # Get image path
        image_path = get_image_path(image_id, img_root)
        if not os.path.exists(image_path):
            print(f"  ✗ Image not found: {image_path}")
            continue
        
        # Load image
        image_np = ImageProcessor.load_image(image_path, target_size=None)
        if image_np is None:
            print(f"  ✗ Failed to load image")
            continue
        
        # Test each mode
        image_results = {'image_id': image_id, 'image_path': image_path, 'modes': {}}
        
        for mode in modes:
            # Get detector-specific kwargs
            detector_kwargs = {}
            if mode == 'haar':
                detector_kwargs['scale_factor'] = 1.1
                detector_kwargs['min_neighbors'] = 3
                detector_kwargs['min_size'] = 50
            
            result = test_detection_mode(
                image_np,
                mode,
                device=device,
                checkpoint_dir=checkpoint_dir,
                strict_detection=strict_detection,
                detector_kwargs=detector_kwargs
            )
            
            image_results['modes'][mode] = result
            
            # Update statistics
            if result['success']:
                mode_stats[mode]['success'] += 1
                if result['bbox']:
                    x, y, w, h = result['bbox']
                    bbox_area = w * h
                    mode_stats[mode]['bbox_sizes'].append(bbox_area)
                if 'mask_area_ratio' in result['metrics']:
                    mode_stats[mode]['mask_areas'].append(result['metrics']['mask_area_ratio'])
            else:
                mode_stats[mode]['failure'] += 1
        
        all_results.append(image_results)
    
    # Calculate aggregate statistics
    summary = {
        'total_images': len(image_ids),
        'modes_tested': modes,
        'strict_detection': strict_detection,
        'mode_statistics': {}
    }
    
    for mode in modes:
        stats = mode_stats[mode]
        total = stats['success'] + stats['failure']
        success_rate = stats['success'] / total if total > 0 else 0.0
        
        avg_bbox_size = np.mean(stats['bbox_sizes']) if stats['bbox_sizes'] else 0.0
        avg_mask_area = np.mean(stats['mask_areas']) if stats['mask_areas'] else 0.0
        
        summary['mode_statistics'][mode] = {
            'success_count': stats['success'],
            'failure_count': stats['failure'],
            'success_rate': float(success_rate),
            'average_bbox_size': float(avg_bbox_size),
            'average_mask_area_ratio': float(avg_mask_area)
        }
    
    return {
        'summary': summary,
        'detailed_results': all_results
    }


def main():
    parser = argparse.ArgumentParser(description='Batch test face detection modes on multiple images')
    parser.add_argument('--img-root', type=str, default='./dataset/CelebAMask-HQ/CelebA-HQ-img',
                        help='Root directory for images')
    parser.add_argument('--image-ids', type=int, nargs='*', default=None,
                        help='List of image IDs to test (if not provided, will sample from dataset)')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='Maximum number of images to test (if image-ids not provided)')
    parser.add_argument('--modes', type=str, default='haar',
                        help='Comma-separated list of modes to test (default: haar)')
    parser.add_argument('--output-dir', type=str, default='experiments/results/face_detection_batch_tests',
                        help='Directory to save results')
    parser.add_argument('--strict', action='store_true',
                        help='Use strict detection mode')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for detectors')
    parser.add_argument('--checkpoint-dir', type=str, default='weights',
                        help='Directory containing detector weights')
    
    args = parser.parse_args()
    
    # Determine image IDs
    if args.image_ids:
        image_ids = args.image_ids
    else:
        print(f"Finding available images in: {args.img_root}")
        image_ids = get_available_image_ids(args.img_root, max_samples=args.max_samples)
        if not image_ids:
            print(f"ERROR: No images found in {args.img_root}")
            return 1
        print(f"Found {len(image_ids)} images")
    
    # Parse modes
    modes = [m.strip() for m in args.modes.split(',')]
    valid_modes = ['haar']  # BlazeFace removed - only Haar supported
    for mode in modes:
        if mode not in valid_modes:
            print(f"ERROR: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}")
            return 1
    
    # Run batch test
    results = test_batch(
        image_ids,
        args.img_root,
        modes,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        strict_detection=args.strict,
        output_dir=args.output_dir
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    summary_path = output_dir / 'batch_test_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    
    # Save detailed results (remove tensors for JSON serialization)
    detailed_results_serializable = []
    for img_result in results['detailed_results']:
        serializable_result = {
            'image_id': img_result['image_id'],
            'image_path': img_result['image_path'],
            'modes': {}
        }
        for mode, mode_result in img_result['modes'].items():
            serializable_result['modes'][mode] = {
                'success': mode_result['success'],
                'bbox': mode_result.get('bbox'),
                'metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer, np.number)) else 
                                (float(v) if hasattr(v, 'item') else v))
                           for k, v in mode_result.get('metrics', {}).items()},
                'error': mode_result.get('error')
            }
        detailed_results_serializable.append(serializable_result)
    
    detailed_path = output_dir / 'batch_test_results.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results_serializable, f, indent=2)
    print(f"Saved detailed results to: {detailed_path}")
    
    # Save CSV for easy analysis
    csv_path = output_dir / 'batch_test_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['image_id', 'mode', 'success', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 
                        'bbox_area', 'mask_area_ratio', 'error'])
        
        # Data rows
        for img_result in results['detailed_results']:
            image_id = img_result['image_id']
            for mode, mode_result in img_result['modes'].items():
                bbox = mode_result.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    bbox_area = w * h
                else:
                    x, y, w, h, bbox_area = None, None, None, None, None
                
                mask_area = mode_result['metrics'].get('mask_area_ratio', None)
                error = mode_result.get('error', '')
                
                writer.writerow([
                    image_id, mode, mode_result['success'],
                    x, y, w, h, bbox_area,
                    mask_area, error
                ])
    
    print(f"Saved CSV to: {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Batch Test Summary")
    print(f"{'='*60}")
    print(f"Total images tested: {results['summary']['total_images']}")
    print(f"\nMode Statistics:")
    print(f"{'Mode':<20} {'Success':<10} {'Failure':<10} {'Success Rate':<15} {'Avg Bbox Size':<15} {'Avg Mask Area':<15}")
    print("-" * 85)
    
    for mode in modes:
        stats = results['summary']['mode_statistics'][mode]
        print(f"{mode:<20} {stats['success_count']:<10} {stats['failure_count']:<10} "
              f"{stats['success_rate']:.2%:<15} {stats['average_bbox_size']:.1f:<15} "
              f"{stats['average_mask_area_ratio']:.4f:<15}")
    
    # Identify best mode
    best_mode = max(modes, key=lambda m: results['summary']['mode_statistics'][m]['success_rate'])
    print(f"\nBest mode (by success rate): {best_mode}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

