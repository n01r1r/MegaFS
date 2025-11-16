"""
MegaFS Local Runner
Execute face swapping locally with CUDA support.
This script mirrors the functionality of MegaFS.ipynb but for local execution.
"""

import os
import sys
import argparse
from pathlib import Path

# Default local paths (can be edited by user)
DEFAULT_DATASET_ROOT = "./dataset/CelebAMask-HQ"
DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_WEIGHTS_DIR = "./weights"
DEFAULT_DATA_MAP = "./data_map.json"

def setup_environment():
    """Setup and import basic libraries"""
    print("=" * 60)
    print("MegaFS Local Runner - Setting up environment")
    print("=" * 60)
    
    try:
        import torch
        import cv2
        import numpy as np
        import torchvision
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        print("[OK] All basic libraries imported")
        print(f"[OK] PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"[OK] CUDA available - Device: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA version: {torch.version.cuda}")
        else:
            print("[WARN] CUDA not available - will use CPU (slower)")
        
        return torch
        
    except ImportError as e:
        print(f"[FAIL] Failed to import libraries: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)

def import_megafs_components():
    """Import MegaFS components"""
    print("\n" + "=" * 60)
    print("Importing MegaFS components")
    print("=" * 60)
    
    try:
        from config import Config, DEFAULT_CONFIGS
        from models.megafs import MegaFS
        from models.weight_loaders import verify_all_weights
        from utils.debug_utils import check_system_requirements
        from utils.data_utils import DataMapManager
        
        print("[OK] All MegaFS components imported")
        return Config, DEFAULT_CONFIGS, MegaFS, verify_all_weights, check_system_requirements, DataMapManager
        
    except ImportError as e:
        print(f"[FAIL] Failed to import MegaFS components: {e}")
        print("Make sure you're running from the MegaFS directory")
        sys.exit(1)

def setup_paths(dataset_root=None, weights_dir=None, data_map_path=None):
    """Setup local paths"""
    print("\n" + "=" * 60)
    print("Setting up local paths")
    print("=" * 60)
    
    # Use defaults if not provided
    dataset_root = dataset_root or DEFAULT_DATASET_ROOT
    weights_dir = weights_dir or DEFAULT_WEIGHTS_DIR
    data_map_path = data_map_path or DEFAULT_DATA_MAP
    
    # Convert to absolute paths
    dataset_root = os.path.abspath(dataset_root)
    weights_dir = os.path.abspath(weights_dir)
    data_map_path = os.path.abspath(data_map_path)
    
    # Derived paths
    img_dir = os.path.join(dataset_root, "CelebA-HQ-img")
    mask_dir = os.path.join(dataset_root, "CelebAMask-HQ-mask-anno")
    
    print(f"Dataset root: {dataset_root}")
    print(f"Weights directory: {weights_dir}")
    print(f"Data map path: {data_map_path}")
    print(f"Image directory: {img_dir}")
    print(f"Mask directory: {mask_dir}")
    
    # Check if directories exist
    if not os.path.exists(dataset_root):
        print(f"[WARN] Warning: Dataset root not found: {dataset_root}")
        print("Please ensure the dataset is extracted to the correct location")
    
    if not os.path.exists(weights_dir):
        print(f"[WARN] Warning: Weights directory not found: {weights_dir}")
        print("Please ensure weight files are in the weights directory")
    
    if not os.path.exists(data_map_path):
        print(f"[WARN] Warning: Data map not found: {data_map_path}")
        print("The data_map.json will be generated if needed")
    
    return {
        'dataset_root': dataset_root,
        'weights_dir': weights_dir,
        'data_map_path': data_map_path,
        'img_dir': img_dir,
        'mask_dir': mask_dir
    }

def load_data_map(data_map_path, dataset_root):
    """Load and verify data map"""
    print("\n" + "=" * 60)
    print("Loading data map")
    print("=" * 60)
    
    if not os.path.exists(data_map_path):
        print(f"[FAIL] Data map not found at: {data_map_path}")
        print("Creating new data map...")
        from create_datamap import build_data_map
        import json
        
        img_dir = os.path.join(dataset_root, "CelebA-HQ-img")
        mask_dir = os.path.join(dataset_root, "CelebAMask-HQ-mask-anno")
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            data_map = build_data_map(dataset_root, img_dir, mask_dir)
            
            # Save data map
            with open(data_map_path, 'w') as f:
                json.dump(data_map, f, indent=4)
            
            print(f"[OK] Data map created and saved to: {data_map_path}")
            print(f"[OK] Found {len(data_map)} entries")
        else:
            print("[FAIL] Cannot create data map - required directories not found")
            return None
    else:
        print(f"[OK] Data map found at: {data_map_path}")
    
    # Load data map using DataMapManager
    DataMapManager_module = import_megafs_components()[5]  # DataMapManager
    data_manager = DataMapManager_module(data_map_path)
    
    valid_ids = data_manager.get_valid_ids(dataset_root, sample_size=None)
    print(f"[OK] Found {len(valid_ids)} valid entries")
    
    # Verify sample
    if valid_ids:
        stats = data_manager.verify_sample(sample_size=10, dataset_root=dataset_root)
        print(f"[OK] Sample verification - {stats}")
    
    return data_manager, valid_ids

def setup_configuration(paths, swap_type='ftm', debug=True):
    """Setup configuration"""
    print("\n" + "=" * 60)
    print("Setting up configuration")
    print("=" * 60)
    
    Config_module = import_megafs_components()[0]  # Config
    
    config = Config_module(
        swap_type=swap_type,
        dataset_root=paths['dataset_root'],
        img_root=paths['img_dir'],
        mask_root=paths['mask_dir'],
        checkpoint_dir=paths['weights_dir']
    )
    
    config.print_config()
    return config

def verify_weights(weights_dir):
    """Verify weight files exist"""
    print("\n" + "=" * 60)
    print("Verifying weight files")
    print("=" * 60)
    
    verify_all_weights_module = import_megafs_components()[3]  # verify_all_weights
    
    if verify_all_weights_module(weights_dir):
        print("[OK] All weight files verified")
        return True
    else:
        print("[FAIL] Some weight files missing")
        print("Required files:")
        print("  - ftm_final.pth")
        print("  - injection_final.pth")
        print("  - lcr_final.pth")
        print("  - stylegan2-ffhq-config-f.pth")
        return False

def initialize_model(config, data_map, swap_type='ftm', enable_grads=False):
    """Initialize MegaFS model"""
    print("\n" + "=" * 60)
    print(f"Initializing {swap_type.upper()}-MegaFS model")
    print("=" * 60)
    
    MegaFS_module = import_megafs_components()[2]  # MegaFS
    
    try:
        # Update swap type in config
        config.swap.swap_type = swap_type
        
        # Determine device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        handler = MegaFS_module(
            config=config,
            data_map=data_map,
            debug=True,
            enable_grads=enable_grads,
            device=device
        )
        
        print(f"[OK] {swap_type.upper()}-MegaFS model initialized")
        print(f"[OK] Device: {device}")
        if enable_grads:
            print("[OK] Gradient computation enabled")
        
        return handler
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_single_swap(handler, src_id, tgt_id, refine=True, save_path=None, 
                    src_adv_path=None, tgt_adv_path=None):
    """Run face swap for a single image pair.
    
    Args:
        handler: MegaFS model instance
        src_id: Source image ID
        tgt_id: Target image ID
        refine: Whether to apply refinement
        save_path: Optional path to save result
        src_adv_path: Optional path to adversarial source image (if None, uses original)
        tgt_adv_path: Optional path to adversarial target image (if None, uses original)
    
    Returns:
        Tuple of (result_path, result_image) or (None, None) on failure
    """
    if not handler:
        print("[FAIL] Error: Handler not initialized")
        return None, None
    
    print("\n" + "=" * 60)
    print(f"Running face swap - Source ID: {src_id}, Target ID: {tgt_id}")
    if src_adv_path:
        print(f"  Using adversarial source: {src_adv_path}")
    if tgt_adv_path:
        print(f"  Using adversarial target: {tgt_adv_path}")
    print("=" * 60)
    
    try:
        # Store original read_pair method
        original_read_pair = handler.read_pair
        
        # Create patched read_pair if adversarial paths are provided
        if src_adv_path or tgt_adv_path:
            from utils.image_utils import ImageProcessor
            from models.megafs import encode_segmentation_rgb
            
            def patched_read_pair(src_idx, tgt_idx):
                use_adv_src = (src_idx == src_id and src_adv_path and os.path.exists(src_adv_path))
                use_adv_tgt = (tgt_idx == tgt_id and tgt_adv_path and os.path.exists(tgt_adv_path))
                
                if not use_adv_src and not use_adv_tgt:
                    return original_read_pair(src_idx, tgt_idx)
                
                # Load images
                if use_adv_src and use_adv_tgt:
                    src_img = ImageProcessor.load_image(src_adv_path, target_size=None)
                    tgt_img = ImageProcessor.load_image(tgt_adv_path, target_size=None)
                    # Load mask for target
                    _, tgt_mask_path = handler.data_manager.resolve_paths_for_id(
                        tgt_id, handler.config.paths.dataset_root
                    )
                    tgt_mask = None
                    if tgt_mask_path and os.path.exists(tgt_mask_path):
                        tgt_mask_raw = ImageProcessor.load_image(tgt_mask_path, target_size=None)
                        if tgt_mask_raw is not None:
                            tgt_mask = encode_segmentation_rgb(tgt_mask_raw)
                elif use_adv_src:
                    src_img = ImageProcessor.load_image(src_adv_path, target_size=None)
                    _, tgt_img, tgt_mask = original_read_pair(src_idx, tgt_idx)
                elif use_adv_tgt:
                    src_img, _, _ = original_read_pair(src_idx, tgt_idx)
                    tgt_img = ImageProcessor.load_image(tgt_adv_path, target_size=None)
                    # Load mask for target
                    _, tgt_mask_path = handler.data_manager.resolve_paths_for_id(
                        tgt_id, handler.config.paths.dataset_root
                    )
                    tgt_mask = None
                    if tgt_mask_path and os.path.exists(tgt_mask_path):
                        tgt_mask_raw = ImageProcessor.load_image(tgt_mask_path, target_size=None)
                        if tgt_mask_raw is not None:
                            tgt_mask = encode_segmentation_rgb(tgt_mask_raw)
                else:
                    return original_read_pair(src_idx, tgt_idx)
                
                return src_img, tgt_img, tgt_mask
            
            # Apply patch temporarily
            handler.read_pair = patched_read_pair
        
        # Ensure handler is in eval mode for inference
        was_training = handler.training
        if was_training:
            handler.eval()
            if hasattr(handler, 'generator'):
                handler.generator.eval()
        
        try:
            import torch
            with torch.no_grad():
                result_path, result_image = handler.run(
                    src_idx=src_id,
                    tgt_idx=tgt_id,
                    refine=refine,
                    save_path=save_path
                )
        finally:
            # Restore original read_pair
            if src_adv_path or tgt_adv_path:
                handler.read_pair = original_read_pair
            # Restore training mode if needed
            if was_training:
                handler.train()
                if hasattr(handler, 'generator'):
                    handler.generator.train()
        
        if result_path:
            print(f"[OK] Result saved to: {result_path}")
        else:
            print("[WARN] Face swap completed but not saved to file")
            
        return result_path, result_image
        
    except Exception as e:
        print(f"[FAIL] Face swap failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='MegaFS Local Runner')
    parser.add_argument('--dataset-root', type=str, default=DEFAULT_DATASET_ROOT,
                        help=f'Path to dataset root (default: {DEFAULT_DATASET_ROOT})')
    parser.add_argument('--weights-dir', type=str, default=DEFAULT_WEIGHTS_DIR,
                        help=f'Path to weights directory (default: {DEFAULT_WEIGHTS_DIR})')
    parser.add_argument('--data-map', type=str, default=DEFAULT_DATA_MAP,
                        help=f'Path to data map JSON (default: {DEFAULT_DATA_MAP})')
    parser.add_argument('--swap-type', type=str, default='ftm',
                        choices=['ftm', 'injection', 'lcr'],
                        help='Swap method type (default: ftm)')
    parser.add_argument('--src-id', type=int, default=2332,
                        help='Source image ID (default: 2332)')
    parser.add_argument('--tgt-id', type=int, default=2107,
                        help='Target image ID (default: 2107)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable refinement step')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for results')
    parser.add_argument('--enable-grads', action='store_true',
                        help='Enable gradient computation for experiments')
    parser.add_argument('--src-adv-path', type=str, default=None,
                        help='Path to adversarial source image (if None, uses original)')
    parser.add_argument('--tgt-adv-path', type=str, default=None,
                        help='Path to adversarial target image (if None, uses original)')
    
    args = parser.parse_args()
    
    # Setup environment
    torch = setup_environment()
    
    # Import components
    Config, DEFAULT_CONFIGS, MegaFS, verify_all_weights, check_system_requirements, DataMapManager = import_megafs_components()
    
    # Setup paths
    paths = setup_paths(
        dataset_root=args.dataset_root,
        weights_dir=args.weights_dir,
        data_map_path=args.data_map
    )
    
    # Check system requirements
    print("\n" + "=" * 60)
    print("Checking system requirements")
    print("=" * 60)
    check_system_requirements()
    
    # Load data map
    data_manager, valid_ids = load_data_map(paths['data_map_path'], paths['dataset_root'])
    
    if data_manager is None or not valid_ids:
        print("[FAIL] Failed to load data map")
        sys.exit(1)
    
    # Setup configuration
    config = setup_configuration(paths, swap_type=args.swap_type)
    
    # Verify weights
    if not verify_weights(paths['weights_dir']):
        print("[WARN] Warning: Some weight files are missing")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Initialize model
    data_map = data_manager.data_map
    handler = initialize_model(config, data_map, 
                              swap_type=args.swap_type,
                              enable_grads=args.enable_grads)
    
    if handler is None:
        print("[FAIL] Failed to initialize model")
        sys.exit(1)
    
    # Validate IDs
    src_id, tgt_id = args.src_id, args.tgt_id
    
    if src_id not in valid_ids:
        print(f"[WARN] Warning: Source ID {src_id} not in valid set")
        print(f"Available IDs: {len(valid_ids)}")
        if len(valid_ids) >= 1:
            src_id = valid_ids[0]
            print(f"Using first valid ID: {src_id}")
    
    if tgt_id not in valid_ids:
        print(f"[WARN] Warning: Target ID {tgt_id} not in valid set")
        if len(valid_ids) >= 2:
            tgt_id = valid_ids[1]
            print(f"Using second valid ID: {tgt_id}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save path with method name
    save_path = os.path.join(args.output_dir, f'swap_{src_id}_to_{tgt_id}_{args.swap_type}.jpg')
    
    # Run face swap
    print("\n" + "=" * 60)
    print("Starting face swap")
    print("=" * 60)
    
    result_path, result_image = run_single_swap(
        handler, 
        src_id, 
        tgt_id,
        refine=not args.no_refine,
        save_path=save_path,
        src_adv_path=args.src_adv_path,
        tgt_adv_path=args.tgt_adv_path
    )
    
    if result_path:
        print(f"\n[OK] SUCCESS: Face swap completed")
        print(f"[OK] Result saved to: {result_path}")
        print("\nTo view the result, open the saved image file.")
    else:
        print("\n[FAIL] Face swap failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

