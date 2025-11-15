"""
Adversarial attack runner for HieRFE dual-target strategy
Execute attacks on CelebA-HQ images with configurable parameters
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
import random
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import Config
from models.megafs import MegaFS
from utils.attack_utils import DualTargetPGDAttack, compute_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load attack configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict[str, Any]) -> MegaFS:
    """Setup MegaFS model with gradients enabled."""
    # Create config object
    cfg = Config(
        swap_type=config['model']['swap_type'],
        dataset_root=config['paths']['dataset_root'],
        img_root=config['paths']['img_root'],
        mask_root=config['paths']['mask_root'],
        checkpoint_dir=config['paths']['checkpoint_dir']
    )
    
    # Initialize model with gradients enabled
    model = MegaFS(
        config=cfg,
        debug=True,
        enable_grads=config['model']['enable_grads'],
        device=config['device']
    )
    
    return model


def get_image_path(image_id: int, img_root: str) -> str:
    """Get full path to image file."""
    # Prefer non-padded filename (e.g., 2332.jpg). Fallback to 5-digit padded if needed.
    non_padded = os.path.join(img_root, f"{image_id}.jpg")
    if os.path.exists(non_padded):
        return non_padded
    padded = os.path.join(img_root, f"{image_id:05d}.jpg")
    return padded if os.path.exists(padded) else non_padded


def make_exp_output_dir(base_dir: str, cfg: Dict[str, Any], image_id: int) -> str:
    """Create and return a unique output directory for given config and image id."""
    exp_name = (
        f"exp_l1_{cfg['attack']['lambda_1']}_"
        f"l2_{cfg['attack']['lambda_2']}_"
        f"e_{cfg['attack']['epsilon']}"
    )
    odir = os.path.join(base_dir, exp_name, str(image_id))
    os.makedirs(odir, exist_ok=True)
    return odir

def make_exp_dir(base_dir: str, cfg: Dict[str, Any]) -> str:
    """Create and return experiment directory without image subfolder."""
    exp_name = (
        f"exp_l1_{cfg['attack']['lambda_1']}_"
        f"l2_{cfg['attack']['lambda_2']}_"
        f"e_{cfg['attack']['epsilon']}"
    )
    edir = os.path.join(base_dir, exp_name)
    os.makedirs(edir, exist_ok=True)
    return edir


def _sweep_job(job_args: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level function for ProcessPoolExecutor sweep execution."""
    try:
        # Reconstruct config
        cfg = json.loads(job_args['config_json'])
        # Apply job-specific overrides
        cfg['attack']['lambda_1'] = float(job_args['lambda_1'])
        cfg['attack']['lambda_2'] = float(job_args['lambda_2'])
        cfg['attack']['epsilon'] = float(job_args['epsilon'])
        cfg['attack']['num_iter'] = int(job_args['num_iter'])

        # Setup model in this process
        model_local = setup_model(cfg)
        out_dir_local = make_exp_output_dir(job_args['base_output_dir'], cfg, job_args['image_id'])

        res = run_single_attack(cfg, model_local, job_args['image_id'], out_dir_local)
        if res is None:
            return {
                'image_id': job_args['image_id'],
                'success': False,
                'lambda_1': job_args['lambda_1'],
                'lambda_2': job_args['lambda_2'],
                'epsilon': job_args['epsilon'],
                'num_iter': job_args['num_iter']
            }
        res.update({
            'lambda_1': job_args['lambda_1'],
            'lambda_2': job_args['lambda_2'],
            'epsilon': job_args['epsilon'],
            'num_iter': job_args['num_iter']
        })
        return res
    except Exception as e:
        return {
            'image_id': job_args.get('image_id'),
            'success': False,
            'error': str(e)
        }


def _sweep_job_pair(job_tuple: tuple) -> Dict[str, Any]:
    """Top-level function for ProcessPoolExecutor pair sweep execution (pickle-safe)."""
    try:
        src_id, tgt_id, l1, l2, eps, nit, config_json, base_output_dir = job_tuple
        # Reconstruct config
        cfg = json.loads(config_json)
        cfg['attack']['lambda_1'] = float(l1)
        cfg['attack']['lambda_2'] = float(l2)
        cfg['attack']['epsilon'] = float(eps)
        cfg['attack']['num_iter'] = int(nit)
        # force full-compare behavior
        cfg.setdefault('experiment', {})
        cfg['experiment']['attack_source'] = True
        cfg['experiment']['full_compare'] = True
        # setup model and exp dir
        model_local = setup_model(cfg)
        exp_dir = make_exp_dir(base_output_dir, cfg)
        return run_pair_attack(cfg, model_local, src_id, tgt_id, exp_dir)
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_single_attack(
    config: Dict[str, Any],
    model: MegaFS,
    image_id: int,
    output_dir: str
) -> Dict[str, Any]:
    """Run attack on a single image."""
    print(f"\n{'='*60}")
    print(f"Attacking image ID: {image_id}")
    print(f"{'='*60}")
    
    # Get image path
    img_root = config['paths']['img_root']
    image_path = get_image_path(image_id, img_root)
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Extract mask generation config
    mask_gen_config = config.get('mask_generation', {})
    detector_method = mask_gen_config.get('method', 'haar')
    strict_detection = mask_gen_config.get('strict_detection', True)
    fallback_method = mask_gen_config.get('fallback_method', None)
    validation_config = mask_gen_config.get('validation', {})
    
    # Build detector kwargs based on method
    detector_kwargs = {}
    if detector_method == 'haar':
        haar_config = mask_gen_config.get('haar', {})
        detector_kwargs['scale_factor'] = haar_config.get('scale_factor', 1.1)
        detector_kwargs['min_neighbors'] = haar_config.get('min_neighbors', 3)
        detector_kwargs['min_size'] = haar_config.get('min_size', 50)
    
    # Create attack instance
    attack = DualTargetPGDAttack(
        identity_extractor=model.encoder,
        epsilon=config['attack']['epsilon'],
        alpha=config['attack']['alpha'],
        num_iter=config['attack']['num_iter'],
        lambda_1=config['attack']['lambda_1'],
        lambda_2=config['attack']['lambda_2'],
        device=config['device'],
        verbose=config['experiment']['verbose'],
        sem_variant=config.get('attack', {}).get('sem_variant', 'mse_f4'),
        preproc=config.get('preprocessing', {}).get('mode', 'none'),
        mask_blur_ks=config.get('mask_generation', {}).get('edge_blur', 0),
        loss_schedule=config.get('attack', {}).get('loss_schedule', False),
        clip_grad=config.get('attack', {}).get('clip_grad', 0.0),
        checkpoint_dir=config['paths']['checkpoint_dir'],
        detector_method=detector_method,
        strict_detection=strict_detection,
        fallback_detector_method=fallback_method,
        min_bbox_area_ratio=validation_config.get('min_bbox_area_ratio', 0.01),
        max_bbox_area_ratio=validation_config.get('max_bbox_area_ratio', 0.95),
        min_bbox_size=validation_config.get('min_bbox_size', 20),
        detector_kwargs=detector_kwargs
    )
    
    # Execute attack
    # Flattened: write directly under experiment/image_id folder
    output_path = output_dir
    os.makedirs(output_path, exist_ok=True)
    
    try:
        adversarial = attack.attack(image_path, output_path)
        
        # Load original for metrics
        from utils.image_utils import ImageProcessor
        original = ImageProcessor.load_image(image_path, target_size=(256, 256))
        
        # Compute metrics
        metrics = compute_metrics(original, adversarial)
        print(f"\nAttack completed!")
        print(f"L2 norm: {metrics.get('L2_norm', 'N/A'):.2f}")
        print(f"L-inf norm: {metrics.get('Linf_norm', 'N/A'):.2f}")
        if 'SSIM' in metrics:
            print(f"SSIM: {metrics['SSIM']:.4f}")
        
        # Save metrics (ensure JSON-serializable types)
        serializable_metrics = {k: (float(v) if hasattr(v, 'item') else float(v) if isinstance(v, (np.floating,)) else v)
                                for k, v in metrics.items()}
        with open(os.path.join(output_path, 'metrics_target.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Helper to ensure model tensors
        def ensure_model_tensor(img_np, device):
            if isinstance(img_np, tuple):
                img_np = img_np[0]
            t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            return ImageProcessor.preprocess_for_model_tensor(t)
        
        # After attack: perform swap comparisons
        try:
            img_root = config['paths']['img_root']
            candidates = [f for f in os.listdir(img_root) if f.lower().endswith('.jpg')]
            current_name = os.path.basename(image_path)
            candidates = [f for f in candidates if f != current_name]
            if candidates:
                src_file = random.choice(candidates)
                src_path = os.path.join(img_root, src_file)
                src_img = ImageProcessor.load_image(src_path, target_size=(256, 256))
                # Prepare tensors
                src_clean_tensor = ensure_model_tensor(src_img, model.device)
                tgt_clean_tensor = ensure_model_tensor(original, model.device)
                tgt_adv_tensor = ensure_model_tensor(adversarial, model.device)
                # Optionally attack source too
                src_adv_img = src_img
                attack_source = bool(config.get('experiment', {}).get('attack_source', False))
                full_compare = bool(config.get('experiment', {}).get('full_compare', False))
                if attack_source:
                    src_adv_img = attack.attack(src_path, output_path, output_prefix='source')
                    # metrics for source
                    m_src = compute_metrics(src_img, src_adv_img)
                    with open(os.path.join(output_path, 'metrics_source.json'), 'w') as f:
                        json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_src.items()}, f, indent=2)
                src_adv_tensor = ensure_model_tensor(src_adv_img, model.device)
                # 4-way swaps
                swap_CC = model.swap(src_clean_tensor, tgt_clean_tensor, return_tensor=False)
                swap_CA = model.swap(src_clean_tensor, tgt_adv_tensor,   return_tensor=False)
                swap_AC = model.swap(src_adv_tensor,   tgt_clean_tensor, return_tensor=False)
                swap_AA = model.swap(src_adv_tensor,   tgt_adv_tensor,   return_tensor=False)
                import cv2
                # Save unified originals/adversarials
                cv2.imwrite(os.path.join(output_path, 'source_clean.jpg'),  src_img[:, :, ::-1])
                cv2.imwrite(os.path.join(output_path, 'target_clean.jpg'),  original[:, :, ::-1])
                cv2.imwrite(os.path.join(output_path, 'target_adv.jpg'),    adversarial[:, :, ::-1])
                if attack_source:
                    cv2.imwrite(os.path.join(output_path, 'source_adv.jpg'), src_adv_img[:, :, ::-1])
                if full_compare:
                    # Save with simple names
                    cv2.imwrite(os.path.join(output_path, 'swap_CC.jpg'), swap_CC[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_CA.jpg'), swap_CA[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_AC.jpg'), swap_AC[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_AA.jpg'), swap_AA[:, :, ::-1])
                    # Also save descriptive variants for clarity
                    cv2.imwrite(os.path.join(output_path, 'swap_CC_(Clean_S_Clean_T).jpg'), swap_CC[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_CA_(Clean_S_Adv_T).jpg'),   swap_CA[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_AC_(Adv_S_Clean_T).jpg'),   swap_AC[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_AA_(Adv_S_Adv_T).jpg'),     swap_AA[:, :, ::-1])
                    top = np.hstack([swap_CC, swap_CA])
                    bottom = np.hstack([swap_AC, swap_AA])
                    grid = np.vstack([top, bottom])
                    cv2.imwrite(os.path.join(output_path, 'comparison_grid_ALL.jpg'), grid[:, :, ::-1])
                else:
                    cv2.imwrite(os.path.join(output_path, 'swap_on_original.jpg'),    swap_CC[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_on_adversarial.jpg'), swap_CA[:, :, ::-1])
                    comp = np.hstack([src_img, original, adversarial, swap_CC, swap_CA])
                    cv2.imwrite(os.path.join(output_path, 'swap_comparison.jpg'), comp[:, :, ::-1])
                # manifest & effective config
                manifest = {
                    'image_id': image_id,
                    'params': {
                        'lambda_1': config['attack']['lambda_1'],
                        'lambda_2': config['attack']['lambda_2'],
                        'epsilon':   config['attack']['epsilon'],
                        'num_iter':  config['attack']['num_iter']
                    },
                    'artifacts': {
                        'source_clean': 'source_clean.jpg',
                        'target_clean': 'target_clean.jpg',
                        'source_adv':   'source_adv.jpg' if attack_source else None,
                        'target_adv':   'target_adv.jpg',
                        'perturbation_source': 'perturbation_source.jpg' if attack_source else None,
                        'perturbation_target': 'perturbation_target.jpg',
                        'masks': ['mask_face_target.jpg', 'mask_bg_target.jpg'],
                        'swaps': {
                            'CC': 'swap_CC.jpg',
                            'CA': 'swap_CA.jpg',
                            'AC': 'swap_AC.jpg',
                            'AA': 'swap_AA.jpg'
                        },
                        'grids': ['comparison_target.jpg', 'comparison_grid_ALL.jpg' if full_compare else 'swap_comparison.jpg'],
                        'metrics_target': 'metrics_target.json',
                        'metrics_source': 'metrics_source.json' if attack_source else None,
                        'loss_curves': ['loss_curves_target.jpg'] + (['loss_curves_source.jpg'] if attack_source else [])
                    }
                }
                with open(os.path.join(output_path, 'effective_config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                with open(os.path.join(output_path, 'manifest.json'), 'w') as f:
                    json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"Warning: swap post-check failed: {e}")
        
        return {
            'image_id': image_id,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR during attack: {e}")
        import traceback
        traceback.print_exc()
        return {
            'image_id': image_id,
            'success': False,
            'error': str(e)
        }


# === MODULE-SCOPE ensure_model_tensor ===
def ensure_model_tensor(img_input, device):
    """Robustly convert to normalized model tensor (NCHW, float, on device)"""
    from utils.image_utils import ImageProcessor
    img_np = img_input
    if isinstance(img_np, tuple):
        img_np = img_np[0]
    if isinstance(img_np, str):
        img_np = ImageProcessor.load_image(img_np, target_size=(256, 256))
    if isinstance(img_np, torch.Tensor):
        t = img_np
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.dtype != torch.float32:
            t = t.float()
        t = t.to(device)
        t = ImageProcessor.preprocess_for_model_tensor(t)
    else:
        t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        t = ImageProcessor.preprocess_for_model_tensor(t)
    assert isinstance(t, torch.Tensor), f"ensure_model_tensor failed: got type {type(t)}"
    return t


def run_pair_attack(
    config: Dict[str, Any],
    model: MegaFS,
    src_id: int,
    tgt_id: int,
    output_dir: str
) -> Dict[str, Any]:
    """Run full 4-way comparison for a specified (source, target) pair into a single folder."""
    print(f"\n{'='*60}")
    print(f"Pair attack - Source ID: {src_id}  Target ID: {tgt_id}")
    print(f"{'='*60}")

    # Resolve paths
    img_root = config['paths']['img_root']
    src_path = get_image_path(src_id, img_root)
    tgt_path = get_image_path(tgt_id, img_root)
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        print(f"ERROR: Missing images for pair: src={src_path}, tgt={tgt_path}")
        return {'success': False, 'error': 'missing_images'}

    # Extract mask generation config
    mask_gen_config = config.get('mask_generation', {})
    detector_method = mask_gen_config.get('method', 'haar')
    strict_detection = mask_gen_config.get('strict_detection', True)
    fallback_method = mask_gen_config.get('fallback_method', None)
    validation_config = mask_gen_config.get('validation', {})
    
    # Build detector kwargs based on method
    detector_kwargs = {}
    if detector_method == 'haar':
        haar_config = mask_gen_config.get('haar', {})
        detector_kwargs['scale_factor'] = haar_config.get('scale_factor', 1.1)
        detector_kwargs['min_neighbors'] = haar_config.get('min_neighbors', 3)
        detector_kwargs['min_size'] = haar_config.get('min_size', 50)
    
    # Attack instance
    attack = DualTargetPGDAttack(
        identity_extractor=model.encoder,
        epsilon=config['attack']['epsilon'],
        alpha=config['attack']['alpha'],
        num_iter=config['attack']['num_iter'],
        lambda_1=config['attack']['lambda_1'],
        lambda_2=config['attack']['lambda_2'],
        device=config['device'],
        verbose=config['experiment']['verbose'],
        sem_variant=config.get('attack', {}).get('sem_variant', 'mse_f4'),
        preproc=config.get('preprocessing', {}).get('mode', 'none'),
        mask_blur_ks=config.get('mask_generation', {}).get('edge_blur', 0),
        loss_schedule=config.get('attack', {}).get('loss_schedule', False),
        clip_grad=config.get('attack', {}).get('clip_grad', 0.0),
        checkpoint_dir=config['paths']['checkpoint_dir'],
        detector_method=detector_method,
        strict_detection=strict_detection,
        fallback_detector_method=fallback_method,
        min_bbox_area_ratio=validation_config.get('min_bbox_area_ratio', 0.01),
        max_bbox_area_ratio=validation_config.get('max_bbox_area_ratio', 0.95),
        min_bbox_size=validation_config.get('min_bbox_size', 20),
        detector_kwargs=detector_kwargs
    )

    # Load clean images
    from utils.image_utils import ImageProcessor
    src_clean = ImageProcessor.load_image(src_path, target_size=(256, 256))
    tgt_clean = ImageProcessor.load_image(tgt_path, target_size=(256, 256))

    # Attack target and source
    tgt_adv = attack.attack(tgt_path, output_dir, output_prefix='target')
    src_adv = attack.attack(src_path, output_dir, output_prefix='source')

    # Compute and save metrics
    m_tgt = compute_metrics(tgt_clean, tgt_adv)
    m_src = compute_metrics(src_clean, src_adv)
    with open(os.path.join(output_dir, 'metrics_target.json'), 'w') as f:
        json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_tgt.items()}, f, indent=2)
    with open(os.path.join(output_dir, 'metrics_source.json'), 'w') as f:
        json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_src.items()}, f, indent=2)

    # Prepare tensors
    src_clean_t = ensure_model_tensor(src_clean, model.device)
    tgt_clean_t = ensure_model_tensor(tgt_clean, model.device)
    src_adv_t   = ensure_model_tensor(src_adv,   model.device)
    tgt_adv_t   = ensure_model_tensor(tgt_adv,   model.device)

    # Optionally, add an explicit runtime assertion just before each swap:
    for name, t in [('src_clean', src_clean_t), ('tgt_clean', tgt_clean_t), ('src_adv', src_adv_t), ('tgt_adv', tgt_adv_t)]:
        print(f'[DEBUG] swap input {name}: type={type(t)}, shape={getattr(t, "shape", None)}')
        assert isinstance(t, torch.Tensor), f"Swap input {name} is not torch.Tensor but {type(t)}"

    # Swaps
    swap_CC = model.swap(src_clean_t, tgt_clean_t, return_tensor=False)
    swap_CA = model.swap(src_clean_t, tgt_adv_t,   return_tensor=False)
    swap_AC = model.swap(src_adv_t,   tgt_clean_t, return_tensor=False)
    swap_AA = model.swap(src_adv_t,   tgt_adv_t,   return_tensor=False)

    # Save raws
    import cv2
    cv2.imwrite(os.path.join(output_dir, 'source_clean.jpg'),  src_clean[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'target_clean.jpg'),  tgt_clean[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'source_adv.jpg'),    src_adv[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'target_adv.jpg'),    tgt_adv[:, :, ::-1])

    # Save swaps simple names
    cv2.imwrite(os.path.join(output_dir, 'swap_CC.jpg'), swap_CC[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'swap_CA.jpg'), swap_CA[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'swap_AC.jpg'), swap_AC[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'swap_AA.jpg'), swap_AA[:, :, ::-1])
    # Grid
    top = np.hstack([swap_CC, swap_CA])
    bottom = np.hstack([swap_AC, swap_AA])
    grid = np.vstack([top, bottom])
    cv2.imwrite(os.path.join(output_dir, 'comparison_grid_ALL.jpg'), grid[:, :, ::-1])

    # Save manifests
    with open(os.path.join(output_dir, 'effective_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    manifest = {
        'pair': {'source_id': src_id, 'target_id': tgt_id},
        'params': {
            'lambda_1': config['attack']['lambda_1'],
            'lambda_2': config['attack']['lambda_2'],
            'epsilon':   config['attack']['epsilon'],
            'num_iter':  config['attack']['num_iter']
        },
        'swaps': {'CC': 'swap_CC.jpg', 'CA': 'swap_CA.jpg', 'AC': 'swap_AC.jpg', 'AA': 'swap_AA.jpg'}
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    return {'success': True, 'image_id': f'{src_id}->{tgt_id}', 'metrics': {'target': m_tgt, 'source': m_src}}


def run_batch_attack(
    config: Dict[str, Any],
    model: MegaFS,
    image_ids: list,
    output_dir: str
) -> None:
    """Run attacks on multiple images."""
    results = []
    
    for image_id in image_ids:
        result = run_single_attack(config, model, image_id, output_dir)
        if result:
            results.append(result)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Batch attack summary")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Successful attacks: {sum(1 for r in results if r.get('success', False))}")
    
    if any('metrics' in r for r in results):
        avg_l2 = np.mean([r['metrics']['L2_norm'] for r in results if 'metrics' in r])
        print(f"Average L2 norm: {avg_l2:.2f}")


def main():
    parser = argparse.ArgumentParser(description='HieRFE Dual-Target Adversarial Attack')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml',
                        help='Path to attack configuration file')
    parser.add_argument('--image-id', type=int, default=2332,
                        help='Single image ID to attack')
    parser.add_argument('--batch', type=str, default=None,
                        help='Comma-separated list of image IDs (e.g., "2332,2107")')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    # Single-value overrides
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--lambda_1', type=float, default=None)
    parser.add_argument('--lambda_2', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=None)
    # Sweep mode (Python-only)
    parser.add_argument('--sweep', action='store_true', help='Enable sweep mode (runs multiple attacks)')
    parser.add_argument('--image-ids', type=int, nargs='*', default=None, help='List of image IDs for sweep')
    parser.add_argument('--num_iter_list', type=int, nargs='*', default=None)
    parser.add_argument('--lambda_1_list', type=float, nargs='*', default=None)
    parser.add_argument('--lambda_2_list', type=float, nargs='*', default=None)
    # Preprocessing and mask options
    parser.add_argument('--preproc', type=str, default=None, choices=['none','homo','homo_clahe'])
    parser.add_argument('--mask-edge-blur', type=int, default=None)
    # SEM variant and schedules
    parser.add_argument('--sem-variant', type=str, default=None,
                        choices=['mse_f4','l1_f4','self_collapse','self_collapse_mid','contrastive_bg'])
    parser.add_argument('--loss-schedule', action='store_true')
    parser.add_argument('--clip-grad', type=float, default=None)
    parser.add_argument('--epsilon_list', type=float, nargs='*', default=None)
    parser.add_argument('--parallel', type=int, default=1, help='Max concurrent workers for sweep')
    parser.add_argument('--attack-source', action='store_true', help='Also attack the source image')
    parser.add_argument('--full-compare', action='store_true', help='Save 4-way swap results (CC, CA, AC, AA)')
    # Pair mode controls
    parser.add_argument('--pair-mode', action='store_true', help='Run 4-way swap for provided pairs')
    parser.add_argument('--pairs', type=str, default=None, help='Comma-separated src:tgt pairs, e.g. "2332:428,2107:123"')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    output_dir = config['experiment']['output_dir']
    
    # Override max samples if specified
    if args.max_samples:
        config['experiment']['max_samples'] = args.max_samples
    # Persist flags into config for downstream use (including in workers)
    config.setdefault('experiment', {})
    full_compare_flag = bool(getattr(args, 'full_compare', False))
    attack_source_flag = bool(getattr(args, 'attack_source', False)) or full_compare_flag
    config['experiment']['attack_source'] = attack_source_flag
    config['experiment']['full_compare'] = full_compare_flag

    # Apply single-value overrides (if provided)
    if args.num_iter is not None:
        config['attack']['num_iter'] = args.num_iter
    if args.lambda_1 is not None:
        config['attack']['lambda_1'] = args.lambda_1
    if args.lambda_2 is not None:
        config['attack']['lambda_2'] = args.lambda_2
    if args.epsilon is not None:
        config['attack']['epsilon'] = args.epsilon
    # New options
    if args.mask_edge_blur is not None:
        config.setdefault('mask_generation', {})
        config['mask_generation']['edge_blur'] = int(args.mask_edge_blur)
    if args.preproc is not None:
        config['preprocessing'] = {'mode': args.preproc}
    if args.sem_variant is not None:
        config.setdefault('attack', {})
        config['attack']['sem_variant'] = args.sem_variant
    if args.loss_schedule:
        config.setdefault('attack', {})
        config['attack']['loss_schedule'] = True
    if args.clip_grad is not None:
        config.setdefault('attack', {})
        config['attack']['clip_grad'] = float(args.clip_grad)

    # Sweep mode
    if args.sweep:
        # Determine image ids
        if args.pairs and args.pair_mode:
            # Parse pairs like "2332:428,2107:123"
            def parse_pairs(pairs_str: str) -> List[tuple]:
                items = []
                for tok in pairs_str.split(','):
                    tok = tok.strip()
                    if not tok:
                        continue
                    s, t = tok.split(':')
                    items.append((int(s), int(t)))
                return items
            pairs = parse_pairs(args.pairs)
            # Build parameter lists
            l1_list = args.lambda_1_list if args.lambda_1_list else [config['attack']['lambda_1']]
            l2_list = args.lambda_2_list if args.lambda_2_list else [config['attack']['lambda_2']]
            e_list  = args.epsilon_list if args.epsilon_list else [config['attack']['epsilon']]
            it_list = args.num_iter_list if args.num_iter_list else [config['attack']['num_iter']]

            combos = []
            for (src_id, tgt_id) in pairs:
                for l1 in l1_list:
                    for l2 in l2_list:
                        for eps in e_list:
                            for nit in it_list:
                                combos.append((src_id, tgt_id, l1, l2, eps, nit))

            print(f"Scheduling {len(combos)} pair-sweep jobs (parallel={args.parallel})...")
            base_output_dir = config['experiment']['output_dir']
            config_json = json.dumps(config)
            results = []

            # Extend combos with config_json and base_output_dir for pickle-safe job runner
            extended_combos = [(src_id, tgt_id, l1, l2, eps, nit, config_json, base_output_dir) 
                               for (src_id, tgt_id, l1, l2, eps, nit) in combos]

            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as executor:
                futures = [executor.submit(_sweep_job_pair, jb) for jb in extended_combos]
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        results.append(res)
                    except Exception as e:
                        results.append({'success': False, 'error': str(e)})

            # Save summary
            os.makedirs(base_output_dir, exist_ok=True)
            with open(os.path.join(base_output_dir, 'sweep_summary.json'), 'w') as f:
                json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
            print("\nSweep completed.")
            return
        elif args.image_ids:
            image_ids: List[int] = args.image_ids
        elif args.batch:
            image_ids = [int(x.strip()) for x in args.batch.split(',')]
        else:
            image_ids = [args.image_id]

        # Build parameter lists (fallback to current config if list not provided)
        l1_list = args.lambda_1_list if args.lambda_1_list else [config['attack']['lambda_1']]
        l2_list = args.lambda_2_list if args.lambda_2_list else [config['attack']['lambda_2']]
        e_list = args.epsilon_list if args.epsilon_list else [config['attack']['epsilon']]
        it_list = args.num_iter_list if args.num_iter_list else [config['attack']['num_iter']]

        # Prepare jobs
        combos = list(itertools.product(image_ids, l1_list, l2_list, e_list, it_list))
        print(f"Scheduling {len(combos)} jobs (parallel={args.parallel})...")

        base_output_dir = output_dir
        config_json = json.dumps(config)
        results = []

        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            futures = []
            for (jid, l1, l2, eps, nit) in combos:
                job_args = {
                    'config_json': config_json,
                    'base_output_dir': base_output_dir,
                    'image_id': int(jid),
                    'lambda_1': float(l1),
                    'lambda_2': float(l2),
                    'epsilon': float(eps),
                    'num_iter': int(nit)
                }
                futures.append(executor.submit(_sweep_job, job_args))

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                    status = 'OK' if res and res.get('success') else 'FAIL'
                    print(f"Job {res.get('image_id')} -> {status}")
                except Exception as e:
                    print(f"Job error: {e}")
                    results.append({'success': False, 'error': str(e)})

        # Save summary at base output dir
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, 'sweep_summary.json'), 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)

        print("\nSweep completed.")
        return
    
    # Setup model
    print("Setting up MegaFS model...")
    model = setup_model(config)
    print(f"[OK] Model loaded on {config['device']}")
    
    # Determine which images to attack
    if args.batch:
        image_ids = [int(x.strip()) for x in args.batch.split(',')]
    else:
        image_ids = [args.image_id]
    
    # Limit samples if specified
    if config['experiment'].get('max_samples'):
        image_ids = image_ids[:config['experiment']['max_samples']]
    
    # Run attack
    if args.pair_mode and args.pairs:
        # Parse explicit pairs and run pair attacks (non-sweep path)
        def parse_pairs_cli(pairs_str: str) -> List[tuple]:
            items = []
            for tok in pairs_str.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                s, t = tok.split(':')
                items.append((int(s), int(t)))
            return items
        pairs = parse_pairs_cli(args.pairs)
        base_exp_dir = make_exp_dir(output_dir, config)
        for (src_id, tgt_id) in pairs:
            pair_dir = os.path.join(base_exp_dir, f"{src_id}_{tgt_id}")
            os.makedirs(pair_dir, exist_ok=True)
            run_pair_attack(config, model, src_id, tgt_id, pair_dir)
    elif args.full_compare and len(image_ids) >= 2:
        # Use the first two as (source, target) and write into a single experiment folder
        exp_dir = make_exp_dir(output_dir, config)
        run_pair_attack(config, model, image_ids[0], image_ids[1], exp_dir)
    elif len(image_ids) == 1:
        out_dir_single = make_exp_output_dir(output_dir, config, image_ids[0])
        run_single_attack(config, model, image_ids[0], out_dir_single)
    else:
        run_batch_attack(config, model, image_ids, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Attack completed! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

