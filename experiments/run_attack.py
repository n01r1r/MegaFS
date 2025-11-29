"""
Adversarial attack runner for HieRFE dual-target strategy
Execute attacks on CelebA-HQ images with configurable parameters
"""

import os
import sys
import argparse
import json
import random
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2

from models.megafs import MegaFS
from utils.attack_utils import DualTargetPGDAttack, compute_metrics
from utils.image_utils import ImageProcessor
from experiments.evaluate_attack import evaluate_image_pair
from experiments.experiment_utils import (
    load_config, setup_model, get_image_path, 
    make_exp_output_dir, make_exp_dir
)


def auto_evaluate_if_enabled(
    config: Dict[str, Any],
    model: MegaFS,
    image_ids: List[int],
    output_dir: str
) -> None:
    """Run evaluation pipeline if enabled in experiment settings."""
    if not config.get('experiment', {}).get('auto_evaluate'):
        return
    for eid in image_ids:
        try:
            evaluate_image_pair(config, model, eid, output_dir)
        except Exception as exc:
            print(f"[WARN] Auto evaluation failed for image {eid}: {exc}")


def _sweep_job(job_args: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level function for ProcessPoolExecutor sweep execution."""
    try:
        # Reconstruct config
        cfg = json.loads(job_args['config_json'])
        # Apply job-specific overrides
        cfg['attack']['lambda_1'] = float(job_args['lambda_1'])
        cfg['attack']['lambda_2'] = float(job_args['lambda_2'])
        cfg['attack']['lambda_sim'] = float(job_args['lambda_sim'])
        cfg['attack']['lambda_tv'] = float(job_args['lambda_tv'])
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
                'lambda_sim': job_args['lambda_sim'],
                'lambda_tv': job_args['lambda_tv'],
                'epsilon': job_args['epsilon'],
                'num_iter': job_args['num_iter']
            }
        res.update({
            'lambda_1': job_args['lambda_1'],
            'lambda_2': job_args['lambda_2'],
            'lambda_sim': job_args['lambda_sim'],
            'lambda_tv': job_args['lambda_tv'],
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
        src_id, tgt_id, l1, l2, eps, nit, l_sim, l_tv, alpha_val, config_json, base_output_dir = job_tuple
        # Reconstruct config
        cfg = json.loads(config_json)
        cfg['attack']['lambda_1'] = float(l1)
        cfg['attack']['lambda_2'] = float(l2)
        cfg['attack']['lambda_sim'] = float(l_sim)
        cfg['attack']['lambda_tv'] = float(l_tv)
        cfg['attack']['epsilon'] = float(eps)
        cfg['attack']['num_iter'] = int(nit)
        cfg['attack']['alpha'] = float(alpha_val)
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
        lambda_sim=config.get('attack', {}).get('lambda_sim', 0.0),
        lambda_tv=config.get('attack', {}).get('lambda_tv', 0.0),
        device=config['device'],
        verbose=config['experiment']['verbose'],
        sem_variant=config.get('attack', {}).get('sem_variant', 'self_collapse'),
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
        detector_kwargs=detector_kwargs,
        sim_loss_type=config.get('attack', {}).get('sim_loss_type', 'mse'),
        structure_weakening_factor=config.get('attack', {}).get('structure_weakening_factor', 0.7),
        early_stop_threshold=config.get('attack', {}).get('early_stop_threshold', 0.2),
        convergence_window=config.get('attack', {}).get('convergence_window', 1000),
        convergence_tolerance=config.get('attack', {}).get('convergence_tolerance', 1e-6),
        min_iter_for_convergence=config.get('attack', {}).get('min_iter_for_convergence', 1000),
        maximize_similarity=config.get('attack', {}).get('maximize_similarity', False),
        random_init=config.get('attack', {}).get('random_init', False),
        target_type=config.get('attack', {}).get('target_type', 'image')
    )
    
    # Execute attack
    output_path = output_dir
    os.makedirs(output_path, exist_ok=True)
    
    try:
        adversarial = attack.attack(image_path, output_path)
        
        # Load original for metrics
        original = ImageProcessor.load_image(image_path, target_size=(256, 256))
        
        # Compute metrics
        metrics = compute_metrics(original, adversarial)
        print(f"\nAttack completed!")
        print(f"L2 norm: {metrics.get('L2_norm', 'N/A'):.2f}")
        print(f"L-inf norm: {metrics.get('Linf_norm', 'N/A'):.2f}")
        if 'SSIM' in metrics:
            print(f"SSIM: {metrics['SSIM']:.4f}")
        
        # Save metrics
        serializable_metrics = {k: (float(v) if hasattr(v, 'item') else float(v) if isinstance(v, (np.floating,)) else v)
                                for k, v in metrics.items()}
        with open(os.path.join(output_path, 'metrics_target.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
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
                
                # Optionally attack source too
                src_adv_img = src_img
                attack_source = bool(config.get('experiment', {}).get('attack_source', False))
                full_compare = bool(config.get('experiment', {}).get('full_compare', False))
                
                if attack_source:
                    src_adv_img = attack.attack(src_path, output_path, output_prefix='source')
                    m_src = compute_metrics(src_img, src_adv_img)
                    with open(os.path.join(output_path, 'metrics_source.json'), 'w') as f:
                        json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_src.items()}, f, indent=2)
                
                # Extract source image ID
                src_filename = os.path.basename(src_path)
                src_id_from_file = int(os.path.splitext(src_filename)[0])
                
                # Load full-resolution images
                src_img_full, tgt_img_full, _ = model.read_pair(src_id_from_file, image_id)
                full_h, full_w = tgt_img_full.shape[:2]
                src_h, src_w = src_img_full.shape[:2]
                
                # Upscale adversarial images
                tgt_adv_full = cv2.resize(adversarial, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
                src_adv_full = cv2.resize(src_adv_img, (src_w, src_h), interpolation=cv2.INTER_LINEAR) if attack_source else src_img_full.copy()
                
                # Save full-resolution adversarial images
                src_adv_path = os.path.join(output_path, f'source_adv_{src_id_from_file}.jpg') if attack_source else None
                tgt_adv_path = os.path.join(output_path, f'target_adv_{image_id}.jpg')
                if attack_source:
                    cv2.imwrite(src_adv_path, src_adv_full[:, :, ::-1])
                cv2.imwrite(tgt_adv_path, tgt_adv_full[:, :, ::-1])
                
                # Save full-resolution clean images
                cv2.imwrite(os.path.join(output_path, 'source_clean.jpg'),  src_img_full[:, :, ::-1])
                cv2.imwrite(os.path.join(output_path, 'target_clean.jpg'),  tgt_img_full[:, :, ::-1])
                cv2.imwrite(os.path.join(output_path, 'target_adv.jpg'),    tgt_adv_full[:, :, ::-1])
                if attack_source:
                    cv2.imwrite(os.path.join(output_path, 'source_adv.jpg'), src_adv_full[:, :, ::-1])
                
                # Generate swap cases via CLI
                print(f"\n[INFO] Generating swap cases using run_local.py...")
                swap_CC_full = get_swap_result_via_cli(config, src_id_from_file, image_id,
                                                       src_adv_path=None, tgt_adv_path=None, refine=True, output_dir=output_path)
                swap_CA_full = get_swap_result_via_cli(config, src_id_from_file, image_id,
                                                       src_adv_path=None, tgt_adv_path=tgt_adv_path, refine=True, output_dir=output_path)
                
                if attack_source:
                    swap_AC_full = get_swap_result_via_cli(config, src_id_from_file, image_id,
                                                           src_adv_path=src_adv_path, tgt_adv_path=None, refine=True, output_dir=output_path)
                    swap_AA_full = get_swap_result_via_cli(config, src_id_from_file, image_id,
                                                           src_adv_path=src_adv_path, tgt_adv_path=tgt_adv_path, refine=True, output_dir=output_path)
                else:
                    swap_AC_full = None
                    swap_AA_full = None
                
                # Save results
                if full_compare:
                    cv2.imwrite(os.path.join(output_path, 'swap_CC.jpg'), swap_CC_full[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_CA.jpg'), swap_CA_full[:, :, ::-1])
                    if swap_AC_full is not None:
                        cv2.imwrite(os.path.join(output_path, 'swap_AC.jpg'), swap_AC_full[:, :, ::-1])
                    if swap_AA_full is not None:
                        cv2.imwrite(os.path.join(output_path, 'swap_AA.jpg'), swap_AA_full[:, :, ::-1])
                    
                    # Create grid
                    top = np.hstack([swap_CC_full, swap_CA_full])
                    if swap_AC_full is not None and swap_AA_full is not None:
                        bottom = np.hstack([swap_AC_full, swap_AA_full])
                        grid = np.vstack([top, bottom])
                    else:
                        grid = top
                    cv2.imwrite(os.path.join(output_path, 'comparison_grid_ALL.jpg'), grid[:, :, ::-1])
                else:
                    cv2.imwrite(os.path.join(output_path, 'swap_on_original.jpg'),    swap_CC_full[:, :, ::-1])
                    cv2.imwrite(os.path.join(output_path, 'swap_on_adversarial.jpg'), swap_CA_full[:, :, ::-1])
                    comp = np.hstack([src_img_full, tgt_img_full, tgt_adv_full, swap_CC_full, swap_CA_full])
                    cv2.imwrite(os.path.join(output_path, 'swap_comparison.jpg'), comp[:, :, ::-1])
                
                # Save manifest
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
                        'swaps': {
                            'CC': 'swap_CC.jpg',
                            'CA': 'swap_CA.jpg',
                            'AC': 'swap_AC.jpg',
                            'AA': 'swap_AA.jpg'
                        },
                        'metrics_target': 'metrics_target.json'
                    }
                }
                with open(os.path.join(output_path, 'effective_config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                with open(os.path.join(output_path, 'manifest.json'), 'w') as f:
                    json.dump(manifest, f, indent=2)
                    
        except Exception as e:
            print(f"Warning: swap post-check failed: {e}")
        
        auto_evaluate_if_enabled(config, model, [image_id], output_path)
        
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


def get_swap_result_via_cli(config: Dict[str, Any], src_id: int, tgt_id: int, 
                             src_adv_path: Optional[str] = None,
                             tgt_adv_path: Optional[str] = None,
                             refine: bool = True,
                             output_dir: Optional[str] = None) -> np.ndarray:
    """Get swap result by calling run_local.py via CLI."""
    import subprocess
    
    # Get paths from config
    dataset_root = config['paths']['dataset_root']
    weights_dir = config['paths']['checkpoint_dir']
    # Try to get data_map from config, fallback to default location
    data_map_path = config.get('paths', {}).get('data_map', None)
    if data_map_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_map_path = os.path.join(project_root, 'data_map.json')
    swap_type = config['model']['swap_type']
    
    # Build CLI command
    run_local_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_local.py')
    
    cmd = [
        sys.executable,
        run_local_script,
        '--src-id', str(src_id),
        '--tgt-id', str(tgt_id),
        '--dataset-root', dataset_root,
        '--weights-dir', weights_dir,
        '--data-map', data_map_path,
        '--swap-type', swap_type,
    ]
    
    if not refine:
        cmd.append('--no-refine')
    
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    if src_adv_path:
        cmd.extend(['--src-adv-path', os.path.abspath(src_adv_path)])
    
    if tgt_adv_path:
        cmd.extend(['--tgt-adv-path', os.path.abspath(tgt_adv_path)])
    
    # Run subprocess
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        stdout_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # print(f"  [run_local] {line}")
                stdout_lines.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, output='\\n'.join(stdout_lines))
        
        result = type('obj', (object,), {'stdout': '\\n'.join(stdout_lines), 'returncode': process.returncode})()
        
        # Parse output to find saved image path
        output_lines = result.stdout.split('\\n')
        result_path = None
        for line in output_lines:
            if '[OK] Result saved to:' in line or 'Result saved to:' in line:
                parts = line.split('Result saved to:')
                if len(parts) > 1:
                    result_path = parts[1].strip()
                    break
        
        if result_path is None:
            if output_dir:
                result_path = os.path.join(output_dir, f'swap_{src_id}_to_{tgt_id}_{swap_type}.jpg')
            else:
                result_path = os.path.join('./outputs', f'swap_{src_id}_to_{tgt_id}_{swap_type}.jpg')
        
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Swap result not found at: {result_path}")
        
        result_image = cv2.imread(result_path)
        if result_image is None:
            raise RuntimeError(f"Failed to load swap result from: {result_path}")
        
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        num_images = 4 if refine else 3
        img_width = result_image.shape[1] // num_images
        
        if refine:
            swapped_face = result_image[:, img_width * 3:img_width * 4]
        else:
            swapped_face = result_image[:, img_width * 2:img_width * 3]
        
        return swapped_face
        
    except Exception as e:
        raise RuntimeError(f"Failed to get swap result via CLI: {e}")


def run_pair_attack(
    config: Dict[str, Any],
    model: MegaFS,
    src_id: int,
    tgt_id: int,
    output_dir: str
) -> Dict[str, Any]:
    """Run full 4-way comparison for a specified (source, target) pair."""
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
    
    # Build detector kwargs
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
        lambda_sim=config.get('attack', {}).get('lambda_sim', 0.0),
        lambda_tv=config.get('attack', {}).get('lambda_tv', 0.0),
        device=config['device'],
        verbose=config['experiment']['verbose'],
        sem_variant=config.get('attack', {}).get('sem_variant', 'self_collapse'),
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
        detector_kwargs=detector_kwargs,
        sim_loss_type=config.get('attack', {}).get('sim_loss_type', 'mse'),
        structure_weakening_factor=config.get('attack', {}).get('structure_weakening_factor', 0.7),
        early_stop_threshold=config.get('attack', {}).get('early_stop_threshold', 0.2),
        convergence_window=config.get('attack', {}).get('convergence_window', 1000),
        convergence_tolerance=config.get('attack', {}).get('convergence_tolerance', 1e-6),
        min_iter_for_convergence=config.get('attack', {}).get('min_iter_for_convergence', 1000),
        maximize_similarity=config.get('attack', {}).get('maximize_similarity', False),
        random_init=config.get('attack', {}).get('random_init', False),
        target_type=config.get('attack', {}).get('target_type', 'image')
    )

    # Load clean images at 256x256 for attack
    src_clean_256 = ImageProcessor.load_image(src_path, target_size=(256, 256))
    tgt_clean_256 = ImageProcessor.load_image(tgt_path, target_size=(256, 256))

    # Attack source only (or both if attack_target is enabled)
    attack_target = config.get('experiment', {}).get('attack_target', False)
    
    if attack_target:
        tgt_adv_256 = attack.attack(tgt_path, output_dir, output_prefix='target', target_image_path=src_path)
    else:
        tgt_adv_256 = tgt_clean_256.copy()
    
    src_adv_256 = attack.attack(src_path, output_dir, output_prefix='source', target_image_path=tgt_path)

    # Compute and save metrics
    if attack_target:
        m_tgt = compute_metrics(tgt_clean_256, tgt_adv_256)
        with open(os.path.join(output_dir, 'metrics_target.json'), 'w') as f:
            json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_tgt.items()}, f, indent=2)
    
    m_src = compute_metrics(src_clean_256, src_adv_256)
    with open(os.path.join(output_dir, 'metrics_source.json'), 'w') as f:
        json.dump({k: (float(v) if hasattr(v, 'item') else float(v)) for k, v in m_src.items()}, f, indent=2)

    # Load full-resolution clean images
    src_clean_full, tgt_clean_full, _ = model.read_pair(src_id, tgt_id)
    
    # Upscale adversarial images
    tgt_h, tgt_w = tgt_clean_full.shape[:2]
    src_h, src_w = src_clean_full.shape[:2]
    if attack_target:
        tgt_adv_full = cv2.resize(tgt_adv_256, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    else:
        tgt_adv_full = tgt_clean_full.copy()
    src_adv_full = cv2.resize(src_adv_256, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    
    # Save full-resolution adversarial images
    src_adv_path = os.path.join(output_dir, f'source_adv_{src_id}.jpg')
    tgt_adv_path = os.path.join(output_dir, f'target_adv_{tgt_id}.jpg') if attack_target else None
    cv2.imwrite(src_adv_path, src_adv_full[:, :, ::-1])
    if attack_target:
        cv2.imwrite(tgt_adv_path, tgt_adv_full[:, :, ::-1])
    
    # Save clean images
    cv2.imwrite(os.path.join(output_dir, 'source_clean.jpg'),  src_clean_full[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'target_clean.jpg'),  tgt_clean_full[:, :, ::-1])
    cv2.imwrite(os.path.join(output_dir, 'source_adv.jpg'),    src_adv_full[:, :, ::-1])
    if attack_target:
        cv2.imwrite(os.path.join(output_dir, 'target_adv.jpg'),    tgt_adv_full[:, :, ::-1])

    # Generate swap cases
    if attack_target:
        print(f"\n[INFO] Generating 4 swap cases using run_local.py...")
        swap_CC = get_swap_result_via_cli(config, src_id, tgt_id, 
                                          src_adv_path=None, tgt_adv_path=None, refine=True, output_dir=output_dir)
        swap_CA = get_swap_result_via_cli(config, src_id, tgt_id,
                                          src_adv_path=None, tgt_adv_path=tgt_adv_path, refine=True, output_dir=output_dir)
        swap_AC = get_swap_result_via_cli(config, src_id, tgt_id,
                                          src_adv_path=src_adv_path, tgt_adv_path=None, refine=True, output_dir=output_dir)
        swap_AA = get_swap_result_via_cli(config, src_id, tgt_id,
                                          src_adv_path=src_adv_path, tgt_adv_path=tgt_adv_path, refine=True, output_dir=output_dir)
        
        cv2.imwrite(os.path.join(output_dir, 'swap_CC.jpg'), swap_CC[:, :, ::-1])
        cv2.imwrite(os.path.join(output_dir, 'swap_CA.jpg'), swap_CA[:, :, ::-1])
        cv2.imwrite(os.path.join(output_dir, 'swap_AC.jpg'), swap_AC[:, :, ::-1])
        cv2.imwrite(os.path.join(output_dir, 'swap_AA.jpg'), swap_AA[:, :, ::-1])
        
        top = np.hstack([swap_CC, swap_CA])
        bottom = np.hstack([swap_AC, swap_AA])
        grid = np.vstack([top, bottom])
        cv2.imwrite(os.path.join(output_dir, 'comparison_grid_ALL.jpg'), grid[:, :, ::-1])
    else:
        print(f"\n[INFO] Generating 2 swap cases using run_local.py (source only attack)...")
        swap_CC = get_swap_result_via_cli(config, src_id, tgt_id, 
                                          src_adv_path=None, tgt_adv_path=None, refine=True, output_dir=output_dir)
        swap_AC = get_swap_result_via_cli(config, src_id, tgt_id,
                                          src_adv_path=src_adv_path, tgt_adv_path=None, refine=True, output_dir=output_dir)
        
        cv2.imwrite(os.path.join(output_dir, 'swap_CC.jpg'), swap_CC[:, :, ::-1])
        cv2.imwrite(os.path.join(output_dir, 'swap_AC.jpg'), swap_AC[:, :, ::-1])
        
        grid = np.hstack([swap_CC, swap_AC])
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
        'artifacts': {
            'source_clean': 'source_clean.jpg',
            'target_clean': 'target_clean.jpg',
            'source_adv':   'source_adv.jpg',
            'target_adv':   'target_adv.jpg' if attack_target else None,
            'swaps': {
                'CC': 'swap_CC.jpg',
                'CA': 'swap_CA.jpg' if attack_target else None,
                'AC': 'swap_AC.jpg',
                'AA': 'swap_AA.jpg' if attack_target else None
            }
        }
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    return {
        'src_id': src_id,
        'tgt_id': tgt_id,
        'success': True
    }


def main():
    parser = argparse.ArgumentParser(description='Run adversarial attack on MegaFS')
    parser.add_argument('--config', type=str, default='configs/attack_config.yaml', help='Path to attack config')
    parser.add_argument('--image-id', type=int, help='Single image ID to attack')
    parser.add_argument('--output-dir', type=str, default='experiments/results', help='Output directory')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--pair-mode', action='store_true', help='Run in pair attack mode')
    parser.add_argument('--pairs', type=str, help='Pairs to attack (e.g., "100:200,300:400")')
    
    # Overrides
    parser.add_argument('--lambda_1', type=float, help='Override lambda_1')
    parser.add_argument('--lambda_2', type=float, help='Override lambda_2')
    parser.add_argument('--lambda_sim', type=float, help='Override lambda_sim')
    parser.add_argument('--lambda_tv', type=float, help='Override lambda_tv')
    parser.add_argument('--epsilon', type=float, help='Override epsilon')
    parser.add_argument('--num_iter', type=int, help='Override num_iter')
    parser.add_argument('--alpha', type=float, help='Override alpha')
    parser.add_argument('--mask-edge-blur', type=int, help='Override mask edge blur kernel size')
    parser.add_argument('--target-type', type=str, help='Override target type (image/zero/random)')
    parser.add_argument('--random-init', action='store_true', help='Enable random initialization')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.lambda_1 is not None: config['attack']['lambda_1'] = args.lambda_1
    if args.lambda_2 is not None: config['attack']['lambda_2'] = args.lambda_2
    if args.lambda_sim is not None: config['attack']['lambda_sim'] = args.lambda_sim
    if args.lambda_tv is not None: config['attack']['lambda_tv'] = args.lambda_tv
    if args.epsilon is not None: config['attack']['epsilon'] = args.epsilon
    if args.num_iter is not None: config['attack']['num_iter'] = args.num_iter
    if args.alpha is not None: config['attack']['alpha'] = args.alpha
    if args.mask_edge_blur is not None: 
        config.setdefault('mask_generation', {})['edge_blur'] = args.mask_edge_blur
    if args.target_type is not None: config['attack']['target_type'] = args.target_type
    if args.random_init: config['attack']['random_init'] = True
    
    # Setup output directory
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run mode
    if args.pair_mode:
        if not args.pairs:
            print("Error: --pairs required for pair mode")
            return
        
        pairs = []
        for p in args.pairs.split(','):
            s, t = map(int, p.split(':'))
            pairs.append((s, t))
            
        print(f"Running pair attack on {len(pairs)} pairs...")
        
        # Prepare jobs
        jobs = []
        for s, t in pairs:
            # Tuple for pickle-safe multiprocessing
            job = (
                s, t, 
                config['attack']['lambda_1'],
                config['attack']['lambda_2'],
                config['attack']['epsilon'],
                config['attack']['num_iter'],
                config['attack'].get('lambda_sim', 0.0),
                config['attack'].get('lambda_tv', 0.0),
                config['attack'].get('alpha', 1.0),
                json.dumps(config),
                base_output_dir
            )
            jobs.append(job)
            
        if args.parallel > 1:
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(_sweep_job_pair, job) for job in jobs]
                for future in as_completed(futures):
                    res = future.result()
                    if res.get('success'):
                        print(f"[OK] Pair {res.get('src_id')}:{res.get('tgt_id')} finished")
                    else:
                        print(f"[FAIL] Pair failed: {res.get('error')}")
        else:
            # Serial execution
            model = setup_model(config)
            exp_dir = make_exp_dir(base_output_dir, config)
            for s, t in pairs:
                run_pair_attack(config, model, s, t, exp_dir)
                
    elif args.image_id:
        # Single image attack
        model = setup_model(config)
        out_dir = make_exp_output_dir(base_output_dir, config, args.image_id)
        run_single_attack(config, model, args.image_id, out_dir)
        
    else:
        print("Please specify --image-id or --pair-mode")


if __name__ == "__main__":
    main()
