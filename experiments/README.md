# Adversarial Attack Experiments

This directory contains scripts for running dual-target adversarial attacks on HieRFE.

## Overview

The implementation uses **self-supervised mask generation** from HieRFE's FPN feature maps, eliminating the need for external segmentation models.

## Key Features

- **Self-supervised masks**: Extract face masks directly from FPN features (`f8`, `f16`, `f32`)
- **Dual-target optimization**:
  - L_ID: Destroy identity in face region (A1)
  - L_SEM: Inject face structure into background (A2)
- **Minimal code changes**: No modifications to existing MegaFS code
- **Reproducible**: Configuration-based hyperparameters

## Usage

### Basic Attack

Run an attack on a single image:

```bash
python experiments/run_attack.py \
  --config configs/attack_config.yaml \
  --image-id 2332 \
  --output-dir experiments/results/attack_2332
```

### Batch Attack

Attack multiple images:

```bash
python experiments/run_attack.py \
  --config configs/attack_config.yaml \
  --batch "2332,2107,3000" \
  --max-samples 10
```

### Evaluation

Evaluate attack effectiveness by comparing clean vs adversarial face swaps:

```bash
python experiments/evaluate_attack.py \
  --config configs/attack_config.yaml \
  --image-id 2332 \
  --output-dir experiments/evaluation
```

## Configuration

Edit `configs/attack_config.yaml` to adjust:

- **PGD parameters**: `epsilon`, `alpha`, `num_iter`
- **Loss weights**: `lambda_1` (identity destruction), `lambda_2` (semantic collapse)
- **Mask generation**: `feature_layers`, `threshold`, `mask_type`
- **Paths**: Dataset, weights, output directories

## Output Files

Each attack generates:

```
results/
└── image_XXXXX/
    ├── original.jpg          # Original image
    ├── adversarial.jpg       # Adversarial image
    ├── mask_face.jpg         # Generated face mask (M1)
    ├── mask_bg.jpg           # Generated background mask (M2)
    ├── perturbation.jpg     # Visualized perturbation
    ├── comparison.jpg        # Side-by-side comparison
    ├── loss_curves.jpg       # Loss curves over iterations
    └── metrics.json          # Computed metrics (L2, L-inf, SSIM, etc.)
```

## Implementation Details

### Mask Generation

1. Extract FPN features (`f8`, `f16`, `f32`) from HieRFE
2. Compute channel-wise attention: `feature.pow(2).mean(dim=1)`
3. Interpolate to input size
4. Apply threshold to create binary mask

### Attack Loop

1. Generate masks from clean image (detached)
2. Initialize perturbation `delta = 0`
3. For each iteration:
   - Compute `adv_image = original + delta`
   - Extract features from face region (A1) and background (A2)
   - Compute L_ID (identity destruction) and L_SEM (semantic injection)
   - Update `delta` via PGD: `delta -= alpha * sign(gradient)`
   - Clamp `delta` to `[-epsilon, epsilon]`

### Gradients

- MegaFS is initialized with `enable_grads=True` to allow gradient computation
- HieRFE inherits from `nn.Module`, so autograd works automatically
- Masks are detached to prevent gradient flow during PGD

## Tests

Run unit tests:

```bash
python -m pytest tests/test_attack.py -v
```

Tests cover:
- Mask generation and shape validation
- Gradient flow through HieRFE
- MegaFS gradient mode compatibility
- Attack class initialization

## Requirements

See `requirements.txt` for all dependencies. Key requirements:

- PyTorch >= 2.1.0
- OpenCV
- NumPy
- Matplotlib (for visualization)

## References

Based on the dual-target adversarial attack strategy for face swapping systems:
- **L_ID**: Identity destruction via latent space manipulation
- **L_SEM**: Semantic collapse through structural feature injection
- **Self-supervised masking**: FPN attention maps for region separation

