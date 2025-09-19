# MegaFS (Unofficial)

Unofficial implementation of "One Shot Face Swapping on Megapixels (CVPR 2021)" with small interface cleanups and a dataset path mapping utility.

- Reference repository: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)


## Attribution / License

- The method and official resources are from the CVPR 2021 paper and the reference repository above.
- Please follow the original dataset and pre-trained weights licenses and terms. The upstream repo references CelebA‑HQ and provides swapped datasets; see their README and license notices.
- CelebA‑HQ is non‑commercial; the upstream swapped datasets are distributed under Creative Commons CC BY‑NC 4.0 per the reference repo.
- This repository is for research and educational purposes only. If you use or distribute models/data, ensure compliance with the original licenses and any third‑party terms.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+ (with CUDA support recommended)
- OpenCV (`opencv-python`)
- NumPy
- tqdm (optional, for progress bars)

### Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA support, install a CUDA-enabled PyTorch matching your system from the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Features

- Hierarchical representation encoder (HieRFE) for richer facial details
- Multiple swap modes: FTM, ID Injection, and LCR
- StyleGAN2-based synthesis for stable, high-quality outputs
- Data path mapping utility (`create_datamap.py`) for robust dataset routing

## Setup

1) Dataset
   - Download CelebA‑HQ and CelebAMask‑HQ (or equivalent). Ensure the following structure under your dataset root:
     - `CelebA-HQ-img/` with `<id>.jpg`
     - `CelebAMask-HQ-mask-anno/` with subfolders containing `<id>_*.png`

2) Weights
   - Place MegaFS checkpoints and StyleGAN2 weights in `weights/`:
     - `{swap_type}_final.pth` (e.g., `ftm_final.pth`)
     - `stylegan2-ffhq-config-f.pth`

3) Data map
   - From the dataset root, generate a data map once:
     - `python create_datamap.py`
   - Pass the loaded mapping to `MegaFS` as shown below.

## Usage

### Google Colab (Recommended)

The easiest way to use MegaFS is through Google Colab:

1. **Open the notebook**: `MegaFS.ipynb`
2. **Upload your dataset** to Google Drive:
   - Upload `celeba_mask_hq.zip` to `/content/drive/MyDrive/Datasets/`
3. **Upload weight files** to Google Drive:
   - Place all weight files in `/content/drive/MyDrive/Datasets/weights/`
4. **Run the notebook**: The notebook will automatically:
   - Clone this repository
   - Mount Google Drive
   - Extract the dataset
   - Generate data mapping
   - Initialize the models

### Local Usage

Basic programmatic usage:

## Datasets

- Training/Inference commonly use CelebA‑HQ images and parsing masks (e.g., CelebAMask‑HQ annotations). Directory convention assumed by this repo:
  - `CelebA-HQ-img/` contains images named `<id>.jpg`
  - `CelebAMask-HQ-mask-anno/` contains subfolders with masks named like `<id>_*.png`

Upstream dataset description and download pointers are listed in the reference repo’s README. See: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)


## Weights

Place weights under `weights/` (create the folder if it doesn’t exist).

- MegaFS encoder/swapper checkpoint: `{swap_type}_final.pth` (e.g., `ftm_final.pth`)
- StyleGAN2 generator: `stylegan2-ffhq-config-f.pth`

These files are not provided here. Obtain from the official sources, train them yourself, or convert from compatible releases, respecting the original licenses.


## New: Data Path Mapping

This repo adds a simple mapping utility to robustly resolve image/mask paths when directory layouts vary.

- Script: `create_datamap.py`
- Output: `data_map.json` mapping integer `id` to relative paths:
  - `{"<id>": {"image_path": "CelebA-HQ-img/<id>.jpg", "mask_path": "CelebAMask-HQ-mask-anno/.../<id>_*.png"}}`

Usage (run from the dataset root that contains both folders):

```bash
python create_datamap.py
```

Then pass the loaded mapping to `MegaFS`:

```python
from config import DEFAULT_CONFIGS
from models.megafs import MegaFS

# Use predefined configuration
config = DEFAULT_CONFIGS["local"]  # or "colab" for Colab environment

# Initialize MegaFS with configuration
megafs = MegaFS(
    config=config,
    debug=True  # Enable debug logging
)

# Run face swap
result_path, result_image = megafs.run(
    src_idx=100,
    tgt_idx=200,
    refine=True,
    save_path="result.jpg"
)
```

### Modular Architecture

This implementation features a modular architecture for better debugging and maintenance:

- **`config.py`**: Centralized configuration management
- **`models/`**: Model definitions and weight loading
- **`utils/`**: Image processing, data management, and debugging utilities
- **`MegaFS.ipynb`**: Interactive notebook for Colab usage

## Reference

- Paper: [`One Shot Face Swapping on Megapixels` (arXiv:2105.04932)](https://arxiv.org/abs/2105.04932)
- Repo: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)
