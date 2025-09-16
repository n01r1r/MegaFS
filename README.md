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
pip install torch torchvision opencv-python numpy tqdm
```

For CUDA support, install PyTorch with CUDA from the [official PyTorch website](https://pytorch.org/get-started/locally/).

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
from models.megafs import MegaFS
import json, os

with open(os.path.join(".", "data_map.json"), "r", encoding="utf-8") as f:
    data_map = {int(k): v for k, v in json.load(f).items()}

megafs = MegaFS(
    swap_type="ftm",                # or "injection", "lcr"
    img_root="./CelebA-HQ-img",
    mask_root="./CelebAMask-HQ-mask-anno",
    checkpoint_dir="./weights",
    data_map=data_map,
)
```

## Reference

- Paper: [`One Shot Face Swapping on Megapixels` (arXiv:2105.04932)](https://arxiv.org/abs/2105.04932)
- Repo: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)
