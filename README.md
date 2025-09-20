# MegaFS (Unofficial)

A modular implementation of "One Shot Face Swapping on Megapixels (CVPR 2021)" with enhanced debugging capabilities, comprehensive configuration management, and improved maintainability.

- **Reference**: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)
- **Paper**: [`One Shot Face Swapping on Megapixels` (arXiv:2105.04932)](https://arxiv.org/abs/2105.04932)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, utilities, and configuration
- **Multiple Swap Methods**: Support for FTM, ID Injection, and LCR face swapping techniques
- **Hierarchical Feature Encoding**: HieRFE encoder for rich facial detail extraction
- **StyleGAN2 Integration**: High-quality face synthesis with StyleGAN2 generator
- **Comprehensive Debugging**: Built-in logging, profiling, and system monitoring
- **Data Management**: Automated dataset mapping and path resolution
- **Colab Ready**: Interactive Jupyter notebook for Google Colab usage

## Requirements

- Python 3.7+
- PyTorch 1.7+ (CUDA support recommended)
- OpenCV (`opencv-python`)
- NumPy
- tqdm (optional, for progress bars)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/n01r1r/MegaFS.git
   cd MegaFS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For CUDA support**, install a CUDA-enabled PyTorch from the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Project Structure

```
MegaFS/
├── models/                 # Core model implementations
│   ├── megafs.py          # Main MegaFS class
│   ├── hierfe.py          # Hierarchical Region Feature Encoder
│   ├── face_transfer.py   # Face transfer modules (FTM, Injection, LCR)
│   ├── stylegan2.py       # StyleGAN2 generator
│   ├── model_factory.py   # Model creation factory
│   └── weight_loaders.py  # Weight loading utilities
├── utils/                 # Utility modules
│   ├── data_utils.py      # Data management and mapping
│   ├── image_utils.py     # Image processing utilities
│   └── debug_utils.py     # Debugging and profiling tools
├── config.py              # Configuration management
├── create_datamap.py      # Dataset mapping utility
├── MegaFS.ipynb          # Interactive Colab notebook
└── requirements.txt       # Python dependencies
```

## Quick Start

### Google Colab (Recommended)

1. **Open the notebook**: `MegaFS.ipynb`
2. **Upload your dataset** to Google Drive:
   - Upload `celeba_mask_hq.zip` to `/content/drive/MyDrive/Datasets/`
3. **Upload weight files** to Google Drive:
   - Place all weight files in `/content/drive/MyDrive/Datasets/weights/`
4. **Run the notebook**: Everything will be set up automatically

### Local Usage

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

## Configuration

The modular configuration system supports multiple environments:

```python
from config import Config, DEFAULT_CONFIGS

# Use predefined configurations
config = DEFAULT_CONFIGS["local"]    # Local development
config = DEFAULT_CONFIGS["colab"]    # Google Colab

# Or create custom configuration
config = Config(
    swap_type="ftm",                 # "ftm", "injection", or "lcr"
    dataset_root="./CelebAMask-HQ",
    img_root="./CelebAMask-HQ/CelebA-HQ-img",
    mask_root="./CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    checkpoint_dir="./weights"
)
```

## Dataset Setup

### Required Datasets

1. **CelebA-HQ**: High-quality face images
   - Structure: `CelebA-HQ-img/<id>.jpg`
2. **CelebAMask-HQ**: Segmentation masks
   - Structure: `CelebAMask-HQ-mask-anno/*/<id>_*.png`

### Data Mapping

The codebase uses a data mapping system for robust path resolution. The `DataMapManager` class handles automatic path resolution for images and masks:

```python
from utils.data_utils import DataMapManager

# Initialize data manager
data_manager = DataMapManager("data_map.json")

# Resolve paths for specific IDs
image_path, mask_path = data_manager.resolve_paths_for_id(100, dataset_root)
```

Generate dataset mapping:

```bash
# Run from dataset root directory
python create_datamap.py
```

This creates `data_map.json` with automatic path mapping that the MegaFS class uses internally.

## Weight Files

Place the following weight files in the `weights/` directory:

- **MegaFS checkpoints**: `{swap_type}_final.pth`
  - `ftm_final.pth`
  - `injection_final.pth`
  - `lcr_final.pth`
- **StyleGAN2 generator**: `stylegan2-ffhq-config-f.pth`

> **Note**: Weight files are not included. Obtain from official sources or train your own models.

## Architecture

### Core Components

1. **HieRFE (Hierarchical Region Feature Encoder)**
   - ResNet50 backbone with FPN
   - Multi-scale feature extraction
   - StyleMapping layers for latent generation

2. **FaceTransferModule**
   - **FTM**: Transfer Cell with multiple blocks
   - **Injection**: ID injection with normalization
   - **LCR**: Latent Code Regularization

3. **StyleGAN2 Generator**
   - High-resolution face synthesis
   - 1024x1024 output resolution
   - 18 latent dimensions

### Processing Pipeline

1. **Preprocessing**: Load and resize images to 256x256
2. **Encoding**: Extract hierarchical features with HieRFE
3. **Transfer**: Apply face transfer using selected method
4. **Generation**: Synthesize high-resolution result with StyleGAN2
5. **Postprocessing**: Apply mask blending and refinement

## Debugging & Profiling

The modular design includes comprehensive debugging tools:

```python
# Enable debug logging
megafs = MegaFS(config=config, debug=True)

# Access debug utilities
megafs.debug_logger.log("Custom message")
megafs.profiler.start_timer("operation")
# ... perform operation ...
duration = megafs.profiler.end_timer("operation")
```

## Usage Examples

### Single Image Swap

```python
# Basic face swap
result_path, result_image = megafs.run(
    src_idx=100,      # Source image ID
    tgt_idx=200,      # Target image ID
    refine=True,       # Apply refinement
    save_path="swap_result.jpg"
)
```

### Batch Processing

```python
# Process multiple pairs
pairs = [(100, 200), (300, 400), (500, 600)]
for src_id, tgt_id in pairs:
    result_path, result_image = megafs.run(
        src_idx=src_id,
        tgt_idx=tgt_id,
        refine=True
    )
```

### Custom Configuration

```python
# Advanced configuration
from config import Config

config = Config(
    swap_type="injection",
    dataset_root="/path/to/dataset",
    img_root="/path/to/images",
    mask_root="/path/to/masks",
    checkpoint_dir="/path/to/weights"
)

megafs = MegaFS(config=config, debug=True)
```

## Contributing

This is an unofficial implementation focused on modularity and maintainability. Contributions are welcome for:

- Bug fixes and improvements
- Additional swap methods
- Performance optimizations
- Documentation enhancements

## License

- **Method**: Based on CVPR 2021 paper "One Shot Face Swapping on Megapixels"
- **Datasets**: CelebA-HQ is non-commercial; follow original licenses
- **Usage**: Research and educational purposes only
- **Compliance**: Ensure adherence to original dataset and model licenses

## Acknowledgments

- Original paper authors and the reference implementation
- StyleGAN2 authors for the generator architecture
- CelebA-HQ and CelebAMask-HQ dataset creators

## References

- [Original Paper](https://arxiv.org/abs/2105.04932)
- [Reference Implementation](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)
- [StyleGAN2](https://github.com/NVlabs/stylegan2)
- [CelebA-HQ Dataset](https://github.com/tkarras/progressive_growing_of_gans)