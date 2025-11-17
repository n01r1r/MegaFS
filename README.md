# MegaFS (Unofficial)

A modular implementation of "One Shot Face Swapping on Megapixels (CVPR 2021)" with enhanced debugging capabilities, comprehensive configuration management, and improved maintainability.

- **Reference**: [`zyainfal/One-Shot-Face-Swapping-on-Megapixels`](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)
- **Paper**: [`One Shot Face Swapping on Megapixels` (arXiv:2105.04932)](https://arxiv.org/abs/2105.04932)

## What's New

### v3 - Architecture Improvements (Latest)
- **PyTorch 2.1+ Upgrade**: Full CUDA 11.x/12.x compatibility for modern GPUs (A100, RTX 30xx/40xx)
- **nn.Module Integration**: MegaFS now inherits from `torch.nn.Module` for standard PyTorch workflows
- **Device Management**: Automatic device detection and centralized management
- **Flexible Dataset Loading**: Optional data_map.json with folder-based auto-discovery
- **Simplified ModelFactory**: Dictionary-based weight loading for maintainability

### Previous Features
- **Gradient Compatibility**: Full support for gradient-based experiments
- **PyTorch Dataset Integration**: Built-in Dataset/DataLoader with train/val/test splits
- **YAML Configuration**: Easy configuration management
- **Training Infrastructure**: Modular `BaseTrainer` class

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, utilities, and configuration
- **Multiple Swap Methods**: Support for FTM, ID Injection, and LCR face swapping techniques
- **Hierarchical Feature Encoding**: HieRFE encoder for rich facial detail extraction
- **StyleGAN2 Integration**: High-quality face synthesis with StyleGAN2 generator
- **Comprehensive Debugging**: Built-in logging, profiling, and system monitoring
- **Data Management**: Automated dataset mapping and path resolution
- **Image Similarity Evaluation**: Comprehensive metrics including LPIPS, PSNR, SSIM, and MSE
- **Batch Processing**: Efficient evaluation of multiple image pairs
- **Statistical Analysis**: Mean, std, min, max, median across all results
- **Visualization**: Charts and graphs for result analysis
- **Colab Ready**: Interactive Jupyter notebooks for Google Colab usage
- **Gradient Compatibility**: Full support for gradient-based experiments and attacks
- **PyTorch Dataset Integration**: Built-in Dataset/DataLoader support for training
- **YAML Configuration**: Easy configuration management with YAML files
- **Training Infrastructure**: Modular trainer base class for custom experiments

## Requirements

- Python 3.7+
- PyTorch 2.1+ (CUDA support recommended for modern GPUs: A100, 30xx series, etc.)
- OpenCV (`opencv-python`)
- NumPy
- tqdm (optional, for progress bars)
- PyYAML (for YAML configuration support)
- pytest (for testing, optional)

### CUDA Compatibility

The project now uses PyTorch 2.1.0, which is compatible with modern CUDA versions (11.x and 12.x), making it suitable for:
- NVIDIA A100 GPUs
- RTX 30xx/40xx series
- Other modern CUDA-enabled GPUs

## Installation

### Prerequisites

- NVIDIA GPU (RTX 30xx/40xx or A100 recommended) with CUDA 11.x or 12.x drivers
- Python 3.10+
- Git

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/n01r1r/MegaFS.git
   cd MegaFS
   ```

2. **Create a virtual environment (highly recommended)**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   
   For CUDA 12.x:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   
   For CUDA 11.x:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```
   
   For CPU-only installation:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   
   **Why `--extra-index-url`?** This ensures pip pulls PyTorch packages from the official PyTorch server, which resolves dependency conflicts with NumPy and other packages.

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Project Structure

```
MegaFS/
├── models/                 # Core model implementations
│   ├── megafs.py          # Main MegaFS class (with gradient support)
│   ├── hierfe.py          # Hierarchical Region Feature Encoder
│   ├── face_transfer.py   # Face transfer modules (FTM, Injection, LCR)
│   ├── stylegan2.py       # StyleGAN2 generator
│   ├── model_factory.py   # Model creation factory
│   └── weight_loaders.py  # Weight loading utilities
├── data/                   # Dataset integration (NEW)
│   ├── __init__.py
│   └── face_swap_dataset.py # PyTorch Dataset with train/val/test splits
├── training/               # Training infrastructure (NEW)
│   ├── __init__.py
│   ├── trainer.py         # Base trainer for experiments
│   └── example_experiment.py # Example experiment template
├── configs/                # Configuration files (NEW)
│   ├── default.yaml       # Default configuration
│   └── experiment.yaml    # Experiment configuration
├── utils/                 # Utility modules
│   ├── data_utils.py      # Data management and mapping
│   ├── image_utils.py     # Image processing utilities
│   ├── debug_utils.py     # Debugging and profiling tools
│   └── metrics.py         # Image similarity evaluation metrics
├── config.py              # Configuration management (with YAML support)
├── create_datamap.py      # Dataset mapping utility
├── MegaFS.ipynb          # Interactive Colab notebook
├── MegaFS_Evaluation.ipynb # Image similarity evaluation notebook
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

#### Option 1: Using the Local Runner Script (Easiest)

```bash
# Basic usage with default settings
python run_local.py

# Run with custom IDs
python run_local.py --src-id 100 --tgt-id 200

# Use different swap method
python run_local.py --swap-type injection

# Custom dataset and weights paths
python run_local.py --dataset-root ./my_dataset --weights-dir ./my_weights

# Enable gradients for experiments
python run_local.py --enable-grads

# See all options
python run_local.py --help
```

**Key Options:**
- `--src-id`, `--tgt-id`: Source and target image IDs
- `--swap-type`: Swap method (`ftm`, `injection`, `lcr`)
- `--dataset-root`: Path to dataset (default: `./dataset/CelebAMask-HQ`)
- `--weights-dir`: Path to weights (default: `./weights`)
- `--output-dir`: Output directory (default: `./outputs`)
- `--no-refine`: Faster processing without refinement
- `--enable-grads`: Enable gradients for experiments

#### Option 2: Programmatic Usage

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

## Image Similarity Evaluation

The framework includes comprehensive image similarity evaluation capabilities with multiple metrics:

### Available Metrics

- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)  
- **SSIM**: Structural Similarity Index (higher is better, range [0,1])
- **MSE**: Mean Squared Error (lower is better)

### Evaluation Notebook

Use `MegaFS_Evaluation.ipynb` for comprehensive evaluation:

1. **Open the evaluation notebook** in Google Colab
2. **Upload your dataset** and weight files to Google Drive
3. **Configure evaluation parameters**:
   - Evaluation size (number of image pairs)
   - Swap methods to compare (FTM, Injection, LCR)
   - Refinement settings
4. **Run evaluation** - automatically processes all methods
5. **View results** - statistical analysis and visualizations

### Programmatic Evaluation

```python
from utils.metrics import ImageMetrics, FaceSwapEvaluator
from models.megafs import MegaFS

# Initialize evaluator
evaluator = FaceSwapEvaluator(use_gpu=True)

# Run face swap evaluation
results = evaluator.evaluate_pair(source_img, target_img, swapped_img, refined_img)

# Calculate statistics across multiple results
stats = evaluator.calculate_statistics(all_results)
```

### Batch Evaluation

```python
# Evaluate multiple image pairs
batch_results = run_batch_evaluation(
    handler_instance=megafs_handler,
    id_pairs=[(100, 200), (300, 400), (500, 600)],
    refine=True,
    max_pairs=50
)

# Generate comprehensive statistics
statistics = evaluator.calculate_statistics(batch_results)
```

## Gradient-Based Experiments (NEW)

The framework now supports gradient-based experiments for research purposes, including adversarial attacks with improved visual quality.

### Attack Loss Functions

The adversarial attack implementation uses three loss components for imperceptible perturbations:

1. **L_ID (Identity Destruction)**: Minimizes cosine similarity to destroy identity in face region
2. **L_SIM (Similarity Preservation)**: Maintains visual similarity using LPIPS perceptual loss (recommended) or MSE/L1
3. **L_TV (Total Variation)**: Encourages smooth perturbations by minimizing adjacent pixel differences

**Total Loss = λ₁ × L_ID + λ_sim × L_SIM + λ_tv × L_TV**

**Note**: L_SEM (Semantic Collapse) has been removed to improve visual quality while maintaining attack effectiveness.

### Key Features

- **LPIPS Support**: Perceptual similarity loss for better visual quality preservation
- **Total Variation Loss**: Smooth perturbations without high-frequency noise
- **Early Stopping**: Automatic termination when L_ID < 0.2 threshold is reached
- **Optimized Hyperparameters**: Reduced iterations (300) and epsilon (8.0) for faster, cleaner attacks

### Enabling Gradients

```python
from config import Config
from models.megafs import MegaFS
import torch

# Load configuration from YAML
config = Config.from_yaml('configs/experiment.yaml')

# Initialize with device and gradients
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MegaFS(config=config, enable_grads=True, device=device)

# Use standard PyTorch methods
model.train()  # Enable gradients
model.eval()  # Disable gradients
```

### Using PyTorch Dataset

```python
from data import create_dataloaders
from models.megafs import MegaFS
import torch

# Create dataloaders with train/val/test splits
# Option 1: Use data_map.json (recommended for CelebA-HQ)
dataloaders = create_dataloaders(
    dataset_root='./dataset/CelebAMask-HQ',
    data_map_path='./data_map.json',
    batch_size=8,
    num_workers=4
)

# Option 2: Auto-discover from folder structure (for custom datasets)
dataloaders_custom = create_dataloaders(
    dataset_root='./my_custom_images',
    use_data_map=False,  # No data_map.json needed
    batch_size=8
)

# Use in experiments
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for batch in dataloaders['train']:
    source = batch['source'].to(device).requires_grad_(True)
    target = batch['target'].to(device)
    
    # Forward pass with gradients
    output = model.forward(source, target)
    
    # Compute loss and backpropagate
    loss = your_loss_function(output, target)
    loss.backward()
```

### Training Infrastructure

```python
from training import BaseTrainer
from training.example_experiment import ExampleExperiment

# Extend BaseTrainer for custom experiments
class MyExperiment(BaseTrainer):
    def compute_loss(self, source, target, batch):
        output = self.model.forward(source, target)
        return your_custom_loss(output, target)

# Create trainer
trainer = MyExperiment(
    model=model,
    dataloaders=dataloaders,
    device='cuda'
)

# Run training
trainer.fit(num_epochs=10)

# Test
test_metrics = trainer.test()
```

### YAML Configuration

Create `configs/experiment.yaml`:

```yaml
# Experiment configuration
swap_type: ftm
dataset_root: ./dataset/CelebAMask-HQ
checkpoint_dir: ./weights

experiment:
  enable_grads: true  # Enable gradients
  batch_size: 4
  num_workers: 2
  seed: 42

data_split:
  train: 0.7
  val: 0.15
  test: 0.15
```

Load configuration:
```python
config = Config.from_yaml('configs/experiment.yaml')
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