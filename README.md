# Towards Simulation-to-Reality Transfer in EIT Tactile Sensing: A Noise-Augmented Deep Learning Approach

A project investigating whether physically-motivated noise modelling during training improves the robustness of deep learning classifiers for touch type identification using simulated Electrical Impedance Tomography (EIT) measurements.

## Overview

This project:
1. Simulates EIT measurements on a prosthetic arm e-skin model using EIDORS
2. Applies a configurable, physically-motivated multi-component noise model
3. Trains and evaluates ML classifiers (SVM, Random Forest, MLP, 1D-CNN)
4. Performs systematic ablation to identify critical noise components
5. Demonstrates that noise-augmented training improves classifier robustness

## Touch Classes

| Class | Description | Simulation |
|-------|-------------|------------|
| 1 | No contact | Baseline (σ=1.0) |
| 2 | Light touch | Small Δσ, medium area |
| 3 | Firm press | Large Δσ, medium area |
| 4 | Point contact | Medium Δσ, very small area |
| 5 | Distributed contact | Medium Δσ, large area |

## Project Structure

```
├── matlab/                      # EIDORS simulation & dataset generation
│   ├── main.m                   # Main dataset generation script
│   ├── create_mesh.m            # 2D/3D mesh creation
│   ├── generate_sample.m        # Single sample generation
│   ├── encode_label.m           # Touch class encoding
│   ├── setup_eidors.m           # EIDORS initialisation
│   ├── noise_model/             # Configurable noise components
│   │   ├── add_noise.m          # Noise application (5 components)
│   │   └── load_noise_params.m  # YAML config loader
│   ├── configs/
│   │   └── noise_params.yaml    # Noise parameters (literature-justified)
│   └── utils/
│       └── get_element_centers.m
├── python/                      # ML training & evaluation pipeline
│   ├── train.py                 # Model training (CLI entry point)
│   ├── evaluate.py              # Evaluation & metrics
│   ├── ablation.py              # Ablation study runner
│   ├── visualisation.py         # Publication-quality plotting
│   ├── models/
│   │   ├── cnn1d.py             # 1D-CNN with adaptive pooling
│   │   └── baselines.py         # SVM, RF, MLP
│   ├── data/
│   │   └── load_dataset.py      # Dataset loading & splitting
│   └── configs/
│       ├── config.yaml          # Central hyperparameter config
│       └── loader.py            # YAML config loader utility
├── notebooks/                   # Jupyter notebooks for EDA & visualisation
├── results/                     # Generated outputs
│   ├── figures/                 # Saved plots (PNG + PDF)
│   ├── tables/                  # CSV metrics, confusion matrices
│   └── models/                  # Serialised model checkpoints
├── tests/                       # Unit tests
│   ├── test_baselines.py
│   ├── test_cnn1d.py
│   └── test_data.py
├── docs/                        # Additional documentation
│   ├── NOISE_MODEL.md           # Noise model specification
│   └── SETUP.md                 # Detailed setup guide
├── pyproject.toml               # Python project config (uv)
└── README.md
```

## Setup

### Prerequisites

- MATLAB R2019b+ with EIDORS (http://eidors3d.sourceforge.net/)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone repository
git clone https://github.com/ronniepiku/eit-sim2real.git
cd eit-sim2real

# Install Python dependencies
uv sync

# Download EIDORS and extract to matlab/eidors/
# See: http://eidors3d.sourceforge.net/download.shtml
```

### Dataset Generation (MATLAB)

```matlab
cd matlab
setup_eidors()   % Initialise EIDORS
main             % Generate dataset (saves to data/)
```

### Training (Python)

All Python scripts load defaults from `python/configs/config.yaml`.
CLI arguments override config values when provided.

```bash
# Train 1D-CNN on noisy data (default)
uv run python python/train.py --model cnn1d

# Train on clean data instead
uv run python python/train.py --model cnn1d --no-noise

# Train baseline models (saved as .joblib)
uv run python python/train.py --model svm
uv run python python/train.py --model random_forest
uv run python python/train.py --model mlp

# Override hyperparameters
uv run python python/train.py --model cnn1d --epochs 200 --lr 0.0005 --batch-size 128
```

### Evaluation

```bash
# Evaluate a trained CNN
uv run python python/evaluate.py --model-path results/models/cnn1d_noisy_best.pt

# Evaluate a baseline model
uv run python python/evaluate.py --model-path results/models/svm_noisy.joblib

# Include robustness sweep
uv run python python/evaluate.py --model-path results/models/cnn1d_noisy_best.pt --robustness

# Evaluate on clean test data
uv run python python/evaluate.py --model-path results/models/cnn1d_noisy_best.pt --eval-on clean
```

Results are saved to `results/tables/` as JSON metrics, `.npy` confusion matrices,
and plain-text classification reports.

### Ablation Study

```bash
# Run with Random Forest (default — fast)
uv run python python/ablation.py

# Run with 1D-CNN
uv run python python/ablation.py --model cnn1d

# Run full per-component noise ablation (requires per-config datasets)
uv run python python/ablation.py --model random_forest --all-configs
```

### Visualisation

The `python/visualisation.py` module provides functions for:

| Function | Description |
|----------|-------------|
| `plot_training_history()` | Loss and accuracy curves |
| `plot_confusion_matrix()` | Normalised / raw confusion matrix heatmap |
| `plot_per_class_f1()` | Per-class F1 bar chart |
| `plot_robustness()` | Accuracy/F1 vs. noise level |
| `plot_robustness_comparison()` | Multi-model robustness overlay |
| `plot_pca()` | 2D PCA scatter of feature space |
| `plot_tsne()` | 2D t-SNE scatter of feature space |
| `plot_ablation_heatmap()` | Ablation results heatmap from CSV |
| `save_figure()` | Save to PNG + PDF at 300 DPI |

Example usage in a notebook or script:

```python
from python.visualisation import plot_confusion_matrix, save_figure
import numpy as np

cm = np.load("results/tables/cnn1d_noisy_best_confusion_matrix.npy")
fig = plot_confusion_matrix(cm)
save_figure(fig, "results/figures/confusion_matrix")
```

## Noise Model

All noise parameters are justified from published EIT hardware characterisations:

| Component | Reference |
|-----------|-----------|
| Gaussian measurement noise (SNR 40–60 dB) | Adler & Lionheart (2006) |
| Electrode contact impedance variation | Vilhunen et al. (2002) |
| Systematic temporal drift | Boone & Holder (1996) |
| Quantisation noise (16-bit ADC) | Hardware specifications |
| Electrode positioning error | Kolehmainen et al. (1997) |

See [`docs/NOISE_MODEL.md`](docs/NOISE_MODEL.md) for the full specification and
[`matlab/configs/noise_params.yaml`](matlab/configs/noise_params.yaml) for the parameter file.

## Reproducibility

- All random seeds are fixed (`seed: 42` in `config.yaml`)
- Dataset generation is deterministic given the same EIDORS version
- Noise parameters are stored in version-controlled YAML
- Train/val/test split ratios: 70/15/15 (stratified)
- CNN training uses early stopping (patience = 20 epochs)

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{eit_touch_2026,
  title={Towards Simulation-to-Reality Transfer in {EIT} Tactile Sensing: A Noise-Augmented Deep Learning Approach},
  author={Ronald Piku},
  year={2026},
  school={University of Bath}
}
```

## License

MIT
