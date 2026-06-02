# Towards Simulation-to-Reality Transfer in EIT Tactile Sensing: A Noise-Augmented Deep Learning Approach

Project investigating whether physically-motivated noise modelling during training improves the robustness of machine learning classifiers for touch-type identification using simulated Electrical Impedance Tomography (EIT) measurements on an ionic hydrogel substrate.

## Overview

This project:

1. Simulates EIT boundary voltage measurements using EIDORS (FEM forward solver)
2. Models an ionic hydrogel substrate with positive piezoresistive response (conductivity decreases under pressure)
3. Applies a configurable 4-component physically-motivated noise model
4. Trains and evaluates 4 classifiers (SVM, Random Forest, MLP, 1D-CNN)
5. Supports online noise augmentation with multi-severity domain randomisation
6. Performs systematic ablation (single-component + leave-one-out) to identify critical noise sources
7. Demonstrates that noise-augmented training bridges the simulation-to-reality gap
8. Generates EIDORS inverse reconstructions for representative clean/noisy class examples

## Touch Classes

| Class | Description | Conductivity (S/m) | Radius | Evidence |
|-------|-------------|-------------------|--------|----------|
| 1 | No contact | 1.00 (baseline) | — | Homogeneous reference |
| 2 | Light touch | [0.85, 0.95] | [0.06, 0.10] | 5–15% decrease |
| 3 | Firm press | [0.55, 0.75] | [0.08, 0.12] | 25–45% decrease |
| 4 | Point contact | [0.60, 0.80] | [0.02, 0.04] | Small area, high pressure |
| 5 | Distributed | [0.60, 0.80] | [0.12, 0.20] | Large area contact |

All contact classes satisfy σ < σ₀, consistent with the positive piezoresistive response of ionic hydrogels (Lee et al., 2019).

## Project Structure

```
├── matlab/                          # EIDORS simulation & dataset generation
│   ├── main.m                       # Main dataset generation (25,000 samples)
│   ├── create_mesh.m                # 2D/3D mesh creation (16 electrodes)
│   ├── generate_sample.m            # Single-sample forward solve
│   ├── generate_validation_reconstructions.m  # EIDORS inverse reconstructions for class inspection
│   ├── encode_label.m               # Touch class encoding
│   ├── setup_eidors.m               # EIDORS path initialisation
│   ├── validate_mesh_convergence.m  # Mesh refinement convergence study
│   ├── noise_model/
│   │   ├── add_noise.m              # 4-component noise application
│   │   └── load_noise_params.m      # YAML parameter loader
│   ├── configs/
│   │   └── noise_params.yaml        # Literature-justified noise parameters
│   └── utils/
│       └── get_element_centers.m
├── python/                          # ML training & evaluation pipeline
│   ├── utils.py                     # Shared utilities (device, seeds, predict helpers)
│   ├── train.py                     # Model training (CNN + baselines) with online noise augmentation
│   ├── evaluate.py                  # Evaluation, robustness & severity sweep (Python noise model)
│   ├── ablation.py                  # Ablation study (Python-side noise, no MATLAB dependency)
│   ├── run_all_experiments.py       # Master experiment runner (all models × datasets × conditions)
│   ├── statistical_tests.py         # 5-fold CV, paired t-tests, Cohen's d
│   ├── architecture_sweep.py        # CNN depth selection validation
│   ├── hyperparameter_optimisation.py # Grid search for CNN hyperparameters
│   ├── log_environment.py           # Hardware/package version logging
│   ├── validate_dataset.py          # Dataset integrity & EDA report
│   ├── visualisation.py             # Publication-quality plotting
│   ├── models/
│   │   ├── cnn1d.py                 # EITConv1D (3×Conv1D, ~56k params)
│   │   └── baselines.py             # SVM, RF, MLP wrappers
│   ├── data/
│   │   ├── load_dataset.py          # Loading, splitting, normalisation
│   │   └── noise.py                 # Python-side 4-component noise model (on-the-fly augmentation)
│   └── configs/
│       ├── config.yaml              # Central hyperparameter config + noise augmentation settings
│       └── loader.py                # YAML config loader
├── dissertation/                    # LaTeX dissertation source
│   ├── main.tex
│   ├── references.bib
│   └── chapters/
├── tests/                           # Unit tests (45 tests)
│   ├── test_baselines.py
│   ├── test_cnn1d.py
│   ├── test_config.py
│   ├── test_data.py
│   └── test_noise.py
├── data/                            # Generated datasets (.mat)
├── results/                         # Outputs (models, figures, tables)
│   ├── models/                      # Trained checkpoints (.pt/.joblib)
│   ├── figures/                     # Plots grouped by model and data condition
│   │   └── <model>/<noisy|clean>/
│   └── tables/                      # Evaluation JSON/TXT/NPY outputs
├── docs/                            # Additional documentation
│   ├── NOISE_MODEL.md               # Noise model full specification
│   └── SETUP.md                     # Detailed setup guide
└── pyproject.toml                   # Python project config (uv)
```

## Quick Start

### Prerequisites

- MATLAB R2019b+ with EIDORS v3.12-ng
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- (Optional, for GPU acceleration) NVIDIA GPU with CUDA 12.6 and matching drivers
  - If you plan to use an NVIDIA GPU, install CUDA 12.6 — the project is
     tested with PyTorch cu126 wheels. Install a matching `torch` build using
     the PyTorch index for `cu126` (see `docs/SETUP.md`).

### Installation

```bash
git clone https://github.com/ronniepiku/eit-sim2real.git
cd eit-sim2real
uv sync
```

See [`docs/SETUP.md`](docs/SETUP.md) for detailed installation instructions.

### Generate Dataset (MATLAB)

```matlab
cd matlab
setup_eidors()   % Initialise EIDORS paths
main             % Generate 25,000 samples → data/eit_dataset.mat
```

### Train Models (Python)

```bash
# Train 1D-CNN on noisy data (default configuration)
uv run python python/train.py --model cnn1d

# Train on clean data (for comparison)
uv run python python/train.py --model cnn1d --no-noise

# Optional: choose custom output roots
# - model files  -> --output-dir
# - figures      -> --figures-dir/<model>/<noisy|clean>/
uv run python python/train.py \
  --model cnn1d \
  --output-dir results/models \
  --figures-dir results/figures

# Train all baselines under all 4 conditions
uv run python python/run_all_experiments.py
```

#### Using Custom Datasets

All scripts support loading datasets from custom paths via `--data-path`:

```bash
# Train on custom dataset
uv run python python/train.py --model cnn1d --data-path /path/to/custom_data.mat

# Evaluate on custom dataset
uv run python python/evaluate.py \
  --model-path results/models/cnn1d_noisy_best.pt \
  --data-path /path/to/custom_data.mat

# Run baselines on custom dataset
uv run python python/run_all_experiments.py --data-path /path/to/custom_data.mat

# Run statistical tests on custom dataset
uv run python python/statistical_tests.py --model cnn1d --data-path /path/to/custom_data.mat
```

**Using Cleaned & Reduced Datasets**

The EDA notebook produces cleaned datasets with duplicate removal, redundant feature removal, and optional dimensionality reduction:

```bash
# PCA-reduced (7 components) — use with shallow models
uv run python python/train.py --model random_forest --data-path data/cleaned/eit_cleaned_pca.mat

# LDA-reduced (4 components, supervised) — use with shallow models
uv run python python/train.py --model svm --data-path data/cleaned/eit_cleaned_lda.mat

# UMAP-reduced (2 components, unsupervised) — visualisation-focused
uv run python python/train.py --model random_forest --data-path data/cleaned/eit_cleaned_umap.mat
```

**Note on model compatibility**: The CNN requires ≥8 features for its pooling layers.
In this repository, use `data/eit_dataset.mat` (208 features) for CNN experiments. Use non-CNN models (SVM/RF/MLP) for dimensionality-reduced datasets.

See [`docs/SETUP.md`](docs/SETUP.md#using-custom-datasets) for dataset format requirements.

### Evaluate

```bash
# Standard evaluation with robustness and severity sweep
uv run python python/evaluate.py \
    --model-path results/models/cnn1d_noisy_best.pt \
    --robustness

# Statistical significance testing (5-fold CV + paired t-tests)
uv run python python/statistical_tests.py --model cnn1d
```

### Ablation Study

```bash
# Core 4-condition mismatch experiment (uses Python-side noise injection)
uv run python python/ablation.py --model cnn1d

# Full per-component ablation (single-component + leave-one-out, no MATLAB dependency)
uv run python python/ablation.py --model cnn1d --all-configs

# Multi-severity trained CNN (domain randomisation over [0.5×, 2.0×])
# Enable in config.yaml: noise_augmentation.enabled: true
uv run python python/train.py --model cnn1d
```

### Dataset Reconstructions

To generate true inverse-reconstructed class examples for the report, run the MATLAB helper after dataset generation:

```matlab
cd matlab
setup_eidors()
generate_validation_reconstructions()
```

This writes clean and noisy reconstruction figures to `results/dataset_validation/reconstructions/` and the dataset validator will pick them up automatically when present.

## Evaluation Framework

Models are evaluated under a 2×2 condition matrix:

|                      | Evaluate on Clean | Evaluate on Noisy |
|----------------------|:-----------------:|:-----------------:|
| **Train on Clean**   | Ceiling           | Vulnerability     |
| **Train on Noisy**   | Generalisation    | Robustness        |

Additionally:

- **Severity sweep**: noise multipliers {0.0×, 0.5×, 1.0×, 1.5×, 2.0×, 2.5×, 3.0×} using Python noise model
- **Statistical testing**: 5-fold stratified CV, paired t-tests with Bonferroni correction, Cohen's d effect sizes
- **Ablation**: single-component (4 exps) + leave-one-out (4 exps) via Python-side noise injection

## Noise Model

Four physically-motivated components applied sequentially:

| # | Component | Parameter | Default | Reference |
|---|-----------|-----------|---------|----------|
| 1 | Gaussian measurement noise | SNR | 40 dB | Adler & Lionheart (2006) |
| 2 | Contact impedance variation | σ_z | 10% | Vilhunen et al. (2002) |
| 3 | Electrode positioning bias | b_max | 0.02 | Kolehmainen et al. (1997) |
| 4 | Quantisation noise | bits / V_range | 16 / 1.0 V | Hardware specs |

The noise model is implemented in both MATLAB (`matlab/noise_model/add_noise.m`) for dataset
generation and Python (`python/data/noise.py`) for on-the-fly training augmentation. The Python
implementation supports:

- Per-component toggling for ablation studies
- Severity scaling (all parameters scaled by a single multiplier)
- Multi-severity domain randomisation (severity sampled per-batch from a configurable range)

See [`docs/NOISE_MODEL.md`](docs/NOISE_MODEL.md) for detailed derivations.

## Scripts Reference

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Train any model | `uv run python python/train.py --model cnn1d` |
| `evaluate.py` | Evaluate + robustness | `uv run python python/evaluate.py --model-path X --robustness` |
| `run_all_experiments.py` | All models × datasets × conditions | `uv run python python/run_all_experiments.py` |
| `ablation.py` | Noise component ablation | `uv run python python/ablation.py --model cnn1d` |
| `statistical_tests.py` | 5-fold CV + significance | `uv run python python/statistical_tests.py --model rf` |
| `architecture_sweep.py` | CNN depth validation | `uv run python python/architecture_sweep.py` |
| `hyperparameter_optimisation.py` | Grid search for CNN | `uv run python python/hyperparameter_optimisation.py` |
**All scripts support `--data-path <path>` to load custom datasets.**

## Output Layout

Training now separates model artifacts from figures:

- Models are saved to `results/models/` (or `--output-dir`).
- Figures are saved to `results/figures/<model>/<noisy|clean>/` (or under `--figures-dir`).

Examples:

- `results/models/cnn1d_noisy_best.pt`
- `results/figures/cnn1d/noisy/cnn1d_noisy_cm_test.png`
- `results/figures/random_forest/clean/random_forest_clean_per_class_metrics_val.png`

## Reproducibility

- All random seeds fixed (`seed: 42` in `config.yaml`, MATLAB `rng(42)`)
- Dataset generation is deterministic given same EIDORS version
- Noise parameters stored in version-controlled YAML (`matlab/configs/noise_params.yaml`)
- Python noise model mirrors MATLAB implementation for consistency
- Train/val/test split: 70/15/15 (stratified)
- Environment logged via `log_environment.py` → `results/environment.json`
- CNN training uses early stopping (patience=40) and LR scheduling (patience=10)
- Ablation study uses Python-side noise (fresh realisation each epoch for CNN)
- All dependencies (including PyTorch) declared in `pyproject.toml`
- Logging configured only at CLI entry points (modules use `logging.getLogger(__name__)`)

## Running Tests

```bash
uv run pytest tests/ -v
```

All 45 tests cover:

- **CNN architecture** (`test_cnn1d.py`): output shapes, adaptive pooling, gradient flow
- **Baseline models** (`test_baselines.py`): creation, training, prediction
- **Data pipeline** (`test_data.py`): splitting, stratification, normalisation, CV folds
- **Noise model** (`test_noise.py`): config construction, noise application, reproducibility, severity scaling
- **Configuration** (`test_config.py`): YAML loading, required keys, error handling

## Citation

```bibtex
@mastersthesis{piku2026eit,
  title   = {Towards Simulation-to-Reality Transfer in {EIT} Tactile Sensing:
             A Noise-Augmented Deep Learning Approach},
  author  = {Piku, Ronald},
  year    = {2026},
  school  = {University of Bath}
}
```

## License

MIT
