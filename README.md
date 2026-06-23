# Towards Simulation-to-Reality Transfer in EIT Tactile Sensing: A Noise-Augmented Deep Learning Approach

[![CI](https://github.com/ronniepiku/eit-sim2real/actions/workflows/ci.yml/badge.svg)](https://github.com/ronniepiku/eit-sim2real/actions/workflows/ci.yml)

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
| 4 | Point contact | [0.35, 0.55] | [0.02, 0.05] | Small area, high pressure |
| 5 | Distributed | [0.80, 0.92] | [0.15, 0.25] | Large area contact |

All contact classes satisfy σ < σ₀, consistent with the positive piezoresistive response of ionic hydrogels (Lee et al., 2019).

## Project Structure

```
├── src/eit_sim2real/               # Installable Python package
│   ├── cli/                        # Click CLI entry points
│   │   ├── train.py                # eit train {cnn,baselines}
│   │   ├── evaluate.py             # eit evaluate
│   │   ├── experiments.py          # eit experiments {run-all,ablation,...}
│   │   ├── validate.py             # eit validate-dataset
│   │   └── environment.py          # eit log-environment
│   ├── configs/
│   │   └── config.yaml             # Central hyperparameter config
│   ├── data/
│   │   ├── load_dataset.py         # Loading, splitting, normalisation
│   │   └── noise.py                # 4-component noise model (on-the-fly augmentation)
│   ├── models/
│   │   ├── cnn1d.py                # EITConv1D (3×Conv1D, ~56k params)
│   │   └── baselines.py            # SVM, RF, MLP wrappers
│   ├── experiments/
│   │   ├── grid.py                 # Model × Dataset × Condition grid
│   │   ├── ablation.py             # Noise component ablation study
│   │   ├── hyperopt.py             # Hyperparameter grid search & architecture sweep
│   │   ├── additional.py           # Memorisation experiments (fixed-bias, different-draw)
│   │   └── mesh_refinement.py      # Cross-mesh evaluation study
│   ├── constants.py                # Shared constants (classes, noise components)
│   ├── utils.py                    # Device, seeds, prediction helpers
│   ├── train.py                    # CNN training logic
│   ├── evaluate.py                 # Evaluation & robustness sweeps
│   ├── visualisation.py            # Publication-quality plotting
│   ├── validate_dataset.py         # Dataset integrity & EDA report
│   └── log_environment.py          # Hardware/package version logging
├── matlab/                         # EIDORS simulation & dataset generation
│   ├── main.m                      # Main dataset generation (25,000 samples)
│   ├── create_mesh.m               # 2D/3D mesh creation (16 electrodes)
│   ├── generate_sample.m           # Single-sample forward solve
│   ├── setup_dependencies.m        # Auto-download EIDORS from SourceForge
│   ├── setup_eidors.m              # EIDORS path initialisation
│   ├── noise_model/
│   │   ├── add_noise.m             # 4-component noise application
│   │   └── load_noise_params.m     # YAML parameter loader
│   └── configs/
│       └── noise_params.yaml       # Literature-justified noise parameters
├── tests/                          # Unit + integration tests
├── dissertation/                   # LaTeX dissertation source
├── scripts/
│   └── run_pipeline.ps1            # Local CI-equivalent (Windows)
├── .github/workflows/ci.yml        # GitHub Actions CI (lint, typecheck, test)
├── pyproject.toml                  # Project config (hatchling build)
└── docs/                           # Additional documentation
    ├── SETUP.md                    # Installation & quick-start guide
    ├── NOISE_MODEL.md              # Noise model derivations
    ├── RAW_DATA_CREATION.md        # MATLAB dataset generation
    ├── EDA_FEATURE_REDUCTION.md    # EDA & feature reduction pipeline
    ├── HYPERPARAMETER_OPTIMISATION.md  # Hyperopt methodology
    └── ADDITIONAL_EXPERIMENTS.md   # Extended experiment details
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- MATLAB R2019b+ with EIDORS v3.12-ng (for dataset generation only)
- (Optional) NVIDIA GPU with CUDA 12.6 for accelerated CNN training

### Installation

```bash
git clone https://github.com/ronniepiku/eit-sim2real.git
cd eit-sim2real
uv sync

# Install PyTorch separately (CPU):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA 12.6:
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install the package in editable mode:
uv pip install -e .
```

See [`docs/SETUP.md`](docs/SETUP.md) for detailed installation instructions.

### Generate Dataset (MATLAB)

```matlab
cd matlab
setup_dependencies()  % Download EIDORS (first time only)
setup_eidors()        % Initialise paths
main                  % Generate 25,000 samples → data/eit_dataset.mat
```

### CLI Usage

After installation, the `eit` command is available:

```bash
# Train 1D-CNN with noise augmentation (default)
eit train cnn

# Train on clean data
eit train cnn --no-noise

# Train all baselines (SVM, RF, MLP)
eit train baselines

# Evaluate a trained model
eit evaluate --model-path results/models/cnn1d_noisy_best.pt

# Run the full experiment suite (grid, ablation, hyperopt, architecture-sweep, extended, additional)
eit experiments run-all

# Run all experiments except grid
eit experiments run-all --skip-grid

# Run only grid and ablation
eit experiments run-all --skip-hyperopt --skip-architecture-sweep --skip-extended --skip-additional

# Include mesh refinement study (requires fine-mesh dataset)
eit experiments run-all --include-mesh-refinement

# Run individual experiment types
eit experiments ablation                     # Noise component ablation
eit experiments additional                   # Memorisation experiments
eit experiments hyperopt --mode=grid-search  # Full hyperparameter grid search
eit experiments hyperopt --mode=arch-sweep   # Quick architecture depth sweep
eit experiments extended                     # All extended analyses (calibration, robustness, etc.)
eit experiments mesh-refinement              # Cross-mesh evaluation

# Validate dataset integrity
eit validate-dataset

# Log environment
eit log-environment
```

All commands support `--help` for full option details.

### Using Custom Datasets

```bash
# Train on a custom .mat file via config override
eit train cnn --config path/to/config.yaml
```

The `data.path` field in config.yaml controls which dataset is loaded. Cleaned/reduced datasets are available at:
- `data/cleaned/eit_cleaned_pca.mat` — PCA-reduced (7 components)
- `data/cleaned/eit_cleaned_lda.mat` — LDA-reduced (4 components)
- `data/cleaned/eit_cleaned_umap.mat` — UMAP-reduced (2 components)

**Note**: The CNN requires ≥8 features for its pooling layers. Use `data/eit_dataset.mat` (208 features) for CNN experiments; use SVM/RF/MLP for reduced datasets.

### Dataset Reconstructions

To generate true inverse-reconstructed class examples, run after dataset generation:

```matlab
cd matlab
setup_eidors()
generate_validation_reconstructions()
```

## Evaluation Framework

Models are evaluated under a 2×2 condition matrix:

|                      | Evaluate on Clean | Evaluate on Noisy |
|----------------------|:-----------------:|:-----------------:|
| **Train on Clean**   | Ceiling           | Vulnerability     |
| **Train on Noisy**   | Generalisation    | Robustness        |

Additionally:

- **Severity sweep**: noise multipliers {0.0×, 0.5×, 1.0×, 1.5×, 2.0×, 2.5×, 3.0×}
- **Statistical testing**: 5-fold stratified CV, paired t-tests with Bonferroni correction, Cohen's d
- **Ablation**: single-component (4 exps) + leave-one-out (4 exps) via Python-side noise injection

## Noise Model

Four physically-motivated components applied sequentially:

| # | Component | Parameter | Default | Reference |
|---|-----------|-----------|---------|----------|
| 1 | Gaussian measurement noise | SNR | 40 dB | Adler & Lionheart (2006) |
| 2 | Contact impedance variation | σ_z | 10% | Vilhunen et al. (2002) |
| 3 | Electrode positioning bias | b_max | 0.02 | Kolehmainen et al. (1997) |
| 4 | Quantisation noise | bits / V_range | 16 / 1.0 V | Hardware specs |

Implemented in both MATLAB (`matlab/noise_model/add_noise.m`) for dataset generation and Python (`src/eit_sim2real/data/noise.py`) for on-the-fly training augmentation. Features:

- Per-component toggling for ablation studies
- Severity scaling (all parameters scaled by a single multiplier)
- Multi-severity domain randomisation (severity sampled per-batch from a configurable range)

See [`docs/NOISE_MODEL.md`](docs/NOISE_MODEL.md) for detailed derivations.

## Reproducibility

- All random seeds fixed (`seed: 42` in `config.yaml`, MATLAB `rng(42)`)
- Dataset generation is deterministic given same EIDORS version
- Noise parameters stored in version-controlled YAML (`matlab/configs/noise_params.yaml`)
- Python noise model mirrors MATLAB implementation for consistency
- Train/val/test split: 70/15/15 (stratified)
- Environment logged via `eit log-environment` → `results/environment.json`
- CNN training uses early stopping (patience=40) and LR scheduling (patience=10)
- All dependencies locked via `uv.lock`

## Development

```bash
# Install with dev tools
uv pip install -e .
uv pip install pytest pytest-cov ruff mypy types-PyYAML pre-commit

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=eit_sim2real

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/eit_sim2real/ --ignore-missing-imports

# Full local pipeline (Windows PowerShell)
.\scripts\run_pipeline.ps1
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development workflow details.

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
