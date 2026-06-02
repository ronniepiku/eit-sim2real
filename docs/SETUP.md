# Setup & Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Windows 11 / Ubuntu 22.04 |
| MATLAB | R2019b | R2024b |
| Python | 3.11 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | Not required | NVIDIA with CUDA 12.6 (for faster CNN training; tested with PyTorch cu126) |
| Disk | 500 MB | 2 GB (including EIDORS + datasets) |

## Python Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

### Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
cd eit-sim2real
uv sync
```

This installs all runtime and development dependencies specified in `pyproject.toml`:

- **Runtime**: torch, numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, pyyaml, joblib, h5py, umap-learn
- **Dev**: pytest, ruff, mypy, jupyter, ipykernel, types-PyYAML

If you plan to use an NVIDIA GPU for training, this project was tested with
CUDA 12.6 and the corresponding `cu126` PyTorch wheels. After running
`uv sync`, install the matching GPU build of PyTorch (PowerShell example):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If you don't have CUDA or do not require GPU acceleration, the CPU-only
PyTorch wheels installed by `uv sync` will work.

### Verify Python Installation

```bash
# Run test suite (should pass all 45 tests)
uv run python -m pytest tests/ -v

# Verify imports work
uv run python -c "from python.data.load_dataset import load_mat_dataset; print('OK')"

# Log environment details
uv run python python/log_environment.py
```

## EIDORS Installation

EIDORS v3.12-ng is included in this repository under `matlab/eidors-v3.12-ng/`.
No separate download is required.

If you need a fresh installation:

1. Download from: <http://eidors3d.sourceforge.net/download.shtml>
2. Extract to `matlab/eidors-v3.12-ng/`
3. Verify the directory structure:

   ```
   matlab/eidors-v3.12-ng/
   ├── eidors/
   │   ├── startup.m
   │   ├── eidors_startup.m
   │   └── ...
   ├── Netgen-5.3_x64/
   └── htdocs/
   ```

## MATLAB Verification

```matlab
cd matlab
setup_eidors()
% Should print: "EIDORS initialised successfully"

% Quick test: create a mesh and verify measurements
[fmdl, vh] = create_mesh();
fprintf('Mesh: %d elements, %d measurements\n', size(fmdl.elems, 1), length(vh.meas));
% Expected: 16-electrode adjacent pattern → 208 measurements

% Optional: validate mesh convergence (compares coarse/medium/fine)
results = validate_mesh_convergence(100);
```

## Dataset Generation

```matlab
cd matlab
setup_eidors()
main    % Generates 25,000 samples (5,000/class) → data/eit_dataset.mat
```

**Expected output**: `data/eit_dataset.mat` containing:

- `dataset_X_clean` — (25000 × 208) clean voltage difference vectors
- `dataset_X_noisy` — (25000 × 208) noise-corrupted vectors
- `dataset_y` — (25000 × 1) class labels (1-indexed)

Generation takes approximately 2–4 hours on a single CPU core.

## EIT Reconstruction Figures

If you want true inverse-reconstructed class images, run the MATLAB helper after dataset generation:

```matlab
cd matlab
setup_eidors()
generate_validation_reconstructions()
```

This creates random and mean reconstructed images for each class in `results/dataset_validation/reconstructions/`. The Python validator will use those figures when they are available.

## Running the Full Pipeline

After dataset generation, the following sequence reproduces all results:

```bash
# 1. Validate dataset integrity
uv run python python/validate_dataset.py

# 2. Log environment for reproducibility
uv run python python/log_environment.py

# 3. Run master experiment suite (recommended — single command)
uv run python python/run_all_experiments.py --seeds 5
```

### Master Experiment Runner (Recommended)

The master runner executes all experiments, generates figures, runs severity
sweeps, and produces a comprehensive Markdown report:

```bash
# Full suite: 3 datasets × 4 models × 8 conditions × 5 seeds + severity sweeps
uv run python python/run_all_experiments.py

# Quick validation run (fewer seeds)
uv run python python/run_all_experiments.py --seeds 3

# Specific datasets only
uv run python python/run_all_experiments.py --datasets raw pca

# Custom CNN training duration
uv run python python/run_all_experiments.py --epochs 300 --early-stopping-patience 50
```

**Training conditions executed by the master runner:**

| Condition | Training Data | Evaluation Data | Purpose |
|-----------|--------------|-----------------|---------|
| `clean_train_clean_eval` | Clean | Clean | Ceiling performance |
| `clean_train_noisy_eval` | Clean | Noisy | Vulnerability baseline |
| `noisy_train_noisy_eval` | Noisy (multi-severity augmented) | Noisy | Standard noise training |
| `noisy_train_clean_eval` | Noisy (multi-severity augmented) | Clean | Generalisation check |
| `augmented_train_noisy_eval` | Clean + online multi-severity noise | Noisy | Domain randomisation |
| `augmented_train_clean_eval` | Clean + online multi-severity noise | Clean | Augmented generalisation |
| `mixed_train_noisy_eval` | Mixed clean+noisy batches | Noisy | Best robustness regime |
| `mixed_train_clean_eval` | Mixed clean+noisy batches | Clean | Dual-domain performance |

**Key features of the updated pipeline:**

- **Multi-severity noise augmentation**: Severity sampled uniformly from [0.5, 2.0]
  each batch, preventing noise-level memorisation
- **Mixed clean+noisy training**: 30% clean / 70% noise-augmented per batch,
  maintaining clean-data representations while learning noise robustness
- **Stronger regularisation for noisy models**: Higher dropout (0.4), weight
  decay (1e-3), and label smoothing (0.05)
- **Severity sweep**: After main experiments, evaluates CNN models across
  severity levels [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] to assess generalisation

**Outputs:**

- `results/reports/experiment_report.md` — Full Markdown report with analysis
- `results/reports/all_results.csv` — Consolidated results table
- `results/reports/all_results.json` — Machine-readable results
- `results/reports/accuracy_pivot.csv` — Accuracy by model × condition
- `results/reports/f1_pivot.csv` — F1 by model × condition
- `results/reports/severity_sweep_results.json` — Severity generalisation curves
- `results/figures/<dataset>/<model>/<condition>/` — Confusion matrices, training curves

### Individual Script Usage

For targeted runs or debugging:

```bash
# Train CNN under both conditions
uv run python python/train.py --model cnn1d                # noisy training
uv run python python/train.py --model cnn1d --no-noise     # clean training

# Train all models under all conditions
uv run python python/run_all_experiments.py

# Evaluate with robustness + severity sweep
uv run python python/evaluate.py \
    --model-path results/models/cnn1d_noisy_best.pt --robustness

# Run ablation study (Python-side noise, no MATLAB dependency)
uv run python python/ablation.py --model cnn1d
uv run python python/ablation.py --model cnn1d --all-configs

# Statistical significance testing
uv run python python/statistical_tests.py --model cnn1d
uv run python python/statistical_tests.py --model random_forest
```

### Using Custom Datasets

All training and evaluation scripts support loading datasets from custom paths
via the `--data-path` CLI argument. This allows you to:

- Use alternative dataset files
- Evaluate on external datasets
- Test different dataset versions

**Example: Use a custom dataset file**

```bash
# Train CNN on custom dataset
uv run python python/train.py --model cnn1d --data-path /path/to/custom_dataset.mat

# Run all experiments on custom dataset
uv run python python/run_all_experiments.py --data-path /path/to/custom_dataset.mat

# Validate custom dataset
uv run python python/validate_dataset.py --data-path /path/to/custom_dataset.mat

# Evaluate model on custom dataset
uv run python python/evaluate.py \
    --model-path results/models/cnn1d_noisy_best.pt \
    --data-path /path/to/custom_dataset.mat

# Run statistical tests on custom dataset
uv run python python/statistical_tests.py --data-path /path/to/custom_dataset.mat

# Run architecture sweep on custom dataset
uv run python python/architecture_sweep.py --data-path /path/to/custom_dataset.mat

# Run ablation study on custom dataset
uv run python python/ablation.py --data-path /path/to/custom_dataset.mat
```

**Dataset Format Requirements**

Custom datasets must be in MATLAB `.mat` format with the following structure:

```matlab
% Original dataset format (from MATLAB generation):
dataset_X_clean    % (n_samples, n_features) clean measurements
dataset_X_noisy    % (n_samples, n_features) noisy measurements
dataset_y          % (n_samples, 1) class labels (1-indexed, 1-5 for 5 classes)

% Exported reduced-dataset format (from EDA notebook):
X_clean            % (n_samples, n_features) clean measurements (0-indexed labels)
X_noisy            % (n_samples, n_features) noisy measurements
y                  % (n_samples, 1) class labels (0-indexed, 0-4 for 5 classes)

% Alternative single-X format:
dataset_X          % Used if dataset_X_clean/dataset_X_noisy not present
dataset_y

% Supported class labels:
% 1 (original) or 0 (zero-indexed) = No contact        (baseline/reference)
% 2 (original) or 1 (zero-indexed) = Light touch
% 3 (original) or 2 (zero-indexed) = Firm press
% 4 (original) or 3 (zero-indexed) = Point contact
% 5 (original) or 4 (zero-indexed) = Distributed
```

**Using Dimensionality-Reduced Datasets**

The EDA notebook (`notebooks/eda_analysis.ipynb`) produces dimensionality-reduced
datasets using a leakage-safe preprocessing comparison, plus a decision log:

```bash
# PCA-reduced dataset (7 components) — suitable for shallow models
uv run python python/train.py --model svm --data-path data/cleaned/eit_cleaned_pca.mat
uv run python python/train.py --model random_forest --data-path data/cleaned/eit_cleaned_pca.mat

# LDA-reduced dataset (4 components, supervised) — best for shallow models
uv run python python/train.py --model svm --data-path data/cleaned/eit_cleaned_lda.mat
uv run python python/train.py --model mlp --data-path data/cleaned/eit_cleaned_lda.mat

# View selected preprocessing route and CV summary
cat data/cleaned/eda_decision_log.json
```

> **Note**: Primary training now uses `data/eit_dataset.mat` directly.
> The EDA notebook exports reduced variants only: `pca` (7 components),
> `lda` (4 components), and `umap` (2 components for visualisation).

**CNN vs. Non-CNN Models and Dataset Sizes**

- **CNN (1D-Conv1D)**: Requires at least 8 features due to 3 pooling layers
  - ✓ Works with: eit_dataset.mat (208 features)
  - ✗ Too small: eit_cleaned_pca.mat (7), eit_cleaned_lda.mat (4)

- **Non-CNN models** (SVM, Random Forest, MLP): Work with any dataset size
  - ✓ All reduced datasets work
  - Good for studying effect of dimensionality reduction

**If using H5PY-compatible `.mat` files** (newer MATLAB versions):
The loader automatically handles both scipy-compatible and HDF5-based `.mat`
files, so no additional configuration is needed.

## Configuration

All hyperparameters are centralised in `python/configs/config.yaml`:

| Section | Controls |
|---------|----------|
| `data` | Dataset path, noise toggle, split ratios |
| `cnn` | Channel sizes, FC dimension, dropout |
| `training` | Epochs, batch size, LR, early stopping |
| `baselines` | SVM/RF/MLP hyperparameters |
| `evaluation` | CV folds, noise levels, severity multipliers |
| `noise_augmentation` | Online noise injection params, severity range for domain randomisation |
| `seed` | Global random seed (42) |

CLI arguments override config values when provided.

### Online Noise Augmentation

The `noise_augmentation` section in `config.yaml` controls Python-side noise injection
during training. When `enabled: true`, clean data is augmented on-the-fly with the
4-component noise model. Key settings:

- `severity_range: [0.5, 2.0]` — sample severity uniformly per batch (multi-severity training)
- Set `severity_range: null` to use fixed severity of 1.0
- All noise parameters (SNR, contact impedance std, bias, ADC bits) match the MATLAB config

## Troubleshooting

### EIDORS `startup.m` not found

Ensure `setup_eidors.m` is on the MATLAB path and that `matlab/eidors-v3.12-ng/eidors/`
contains `startup.m`. Run from the `matlab/` directory.

### Netgen not found (3D models only)

3D cylindrical models require Netgen. If unavailable, use the default 2D circle:

```matlab
opts.geometry = '2d_circle';  % Default — does not require Netgen
```

### CUDA not detected by PyTorch

- Verify: `uv run python -c "import torch; print(torch.cuda.is_available())"`
- Ensure NVIDIA drivers and CUDA toolkit are installed
- Training works on CPU (slower but functional)

Note: ensure your installed PyTorch build matches your CUDA toolkit version
(this repository's tested configuration uses CUDA 12.6 / `cu126` wheels). If
you installed a `cu126` wheel, ensure the NVIDIA driver on your system
supports CUDA 12.6; otherwise install a PyTorch wheel that matches your
system CUDA or use the CPU-only wheel.

### Memory issues during dataset generation

- 25,000 samples × 208 features × 8 bytes × 2 (clean + noisy) ≈ 84 MB — should fit in 8 GB RAM
- If issues persist, reduce `samples_per_class` in `matlab/main.m`

### Import errors from Python scripts

All scripts must be run from the project root or from `python/`:

```bash
# From project root:
uv run python python/train.py --model cnn1d

# Or from python/ directory:
cd python
uv run train.py --model cnn1d
```
