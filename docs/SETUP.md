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
- **Runtime**: torch, numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, pyyaml, joblib
- **Dev**: pytest, ruff, mypy, jupyter, ipykernel

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
# Run test suite (should pass all 24 tests)
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
1. Download from: http://eidors3d.sourceforge.net/download.shtml
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

If you want true inverse-reconstructed class images for the dissertation report, run the MATLAB helper after dataset generation:

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

# 3. Train CNN under both conditions
uv run python python/train.py --model cnn1d                # noisy training
uv run python python/train.py --model cnn1d --no-noise     # clean training

# 4. Train all baselines under 4 conditions
uv run python python/run_baselines.py

# 5. Evaluate with robustness + severity sweep
uv run python python/evaluate.py \
    --model-path results/models/cnn1d_noisy_best.pt --robustness

# 6. Run ablation study
uv run python python/ablation.py --model cnn1d

# 7. Statistical significance testing
uv run python python/statistical_tests.py --model cnn1d
uv run python python/statistical_tests.py --model random_forest
```

## Configuration

All hyperparameters are centralised in `python/configs/config.yaml`:

| Section | Controls |
|---------|----------|
| `data` | Dataset path, noise toggle, split ratios |
| `cnn` | Channel sizes, FC dimension, dropout |
| `training` | Epochs, batch size, LR, early stopping |
| `baselines` | SVM/RF/MLP hyperparameters |
| `evaluation` | CV folds, noise levels, severity multipliers |
| `seed` | Global random seed (42) |

CLI arguments override config values when provided.

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
