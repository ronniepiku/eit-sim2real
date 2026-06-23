# CLI Usage Guide

This document provides comprehensive documentation for the `eit` command-line interface.

## Installation

The `eit` command is installed when you install the package:

```bash
uv pip install -e .
eit --version
```

## Global Options

All commands support:

```bash
eit --version        # Show package version
eit --help          # Show general help
eit COMMAND --help  # Show help for specific command
```

## Commands Overview

### `eit train` — Model Training

Train neural networks and baseline classifiers on EIT data.

#### Syntax

```bash
eit train {cnn,baselines} [OPTIONS]
```

#### CNN Training

```bash
# Train 1D-CNN with noise augmentation (default)
eit train cnn

# Train on clean data only (no noise augmentation)
eit train cnn --no-noise

# Custom configuration
eit train cnn --config path/to/config.yaml

# Specify output directory
eit train cnn --output-dir results/custom_models

# Resume training from checkpoint
eit train cnn --resume-from path/to/checkpoint.pt

# Custom hyperparameters
eit train cnn --epochs 300 --batch-size 32 --learning-rate 0.001
```

#### Baseline Training

```bash
# Train all baselines (SVM, RF, MLP)
eit train baselines

# Train single baseline
eit train baselines --model svm
eit train baselines --model random_forest
eit train baselines --model mlp
```

### `eit evaluate` — Model Evaluation

Evaluate trained models on validation/test datasets.

#### Syntax

```bash
eit evaluate [OPTIONS]
```

#### Examples

```bash
# Evaluate a trained CNN
eit evaluate --model-path results/models/cnn1d_noisy_best.pt

# Evaluate with custom config
eit evaluate --model-path results/models/cnn1d_noisy_best.pt --config path/to/config.yaml

# Evaluate with test-time noise augmentation
eit evaluate --model-path results/models/cnn1d_clean_best.pt --noise gaussian

# Gaussian-only evaluation (SNR sweep)
eit evaluate --gaussian-only

# Specify output directory
eit evaluate --model-path results/models/cnn1d_noisy_best.pt --output-dir results/evaluation
```

### `eit experiments` — Experiment Pipelines

Run comprehensive experiment suites for research.

#### Master Command: `run-all`

Run all experiment types with selective enabling/disabling.

##### Syntax

```bash
eit experiments run-all [OPTIONS]
```

##### Default Behavior

By default, runs:
1. **Grid** — Model × Dataset × Condition combinations
2. **Ablation** — Noise component analysis
3. **Hyperopt** — Hyperparameter grid search
4. **Architecture-sweep** — CNN depth optimization
5. **Extended** — Statistical tests, robustness analysis, calibration
6. **Additional** — Memorization experiments

##### Examples

```bash
# Run all experiments (full suite)
eit experiments run-all

# Skip specific experiment types
eit experiments run-all --skip-grid                    # Skip grid, run others
eit experiments run-all --skip-hyperopt --skip-ablation  # Run only grid, extended, additional
eit experiments run-all --skip-grid --skip-ablation    # Run hyperopt, extended, additional only

# Include optional mesh refinement (requires fine-mesh dataset)
eit experiments run-all --include-mesh-refinement

# Combine skip flags
eit experiments run-all --skip-additional --skip-architecture-sweep

# With custom config
eit experiments run-all --config path/to/config.yaml
```

#### Individual Experiment Commands

##### `ablation` — Noise Component Ablation

Analyze the impact of each noise component through single-component and leave-one-out studies.

```bash
# Default (5 seeds, CNN model)
eit experiments ablation

# Custom parameters
eit experiments ablation --model cnn1d --n-seeds 10 --seed 42

# Test other models
eit experiments ablation --model svm
eit experiments ablation --model random_forest
eit experiments ablation --model mlp

# Skip severity sweep
eit experiments ablation --no-severity-sweep
```

##### `hyperopt` — Hyperparameter Optimization

Search hyperparameter space using grid search or architecture sweep.

```bash
# Full grid search (comprehensive, ~200 configs × k-fold CV)
eit experiments hyperopt --mode grid-search

# Resume interrupted grid search
eit experiments hyperopt --mode grid-search --resume

# Quick architecture depth sweep (2-5 conv blocks)
eit experiments hyperopt --mode arch-sweep

# Train final model with best hyperparameters
eit experiments hyperopt --mode grid-search --final-only
```

##### `architecture-sweep` — CNN Depth Search

Fast focused search over CNN depths (2-5 convolutional blocks).

```bash
eit experiments architecture-sweep
```

Equivalent to: `eit experiments hyperopt --mode arch-sweep`

##### `extended` — Extended Analyses

Run all extended robustness and analysis experiments.

```bash
# Default (5 seeds, 200 epochs)
eit experiments extended

# Custom parameters
eit experiments extended --seeds 10 --seed 42 --epochs 250

# Custom output paths
eit experiments extended --output-dir results/custom --figures-dir figures/custom

# Early stopping patience
eit experiments extended --early-stopping-patience 50
```

This command runs:
1. Statistical tests (paired t-tests across conditions)
2. Dataset size effects (learning curves)
3. Noise-type severity sweep (per-component analysis)
4. Gaussian-only evaluation (SNR sweep)
5. Confidence calibration (ECE/MCE analysis)
6. Per-class robustness (class-wise degradation)
7. Noise parameter sensitivity (4-parameter sweep)
8. Hyperparameter sensitivity (CNN tuning analysis)

##### `additional` — Memorization Experiments

Run additional memorization and edge-case studies.

```bash
eit experiments additional
```

Includes:
- Fixed-bias experiment (memorization when class distribution is biased)
- Different-draw experiment (out-of-distribution test)

##### `mesh-refinement` — Cross-Mesh Evaluation

Evaluate models trained on one mesh on data from a finer mesh.

```bash
# Requires fine-mesh dataset
eit experiments mesh-refinement

# Custom paths
eit experiments mesh-refinement \
  --baseline-dataset data/eit_dataset.mat \
  --fine-mesh-dataset data/eit_dataset_mesh_f.mat \
  --noisy-model results/models/cnn1d_noisy_best.pt \
  --clean-model results/models/cnn1d_clean_best.pt
```

### `eit validate-dataset` — Dataset Validation

Validate and analyze dataset integrity.

```bash
# Validate default dataset
eit validate-dataset

# Validate custom dataset
eit validate-dataset --dataset-path data/custom_dataset.mat

# Generate EDA report
eit validate-dataset --generate-eda
```

### `eit log-environment` — Environment Information

Log system and package version information.

```bash
eit log-environment

# Saves to results/environment.json
```

## Configuration

Most commands support `--config` to override defaults:

```bash
eit train cnn --config config_custom.yaml
eit evaluate --config config_custom.yaml --model-path results/models/cnn1d_noisy_best.pt
```

Config file (`config.yaml`) controls:
- Dataset path
- Noise model parameters
- Default hyperparameters
- Output paths

See `src/eit_sim2real/configs/config.yaml` for full structure.

## Output Structure

Experiments generate results in predictable locations:

```
results/
├── models/
│   ├── cnn1d_clean_best.pt              # Best CNN on clean data
│   ├── cnn1d_noisy_best.pt              # Best CNN with augmentation
│   ├── svm_{clean,noisy}.joblib
│   ├── random_forest_{clean,noisy}.joblib
│   └── mlp_{clean,noisy}.joblib
├── reports/
│   ├── grid_results.json                # Grid experiment results
│   ├── ablation_results.csv             # Ablation study summary
│   ├── dataset_size_results.json        # Learning curves
│   ├── all_results.csv                  # Comprehensive results table
│   └── *.md                             # Generated markdown reports
├── figures/
│   ├── training_curves_*.png
│   ├── confusion_matrices_*.png
│   ├── robustness_*.png
│   ├── calibration_*.png
│   └── ...
├── additional_experiments/
│   ├── fixed_bias/                      # Fixed-bias results
│   ├── different_draw/                  # Different-draw results
│   └── mesh_refinement/                 # Cross-mesh evaluation
├── hyperparameter_optimisation/
│   ├── grid_search_results.csv
│   ├── best_config.json
│   └── ...
├── architecture_sweep/
│   ├── arch_sweep_results.json
│   └── ...
└── environment.json                     # System info at runtime
```

## Reproducibility

To ensure reproducible results:

1. **Fix random seeds** (already done in `config.yaml`):
   ```bash
   eit train cnn --seed 42
   eit experiments ablation --seed 42
   ```

2. **Use version-controlled datasets**:
   ```bash
   git lfs pull  # If using Git LFS for large datasets
   ```

3. **Log environment**:
   ```bash
   eit log-environment  # Saves to results/environment.json
   ```

4. **Document configuration**:
   ```bash
   cp config.yaml results/config_used.yaml
   ```

## Troubleshooting

### Model Not Found

```bash
eit evaluate --model-path results/models/cnn1d_noisy_best.pt
# Error: File not found
# → Run `eit train cnn` first to generate models
```

### Dataset Not Found

```bash
eit train cnn
# Error: data/eit_dataset.mat not found
# → Generate dataset via MATLAB scripts, or use `--config` with custom path
```

### GPU Not Available

```bash
eit train cnn
# Warning: CUDA not available, using CPU
# → Install CUDA-enabled PyTorch: see docs/SETUP.md
```

### Out of Memory

```bash
# Reduce batch size
eit train cnn --batch-size 32

# Use smaller dataset
eit train cnn --config config_small.yaml

# Reduce model complexity (fewer channels)
# → Edit config.yaml manually
```

## Advanced Usage

### Custom Hyperparameter Search

```bash
# Via config modification (see pyproject.toml)
eit experiments hyperopt --mode grid-search --final-only
```

### Batch Processing Multiple Configs

```bash
for config in configs/*.yaml; do
  eit train cnn --config "$config" --output-dir "results/$(basename $config .yaml)"
done
```

### Parallel Experiment Runs

```bash
# Run grid and ablation in parallel (requires joblib support)
eit experiments run-all --skip-hyperopt --skip-extended &
eit experiments hyperopt --mode arch-sweep &
wait
```

## Performance Tips

1. **Use GPU for CNN training**:
   ```bash
   # Install CUDA-enabled PyTorch (see SETUP.md)
   eit train cnn  # ~10x faster on NVIDIA GPU
   ```

2. **Resume interrupted runs**:
   ```bash
   eit experiments hyperopt --mode grid-search --resume
   ```

3. **Skip slow experiments**:
   ```bash
   # Skip mesh refinement (requires fine-mesh dataset)
   eit experiments run-all --skip-mesh-refinement
   ```

4. **Reduce seeds for exploration**:
   ```bash
   eit experiments ablation --n-seeds 3  # Default is 5
   ```

## Getting Help

```bash
# General help
eit --help

# Command-specific help
eit train --help
eit experiments --help
eit experiments run-all --help

# Report issues
# → GitHub: https://github.com/ronniepiku/eit-sim2real/issues
```
