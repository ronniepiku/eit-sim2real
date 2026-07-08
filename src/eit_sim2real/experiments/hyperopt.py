"""Unified hyperparameter optimisation and architecture search for EIT 1D-CNN.

Supports two modes:
  1. Architecture sweep (--mode=arch-sweep): Quick focused depth search with dev subset.
     Justifies the 3-block design choice. Uses single train/val split.
  2. Full grid search (--mode=grid-search, default): Comprehensive hyperparameter grid
     with k-fold CV, optimises for robustness (harmonic mean of clean & noisy F1).

Results are saved to results/hyperparameter_optimisation/ or results/architecture_sweep/.

Usage:
    # Architecture sweep (fast, exploratory)
    eit experiments hyperopt --mode=arch-sweep

    # Full grid search (comprehensive)
    eit experiments hyperopt --mode=grid-search
    eit experiments hyperopt --mode=grid-search --resume

    # Train final model with best config
    eit experiments hyperopt --mode=grid-search --final-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from eit_sim2real.configs import load_config
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.data.noise import (
    NoiseConfig,
    apply_noise_batch_vectorised,
    apply_noise_in_scaled_space,
)
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.train import train_cnn
from eit_sim2real.utils import (
    count_parameters,
    get_device,
    predict_cnn,
    rescale_cross_condition,
    set_seeds,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

# Full grid search space
FULL_SEARCH_SPACE = {
    "channels": [
        [32, 64, 128],  # Default (3 blocks)
        [64, 128, 256],  # Wider
        [32, 64, 128, 256],  # Deeper (4 blocks)
        [64, 128, 256, 512],  # Wider + deeper
    ],
    "fc_dim": [128, 256],
    "dropout": [0.2, 0.3, 0.5],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [32, 64, 128],
    "weight_decay": [1e-4, 1e-3],
    "noise_augmentation": [False, True],
}

# Architecture sweep space (focused depth search with fixed other hyperparameters)
ARCH_SWEEP_SPACE = {
    "channels": [
        [32, 64],  # 2 blocks
        [32, 64, 128],  # 3 blocks (default)
        [32, 64, 128, 256],  # 4 blocks
        [32, 64, 128, 256, 512],  # 5 blocks
    ],
    "fc_dim": [128],  # Fixed
    "dropout": [0.3],  # Fixed
    "learning_rate": [1e-3],  # Fixed
    "batch_size": [64],  # Fixed
    "weight_decay": [1e-4],  # Fixed
    "noise_augmentation": [False],  # No augmentation for clean-data search
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HParamConfig:
    """A single hyperparameter configuration."""

    channels: list[int]
    fc_dim: int
    dropout: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    noise_augmentation: bool

    @property
    def uid(self) -> str:
        """Deterministic hash for this configuration."""
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""

    config: HParamConfig
    fold_clean_f1: list[float] = field(default_factory=list)
    fold_noisy_f1: list[float] = field(default_factory=list)
    fold_robustness: list[float] = field(default_factory=list)
    mean_clean_f1: float = 0.0
    mean_noisy_f1: float = 0.0
    mean_robustness: float = 0.0
    std_robustness: float = 0.0
    training_time_s: float = 0.0
    n_params: int = 0
    epochs_used: list[int] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute mean/std from per-fold scores."""
        self.mean_clean_f1 = float(np.mean(self.fold_clean_f1))
        self.mean_noisy_f1 = (
            float(np.mean(self.fold_noisy_f1)) if self.fold_noisy_f1 else 0.0
        )
        self.mean_robustness = (
            float(np.mean(self.fold_robustness))
            if self.fold_robustness
            else self.mean_clean_f1
        )
        self.std_robustness = (
            float(np.std(self.fold_robustness))
            if len(self.fold_robustness) > 1
            else 0.0
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def harmonic_mean(a: float, b: float) -> float:
    """Compute harmonic mean of two values, returning 0 if either is 0."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def generate_grid(search_space: dict) -> list[HParamConfig]:
    """Generate all combinations from a search space."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    configs = []
    for combo in product(*values):
        params = dict(zip(keys, combo, strict=True))
        configs.append(HParamConfig(**params))
    return configs


# ---------------------------------------------------------------------------
# Training functions (shared by both modes)
# ---------------------------------------------------------------------------


def train_fold(
    hparams: HParamConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_clean: np.ndarray,
    X_val_noisy: np.ndarray | None,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int,
    early_stopping_patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    noise_config: NoiseConfig | None,
    input_scaler: object | None,
    severity_range: tuple[float, float] | None,
    device: torch.device,
    seed: int,
) -> tuple[float, float, int, int]:
    """Train a single fold and return (clean_f1, noisy_f1, epochs_used, n_params).

    Args:
        hparams: Hyperparameter configuration.
        X_train: Training features (scaled, clean).
        y_train: Training labels.
        X_val_clean: Validation features (clean, scaled).
        X_val_noisy: Validation features (noisy, scaled). If None, uses X_val_clean.
        y_val: Validation labels.
        n_classes: Number of output classes.
        epochs: Maximum training epochs.
        early_stopping_patience: Early stopping patience.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR scheduler reduction factor.
        noise_config: Noise config for online augmentation (if enabled).
        input_scaler: Scaler mapping the model input space to raw voltage space.
        severity_range: Severity range for domain randomisation.
        device: Torch device.
        seed: Random seed.

    Returns:
        Tuple of (clean_macro_f1, noisy_macro_f1, epochs_trained, n_params).
    """
    set_seeds(seed)
    n_features = X_train.shape[1]

    model = EITConv1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=hparams.channels,
        fc_dim=hparams.fc_dim,
        dropout=hparams.dropout,
    ).to(device)

    n_params = count_parameters(model)

    # Data loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=hparams.batch_size, shuffle=True, pin_memory=True
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    # Online noise augmentation
    augment = hparams.noise_augmentation and noise_config is not None
    aug_rng = np.random.default_rng(seed) if augment else None

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        for X_batch, y_batch in train_loader:
            if augment:
                X_np = X_batch.numpy()
                if severity_range is not None:
                    noise_config.severity = float(
                        aug_rng.uniform(severity_range[0], severity_range[1])
                    )
                if input_scaler is not None:
                    X_np = apply_noise_in_scaled_space(
                        X_np,
                        input_scaler,
                        noise_config,
                        rng=aug_rng,
                    )
                else:
                    X_np = apply_noise_batch_vectorised(
                        X_np,
                        noise_config,
                        rng=aug_rng,
                    )
                X_batch = torch.from_numpy(X_np).float()

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # --- Validation (on clean for early stopping) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            X_val_t = torch.from_numpy(X_val_clean).float().to(device)
            y_val_t = torch.from_numpy(y_val).long().to(device)
            logits = model(X_val_t)
            val_loss = criterion(logits, y_val_t).item()

        scheduler.step(val_loss)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            break

    # Load best model and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    model.eval()
    with torch.no_grad():
        # Clean evaluation
        X_clean_t = torch.from_numpy(X_val_clean).float().to(device)
        preds_clean = model(X_clean_t).argmax(dim=1).cpu().numpy()
        clean_f1 = f1_score(y_val, preds_clean, average="macro")

        # Noisy evaluation (if provided, else same as clean)
        if X_val_noisy is not None:
            X_noisy_t = torch.from_numpy(X_val_noisy).float().to(device)
            preds_noisy = model(X_noisy_t).argmax(dim=1).cpu().numpy()
            noisy_f1 = f1_score(y_val, preds_noisy, average="macro")
        else:
            noisy_f1 = clean_f1

    return float(clean_f1), float(noisy_f1), epochs_trained, n_params


# ---------------------------------------------------------------------------
# Architecture sweep mode
# ---------------------------------------------------------------------------


def run_architecture_sweep(
    X: np.ndarray,
    y: np.ndarray,
    dev_fraction: float = 0.1,
    epochs: int = 50,
    early_stopping_patience: int = 20,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    seed: int = 42,
    output_dir: Path = Path("results/architecture_sweep"),
) -> Path:
    """Run architecture sweep (focused depth search with fixed other hyperparameters).

    Uses a dev subset for efficiency and single train/val split (no k-fold).
    Only evaluates on clean data (no noise augmentation).

    Args:
        X: Feature matrix.
        y: Label vector.
        dev_fraction: Fraction of training data to use (development subset).
        epochs: Max epochs per trial.
        early_stopping_patience: Early stopping patience.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR scheduler factor.
        seed: Random seed.
        output_dir: Directory for saving results.

    Returns:
        Path to the saved CSV with results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device())
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seeds(seed)

    # Load and split data
    dataset = prepare_splits(X, y, random_state=seed)

    # Use development subset for efficient sweep
    n_dev = int(len(dataset.X_train) * dev_fraction)
    indices = np.random.permutation(len(dataset.X_train))[:n_dev]
    X_train_dev = dataset.X_train[indices]
    y_train_dev = dataset.y_train[indices]

    logger.info(
        "Architecture sweep: %d training samples, full validation set (%d samples)",
        n_dev,
        len(dataset.X_val),
    )

    # Generate grid and run trials
    grid = generate_grid(ARCH_SWEEP_SPACE)
    n_classes = len(np.unique(y))
    results = []

    for trial_idx, hparams in enumerate(grid):
        logger.info(
            f"[{trial_idx + 1}/{len(grid)}] "
            f"channels={hparams.channels} ({len(hparams.channels)} blocks)"
        )

        clean_f1, _, epochs_used, n_params = train_fold(
            hparams=hparams,
            X_train=X_train_dev,
            y_train=y_train_dev,
            X_val_clean=dataset.X_val,
            X_val_noisy=None,
            y_val=dataset.y_val,
            n_classes=n_classes,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            noise_config=None,
            severity_range=None,
            device=device,
            seed=seed,
        )

        results.append(
            {
                "n_blocks": len(hparams.channels),
                "channels": str(hparams.channels),
                "n_params": n_params,
                "clean_f1": clean_f1,
                "epochs_used": epochs_used,
            }
        )

        logger.info(
            "  Val F1: %.4f | Params: %s | Epochs: %d",
            clean_f1,
            f"{n_params:,}",
            epochs_used,
        )

    # Save results
    df = pd.DataFrame(results)
    output_path = (output_dir / "architecture_sweep.csv").resolve()
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    # Print summary
    summary = df[["n_blocks", "channels", "n_params", "clean_f1", "epochs_used"]].copy()
    logger.info("\nArchitecture sweep summary:\n%s", summary.to_string(index=False))

    # Report best
    best = df.loc[df["clean_f1"].idxmax()]
    logger.info(
        "Best architecture: %d blocks (F1: %.4f, params: %s)",
        int(best["n_blocks"]),
        best["clean_f1"],
        f"{int(best['n_params']):,}",
    )

    return output_path


# ---------------------------------------------------------------------------
# Full grid search mode
# ---------------------------------------------------------------------------


def run_grid_search(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    noise_config: NoiseConfig | None = None,
    severity_range: tuple[float, float] | None = None,
    seed: int = 42,
    output_dir: Path = Path("results/hyperparameter_optimisation"),
    resume: bool = False,
) -> tuple[pd.DataFrame, HParamConfig, TrialResult]:
    """Run full grid search with cross-validation.

    Optimises for robustness: harmonic mean of clean-F1 and noisy-F1.

    Args:
        X_clean: Clean feature matrix (unscaled).
        X_noisy: Noisy feature matrix (unscaled).
        y: Label vector.
        n_folds: Number of CV folds.
        epochs: Max epochs per trial.
        early_stopping_patience: Early stopping patience.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR scheduler factor.
        noise_config: Noise config for online augmentation trials.
        severity_range: Severity range for domain randomisation.
        seed: Master random seed.
        output_dir: Directory for saving results.
        resume: If True, skip already-completed trials.

    Returns:
        Tuple of (results_df, best_config, best_result).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "grid_search_results.csv"
    checkpoint_path = output_dir / "grid_search_checkpoint.json"

    device = torch.device(get_device())
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    grid = generate_grid(FULL_SEARCH_SPACE)
    n_total = len(grid)
    logger.info(f"Grid search: {n_total} configurations x {n_folds} folds")

    # Load completed trials for resume
    completed_uids: set[str] = set()
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            completed_uids = set(json.load(f).get("completed", []))
        logger.info(f"Resuming: {len(completed_uids)} trials already completed")

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(X_clean, y))
    n_classes = len(np.unique(y))

    all_results: list[TrialResult] = []
    best_robustness = -1.0
    best_result: TrialResult | None = None
    best_config: HParamConfig | None = None

    for trial_idx, hparams in enumerate(grid):
        if hparams.uid in completed_uids:
            continue

        logger.info(
            f"[{trial_idx + 1}/{n_total}] "
            f"channels={hparams.channels}, fc={hparams.fc_dim}, "
            f"drop={hparams.dropout}, lr={hparams.learning_rate}, "
            f"bs={hparams.batch_size}, wd={hparams.weight_decay}, "
            f"aug={hparams.noise_augmentation}"
        )

        trial = TrialResult(config=hparams)
        t_start = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Split and scale (fit scaler on train fold only)
            scaler = RobustScaler()
            X_train_fold = scaler.fit_transform(X_clean[train_idx])
            X_val_clean_fold = scaler.transform(X_clean[val_idx])
            X_val_noisy_fold = (
                scaler.transform(X_noisy[val_idx]) if X_noisy is not None else None
            )
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            fold_seed = seed + fold_idx

            clean_f1, noisy_f1, epochs_used, n_params = train_fold(
                hparams=hparams,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val_clean=X_val_clean_fold,
                X_val_noisy=X_val_noisy_fold,
                y_val=y_val_fold,
                n_classes=n_classes,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                noise_config=noise_config,
                input_scaler=scaler,
                severity_range=severity_range,
                device=device,
                seed=fold_seed,
            )

            robustness = harmonic_mean(clean_f1, noisy_f1)
            trial.fold_clean_f1.append(clean_f1)
            trial.fold_noisy_f1.append(noisy_f1)
            trial.fold_robustness.append(robustness)
            trial.epochs_used.append(epochs_used)
            trial.n_params = n_params

            logger.info(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"clean_F1={clean_f1:.4f}, noisy_F1={noisy_f1:.4f}, "
                f"robustness={robustness:.4f} (epochs={epochs_used})"
            )

        trial.training_time_s = time.time() - t_start
        trial.compute_aggregates()

        logger.info(
            f"  → Mean robustness: {trial.mean_robustness:.4f} "
            f"(±{trial.std_robustness:.4f}), "
            f"clean={trial.mean_clean_f1:.4f}, noisy={trial.mean_noisy_f1:.4f}, "
            f"time={trial.training_time_s:.1f}s"
        )

        all_results.append(trial)

        # Track best
        if trial.mean_robustness > best_robustness:
            best_robustness = trial.mean_robustness
            best_result = trial
            best_config = hparams
            logger.info(f"  ★ New best robustness: {best_robustness:.4f}")

        # Save checkpoint incrementally
        completed_uids.add(hparams.uid)
        _save_checkpoint(checkpoint_path, completed_uids)
        _save_results_csv(results_path, all_results)

    # Final report
    results_df = _build_results_dataframe(all_results)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    return results_df, best_config, best_result


def _save_checkpoint(path: Path, completed_uids: set[str]) -> None:
    """Save checkpoint of completed trial UIDs."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed": list(completed_uids)}, f)


def _save_results_csv(path: Path, results: list[TrialResult]) -> None:
    """Save current results to CSV."""
    df = _build_results_dataframe(results)
    df.to_csv(path, index=False)


def _build_results_dataframe(results: list[TrialResult]) -> pd.DataFrame:
    """Convert trial results to a DataFrame."""
    rows = []
    for trial in results:
        rows.append(
            {
                "uid": trial.config.uid,
                "channels": str(trial.config.channels),
                "fc_dim": trial.config.fc_dim,
                "dropout": trial.config.dropout,
                "learning_rate": trial.config.learning_rate,
                "batch_size": trial.config.batch_size,
                "weight_decay": trial.config.weight_decay,
                "noise_augmentation": trial.config.noise_augmentation,
                "n_params": trial.n_params,
                "mean_clean_f1": trial.mean_clean_f1,
                "mean_noisy_f1": trial.mean_noisy_f1,
                "mean_robustness": trial.mean_robustness,
                "std_robustness": trial.std_robustness,
                "training_time_s": trial.training_time_s,
                "mean_epochs": float(np.mean(trial.epochs_used)),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("mean_robustness", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final model training
# ---------------------------------------------------------------------------


def train_final_model(
    best_config: HParamConfig,
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    noise_config: NoiseConfig | None = None,
    severity_range: tuple[float, float] | None = None,
    seed: int = 42,
    output_dir: Path = Path("results/hyperparameter_optimisation"),
) -> Path:
    """Train the final model with the best hyperparameters on the full train split.

    Uses a held-out test set for final evaluation. The model is saved along
    with its configuration and evaluation metrics.

    Returns:
        Path to the saved model checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device())
    set_seeds(seed)

    # Prepare train/val/test splits
    dataset = prepare_splits(X_clean, y, random_state=seed, scaler_type="robust")

    # Reconstruct full index mapping for noisy test set
    _, X_noisy_test_split, _, _ = train_test_split(
        X_noisy, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_test_noisy_scaled = dataset.scaler.transform(X_noisy_test_split)

    n_classes = len(np.unique(y))
    n_features = dataset.X_train.shape[1]

    logger.info(f"Training final model with config: {asdict(best_config)}")
    logger.info(
        f"Train: {len(dataset.y_train)}, Val: {len(dataset.y_val)}, Test: {len(dataset.y_test)}"
    )

    model = EITConv1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=best_config.channels,
        fc_dim=best_config.fc_dim,
        dropout=best_config.dropout,
    ).to(device)

    # Data loaders
    train_ds = TensorDataset(
        torch.from_numpy(dataset.X_train).float(),
        torch.from_numpy(dataset.y_train).long(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=best_config.batch_size, shuffle=True, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config.learning_rate,
        weight_decay=best_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    # Online augmentation
    augment = best_config.noise_augmentation and noise_config is not None
    aug_rng = np.random.default_rng(seed) if augment else None

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_clean_f1": [], "val_noisy_f1": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            if augment:
                X_np = X_batch.numpy()
                if severity_range is not None:
                    noise_config.severity = float(
                        aug_rng.uniform(severity_range[0], severity_range[1])
                    )
                X_np = apply_noise_in_scaled_space(
                    X_np,
                    dataset.scaler,
                    noise_config,
                    rng=aug_rng,
                )
                X_batch = torch.from_numpy(X_np).float()

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.from_numpy(dataset.X_val).float().to(device)
            y_val_t = torch.from_numpy(dataset.y_val).long().to(device)
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {epoch_loss / n_batches:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Clean test
        X_test_t = torch.from_numpy(dataset.X_test).float().to(device)
        preds_clean = model(X_test_t).argmax(dim=1).cpu().numpy()
        clean_f1 = f1_score(dataset.y_test, preds_clean, average="macro")

        # Noisy test
        X_noisy_t = torch.from_numpy(X_test_noisy_scaled).float().to(device)
        preds_noisy = model(X_noisy_t).argmax(dim=1).cpu().numpy()
        noisy_f1 = f1_score(dataset.y_test, preds_noisy, average="macro")

    robustness = harmonic_mean(clean_f1, noisy_f1)
    logger.info(
        f"Final model: clean_F1={clean_f1:.4f}, noisy_F1={noisy_f1:.4f}, "
        f"robustness={robustness:.4f}"
    )

    # Save model and metadata
    model_path = output_dir / "cnn1d_optimised_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(best_config),
            "n_features": n_features,
            "n_classes": n_classes,
            "metrics": {
                "clean_f1": clean_f1,
                "noisy_f1": noisy_f1,
                "robustness": robustness,
            },
            "training": {
                "epochs_trained": epoch + 1,
                "best_val_loss": best_val_loss,
            },
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = output_dir / "final_model_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save comprehensive report
    report = {
        "best_hyperparameters": asdict(best_config),
        "final_metrics": {
            "clean_macro_f1": clean_f1,
            "noisy_macro_f1": noisy_f1,
            "robustness_score": robustness,
        },
        "model_info": {
            "n_parameters": count_parameters(model),
            "n_features": n_features,
            "n_classes": n_classes,
            "architecture": f"EITConv1D(channels={best_config.channels}, "
            f"fc_dim={best_config.fc_dim}, dropout={best_config.dropout})",
        },
        "training_details": {
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "early_stopping_patience": early_stopping_patience,
            "noise_augmentation": best_config.noise_augmentation,
        },
    }
    report_path = output_dir / "optimisation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    return model_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified hyperparameter optimisation and architecture search for EIT 1D-CNN."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (matches train.py).",
    )
    parser.add_argument(
        "--mode",
        choices=["arch-sweep", "grid-search"],
        default="grid-search",
        help="Search mode: 'arch-sweep' for focused depth search, "
        "'grid-search' for comprehensive tuning (default: grid-search).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset .mat file (default: from config.yaml).",
    )
    parser.add_argument(
        "--noise/--no-noise",
        dest="noise",
        action="store_true",
        default=True,
        help="Train with noise augmentation (matches train.py, default: True).",
    )
    parser.add_argument(
        "--no-noise",
        dest="noise",
        action="store_false",
        help="Train without noise augmentation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: results/architecture_sweep or "
        "results/hyperparameter_optimisation based on mode).",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data for architecture sweep dev subset (default: 0.1).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds for grid search (default: 3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs (default: from training.epochs in config for grid-search).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume grid search from checkpoint (skip completed trials).",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip search, train final model from existing results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config).",
    )
    return parser.parse_args()


def run_hyperparameter_sensitivity(
    data_path: Path,
    seed: int = 42,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports"),
    figures_dir: Path = Path("results/figures"),
) -> dict:
    """Evaluate CNN sensitivity to key training hyperparameters.

    Tests that results aren't artefacts of specific hyperparameter choices
    by sweeping: learning rate, dropout, weight decay, severity range.
    """
    logger.info("── Hyperparameter Sensitivity ──")
    device = get_device()
    noise_cfg = NoiseConfig()

    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)

    ds_clean = prepare_splits(X_clean, y, random_state=seed)
    ds_noisy = prepare_splits(X_noisy, y, random_state=seed)

    # Noisy test data rescaled into clean-scaler space for cross-condition eval
    X_noisy_test_clean_space = rescale_cross_condition(
        ds_noisy.X_test, ds_noisy.scaler, ds_clean.scaler
    )

    results = {}

    # Define sweeps: (param_name, values, default, label)
    hp_sweeps = {
        "learning_rate": {
            "values": [5e-4, 1e-3, 2e-3, 5e-3],
            "default": 1e-3,
            "label": "Learning Rate",
        },
        "dropout": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5],
            "default": 0.4,
            "label": "Dropout",
        },
        "weight_decay": {
            "values": [1e-5, 1e-4, 1e-3, 1e-2],
            "default": 1e-3,
            "label": "Weight Decay",
        },
        "severity_range_max": {
            "values": [1.0, 1.5, 2.0, 2.5, 3.0],
            "default": 2.0,
            "label": "Max Severity (training)",
        },
    }

    for param_name, sweep_info in hp_sweeps.items():
        accs_noisy = []
        accs_clean = []

        for val in sweep_info["values"]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            # Build training kwargs
            train_kwargs = {
                "epochs": epochs,
                "early_stopping_patience": early_stopping_patience,
                "device": device,
                "noise_config": noise_cfg,
                "severity_range": (0.5, 2.0),
                "weight_decay": 1e-3,
                "dropout": 0.4,
                "label_smoothing": 0.05,
            }

            # Override the swept parameter
            if param_name == "learning_rate":
                train_kwargs["lr"] = val
            elif param_name == "dropout":
                train_kwargs["dropout"] = val
            elif param_name == "weight_decay":
                train_kwargs["weight_decay"] = val
            elif param_name == "severity_range_max":
                train_kwargs["severity_range"] = (0.5, val)

            model, _ = train_cnn(
                ds_clean.X_train,
                ds_clean.y_train,
                ds_clean.X_val,
                ds_clean.y_val,
                **train_kwargs,
                input_scaler=ds_clean.scaler,
            )

            y_pred_noisy = predict_cnn(model, X_noisy_test_clean_space, device)
            y_pred_clean = predict_cnn(model, ds_clean.X_test, device)
            accs_noisy.append(float(accuracy_score(ds_noisy.y_test, y_pred_noisy)))
            accs_clean.append(float(accuracy_score(ds_clean.y_test, y_pred_clean)))

        results[param_name] = {
            "values": sweep_info["values"],
            "accuracies_noisy": accs_noisy,
            "accuracies_clean": accs_clean,
            "label": sweep_info["label"],
            "default": sweep_info["default"],
        }
        logger.info(
            f"  {sweep_info['label']}: noisy range "
            f"[{min(accs_noisy):.3f}, {max(accs_noisy):.3f}]"
        )

    # Generate figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, (param_name, data) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        values = data["values"]
        x_pos = range(len(values))

        ax.plot(
            x_pos,
            data["accuracies_noisy"],
            "o-",
            label="Noisy eval",
            linewidth=2,
            color="tab:blue",
        )
        ax.plot(
            x_pos,
            data["accuracies_clean"],
            "s--",
            label="Clean eval",
            linewidth=2,
            color="tab:green",
        )

        # Mark default
        if data["default"] in values:
            def_idx = values.index(data["default"])
            ax.axvline(def_idx, color="red", linestyle=":", alpha=0.5)
            ax.scatter(
                [def_idx],
                [data["accuracies_noisy"][def_idx]],
                color="red",
                s=80,
                zorder=5,
                marker="*",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{v}" for v in values], fontsize=9)
        ax.set_xlabel(data["label"])
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Sensitivity: {data['label']}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "CNN Hyperparameter Sensitivity (Augmented Training)", fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig_path = figures_dir / "hyperparameter_sensitivity.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    out_path = output_dir / "hyperparameter_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {out_path}")
    return results


def main() -> None:
    """Main entry point for unified hyperparameter optimisation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Determine output directory based on mode
    if args.output_dir is None:
        if args.mode == "arch-sweep":
            args.output_dir = Path("results/architecture_sweep")
        else:
            args.output_dir = Path("results/hyperparameter_optimisation")

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_path = args.data_path or Path(cfg["data"]["path"])

    # Trigger device banner early
    get_device()

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Noise augmentation: {'enabled' if args.noise else 'disabled'}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {args.output_dir}")

    if args.mode == "arch-sweep":
        # --- Architecture Sweep Mode ---
        logger.info("Running architecture sweep (focused depth search)...")
        X, y = load_mat_dataset(data_path, use_noisy=False)

        epochs = args.epochs or 50
        run_architecture_sweep(
            X=X,
            y=y,
            dev_fraction=args.dev_fraction,
            epochs=epochs,
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
            scheduler_patience=cfg["training"]["scheduler_patience"],
            scheduler_factor=cfg["training"]["scheduler_factor"],
            seed=seed,
            output_dir=args.output_dir,
        )

    else:  # grid-search
        # --- Grid Search Mode ---
        logger.info("Running full grid search (comprehensive hyperparameter tuning)...")

        # Load data
        X_clean, y = load_mat_dataset(data_path, use_noisy=False)
        if args.noise:
            X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)
        else:
            X_noisy = None  # Will skip noise augmentation in grid search

        logger.info(f"Dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

        # Setup noise config only if noise is enabled
        if args.noise:
            noise_cfg_section = cfg.get("noise_augmentation", {})
            noise_config = NoiseConfig(
                enabled=True,
                snr_db=noise_cfg_section.get("snr_db", 40.0),
                noise_floor=noise_cfg_section.get("noise_floor", 1e-4),
                contact_impedance_std_percent=noise_cfg_section.get(
                    "contact_impedance_std_percent", 10.0
                ),
                max_bias=noise_cfg_section.get("max_bias", 0.02),
                adc_bits=noise_cfg_section.get("adc_bits", 16),
                voltage_range=noise_cfg_section.get("voltage_range", 1.0),
                n_electrodes=noise_cfg_section.get("n_electrodes", 16),
            )
            severity_range_cfg = noise_cfg_section.get("severity_range")
            severity_range = (
                tuple(severity_range_cfg) if severity_range_cfg else (0.5, 2.0)
            )
        else:
            noise_config = None
            severity_range = None

        if not args.final_only:
            # Run grid search
            epochs = args.epochs or cfg["training"]["epochs"]
            results_df, best_config, best_result = run_grid_search(
                X_clean=X_clean,
                X_noisy=X_noisy,
                y=y,
                n_folds=args.n_folds,
                epochs=epochs,
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                scheduler_patience=cfg["training"]["scheduler_patience"],
                scheduler_factor=cfg["training"]["scheduler_factor"],
                noise_config=noise_config,
                severity_range=severity_range,
                seed=seed,
                output_dir=args.output_dir,
                resume=args.resume,
            )

            logger.info(f"\n{'=' * 60}")
            logger.info("GRID SEARCH COMPLETE")
            logger.info(f"{'=' * 60}")
            logger.info(f"Best config: {asdict(best_config)}")
            if args.noise:
                logger.info(
                    f"Best robustness: {best_result.mean_robustness:.4f} "
                    f"(clean={best_result.mean_clean_f1:.4f}, noisy={best_result.mean_noisy_f1:.4f})"
                )
            else:
                logger.info(f"Best clean F1: {best_result.mean_clean_f1:.4f}")
        else:
            # Load best config from existing results
            results_path = args.output_dir / "grid_search_results.csv"
            if not results_path.exists():
                raise FileNotFoundError(
                    f"No results found at {results_path}. Run grid search first."
                )
            results_df = pd.read_csv(results_path)
            best_row = results_df.iloc[0]
            best_config = HParamConfig(
                channels=json.loads(best_row["channels"].replace("'", '"')),
                fc_dim=int(best_row["fc_dim"]),
                dropout=float(best_row["dropout"]),
                learning_rate=float(best_row["learning_rate"]),
                batch_size=int(best_row["batch_size"]),
                weight_decay=float(best_row["weight_decay"]),
                noise_augmentation=bool(best_row["noise_augmentation"]),
            )
            logger.info(f"Loaded best config from {results_path}")

        # Train final model only if noise is enabled
        if args.noise:
            logger.info("\nTraining final optimised model...")
            model_path = train_final_model(
                best_config=best_config,
                X_clean=X_clean,
                X_noisy=X_noisy,
                y=y,
                epochs=cfg["training"]["epochs"],
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                scheduler_patience=cfg["training"]["scheduler_patience"],
                scheduler_factor=cfg["training"]["scheduler_factor"],
                noise_config=noise_config,
                severity_range=severity_range,
                seed=seed,
                output_dir=args.output_dir,
            )

            logger.info(f"\nOptimised model saved to: {model_path}")
            logger.info("Run `python evaluate.py` with this model for full evaluation.")


if __name__ == "__main__":
    main()
