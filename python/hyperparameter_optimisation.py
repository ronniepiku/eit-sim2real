"""Hyperparameter optimisation for the 1D-CNN via grid search.

Optimises for **robustness**: the model must perform well on both clean and
noisy EIT measurements.  The primary objective is the harmonic mean of
clean-test macro-F1 and noisy-test macro-F1 (robustness score), which
penalises models that sacrifice one condition for the other.

Grid search is used (rather than random/Bayesian) per the user's compute
budget.  Results are saved incrementally so partial runs can be resumed.

Usage:
    uv run python python/hyperparameter_optimisation.py
    uv run python python/hyperparameter_optimisation.py --resume
    uv run python python/hyperparameter_optimisation.py --config python/configs/config.yaml
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from data.noise import NoiseConfig, apply_noise_batch_vectorised
from models.cnn1d import EITConv1D
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from utils import count_parameters, get_device, set_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
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
        return hashlib.md5(raw.encode()).hexdigest()[:12]


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
        self.mean_noisy_f1 = float(np.mean(self.fold_noisy_f1))
        self.mean_robustness = float(np.mean(self.fold_robustness))
        self.std_robustness = float(np.std(self.fold_robustness))


def harmonic_mean(a: float, b: float) -> float:
    """Compute harmonic mean of two values, returning 0 if either is 0."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Select the best available device."""
    return torch.device(get_device())


def _set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    set_seeds(seed)


def _count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return count_parameters(model)


def train_fold(
    hparams: HParamConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_clean: np.ndarray,
    X_val_noisy: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    epochs: int,
    early_stopping_patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    noise_config: NoiseConfig | None,
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
        X_val_noisy: Validation features (noisy, scaled).
        y_val: Validation labels.
        n_classes: Number of output classes.
        epochs: Maximum training epochs.
        early_stopping_patience: Early stopping patience.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR scheduler reduction factor.
        noise_config: Noise config for online augmentation (if enabled).
        severity_range: Severity range for domain randomisation.
        device: Torch device.
        seed: Random seed.

    Returns:
        Tuple of (clean_macro_f1, noisy_macro_f1, epochs_trained, n_params).
    """
    _set_seeds(seed)
    n_features = X_train.shape[1]

    model = EITConv1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=hparams.channels,
        fc_dim=hparams.fc_dim,
        dropout=hparams.dropout,
    ).to(device)

    n_params = _count_parameters(model)

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
                X_np = apply_noise_batch_vectorised(X_np, noise_config, rng=aug_rng)
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

        # Noisy evaluation
        X_noisy_t = torch.from_numpy(X_val_noisy).float().to(device)
        preds_noisy = model(X_noisy_t).argmax(dim=1).cpu().numpy()
        noisy_f1 = f1_score(y_val, preds_noisy, average="macro")

    return float(clean_f1), float(noisy_f1), epochs_trained, n_params


# ---------------------------------------------------------------------------
# Grid search orchestrator
# ---------------------------------------------------------------------------


def generate_grid() -> list[HParamConfig]:
    """Generate all combinations from the search space."""
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    configs = []
    for combo in product(*values):
        params = dict(zip(keys, combo, strict=True))
        configs.append(HParamConfig(**params))
    return configs


def run_grid_search(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    epochs: int = 100,
    early_stopping_patience: int = 20,
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

    device = _get_device()
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    grid = generate_grid()
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
            X_val_noisy_fold = scaler.transform(X_noisy[val_idx])
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
    epochs: int = 150,
    early_stopping_patience: int = 30,
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
    device = _get_device()
    _set_seeds(seed)

    # Prepare train/val/test splits
    dataset = prepare_splits(X_clean, y, random_state=seed, scaler_type="robust")

    # Reconstruct full index mapping for noisy test set
    # Use same split logic to get matching noisy test samples
    from sklearn.model_selection import train_test_split

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
                X_np = apply_noise_batch_vectorised(X_np, noise_config, rng=aug_rng)
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
            "n_parameters": _count_parameters(model),
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
        description="Hyperparameter optimisation for EIT 1D-CNN (grid search)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset .mat file (default: from config.yaml).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hyperparameter_optimisation"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs per trial (default: 100).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip completed trials).",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip grid search, train final model from existing results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for hyperparameter optimisation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_path = args.data_path or Path(cfg["data"]["path"])
    output_dir = args.output_dir

    # Load both clean and noisy data
    logger.info(f"Loading dataset from {data_path}")
    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)
    logger.info(f"Dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

    # Setup noise config for augmentation trials
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
    severity_range = tuple(severity_range_cfg) if severity_range_cfg else (0.5, 2.0)

    if not args.final_only:
        # Run grid search
        results_df, best_config, best_result = run_grid_search(
            X_clean=X_clean,
            X_noisy=X_noisy,
            y=y,
            n_folds=args.n_folds,
            epochs=args.epochs,
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
            scheduler_patience=cfg["training"]["scheduler_patience"],
            scheduler_factor=cfg["training"]["scheduler_factor"],
            noise_config=noise_config,
            severity_range=severity_range,
            seed=seed,
            output_dir=output_dir,
            resume=args.resume,
        )

        logger.info(f"\n{'='*60}")
        logger.info("GRID SEARCH COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Best config: {asdict(best_config)}")
        logger.info(
            f"Best robustness: {best_result.mean_robustness:.4f} "
            f"(clean={best_result.mean_clean_f1:.4f}, noisy={best_result.mean_noisy_f1:.4f})"
        )
    else:
        # Load best config from existing results
        results_path = output_dir / "grid_search_results.csv"
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

    # Train final model with best hyperparameters
    logger.info("\nTraining final optimised model...")
    model_path = train_final_model(
        best_config=best_config,
        X_clean=X_clean,
        X_noisy=X_noisy,
        y=y,
        epochs=150,
        early_stopping_patience=30,
        scheduler_patience=cfg["training"]["scheduler_patience"],
        scheduler_factor=cfg["training"]["scheduler_factor"],
        noise_config=noise_config,
        severity_range=severity_range,
        seed=seed,
        output_dir=output_dir,
    )

    logger.info(f"\nOptimised model saved to: {model_path}")
    logger.info("Run `python evaluate.py` with this model for full evaluation.")


if __name__ == "__main__":
    main()
