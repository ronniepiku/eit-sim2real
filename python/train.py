"""Training script for EIT touch classification models.

Supports both CNN and baseline (sklearn) models with configurable
noise augmentation settings for ablation studies.

Configuration is loaded from ``python/configs/config.yaml`` and can be
overridden via CLI arguments.  Baseline models are persisted with
``joblib`` alongside CNN checkpoints.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from data.noise import NoiseConfig, apply_noise_batch_vectorised
from models.baselines import get_baseline, train_baseline
from models.cnn1d import EITConv1D
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    CLI values override the corresponding ``config.yaml`` entries.
    """
    parser = argparse.ArgumentParser(
        description="Train EIT touch classification models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset .mat file (default: from config.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn1d", "svm", "random_forest", "mlp"],
        default="cnn1d",
        help="Model to train.",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Train on clean measurements instead of noisy.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (CNN only, default: from config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (CNN only, default: from config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Epochs without improvement before early stopping (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models"),
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help=(
            "Directory to save figures. Plots are stored as "
            "<figures-dir>/<model>/<noisy|clean>/"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 5,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 40,
    device: str = "auto",
    noise_config: NoiseConfig | None = None,
    severity_range: tuple[float, float] | None = None,
) -> tuple[EITConv1D, dict[str, list[float]]]:
    """Train the 1D-CNN model with early stopping.

    Training halts early if validation loss does not improve for
    ``early_stopping_patience`` consecutive epochs.  The best model
    (by validation loss) is always returned.

    Args:
        X_train: Training features (clean if noise_config is provided).
        y_train: Training labels (0-indexed).
        X_val: Validation features.
        y_val: Validation labels.
        n_classes: Number of classes.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularisation strength.
        scheduler_patience: Epochs before LR reduction.
        scheduler_factor: LR reduction factor.
        early_stopping_patience: Epochs without improvement before stopping.
        device: Device string ('cpu', 'cuda', or 'auto').
        noise_config: If provided, apply this noise model on-the-fly to
            training data each epoch (online augmentation). X_train should
            be CLEAN data in this case.
        severity_range: If provided with noise_config, sample the severity
            multiplier uniformly from this range each batch. E.g. (0.5, 2.0)
            for multi-severity domain randomisation.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    model = EITConv1D(n_features=n_features, n_classes=n_classes).to(device)

    # Online augmentation setup
    augment = noise_config is not None and noise_config.enabled
    aug_rng = np.random.default_rng(42) if augment else None

    # Data loaders — store raw numpy for augmentation, or pre-tensorise
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            # Apply online noise augmentation if configured
            if augment:
                X_np = X_batch.numpy()
                if severity_range is not None:
                    noise_config.severity = float(
                        aug_rng.uniform(severity_range[0], severity_range[1])
                    )
                X_np = apply_noise_batch_vectorised(X_np, noise_config, rng=aug_rng)
                X_batch = torch.from_numpy(X_np).float()

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        # Epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Best model tracking + early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def main() -> None:
    """Main training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from evaluate import evaluate_and_visualize_baseline, evaluate_and_visualize_cnn
    from visualisation import plot_training_curves

    args = parse_args()

    # Load config, then let CLI args override
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    data_path = args.data_path or Path(cfg["data"]["path"])
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else cfg["training"]["batch_size"]
    )
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else cfg["training"]["early_stopping_patience"]
    )
    lr = args.lr if args.lr is not None else cfg["training"]["learning_rate"]

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    use_noisy = not args.no_noise
    logger.info(f"Loading dataset from {data_path} (noisy={use_noisy})")
    X, y = load_mat_dataset(data_path, use_noisy=use_noisy)
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Prepare splits
    scaler_type = cfg.get("data", {}).get("scaler", "robust")
    dataset = prepare_splits(X, y, random_state=seed, scaler_type=scaler_type)
    logger.info(
        f"Splits: train={len(dataset.y_train)}, "
        f"val={len(dataset.y_val)}, test={len(dataset.y_test)}"
    )

    # Train model
    noise_tag = "noisy" if use_noisy else "clean"
    model_output_dir = args.output_dir
    figures_output_dir = args.figures_dir / args.model / noise_tag

    model_output_dir.mkdir(parents=True, exist_ok=True)
    figures_output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "cnn1d":
        logger.info("Training 1D-CNN...")
        model, history = train_cnn(
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=cfg["training"]["weight_decay"],
            scheduler_patience=cfg["training"]["scheduler_patience"],
            scheduler_factor=cfg["training"]["scheduler_factor"],
            early_stopping_patience=early_stopping_patience,
        )
        # Save model
        model_path = model_output_dir / f"cnn1d_{noise_tag}_best.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"CNN model saved to {model_path}")

        # Generate visualizations
        plot_training_curves(history, figures_output_dir, noise_tag)
        evaluate_and_visualize_cnn(
            model,
            dataset.X_val,
            dataset.y_val,
            dataset.X_test,
            dataset.y_test,
            figures_output_dir,
            noise_tag,
            model_name="cnn1d",
            batch_size=batch_size,
        )
    else:
        logger.info(f"Training baseline: {args.model}...")
        model = get_baseline(args.model, random_state=seed)
        model = train_baseline(model, dataset.X_train, dataset.y_train)

        # Evaluate on validation
        val_acc = model.score(dataset.X_val, dataset.y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")

        # Persist baseline model
        model_path = model_output_dir / f"{args.model}_{noise_tag}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Baseline model saved to {model_path}")

        # Generate visualizations
        evaluate_and_visualize_baseline(
            model,
            dataset.X_val,
            dataset.y_val,
            dataset.X_test,
            dataset.y_test,
            figures_output_dir,
            noise_tag,
            model_name=args.model,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
