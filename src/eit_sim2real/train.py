"""Training functions for EIT touch classification models.

Provides `train_cnn` for standard and augmented training, and `train_cnn_mixed`
for mixed clean+noisy batch training.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eit_sim2real.data.noise import (
    NoiseConfig,
    apply_noise_batch_vectorised,
    apply_noise_in_scaled_space,
)
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.utils import get_device

logger = logging.getLogger(__name__)


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
    input_scaler: object | None = None,
    severity_range: tuple[float, float] | None = None,
    label_smoothing: float = 0.0,
    dropout: float = 0.3,
    channels: list[int] | None = None,
) -> tuple[EITConv1D, dict[str, list[float]]]:
    """Train 1D-CNN with optional online noise augmentation and early stopping.

    Args:
        X_train: Training features (should be clean if noise_config provided).
        y_train: Training labels (0-indexed).
        X_val: Validation features.
        y_val: Validation labels.
        n_classes: Number of output classes.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        scheduler_patience: Epochs before LR reduction.
        scheduler_factor: LR reduction factor.
        early_stopping_patience: Epochs without improvement before stopping.
        device: Compute device ('cpu', 'cuda', or 'auto').
        noise_config: If provided, apply online noise augmentation to training batches.
        input_scaler: Scaler that maps training features to and from raw voltage
            space. When provided, augmentation is applied in raw space before
            transforming back into the model's input space.
        severity_range: If provided with noise_config, sample severity per batch.
        label_smoothing: CrossEntropyLoss label smoothing.
        dropout: Dropout probability.
        channels: Conv block channel sizes.

    Returns:
        Tuple of (best model, training history dict).
    """
    if device == "auto":
        device = get_device()

    n_features = X_train.shape[1]
    model = EITConv1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=channels,
        dropout=dropout,
    ).to(device)

    augment = noise_config is not None and noise_config.enabled
    # Derive augmentation RNG from the training seed (passed via torch manual_seed)
    aug_rng = np.random.default_rng(torch.initial_seed() % 2**32) if augment else None

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0

        for X_batch, y_batch in train_loader:
            if augment and noise_config is not None and aug_rng is not None:
                X_np = X_batch.numpy()
                if severity_range is not None:
                    noise_config.severity = float(aug_rng.uniform(*severity_range))
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

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        ep_train_loss = train_loss / train_total
        ep_val_loss = val_loss / val_total
        history["train_loss"].append(ep_train_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)

        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {ep_train_loss:.4f} | "
                f"Val Loss: {ep_val_loss:.4f} | Val Acc: {val_correct / val_total:.4f}"
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_cnn_mixed(
    X_clean_train: np.ndarray,
    X_noisy_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 5,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 40,
    device: str = "auto",
    noise_config: NoiseConfig | None = None,
    clean_input_scaler: object | None = None,
    severity_range: tuple[float, float] | None = None,
    clean_ratio: float = 0.3,
    label_smoothing: float = 0.05,
    dropout: float = 0.4,
    channels: list[int] | None = None,
) -> tuple[EITConv1D, dict[str, list[float]]]:
    """Train CNN with mixed clean + noise-augmented batches.

    Each batch is composed of `clean_ratio` fraction clean samples and
    `1 - clean_ratio` fraction noise-augmented samples. Prevents
    over-specialisation to a single noise level.

    Args:
        X_clean_train: Clean training features.
        X_noisy_train: Pre-generated noisy training features (fallback if no noise_config).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_classes: Number of output classes.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        scheduler_patience: Epochs before LR reduction.
        scheduler_factor: LR reduction factor.
        early_stopping_patience: Epochs without improvement before stopping.
        device: Compute device.
        noise_config: If provided, apply online noise augmentation.
        clean_input_scaler: Scaler for the clean training space. When provided,
            augmented samples are perturbed in raw voltage space before being
            transformed back into the clean feature space.
        severity_range: Severity sampling range for augmentation.
        clean_ratio: Fraction of each batch that is clean.
        label_smoothing: CrossEntropyLoss label smoothing.
        dropout: Dropout probability.
        channels: Conv block channel sizes.

    Returns:
        Tuple of (best model, training history dict).
    """
    if device == "auto":
        device = get_device()

    n_features = X_clean_train.shape[1]
    model = EITConv1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=channels,
        dropout=dropout,
    ).to(device)

    augment = noise_config is not None and noise_config.enabled
    aug_rng = np.random.default_rng(torch.initial_seed() % 2**32)

    clean_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_clean_train).float(),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0

        for X_clean_batch, y_batch in clean_loader:
            actual_bs = X_clean_batch.shape[0]
            n_clean = max(1, int(actual_bs * clean_ratio))

            X_clean_part = X_clean_batch[:n_clean]
            X_noisy_src = X_clean_batch[n_clean:]
            y_clean_part = y_batch[:n_clean]
            y_noisy_part = y_batch[n_clean:]

            if X_noisy_src.shape[0] > 0:
                X_noisy_np = X_noisy_src.numpy()
                if augment:
                    if noise_config is None:
                        raise ValueError(
                            "noise_config must be provided when augment=True"
                        )
                    if severity_range is not None:
                        noise_config.severity = float(aug_rng.uniform(*severity_range))  # type: ignore[union-attr]
                    if clean_input_scaler is not None:
                        X_noisy_np = apply_noise_in_scaled_space(
                            X_noisy_np,
                            clean_input_scaler,
                            noise_config,
                            rng=aug_rng,
                        )
                    else:
                        X_noisy_np = apply_noise_batch_vectorised(
                            X_noisy_np,
                            noise_config,
                            rng=aug_rng,  # type: ignore[arg-type]
                        )
                else:
                    indices = aug_rng.integers(
                        0, len(X_noisy_train), size=X_noisy_src.shape[0]
                    )
                    X_noisy_np = X_noisy_train[indices]
                    y_noisy_part = torch.from_numpy(y_train[indices]).long()

                X_combined = torch.cat(
                    [X_clean_part, torch.from_numpy(X_noisy_np).float()],
                    dim=0,
                )
                y_combined = torch.cat([y_clean_part, y_noisy_part], dim=0)
            else:
                X_combined = X_clean_part
                y_combined = y_clean_part

            X_combined, y_combined = X_combined.to(device), y_combined.to(device)
            optimizer.zero_grad()
            logits = model(X_combined)
            loss = criterion(logits, y_combined)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_combined.size(0)
            train_correct += (logits.argmax(1) == y_combined).sum().item()
            train_total += X_combined.size(0)

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        ep_train_loss = train_loss / train_total
        ep_val_loss = val_loss / val_total
        history["train_loss"].append(ep_train_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)

        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {ep_train_loss:.4f} | "
                f"Val Loss: {ep_val_loss:.4f} | Val Acc: {val_correct / val_total:.4f}"
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
