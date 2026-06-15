"""Shared utilities for the EIT touch classification pipeline."""

import logging

import numpy as np
import torch

from eit_sim2real.constants import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "count_parameters",
    "get_device",
    "predict_cnn",
    "predict_cnn_with_probs",
    "rescale_cross_condition",
    "set_seeds",
]


_device_logged = False


def get_device() -> str:
    """Return the best available compute device.

    Selects ``cuda`` when available, otherwise ``cpu``. The selection
    decision is logged exactly once per process so callers in tight
    inference loops do not spam the log.
    """
    global _device_logged
    if torch.cuda.is_available():
        device = "cuda"
        if not _device_logged:
            logger.info(
                "CUDA available: using GPU '%s' (CUDA %s, %d device(s))",
                torch.cuda.get_device_name(0),
                torch.version.cuda,
                torch.cuda.device_count(),
            )
            _device_logged = True
    else:
        device = "cpu"
        if not _device_logged:
            logger.warning(
                "CUDA not available: falling back to CPU "
                "(install a CUDA-enabled PyTorch build for GPU acceleration)"
            )
            _device_logged = True
    return device


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and PyTorch.

    Uses the legacy ``np.random.seed()`` because PyTorch's DataLoader
    worker seeding and sklearn utilities depend on the global NumPy RNG
    state.  Prefer ``np.random.default_rng(seed)`` in new non-training code.
    """
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_cnn(
    model: torch.nn.Module, X: np.ndarray, device: str = "auto"
) -> np.ndarray:
    """Get class predictions from a CNN model.

    Args:
        model: Trained PyTorch model.
        X: Input features of shape (n_samples, n_features).
        device: Device for inference ('cpu', 'cuda', or 'auto').

    Returns:
        Predicted class labels of shape (n_samples,).
    """
    if device == "auto":
        device = get_device()
    model.to(device).eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        return model(X_tensor).argmax(dim=1).cpu().numpy()


def predict_cnn_with_probs(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and softmax probabilities from a CNN model.

    Args:
        model: Trained PyTorch model.
        X: Input features of shape (n_samples, n_features).
        batch_size: Batch size for inference.
        device: Device for inference.

    Returns:
        Tuple of (predicted labels, probability matrix).
    """
    if device == "auto":
        device = get_device()
    model.to(device).eval()

    all_preds = []
    all_probs = []
    n = len(X)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            X_batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            logits = model(X_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def rescale_cross_condition(
    X_scaled: np.ndarray,
    source_scaler: object,
    target_scaler: object,
) -> np.ndarray:
    """Re-scale data from one scaler's feature space into another's.

    When evaluating a model trained in target_scaler's space on data that
    was originally scaled by source_scaler, this inverts source_scaler and
    re-applies target_scaler so the features are in the correct space.

    Args:
        X_scaled: Data in source_scaler's feature space.
        source_scaler: The scaler that was used to produce X_scaled.
        target_scaler: The scaler for the model's training space.

    Returns:
        Data transformed into target_scaler's feature space.
    """
    X_raw = source_scaler.inverse_transform(X_scaled)  # type: ignore[attr-defined]
    return target_scaler.transform(X_raw)  # type: ignore[attr-defined]


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
