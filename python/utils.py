"""Shared utilities for the EIT touch classification pipeline."""

import numpy as np
import torch

NUM_CLASSES = 5

CLASS_NAMES: list[str] = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed contact",
]


def get_device() -> str:
    """Return the best available compute device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}"
    )
    return device


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and PyTorch."""
    np.random.seed(seed)
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
    source_scaler,
    target_scaler,
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
    X_raw = source_scaler.inverse_transform(X_scaled)
    return target_scaler.transform(X_raw)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
