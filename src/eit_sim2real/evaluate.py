"""Evaluation utilities for EIT touch classification models.

Provides model evaluation, robustness sweeps, and visualization generation.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from eit_sim2real.constants import CLASS_NAMES
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.utils import get_device, predict_cnn, predict_cnn_with_probs

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a CNN or sklearn model on a test set.

    Args:
        model: Trained PyTorch CNN or sklearn classifier.
        X_test: Test features of shape (n_samples, n_features).
        y_test: True labels of shape (n_samples,).
        device: Device for CNN inference.

    Returns:
        Dict with accuracy, f1_macro, confusion_matrix, y_pred, and report.
    """
    if isinstance(model, EITConv1D):
        if device == "auto":
            device = get_device()
        y_pred = predict_cnn(model, X_test, device=device)
    else:
        y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "report": classification_report(
            y_test,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }


def evaluate_severity_sweep(
    model: Any,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    severity_multipliers: list[float] | None = None,
    device: str = "auto",
    seed: int = 42,
    X_noisy: np.ndarray | None = None,
    noise_config: NoiseConfig | None = None,
) -> dict[str, list[float] | dict[str, float]]:
    """Evaluate model robustness under varying noise severity.

    If X_noisy is provided, uses linear interpolation between clean and noisy.
    Otherwise, uses the full Python 4-component noise model at each severity.

    Args:
        model: Trained CNN or sklearn model.
        X_clean: Clean test features.
        y_test: Test labels.
        severity_multipliers: List of severity multipliers.
        device: Device for CNN inference.
        seed: Random seed for noise generation.
        X_noisy: Optional noisy test features at 1.0x severity.
        noise_config: Base noise configuration to scale. If None, uses the
            project default :class:`NoiseConfig`. The ``severity`` field of the
            base config is overwritten per multiplier; all other parameters
            (component flags, SNR, max_bias, etc.) are preserved.

    Returns:
        Dictionary with multipliers, accuracies, F1 scores, and degradation metrics.
    """
    if severity_multipliers is None:
        severity_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    base_cfg = noise_config if noise_config is not None else NoiseConfig()

    accuracies: list[float] = []
    f1_scores_list: list[float] = []

    # Precompute noise delta if using interpolation mode
    noise_delta = (X_noisy - X_clean) if X_noisy is not None else None

    for mult in severity_multipliers:
        if mult == 0.0:
            X_test = X_clean
        elif noise_delta is not None:
            X_test = X_clean + mult * noise_delta
        else:
            noise_cfg = dataclass_replace(base_cfg, severity=mult)
            rng = np.random.default_rng(seed)
            X_test = apply_noise_batch_vectorised(X_clean, noise_cfg, rng=rng)

        results = evaluate_model(model, X_test, y_test, device=device)
        accuracies.append(results["accuracy"])
        f1_scores_list.append(results["f1_macro"])

        logger.info(
            f"  Severity {mult:.1f}x | "
            f"Acc: {results['accuracy']:.4f} | "
            f"F1: {results['f1_macro']:.4f}"
        )

    # Compute degradation metrics
    metrics: dict[str, float] = {}
    eval_mults = [m for m in severity_multipliers if m > 0]
    eval_accs = [a for m, a in zip(severity_multipliers, accuracies) if m > 0]

    if len(eval_accs) >= 2:
        from numpy.polynomial.polynomial import polyfit

        coeffs = polyfit(eval_mults, eval_accs, 1)
        metrics["degradation_slope"] = float(coeffs[1])

    for target in [2.0, 3.0]:
        if target in severity_multipliers:
            idx = severity_multipliers.index(target)
            metrics[f"accuracy_at_{target:.0f}x"] = accuracies[idx]
            metrics[f"f1_at_{target:.0f}x"] = f1_scores_list[idx]

    if 0.0 in severity_multipliers:
        metrics["accuracy_at_0x"] = accuracies[severity_multipliers.index(0.0)]

    return {
        "severity_multipliers": severity_multipliers,
        "accuracies": accuracies,
        "f1_scores": f1_scores_list,
        "metrics": metrics,
    }


def evaluate_robustness(
    model: Any,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list[float] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Evaluate model robustness under varying Gaussian noise levels.

    Args:
        model: Trained model.
        X_clean: Clean test features.
        y_test: Test labels.
        noise_levels: Standard deviations of additive Gaussian noise.
        device: Device for CNN inference.
        seed: Random seed.

    Returns:
        Dictionary with noise levels, accuracies, and F1 scores.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    rng = np.random.default_rng(seed)
    accuracies: list[float] = []
    f1_scores: list[float] = []

    for noise_std in noise_levels:
        X_noisy = X_clean + noise_std * rng.standard_normal(
            X_clean.shape, dtype=np.float32
        )
        results = evaluate_model(model, X_noisy, y_test, device=device)
        accuracies.append(results["accuracy"])
        f1_scores.append(results["f1_macro"])
        logger.info(f"  Noise σ={noise_std:.3f} | Acc: {results['accuracy']:.4f}")

    return {
        "noise_levels": noise_levels,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
    }


def evaluate_and_visualize(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    noise_tag: str,
    model_name: str,
    device: str = "auto",
) -> None:
    """Evaluate model and generate all visualizations.

    Args:
        model: Trained model (CNN or sklearn).
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        output_dir: Directory to save visualizations.
        noise_tag: Noise tag for file naming.
        model_name: Name of the model.
        device: Device for CNN inference.
    """
    from eit_sim2real.visualisation import (
        plot_confusion_matrix_and_save,
        plot_per_class_metrics_and_save,
        plot_precision_recall_curves_and_save,
        plot_roc_curves_and_save,
    )

    logger.info(f"Generating {model_name} visualizations...")

    is_cnn = isinstance(model, EITConv1D)

    for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_probs: np.ndarray | None = None
        if is_cnn:
            y_pred, y_probs = predict_cnn_with_probs(model, X, device=device)
        else:
            y_pred = model.predict(X)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X)

        plot_confusion_matrix_and_save(
            y_true,
            y_pred,
            output_dir,
            model_name,
            noise_tag,
            split_name,
        )
        plot_per_class_metrics_and_save(
            y_true,
            y_pred,
            output_dir,
            model_name,
            noise_tag,
            split_name,
        )

        if y_probs is not None:
            n_classes = y_probs.shape[1]
            plot_roc_curves_and_save(
                y_true,
                y_probs,
                output_dir,
                model_name,
                noise_tag,
                split_name,
                n_classes,
            )
            plot_precision_recall_curves_and_save(
                y_true,
                y_probs,
                output_dir,
                model_name,
                noise_tag,
                split_name,
                n_classes,
            )

    # Log metrics from the last iteration (test split)
    if is_cnn:
        y_pred_test = predict_cnn(model, X_test, device=device)
    else:
        y_pred_test = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred_test))

    if is_cnn:
        y_pred_val = predict_cnn(model, X_val, device=device)
    else:
        y_pred_val = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, y_pred_val))

    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")


def load_model(model_path: Path, n_features: int) -> Any:
    """Load a CNN checkpoint or sklearn joblib model from disk.

    Args:
        model_path: Path to .pt (CNN) or .joblib (sklearn) model file.
        n_features: Number of input features (needed for CNN architecture).

    Returns:
        Loaded model ready for inference.

    Raises:
        ValueError: If model format is unsupported.

    Security:
        CNN (.pt) files are loaded with ``weights_only=True`` to prevent
        arbitrary code execution.  Joblib (.joblib) files can execute
        arbitrary code on load — only load models you have trained locally.
    """
    import joblib
    import torch

    if model_path.suffix == ".pt":
        model = EITConv1D(n_features=n_features)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    elif model_path.suffix == ".joblib":
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
