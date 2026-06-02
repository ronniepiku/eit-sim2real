"""Evaluation utilities for EIT touch classification models.

Provides model evaluation, robustness sweeps, and visualization generation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from data.noise import NoiseConfig, apply_noise_batch_vectorised
from models.cnn1d import EITConv1D
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from utils import CLASS_NAMES, get_device, predict_cnn, predict_cnn_with_probs

logger = logging.getLogger(__name__)


def evaluate_model(
    model: EITConv1D | object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "auto",
) -> dict[str, float | np.ndarray | str]:
    """Evaluate a CNN or sklearn model on a test set.

    Returns dict with accuracy, f1, confusion matrix, predictions and report.
    """
    if isinstance(model, EITConv1D):
        if device == "auto":
            device = get_device()
        y_pred = predict_cnn(model, X_test, device=device)
    else:
        y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=4
        ),
    }


def evaluate_severity_sweep(
    model: EITConv1D | object,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    severity_multipliers: list[float] | None = None,
    device: str = "auto",
    seed: int = 42,
    X_noisy: np.ndarray | None = None,
) -> dict[str, list[float]]:
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
        X_noisy: Optional noisy test features at 1.0x severity (for delta-based sweep).

    Returns:
        Dictionary with multipliers, accuracies, F1 scores, and degradation metrics.
    """
    if severity_multipliers is None:
        severity_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

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
            noise_cfg = NoiseConfig(severity=mult)
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
    model: EITConv1D | object,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list[float] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Evaluate model robustness under varying Gaussian noise levels."""
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
    model: EITConv1D | object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    noise_tag: str,
    model_name: str,
    device: str = "auto",
) -> None:
    """Evaluate model and generate all visualizations (unified CNN + baseline).

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
    from visualisation import (
        plot_confusion_matrix_and_save,
        plot_per_class_metrics_and_save,
        plot_precision_recall_curves_and_save,
        plot_roc_curves_and_save,
    )

    logger.info(f"Generating {model_name} visualizations...")

    is_cnn = isinstance(model, EITConv1D)

    for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if is_cnn:
            y_pred, y_probs = predict_cnn_with_probs(model, X, device=device)
        else:
            y_pred = model.predict(X)
            y_probs = (
                model.predict_proba(X) if hasattr(model, "predict_proba") else None
            )

        plot_confusion_matrix_and_save(
            y_true, y_pred, output_dir, model_name, noise_tag, split_name
        )
        plot_per_class_metrics_and_save(
            y_true, y_pred, output_dir, model_name, noise_tag, split_name
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

    # Log metrics
    val_results = evaluate_model(model, X_val, y_val, device=device)
    test_results = evaluate_model(model, X_test, y_test, device=device)
    logger.info(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")


# Keep old names as aliases for backward compatibility
evaluate_and_visualize_cnn = evaluate_and_visualize
evaluate_and_visualize_baseline = evaluate_and_visualize


def _load_model(model_path: Path, n_features: int) -> EITConv1D | object:
    """Load a CNN checkpoint or sklearn joblib model from disk."""
    if model_path.suffix == ".pt":
        model = EITConv1D(n_features=n_features)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model
    elif model_path.suffix == ".joblib":
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained EIT models.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--eval-on", choices=["noisy", "clean"], default="noisy")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    cfg = load_config()
    np.random.seed(args.seed)
    data_path = args.data_path or Path(cfg["data"]["path"])

    use_noisy = args.eval_on == "noisy"
    scaler_type = cfg.get("data", {}).get("scaler", "robust")
    X, y = load_mat_dataset(data_path, use_noisy=use_noisy)
    dataset = prepare_splits(X, y, random_state=args.seed, scaler_type=scaler_type)

    n_features = dataset.X_test.shape[1]
    model = _load_model(args.model_path, n_features)
    model_name = args.model_path.stem

    logger.info(f"Evaluating {model_name} on {args.eval_on} test set...")
    results = evaluate_model(model, dataset.X_test, dataset.y_test)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"\n{results['report']}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"accuracy": results["accuracy"], "f1_macro": results["f1_macro"]},
            fh,
            indent=2,
        )

    if args.robustness:
        X_clean, _ = load_mat_dataset(data_path, use_noisy=False)
        dataset_clean = prepare_splits(
            X_clean, y, random_state=args.seed, scaler_type=scaler_type
        )

        rob = evaluate_robustness(
            model,
            dataset_clean.X_test,
            dataset.y_test,
            noise_levels=cfg["evaluation"]["robustness_noise_levels"],
            seed=args.seed,
        )
        rob_path = args.output_dir / f"{model_name}_robustness.json"
        with open(rob_path, "w", encoding="utf-8") as fh:
            json.dump(rob, fh, indent=2)

        X_noisy_all, _ = load_mat_dataset(data_path, use_noisy=True)
        dataset_noisy = prepare_splits(
            X_noisy_all, y, random_state=args.seed, scaler_type=scaler_type
        )
        sev = evaluate_severity_sweep(
            model,
            dataset_clean.X_test,
            dataset.y_test,
            severity_multipliers=cfg["evaluation"].get(
                "severity_multipliers", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            ),
            X_noisy=dataset_noisy.X_test,
        )
        sev_path = args.output_dir / f"{model_name}_severity_sweep.json"
        with open(sev_path, "w", encoding="utf-8") as fh:
            json.dump(sev, fh, indent=2)


if __name__ == "__main__":
    main()
