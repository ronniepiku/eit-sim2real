"""Evaluation script for trained EIT touch classification models.

Computes accuracy, macro-F1, per-class precision/recall, and confusion matrix.
Supports both CNN and sklearn baseline models.  Results are saved to the
``results/`` directory for reproducibility and downstream visualisation.
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
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed contact",
]


def evaluate_model(
    model: EITConv1D | object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict[str, float | np.ndarray | str]:
    """Evaluate a CNN or sklearn model on a test set.

    Args:
        model: A trained ``EITConv1D`` or any sklearn classifier exposing
               a ``.predict()`` method.
        X_test: Test features.
        y_test: Test labels (0-indexed).
        device: Device for CNN inference (ignored for sklearn models).

    Returns:
        Dictionary with accuracy, f1, confusion matrix, predictions and report.
    """
    if isinstance(model, EITConv1D):
        model.eval()
        model.to(device)
        X_tensor = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            y_pred = logits.argmax(dim=1).cpu().numpy()
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


def evaluate_robustness(
    model: EITConv1D | object,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list[float] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Evaluate model robustness under varying Gaussian noise levels.

    Uses a fixed random seed so that the same noise realisation is applied
    across different models, ensuring fair comparison.

    Args:
        model: Trained CNN or sklearn model.
        X_clean: Clean test features (before noise).
        y_test: Test labels.
        noise_levels: List of noise standard deviations to test.
        device: Device for CNN inference.
        seed: Random seed for reproducible noise generation.

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

        logger.info(
            f"  Noise σ={noise_std:.3f} | "
            f"Acc: {results['accuracy']:.4f} | "
            f"F1: {results['f1_macro']:.4f}"
        )

    return {
        "noise_levels": noise_levels,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
    }


def evaluate_severity_sweep(
    model: EITConv1D | object,
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    y_test: np.ndarray,
    severity_multipliers: list[float] | None = None,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Evaluate model robustness under severity-scaled noise.

    Applies noise at varying severity multipliers following Hendrycks & Dietterich
    (2019). The noise perturbation is computed as the difference between noisy and
    clean data, then scaled by the severity multiplier.

    Args:
        model: Trained CNN or sklearn model.
        X_clean: Clean test features.
        X_noisy: Noisy test features (at 1.0x severity).
        y_test: Test labels.
        severity_multipliers: List of multipliers (e.g., [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).
        device: Device for CNN inference.

    Returns:
        Dictionary with multipliers, accuracies, F1 scores, and degradation metrics.
    """
    if severity_multipliers is None:
        severity_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Compute the noise perturbation at 1x
    noise_delta = X_noisy - X_clean

    accuracies: list[float] = []
    f1_scores: list[float] = []

    for mult in severity_multipliers:
        X_test_scaled = X_clean + mult * noise_delta
        results = evaluate_model(model, X_test_scaled, y_test, device=device)
        accuracies.append(results["accuracy"])
        f1_scores.append(results["f1_macro"])

        logger.info(
            f"  Severity {mult:.1f}x | "
            f"Acc: {results['accuracy']:.4f} | "
            f"F1: {results['f1_macro']:.4f}"
        )

    # Compute robustness metrics
    metrics: dict[str, float | None] = {}
    if len(accuracies) >= 2:
        # Graceful degradation rate (slope of accuracy vs severity)
        from numpy.polynomial.polynomial import polyfit

        coeffs = polyfit(severity_multipliers, accuracies, 1)
        metrics["degradation_slope"] = float(coeffs[1])

    # Robustness at 2x and 3x (if available)
    for target in [2.0, 3.0]:
        if target in severity_multipliers:
            idx = severity_multipliers.index(target)
            metrics[f"accuracy_at_{target:.0f}x"] = accuracies[idx]
            metrics[f"f1_at_{target:.0f}x"] = f1_scores[idx]

    return {
        "severity_multipliers": severity_multipliers,
        "accuracies": accuracies,
        "f1_scores": f1_scores,
        "metrics": metrics,
    }


def evaluate_severity_sweep_python(
    model: EITConv1D | object,
    X_clean: np.ndarray,
    y_test: np.ndarray,
    severity_multipliers: list[float] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Evaluate model robustness using the full Python noise model at varying severity.

    Unlike evaluate_severity_sweep (which scales a fixed noise delta), this function
    generates fresh noise at each severity level using the full 4-component model.
    This provides a more realistic evaluation of robustness to varying noise intensity.

    Args:
        model: Trained CNN or sklearn model.
        X_clean: Clean test features.
        y_test: Test labels.
        severity_multipliers: List of severity factors. 1.0 = default noise params.
        device: Device for CNN inference.
        seed: Random seed for reproducible noise generation.

    Returns:
        Dictionary with multipliers, accuracies, F1 scores, and degradation metrics.
    """
    if severity_multipliers is None:
        severity_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    accuracies: list[float] = []
    f1_scores_list: list[float] = []

    for mult in severity_multipliers:
        if mult == 0.0:
            X_test = X_clean
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

    # Compute degradation metrics (exclude 0.0 if present)
    eval_mults = [m for m in severity_multipliers if m > 0]
    eval_accs = [
        a for m, a in zip(severity_multipliers, accuracies, strict=True) if m > 0
    ]

    metrics: dict[str, float] = {}
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


def _save_results(
    results: dict,
    output_dir: Path,
    model_name: str,
) -> None:
    """Persist evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix as .npy
    cm_path = output_dir / f"{model_name}_confusion_matrix.npy"
    np.save(cm_path, results["confusion_matrix"])

    # Save scalar metrics as JSON
    metrics = {
        "model": model_name,
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
    }
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Save classification report as text
    report_path = output_dir / f"{model_name}_report.txt"
    report_path.write_text(results["report"], encoding="utf-8")

    logger.info(f"Results saved to {output_dir}/{model_name}_*")


def _load_model(
    model_path: Path,
    n_features: int,
) -> EITConv1D | object:
    """Load a CNN checkpoint or sklearn joblib model from disk."""
    if model_path.suffix == ".pt":
        model = EITConv1D(n_features=n_features)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model
    elif model_path.suffix == ".joblib":
        return joblib.load(model_path)
    else:
        raise ValueError(
            f"Unsupported model file format: {model_path.suffix}. "
            "Expected .pt (CNN) or .joblib (sklearn)."
        )


def get_cnn_predictions(
    model: EITConv1D,
    X: np.ndarray,
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from CNN model.

    Args:
        model: Trained CNN model.
        X: Input features.
        batch_size: Batch size for prediction.
        device: Device to use.

    Returns:
        Tuple of (predicted labels, predicted probabilities).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def evaluate_and_visualize_cnn(
    model: EITConv1D,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    noise_tag: str,
    model_name: str = "cnn1d",
    batch_size: int = 64,
    device: str = "auto",
) -> None:
    """Evaluate CNN model and generate all visualizations.

    Args:
        model: Trained CNN model.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        output_dir: Directory to save visualizations.
        noise_tag: Noise tag for file naming.
        model_name: Name of the model.
        batch_size: Batch size for prediction.
        device: Device to use.
    """
    # Import visualization functions
    from visualisation import (
        plot_confusion_matrix_and_save,
        plot_per_class_metrics_and_save,
        plot_precision_recall_curves_and_save,
        plot_roc_curves_and_save,
    )

    logger.info("Generating CNN visualizations...")

    # Get predictions
    y_val_pred, y_val_probs = get_cnn_predictions(
        model, X_val, batch_size=batch_size, device=device
    )
    y_test_pred, y_test_probs = get_cnn_predictions(
        model, X_test, batch_size=batch_size, device=device
    )

    n_classes = y_val_probs.shape[1]

    # Validation set visualizations
    plot_confusion_matrix_and_save(
        y_val, y_val_pred, output_dir, model_name, noise_tag, split_name="val"
    )
    plot_roc_curves_and_save(
        y_val,
        y_val_probs,
        output_dir,
        model_name,
        noise_tag,
        split_name="val",
        n_classes=n_classes,
    )
    plot_precision_recall_curves_and_save(
        y_val,
        y_val_probs,
        output_dir,
        model_name,
        noise_tag,
        split_name="val",
        n_classes=n_classes,
    )
    plot_per_class_metrics_and_save(
        y_val, y_val_pred, output_dir, model_name, noise_tag, split_name="val"
    )

    # Test set visualizations
    plot_confusion_matrix_and_save(
        y_test, y_test_pred, output_dir, model_name, noise_tag, split_name="test"
    )
    plot_roc_curves_and_save(
        y_test,
        y_test_probs,
        output_dir,
        model_name,
        noise_tag,
        split_name="test",
        n_classes=n_classes,
    )
    plot_precision_recall_curves_and_save(
        y_test,
        y_test_probs,
        output_dir,
        model_name,
        noise_tag,
        split_name="test",
        n_classes=n_classes,
    )
    plot_per_class_metrics_and_save(
        y_test, y_test_pred, output_dir, model_name, noise_tag, split_name="test"
    )

    # Compute and log metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")


def evaluate_and_visualize_baseline(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    noise_tag: str,
    model_name: str,
) -> None:
    """Evaluate baseline model and generate visualizations.

    Args:
        model: Trained baseline model (sklearn).
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        output_dir: Directory to save visualizations.
        noise_tag: Noise tag for file naming.
        model_name: Name of the model (e.g., 'svm', 'random_forest').
    """
    # Import visualization functions
    from visualisation import (
        plot_confusion_matrix_and_save,
        plot_per_class_metrics_and_save,
        plot_precision_recall_curves_and_save,
        plot_roc_curves_and_save,
    )

    logger.info(f"Generating {model_name} visualizations...")

    # Get predictions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        y_val_probs = model.predict_proba(X_val)
        y_test_probs = model.predict_proba(X_test)
        n_classes = y_val_probs.shape[1]
    else:
        y_val_probs = None
        y_test_probs = None
        n_classes = len(np.unique(y_val))

    # Validation set visualizations
    plot_confusion_matrix_and_save(
        y_val, y_val_pred, output_dir, model_name, noise_tag, split_name="val"
    )
    plot_per_class_metrics_and_save(
        y_val, y_val_pred, output_dir, model_name, noise_tag, split_name="val"
    )

    if y_val_probs is not None:
        plot_roc_curves_and_save(
            y_val,
            y_val_probs,
            output_dir,
            model_name,
            noise_tag,
            split_name="val",
            n_classes=n_classes,
        )
        plot_precision_recall_curves_and_save(
            y_val,
            y_val_probs,
            output_dir,
            model_name,
            noise_tag,
            split_name="val",
            n_classes=n_classes,
        )

    # Test set visualizations
    plot_confusion_matrix_and_save(
        y_test, y_test_pred, output_dir, model_name, noise_tag, split_name="test"
    )
    plot_per_class_metrics_and_save(
        y_test, y_test_pred, output_dir, model_name, noise_tag, split_name="test"
    )

    if y_test_probs is not None:
        plot_roc_curves_and_save(
            y_test,
            y_test_probs,
            output_dir,
            model_name,
            noise_tag,
            split_name="test",
            n_classes=n_classes,
        )
        plot_precision_recall_curves_and_save(
            y_test,
            y_test_probs,
            output_dir,
            model_name,
            noise_tag,
            split_name="test",
            n_classes=n_classes,
        )

    # Compute and log metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained EIT classification models."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset .mat file (default: from config.yaml).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.pt for CNN, .joblib for baselines).",
    )
    parser.add_argument(
        "--eval-on",
        choices=["noisy", "clean"],
        default="noisy",
        help="Evaluate on noisy or clean test data.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness evaluation with varying noise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    cfg = load_config()

    np.random.seed(args.seed)
    data_path = args.data_path or Path(cfg["data"]["path"])

    # Load data
    use_noisy = args.eval_on == "noisy"
    X, y = load_mat_dataset(data_path, use_noisy=use_noisy)
    dataset = prepare_splits(X, y, random_state=args.seed)

    # Load model
    n_features = dataset.X_test.shape[1]
    model = _load_model(args.model_path, n_features)
    model_name = args.model_path.stem

    # Evaluate
    logger.info(f"Evaluating {model_name} on {args.eval_on} test set...")
    results = evaluate_model(model, dataset.X_test, dataset.y_test)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"\n{results['report']}")

    _save_results(results, args.output_dir, model_name)

    # Robustness evaluation
    if args.robustness:
        logger.info("\nRobustness evaluation (varying noise levels):")
        # Always use clean test features as base for noise injection
        X_clean_test, _ = load_mat_dataset(data_path, use_noisy=False)
        dataset_clean = prepare_splits(X_clean_test, y, random_state=args.seed)
        rob = evaluate_robustness(
            model,
            dataset_clean.X_test,
            dataset.y_test,
            noise_levels=cfg["evaluation"]["robustness_noise_levels"],
            seed=args.seed,
        )
        rob_path = args.output_dir / f"{model_name}_robustness.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(rob_path, "w", encoding="utf-8") as fh:
            json.dump(rob, fh, indent=2)
        logger.info(f"Robustness results saved to {rob_path}")

        # Severity sweep (Hendrycks-style with noise multipliers)
        logger.info("\nSeverity sweep evaluation:")
        X_noisy_all, _ = load_mat_dataset(data_path, use_noisy=True)
        dataset_noisy = prepare_splits(X_noisy_all, y, random_state=args.seed)
        severity_mults = cfg["evaluation"].get(
            "severity_multipliers", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        )
        sev = evaluate_severity_sweep(
            model,
            dataset_clean.X_test,
            dataset_noisy.X_test,
            dataset.y_test,
            severity_multipliers=severity_mults,
        )
        sev_path = args.output_dir / f"{model_name}_severity_sweep.json"
        with open(sev_path, "w", encoding="utf-8") as fh:
            json.dump(sev, fh, indent=2)
        logger.info(f"Severity sweep results saved to {sev_path}")


if __name__ == "__main__":
    main()
