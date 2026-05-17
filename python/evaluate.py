"""Evaluation script for trained EIT touch classification models.

Computes accuracy, macro-F1, per-class precision/recall, and confusion matrix.
Supports both CNN and sklearn baseline models.  Results are saved to the
``results/`` directory for reproducibility and downstream visualisation.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from models.cnn1d import EITConv1D
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed",
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
        rob = evaluate_robustness(
            model,
            dataset.X_test,
            dataset.y_test,
            noise_levels=cfg["evaluation"]["robustness_noise_levels"],
            seed=args.seed,
        )
        rob_path = args.output_dir / f"{model_name}_robustness.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(rob_path, "w", encoding="utf-8") as fh:
            json.dump(rob, fh, indent=2)
        logger.info(f"Robustness results saved to {rob_path}")


if __name__ == "__main__":
    main()
