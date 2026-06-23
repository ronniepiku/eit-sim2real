"""Evaluation utilities for EIT touch classification models.

Provides model evaluation, robustness sweeps, and visualization generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from eit_sim2real.constants import CLASS_NAMES
from eit_sim2real.data import prepare_splits
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.train import train_cnn
from eit_sim2real.utils import (
    get_device,
    predict_cnn,
    predict_cnn_with_probs,
    rescale_cross_condition,
)

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


def run_gaussian_only_evaluation(
    data_path: Path,
    seed: int = 42,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports"),
    figures_dir: Path = Path("results/figures"),
) -> dict[str, Any]:
    """Evaluate models under Gaussian-only noise for literature comparison.

    Trains clean, Gaussian-augmented, and full-noise-augmented CNNs plus a
    Gaussian-trained RF baseline, then evaluates all models over an SNR sweep.
    """
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    from eit_sim2real.data import load_mat_dataset, prepare_splits
    from eit_sim2real.models import get_baseline, train_baseline
    from eit_sim2real.train import train_cnn

    logger.info("── Gaussian-Only Evaluation (Literature Comparison) ──")
    device = get_device()
    snr_levels = [60, 50, 40, 30, 20]
    noise_cfg_full = NoiseConfig()
    noise_cfg_gauss = NoiseConfig.only("gaussian")

    X_clean, y = load_mat_dataset(data_path, use_noisy=False)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds = prepare_splits(X_clean, y, random_state=seed)

    results: dict[str, Any] = {"snr_levels_db": snr_levels}

    # Train models
    logger.info("  Training clean CNN...")
    torch.manual_seed(seed)
    cnn_clean, _ = train_cnn(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )

    logger.info("  Training Gaussian-augmented CNN...")
    torch.manual_seed(seed)
    cnn_gauss, _ = train_cnn(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_cfg_gauss,
        severity_range=(0.5, 2.0),
        weight_decay=1e-3,
        dropout=0.4,
        label_smoothing=0.05,
    )

    logger.info("  Training full-noise-augmented CNN...")
    torch.manual_seed(seed)
    cnn_full, _ = train_cnn(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_cfg_full,
        severity_range=(0.5, 2.0),
        weight_decay=1e-3,
        dropout=0.4,
        label_smoothing=0.05,
    )

    # Train RF baseline on Gaussian-noisy data
    logger.info("  Training RF baseline...")
    rng_train = np.random.default_rng(seed)
    X_train_gauss = apply_noise_batch_vectorised(
        ds.X_train, noise_cfg_gauss, rng=rng_train
    )
    rf = get_baseline("random_forest", random_state=seed)
    rf = train_baseline(rf, X_train_gauss, ds.y_train)

    models: dict[str, Any] = {
        "CNN (clean-trained)": cnn_clean,
        "CNN (Gaussian-augmented)": cnn_gauss,
        "CNN (full-noise-augmented)": cnn_full,
        "Random Forest (Gaussian-trained)": rf,
    }

    # Evaluate at each SNR level
    for model_name, model in models.items():
        accs: list[float] = []
        f1s: list[float] = []
        for snr in snr_levels:
            # 40dB corresponds to severity=1.0 in the current noise model setup.
            severity = 40.0 / snr
            cfg = NoiseConfig.only("gaussian")
            cfg.severity = severity
            rng = np.random.default_rng(seed + 500)
            X_te = apply_noise_batch_vectorised(ds.X_test, cfg, rng=rng)

            if isinstance(model, EITConv1D):
                y_pred = predict_cnn(model, X_te, device)
            else:
                y_pred = model.predict(X_te)

            accs.append(float(accuracy_score(ds.y_test, y_pred)))
            f1s.append(float(f1_score(ds.y_test, y_pred, average="macro")))

        results[model_name] = {"accuracies": accs, "f1_scores": f1s}
        logger.info(
            f"  {model_name}: {accs[0]:.3f}@{snr_levels[0]}dB → {accs[-1]:.3f}@{snr_levels[-1]}dB"
        )

    # Also report clean baseline (no-noise test)
    clean_acc_clean = float(
        accuracy_score(ds.y_test, predict_cnn(cnn_clean, ds.X_test, device))
    )
    clean_acc_gauss = float(
        accuracy_score(ds.y_test, predict_cnn(cnn_gauss, ds.X_test, device))
    )
    clean_acc_full = float(
        accuracy_score(ds.y_test, predict_cnn(cnn_full, ds.X_test, device))
    )
    results["clean_eval_baselines"] = {
        "CNN (clean-trained)": clean_acc_clean,
        "CNN (Gaussian-augmented)": clean_acc_gauss,
        "CNN (full-noise-augmented)": clean_acc_full,
    }

    # Generate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("colorblind", len(models))
    markers = ["o", "s", "D", "^"]

    for i, (model_name, data) in enumerate([(k, results[k]) for k in models]):
        ax.plot(
            snr_levels,
            data["accuracies"],
            f"{markers[i]}-",
            label=model_name,
            color=colors[i],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Gaussian-Only Noise Robustness (Literature Comparison)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.set_xticks(snr_levels)
    fig.tight_layout()
    fig_path = figures_dir / "gaussian_only_evaluation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    out_path = output_dir / "gaussian_only_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {out_path}")
    return results


def run_calibration_analysis(
    data_path: Path,
    seed: int = 42,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports"),
    figures_dir: Path = Path("results/figures"),
) -> dict[str, Any]:
    """Analyse confidence calibration of CNN under clean and noisy conditions.

    Computes reliability diagrams (calibration curves), Expected Calibration
    Error (ECE), and maximum calibration error. A well-calibrated model has
    confidence scores that match actual accuracy.
    """
    from eit_sim2real.data import load_mat_dataset

    logger.info("── Confidence Calibration Analysis ──")
    device = get_device()
    noise_cfg = NoiseConfig()
    n_bins = 10

    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds_clean = prepare_splits(X_clean, y, random_state=seed)
    ds_noisy = prepare_splits(X_noisy, y, random_state=seed)

    logger.info("  Training clean CNN...")
    torch.manual_seed(seed)
    cnn_clean, _ = train_cnn(
        ds_clean.X_train,
        ds_clean.y_train,
        ds_clean.X_val,
        ds_clean.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )

    logger.info("  Training noise-augmented CNN...")
    torch.manual_seed(seed)
    cnn_aug, _ = train_cnn(
        ds_clean.X_train,
        ds_clean.y_train,
        ds_clean.X_val,
        ds_clean.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_cfg,
        severity_range=(0.5, 2.0),
        weight_decay=1e-3,
        dropout=0.4,
        label_smoothing=0.05,
    )

    def _get_probs(model: EITConv1D, X: np.ndarray) -> np.ndarray:
        model.to(device).eval()
        X_t = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            logits = model(X_t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def _compute_calibration(
        probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> dict[str, Any]:
        """Compute binned calibration metrics."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies_arr = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            if mask.sum() > 0:
                bin_accs.append(float(accuracies_arr[mask].mean()))
                bin_confs.append(float(confidences[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_accs.append(0.0)
                bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_counts.append(0)

        total = len(y_true)
        ece = sum(
            (count / total) * abs(acc - conf)
            for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        )
        mce = (
            max(
                abs(acc - conf)
                for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
                if count > 0
            )
            if any(c > 0 for c in bin_counts)
            else 0.0
        )

        return {
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
            "ece": float(ece),
            "mce": float(mce),
            "mean_confidence": float(confidences.mean()),
            "overall_accuracy": float(accuracies_arr.mean()),
        }

    X_noisy_test_clean_space = rescale_cross_condition(
        ds_noisy.X_test, ds_noisy.scaler, ds_clean.scaler
    )

    conditions = {
        "Clean CNN → Clean data": (cnn_clean, ds_clean.X_test, ds_clean.y_test),
        "Clean CNN → Noisy data": (
            cnn_clean,
            X_noisy_test_clean_space,
            ds_noisy.y_test,
        ),
        "Augmented CNN → Clean data": (cnn_aug, ds_clean.X_test, ds_clean.y_test),
        "Augmented CNN → Noisy data": (
            cnn_aug,
            X_noisy_test_clean_space,
            ds_noisy.y_test,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for cond_name, (model, X_te, y_te) in conditions.items():
        probs = _get_probs(model, X_te)
        cal = _compute_calibration(probs, y_te, n_bins=n_bins)
        results[cond_name] = cal
        logger.info(
            f"  {cond_name}: ECE={cal['ece']:.4f}, MCE={cal['mce']:.4f}, "
            f"MeanConf={cal['mean_confidence']:.3f}, Acc={cal['overall_accuracy']:.3f}"
        )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for idx, (cond_name, cal) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{cond_name}\nECE={cal['ece']:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Reliability Diagrams: CNN Confidence Calibration", fontsize=12, y=1.02
    )
    fig.tight_layout()
    fig_path = figures_dir / "calibration_reliability_diagrams.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    for idx, (cond_name, (model, X_te, y_te)) in enumerate(conditions.items()):
        probs = _get_probs(model, X_te)
        confidences = np.max(probs, axis=1)
        correct = np.argmax(probs, axis=1) == y_te

        ax = axes2[idx]
        ax.hist(
            confidences[correct],
            bins=20,
            alpha=0.6,
            label="Correct",
            color="green",
            density=True,
        )
        ax.hist(
            confidences[~correct],
            bins=20,
            alpha=0.6,
            label="Incorrect",
            color="red",
            density=True,
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.set_title(cond_name, fontsize=10)
        ax.legend(fontsize=8)

    fig2.suptitle(
        "Confidence Distributions: Correct vs Incorrect Predictions",
        fontsize=12,
        y=1.02,
    )
    fig2.tight_layout()
    fig_path2 = figures_dir / "calibration_confidence_distributions.png"
    fig2.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved: {fig_path2}")

    out_path = output_dir / "calibration_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {out_path}")
    return results


def run_per_class_robustness(
    data_path: Path,
    seed: int = 42,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports"),
    figures_dir: Path = Path("results/figures"),
) -> dict[str, Any]:
    """Analyse per-class accuracy degradation across severity levels."""
    from eit_sim2real.data import load_mat_dataset

    logger.info("── Per-Class Robustness Breakdown ──")
    device = get_device()
    noise_cfg = NoiseConfig()
    severity_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    X_clean, y = load_mat_dataset(data_path, use_noisy=False)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds = prepare_splits(X_clean, y, random_state=seed)

    logger.info("  Training noise-augmented CNN...")
    model, _ = train_cnn(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_cfg,
        severity_range=(0.5, 2.0),
        weight_decay=1e-3,
        dropout=0.4,
        label_smoothing=0.05,
    )

    n_classes = len(CLASS_NAMES)
    results: dict[str, Any] = {"severity_levels": severity_levels, "classes": {}}

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_mask = ds.y_test == cls_idx
        cls_accs = []
        cls_f1s = []

        for sev in severity_levels:
            if sev == 0.0:
                X_te = ds.X_test
            else:
                cfg = NoiseConfig(severity=sev)
                rng = np.random.default_rng(seed + 300)
                X_te = apply_noise_batch_vectorised(ds.X_test, cfg, rng=rng)

            y_pred = predict_cnn(model, X_te, device)

            cls_correct = (y_pred[cls_mask] == cls_idx).sum()
            cls_total = cls_mask.sum()
            cls_accs.append(float(cls_correct / cls_total) if cls_total > 0 else 0.0)

            cls_f1 = float(f1_score(ds.y_test, y_pred, average=None)[cls_idx])
            cls_f1s.append(cls_f1)

        results["classes"][cls_name] = {
            "accuracies": cls_accs,
            "f1_scores": cls_f1s,
            "n_samples": int(cls_mask.sum()),
        }
        logger.info(f"  {cls_name}: {cls_accs[0]:.3f}@0x → {cls_accs[-1]:.3f}@3x")

    overall_accs = []
    for sev in severity_levels:
        if sev == 0.0:
            X_te = ds.X_test
        else:
            cfg = NoiseConfig(severity=sev)
            rng = np.random.default_rng(seed + 300)
            X_te = apply_noise_batch_vectorised(ds.X_test, cfg, rng=rng)
        y_pred = predict_cnn(model, X_te, device)
        overall_accs.append(float(accuracy_score(ds.y_test, y_pred)))
    results["overall"] = overall_accs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("colorblind", n_classes)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        data = results["classes"][cls_name]
        ax1.plot(
            severity_levels,
            data["accuracies"],
            "o-",
            label=cls_name,
            color=colors[cls_idx],
            linewidth=2,
        )
        ax2.plot(
            severity_levels,
            data["f1_scores"],
            "s-",
            label=cls_name,
            color=colors[cls_idx],
            linewidth=2,
        )

    ax1.plot(
        severity_levels, overall_accs, "k--", linewidth=2.5, label="Overall", alpha=0.7
    )
    ax1.set_xlabel("Severity Multiplier")
    ax1.set_ylabel("Per-Class Recall")
    ax1.set_title("Per-Class Accuracy vs Noise Severity")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("Severity Multiplier")
    ax2.set_ylabel("Per-Class F1")
    ax2.set_title("Per-Class F1 vs Noise Severity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Per-Class Robustness Analysis (Augmented CNN)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig_path = figures_dir / "per_class_robustness.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    fig2, ax = plt.subplots(figsize=(10, 5))
    degradations = []
    names = []
    for cls_name, data in results["classes"].items():
        deg = data["accuracies"][0] - data["accuracies"][-1]
        degradations.append(deg)
        names.append(cls_name)

    bars = ax.bar(names, degradations, color=colors, edgecolor="black", linewidth=0.5)
    for bar, deg in zip(bars, degradations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{deg:.3f}",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("Accuracy Drop (0× → 3× severity)")
    ax.set_title("Per-Class Degradation Under Maximum Noise")
    ax.axhline(
        np.mean(degradations),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(degradations):.3f}",
    )
    ax.legend()
    fig2.tight_layout()
    fig_path2 = figures_dir / "per_class_degradation_bar.png"
    fig2.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved: {fig_path2}")

    out_path = output_dir / "per_class_robustness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {out_path}")
    return results


def run_noise_parameter_sensitivity(
    data_path: Path,
    seed: int = 42,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports"),
    figures_dir: Path = Path("results/figures"),
) -> dict[str, Any]:
    """Vary noise model parameters to test sensitivity of conclusions."""
    from eit_sim2real.data import load_mat_dataset

    logger.info("── Noise Parameter Sensitivity ──")
    device = get_device()

    X_clean, y = load_mat_dataset(data_path, use_noisy=False)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    ds = prepare_splits(X_clean, y, random_state=seed)

    logger.info("  Training noise-augmented CNN (default params)...")
    noise_cfg = NoiseConfig()
    model, _ = train_cnn(
        ds.X_train,
        ds.y_train,
        ds.X_val,
        ds.y_val,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        noise_config=noise_cfg,
        severity_range=(0.5, 2.0),
        weight_decay=1e-3,
        dropout=0.4,
        label_smoothing=0.05,
    )

    results: dict[str, dict[str, Any]] = {}

    sweeps = {
        "snr_db": {
            "values": [60, 50, 40, 30, 20],
            "default": 40,
            "unit": "dB",
            "label": "Gaussian SNR",
        },
        "contact_impedance_std_percent": {
            "values": [5, 10, 15, 20, 25],
            "default": 10,
            "unit": "%",
            "label": "Contact Impedance σ",
        },
        "max_bias": {
            "values": [0.005, 0.01, 0.02, 0.04, 0.08],
            "default": 0.02,
            "unit": "",
            "label": "Electrode Bias Max",
        },
        "adc_bits": {
            "values": [8, 10, 12, 14, 16],
            "default": 16,
            "unit": "bits",
            "label": "ADC Resolution",
        },
    }

    for param_name, sweep_info in sweeps.items():
        sweep_values = cast(list[float | int], sweep_info["values"])
        sweep_label = cast(str, sweep_info["label"])
        accs = []
        f1s = []
        for val in sweep_values:
            cfg = NoiseConfig()
            setattr(cfg, param_name, val)
            rng = np.random.default_rng(seed + 400)
            X_te = apply_noise_batch_vectorised(ds.X_test, cfg, rng=rng)

            y_pred = predict_cnn(model, X_te, device)
            acc = float(accuracy_score(ds.y_test, y_pred))
            f1_val = float(f1_score(ds.y_test, y_pred, average="macro"))
            accs.append(acc)
            f1s.append(f1_val)

        results[param_name] = {
            "values": sweep_values,
            "accuracies": accs,
            "f1_scores": f1s,
            "label": sweep_label,
            "unit": sweep_info["unit"],
            "default": sweep_info["default"],
        }
        logger.info(
            f"  {sweep_label}: "
            f"{accs[0]:.3f}@{sweep_values[0]} → {accs[-1]:.3f}@{sweep_values[-1]}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = sns.color_palette("colorblind", 4)

    for idx, (param_name, data) in enumerate(results.items()):
        values = cast(list[float | int], data["values"])
        accuracies = cast(list[float], data["accuracies"])
        default_value = cast(float | int, data["default"])
        label = cast(str, data["label"])
        unit = cast(str, data["unit"])

        ax = axes[idx // 2, idx % 2]
        ax.plot(
            values,
            accuracies,
            "o-",
            color=colors[idx],
            linewidth=2,
            markersize=8,
        )

        default_idx = values.index(default_value)
        ax.axvline(
            default_value,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Default ({default_value})",
        )
        ax.scatter(
            [default_value],
            [accuracies[default_idx]],
            color="red",
            s=100,
            zorder=5,
            marker="*",
        )

        unit_str = f" ({unit})" if unit else ""
        ax.set_xlabel(f"{label}{unit_str}")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Sensitivity: {label}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)

    fig.suptitle("Noise Parameter Sensitivity Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    fig_path = figures_dir / "noise_parameter_sensitivity.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    out_path = output_dir / "noise_parameter_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {out_path}")
    return results


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
