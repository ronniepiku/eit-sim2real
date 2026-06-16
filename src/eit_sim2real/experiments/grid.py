"""Master experiment runner for EIT touch classification.

Runs all model x dataset x condition combinations across multiple seeds,
collects metrics with uncertainty estimates (mean +/- std), generates all
figures/tables, and produces a final Markdown report.

Datasets:
    - Raw (data/eit_dataset.mat) -- 208 features
    - PCA (data/cleaned/eit_cleaned_pca.mat) -- 7 features
    - LDA (data/cleaned/eit_cleaned_lda.mat) -- 4 features

Models:
    - 1D-CNN (only on datasets with >=8 features)
    - SVM (RBF kernel)
    - Random Forest (500 trees)
    - MLP (2x128 hidden)

Conditions:
    - Clean->Clean: baseline ceiling (no noise)
    - Clean->Noisy: vulnerability without noise training
    - Noisy->Noisy: fixed-severity noise training + noisy eval
    - Noisy->Clean: noise-trained model on clean data (generalisation)
    - Augmented->Noisy: multi-severity online augmentation + noisy eval
    - Augmented->Clean: multi-severity model on clean data
    - Mixed->Noisy: mixed clean+noisy batches with multi-severity + noisy eval
    - Mixed->Clean: mixed-trained model on clean data

Usage:
    eit experiments run-all
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from eit_sim2real.constants import (
    DEFAULT_SEVERITY_RANGE,
    MIXED_CNN_PARAMS,
    NOISY_CNN_PARAMS,
)
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.experiments.ablation import generate_ablation_report, run_ablation
from eit_sim2real.experiments.additional import run_additional_experiments
from eit_sim2real.experiments.extended import run_all_extended_experiments
from eit_sim2real.models import get_baseline, train_baseline
from eit_sim2real.train import train_cnn, train_cnn_mixed
from eit_sim2real.utils import predict_cnn, rescale_cross_condition

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

DATASETS = {
    "raw": Path("data/eit_dataset.mat"),
    "pca": Path("data/cleaned/eit_cleaned_pca.mat"),
    "lda": Path("data/cleaned/eit_cleaned_lda.mat"),
}

ALL_MODELS = ["cnn1d", "svm", "random_forest", "mlp"]
CNN_MIN_FEATURES = 8

# Standard conditions (all models)
BASE_CONDITIONS = [
    ("clean_train_clean_eval", "clean", "clean"),
    ("clean_train_noisy_eval", "clean", "noisy"),
    ("noisy_train_noisy_eval", "noisy", "noisy"),
    ("noisy_train_clean_eval", "noisy", "clean"),
]

# Extended conditions (CNN only — require special training regimes)
CNN_EXTENDED_CONDITIONS = [
    ("augmented_train_noisy_eval", "augmented", "noisy"),
    ("augmented_train_clean_eval", "augmented", "clean"),
    ("mixed_train_noisy_eval", "mixed", "noisy"),
    ("mixed_train_clean_eval", "mixed", "clean"),
]

# Training hyperparameters for noisy/augmented/mixed conditions are imported
# from ``eit_sim2real.constants`` (NOISY_CNN_PARAMS, MIXED_CNN_PARAMS,
# DEFAULT_SEVERITY_RANGE) to avoid the previous duplication between this
# module, ``cli/train.py`` and ``configs/config.yaml``.


# ── Helpers ───────────────────────────────────────────────────────────


def _get_noise_config() -> NoiseConfig:
    """Return the default 4-component noise config for online augmentation."""
    return NoiseConfig(
        enabled=True,
        gaussian_enabled=True,
        contact_impedance_enabled=True,
        electrode_bias_enabled=True,
        quantisation_enabled=True,
    )


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all metrics for a set of predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


# ── Main experiment loop ──────────────────────────────────────────────


def run_experiments(
    datasets: list[str],
    seeds: list[int],
    epochs: int,
    early_stopping_patience: int,
    output_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Run all experiments and return consolidated results."""
    all_results: list[dict] = []
    total_experiments = 0

    # Count total experiments for progress
    for ds_name in datasets:
        ds_path = DATASETS[ds_name]
        if not ds_path.exists():
            logger.warning(f"Dataset {ds_name} not found at {ds_path}, skipping.")
            continue
        X_check, _ = load_mat_dataset(ds_path, use_noisy=False)
        n_features = X_check.shape[1]
        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]
        # Base conditions for all models
        n_base = len(models_for_ds) * len(BASE_CONDITIONS) * len(seeds)
        # Extended conditions for CNN only
        n_ext = 0
        if n_features >= CNN_MIN_FEATURES:
            n_ext = len(CNN_EXTENDED_CONDITIONS) * len(seeds)
        total_experiments += n_base + n_ext

    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(
        "Validation logs are in-domain only (Val Acc uses each condition's val split); "
        "cross-domain robustness is reported from held-out test metrics by condition."
    )
    experiment_idx = 0

    for ds_name in datasets:
        ds_path = DATASETS[ds_name]
        if not ds_path.exists():
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"DATASET: {ds_name} ({ds_path})")
        logger.info(f"{'=' * 70}")

        # Load both clean and noisy versions
        X_clean, y = load_mat_dataset(ds_path, use_noisy=False)
        X_noisy, _ = load_mat_dataset(ds_path, use_noisy=True)
        normalize_features = ds_name == "raw"
        if not normalize_features:
            logger.info(
                "  Using pre-transformed features for reduced dataset; "
                "skipping additional scaling."
            )
        n_features = X_clean.shape[1]
        n_samples = X_clean.shape[0]

        logger.info(f"  Samples: {n_samples}, Features: {n_features}")

        # Determine which models can run
        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]
        if "cnn1d" not in models_for_ds:
            logger.info(
                f"  Skipping CNN (requires >={CNN_MIN_FEATURES} features, "
                f"got {n_features})"
            )

        # ── Base conditions (all models) ──
        for model_name in models_for_ds:
            for condition_name, train_key, eval_key in BASE_CONDITIONS:
                seed_results: list[dict] = []

                for seed in seeds:
                    experiment_idx += 1
                    logger.info(
                        f"  [{experiment_idx}/{total_experiments}] "
                        f"{model_name} | {condition_name} | seed={seed}"
                    )

                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    ds_clean = prepare_splits(
                        X_clean, y, random_state=seed, normalize=normalize_features
                    )
                    ds_noisy = prepare_splits(
                        X_noisy, y, random_state=seed, normalize=normalize_features
                    )

                    splits = {"clean": ds_clean, "noisy": ds_noisy}
                    train_ds = splits[train_key]
                    eval_ds = splits[eval_key]

                    # Cross-condition: rescale eval test data into training feature space
                    if eval_key != train_key and normalize_features:
                        X_test_eval = rescale_cross_condition(
                            eval_ds.X_test, eval_ds.scaler, train_ds.scaler
                        )
                    else:
                        X_test_eval = eval_ds.X_test

                    start_time = time.time()
                    if model_name == "cnn1d":
                        # Noisy base condition should train on the fixed noisy dataset only.
                        # Keep augmentation for explicit augmented/mixed conditions.
                        if train_key == "noisy":
                            model, history = train_cnn(
                                train_ds.X_train,
                                train_ds.y_train,
                                train_ds.X_val,
                                train_ds.y_val,
                                epochs=epochs,
                                early_stopping_patience=early_stopping_patience,
                                weight_decay=NOISY_CNN_PARAMS["weight_decay"],
                                dropout=NOISY_CNN_PARAMS["dropout"],
                                label_smoothing=NOISY_CNN_PARAMS["label_smoothing"],
                            )
                        else:
                            model, history = train_cnn(
                                train_ds.X_train,
                                train_ds.y_train,
                                train_ds.X_val,
                                train_ds.y_val,
                                epochs=epochs,
                                early_stopping_patience=early_stopping_patience,
                            )
                        y_pred = predict_cnn(model, X_test_eval)
                        stopped_epoch = len(history["train_loss"])
                    else:
                        model = get_baseline(model_name, random_state=seed)
                        model = train_baseline(
                            model, train_ds.X_train, train_ds.y_train
                        )
                        y_pred = model.predict(X_test_eval)
                        stopped_epoch = None

                    train_time = time.time() - start_time

                    metrics = evaluate_predictions(eval_ds.y_test, y_pred)
                    metrics["train_time_s"] = train_time
                    if stopped_epoch is not None:
                        metrics["stopped_epoch"] = stopped_epoch

                    seed_results.append(metrics)

                result_entry = _aggregate_seed_results(
                    seed_results,
                    ds_name,
                    n_features,
                    model_name,
                    condition_name,
                    train_key,
                    eval_key,
                    seeds,
                )
                all_results.append(result_entry)
                _log_result(result_entry)

        # ── Extended CNN conditions (augmented + mixed) ──
        if n_features >= CNN_MIN_FEATURES:
            for condition_name, train_key, eval_key in CNN_EXTENDED_CONDITIONS:
                seed_results = []

                for seed in seeds:
                    experiment_idx += 1
                    logger.info(
                        f"  [{experiment_idx}/{total_experiments}] "
                        f"cnn1d | {condition_name} | seed={seed}"
                    )

                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    ds_clean = prepare_splits(
                        X_clean, y, random_state=seed, normalize=normalize_features
                    )
                    ds_noisy = prepare_splits(
                        X_noisy, y, random_state=seed, normalize=normalize_features
                    )

                    # Augmented/mixed always train in clean-scaler space
                    if eval_key == "noisy" and normalize_features:
                        X_test_eval = rescale_cross_condition(
                            ds_noisy.X_test, ds_noisy.scaler, ds_clean.scaler
                        )
                    else:
                        X_test_eval = ds_clean.X_test
                    noise_cfg = _get_noise_config()

                    start_time = time.time()

                    if train_key == "augmented":
                        # Multi-severity online augmentation on clean data
                        model, history = train_cnn(
                            ds_clean.X_train,
                            ds_clean.y_train,
                            ds_clean.X_val,
                            ds_clean.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                            noise_config=noise_cfg,
                            severity_range=DEFAULT_SEVERITY_RANGE,
                            seed=seed,
                            weight_decay=NOISY_CNN_PARAMS["weight_decay"],
                            dropout=NOISY_CNN_PARAMS["dropout"],
                            label_smoothing=NOISY_CNN_PARAMS["label_smoothing"],
                        )
                    elif train_key == "mixed":
                        # Mixed clean + multi-severity noisy batches
                        model, history = train_cnn_mixed(
                            ds_clean.X_train,
                            ds_noisy.X_train,
                            ds_clean.y_train,
                            ds_clean.X_val,
                            ds_clean.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                            noise_config=noise_cfg,
                            severity_range=DEFAULT_SEVERITY_RANGE,
                            seed=seed,
                            **MIXED_CNN_PARAMS,
                        )
                    else:
                        raise ValueError(f"Unknown train_key: {train_key}")

                    y_pred = predict_cnn(model, X_test_eval)
                    stopped_epoch = len(history["train_loss"])
                    train_time = time.time() - start_time

                    metrics = evaluate_predictions(ds_clean.y_test, y_pred)
                    metrics["train_time_s"] = train_time
                    metrics["stopped_epoch"] = stopped_epoch

                    seed_results.append(metrics)

                result_entry = _aggregate_seed_results(
                    seed_results,
                    ds_name,
                    n_features,
                    "cnn1d",
                    condition_name,
                    train_key,
                    eval_key,
                    seeds,
                )
                all_results.append(result_entry)
                _log_result(result_entry)

    # Generate figures for best seed (seed[0]) of each key combination
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FIGURES (using first seed)")
    logger.info("=" * 70)
    _generate_all_figures(
        datasets, seeds[0], epochs, early_stopping_patience, figures_dir
    )

    return pd.DataFrame(all_results)


def _aggregate_seed_results(
    seed_results: list[dict],
    ds_name: str,
    n_features: int,
    model_name: str,
    condition_name: str,
    train_key: str,
    eval_key: str,
    seeds: list[int],
) -> dict:
    """Aggregate metrics across seeds into a single result entry."""
    acc_values = [r["accuracy"] for r in seed_results]
    f1_values = [r["f1_macro"] for r in seed_results]
    prec_values = [r["precision_macro"] for r in seed_results]
    rec_values = [r["recall_macro"] for r in seed_results]

    result_entry = {
        "dataset": ds_name,
        "n_features": n_features,
        "model": model_name,
        "condition": condition_name,
        "train_data": train_key,
        "eval_data": eval_key,
        "n_seeds": len(seeds),
        "accuracy_mean": float(np.mean(acc_values)),
        "accuracy_std": float(np.std(acc_values)),
        "f1_macro_mean": float(np.mean(f1_values)),
        "f1_macro_std": float(np.std(f1_values)),
        "precision_macro_mean": float(np.mean(prec_values)),
        "precision_macro_std": float(np.std(prec_values)),
        "recall_macro_mean": float(np.mean(rec_values)),
        "recall_macro_std": float(np.std(rec_values)),
        "train_time_mean_s": float(np.mean([r["train_time_s"] for r in seed_results])),
    }
    if "stopped_epoch" in seed_results[0]:
        epochs_stopped = [r["stopped_epoch"] for r in seed_results]
        result_entry["stopped_epoch_mean"] = float(np.mean(epochs_stopped))

    return result_entry


def _log_result(result_entry: dict) -> None:
    """Log a single aggregated result."""
    logger.info(
        f"    -> Acc: {result_entry['accuracy_mean']:.4f} "
        f"+/- {result_entry['accuracy_std']:.4f} | "
        f"F1: {result_entry['f1_macro_mean']:.4f} "
        f"+/- {result_entry['f1_macro_std']:.4f}"
    )


def _generate_all_figures(
    datasets: list[str],
    seed: int,
    epochs: int,
    early_stopping_patience: int,
    figures_dir: Path,
) -> None:
    """Generate confusion matrices, training curves, and per-class metrics."""
    from visualisation import (
        plot_confusion_matrix_and_save,
        plot_per_class_metrics_and_save,
        plot_training_curves,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_conditions = BASE_CONDITIONS + CNN_EXTENDED_CONDITIONS

    for ds_name in datasets:
        ds_path = DATASETS[ds_name]
        if not ds_path.exists():
            continue

        X_clean, y = load_mat_dataset(ds_path, use_noisy=False)
        X_noisy, _ = load_mat_dataset(ds_path, use_noisy=True)
        normalize_features = ds_name == "raw"
        n_features = X_clean.shape[1]

        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]

        ds_clean = prepare_splits(
            X_clean, y, random_state=seed, normalize=normalize_features
        )
        ds_noisy = prepare_splits(
            X_noisy, y, random_state=seed, normalize=normalize_features
        )

        for model_name in models_for_ds:
            conditions_for_model = (
                all_conditions if model_name == "cnn1d" else BASE_CONDITIONS
            )

            for condition_name, train_key, eval_key in conditions_for_model:
                fig_output = figures_dir / ds_name / model_name / condition_name
                fig_output.mkdir(parents=True, exist_ok=True)

                # Determine correctly-scaled eval test data
                if eval_key == "noisy":
                    eval_ds = ds_noisy
                else:
                    eval_ds = ds_clean

                # Determine the training scaler space
                if train_key in ("clean", "augmented", "mixed"):
                    train_ds_for_scaler = ds_clean
                else:  # noisy
                    train_ds_for_scaler = ds_noisy

                if (
                    eval_key != train_key
                    and normalize_features
                    and train_key not in ("augmented", "mixed")
                ):
                    X_test_eval = rescale_cross_condition(
                        eval_ds.X_test, eval_ds.scaler, train_ds_for_scaler.scaler
                    )
                elif (
                    train_key in ("augmented", "mixed")
                    and eval_key == "noisy"
                    and normalize_features
                ):
                    X_test_eval = rescale_cross_condition(
                        ds_noisy.X_test, ds_noisy.scaler, ds_clean.scaler
                    )
                else:
                    X_test_eval = eval_ds.X_test

                np.random.seed(seed)
                torch.manual_seed(seed)

                if model_name == "cnn1d":
                    noise_cfg = _get_noise_config()

                    if train_key == "mixed":
                        model, history = train_cnn_mixed(
                            ds_clean.X_train,
                            ds_noisy.X_train,
                            ds_clean.y_train,
                            ds_clean.X_val,
                            ds_clean.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                            noise_config=noise_cfg,
                            severity_range=DEFAULT_SEVERITY_RANGE,
                            seed=seed,
                            **MIXED_CNN_PARAMS,
                        )
                    elif train_key == "augmented":
                        model, history = train_cnn(
                            ds_clean.X_train,
                            ds_clean.y_train,
                            ds_clean.X_val,
                            ds_clean.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                            noise_config=noise_cfg,
                            severity_range=DEFAULT_SEVERITY_RANGE,
                            seed=seed,
                            weight_decay=NOISY_CNN_PARAMS["weight_decay"],
                            dropout=NOISY_CNN_PARAMS["dropout"],
                            label_smoothing=NOISY_CNN_PARAMS["label_smoothing"],
                        )
                    elif train_key == "noisy":
                        model, history = train_cnn(
                            ds_noisy.X_train,
                            ds_noisy.y_train,
                            ds_noisy.X_val,
                            ds_noisy.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                            weight_decay=NOISY_CNN_PARAMS["weight_decay"],
                            dropout=NOISY_CNN_PARAMS["dropout"],
                            label_smoothing=NOISY_CNN_PARAMS["label_smoothing"],
                        )
                    else:
                        model, history = train_cnn(
                            ds_clean.X_train,
                            ds_clean.y_train,
                            ds_clean.X_val,
                            ds_clean.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                        )

                    y_pred = predict_cnn(model, X_test_eval)
                    plot_training_curves(history, fig_output, condition_name)
                else:
                    train_ds = ds_noisy if train_key == "noisy" else ds_clean
                    model = get_baseline(model_name, random_state=seed)
                    model = train_baseline(model, train_ds.X_train, train_ds.y_train)
                    y_pred = model.predict(X_test_eval)

                plot_confusion_matrix_and_save(
                    eval_ds.y_test,
                    y_pred,
                    fig_output,
                    model_name,
                    condition_name,
                    split_name="test",
                )
                plot_per_class_metrics_and_save(
                    eval_ds.y_test,
                    y_pred,
                    fig_output,
                    model_name,
                    condition_name,
                    split_name="test",
                )


# ── Report generation ─────────────────────────────────────────────────


def generate_report(df: pd.DataFrame, output_dir: Path, runtime_s: float) -> None:
    """Generate a comprehensive Markdown report from results."""
    all_conditions = BASE_CONDITIONS + CNN_EXTENDED_CONDITIONS
    report_lines: list[str] = []
    report_lines.append("# EIT Touch Classification — Experiment Report")
    report_lines.append(
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(f"**Total runtime**: {runtime_s / 60:.1f} minutes")
    report_lines.append(f"**Seeds**: {df['n_seeds'].iloc[0]}")
    report_lines.append(f"**Datasets**: {', '.join(df['dataset'].unique())}")
    report_lines.append(f"**Models**: {', '.join(df['model'].unique())}")
    report_lines.append(f"**Conditions**: {len(df['condition'].unique())}")
    report_lines.append(f"**Total experiments**: {len(df)}")

    # ── Summary: Best models per condition ──
    report_lines.append("\n---\n## 1. Best Models by Condition\n")
    for condition_name, _, _ in all_conditions:
        subset = df[df["condition"] == condition_name]
        if subset.empty:
            continue
        best = subset.loc[subset["accuracy_mean"].idxmax()]
        report_lines.append(
            f"- **{condition_name}**: {best['model']} on {best['dataset']} "
            f"— Acc: {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}, "
            f"F1: {best['f1_macro_mean']:.4f} ± {best['f1_macro_std']:.4f}"
        )

    # ── Per-dataset summary ──
    report_lines.append("\n---\n## 2. Results by Dataset\n")
    for ds_name in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds_name]
        n_feat = ds_df["n_features"].iloc[0]
        report_lines.append(f"\n### Dataset: {ds_name} ({n_feat} features)\n")
        report_lines.append("| Model | Condition | Accuracy | F1 (macro) |")
        report_lines.append("|-------|-----------|----------|------------|")
        for _, row in ds_df.sort_values(["model", "condition"]).iterrows():
            report_lines.append(
                f"| {row['model']} | {row['condition']} | "
                f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                f"{row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} |"
            )

    # ── Key comparisons ──
    report_lines.append("\n---\n## 3. Key Comparisons\n")

    # 3.1 Training regime comparison (CNN on raw, noisy eval)
    report_lines.append("### 3.1 CNN Training Regime Comparison (noisy evaluation)\n")
    cnn_noisy_eval = df[(df["model"] == "cnn1d") & (df["eval_data"] == "noisy")]
    if not cnn_noisy_eval.empty:
        report_lines.append("| Dataset | Condition | Accuracy | F1 |")
        report_lines.append("|---------|-----------|----------|-----|")
        for _, row in cnn_noisy_eval.sort_values(
            ["dataset", "accuracy_mean"], ascending=[True, False]
        ).iterrows():
            report_lines.append(
                f"| {row['dataset']} | {row['condition']} | "
                f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                f"{row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} |"
            )
    else:
        report_lines.append("_No CNN results for noisy evaluation._")

    # 3.2 Best model per dataset on clean→clean
    report_lines.append("\n### 3.2 Best Model per Dataset (clean→clean — ceiling)\n")
    clean_clean = df[df["condition"] == "clean_train_clean_eval"]
    if not clean_clean.empty:
        report_lines.append("| Dataset | Best Model | Accuracy | F1 |")
        report_lines.append("|---------|-----------|----------|-----|")
        for ds_name in clean_clean["dataset"].unique():
            ds_subset = clean_clean[clean_clean["dataset"] == ds_name]
            best = ds_subset.loc[ds_subset["accuracy_mean"].idxmax()]
            report_lines.append(
                f"| {ds_name} | {best['model']} | "
                f"{best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f} | "
                f"{best['f1_macro_mean']:.4f} ± {best['f1_macro_std']:.4f} |"
            )

    # 3.3 Effect of noise training
    report_lines.append(
        "\n### 3.3 Effect of Noise Training (noisy→noisy vs clean→noisy)\n"
    )
    report_lines.append(
        "| Dataset | Model | Clean→Noisy Acc | Noisy→Noisy Acc | Δ Accuracy |"
    )
    report_lines.append(
        "|---------|-------|-----------------|-----------------|------------|"
    )
    for ds_name in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds_name]
        for model_name in ds_df["model"].unique():
            c2n = ds_df[
                (ds_df["model"] == model_name)
                & (ds_df["condition"] == "clean_train_noisy_eval")
            ]
            n2n = ds_df[
                (ds_df["model"] == model_name)
                & (ds_df["condition"] == "noisy_train_noisy_eval")
            ]
            if not c2n.empty and not n2n.empty:
                c2n_acc = c2n.iloc[0]["accuracy_mean"]
                n2n_acc = n2n.iloc[0]["accuracy_mean"]
                delta = n2n_acc - c2n_acc
                report_lines.append(
                    f"| {ds_name} | {model_name} | {c2n_acc:.4f} | {n2n_acc:.4f} | "
                    f"{'+' if delta >= 0 else ''}{delta:.4f} |"
                )

    # 3.4 Augmented vs Mixed vs Fixed-noise (CNN only)
    report_lines.append(
        "\n### 3.4 Training Regime Comparison: "
        "Fixed Noise vs Augmented vs Mixed (CNN)\n"
    )
    report_lines.append(
        "| Dataset | Fixed Noisy→Noisy | Augmented→Noisy | Mixed→Noisy | Mixed→Clean |"
    )
    report_lines.append(
        "|---------|-------------------|-----------------|-------------|------------|"
    )
    for ds_name in df["dataset"].unique():
        ds_cnn = df[(df["dataset"] == ds_name) & (df["model"] == "cnn1d")]
        if ds_cnn.empty:
            continue
        n2n = ds_cnn[ds_cnn["condition"] == "noisy_train_noisy_eval"]
        a2n = ds_cnn[ds_cnn["condition"] == "augmented_train_noisy_eval"]
        m2n = ds_cnn[ds_cnn["condition"] == "mixed_train_noisy_eval"]
        m2c = ds_cnn[ds_cnn["condition"] == "mixed_train_clean_eval"]

        def _fmt(subset: pd.DataFrame) -> str:
            if subset.empty:
                return "—"
            return f"{subset.iloc[0]['accuracy_mean']:.4f}"

        report_lines.append(
            f"| {ds_name} | {_fmt(n2n)} | {_fmt(a2n)} | {_fmt(m2n)} | {_fmt(m2c)} |"
        )

    # 3.5 Feature engineering impact
    report_lines.append(
        "\n### 3.5 Feature Engineering Impact (clean→clean baseline comparison)\n"
    )
    report_lines.append("| Model | Raw Acc | PCA Acc | LDA Acc |")
    report_lines.append("|-------|---------|---------|---------|")
    cc = df[df["condition"] == "clean_train_clean_eval"]
    for model_name in cc["model"].unique():
        model_cc = cc[cc["model"] == model_name]
        row_str = f"| {model_name} |"
        for ds in ["raw", "pca", "lda"]:
            ds_row = model_cc[model_cc["dataset"] == ds]
            if not ds_row.empty:
                row_str += f" {ds_row.iloc[0]['accuracy_mean']:.4f} |"
            else:
                row_str += " — |"
        report_lines.append(row_str)

    # ── Overfitting indicators (CNN) ──
    report_lines.append("\n---\n## 4. CNN Training Behavior\n")
    cnn_results = df[df["model"] == "cnn1d"]
    if not cnn_results.empty and "stopped_epoch_mean" in cnn_results.columns:
        report_lines.append(
            "| Dataset | Condition | Stopped Epoch (mean) | Train Time (s) |"
        )
        report_lines.append(
            "|---------|-----------|---------------------|----------------|"
        )
        for _, row in cnn_results.iterrows():
            ep = row.get("stopped_epoch_mean", "—")
            ep_str = f"{ep:.0f}" if isinstance(ep, float) else str(ep)
            report_lines.append(
                f"| {row['dataset']} | {row['condition']} | {ep_str} | "
                f"{row['train_time_mean_s']:.1f} |"
            )

    # ── Statistical summary ──
    report_lines.append("\n---\n## 5. Statistical Summary\n")
    report_lines.append(
        f"Uncertainty reported as mean ± std over {df['n_seeds'].iloc[0]} random seeds."
    )
    report_lines.append(
        "Larger std indicates training instability or sensitivity to initialization.\n"
    )

    high_var = df[df["accuracy_std"] > 0.02]
    if not high_var.empty:
        report_lines.append("**High variance experiments (std > 0.02):**\n")
        for _, row in high_var.iterrows():
            report_lines.append(
                f"- {row['model']} on {row['dataset']} ({row['condition']}): "
                f"std={row['accuracy_std']:.4f}"
            )
    else:
        report_lines.append(
            "All experiments show low variance (std <= 0.02) — good reproducibility."
        )

    # ── Conclusions ──
    report_lines.append("\n---\n## 6. Key Findings\n")

    # Find overall best noisy performer
    noisy_eval = df[df["eval_data"] == "noisy"]
    if not noisy_eval.empty:
        best_noisy = noisy_eval.loc[noisy_eval["accuracy_mean"].idxmax()]
        report_lines.append(
            f"1. **Best noisy-domain performer**: {best_noisy['model']} on "
            f"{best_noisy['dataset']} ({best_noisy['condition']}) — "
            f"{best_noisy['accuracy_mean']:.4f} accuracy"
        )

    # Find overall best clean performer
    clean_eval = df[df["eval_data"] == "clean"]
    if not clean_eval.empty:
        best_clean = clean_eval.loc[clean_eval["accuracy_mean"].idxmax()]
        report_lines.append(
            f"2. **Best clean-domain performer**: {best_clean['model']} on "
            f"{best_clean['dataset']} ({best_clean['condition']}) — "
            f"{best_clean['accuracy_mean']:.4f} accuracy"
        )

    # Mixed regime assessment
    mixed_noisy = df[df["condition"] == "mixed_train_noisy_eval"]
    mixed_clean = df[df["condition"] == "mixed_train_clean_eval"]
    if not mixed_noisy.empty and not mixed_clean.empty:
        best_mixed_n = mixed_noisy.loc[mixed_noisy["accuracy_mean"].idxmax()]
        best_mixed_c = mixed_clean.loc[mixed_clean["accuracy_mean"].idxmax()]
        report_lines.append(
            f"3. **Best mixed-trained model (noisy eval)**: "
            f"{best_mixed_n['accuracy_mean']:.4f} on {best_mixed_n['dataset']}"
        )
        report_lines.append(
            f"4. **Best mixed-trained model (clean eval)**: "
            f"{best_mixed_c['accuracy_mean']:.4f} on {best_mixed_c['dataset']}"
        )

    # Augmented vs fixed-noise
    augmented_noisy = df[df["condition"] == "augmented_train_noisy_eval"]
    fixed_noisy = df[df["condition"] == "noisy_train_noisy_eval"]
    if not augmented_noisy.empty and not fixed_noisy.empty:
        aug_best = augmented_noisy["accuracy_mean"].max()
        fixed_best = fixed_noisy[fixed_noisy["model"] == "cnn1d"]["accuracy_mean"].max()
        delta = aug_best - fixed_best
        report_lines.append(
            f"5. **Multi-severity augmentation vs fixed noise**: "
            f"{'+' if delta >= 0 else ''}{delta * 100:.1f}pp difference"
        )

    report_lines.append("\n---\n*Report generated by `eit experiments run-all`*\n")

    # Write report
    report_path = output_dir / "experiment_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all EIT classification experiments with uncertainty."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=["raw", "pca", "lda"],
        help="Which datasets to include.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds for uncertainty estimation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max CNN epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=40,
        help="CNN early stopping patience.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports"),
        help="Directory for report and tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation study.",
    )
    parser.add_argument(
        "--ablation-skip-all-configs",
        action="store_true",
        help="Skip exhaustive component/order ablation in ablation study.",
    )
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="Skip extended experiments (stats, ensemble, t-SNE, etc.).",
    )
    parser.add_argument(
        "--skip-additional",
        action="store_true",
        help="Skip additional memorisation experiments (fixed-bias, different-draw).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 42 + args.seeds))

    logger.info("=" * 70)
    logger.info("EIT TOUCH CLASSIFICATION — MASTER EXPERIMENT RUNNER")
    logger.info("=" * 70)
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"CNN epochs: {args.epochs}, patience: {args.early_stopping_patience}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Figures: {args.figures_dir}")
    logger.info("=" * 70)

    start_time = time.time()

    # Run all experiments
    df = run_experiments(
        datasets=args.datasets,
        seeds=seeds,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )

    # Save consolidated CSV
    csv_path = args.output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Consolidated results saved to {csv_path}")

    # Save as JSON for programmatic access
    json_path = args.output_dir / "all_results.json"
    df.to_json(json_path, orient="records", indent=2)

    # Generate pivot tables
    acc_pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="condition",
        values="accuracy_mean",
        aggfunc="first",
    )
    acc_pivot_path = args.output_dir / "accuracy_pivot.csv"
    acc_pivot.to_csv(acc_pivot_path)

    f1_pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="condition",
        values="f1_macro_mean",
        aggfunc="first",
    )
    f1_pivot_path = args.output_dir / "f1_pivot.csv"
    f1_pivot.to_csv(f1_pivot_path)

    logger.info(f"Pivot tables saved to {args.output_dir}/")

    # ── Severity sweep on best CNN models ──
    logger.info("\n" + "=" * 70)
    logger.info("SEVERITY SWEEP — evaluating CNN generalisation across noise levels")
    logger.info("=" * 70)
    severity_results = _run_severity_sweeps(
        args.datasets,
        seeds[0],
        args.epochs,
        args.early_stopping_patience,
        args.output_dir,
    )
    if severity_results:
        sev_path = args.output_dir / "severity_sweep_results.json"
        with open(sev_path, "w", encoding="utf-8") as fh:
            json.dump(severity_results, fh, indent=2)
        logger.info(f"Severity sweep results saved to {sev_path}")

    # ── Ablation study ──
    if not args.skip_ablation:
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY — noise component analysis")
        logger.info("=" * 70)
        ablation_figures_dir = args.figures_dir / "ablation"
        ablation_study = run_ablation(
            data_path=DATASETS["raw"],
            model_name="cnn1d",
            seed=seeds[0],
            run_all_configs=not args.ablation_skip_all_configs,
            run_severity_sweep=True,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            figures_dir=ablation_figures_dir,
            output_dir=args.output_dir,
        )
        ablation_csv = args.output_dir / "ablation_results.csv"
        ablation_study.save(ablation_csv)
        ablation_time = time.time() - start_time
        generate_ablation_report(
            ablation_study, args.output_dir, ablation_time, "cnn1d"
        )
        logger.info(
            f"Ablation study complete — {len(ablation_study.results)} experiments"
        )

    # ── Extended experiments ──
    if not args.skip_extended:
        logger.info("\n" + "=" * 70)
        logger.info(
            "EXTENDED EXPERIMENTS — stats, dataset size, ensemble, t-SNE, severity, Gaussian"
        )
        logger.info("=" * 70)
        run_all_extended_experiments(
            data_path=DATASETS["raw"],
            seeds=seeds,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )

    # ── Additional memorisation experiments ──
    if not args.skip_additional:
        logger.info("\n" + "=" * 70)
        logger.info(
            "ADDITIONAL EXPERIMENTS — fixed-bias augmentation & different-draw test"
        )
        logger.info("=" * 70)
        run_additional_experiments(
            data_path=DATASETS["raw"],
            seeds=seeds,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            output_dir=args.output_dir / "additional",
            models_dir=Path("results/models"),
        )

    # Generate report
    total_time = time.time() - start_time
    generate_report(df, args.output_dir, total_time)

    logger.info("\n" + "=" * 70)
    logger.info(f"ALL EXPERIMENTS COMPLETE — {total_time / 60:.1f} minutes total")
    logger.info(f"Report: {args.output_dir / 'experiment_report.md'}")
    logger.info(f"CSV: {csv_path}")
    logger.info("=" * 70)


def _run_severity_sweeps(
    datasets: list[str],
    seed: int,
    epochs: int,
    early_stopping_patience: int,
    output_dir: Path,
) -> dict:
    """Run severity sweeps for CNN under each training regime on raw data.

    Trains CNN with each regime (fixed noisy, augmented, mixed), then
    evaluates across severity multipliers [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
    """
    severity_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results: dict = {}

    # Only run on raw dataset (208 features) for meaningful comparison
    if "raw" not in datasets:
        return results

    ds_path = DATASETS["raw"]
    if not ds_path.exists():
        return results

    X_clean, y = load_mat_dataset(ds_path, use_noisy=False)
    X_noisy, _ = load_mat_dataset(ds_path, use_noisy=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds_clean = prepare_splits(X_clean, y, random_state=seed)
    ds_noisy = prepare_splits(X_noisy, y, random_state=seed)
    noise_cfg = _get_noise_config()

    regimes = {
        "noisy_fixed": lambda: train_cnn(
            ds_noisy.X_train,
            ds_noisy.y_train,
            ds_noisy.X_val,
            ds_noisy.y_val,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **NOISY_CNN_PARAMS,
        ),
        "augmented": lambda: train_cnn(
            ds_clean.X_train,
            ds_clean.y_train,
            ds_clean.X_val,
            ds_clean.y_val,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            noise_config=noise_cfg,
            severity_range=DEFAULT_SEVERITY_RANGE,
            seed=seed,
            **NOISY_CNN_PARAMS,
        ),
        "mixed": lambda: train_cnn_mixed(
            ds_clean.X_train,
            ds_noisy.X_train,
            ds_clean.y_train,
            ds_clean.X_val,
            ds_clean.y_val,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            noise_config=noise_cfg,
            severity_range=DEFAULT_SEVERITY_RANGE,
            seed=seed,
            **MIXED_CNN_PARAMS,
        ),
    }

    for regime_name, train_fn in regimes.items():
        logger.info(f"  Training CNN ({regime_name}) for severity sweep...")
        np.random.seed(seed)
        torch.manual_seed(seed)

        model, _ = train_fn()

        # Determine base test data in the regime's native feature space.
        # For noisy_fixed, model operates in noisy-scaler space;
        # for augmented/mixed, model operates in clean-scaler space.
        if regime_name == "noisy_fixed":
            # Get raw clean test data and scale into noisy feature space
            base_test = rescale_cross_condition(
                ds_clean.X_test, ds_clean.scaler, ds_noisy.scaler
            )
        else:
            base_test = ds_clean.X_test

        # Evaluate at each severity level
        accuracies = []
        f1_scores = []
        for mult in severity_multipliers:
            if mult == 0.0:
                X_test = base_test
            else:
                sweep_cfg = NoiseConfig(severity=mult)
                rng = np.random.default_rng(seed)
                X_test = apply_noise_batch_vectorised(base_test, sweep_cfg, rng=rng)

            y_pred = predict_cnn(model, X_test)
            acc = float(accuracy_score(ds_clean.y_test, y_pred))
            f1 = float(f1_score(ds_clean.y_test, y_pred, average="macro"))
            accuracies.append(acc)
            f1_scores.append(f1)

            logger.info(f"    {regime_name} @ {mult:.1f}x: Acc={acc:.4f}, F1={f1:.4f}")

        results[regime_name] = {
            "severity_multipliers": severity_multipliers,
            "accuracies": accuracies,
            "f1_scores": f1_scores,
        }

    return results


if __name__ == "__main__":
    main()
