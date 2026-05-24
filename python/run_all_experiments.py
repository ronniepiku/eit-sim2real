"""Master experiment runner for EIT touch classification.

Runs all model × dataset × condition combinations across multiple seeds,
collects metrics with uncertainty estimates (mean ± std), generates all
figures/tables, and produces a final Markdown report.

Datasets:
    - Raw (data/eit_dataset.mat) — 208 features
    - Cleaned (data/cleaned/eit_cleaned.mat) — 22 features
    - PCA (data/cleaned/eit_cleaned_pca.mat) — 7 features
    - LDA (data/cleaned/eit_cleaned_lda.mat) — 4 features

Models:
    - 1D-CNN (only on datasets with ≥8 features)
    - SVM (RBF kernel)
    - Random Forest (500 trees)
    - MLP (2×128 hidden)

Conditions:
    - Clean→Clean, Clean→Noisy, Noisy→Noisy, Noisy→Clean

Usage:
    uv run python python/run_all_experiments.py
    uv run python python/run_all_experiments.py --seeds 5
    uv run python python/run_all_experiments.py --datasets raw cleaned
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
from configs.loader import load_config
from models.baselines import get_baseline, train_baseline
from models.cnn1d import EITConv1D
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data.load_dataset import load_mat_dataset, prepare_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

DATASETS = {
    "raw": Path("data/eit_dataset.mat"),
    "cleaned": Path("data/cleaned/eit_cleaned.mat"),
    "pca": Path("data/cleaned/eit_cleaned_pca.mat"),
    "lda": Path("data/cleaned/eit_cleaned_lda.mat"),
}

ALL_MODELS = ["cnn1d", "svm", "random_forest", "mlp"]
CNN_MIN_FEATURES = 8

CONDITIONS = [
    ("clean_train_clean_eval", "clean", "clean"),
    ("clean_train_noisy_eval", "clean", "noisy"),
    ("noisy_train_noisy_eval", "noisy", "noisy"),
    ("noisy_train_clean_eval", "noisy", "clean"),
]

CLASS_NAMES = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed",
]


# ── Training helpers ──────────────────────────────────────────────────


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 40,
    device: str = "auto",
) -> tuple[EITConv1D, dict]:
    """Train CNN and return model + history."""
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = EITConv1D(n_features=n_features, n_classes=n_classes).to(device)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history


def predict_cnn(model: EITConv1D, X: np.ndarray, device: str = "auto") -> np.ndarray:
    """Get CNN predictions."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    return logits.argmax(dim=1).cpu().numpy()


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
        # Check feature count
        X_check, _ = load_mat_dataset(ds_path, use_noisy=False)
        n_features = X_check.shape[1]
        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]
        total_experiments += len(models_for_ds) * len(CONDITIONS) * len(seeds)

    logger.info(f"Total experiments to run: {total_experiments}")
    experiment_idx = 0

    for ds_name in datasets:
        ds_path = DATASETS[ds_name]
        if not ds_path.exists():
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"DATASET: {ds_name} ({ds_path})")
        logger.info(f"{'='*70}")

        # Load both clean and noisy versions
        X_clean, y = load_mat_dataset(ds_path, use_noisy=False)
        X_noisy, _ = load_mat_dataset(ds_path, use_noisy=True)
        n_features = X_clean.shape[1]
        n_samples = X_clean.shape[0]

        logger.info(f"  Samples: {n_samples}, Features: {n_features}")

        # Determine which models can run
        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]
        if "cnn1d" not in models_for_ds:
            logger.info(
                f"  Skipping CNN (requires ≥{CNN_MIN_FEATURES} features, got {n_features})"
            )

        for model_name in models_for_ds:
            for condition_name, train_key, eval_key in CONDITIONS:
                seed_results: list[dict] = []

                for seed in seeds:
                    experiment_idx += 1
                    logger.info(
                        f"  [{experiment_idx}/{total_experiments}] "
                        f"{model_name} | {condition_name} | seed={seed}"
                    )

                    # Set seeds
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # Prepare splits for both clean and noisy
                    ds_clean = prepare_splits(X_clean, y, random_state=seed)
                    ds_noisy = prepare_splits(X_noisy, y, random_state=seed)

                    # Select train/eval data
                    splits = {"clean": ds_clean, "noisy": ds_noisy}
                    train_ds = splits[train_key]
                    eval_ds = splits[eval_key]

                    # Train
                    start_time = time.time()
                    if model_name == "cnn1d":
                        model, history = train_cnn_model(
                            train_ds.X_train,
                            train_ds.y_train,
                            train_ds.X_val,
                            train_ds.y_val,
                            epochs=epochs,
                            early_stopping_patience=early_stopping_patience,
                        )
                        y_pred = predict_cnn(model, eval_ds.X_test)
                        stopped_epoch = len(history["train_loss"])
                    else:
                        model = get_baseline(model_name, random_state=seed)
                        model = train_baseline(
                            model, train_ds.X_train, train_ds.y_train
                        )
                        y_pred = model.predict(eval_ds.X_test)
                        stopped_epoch = None

                    train_time = time.time() - start_time

                    # Evaluate
                    metrics = evaluate_predictions(eval_ds.y_test, y_pred)
                    metrics["train_time_s"] = train_time
                    if stopped_epoch is not None:
                        metrics["stopped_epoch"] = stopped_epoch

                    seed_results.append(metrics)

                # Aggregate across seeds
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
                    "train_time_mean_s": float(
                        np.mean([r["train_time_s"] for r in seed_results])
                    ),
                }
                if "stopped_epoch" in seed_results[0]:
                    epochs_stopped = [r["stopped_epoch"] for r in seed_results]
                    result_entry["stopped_epoch_mean"] = float(np.mean(epochs_stopped))

                all_results.append(result_entry)

                logger.info(
                    f"    → Acc: {result_entry['accuracy_mean']:.4f} ± {result_entry['accuracy_std']:.4f} | "
                    f"F1: {result_entry['f1_macro_mean']:.4f} ± {result_entry['f1_macro_std']:.4f}"
                )

    # Now generate figures for best seed (seed[0]) of each key combination
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FIGURES (using first seed)")
    logger.info("=" * 70)
    _generate_all_figures(
        datasets, seeds[0], epochs, early_stopping_patience, figures_dir
    )

    return pd.DataFrame(all_results)


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

    for ds_name in datasets:
        ds_path = DATASETS[ds_name]
        if not ds_path.exists():
            continue

        X_clean, y = load_mat_dataset(ds_path, use_noisy=False)
        X_noisy, _ = load_mat_dataset(ds_path, use_noisy=True)
        n_features = X_clean.shape[1]

        models_for_ds = [
            m for m in ALL_MODELS if m != "cnn1d" or n_features >= CNN_MIN_FEATURES
        ]

        ds_clean = prepare_splits(X_clean, y, random_state=seed)
        ds_noisy = prepare_splits(X_noisy, y, random_state=seed)
        splits = {"clean": ds_clean, "noisy": ds_noisy}

        for model_name in models_for_ds:
            for condition_name, train_key, eval_key in CONDITIONS:
                fig_output = figures_dir / ds_name / model_name / condition_name
                fig_output.mkdir(parents=True, exist_ok=True)

                train_ds = splits[train_key]
                eval_ds = splits[eval_key]

                np.random.seed(seed)
                torch.manual_seed(seed)

                if model_name == "cnn1d":
                    model, history = train_cnn_model(
                        train_ds.X_train,
                        train_ds.y_train,
                        train_ds.X_val,
                        train_ds.y_val,
                        epochs=epochs,
                        early_stopping_patience=early_stopping_patience,
                    )
                    y_pred = predict_cnn(model, eval_ds.X_test)
                    plot_training_curves(history, fig_output, condition_name)
                else:
                    model = get_baseline(model_name, random_state=seed)
                    model = train_baseline(model, train_ds.X_train, train_ds.y_train)
                    y_pred = model.predict(eval_ds.X_test)

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
    report_lines: list[str] = []
    report_lines.append("# EIT Touch Classification — Experiment Report")
    report_lines.append(
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(f"**Total runtime**: {runtime_s / 60:.1f} minutes")
    report_lines.append(f"**Seeds**: {df['n_seeds'].iloc[0]}")
    report_lines.append(f"**Datasets**: {', '.join(df['dataset'].unique())}")
    report_lines.append(f"**Models**: {', '.join(df['model'].unique())}")
    report_lines.append(f"**Conditions**: {len(CONDITIONS)}")
    report_lines.append(f"**Total experiments**: {len(df)}")

    # ── Summary: Best models per condition ──
    report_lines.append("\n---\n## 1. Best Models by Condition\n")
    for condition_name, _, _ in CONDITIONS:
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

    # CNN raw vs cleaned on noisy eval
    report_lines.append("### 3.1 CNN: Raw vs Feature-Engineered (noisy→noisy)\n")
    cnn_noisy = df[
        (df["model"] == "cnn1d") & (df["condition"] == "noisy_train_noisy_eval")
    ]
    if not cnn_noisy.empty:
        report_lines.append("| Dataset | Accuracy | F1 |")
        report_lines.append("|---------|----------|-----|")
        for _, row in cnn_noisy.iterrows():
            report_lines.append(
                f"| {row['dataset']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
                f"{row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} |"
            )
    else:
        report_lines.append("_No CNN results for noisy→noisy condition._")

    # Best model per dataset on clean→clean
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

    # Robustness: noisy train helps?
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

    # Feature engineering impact
    report_lines.append(
        "\n### 3.4 Feature Engineering Impact (clean→clean baseline comparison)\n"
    )
    report_lines.append("| Model | Raw Acc | Cleaned Acc | PCA Acc | LDA Acc |")
    report_lines.append("|-------|---------|-------------|---------|---------|")
    cc = df[df["condition"] == "clean_train_clean_eval"]
    for model_name in cc["model"].unique():
        model_cc = cc[cc["model"] == model_name]
        row_str = f"| {model_name} |"
        for ds in ["raw", "cleaned", "pca", "lda"]:
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
                f"| {row['dataset']} | {row['condition']} | {ep_str} | {row['train_time_mean_s']:.1f} |"
            )

    # ── Statistical summary ──
    report_lines.append("\n---\n## 5. Statistical Summary\n")
    report_lines.append("Uncertainty reported as mean ± std over 3 random seeds.")
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
            "All experiments show low variance (std ≤ 0.02) — good reproducibility."
        )

    # ── Conclusions ──
    report_lines.append("\n---\n## 6. Key Findings\n")

    # Find overall best noisy performer
    noisy_eval = df[df["eval_data"] == "noisy"]
    if not noisy_eval.empty:
        best_noisy = noisy_eval.loc[noisy_eval["accuracy_mean"].idxmax()]
        report_lines.append(
            f"1. **Best noisy-domain performer**: {best_noisy['model']} on {best_noisy['dataset']} "
            f"({best_noisy['condition']}) — {best_noisy['accuracy_mean']:.4f} accuracy"
        )

    # Find overall best clean performer
    clean_eval = df[df["eval_data"] == "clean"]
    if not clean_eval.empty:
        best_clean = clean_eval.loc[clean_eval["accuracy_mean"].idxmax()]
        report_lines.append(
            f"2. **Best clean-domain performer**: {best_clean['model']} on {best_clean['dataset']} "
            f"({best_clean['condition']}) — {best_clean['accuracy_mean']:.4f} accuracy"
        )

    # Feature engineering effect
    raw_cc = df[
        (df["dataset"] == "raw") & (df["condition"] == "clean_train_clean_eval")
    ]
    cleaned_cc = df[
        (df["dataset"] == "cleaned") & (df["condition"] == "clean_train_clean_eval")
    ]
    if not raw_cc.empty and not cleaned_cc.empty:
        raw_best = raw_cc["accuracy_mean"].max()
        cleaned_best = cleaned_cc["accuracy_mean"].max()
        if cleaned_best > raw_best:
            report_lines.append(
                f"3. **Feature engineering on clean data**: Improved ceiling by "
                f"{(cleaned_best - raw_best)*100:.1f}pp"
            )
        else:
            report_lines.append(
                f"3. **Feature engineering on clean data**: Decreased ceiling by "
                f"{(raw_best - cleaned_best)*100:.1f}pp"
            )

    report_lines.append(
        "\n---\n*Report generated by `python/run_all_experiments.py`*\n"
    )

    # Write report
    report_path = output_dir / "experiment_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all EIT classification experiments with uncertainty."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=["raw", "cleaned", "pca", "lda"],
        help="Which datasets to include.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    total_time = time.time() - start_time

    # Save consolidated CSV
    csv_path = args.output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Consolidated results saved to {csv_path}")

    # Save as JSON for programmatic access
    json_path = args.output_dir / "all_results.json"
    df.to_json(json_path, orient="records", indent=2)

    # Generate pivot tables
    # Accuracy pivot
    acc_pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="condition",
        values="accuracy_mean",
        aggfunc="first",
    )
    acc_pivot_path = args.output_dir / "accuracy_pivot.csv"
    acc_pivot.to_csv(acc_pivot_path)

    # F1 pivot
    f1_pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="condition",
        values="f1_macro_mean",
        aggfunc="first",
    )
    f1_pivot_path = args.output_dir / "f1_pivot.csv"
    f1_pivot.to_csv(f1_pivot_path)

    logger.info(f"Pivot tables saved to {args.output_dir}/")

    # Generate report
    generate_report(df, args.output_dir, total_time)

    logger.info("\n" + "=" * 70)
    logger.info(f"ALL EXPERIMENTS COMPLETE — {total_time / 60:.1f} minutes total")
    logger.info(f"Report: {args.output_dir / 'experiment_report.md'}")
    logger.info(f"CSV: {csv_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
