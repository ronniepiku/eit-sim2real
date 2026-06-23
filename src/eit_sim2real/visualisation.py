"""Visualisation utilities for EIT touch classification results.

Provides publication-quality plotting functions for training curves,
confusion matrices, per-class metrics, robustness curves, and feature
space visualisations.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.metrics import (
    confusion_matrix as cm_sklearn,
)

from eit_sim2real.constants import CLASS_NAMES

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


# ── Core save helper ──────────────────────────────────────────────────


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Save a figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Training curves ───────────────────────────────────────────────────


def plot_training_curves(
    history: dict[str, list[float]], output_dir: Path, noise_tag: str
) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / f"cnn1d_{noise_tag}_training_curves.png")
    logger.info(f"Training curves saved to {output_dir}")


# ── Confusion matrix ──────────────────────────────────────────────────


def plot_confusion_matrix_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
) -> None:
    """Plot confusion matrix and save to file."""
    cm = cm_sklearn(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = CLASS_NAMES[:n_classes] if n_classes <= len(CLASS_NAMES) else None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        ax=ax,
        square=True,
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name} ({split_name})")
    fig.tight_layout()
    save_figure(fig, output_dir / f"{model_name}_{noise_tag}_cm_{split_name}.png")


# ── Per-class metrics ─────────────────────────────────────────────────


def plot_per_class_metrics_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
) -> None:
    """Plot per-class precision, recall, and F1-score and save."""
    n_classes = len(np.unique(y_true))
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, prec, width, label="Precision", alpha=0.8)
    ax.bar(x, rec, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics - {model_name} ({split_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {i}" for i in range(n_classes)])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(
        fig, output_dir / f"{model_name}_{noise_tag}_per_class_metrics_{split_name}.png"
    )


# ── ROC curves ────────────────────────────────────────────────────────


def plot_roc_curves_and_save(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
    n_classes: int = 5,
) -> None:
    """Plot One-vs-Rest ROC curves and save."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab10")

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors(i / n_classes),
            lw=2,
            label=f"Class {i} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {model_name} ({split_name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir / f"{model_name}_{noise_tag}_roc_{split_name}.png")


# ── Precision-Recall curves ───────────────────────────────────────────


def plot_precision_recall_curves_and_save(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
    n_classes: int = 5,
) -> None:
    """Plot Precision-Recall curves and save."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab10")

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
        ax.plot(
            recall, precision, color=colors(i / n_classes), lw=2, label=f"Class {i}"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves - {model_name} ({split_name})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(
        fig, output_dir / f"{model_name}_{noise_tag}_pr_curves_{split_name}.png"
    )


# ── Feature space visualisation ───────────────────────────────────────


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "PCA - EIT Feature Space",
) -> plt.Figure:
    """2D PCA scatter plot coloured by class."""
    if class_names is None:
        class_names = CLASS_NAMES

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.6, s=15, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(markerscale=3)
    fig.tight_layout()
    return fig


def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    perplexity: float = 30.0,
    seed: int = 42,
    title: str = "t-SNE - EIT Feature Space",
) -> plt.Figure:
    """2D t-SNE scatter plot coloured by class."""
    if class_names is None:
        class_names = CLASS_NAMES

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.6, s=15, edgecolors="none"
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    fig.tight_layout()
    return fig


# ── Robustness curves ─────────────────────────────────────────────────


def plot_robustness(
    robustness: dict[str, list[float]],
    title: str = "Robustness Under Noise",
) -> plt.Figure:
    """Plot accuracy and F1 vs. noise level."""
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = robustness["noise_levels"]
    ax.plot(levels, robustness["accuracies"], "o-", label="Accuracy", linewidth=2)
    ax.plot(levels, robustness["f1_scores"], "s--", label="F1 (macro)", linewidth=2)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


# ── Ablation heatmap ─────────────────────────────────────────────────


def plot_ablation_heatmap(
    csv_path: Path | str,
    metric: str = "test_acc",
    title: str = "Ablation Study - Test Accuracy",
) -> plt.Figure:
    """Heatmap of ablation study results from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(
        index="description", columns="model", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_title(title)
    ax.set_ylabel("Noise Configuration")
    fig.tight_layout()
    return fig
