"""Visualisation utilities for EIT touch classification results.

Provides publication-quality plotting functions for:
- Training history curves (loss and accuracy)
- Confusion matrices (normalised and raw)
- Per-class F1 bar charts
- Robustness degradation curves under varying noise
- PCA / t-SNE feature-space scatter plots
- Ablation study heatmaps

All functions return ``matplotlib.figure.Figure`` objects so that callers
can either save or display them as needed.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.metrics import (
    confusion_matrix as cm_sklearn,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Consistent style across all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

CLASS_NAMES = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed",
]


# ── Training history ───────────────────────────────────────────────────


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
) -> plt.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
                 ``train_acc``, ``val_acc``.
        title: Figure title.

    Returns:
        Matplotlib figure with two subplots.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        class_names: Class labels for axes.
        normalize: If True, show row-normalised percentages.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Per-class F1 ──────────────────────────────────────────────────────


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Per-Class F1 Score",
) -> plt.Figure:
    """Bar chart of per-class F1 scores.

    Args:
        y_true: Ground truth labels (0-indexed).
        y_pred: Predicted labels.
        class_names: Class labels.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    f1_scores = [report[name]["f1-score"] for name in class_names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(class_names, f1_scores, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


# ── Robustness degradation ───────────────────────────────────────────


def plot_robustness(
    robustness: dict[str, list[float]],
    title: str = "Robustness Under Noise",
) -> plt.Figure:
    """Plot accuracy and F1 vs. noise level.

    Args:
        robustness: Dict with ``noise_levels``, ``accuracies``, ``f1_scores``.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
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


def plot_robustness_comparison(
    results: dict[str, dict[str, list[float]]],
    metric: str = "accuracies",
    title: str = "Robustness Comparison",
) -> plt.Figure:
    """Compare robustness across multiple models.

    Args:
        results: ``{model_name: robustness_dict}`` mapping.
        metric: Key to plot (``accuracies`` or ``f1_scores``).
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, rob in results.items():
        ax.plot(rob["noise_levels"], rob[metric], "o-", label=name, linewidth=2)

    ylabel = "Accuracy" if metric == "accuracies" else "F1 (macro)"
    ax.set_xlabel("Noise σ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


# ── Feature space visualisation ──────────────────────────────────────


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "PCA – EIT Feature Space",
) -> plt.Figure:
    """2D PCA scatter plot of feature vectors coloured by class.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels (0-indexed).
        class_names: Class labels for the legend.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none",
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
    title: str = "t-SNE – EIT Feature Space",
) -> plt.Figure:
    """2D t-SNE scatter plot of feature vectors coloured by class.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels (0-indexed).
        class_names: Class labels.
        perplexity: t-SNE perplexity parameter.
        seed: Random state for reproducibility.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    fig.tight_layout()
    return fig


# ── Ablation heatmap ─────────────────────────────────────────────────


def plot_ablation_heatmap(
    csv_path: Path | str,
    metric: str = "test_acc",
    title: str = "Ablation Study – Test Accuracy",
) -> plt.Figure:
    """Heatmap of ablation study results.

    Rows = noise configurations, columns = metrics or models.

    Args:
        csv_path: Path to CSV file saved by :meth:`AblationStudy.save`.
        metric: Column to visualise.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    df = pd.read_csv(csv_path)

    # Pivot: rows = description, value = metric
    pivot = df.pivot_table(
        index="description",
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric},
    )
    ax.set_title(title)
    ax.set_ylabel("Noise Configuration")
    fig.tight_layout()
    return fig


# ── Convenience save helper ──────────────────────────────────────────


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> None:
    """Save a figure in multiple formats.

    Args:
        fig: Matplotlib figure to save.
        path: Base path (extension is replaced per format).
        dpi: Resolution for raster formats.
        formats: Iterable of file extensions to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


# ── Direct plotting with file saving ──────────────────────────────────


def plot_training_curves(
    history: dict[str, list[float]], output_dir: Path, noise_tag: str
) -> None:
    """Plot training and validation loss/accuracy curves and save to file.

    Args:
        history: Training history dictionary.
        output_dir: Directory to save plots.
        noise_tag: Noise tag for file naming.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(
        "Training and Validation Accuracy", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"cnn1d_{noise_tag}_training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Training curves saved to {output_path}")
    plt.close()


def plot_confusion_matrix_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
) -> None:
    """Plot confusion matrix and save to file.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_dir: Directory to save plots.
        model_name: Name of the model.
        noise_tag: Noise tag for file naming.
        split_name: Name of the data split (e.g., 'val', 'test').
    """
    cm = cm_sklearn(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        ax=ax,
        square=True,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix - {model_name} ({split_name})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = output_dir / f"{model_name}_{noise_tag}_cm_{split_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_roc_curves_and_save(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
    n_classes: int = 5,
) -> None:
    """Plot ROC curves for multiclass classification (One-vs-Rest) and save.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities for each class.
        output_dir: Directory to save plots.
        model_name: Name of the model.
        noise_tag: Noise tag for file naming.
        split_name: Name of the data split.
        n_classes: Number of classes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f"Class {i} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"ROC Curves - {model_name} ({split_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{model_name}_{noise_tag}_roc_{split_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"ROC curves saved to {output_path}")
    plt.close()


def plot_precision_recall_curves_and_save(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
    n_classes: int = 5,
) -> None:
    """Plot Precision-Recall curves for multiclass classification and save.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities for each class.
        output_dir: Directory to save plots.
        model_name: Name of the model.
        noise_tag: Noise tag for file naming.
        split_name: Name of the data split.
        n_classes: Number of classes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, i])

        ax.plot(
            recall,
            precision,
            color=colors[i],
            lw=2,
            label=f"Class {i}",
        )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Precision-Recall Curves - {model_name} ({split_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{model_name}_{noise_tag}_pr_curves_{split_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Precision-Recall curves saved to {output_path}")
    plt.close()


def plot_per_class_metrics_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str,
    noise_tag: str,
    split_name: str = "test",
) -> None:
    """Plot per-class precision, recall, and F1-score and save.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_dir: Directory to save plots.
        model_name: Name of the model.
        noise_tag: Noise tag for file naming.
        split_name: Name of the data split.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    n_classes = len(np.unique(y_true))
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    ax.bar(x, recall, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Per-Class Metrics - {model_name} ({split_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {i}" for i in range(n_classes)])
    ax.legend(fontsize=11)
    ax.set_ylim((0, 1.1))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = (
        output_dir / f"{model_name}_{noise_tag}_per_class_metrics_{split_name}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Per-class metrics saved to {output_path}")
    plt.close()
