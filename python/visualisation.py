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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

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
