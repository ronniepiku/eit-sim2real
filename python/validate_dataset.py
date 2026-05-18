"""Comprehensive validation report for MATLAB-generated EIT datasets.

This script produces dissertation-ready figures and tables to assess:
1) dataset integrity and class balance,
2) signal-scale and per-class statistics,
3) feature-space separability using PCA,
4) class-level similarity structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# Consistent plotting style for report figures.
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "eit_dataset_numpy.mat"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results" / "dataset_validation"
DEFAULT_RECONSTRUCTION_DIR = DEFAULT_OUTPUT_DIR / "reconstructions"

DEFAULT_CLASS_NAMES = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed",
]

POLAR_IMAGE_SHAPE = (16, 13)


def load_dataset(data_path: Path | str, use_noisy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load EIT feature matrix and labels from a MATLAB .mat file."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    mat = sio.loadmat(str(data_path))
    if "dataset_y" not in mat:
        raise KeyError("Missing 'dataset_y' in MAT file")

    key = "dataset_X_noisy" if use_noisy else "dataset_X_clean"
    if key not in mat:
        key = "dataset_X"
    if key not in mat:
        raise KeyError(
            "Missing feature matrix key in MAT file. Expected one of "
            "'dataset_X_noisy', 'dataset_X_clean', or 'dataset_X'."
        )

    X = np.asarray(mat[key], dtype=np.float32)
    y_raw = np.asarray(mat["dataset_y"], dtype=np.int64).ravel()
    y = normalize_labels(y_raw)
    return X, y


def normalize_labels(y: np.ndarray) -> np.ndarray:
    """Return 0-indexed labels from either MATLAB (1-indexed) or Python format."""
    min_label = int(y.min())
    if min_label == 0:
        return y
    if min_label == 1:
        return y - 1
    raise ValueError(f"Unsupported label encoding with min label {min_label}")


def class_name_map(unique_labels: np.ndarray) -> dict[int, str]:
    """Map integer labels to readable names."""
    mapping: dict[int, str] = {}
    for label in unique_labels:
        idx = int(label)
        mapping[idx] = DEFAULT_CLASS_NAMES[idx] if idx < len(DEFAULT_CLASS_NAMES) else f"Class {idx}"
    return mapping


def vector_to_polar_grid(vector: np.ndarray) -> np.ndarray:
    """Reshape a 1D measurement vector into a polar display grid.

    The dataset uses a 16-electrode adjacent pattern, so the 208 measurements
    are displayed as 16 angular sectors and 13 radial bins. This is a visual
    inspection aid rather than a physical reconstruction.
    """
    vector = np.asarray(vector)
    if vector.ndim == 2:
        return vector
    rows, cols = POLAR_IMAGE_SHAPE
    if vector.size != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} features for polar display, got {vector.size}."
        )
    return vector.reshape(rows, cols)


def reconstruction_figure_paths(reconstruction_dir: Path) -> dict[str, Path]:
    """Return expected reconstruction figure paths produced by MATLAB."""
    return {
        "clean_random": reconstruction_dir / "random_reconstructed_class_images_clean.png",
        "clean_mean": reconstruction_dir / "mean_reconstructed_class_images_clean.png",
        "noisy_random": reconstruction_dir / "random_reconstructed_class_images_noisy.png",
        "noisy_mean": reconstruction_dir / "mean_reconstructed_class_images_noisy.png",
    }


def available_reconstruction_figures(reconstruction_dir: Path) -> dict[str, Path]:
    """Collect reconstruction figures that already exist on disk."""
    expected = reconstruction_figure_paths(reconstruction_dir)
    return {name: path for name, path in expected.items() if path.exists()}


def sample_level_summary(X: np.ndarray) -> pd.DataFrame:
    """Create per-sample scalar descriptors to support compact statistics plots."""
    abs_X = np.abs(X)
    return pd.DataFrame(
        {
            "sample_mean": X.mean(axis=1),
            "sample_std": X.std(axis=1),
            "sample_l2": np.linalg.norm(X, axis=1),
            "sample_abs_mean": abs_X.mean(axis=1),
            "sample_abs_max": abs_X.max(axis=1),
        }
    )


def compute_tables(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
) -> dict[str, pd.DataFrame]:
    """Build all tabular outputs used in the report."""
    n_samples, n_features = X.shape

    dist_counts = pd.Series(y).value_counts().sort_index()
    distribution = pd.DataFrame(
        {
            "class_label": dist_counts.index.astype(int),
            "class_name": [name_by_label[int(lbl)] for lbl in dist_counts.index],
            "count": dist_counts.values,
            "percentage": 100.0 * dist_counts.values / n_samples,
        }
    )

    overview = pd.DataFrame(
        {
            "n_samples": [n_samples],
            "n_features": [n_features],
            "n_classes": [len(labels)],
            "min_value": [float(np.min(X))],
            "max_value": [float(np.max(X))],
            "mean_value": [float(np.mean(X))],
            "std_value": [float(np.std(X))],
            "has_nan": [bool(np.isnan(X).any())],
            "has_inf": [bool(np.isinf(X).any())],
            "duplicated_rows": [int(pd.DataFrame(X).duplicated().sum())],
        }
    )

    rows: list[dict[str, float | int | str]] = []
    for label in labels:
        X_cls = X[y == label]
        rows.append(
            {
                "class_label": int(label),
                "class_name": name_by_label[int(label)],
                "n_samples": int(X_cls.shape[0]),
                "min": float(np.min(X_cls)),
                "max": float(np.max(X_cls)),
                "mean": float(np.mean(X_cls)),
                "std": float(np.std(X_cls)),
                "p01": float(np.percentile(X_cls, 1)),
                "p50": float(np.percentile(X_cls, 50)),
                "p99": float(np.percentile(X_cls, 99)),
                "mean_l2": float(np.linalg.norm(X_cls, axis=1).mean()),
            }
        )
    per_class_stats = pd.DataFrame(rows)

    sample_stats = sample_level_summary(X)
    sample_stats.insert(0, "class_label", y)
    sample_stats.insert(1, "class_name", [name_by_label[int(lbl)] for lbl in y])

    centroid_rows = []
    centroids = []
    for label in labels:
        centroid = X[y == label].mean(axis=0)
        centroids.append(centroid)
        centroid_rows.append(
            {
                "class_label": int(label),
                "class_name": name_by_label[int(label)],
                "centroid_l2": float(np.linalg.norm(centroid)),
            }
        )
    centroid_df = pd.DataFrame(centroid_rows)

    centroid_matrix = pairwise_distances(np.vstack(centroids), metric="euclidean")
    centroid_distance = pd.DataFrame(
        centroid_matrix,
        index=[name_by_label[int(lbl)] for lbl in labels],
        columns=[name_by_label[int(lbl)] for lbl in labels],
    )

    return {
        "dataset_overview": overview,
        "class_distribution": distribution,
        "per_class_signal_stats": per_class_stats,
        "sample_level_stats": sample_stats,
        "centroid_norms": centroid_df,
        "centroid_distance_matrix": centroid_distance,
    }


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 300) -> None:
    """Save figure as PNG and PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(distribution: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart of class counts with percentages."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(distribution["class_name"], distribution["count"], edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=20)
    for bar, pct in zip(bars, distribution["percentage"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    save_figure(fig, fig_dir, "class_distribution")


def plot_sample_descriptor_boxplots(sample_stats: pd.DataFrame, fig_dir: Path) -> None:
    """Compare compact signal descriptors per class."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    columns = ["sample_l2", "sample_std", "sample_abs_mean"]
    ylabels = ["L2 norm", "Within-sample std", "Mean |voltage|"]

    for ax, column, ylabel in zip(axes, columns, ylabels, strict=True):
        sns.boxplot(data=sample_stats, x="class_name", y=column, ax=ax, showfliers=False)
        ax.set_xlabel("Class")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Per-Sample Signal Descriptor Distributions", y=1.04)
    fig.tight_layout()
    save_figure(fig, fig_dir, "sample_descriptor_boxplots")


def plot_voltage_traces_by_class(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
    fig_dir: Path,
    rng: np.random.Generator,
    max_traces_per_class: int,
) -> None:
    """Overlay random traces and class mean for each class."""
    n_classes = len(labels)
    ncols = 2
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        X_cls = X[y == label]
        n_pick = min(max_traces_per_class, X_cls.shape[0])
        pick = rng.choice(X_cls.shape[0], size=n_pick, replace=False)
        for row in X_cls[pick]:
            ax.plot(row, color="tab:blue", alpha=0.15, linewidth=0.8)

        mean_trace = X_cls.mean(axis=0)
        std_trace = X_cls.std(axis=0)
        x_axis = np.arange(X_cls.shape[1])
        ax.plot(x_axis, mean_trace, color="black", linewidth=2, label="Class mean")
        ax.fill_between(
            x_axis,
            mean_trace - std_trace,
            mean_trace + std_trace,
            color="tab:orange",
            alpha=0.25,
            label="Mean +/- 1 std",
        )
        ax.set_title(name_by_label[int(label)])
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Voltage")
        ax.grid(alpha=0.25)

    for idx in range(n_classes, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles, labels_plot = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_plot, loc="upper center", ncol=2)
    fig.suptitle("Voltage Trace Profiles by Class", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "voltage_traces_by_class")


def plot_pca_outputs(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
    fig_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Create PCA scatter and scree/cumulative variance plots."""
    n_comp = min(20, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    ratios = pca.explained_variance_ratio_

    fig_scatter, ax = plt.subplots(figsize=(8.5, 6.5))
    for label in labels:
        mask = y == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=16,
            alpha=0.65,
            label=name_by_label[int(label)],
        )
    ax.set_xlabel(f"PC1 ({ratios[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.1%} variance)")
    ax.set_title("PCA Projection of EIT Dataset")
    ax.legend(markerscale=1.5)
    ax.grid(alpha=0.25)
    fig_scatter.tight_layout()
    save_figure(fig_scatter, fig_dir, "pca_scatter_2d")

    fig_var, ax_var = plt.subplots(figsize=(8.5, 4.8))
    ax_var.bar(np.arange(1, len(ratios) + 1), ratios, alpha=0.8, label="Per-component")
    ax_var.plot(
        np.arange(1, len(ratios) + 1),
        np.cumsum(ratios),
        "o-",
        color="black",
        linewidth=1.5,
        markersize=3,
        label="Cumulative",
    )
    ax_var.set_xlabel("Principal component")
    ax_var.set_ylabel("Explained variance ratio")
    ax_var.set_title("PCA Explained Variance")
    ax_var.set_ylim(0, 1.05)
    ax_var.legend()
    fig_var.tight_layout()
    save_figure(fig_var, fig_dir, "pca_explained_variance")

    return X_pca, ratios


def plot_centroid_distance_heatmap(
    tables: dict[str, pd.DataFrame],
    fig_dir: Path,
) -> None:
    """Heatmap of pairwise centroid distances between classes."""
    distance_df = tables["centroid_distance_matrix"]
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    sns.heatmap(
        distance_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        square=True,
        cbar_kws={"label": "Euclidean distance"},
        ax=ax,
    )
    ax.set_title("Inter-Class Centroid Distance Matrix")
    fig.tight_layout()
    save_figure(fig, fig_dir, "centroid_distance_heatmap")


def _plot_polar_panel(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    """Render a single polar measurement image."""
    n_theta, n_r = grid.shape
    theta_edges = np.linspace(0, 2 * np.pi, n_theta + 1)
    r_edges = np.linspace(0.0, 1.0, n_r + 1)
    theta_mesh, r_mesh = np.meshgrid(theta_edges, r_edges)

    ax.grid(False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.pcolormesh(
        theta_mesh,
        r_mesh,
        grid.T,
        cmap="viridis",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticklabels([])
    ax.set_title(title, va="bottom")


def plot_random_class_images(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
    fig_dir: Path,
    rng: np.random.Generator,
) -> None:
    """Show one randomly selected circular measurement image per class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw={"projection": "polar"})
    axes_flat = axes.ravel()
    vmin = float(np.percentile(X, 1))
    vmax = float(np.percentile(X, 99))

    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        class_samples = np.flatnonzero(y == label)
        sample_idx = int(rng.choice(class_samples))
        image = vector_to_polar_grid(X[sample_idx])
        _plot_polar_panel(ax, image, name_by_label[int(label)], vmin, vmax)

    for idx in range(len(labels), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Random Circular Measurement Images by Class", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "random_class_measurement_images")


def plot_mean_class_images(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
    fig_dir: Path,
) -> None:
    """Show the average circular measurement image for each class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw={"projection": "polar"})
    axes_flat = axes.ravel()
    class_means = [X[y == label].mean(axis=0) for label in labels]
    vmin = float(np.min(class_means))
    vmax = float(np.max(class_means))

    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        image = vector_to_polar_grid(class_means[idx])
        _plot_polar_panel(ax, image, f"Mean: {name_by_label[int(label)]}", vmin, vmax)

    for idx in range(len(labels), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Class Mean Circular Measurement Images", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_dir, "mean_class_measurement_images")


def plot_tsne_embedding(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    name_by_label: dict[int, str],
    fig_dir: Path,
    rng: np.random.Generator,
    max_points: int = 5000,
    perplexity: float = 30.0,
) -> None:
    """Create a t-SNE scatter plot using a stratified subset of the dataset.

    t-SNE is run on a manageable subset for speed and readability. A small PCA
    pre-reduction is applied before t-SNE to denoise the feature space.
    """
    per_class = max(1, max_points // len(labels))
    selected_indices: list[int] = []
    for label in labels:
        class_indices = np.flatnonzero(y == label)
        n_pick = min(per_class, class_indices.size)
        selected_indices.extend(rng.choice(class_indices, size=n_pick, replace=False).tolist())

    selected_indices = np.array(selected_indices)
    X_sel = X[selected_indices]
    y_sel = y[selected_indices]

    n_pca = min(50, X_sel.shape[1], X_sel.shape[0] - 1)
    if n_pca >= 2:
        X_sel = PCA(n_components=n_pca, random_state=42).fit_transform(X_sel)

    tsne_perplexity = min(perplexity, max(5.0, (X_sel.shape[0] - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    X_2d = tsne.fit_transform(X_sel)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for label in labels:
        mask = y_sel == label
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=14,
            alpha=0.6,
            label=name_by_label[int(label)],
            edgecolors="none",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE Projection (n={X_sel.shape[0]})")
    ax.legend(markerscale=1.4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, fig_dir, "tsne_embedding")


def save_tables(tables: dict[str, pd.DataFrame], table_dir: Path) -> None:
    """Write all tables as CSV files."""
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=True if "matrix" in name else False)


def write_report_markdown(
    output_dir: Path,
    data_path: Path,
    use_noisy: bool,
    tables: dict[str, pd.DataFrame],
    pca_ratios: np.ndarray,
    reconstruction_figures: dict[str, Path],
) -> None:
    """Create a lightweight narrative summary for direct dissertation use."""
    overview = tables["dataset_overview"].iloc[0]
    class_dist = tables["class_distribution"]
    dominant = class_dist.sort_values("count", ascending=False).iloc[0]
    minority = class_dist.sort_values("count", ascending=True).iloc[0]

    pca2 = float(pca_ratios[:2].sum()) if len(pca_ratios) >= 2 else float(pca_ratios.sum())
    pca5 = float(pca_ratios[:5].sum()) if len(pca_ratios) >= 5 else float(pca_ratios.sum())

    report_path = output_dir / "validation_report.md"
    report_lines = [
        "# Dataset Validation Report",
        "",
        "## Dataset Source",
        f"- Data file: {data_path}",
        f"- Variant: {'Noisy measurements' if use_noisy else 'Clean measurements'}",
        "",
        "## Core Summary",
        f"- Samples: {int(overview['n_samples'])}",
        f"- Features per sample: {int(overview['n_features'])}",
        f"- Classes: {int(overview['n_classes'])}",
        f"- Value range: [{overview['min_value']:.6f}, {overview['max_value']:.6f}]",
        f"- Global mean +/- std: {overview['mean_value']:.6f} +/- {overview['std_value']:.6f}",
        f"- NaN present: {bool(overview['has_nan'])}",
        f"- Inf present: {bool(overview['has_inf'])}",
        f"- Duplicated rows: {int(overview['duplicated_rows'])}",
        "",
        "## Class Balance",
        f"- Largest class: {dominant['class_name']} ({int(dominant['count'])} samples, {dominant['percentage']:.2f}%)",
        f"- Smallest class: {minority['class_name']} ({int(minority['count'])} samples, {minority['percentage']:.2f}%)",
        "",
        "## PCA Separability Indicators",
        f"- Variance captured by first 2 PCs: {100.0 * pca2:.2f}%",
        f"- Variance captured by first 5 PCs: {100.0 * pca5:.2f}%",
        "",
        "## Reconstructed Class Images",
    ]

    if reconstruction_figures:
        for name, path in sorted(reconstruction_figures.items()):
            report_lines.append(f"- {name.replace('_', ' ').title()}: {path}")
    else:
        report_lines.extend(
            [
                "- No reconstruction figures were found. Run matlab/generate_validation_reconstructions.m to generate them.",
                "- Fallback measurement-layout figures may be used for quick inspection, but they are not inverse reconstructions.",
            ]
        )

    report_lines.extend([
        "",
        "## Additional Visuals",
        "- t-SNE embedding: figures/tsne_embedding.(png|pdf)",
        "",
        "## Outputs",
        "- Figures: results/dataset_validation/figures",
        "- Tables: results/dataset_validation/tables",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive EIT dataset validation report")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to MATLAB dataset (.mat)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures/tables/report will be saved",
    )
    parser.add_argument(
        "--reconstruction-dir",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_DIR,
        help="Directory containing MATLAB EIDORS reconstruction figures",
    )
    parser.add_argument(
        "--use-noisy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use noisy dataset_X_noisy (or clean dataset_X_clean)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible trace sampling",
    )
    parser.add_argument(
        "--max-traces-per-class",
        type=int,
        default=30,
        help="Maximum random traces overlaid in each class panel",
    )
    return parser.parse_args()


def validate_dataset(args: argparse.Namespace) -> None:
    """Run complete validation workflow and write all outputs to disk."""
    print("=" * 72)
    print("EIT Dataset Validation Report")
    print("=" * 72)
    print(f"Loading dataset from: {args.data_path}")
    print(f"Using noisy measurements: {args.use_noisy}")

    X, y = load_dataset(args.data_path, use_noisy=args.use_noisy)
    labels = np.unique(y)
    name_by_label = class_name_map(labels)

    output_dir = args.output_dir
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_figures = available_reconstruction_figures(args.reconstruction_dir)

    print(f"Dataset shape: {X.shape}")
    print(f"Classes found: {labels.tolist()}")

    tables = compute_tables(X, y, labels, name_by_label)
    save_tables(tables, table_dir)
    print(f"Saved tables to: {table_dir}")

    rng = np.random.default_rng(args.seed)
    sample_stats = tables["sample_level_stats"]

    plot_class_distribution(tables["class_distribution"], fig_dir)
    plot_sample_descriptor_boxplots(sample_stats, fig_dir)
    plot_voltage_traces_by_class(
        X,
        y,
        labels,
        name_by_label,
        fig_dir,
        rng,
        args.max_traces_per_class,
    )
    _, pca_ratios = plot_pca_outputs(X, y, labels, name_by_label, fig_dir)
    plot_centroid_distance_heatmap(tables, fig_dir)
    if recon_figures:
        print(f"Using {len(recon_figures)} MATLAB reconstruction figure(s) from: {args.reconstruction_dir}")
    else:
        plot_random_class_images(X, y, labels, name_by_label, fig_dir, rng)
        plot_mean_class_images(X, y, labels, name_by_label, fig_dir)
    plot_tsne_embedding(X, y, labels, name_by_label, fig_dir, rng)
    print(f"Saved figures to: {fig_dir}")

    write_report_markdown(
        output_dir,
        args.data_path,
        args.use_noisy,
        tables,
        pca_ratios,
        recon_figures,
    )
    print(f"Saved report to: {output_dir / 'validation_report.md'}")

    print("\nValidation complete.")
    print("Use the CSV tables and vector-PDF figures directly in your dissertation.")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    validate_dataset(args)


if __name__ == "__main__":
    main()
