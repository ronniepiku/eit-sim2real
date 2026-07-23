"""Generate a publication-ready figure for the EITConv1D architecture.

The figure is saved in raster and vector formats so it can be used
directly in the dissertation without relying on wide TikZ diagrams.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from eit_sim2real.models.cnn1d import EITConv1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/methodology"),
        help="Directory where the figure files should be written.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=208,
        help="Input feature length used for the diagram.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability to display in the classifier head.",
    )
    return parser.parse_args()


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.1,
        edgecolor="#233043",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.73,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#14202b",
    )
    ax.text(
        x + width / 2,
        y + height * 0.33,
        body,
        ha="center",
        va="center",
        fontsize=10,
        color="#14202b",
        linespacing=1.45,
    )


def add_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.3,
        color="#4a5568",
    )
    ax.add_patch(arrow)


def build_diagram(n_features: int, dropout: float) -> plt.Figure:
    _ = EITConv1D(n_features=n_features, dropout=dropout)

    fig, ax = plt.subplots(figsize=(8.4, 12.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    x = 0.16
    width = 0.66
    height = 0.094
    ys = [0.87, 0.715, 0.56, 0.405, 0.25, 0.095, -0.06]

    boxes = [
        (
            "Input",
            f"Voltage-difference vector\n(B, 1, {n_features})",
            "#eef2ff",
        ),
        (
            "Conv Block 1",
            "Conv1D 1→32, k=5, p=2\nBN + ReLU + MaxPool(2)\n(B, 32, 104)",
            "#dbeafe",
        ),
        (
            "Conv Block 2",
            "Conv1D 32→64, k=5, p=2\nBN + ReLU + MaxPool(2)\n(B, 64, 52)",
            "#dbeafe",
        ),
        (
            "Conv Block 3",
            "Conv1D 64→128, k=5, p=2\nBN + ReLU + MaxPool(2)\n(B, 128, 26)",
            "#dbeafe",
        ),
        (
            "Adaptive Pool",
            "AdaptiveAvgPool1d(1)\n(B, 128, 1)",
            "#dcfce7",
        ),
        (
            "Classifier Head",
            f"Flatten → Linear 128→128\nReLU + Dropout(p={dropout:.1f})\n(B, 128)",
            "#fef3c7",
        ),
        (
            "Output Layer",
            "Linear 128→5\nLogits: (B, 5)",
            "#fee2e2",
        ),
    ]

    for (title, body, facecolor), y in zip(boxes, ys, strict=True):
        add_box(ax, x, y, width, height, title, body, facecolor)

    for top_y, bottom_y in zip(ys[:-1], ys[1:], strict=True):
        add_arrow(ax, 0.5, top_y, 0.5, bottom_y + height)

    return fig


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_diagram(args.n_features, args.dropout)
    stem = output_dir / "cnn_architecture"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved architecture figure to {stem.with_suffix('.png')}")
    print(f"Saved architecture figure to {stem.with_suffix('.pdf')}")
    print(f"Saved architecture figure to {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()