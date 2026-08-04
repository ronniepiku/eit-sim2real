"""Regenerate the severity-sweep figure from saved sweep results.

The sweep itself is expensive (it retrains one CNN per training regime), so the
plot is rebuilt from ``results/reports/severity_sweep_results.json`` rather than
by re-running the experiment.

Usage:
    python scripts/generate_severity_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "reports" / "severity_sweep_results.json"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Presentation labels for the three training regimes.
REGIME_STYLE = {
    "noisy_fixed": ("Fixed noisy dataset", "#0072B2", "o", "-"),
    "augmented": ("Online augmentation", "#D55E00", "s", "--"),
    "mixed": ("Mixed (30% clean)", "#009E73", "^", "-."),
}


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"Sweep results not found: {RESULTS}")

    with open(RESULTS, encoding="utf-8") as fh:
        data = json.load(fh)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for metric, ax, ylabel in (
        ("accuracies", axes[0], "Accuracy"),
        ("f1_scores", axes[1], "Macro-F1"),
    ):
        for regime, (label, colour, marker, style) in REGIME_STYLE.items():
            if regime not in data:
                continue
            x = data[regime]["severity_multipliers"]
            y = data[regime][metric]
            ax.plot(
                x,
                y,
                marker=marker,
                linestyle=style,
                color=colour,
                label=label,
                linewidth=1.8,
                markersize=5,
            )
        # 1.0x is the severity the fixed-noise model was trained at.
        ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.2)
        ax.annotate(
            "training severity",
            xy=(1.0, ax.get_ylim()[0]),
            xytext=(1.06, 0.02),
            textcoords=("data", "axes fraction"),
            fontsize=8,
            color="grey",
        )
        ax.set_xlabel(r"Noise severity multiplier ($\times$)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(
        "Robustness under severity scaling by training regime",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIG_DIR / f"noise_type_severity_sweep.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
