"""Ablation study runner for EIT touch classification.

Systematically trains and evaluates models with different noise component
combinations to identify which noise sources most impact robustness.

Experiments:
1. **Core mismatch** - 4 conditions: clean→clean, clean→noisy, noisy→noisy,
   noisy→clean (baseline comparisons).
2. **Single-component isolation** - Train/evaluate with only ONE noise type
   active at a time to measure individual component impact.
3. **Exhaustive subset/order ablation** - All non-empty component subsets
   with physically-constrained orderings (optional, expensive).
4. **Per-component severity sweep** - For each noise type, sweep severity
   [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] while others stay at 1.0.

Outputs:
- results/tables/ablation_results.csv
- results/reports/ablation_report.md
- results/figures/ablation/ (heatmaps, bar charts, severity curves)
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score

from eit_sim2real.configs import load_config
from eit_sim2real.constants import COMPONENT_LABELS, NOISE_COMPONENTS
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.data.noise import (
    NoiseConfig,
    apply_noise_batch_vectorised,
    apply_noise_in_scaled_space,
)
from eit_sim2real.experiments.protocols import NOISY_CNN_PARAMS
from eit_sim2real.models import get_baseline, train_baseline
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.train import train_cnn
from eit_sim2real.utils import get_device, predict_cnn

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


@dataclass
class AblationResult:
    """Result from a single ablation experiment."""

    noise_config: dict[str, bool]
    model_name: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    test_f1_macro: float
    description: str = ""
    component_order: tuple[str, ...] | None = None
    train_time_s: float = 0.0


@dataclass
class AblationStudy:
    """Container for full ablation study results."""

    results: list[AblationResult] = field(default_factory=list)
    severity_sweep_results: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        rows = []
        for r in self.results:
            row = {
                "model": r.model_name,
                "description": r.description,
                "train_acc": r.train_accuracy,
                "val_acc": r.val_accuracy,
                "test_acc": r.test_accuracy,
                "test_f1": r.test_f1_macro,
                "train_time_s": r.train_time_s,
                "noise_order": " > ".join(r.component_order or ()),
                "noise_n_components": sum(r.noise_config.values()),
            }
            for comp in NOISE_COMPONENTS:
                row[f"noise_{comp}"] = r.noise_config.get(comp, False)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, output_path: Path) -> None:
        """Save ablation results to CSV."""
        df = self.to_dataframe()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")


# ── Noise config generation ───────────────────────────────────────────


def generate_single_component_configs() -> list[tuple[str, NoiseConfig]]:
    """Generate configs with only one noise component active at a time."""
    configs: list[tuple[str, NoiseConfig]] = []
    for comp in NOISE_COMPONENTS:
        cfg = NoiseConfig(
            enabled=True,
            gaussian_enabled=(comp == "gaussian"),
            contact_impedance_enabled=(comp == "contact_impedance"),
            electrode_bias_enabled=(comp == "electrode_bias"),
            quantisation_enabled=(comp == "quantisation"),
        )
        configs.append((f"single_{comp}", cfg))
    return configs


def generate_ablation_configs() -> list[tuple[str, NoiseConfig]]:
    """Generate all physically-meaningful noise configs for ablation.

    Returns:
        List of (description, NoiseConfig) tuples. Includes every non-empty
        component subset (1/2/3/4 components) and all physically admissible
        orderings under these rules:
        - Quantisation is always last when enabled.
        - Contact impedance is applied before electrode bias when both enabled.
    """
    configs: list[tuple[str, NoiseConfig]] = []

    for n_components in range(1, len(NOISE_COMPONENTS) + 1):
        for subset in itertools.combinations(NOISE_COMPONENTS, n_components):
            for order in _physically_valid_orders(subset):
                cfg = NoiseConfig(
                    enabled=True,
                    gaussian_enabled="gaussian" in subset,
                    contact_impedance_enabled="contact_impedance" in subset,
                    electrode_bias_enabled="electrode_bias" in subset,
                    quantisation_enabled="quantisation" in subset,
                    component_order=order,
                )
                subset_str = "+".join(subset)
                order_str = "->".join(order)
                configs.append((f"subset_{subset_str}__order_{order_str}", cfg))

    return configs


def _physically_valid_orders(components: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Return physically constrained component orderings for a subset.

    Rules:
        1) quantisation is fixed as the last stage when present.
        2) contact impedance precedes electrode bias when both are present.
    """
    active = list(components)
    includes_quant = "quantisation" in active

    non_quant = [c for c in active if c != "quantisation"]
    candidate_orders = list(itertools.permutations(non_quant))
    valid_non_quant: list[tuple[str, ...]] = []

    for order in candidate_orders:
        if (
            "contact_impedance" in order
            and "electrode_bias" in order
            and order.index("contact_impedance") > order.index("electrode_bias")
        ):
            continue
        valid_non_quant.append(order)

    if includes_quant:
        return [order + ("quantisation",) for order in valid_non_quant]
    return valid_non_quant


# ── Evaluation utilities ──────────────────────────────────────────────


def _evaluate_sklearn(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (train_acc, val_acc, test_acc, test_f1) for an sklearn model."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="macro")
    return train_acc, val_acc, test_acc, test_f1


def _evaluate_cnn(
    model: EITConv1D,
    X: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Return (accuracy, f1_macro) for a CNN model on a given split."""
    if device is None:
        device = get_device()
    model.to(device).eval()
    all_preds = []
    n = len(X)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            X_batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    all_preds_arr = np.concatenate(all_preds)
    return float(accuracy_score(y, all_preds_arr)), float(
        f1_score(y, all_preds_arr, average="macro")
    )


# ── Figure generation ─────────────────────────────────────────────────


def _plot_component_impact_bar(df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart: single-component isolation impact on test accuracy."""
    single = df[df["description"].str.startswith("single_")].copy()
    if single.empty:
        return

    full = df[df["description"] == "train_noisy_eval_noisy"]
    no_noise = df[df["description"] == "train_clean_eval_clean"]

    fig, ax = plt.subplots(figsize=(10, 5))

    single = single.sort_values("test_acc", ascending=False)
    labels = [
        COMPONENT_LABELS.get(d.replace("single_", ""), d) for d in single["description"]
    ]
    colors = sns.color_palette("colorblind", n_colors=len(labels) + 2)

    bars = ax.bar(
        labels,
        single["test_acc"].values,
        color=colors[: len(labels)],
        edgecolor="black",
        linewidth=0.5,
    )

    if not full.empty:
        ax.axhline(
            full.iloc[0]["test_acc"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"All noise ({full.iloc[0]['test_acc']:.3f})",
        )
    if not no_noise.empty:
        ax.axhline(
            no_noise.iloc[0]["test_acc"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"No noise ({no_noise.iloc[0]['test_acc']:.3f})",
        )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Single-Component Noise Isolation: Impact on Classification")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = figures_dir / "single_component_impact.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def _plot_component_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Heatmap of test accuracy by component combination."""
    subset_df = df[df["description"].str.startswith("subset_")].copy()
    if subset_df.empty:
        return

    comp_cols = [f"noise_{c}" for c in NOISE_COMPONENTS]
    subset_df["comp_key"] = subset_df[comp_cols].apply(
        lambda row: tuple(row.values), axis=1
    )
    best_per_subset = subset_df.loc[subset_df.groupby("comp_key")["test_acc"].idxmax()]

    n_subsets = len(best_per_subset)
    matrix = np.zeros((n_subsets, len(NOISE_COMPONENTS)))
    labels_y = []
    acc_values = []

    for idx, (_, row) in enumerate(
        best_per_subset.sort_values("test_acc", ascending=False).iterrows()
    ):
        for j, comp in enumerate(NOISE_COMPONENTS):
            matrix[idx, j] = 1.0 if row[f"noise_{comp}"] else 0.0
        active = [COMPONENT_LABELS[c] for c in NOISE_COMPONENTS if row[f"noise_{c}"]]
        labels_y.append(" + ".join(active))
        acc_values.append(row["test_acc"])

    fig, (ax_heat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(14, max(6, n_subsets * 0.4)),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap="Blues",
        cbar=False,
        xticklabels=[COMPONENT_LABELS[c] for c in NOISE_COMPONENTS],
        yticklabels=labels_y,
        linewidths=0.5,
    )
    ax_heat.set_title("Active Noise Components")

    ax_bar.barh(
        range(n_subsets),
        acc_values,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_bar.set_yticks(range(n_subsets))
    ax_bar.set_yticklabels([""] * n_subsets)
    ax_bar.set_xlabel("Test Accuracy")
    ax_bar.set_title("Best Ordering Accuracy")
    ax_bar.invert_yaxis()

    fig.suptitle("Noise Component Ablation: Subset Analysis", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = figures_dir / "component_subset_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def _plot_ordering_impact(df: pd.DataFrame, figures_dir: Path) -> None:
    """Show ordering impact for the full 4-component configuration."""
    full_subset = df[
        (df["noise_gaussian"] == True)  # noqa: E712
        & (df["noise_contact_impedance"] == True)  # noqa: E712
        & (df["noise_electrode_bias"] == True)  # noqa: E712
        & (df["noise_quantisation"] == True)  # noqa: E712
    ].copy()
    if full_subset.empty or len(full_subset) < 2:
        return

    full_subset = full_subset.sort_values("test_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(full_subset) * 0.35)))
    ax.barh(
        full_subset["noise_order"],
        full_subset["test_acc"],
        color="darkorange",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Impact of Noise Application Order (All 4 Components)")
    ax.invert_yaxis()
    fig.tight_layout()

    out_path = figures_dir / "ordering_impact.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def _plot_severity_sweep(sweep_results: dict, figures_dir: Path) -> None:
    """Plot per-component severity sweep curves."""
    if not sweep_results:
        return

    fig, (ax_acc, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("colorblind", n_colors=len(sweep_results))

    for idx, (comp_name, data) in enumerate(sweep_results.items()):
        label = COMPONENT_LABELS.get(comp_name, comp_name)
        severities = data["severity_levels"]
        ax_acc.plot(
            severities,
            data["accuracies"],
            marker="o",
            label=label,
            color=colors[idx],
            linewidth=2,
        )
        ax_f1.plot(
            severities,
            data["f1_scores"],
            marker="s",
            label=label,
            color=colors[idx],
            linewidth=2,
        )

    ax_acc.set_xlabel("Severity Multiplier")
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Per-Component Severity Sweep: Accuracy")
    ax_acc.legend()
    ax_acc.set_ylim(0, 1.0)
    ax_acc.grid(True, alpha=0.3)

    ax_f1.set_xlabel("Severity Multiplier")
    ax_f1.set_ylabel("Test F1 (macro)")
    ax_f1.set_title("Per-Component Severity Sweep: F1")
    ax_f1.legend()
    ax_f1.set_ylim(0, 1.0)
    ax_f1.grid(True, alpha=0.3)

    fig.suptitle("Noise Robustness by Component Type", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = figures_dir / "per_component_severity_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def _plot_mismatch_summary(df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart for the 4 core mismatch conditions."""
    core_names = [
        "train_clean_eval_clean",
        "train_clean_eval_noisy",
        "train_noisy_eval_noisy",
        "train_noisy_eval_clean",
    ]
    core_labels = ["Clean→Clean", "Clean→Noisy", "Noisy→Noisy", "Noisy→Clean"]
    core = df[df["description"].isin(core_names)].copy()
    if core.empty:
        return

    core["_order"] = core["description"].map({n: i for i, n in enumerate(core_names)})
    core = core.sort_values("_order")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        core_labels[: len(core)],
        core["test_acc"].values,
        color=sns.color_palette("colorblind", 4),
        edgecolor="black",
        linewidth=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Core Mismatch Experiment: Train/Eval Condition Impact")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out_path = figures_dir / "mismatch_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ── Report generation ─────────────────────────────────────────────────


def generate_ablation_report(
    study: AblationStudy,
    output_dir: Path,
    runtime_s: float,
    model_name: str,
) -> None:
    """Generate a comprehensive Markdown ablation report."""
    df = study.to_dataframe()
    lines: list[str] = []

    lines.append("# EIT Touch Classification — Ablation Study Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model**: {model_name}")
    lines.append(f"**Total runtime**: {runtime_s / 60:.1f} minutes")
    lines.append(f"**Total experiments**: {len(df)}")
    lines.append(
        "**Metric provenance**: Accuracy and F1 values below are held-out test "
        "metrics from the ablation runs; validation accuracy is used only during "
        "training and is not the headline result."
    )
    lines.append(
        "**Evidence tiering**: Core mismatch conditions are confirmatory and are "
        "aligned to the main grid protocol. Component isolation, ordering, and "
        "severity sweeps are exploratory mechanism analyses."
    )

    # ── 1. Core mismatch ──
    lines.append("\n---\n## 1. Core Mismatch Conditions\n")
    core = df[df["description"].str.startswith("train_")]
    if not core.empty:
        lines.append("| Condition | Test Acc | Test F1 | Train Time (s) |")
        lines.append("|-----------|----------|---------|----------------|")
        for _, row in core.iterrows():
            lines.append(
                f"| {row['description']} | {row['test_acc']:.4f} | "
                f"{row['test_f1']:.4f} | {row['train_time_s']:.1f} |"
            )

    # ── 2. Single-component isolation ──
    lines.append("\n---\n## 2. Single-Component Isolation\n")
    lines.append(
        "Each row shows the result of training with ONLY that noise type active.\n"
    )
    single = df[df["description"].str.startswith("single_")]
    if not single.empty:
        single_sorted = single.sort_values("test_acc", ascending=True)
        lines.append("| Noise Component | Test Acc | Test F1 | Impact vs Full Noise |")
        lines.append("|-----------------|----------|---------|---------------------|")
        full_row = df[df["description"] == "train_noisy_eval_noisy"]
        full_acc = full_row.iloc[0]["test_acc"] if not full_row.empty else None
        for _, row in single_sorted.iterrows():
            comp = row["description"].replace("single_", "")
            label = COMPONENT_LABELS.get(comp, comp)
            impact = ""
            if full_acc is not None:
                delta = row["test_acc"] - full_acc
                impact = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            lines.append(
                f"| {label} | {row['test_acc']:.4f} | {row['test_f1']:.4f} | {impact} |"
            )

        worst = single_sorted.iloc[0]
        best = single_sorted.iloc[-1]
        worst_label = COMPONENT_LABELS.get(
            worst["description"].replace("single_", ""), ""
        )
        best_label = COMPONENT_LABELS.get(
            best["description"].replace("single_", ""), ""
        )
        lines.append(
            f"\n**Most harmful individual component**: "
            f"{worst_label} (acc={worst['test_acc']:.4f})"
        )
        lines.append(
            f"**Least harmful individual component**: "
            f"{best_label} (acc={best['test_acc']:.4f})"
        )
        lines.append(
            f"**Spread**: {best['test_acc'] - worst['test_acc']:.4f} "
            f"accuracy range across components"
        )

    # ── 3. Per-component severity sweep ──
    degradation_rates: dict[str, float] = {}
    if study.severity_sweep_results:
        lines.append("\n---\n## 3. Per-Component Severity Sweep\n")
        lines.append(
            "Model trained on full noise at 1.0× severity, then evaluated "
            "with only one noise type at varying severity.\n"
        )
        lines.append("| Component | 0.0× | 0.5× | 1.0× | 1.5× | 2.0× | 2.5× | 3.0× |")
        lines.append("|-----------|------|------|------|------|------|------|------|")
        for comp, data in study.severity_sweep_results.items():
            label = COMPONENT_LABELS.get(comp, comp)
            acc_strs = " | ".join(f"{a:.3f}" for a in data["accuracies"])
            lines.append(f"| {label} | {acc_strs} |")

        for comp, data in study.severity_sweep_results.items():
            accs = data["accuracies"]
            if len(accs) >= 3:
                degradation_rates[comp] = accs[0] - accs[-1]

        if degradation_rates:
            worst_degrader = max(degradation_rates, key=degradation_rates.get)
            lines.append(
                f"\n**Fastest degradation**: {COMPONENT_LABELS[worst_degrader]} "
                f"(Δacc = -{degradation_rates[worst_degrader]:.4f} from 0x to 3x)"
            )
            best_degrader = min(degradation_rates, key=degradation_rates.get)
            lines.append(
                f"**Most robust to**: {COMPONENT_LABELS[best_degrader]} "
                f"(Δacc = -{degradation_rates[best_degrader]:.4f} from 0x to 3x)"
            )

    # ── 4. Best ordering analysis ──
    subset_df = df[df["description"].str.startswith("subset_")]
    if not subset_df.empty:
        lines.append("\n---\n## 4. Component Ordering Analysis\n")

        full_subset = subset_df[
            (subset_df["noise_gaussian"] == True)  # noqa: E712
            & (subset_df["noise_contact_impedance"] == True)  # noqa: E712
            & (subset_df["noise_electrode_bias"] == True)  # noqa: E712
            & (subset_df["noise_quantisation"] == True)  # noqa: E712
        ]
        if not full_subset.empty:
            lines.append("### Full 4-Component Orderings\n")
            lines.append("| Order | Test Acc | Test F1 |")
            lines.append("|-------|----------|---------|")
            for _, row in full_subset.sort_values(
                "test_acc", ascending=False
            ).iterrows():
                lines.append(
                    f"| {row['noise_order']} | "
                    f"{row['test_acc']:.4f} | {row['test_f1']:.4f} |"
                )

            best_order = full_subset.loc[full_subset["test_acc"].idxmax()]
            worst_order = full_subset.loc[full_subset["test_acc"].idxmin()]
            lines.append(
                f"\n**Best ordering**: {best_order['noise_order']} "
                f"(acc={best_order['test_acc']:.4f})"
            )
            lines.append(
                f"**Worst ordering**: {worst_order['noise_order']} "
                f"(acc={worst_order['test_acc']:.4f})"
            )
            lines.append(
                f"**Ordering effect size**: "
                f"{best_order['test_acc'] - worst_order['test_acc']:.4f}"
            )

        lines.append("\n### Best Configuration per Component Count\n")
        lines.append("| # Components | Best Config | Test Acc | Test F1 |")
        lines.append("|:---:|-----------|----------|---------|")
        for n_comp in range(1, 5):
            n_df = subset_df[subset_df["noise_n_components"] == n_comp]
            if not n_df.empty:
                best = n_df.loc[n_df["test_acc"].idxmax()]
                active = [
                    COMPONENT_LABELS[c] for c in NOISE_COMPONENTS if best[f"noise_{c}"]
                ]
                lines.append(
                    f"| {n_comp} | {' + '.join(active)} | "
                    f"{best['test_acc']:.4f} | {best['test_f1']:.4f} |"
                )

    # ── 5. Summary and conclusions ──
    lines.append("\n---\n## 5. Key Findings\n")
    if not single.empty:
        single_sorted = single.sort_values("test_acc", ascending=True)
        lines.append("1. **Individual component ranking (most → least harmful)**:")
        for _, row in single_sorted.iterrows():
            comp = row["description"].replace("single_", "")
            lines.append(
                f"   - {COMPONENT_LABELS.get(comp, comp)}: {row['test_acc']:.4f}"
            )

    if degradation_rates:
        worst_degrader = max(degradation_rates, key=degradation_rates.get)
        lines.append(
            f"\n2. **Severity sensitivity**: {COMPONENT_LABELS[worst_degrader]} "
            "causes the fastest performance degradation with increasing severity."
        )

    if not subset_df.empty:
        full_subset = subset_df[
            (subset_df["noise_gaussian"] == True)  # noqa: E712
            & (subset_df["noise_contact_impedance"] == True)  # noqa: E712
            & (subset_df["noise_electrode_bias"] == True)  # noqa: E712
            & (subset_df["noise_quantisation"] == True)  # noqa: E712
        ]
        if not full_subset.empty:
            best_order = full_subset.loc[full_subset["test_acc"].idxmax()]
            lines.append(
                f"\n3. **Optimal noise application order**: {best_order['noise_order']}"
            )
            lines.append(
                "   This ordering should be used in the MATLAB generation "
                "pipeline for maximum training effectiveness."
            )

    lines.append("\n---\n*Report generated by `python/ablation.py`*\n")

    report_path = output_dir / "ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Ablation report saved to {report_path}")


# ── Main ablation runner ──────────────────────────────────────────────


def run_ablation(
    data_path: Path,
    model_name: str = "cnn1d",
    seed: int = 42,
    n_seeds: int = 5,
    run_all_configs: bool = True,
    run_severity_sweep: bool = True,
    epochs: int = 200,
    early_stopping_patience: int = 40,
    figures_dir: Path = Path("results/figures/ablation"),
    output_dir: Path = Path("results/reports"),
) -> AblationStudy:
    """Run the ablation study using Python-side noise injection.

    Experiments:
        - Core mismatch (4 experiments): clean→clean, clean→noisy,
            noisy→noisy, noisy→clean.
        - Single-component isolation (4 experiments): one noise type active.
        - Per-component severity sweep: sweep severity per type.
        - Exhaustive component/order ablation (default, disable with
            run_all_configs=False): all non-empty component subsets (1/2/3/4
            components) with all physically constrained orderings.

    Each experiment is repeated over `n_seeds` independent random seeds with
    different stratified splits for statistical robustness (mean ± std).

    Args:
        data_path: Path to the dataset .mat file (must contain clean vectors).
        model_name: Model to train ('cnn1d', 'svm', 'random_forest', 'mlp').
        seed: Base random seed (actual seeds used: seed, seed+1, ..., seed+n_seeds-1).
        n_seeds: Number of independent seeds/splits to run (default: 5).
        run_all_configs: Run all subset/order ablation experiments (default True).
        run_severity_sweep: If True, run per-component severity sweeps.
        epochs: Max CNN training epochs.
        early_stopping_patience: CNN early stopping patience.
        figures_dir: Output directory for figures.
        output_dir: Output directory for report.

    Returns:
        Completed AblationStudy (aggregated across seeds).
    """
    device = get_device()
    seeds = [seed + i for i in range(n_seeds)]
    logger.info(f"Starting ablation study with model: {model_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Running {n_seeds} seeds: {seeds}")

    # Load CLEAN data as the base (noise is applied in Python)
    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy_matlab, _ = load_mat_dataset(data_path, use_noisy=True)

    # Full noise config (all 4 components at default severity)
    full_noise = NoiseConfig()

    # Collect per-seed results: dict[description, list[AblationResult]]
    all_seed_results: dict[str, list[AblationResult]] = {}
    # Severity sweep: dict[comp, list[dict]] per seed
    all_seed_severity: dict[str, list[dict]] = {c: [] for c in NOISE_COMPONENTS}

    for seed_idx, current_seed in enumerate(seeds):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  SEED {seed_idx + 1}/{n_seeds} (seed={current_seed})")
        logger.info(f"{'=' * 60}")

        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(current_seed)

        # Use different splits for each seed
        dataset_clean = prepare_splits(X_clean, y, random_state=current_seed)
        dataset_noisy = prepare_splits(X_noisy_matlab, y, random_state=current_seed)

        def _run_experiment(
            train_data: str,
            eval_data: str,
            noise_cfg: NoiseConfig | None,
            description: str,
            dataset_clean=dataset_clean,
            dataset_noisy=dataset_noisy,
            current_seed=current_seed,
        ) -> AblationResult:
            """Train and evaluate a single ablation experiment."""
            if train_data == "clean":
                X_tr = dataset_clean.X_train
                y_tr = dataset_clean.y_train
                X_v = dataset_clean.X_val
                y_v = dataset_clean.y_val
                train_noise_cfg = None
            elif train_data == "noisy_matlab":
                X_tr = dataset_noisy.X_train
                y_tr = dataset_noisy.y_train
                X_v = dataset_noisy.X_val
                y_v = dataset_noisy.y_val
                train_noise_cfg = None
            elif train_data == "noisy_python":
                X_tr = dataset_clean.X_train
                y_tr = dataset_clean.y_train
                X_v = dataset_clean.X_val
                y_v = dataset_clean.y_val
                train_noise_cfg = noise_cfg
            else:
                raise ValueError(f"Unknown train_data: {train_data}")

            if eval_data == "clean":
                X_te = dataset_clean.X_test
                y_te = dataset_clean.y_test
            elif eval_data == "noisy_matlab":
                X_te = dataset_noisy.X_test
                y_te = dataset_noisy.y_test
            elif eval_data == "noisy":
                eval_rng = np.random.default_rng(current_seed + 999)
                eval_noise_cfg = noise_cfg if noise_cfg is not None else full_noise
                X_te = apply_noise_in_scaled_space(
                    dataset_clean.X_test,
                    dataset_clean.scaler,
                    eval_noise_cfg,
                    rng=eval_rng,
                )
                y_te = dataset_clean.y_test
            else:
                raise ValueError(f"Unknown eval_data: {eval_data}")

            start_time = time.time()
            if model_name == "cnn1d":
                model, _ = train_cnn(
                    X_tr,
                    y_tr,
                    X_v,
                    y_v,
                    epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                    device=device,
                    noise_config=train_noise_cfg,
                    weight_decay=(
                        NOISY_CNN_PARAMS["weight_decay"]
                        if description in ("train_noisy_eval_noisy", "train_noisy_eval_clean")
                        else 1e-4
                    ),
                    dropout=(
                        NOISY_CNN_PARAMS["dropout"]
                        if description in ("train_noisy_eval_noisy", "train_noisy_eval_clean")
                        else 0.3
                    ),
                    label_smoothing=(
                        NOISY_CNN_PARAMS["label_smoothing"]
                        if description in ("train_noisy_eval_noisy", "train_noisy_eval_clean")
                        else 0.0
                    ),
                    input_scaler=(
                        dataset_clean.scaler if train_noise_cfg is not None else None
                    ),
                )
                train_acc, _ = _evaluate_cnn(model, X_tr, y_tr, device=device)
                val_acc, _ = _evaluate_cnn(model, X_v, y_v, device=device)
                test_acc, test_f1 = _evaluate_cnn(model, X_te, y_te, device=device)
            else:
                if train_noise_cfg is not None and train_noise_cfg.enabled:
                    aug_rng = np.random.default_rng(current_seed)
                    X_tr = apply_noise_in_scaled_space(
                        X_tr,
                        dataset_clean.scaler,
                        train_noise_cfg,
                        rng=aug_rng,
                    )

                model = get_baseline(model_name, random_state=current_seed)
                model = train_baseline(model, X_tr, y_tr)
                train_acc, val_acc, test_acc, test_f1 = _evaluate_sklearn(
                    model, X_tr, y_tr, X_v, y_v, X_te, y_te
                )
            train_time = time.time() - start_time

            noise_flags = (
                noise_cfg.component_flags()
                if noise_cfg
                else {c: False for c in NOISE_COMPONENTS}
            )
            result = AblationResult(
                noise_config=noise_flags,
                model_name=model_name,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                test_accuracy=test_acc,
                test_f1_macro=test_f1,
                description=description,
                component_order=(
                    noise_cfg.resolved_component_order() if noise_cfg else ()
                ),
                train_time_s=train_time,
            )
            logger.info(
                f"  {description}: acc={test_acc:.4f}, f1={test_f1:.4f} "
                f"({train_time:.1f}s)"
            )
            return result

        # ── 1. Core mismatch experiments ──────────────────────────────
        logger.info("\n── Core Mismatch Experiments (4) ──")
        for train_data, eval_data, noise_cfg, desc in [
            ("clean", "clean", None, "train_clean_eval_clean"),
            ("clean", "noisy", None, "train_clean_eval_noisy"),
            ("noisy_matlab", "noisy_matlab", None, "train_noisy_eval_noisy"),
            ("noisy_matlab", "clean", None, "train_noisy_eval_clean"),
        ]:
            r = _run_experiment(train_data, eval_data, noise_cfg, desc)
            all_seed_results.setdefault(desc, []).append(r)

        # ── 2. Single-component isolation ─────────────────────────────
        logger.info("\n── Single-Component Isolation (4) ──")
        single_configs = generate_single_component_configs()
        for description, cfg_variant in single_configs:
            r = _run_experiment("noisy_python", "noisy", cfg_variant, description)
            all_seed_results.setdefault(description, []).append(r)

        # ── 3. Exhaustive component/order ablation ────────────────────
        if run_all_configs:
            configs = generate_ablation_configs()
            logger.info(
                f"\n── Exhaustive Component/Order Ablation "
                f"({len(configs)} experiments) ──"
            )
            for idx, (description, cfg_variant) in enumerate(configs, start=1):
                if idx % 10 == 0:
                    logger.info(f"  Progress: {idx}/{len(configs)}")
                r = _run_experiment("noisy_python", "noisy", cfg_variant, description)
                all_seed_results.setdefault(description, []).append(r)

        # ── 4. Per-component severity sweep ───────────────────────────
        if run_severity_sweep and model_name == "cnn1d":
            logger.info("\n── Per-Component Severity Sweep ──")
            logger.info(
                "  Training CNN on full noise, then evaluating per-component..."
            )

            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(current_seed)

            sweep_model, _ = train_cnn(
                dataset_clean.X_train,
                dataset_clean.y_train,
                dataset_clean.X_val,
                dataset_clean.y_val,
                epochs=epochs,
                early_stopping_patience=early_stopping_patience,
                device=device,
                noise_config=full_noise,
                input_scaler=dataset_clean.scaler,
            )

            for comp in NOISE_COMPONENTS:
                comp_accs = []
                comp_f1s = []
                for sev in SEVERITY_LEVELS:
                    if sev == 0.0:
                        X_te = dataset_clean.X_test
                    else:
                        sweep_cfg = NoiseConfig(
                            enabled=True,
                            gaussian_enabled=(comp == "gaussian"),
                            contact_impedance_enabled=(comp == "contact_impedance"),
                            electrode_bias_enabled=(comp == "electrode_bias"),
                            quantisation_enabled=(comp == "quantisation"),
                            severity=sev,
                        )
                        rng = np.random.default_rng(current_seed + 777)
                        X_te = apply_noise_in_scaled_space(
                            dataset_clean.X_test,
                            dataset_clean.scaler,
                            sweep_cfg,
                            rng=rng,
                        )

                    y_pred = predict_cnn(sweep_model, X_te, device=device)
                    acc = float(accuracy_score(dataset_clean.y_test, y_pred))
                    f1 = float(f1_score(dataset_clean.y_test, y_pred, average="macro"))
                    comp_accs.append(acc)
                    comp_f1s.append(f1)

                all_seed_severity[comp].append(
                    {"accuracies": comp_accs, "f1_scores": comp_f1s}
                )
                logger.info(
                    f"  {COMPONENT_LABELS[comp]}: "
                    f"{comp_accs[0]:.3f} → {comp_accs[-1]:.3f}"
                )

    # ── Aggregate results across seeds ────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  AGGREGATING {n_seeds} SEEDS")
    logger.info(f"{'=' * 60}")

    study = AblationStudy()

    for desc, results_list in all_seed_results.items():
        # Use the first result as template for noise config/order metadata,
        # but store aggregated (mean) metrics
        template = results_list[0]
        accs = [r.test_accuracy for r in results_list]
        f1s = [r.test_f1_macro for r in results_list]
        train_accs = [r.train_accuracy for r in results_list]
        val_accs = [r.val_accuracy for r in results_list]
        times = [r.train_time_s for r in results_list]

        study.results.append(
            AblationResult(
                noise_config=template.noise_config,
                model_name=template.model_name,
                train_accuracy=float(np.mean(train_accs)),
                val_accuracy=float(np.mean(val_accs)),
                test_accuracy=float(np.mean(accs)),
                test_f1_macro=float(np.mean(f1s)),
                description=template.description,
                component_order=template.component_order,
                train_time_s=float(np.mean(times)),
            )
        )
        logger.info(
            f"  {desc}: acc={np.mean(accs):.4f} ± {np.std(accs):.4f}, "
            f"f1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        )

    # Aggregate severity sweep: mean across seeds
    if run_severity_sweep and model_name == "cnn1d":
        for comp in NOISE_COMPONENTS:
            if all_seed_severity[comp]:
                all_accs = np.array([s["accuracies"] for s in all_seed_severity[comp]])
                all_f1s = np.array([s["f1_scores"] for s in all_seed_severity[comp]])
                study.severity_sweep_results[comp] = {
                    "severity_levels": SEVERITY_LEVELS,
                    "accuracies": all_accs.mean(axis=0).tolist(),
                    "accuracies_std": all_accs.std(axis=0).tolist(),
                    "f1_scores": all_f1s.mean(axis=0).tolist(),
                    "f1_scores_std": all_f1s.std(axis=0).tolist(),
                }

    # ── Generate figures ──────────────────────────────────────────────
    logger.info("\n── Generating Ablation Figures ──")
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = study.to_dataframe()

    _plot_mismatch_summary(df, figures_dir)
    _plot_component_impact_bar(df, figures_dir)
    _plot_severity_sweep(study.severity_sweep_results, figures_dir)

    if run_all_configs:
        _plot_component_heatmap(df, figures_dir)
        _plot_ordering_impact(df, figures_dir)

    # ── Save per-seed raw results for transparency ────────────────────
    raw_rows = []
    for desc, results_list in all_seed_results.items():
        for seed_idx, r in enumerate(results_list):
            raw_rows.append(
                {
                    "seed": seeds[seed_idx],
                    "description": r.description,
                    "model": r.model_name,
                    "test_acc": r.test_accuracy,
                    "test_f1": r.test_f1_macro,
                    "val_acc": r.val_accuracy,
                    "train_time_s": r.train_time_s,
                }
            )
    raw_df = pd.DataFrame(raw_rows)
    raw_path = output_dir / "ablation_per_seed_results.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Per-seed results saved to {raw_path}")

    logger.info("Ablation study complete.")
    return study


# ── CLI entry point ───────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone ablation run."""
    parser = argparse.ArgumentParser(description="Run ablation study.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to dataset .mat file (default: from config.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn1d",
        choices=["svm", "random_forest", "mlp", "cnn1d"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/ablation_results.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports"),
    )
    parser.add_argument(
        "--skip-all-configs",
        action="store_true",
        help="Skip exhaustive subset/order ablation (only run core + single-component).",
    )
    parser.add_argument(
        "--no-severity-sweep",
        action="store_true",
        help="Skip the per-component severity sweep.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of independent seeds for statistical robustness (default: 5).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    cfg = load_config()
    data_path = args.data_path or Path(cfg["data"]["path"])

    start_time = time.time()
    study = run_ablation(
        data_path,
        model_name=args.model,
        seed=args.seed,
        n_seeds=args.n_seeds,
        run_all_configs=not args.skip_all_configs,
        run_severity_sweep=not args.no_severity_sweep,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        figures_dir=args.figures_dir,
        output_dir=args.output_dir,
    )
    total_time = time.time() - start_time

    study.save(args.output)

    # Save severity sweep as JSON
    if study.severity_sweep_results:
        sev_path = args.output_dir / "ablation_severity_sweep.json"
        sev_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sev_path, "w", encoding="utf-8") as f:
            json.dump(study.severity_sweep_results, f, indent=2)
        logger.info(f"Severity sweep saved to {sev_path}")

    # Generate report
    generate_ablation_report(study, args.output_dir, total_time, args.model)

    # Print summary
    df = study.to_dataframe()
    print("\n=== Ablation Study Summary ===")
    print(df[["description", "test_acc", "test_f1"]].to_string(index=False))
