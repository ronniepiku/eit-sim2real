"""Mesh-refinement training and cross-evaluation study.

This experiment trains one noisy CNN on each of the c, d and f mesh
datasets using the same noisy data model, same hyperparameters, and the
same train/validation/test protocol. Each trained model is then evaluated
on all three test splits to produce a 3x3 cross-mesh matrix. The study
also records training time, inference time, parameter count and peak
memory, and saves the training curves and summary figures needed for the
dissertation.

This replaces the earlier mesh-convergence-only table while keeping
cross-mesh evaluation as part of the same study.
"""

from __future__ import annotations

import csv
import json
import logging
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score

from eit_sim2real.constants import CLASS_NAMES, NUM_CLASSES
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.evaluate import evaluate_model
from eit_sim2real.experiments.protocols import NOISY_CNN_PARAMS
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.train import train_cnn
from eit_sim2real.utils import count_parameters, get_device, set_seeds
from eit_sim2real.visualisation import (
    plot_confusion_matrix_and_save,
    plot_per_class_metrics_and_save,
    plot_training_curves,
)

logger = logging.getLogger(__name__)

MESH_REFINEMENTS = ("c", "d", "f")
NOISY_TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "lr": 1e-3,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 40,
    "weight_decay": NOISY_CNN_PARAMS["weight_decay"],
    "dropout": NOISY_CNN_PARAMS["dropout"],
    "label_smoothing": NOISY_CNN_PARAMS["label_smoothing"],
}


@dataclass(slots=True)
class MeshDataset:
    refinement: str
    path: Path
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_test_raw: np.ndarray
    scaler: Any
    n_samples: int
    n_features: int


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_id
        out[cls_name] = float("nan") if mask.sum() == 0 else float(
            accuracy_score(y_true[mask], y_pred[mask])
        )
    return out


def _validate_dataset_alignment(datasets: dict[str, MeshDataset]) -> None:
    reference = None
    for refinement, dataset in datasets.items():
        counts = tuple(np.bincount(dataset.y_test, minlength=NUM_CLASSES).tolist())
        if reference is None:
            reference = counts
            continue
        if counts != reference:
            raise ValueError(
                f"Class distribution mismatch for refinement {refinement}: "
                f"expected {reference}, got {counts}."
            )


def _profile_call(fn: Any) -> tuple[Any, float, float, float]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start = perf_counter()
    result = fn()
    if cuda_available:
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_py_mb = peak_py / 1024**2
    peak_cuda_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if cuda_available else 0.0
    )
    return result, elapsed, peak_py_mb, peak_cuda_mb


def _load_mesh_dataset(refinement: str, dataset_path: Path, seed: int) -> MeshDataset:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Mesh dataset not found for refinement '{refinement}': {dataset_path}"
        )

    X, y = load_mat_dataset(dataset_path, use_noisy=True)
    splits = prepare_splits(
        X,
        y,
        random_state=seed,
        normalize=True,
        scaler_type="robust",
    )
    X_test_raw = splits.scaler.inverse_transform(splits.X_test)
    return MeshDataset(
        refinement=refinement,
        path=dataset_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        X_test_raw=X_test_raw,
        scaler=splits.scaler,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )


def _load_cnn(n_features: int) -> EITConv1D:
    return EITConv1D(n_features=n_features, n_classes=NUM_CLASSES)


def _prepare_eval_features(source_scaler: Any, target_raw_test: np.ndarray) -> np.ndarray:
    return source_scaler.transform(target_raw_test).astype(np.float32)


def _train_one_model(
    dataset: MeshDataset,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    output_models_dir: Path,
) -> dict[str, Any]:
    set_seeds(seed)

    def _run_training() -> tuple[EITConv1D, dict[str, list[float]]]:
        return train_cnn(
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=NOISY_TRAINING_CONFIG["lr"],
            weight_decay=NOISY_TRAINING_CONFIG["weight_decay"],
            scheduler_patience=NOISY_TRAINING_CONFIG["scheduler_patience"],
            scheduler_factor=NOISY_TRAINING_CONFIG["scheduler_factor"],
            early_stopping_patience=NOISY_TRAINING_CONFIG["early_stopping_patience"],
            device=device,
            dropout=NOISY_TRAINING_CONFIG["dropout"],
            label_smoothing=NOISY_TRAINING_CONFIG["label_smoothing"],
        )

    (model, history), train_time_s, train_peak_py_mb, train_peak_cuda_mb = _profile_call(
        _run_training
    )

    model_path = output_models_dir / f"cnn1d_noisy_mesh_{dataset.refinement}.pt"
    output_models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    return {
        "model": model,
        "history": history,
        "checkpoint": str(model_path),
        "train_time_s": train_time_s,
        "train_peak_python_memory_mb": train_peak_py_mb,
        "train_peak_cuda_memory_mb": train_peak_cuda_mb,
        "train_peak_memory_mb": max(train_peak_py_mb, train_peak_cuda_mb),
        "n_parameters": int(count_parameters(model)),
    }


def _evaluate_one_pair(
    model: EITConv1D,
    source_dataset: MeshDataset,
    target_dataset: MeshDataset,
    device: str,
) -> tuple[dict[str, Any], np.ndarray]:
    X_eval = (
        source_dataset.X_test
        if source_dataset.refinement == target_dataset.refinement
        else _prepare_eval_features(source_dataset.scaler, target_dataset.X_test_raw)
    )

    def _run_eval() -> dict[str, Any]:
        return evaluate_model(model, X_eval, target_dataset.y_test, device=device)

    results, inference_time_s, peak_py_mb, peak_cuda_mb = _profile_call(_run_eval)
    y_pred = results["y_pred"]
    n_eval = int(target_dataset.y_test.shape[0])

    return (
        {
            "accuracy": float(results["accuracy"]),
            "f1_macro": float(results["f1_macro"]),
            "inference_time_s": inference_time_s,
            "inference_time_per_sample_ms": inference_time_s / n_eval * 1000.0,
            "peak_python_memory_mb": peak_py_mb,
            "peak_cuda_memory_mb": peak_cuda_mb,
            "peak_memory_mb": max(peak_py_mb, peak_cuda_mb),
            "n_samples": n_eval,
            "per_class_accuracy": _per_class_accuracy(target_dataset.y_test, y_pred),
        },
        y_pred,
    )


def _plot_matrix(
    matrix: dict[str, dict[str, float]],
    title: str,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    data = np.array([[matrix[r][c] for c in MESH_REFINEMENTS] for r in MESH_REFINEMENTS])
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"Test {r}" for r in MESH_REFINEMENTS],
        yticklabels=[f"Train {r}" for r in MESH_REFINEMENTS],
        cbar_kws={"label": title},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Test mesh")
    ax.set_ylabel("Training mesh")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_resource_summary(model_results: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    refinements = list(MESH_REFINEMENTS)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    train_times = [model_results[r]["train_time_s"] for r in refinements]
    inf_times = [
        model_results[r]["evaluations"][r]["inference_time_per_sample_ms"]
        for r in refinements
    ]
    params = [model_results[r]["n_parameters"] for r in refinements]
    memory = [model_results[r]["train_peak_memory_mb"] for r in refinements]

    panels = [
        (axes[0, 0], train_times, "Training time (s)"),
        (axes[0, 1], inf_times, "Inference time (ms/sample)"),
        (axes[1, 0], memory, "Peak memory (MB)"),
        (axes[1, 1], params, "Parameters"),
    ]

    for ax, values, title in panels:
        ax.bar(refinements, values, color=colors)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("Mesh refinement")

    axes[0, 0].set_ylabel("Seconds")
    axes[0, 1].set_ylabel("Milliseconds")
    axes[1, 0].set_ylabel("MB")
    axes[1, 1].set_ylabel("Count")

    fig.suptitle("Mesh-Refinement Resource Summary", y=1.02, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_reference_noisy_accuracy(report_dir: Path) -> float | None:
    csv_path = report_dir / "all_results.csv"
    if not csv_path.exists():
        return None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("dataset") == "raw"
                and row.get("model") == "cnn1d"
                and row.get("condition") == "noisy_train_noisy_eval"
            ):
                try:
                    return float(row["accuracy_mean"])
                except (TypeError, ValueError, KeyError):
                    return None
    return None


def run_mesh_refinement_evaluation(
    coarse_dataset_path: Path,
    baseline_dataset_path: Path,
    fine_mesh_dataset_path: Path,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    set_seeds(seed)
    device = get_device()

    if output_dir is None:
        output_dir = Path("results/additional_experiments/mesh_refinement")
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    models_dir = Path("results/models/mesh_refinement")

    logger.info("=" * 70)
    logger.info("MESH-REFINEMENT TRAINING + CROSS-EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Coarse dataset:   {coarse_dataset_path}")
    logger.info(f"Baseline dataset: {baseline_dataset_path}")
    logger.info(f"Fine dataset:     {fine_mesh_dataset_path}")

    datasets = {
        "c": _load_mesh_dataset("c", coarse_dataset_path, seed),
        "d": _load_mesh_dataset("d", baseline_dataset_path, seed),
        "f": _load_mesh_dataset("f", fine_mesh_dataset_path, seed),
    }
    _validate_dataset_alignment(datasets)

    n_features = {dataset.n_features for dataset in datasets.values()}
    if len(n_features) != 1:
        raise ValueError(f"Feature-count mismatch across mesh datasets: {sorted(n_features)}")
    n_features_value = n_features.pop()

    summary: dict[str, Any] = {
        "experiment": "mesh_refinement_training_cross_evaluation",
        "description": (
            "Trains one noisy CNN on each of the c, d and f mesh datasets "
            "using the same protocol, then evaluates all trained models "
            "on all three test splits. This preserves cross-mesh "
            "evaluation while replacing the earlier mesh-convergence table."
        ),
        "seed": seed,
        "n_features": int(n_features_value),
        "protocol": {
            "noise_model": "pre-generated noisy datasets with the project-wide 4-component noise model",
            "training_regime": "noisy_train_noisy_eval",
            "train_val_test_split": "70/15/15 stratified",
            "hyperparameters": NOISY_TRAINING_CONFIG,
            "same_protocol_across_meshes": True,
        },
        "datasets": {
            ref: {
                "path": str(dataset.path),
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
            }
            for ref, dataset in datasets.items()
        },
        "models": {},
    }

    reference_acc = _load_reference_noisy_accuracy(Path("results/reports"))
    if reference_acc is not None:
        summary["protocol_consistency"] = {
            "reference_report": "results/reports/all_results.csv",
            "reference_metric": "raw/cnn1d/noisy_train_noisy_eval accuracy_mean",
            "reference_accuracy": reference_acc,
        }

    model_results: dict[str, dict[str, Any]] = {}
    accuracy_matrix: dict[str, dict[str, float]] = {r: {} for r in MESH_REFINEMENTS}
    f1_matrix: dict[str, dict[str, float]] = {r: {} for r in MESH_REFINEMENTS}

    for source_ref in MESH_REFINEMENTS:
        logger.info("\nTraining model on mesh %s...", source_ref)
        source_dataset = datasets[source_ref]
        trained = _train_one_model(
            source_dataset,
            seed=seed,
            device=device,
            epochs=NOISY_TRAINING_CONFIG["epochs"],
            batch_size=NOISY_TRAINING_CONFIG["batch_size"],
            output_models_dir=models_dir,
        )

        model = trained["model"]
        history = trained["history"]
        source_model_dir = figures_dir / source_ref
        plot_training_curves(history, source_model_dir / "training", f"mesh_{source_ref}")

        model_entry: dict[str, Any] = {
            "checkpoint": trained["checkpoint"],
            "n_parameters": trained["n_parameters"],
            "train_time_s": trained["train_time_s"],
            "train_peak_python_memory_mb": trained["train_peak_python_memory_mb"],
            "train_peak_cuda_memory_mb": trained["train_peak_cuda_memory_mb"],
            "train_peak_memory_mb": trained["train_peak_memory_mb"],
            "history": history,
            "evaluations": {},
        }

        for target_ref in MESH_REFINEMENTS:
            logger.info("  Evaluating trained %s model on %s test set...", source_ref, target_ref)
            target_dataset = datasets[target_ref]
            eval_summary, y_pred = _evaluate_one_pair(
                model,
                source_dataset,
                target_dataset,
                device=device,
            )
            model_entry["evaluations"][target_ref] = eval_summary
            accuracy_matrix[source_ref][target_ref] = eval_summary["accuracy"]
            f1_matrix[source_ref][target_ref] = eval_summary["f1_macro"]

            if source_ref == target_ref:
                pair_output = figures_dir / source_ref / "in_mesh"
                plot_confusion_matrix_and_save(
                    target_dataset.y_test,
                    y_pred,
                    pair_output,
                    model_name=f"cnn1d_mesh_{source_ref}",
                    noise_tag=f"mesh_{source_ref}",
                    split_name="test",
                )
                plot_per_class_metrics_and_save(
                    target_dataset.y_test,
                    y_pred,
                    pair_output,
                    model_name=f"cnn1d_mesh_{source_ref}",
                    noise_tag=f"mesh_{source_ref}",
                    split_name="test",
                )

        model_results[source_ref] = model_entry
        summary["models"][source_ref] = model_entry

    _plot_matrix(
        accuracy_matrix,
        "Cross-mesh accuracy",
        figures_dir / "cross_mesh_accuracy_heatmap.png",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_matrix(
        f1_matrix,
        "Cross-mesh macro-F1",
        figures_dir / "cross_mesh_f1_heatmap.png",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_resource_summary(model_results, figures_dir / "resource_summary.png")

    summary["cross_mesh_matrix"] = {
        "accuracy": accuracy_matrix,
        "f1_macro": f1_matrix,
    }

    summary["interpretation"] = (
        "The study compares identical noisy-CNN training protocols across "
        "c/d/f meshes. The cross-mesh matrix isolates discretisation "
        "effects, while the resource summary reports the practical cost of "
        "each refinement level."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(tables_dir / "mesh_study_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "train_mesh",
            "test_mesh",
            "accuracy",
            "f1_macro",
            "inference_time_s",
            "inference_time_per_sample_ms",
            "train_time_s",
            "parameters",
            "train_peak_memory_mb",
        ])
        for source_ref in MESH_REFINEMENTS:
            for target_ref in MESH_REFINEMENTS:
                eval_summary = summary["models"][source_ref]["evaluations"][target_ref]
                writer.writerow([
                    source_ref,
                    target_ref,
                    eval_summary["accuracy"],
                    eval_summary["f1_macro"],
                    eval_summary["inference_time_s"],
                    eval_summary["inference_time_per_sample_ms"],
                    summary["models"][source_ref]["train_time_s"],
                    summary["models"][source_ref]["n_parameters"],
                    summary["models"][source_ref]["train_peak_memory_mb"],
                ])

    logger.info("\nResults saved to: %s", output_dir / "results.json")
    logger.info("Figures saved to: %s", figures_dir)
    logger.info("Summary table saved to: %s", tables_dir / "mesh_study_summary.csv")

    return summary


def main(
    coarse_dataset: str = "data/eit_dataset_mesh_c.mat",
    baseline_dataset: str = "data/eit_dataset.mat",
    fine_mesh_dataset: str = "data/eit_dataset_mesh_f.mat",
    output_dir: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    return run_mesh_refinement_evaluation(
        coarse_dataset_path=Path(coarse_dataset),
        baseline_dataset_path=Path(baseline_dataset),
        fine_mesh_dataset_path=Path(fine_mesh_dataset),
        output_dir=Path(output_dir) if output_dir else None,
        seed=seed,
    )
