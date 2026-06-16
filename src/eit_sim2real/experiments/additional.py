"""Additional dissertation experiments — fixed-bias augmentation & different-draw test.

Both experiments are run under the same 5-seed stratified protocol as the
main grid (see :mod:`eit_sim2real.experiments.grid`) so that the reported
numbers share the same statistical significance. Mean ± std across seeds
is recorded and a single JSON results file is emitted per experiment.

Experiment 1 — Fixed-Bias Augmentation
--------------------------------------
Samples noise ONCE per training instance (using ``base_seed + sample_index``)
and holds it fixed across all epochs. This mimics the deployment scenario
where each physical device has persistent per-electrode characteristics.

Hypothesis: combining the clean-domain awareness of online augmentation
(many different noise patterns) with the noise-persistence that enables
learning noise-accommodation (fixed per sample across epochs).

Important — feature spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~
The MATLAB noise model parameters (``max_bias=0.02 V``, ``noise_floor=1e-4``,
``adc_bits=16``) are calibrated for *raw* voltage difference vectors, not
for RobustScaler-transformed features. To remain consistent with both the
MATLAB-generated test set and the §3 methodology, this module always:

1. Splits raw clean/noisy data into train/val/test indices,
2. Applies fixed-bias noise to *raw* training voltages,
3. Then transforms with a scaler fit on raw clean training data.

Earlier revisions of this experiment applied noise in scaled feature space,
which silently rescaled the noise distribution by ``1/IQR(raw)``. That
behaviour is fixed here.

Experiment 2 — Different Noise Draw
-----------------------------------
Re-evaluates the existing noisy-trained CNN checkpoint on test sets generated
from independent random draws of the parametric noise model. If accuracy
holds, this provides evidence against pure noise memorisation. Multi-seed
aggregation reports the alternative-draw accuracy distribution alongside
the original noise-test sanity check.

Usage
-----
::

    eit experiments additional                   # 5-seed default
    eit experiments additional --seeds 3         # quick sanity check

Or as part of the master pipeline::

    eit experiments run-all                      # runs all 5-seed experiments

"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from eit_sim2real.configs import load_config
from eit_sim2real.constants import NOISY_CNN_PARAMS
from eit_sim2real.data import load_mat_dataset
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.evaluate import evaluate_model
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.utils import get_device

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-bias augmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────


class FixedBiasDataset(Dataset):
    """PyTorch Dataset that applies noise ONCE per sample and caches it.

    Noise is sampled in **raw voltage space** (not scaled feature space) so
    that ``max_bias``, ``noise_floor`` and ``adc_bits`` retain their physical
    meaning, then the cached samples are scaled with the supplied
    ``RobustScaler``.

    Parameters
    ----------
    X_clean_raw:
        Clean training features in **raw voltage space**, before scaling.
    y:
        Training labels.
    noise_config:
        Noise configuration to apply once per sample.
    scaler:
        Fitted scaler used to project the cached raw-noisy voltages into the
        model's feature space (the same scaler used at evaluation time).
    base_seed:
        Per-sample noise seed; sample ``i`` uses ``base_seed + i``.
    """

    def __init__(
        self,
        X_clean_raw: np.ndarray,
        y: np.ndarray,
        noise_config: NoiseConfig,
        scaler: RobustScaler,
        base_seed: int = 123,
    ):
        self.y = torch.from_numpy(y).long()
        self.noise_config = noise_config
        self.n_samples = X_clean_raw.shape[0]

        logger.info(
            "Generating fixed-bias noisy dataset (%d samples, base_seed=%d)...",
            self.n_samples,
            base_seed,
        )
        X_noisy_raw = np.empty_like(X_clean_raw, dtype=np.float32)
        for i in range(self.n_samples):
            rng = np.random.default_rng(base_seed + i)
            noisy_sample = apply_noise_batch_vectorised(
                X_clean_raw[i : i + 1], noise_config, rng=rng
            )
            X_noisy_raw[i] = noisy_sample[0]

        # Project into the scaler's feature space (identical to test pipeline)
        X_noisy_scaled = scaler.transform(X_noisy_raw).astype(np.float32)
        self.X_noisy = torch.from_numpy(X_noisy_scaled).float()
        logger.info("Fixed-bias dataset generation complete.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):  # noqa: ANN201 — Dataset protocol
        return self.X_noisy[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_indices(
    y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train, val, test) indices for the canonical 70/15/15 split."""
    n = len(y)
    indices = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=y[idx_trainval],
    )
    return idx_train, idx_val, idx_test


def _build_noise_config(cfg: dict | None = None) -> NoiseConfig:
    """Construct the headline 4-component NoiseConfig from project config.yaml."""
    if cfg is None:
        cfg = load_config()
    aug = cfg.get("noise_augmentation", {})
    return NoiseConfig.from_config_dict(aug)


def _train_fixed_bias_cnn(
    X_clean_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    noise_config: NoiseConfig,
    scaler: RobustScaler,
    n_features: int,
    epochs: int,
    early_stopping_patience: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    base_seed: int = 123,
    device: str | None = None,
) -> tuple[EITConv1D, dict[str, list[float]]]:
    """Train a CNN on the fixed-bias augmented dataset (noise sampled once per sample)."""
    if device is None:
        device = get_device()

    model = EITConv1D(
        n_features=n_features,
        n_classes=5,
        dropout=float(NOISY_CNN_PARAMS["dropout"]),
    ).to(device)

    train_dataset = FixedBiasDataset(
        X_clean_train_raw, y_train, noise_config, scaler, base_seed=base_seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_scaled).float(),
            torch.from_numpy(y_val).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(NOISY_CNN_PARAMS["label_smoothing"])
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(NOISY_CNN_PARAMS["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += X_batch.size(0)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        ep_train_loss = train_loss / train_total
        ep_val_loss = val_loss / val_total
        history["train_loss"].append(ep_train_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)
        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "[Fixed-Bias] Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch + 1,
                epochs,
                ep_train_loss,
                ep_val_loss,
                val_correct / val_total,
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info("Early stopping at epoch %d.", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Fixed-bias augmentation (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment_fixed_bias(
    data_path: Path,
    seeds: list[int],
    epochs: int,
    early_stopping_patience: int,
    output_dir: Path,
) -> dict:
    """Multi-seed fixed-bias augmentation experiment.

    Across each seed, fits a fresh scaler on raw clean training data, samples
    a per-sample fixed bias in raw voltage space, scales, and trains a CNN
    with NOISY_CNN_PARAMS regularisation. Reports clean / MATLAB-noisy /
    alt-draw test accuracy aggregated over seeds.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT (5-seed): Fixed-Bias Augmentation")
    logger.info("=" * 70)

    cfg = load_config()
    noise_config = _build_noise_config(cfg)

    X_clean_raw, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy_matlab, _ = load_mat_dataset(data_path, use_noisy=True)
    n_features = X_clean_raw.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []

    for seed in seeds:
        logger.info("--- Fixed-bias | seed=%d ---", seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        idx_train, idx_val, idx_test = _make_indices(y, seed)
        X_train_raw = X_clean_raw[idx_train]
        X_val_raw = X_clean_raw[idx_val]
        X_test_raw = X_clean_raw[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        # Scaler fit on RAW clean training data — same convention as
        # ``prepare_splits(normalize=True, scaler='robust')`` in the main grid.
        scaler = RobustScaler().fit(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
        X_test_clean_scaled = scaler.transform(X_test_raw).astype(np.float32)
        X_test_noisy_matlab_scaled = scaler.transform(X_noisy_matlab[idx_test]).astype(
            np.float32
        )

        model, _history = _train_fixed_bias_cnn(
            X_clean_train_raw=X_train_raw,
            y_train=y_train,
            X_val_scaled=X_val_scaled,
            y_val=y_val,
            noise_config=noise_config,
            scaler=scaler,
            n_features=n_features,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            base_seed=seed * 1000,
        )

        clean_metrics = evaluate_model(model, X_test_clean_scaled, y_test)
        noisy_metrics = evaluate_model(model, X_test_noisy_matlab_scaled, y_test)

        # Alternative noise draw: apply noise to RAW clean test, then scale.
        rng_alt = np.random.default_rng(seed + 9999)
        X_test_alt_raw = apply_noise_batch_vectorised(
            X_test_raw, noise_config, rng=rng_alt
        )
        X_test_alt_scaled = scaler.transform(X_test_alt_raw).astype(np.float32)
        alt_metrics = evaluate_model(model, X_test_alt_scaled, y_test)

        per_seed.append(
            {
                "seed": int(seed),
                "clean_accuracy": clean_metrics["accuracy"],
                "clean_f1": clean_metrics["f1_macro"],
                "noisy_accuracy": noisy_metrics["accuracy"],
                "noisy_f1": noisy_metrics["f1_macro"],
                "alt_noise_accuracy": alt_metrics["accuracy"],
                "alt_noise_f1": alt_metrics["f1_macro"],
            }
        )
        logger.info(
            "  seed=%d | clean=%.4f | noisy=%.4f | alt=%.4f",
            seed,
            clean_metrics["accuracy"],
            noisy_metrics["accuracy"],
            alt_metrics["accuracy"],
        )

    summary = _aggregate_runs(per_seed)
    summary["experiment"] = "fixed_bias_augmentation"
    summary["n_seeds"] = len(seeds)
    summary["seeds"] = list(map(int, seeds))
    summary["per_seed"] = per_seed

    out_path = output_dir / "fixed_bias_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Fixed-bias results saved to %s", out_path)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Different noise draw test (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment_different_draw(
    data_path: Path,
    seeds: list[int],
    output_dir: Path,
    models_dir: Path,
    n_alt_draws: int = 10,
) -> dict:
    """Multi-seed evaluation of the headline noisy CNN on independent noise draws.

    For each seed:
      * Re-derives the canonical 70/15/15 indices,
      * Re-fits a RobustScaler on the noisy training fold (matching the
        scaler under which the saved checkpoint was trained),
      * Generates ``n_alt_draws`` independent noise realisations on the raw
        clean test fold and evaluates the saved model on each.

    Args:
        data_path: Path to the .mat dataset.
        seeds: List of seeds (matches the headline grid sequence).
        output_dir: Directory for JSON results.
        models_dir: Directory containing ``cnn1d_noisy_best.pt``.
        n_alt_draws: Number of alternative draws per seed.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT (5-seed): Different Noise Draw Test")
    logger.info("=" * 70)

    model_path = Path(models_dir) / "cnn1d_noisy_best.pt"
    if not model_path.exists():
        logger.warning(
            "Model not found at %s — skipping different-draw experiment. "
            "Train it first via `eit train cnn` or `eit experiments run-all`.",
            model_path,
        )
        return {
            "experiment": "different_noise_draw_test",
            "skipped": True,
            "reason": f"missing checkpoint: {model_path}",
        }

    cfg = load_config()
    noise_config = _build_noise_config(cfg)

    X_clean_raw, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy_matlab, _ = load_mat_dataset(data_path, use_noisy=True)
    n_features = X_clean_raw.shape[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    per_seed: list[dict] = []

    for seed in seeds:
        logger.info("--- Different-draw | seed=%d ---", seed)
        idx_train, _idx_val, idx_test = _make_indices(y, seed)

        # The saved checkpoint was trained against the noisy-fold scaler.
        scaler = RobustScaler().fit(X_noisy_matlab[idx_train])
        X_test_noisy_matlab_scaled = scaler.transform(X_noisy_matlab[idx_test]).astype(
            np.float32
        )
        y_test = y[idx_test]
        X_test_clean_raw = X_clean_raw[idx_test]

        model = EITConv1D(n_features=n_features, n_classes=5)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)

        original = evaluate_model(model, X_test_noisy_matlab_scaled, y_test)

        alt_accs: list[float] = []
        alt_f1s: list[float] = []
        # Use deterministic per-seed alternative draws.
        for k, draw_seed in enumerate(
            range(seed * 10_000 + 1, seed * 10_000 + 1 + n_alt_draws)
        ):
            rng = np.random.default_rng(draw_seed)
            X_alt_raw = apply_noise_batch_vectorised(
                X_test_clean_raw, noise_config, rng=rng
            )
            X_alt_scaled = scaler.transform(X_alt_raw).astype(np.float32)
            metrics = evaluate_model(model, X_alt_scaled, y_test)
            alt_accs.append(metrics["accuracy"])
            alt_f1s.append(metrics["f1_macro"])
            logger.debug(
                "    draw %d (seed=%d): acc=%.4f", k, draw_seed, metrics["accuracy"]
            )

        per_seed.append(
            {
                "seed": int(seed),
                "original_noisy_accuracy": original["accuracy"],
                "original_noisy_f1": original["f1_macro"],
                "alt_draw_mean_accuracy": float(np.mean(alt_accs)),
                "alt_draw_std_accuracy": float(np.std(alt_accs)),
                "alt_draw_min_accuracy": float(np.min(alt_accs)),
                "alt_draw_max_accuracy": float(np.max(alt_accs)),
                "alt_draw_mean_f1": float(np.mean(alt_f1s)),
                "alt_draw_std_f1": float(np.std(alt_f1s)),
                "n_alt_draws": int(n_alt_draws),
            }
        )
        logger.info(
            "  seed=%d | original=%.4f | alt-mean=%.4f ± %.4f",
            seed,
            original["accuracy"],
            np.mean(alt_accs),
            np.std(alt_accs),
        )

    # Cross-seed summary
    flat_orig = [r["original_noisy_accuracy"] for r in per_seed]
    flat_alt = [r["alt_draw_mean_accuracy"] for r in per_seed]
    summary = {
        "experiment": "different_noise_draw_test",
        "n_seeds": len(seeds),
        "seeds": list(map(int, seeds)),
        "n_alt_draws_per_seed": int(n_alt_draws),
        "original_noisy_accuracy_mean": float(np.mean(flat_orig)),
        "original_noisy_accuracy_std": float(np.std(flat_orig)),
        "alt_draw_accuracy_mean": float(np.mean(flat_alt)),
        "alt_draw_accuracy_std": float(np.std(flat_alt)),
        "accuracy_drop_mean": float(np.mean(flat_orig) - np.mean(flat_alt)),
        "per_seed": per_seed,
    }

    out_path = output_dir / "different_draw_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Different-draw results saved to %s", out_path)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate_runs(per_seed: list[dict]) -> dict:
    """Compute mean ± std across seeds for every numeric field."""
    summary: dict = {}
    if not per_seed:
        return summary
    keys = [
        k for k, v in per_seed[0].items() if isinstance(v, (int, float)) and k != "seed"
    ]
    for key in keys:
        values = [r[key] for r in per_seed]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────


def run_additional_experiments(
    data_path: Path,
    seeds: list[int],
    epochs: int = 200,
    early_stopping_patience: int = 40,
    output_dir: Path = Path("results/reports/additional"),
    models_dir: Path = Path("results/models"),
) -> dict:
    """Run all additional experiments under the same multi-seed protocol.

    Mirrors the API of :func:`run_all_extended_experiments` so it can slot
    into ``eit experiments run-all``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    results["fixed_bias"] = run_experiment_fixed_bias(
        data_path=data_path,
        seeds=seeds,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        output_dir=output_dir,
    )
    results["different_draw"] = run_experiment_different_draw(
        data_path=data_path,
        seeds=seeds,
        output_dir=output_dir,
        models_dir=models_dir,
    )

    combined_path = output_dir / "additional_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Combined additional-experiments summary saved to %s", combined_path)

    logger.info("=" * 70)
    logger.info("ALL ADDITIONAL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    fb = results["fixed_bias"]
    logger.info(
        "Fixed-bias  | clean=%.4f ± %.4f | noisy=%.4f ± %.4f | alt=%.4f ± %.4f",
        fb["clean_accuracy_mean"],
        fb["clean_accuracy_std"],
        fb["noisy_accuracy_mean"],
        fb["noisy_accuracy_std"],
        fb["alt_noise_accuracy_mean"],
        fb["alt_noise_accuracy_std"],
    )
    dd = results["different_draw"]
    if not dd.get("skipped"):
        logger.info(
            "Different-draw | original=%.4f ± %.4f | alt=%.4f ± %.4f | drop=%+.4f",
            dd["original_noisy_accuracy_mean"],
            dd["original_noisy_accuracy_std"],
            dd["alt_draw_accuracy_mean"],
            dd["alt_draw_accuracy_std"],
            dd["accuracy_drop_mean"],
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the additional dissertation experiments under the 5-seed protocol."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds (default: 5, matching the headline protocol).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reports/additional"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("results/models"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override path to the .mat dataset (defaults to data.path in config).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_local = load_config()
    seed0 = int(cfg_local.get("seed", 42))
    seed_list = list(range(seed0, seed0 + args.seeds))
    data_path = args.data_path or Path(cfg_local["data"]["path"])

    run_additional_experiments(
        data_path=data_path,
        seeds=seed_list,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
