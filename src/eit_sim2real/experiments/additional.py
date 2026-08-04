"""Additional experiments for dissertation: fixed-bias augmentation & different-draw test.

Experiment 1: Fixed-Bias Augmentation
--------------------------------------
Samples noise ONCE per training instance (using sample index as seed) and holds
it fixed across all epochs. This mimics having many different physical devices
in the training set, where each device has its own persistent electrode bias.

Hypothesis: This should combine the clean-domain awareness of online augmentation
(many different noise patterns) with the noise-persistence that enables learning
noise-accommodation (fixed per sample across epochs).

Experiment 2: Different Noise Draw Test
-----------------------------------------
Evaluates the existing noisy-trained CNN on a NEW noisy test set generated with
a different random seed. If accuracy holds (~76%), this provides strong evidence
against pure noise memorisation. If it drops substantially, the model has
memorised distributional statistics rather than learning robust features.

Usage:
    cd python
    python run_additional_experiments.py

    # Or run individually:
    python run_additional_experiments.py --experiment fixed_bias
    python run_additional_experiments.py --experiment different_draw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eit_sim2real.configs import load_config
from eit_sim2real.constants import PROJECT_ROOT
from eit_sim2real.data import load_mat_dataset, prepare_splits
from eit_sim2real.data.noise import (
    NoiseConfig,
    apply_noise_batch_vectorised,
    apply_noise_in_scaled_space,
)
from eit_sim2real.evaluate import evaluate_model
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.utils import get_device, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "additional_experiments"


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Fixed-Bias Augmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────


class FixedBiasDataset(Dataset):
    """PyTorch Dataset that applies noise ONCE per sample and caches it.

    Each sample gets a unique noise realisation (seeded by sample index),
    which remains fixed across all epochs. This mimics the deployment scenario
    where each physical device has persistent per-electrode characteristics.
    """

    def __init__(
        self,
        X_clean: np.ndarray,
        y: np.ndarray,
        noise_config: NoiseConfig,
        scaler: object | None = None,
        base_seed: int = 123,
    ):
        """
        Args:
            X_clean: Clean (pre-noise) features, shape (n_samples, n_features).
            y: Labels, shape (n_samples,).
            noise_config: Noise configuration to apply.
            scaler: Optional scaler for applying noise in raw voltage space and
                restoring the model input space.
            base_seed: Base seed; each sample i uses seed = base_seed + i.
        """
        self.y = torch.from_numpy(y).long()
        self.noise_config = noise_config
        self.n_samples = X_clean.shape[0]

        # Apply noise ONCE per sample with a deterministic per-sample seed
        logger.info(
            f"Generating fixed-bias noisy dataset ({self.n_samples} samples)..."
        )
        self.X_noisy = np.empty_like(X_clean, dtype=np.float32)
        for i in range(self.n_samples):
            rng = np.random.default_rng(base_seed + i)
            if scaler is not None:
                noisy_sample = apply_noise_in_scaled_space(
                    X_clean[i : i + 1],
                    scaler,
                    noise_config,
                    rng=rng,
                )
            else:
                noisy_sample = apply_noise_batch_vectorised(
                    X_clean[i : i + 1], noise_config, rng=rng
                )
            self.X_noisy[i] = noisy_sample[0]

        self.X_noisy = torch.from_numpy(self.X_noisy).float()
        logger.info("Fixed-bias dataset generation complete.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X_noisy[idx], self.y[idx]


def train_fixed_bias_cnn(
    X_clean_train: np.ndarray,
    y_train: np.ndarray,
    X_val_noisy: np.ndarray,
    y_val: np.ndarray,
    noise_config: NoiseConfig,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 40,
    label_smoothing: float = 0.05,
    dropout: float = 0.4,
    device: str = "auto",
    input_scaler: object | None = None,
    base_seed: int = 123,
) -> tuple[EITConv1D, dict]:
    """Train CNN on fixed-bias augmented data.

    The key difference from online augmentation: noise is sampled ONCE per
    training sample and held constant across all epochs. Each sample gets
    a different noise draw (simulating different devices), but the same sample
    always has the same noise (simulating per-device persistence).

    Args:
        X_clean_train: Clean training features.
        y_train: Training labels.
        X_val_noisy: Noisy validation features (for monitoring).
        y_val: Validation labels.
        noise_config: Noise configuration.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        scheduler_patience: LR scheduler patience.
        scheduler_factor: LR reduction factor.
        early_stopping_patience: Early stopping patience.
        label_smoothing: Label smoothing epsilon.
        dropout: Dropout rate.
        device: Compute device.
        input_scaler: Optional scaler for applying noise in raw voltage space.
        base_seed: Base seed for per-sample noise generation.

    Returns:
        Tuple of (best model, training history).
    """
    if device == "auto":
        device = get_device()

    n_features = X_clean_train.shape[1]
    model = EITConv1D(n_features=n_features, n_classes=5, dropout=dropout).to(device)

    # Create fixed-bias dataset (noise applied once, cached)
    train_dataset = FixedBiasDataset(
        X_clean_train,
        y_train,
        noise_config,
        scaler=input_scaler,
        base_seed=base_seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_noisy).float(),
            torch.from_numpy(y_val).long(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
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

        # Validation
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
        ep_train_acc = train_correct / train_total
        ep_val_acc = val_correct / val_total

        history["train_loss"].append(ep_train_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_acc"].append(ep_val_acc)

        scheduler.step(ep_val_loss)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"[Fixed-Bias] Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} | "
                f"Val Acc: {ep_val_acc:.4f}"
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment_1_fixed_bias():
    """Experiment 1: Fixed-bias augmentation training and evaluation."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Fixed-Bias Augmentation")
    logger.info("=" * 70)

    set_seeds(42)
    cfg = load_config()
    data_path = PROJECT_ROOT / cfg["data"]["path"]

    # Load CLEAN data
    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    logger.info(f"Loaded clean dataset: {X_clean.shape}")

    # Load NOISY data (for test evaluation)
    X_noisy, _ = load_mat_dataset(data_path, use_noisy=True)
    logger.info(f"Loaded noisy dataset: {X_noisy.shape}")

    # Prepare splits (normalise based on clean training data)
    splits = prepare_splits(X_clean, y, normalize=True, scaler_type="robust")
    X_train_clean = splits.X_train
    _X_val_clean = splits.X_val  # noqa: F841
    X_test_clean = splits.X_test
    y_train = splits.y_train
    y_val = splits.y_val
    y_test = splits.y_test
    scaler = splits.scaler

    # Apply same scaler to noisy data
    X_noisy_scaled = scaler.transform(X_noisy)
    # Get test split indices by re-splitting
    from sklearn.model_selection import train_test_split

    n = len(y)
    indices = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15 / 0.85, random_state=42, stratify=y[idx_trainval]
    )
    X_test_noisy = X_noisy_scaled[idx_test]
    X_val_noisy = X_noisy_scaled[idx_val]

    # Configure noise model (same as main experiments)
    noise_config = NoiseConfig(
        enabled=True,
        snr_db=cfg["noise_augmentation"]["snr_db"],
        noise_floor=cfg["noise_augmentation"]["noise_floor"],
        contact_impedance_std_percent=cfg["noise_augmentation"][
            "contact_impedance_std_percent"
        ],
        max_bias=cfg["noise_augmentation"]["max_bias"],
        adc_bits=cfg["noise_augmentation"]["adc_bits"],
        voltage_range=cfg["noise_augmentation"]["voltage_range"],
        n_electrodes=cfg["noise_augmentation"]["n_electrodes"],
        severity=1.0,
    )

    # Train with fixed-bias augmentation
    model, history = train_fixed_bias_cnn(
        X_clean_train=X_train_clean,
        y_train=y_train,
        X_val_noisy=X_val_noisy,
        y_val=y_val,
        noise_config=noise_config,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training_noisy"]["weight_decay"],
        scheduler_patience=cfg["training"]["scheduler_patience"],
        scheduler_factor=cfg["training"]["scheduler_factor"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        label_smoothing=cfg["training_noisy"]["label_smoothing"],
        dropout=cfg["training_noisy"]["dropout"],
        input_scaler=scaler,
    )

    # Evaluate on BOTH clean and noisy test sets
    logger.info("\n--- Evaluation: Fixed-Bias Model ---")

    results_clean = evaluate_model(model, X_test_clean, y_test)
    logger.info(f"  Clean test accuracy:  {results_clean['accuracy']:.4f}")
    logger.info(f"  Clean test F1 macro:  {results_clean['f1_macro']:.4f}")

    results_noisy = evaluate_model(model, X_test_noisy, y_test)
    logger.info(f"  Noisy test accuracy:  {results_noisy['accuracy']:.4f}")
    logger.info(f"  Noisy test F1 macro:  {results_noisy['f1_macro']:.4f}")

    # Also evaluate on a DIFFERENT noise draw (seed 999)
    rng_alt = np.random.default_rng(999)
    X_test_clean_unscaled = scaler.inverse_transform(X_test_clean)
    X_test_alt_noisy = apply_noise_batch_vectorised(
        X_test_clean_unscaled, noise_config, rng=rng_alt
    )
    X_test_alt_noisy_scaled = scaler.transform(X_test_alt_noisy)

    results_alt_noisy = evaluate_model(model, X_test_alt_noisy_scaled, y_test)
    logger.info(f"  Alt-noise test accuracy: {results_alt_noisy['accuracy']:.4f}")
    logger.info(f"  Alt-noise test F1 macro: {results_alt_noisy['f1_macro']:.4f}")

    # Save results
    output_dir = RESULTS_DIR / "fixed_bias"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "cnn1d_fixed_bias_best.pt")

    results_summary = {
        "experiment": "fixed_bias_augmentation",
        "description": (
            "Noise sampled ONCE per training sample, held fixed across epochs. "
            "Mimics per-device persistent noise characteristics."
        ),
        "clean_accuracy": results_clean["accuracy"],
        "clean_f1": results_clean["f1_macro"],
        "noisy_accuracy": results_noisy["accuracy"],
        "noisy_f1": results_noisy["f1_macro"],
        "alt_noise_accuracy": results_alt_noisy["accuracy"],
        "alt_noise_f1": results_alt_noisy["f1_macro"],
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": float(min(history["val_loss"])),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Save training history
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(
        f"\nSUMMARY — Fixed-Bias Augmentation:\n"
        f"  Clean accuracy:     {results_clean['accuracy']:.1%}\n"
        f"  Noisy accuracy:     {results_noisy['accuracy']:.1%}\n"
        f"  Alt-noise accuracy: {results_alt_noisy['accuracy']:.1%}"
    )

    return results_summary


def run_experiment_2_different_draw():
    """Experiment 2: Evaluate existing noisy-trained CNN on different noise draws."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: Different Noise Draw Test (Memorisation Check)")
    logger.info("=" * 70)

    set_seeds(42)
    cfg = load_config()
    data_path = PROJECT_ROOT / cfg["data"]["path"]

    # Load clean data and original noisy data
    X_clean, y = load_mat_dataset(data_path, use_noisy=False)
    X_noisy_original, _ = load_mat_dataset(data_path, use_noisy=True)

    # Prepare splits for NOISY data (this is what the model was trained on)
    # The model was trained on noisy data, normalised with a scaler fit on noisy training data
    splits_noisy = prepare_splits(
        X_noisy_original, y, normalize=True, scaler_type="robust"
    )
    noisy_scaler = splits_noisy.scaler
    X_test_noisy_original = splits_noisy.X_test
    y_test = splits_noisy.y_test

    # Get the corresponding clean test samples (same indices)
    # We need to re-split clean data with same random_state to get matching indices
    from sklearn.model_selection import train_test_split

    n = len(y)
    indices = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=y
    )
    X_test_clean_raw = X_clean[idx_test]  # UN-normalised clean test data

    # Noise config (same parameters as MATLAB generation)
    noise_config = NoiseConfig(
        enabled=True,
        snr_db=cfg["noise_augmentation"]["snr_db"],
        noise_floor=cfg["noise_augmentation"]["noise_floor"],
        contact_impedance_std_percent=cfg["noise_augmentation"][
            "contact_impedance_std_percent"
        ],
        max_bias=cfg["noise_augmentation"]["max_bias"],
        adc_bits=cfg["noise_augmentation"]["adc_bits"],
        voltage_range=cfg["noise_augmentation"]["voltage_range"],
        n_electrodes=cfg["noise_augmentation"]["n_electrodes"],
        severity=1.0,
    )

    # Load the existing noisy-trained CNN
    model_path = PROJECT_ROOT / "results" / "models" / "cnn1d_noisy_best.pt"
    if not model_path.exists():
        logger.error(
            f"Model not found at {model_path}. "
            "Please ensure the main experiments have been run first."
        )
        sys.exit(1)

    n_features = X_clean.shape[1]
    model = EITConv1D(n_features=n_features, n_classes=5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logger.info(f"Loaded model from: {model_path}")

    # Evaluate on original noisy test set (sanity check — should be ~76%)
    results_original = evaluate_model(model, X_test_noisy_original, y_test)
    logger.info(
        f"\nOriginal noisy test (sanity check): {results_original['accuracy']:.4f}"
    )

    # Generate MULTIPLE different noise draws and evaluate
    # Apply noise to raw clean test data, then normalise with the NOISY scaler
    different_seeds = [100, 200, 300, 500, 777, 999, 1234, 2024, 5555, 9999]
    alt_results = []

    for seed in different_seeds:
        rng = np.random.default_rng(seed)
        X_test_alt_raw = apply_noise_batch_vectorised(
            X_test_clean_raw, noise_config, rng=rng
        )
        # Normalise with the SAME scaler that was used for model training
        X_test_alt_scaled = noisy_scaler.transform(X_test_alt_raw)
        result = evaluate_model(model, X_test_alt_scaled, y_test)
        alt_results.append(
            {
                "seed": seed,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
            }
        )
        logger.info(f"  Seed {seed:>5d}: accuracy = {result['accuracy']:.4f}")

    # Summary statistics
    alt_accuracies = [r["accuracy"] for r in alt_results]
    mean_acc = np.mean(alt_accuracies)
    std_acc = np.std(alt_accuracies)
    min_acc = np.min(alt_accuracies)
    max_acc = np.max(alt_accuracies)

    logger.info("\n--- Different-Draw Results (10 seeds) ---")
    logger.info(f"  Original noise test: {results_original['accuracy']:.4f}")
    logger.info(f"  Alt-draw mean:       {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"  Alt-draw range:      [{min_acc:.4f}, {max_acc:.4f}]")

    drop = results_original["accuracy"] - mean_acc
    logger.info(f"  Drop from original:  {drop:+.4f}")
    logger.info("")

    if drop < 0.05:
        logger.info(
            "  INTERPRETATION: Minimal drop (<5pp) — model generalises across "
            "noise instances. Evidence AGAINST pure noise memorisation."
        )
    elif drop < 0.15:
        logger.info(
            "  INTERPRETATION: Moderate drop (5-15pp) — partial memorisation "
            "of specific noise statistics. Model has learned some distributional "
            "regularities but retains partial noise-invariant features."
        )
    else:
        logger.info(
            "  INTERPRETATION: Large drop (>15pp) — model has substantially "
            "memorised the training noise distribution. Evidence FOR noise "
            "memorisation over genuine robustness."
        )

    # Save results
    output_dir = RESULTS_DIR / "different_draw"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        "experiment": "different_noise_draw_test",
        "description": (
            "Evaluate existing noisy-trained CNN on test sets generated with "
            "different random seeds (same parametric noise distribution, "
            "different realisations). Tests memorisation vs accommodation."
        ),
        "original_noisy_accuracy": results_original["accuracy"],
        "original_noisy_f1": results_original["f1_macro"],
        "alt_draw_results": alt_results,
        "alt_draw_mean_accuracy": float(mean_acc),
        "alt_draw_std_accuracy": float(std_acc),
        "alt_draw_min_accuracy": float(min_acc),
        "alt_draw_max_accuracy": float(max_acc),
        "accuracy_drop_from_original": float(drop),
        "n_seeds_tested": len(different_seeds),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")
    return results_summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run additional dissertation experiments (fixed-bias & different-draw)."
    )
    parser.add_argument(
        "--experiment",
        choices=["fixed_bias", "different_draw", "both"],
        default="both",
        help="Which experiment(s) to run. Default: both.",
    )
    args = parser.parse_args(argv)

    results = {}

    if args.experiment in ("fixed_bias", "both"):
        results["fixed_bias"] = run_experiment_1_fixed_bias()

    if args.experiment in ("different_draw", "both"):
        results["different_draw"] = run_experiment_2_different_draw()

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)

    if "fixed_bias" in results:
        r = results["fixed_bias"]
        logger.info(
            f"\nExp 1 (Fixed-Bias Augmentation):\n"
            f"  Clean: {r['clean_accuracy']:.1%} | "
            f"Noisy: {r['noisy_accuracy']:.1%} | "
            f"Alt-noise: {r['alt_noise_accuracy']:.1%}"
        )

    if "different_draw" in results:
        r = results["different_draw"]
        logger.info(
            f"\nExp 2 (Different Noise Draw):\n"
            f"  Original: {r['original_noisy_accuracy']:.1%} | "
            f"Alt-draws: {r['alt_draw_mean_accuracy']:.1%} ± "
            f"{r['alt_draw_std_accuracy']:.1%}\n"
            f"  Drop: {r['accuracy_drop_from_original']:+.1%}"
        )

    logger.info(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
