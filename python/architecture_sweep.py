"""Architecture sweep for EIT 1D-CNN depth selection.

Evaluates CNN architectures with varying numbers of convolutional blocks
to justify the 3-block design choice documented in the methodology.
Results are saved to results/architecture_sweep.csv.

Usage:
    uv run python/architecture_sweep.py --data-path data/eit_dataset.mat
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from configs.loader import load_config
from data.load_dataset import load_mat_dataset, prepare_splits
from models.cnn1d import EITConv1D
from torch.utils.data import DataLoader, TensorDataset
from utils import get_device, set_seeds

logger = logging.getLogger(__name__)


def build_channel_list(n_blocks: int, base: int = 32) -> list[int]:
    """Generate channel list with consistent doubling pattern."""
    return [base * (2**i) for i in range(n_blocks)]


def train_and_evaluate(
    n_blocks: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, float]:
    """Train a CNN with specified depth and return metrics."""
    set_seeds(seed)

    device = get_device()
    channels = build_channel_list(n_blocks)
    n_features = X_train.shape[1]

    model = EITConv1D(
        n_features=n_features,
        n_classes=5,
        channels=channels,
        fc_dim=128,
        dropout=0.3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 20:
            break

    return {
        "n_blocks": n_blocks,
        "channels": str(channels),
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CNN architecture depth sweep.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data to use (development subset).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    data_path = args.data_path or Path(cfg["data"]["path"])

    set_seeds(args.seed)

    # Load and split data
    X, y = load_mat_dataset(data_path, use_noisy=True)
    dataset = prepare_splits(X, y, random_state=args.seed)

    # Use development subset for efficient sweep
    n_dev = int(len(dataset.X_train) * args.dev_fraction)
    indices = np.random.permutation(len(dataset.X_train))[:n_dev]
    X_train_dev = dataset.X_train[indices]
    y_train_dev = dataset.y_train[indices]

    logger.info(
        f"Architecture sweep: {n_dev} training samples, "
        f"full validation set ({len(dataset.X_val)} samples)"
    )

    # Sweep over block counts
    block_counts = [2, 3, 4, 5]
    results = []

    for n_blocks in block_counts:
        logger.info(
            f"\n--- {n_blocks} conv blocks (channels: {build_channel_list(n_blocks)}) ---"
        )
        result = train_and_evaluate(
            n_blocks=n_blocks,
            X_train=X_train_dev,
            y_train=y_train_dev,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            seed=args.seed,
        )
        results.append(result)
        logger.info(
            f"  Val acc: {result['best_val_acc']:.4f} | "
            f"Val loss: {result['best_val_loss']:.4f} | "
            f"Params: {result['n_params']:,} | "
            f"Best epoch: {result['best_epoch']}"
        )

    # Save results
    df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "architecture_sweep.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    # Report best
    best = df.loc[df["best_val_acc"].idxmax()]
    logger.info(
        f"\nBest architecture: {int(best['n_blocks'])} blocks "
        f"(val acc: {best['best_val_acc']:.4f})"
    )


if __name__ == "__main__":
    main()
