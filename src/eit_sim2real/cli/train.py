"""Train commands."""

import click


@click.group()
def train() -> None:
    """Train models."""


@train.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option("--noise/--no-noise", default=True, help="Train with noise augmentation.")
@click.option("--epochs", type=int, default=None, help="Override number of epochs.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/models",
    help="Model output directory.",
)
def cnn(config: str | None, noise: bool, epochs: int | None, output_dir: str) -> None:
    """Train the 1D-CNN model."""
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits
    from eit_sim2real.data.noise import NoiseConfig
    from eit_sim2real.train import train_cnn
    from eit_sim2real.utils import get_device

    cfg = load_config(config)
    click.echo(f"Loading dataset from {cfg['data']['path']}...")
    X, y = load_mat_dataset(cfg["data"]["path"], use_noisy=False)
    splits = prepare_splits(X, y, random_state=cfg.get("seed", 42))

    device = get_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    if epochs is not None:
        train_cfg["epochs"] = epochs

    if noise:
        noise_cfg = NoiseConfig()
        click.echo("Training CNN with noise augmentation...")
        model, history = train_cnn(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            noise_config=noise_cfg,
            input_scaler=splits.scaler,
            device=device,
            **{
                k: v
                for k, v in train_cfg.items()
                if k in ("epochs", "batch_size", "early_stopping_patience")
            },
            lr=train_cfg.get("learning_rate", 1e-3),
        )
        tag = "noisy"
    else:
        click.echo("Training CNN on clean data...")
        model, history = train_cnn(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            device=device,
            **{
                k: v
                for k, v in train_cfg.items()
                if k in ("epochs", "batch_size", "early_stopping_patience")
            },
            lr=train_cfg.get("learning_rate", 1e-3),
        )
        tag = "clean"

    import torch

    save_path = out / f"cnn1d_{tag}_best.pt"
    torch.save(model.state_dict(), save_path)
    click.echo(f"Model saved to {save_path}")


@train.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option("--noise/--no-noise", default=True, help="Train with noise augmentation.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/models",
    help="Model output directory.",
)
def baselines(config: str | None, noise: bool, output_dir: str) -> None:
    """Train all baseline models (SVM, RF, MLP)."""
    from pathlib import Path

    import joblib
    import numpy as np

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits
    from eit_sim2real.data.noise import NoiseConfig, apply_noise_in_scaled_space
    from eit_sim2real.models import get_baseline, train_baseline

    cfg = load_config(config)
    X, y = load_mat_dataset(cfg["data"]["path"], use_noisy=False)
    splits = prepare_splits(X, y, random_state=cfg.get("seed", 42))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train = splits.X_train
    y_train = splits.y_train

    if noise:
        noise_cfg = NoiseConfig()
        rng = np.random.default_rng(cfg.get("seed", 42))
        X_train = apply_noise_in_scaled_space(
            X_train,
            splits.scaler,
            noise_cfg,
            rng=rng,
        )
        tag = "noisy"
    else:
        tag = "clean"

    for name in ("svm", "random_forest", "mlp"):
        click.echo(f"Training {name} ({tag})...")
        model = get_baseline(name)
        model = train_baseline(model, X_train, y_train)
        save_path = out / f"{name}_{tag}.joblib"
        joblib.dump(model, save_path)
        click.echo(f"  Saved to {save_path}")
