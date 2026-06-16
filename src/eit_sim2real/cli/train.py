"""Train commands.

Protocol note
-------------
The ``cnn`` and ``baselines`` subcommands here reproduce a *single seed* of
the headline ``noisy_train_noisy_eval`` (or ``clean_train_clean_eval``)
protocol used by :mod:`eit_sim2real.experiments.grid`. They load the
pre-noised (or clean) split directly from the dataset and do **not** apply
any additional online noise augmentation -- doing so would produce a hybrid
"noisy + augmented" regime that matches no condition in the dissertation
results tables. Use ``eit experiments run-all`` to reproduce the full
5-seed mean used by Table 4.1; use these subcommands to materialise a
single deployable checkpoint at a chosen seed (default ``--seed 42``,
matching the first seed of the run-all sweep).
"""

import click

from eit_sim2real.constants import NOISY_CNN_PARAMS


@click.group()
def train() -> None:
    """Train models."""


@train.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option(
    "--noise/--no-noise",
    default=True,
    help="Train on the pre-noised (default) or clean dataset split.",
)
@click.option("--epochs", type=int, default=None, help="Override number of epochs.")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed (defaults to config seed, typically 42 -- the first "
    "seed used by `eit experiments run-all`).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/models",
    help="Model output directory.",
)
def cnn(
    config: str | None,
    noise: bool,
    epochs: int | None,
    seed: int | None,
    output_dir: str,
) -> None:
    """Train the 1D-CNN model under the headline noisy/clean protocol.

    With ``--noise`` (default) this reproduces a single seed of the
    ``noisy_train_noisy_eval`` condition: pre-noised training data,
    no online augmentation, with the stronger regularisation reported in
    the dissertation methodology (weight_decay=1e-3, dropout=0.4,
    label_smoothing=0.05). With ``--no-noise`` it reproduces the
    ``clean_train_clean_eval`` ceiling.
    """
    from pathlib import Path

    import numpy as np
    import torch

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits_from_config
    from eit_sim2real.train import train_cnn
    from eit_sim2real.utils import get_device

    cfg = load_config(config)
    seed = seed if seed is not None else cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    click.echo(f"Loading dataset from {cfg['data']['path']} (use_noisy={noise})...")
    X, y = load_mat_dataset(cfg["data"]["path"], use_noisy=noise)
    splits = prepare_splits_from_config(X, y, cfg, random_state=seed)

    device = get_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    if epochs is not None:
        train_cfg["epochs"] = epochs
    train_kwargs = {
        k: v
        for k, v in train_cfg.items()
        if k in ("epochs", "batch_size", "early_stopping_patience")
    }
    train_kwargs["lr"] = train_cfg.get("learning_rate", 1e-3)

    if noise:
        click.echo(
            f"Training CNN on pre-noised split (seed={seed}, "
            "no online augmentation, NOISY_CNN_PARAMS)..."
        )
        model, _ = train_cnn(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            device=device,
            seed=seed,
            **train_kwargs,
            **NOISY_CNN_PARAMS,
        )
        tag = "noisy"
    else:
        click.echo(f"Training CNN on clean split (seed={seed})...")
        model, _ = train_cnn(
            splits.X_train,
            splits.y_train,
            splits.X_val,
            splits.y_val,
            device=device,
            seed=seed,
            **train_kwargs,
        )
        tag = "clean"

    save_path = out / f"cnn1d_{tag}_best.pt"
    torch.save(model.state_dict(), save_path)
    click.echo(f"Model saved to {save_path}")


@train.command()
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option(
    "--noise/--no-noise",
    default=True,
    help="Train on the pre-noised (default) or clean dataset split.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed (defaults to config seed, typically 42).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/models",
    help="Model output directory.",
)
def baselines(
    config: str | None, noise: bool, seed: int | None, output_dir: str
) -> None:
    """Train all baseline models (SVM, RF, MLP) under the headline protocol.

    Mirrors the ``noisy_train`` / ``clean_train`` baseline configuration in
    :mod:`eit_sim2real.experiments.grid`: trains on the pre-noised (or
    clean) split directly, with no further noise augmentation. ``--seed``
    seeds both the train/val/test split and the model's ``random_state``.
    """
    from pathlib import Path

    import joblib

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits_from_config
    from eit_sim2real.models import get_baseline, train_baseline

    cfg = load_config(config)
    seed = seed if seed is not None else cfg.get("seed", 42)

    click.echo(f"Loading dataset from {cfg['data']['path']} (use_noisy={noise})...")
    X, y = load_mat_dataset(cfg["data"]["path"], use_noisy=noise)
    splits = prepare_splits_from_config(X, y, cfg, random_state=seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tag = "noisy" if noise else "clean"
    for name in ("svm", "random_forest", "mlp"):
        click.echo(f"Training {name} ({tag}, seed={seed})...")
        model = get_baseline(name, random_state=seed)
        model = train_baseline(model, splits.X_train, splits.y_train)
        save_path = out / f"{name}_{tag}.joblib"
        joblib.dump(model, save_path)
        click.echo(f"  Saved to {save_path}")
