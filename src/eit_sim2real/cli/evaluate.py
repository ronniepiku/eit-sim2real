"""Evaluate commands."""

import click
import numpy as np


@click.command()
@click.option(
    "--model-path",
    required=False,
    type=click.Path(exists=True),
    help="Path to trained model (required unless --gaussian-only is used).",
)
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option("--noise/--no-noise", default=False, help="Apply noise to test data.")
@click.option(
    "--gaussian-only",
    is_flag=True,
    help="Run Gaussian-only literature-comparison evaluation from evaluate.py.",
)
@click.option("--seed", type=int, default=None, help="Override random seed.")
@click.option("--epochs", type=int, default=None, help="Override training epochs.")
@click.option(
    "--early-stopping-patience",
    type=int,
    default=None,
    help="Override early stopping patience.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="results/reports",
    show_default=True,
    help="Output directory for evaluation artifacts.",
)
@click.option(
    "--figures-dir",
    type=click.Path(file_okay=False),
    default="results/figures",
    show_default=True,
    help="Directory for generated figures.",
)
def evaluate(
    model_path: str | None,
    config: str | None,
    noise: bool,
    gaussian_only: bool,
    seed: int | None,
    epochs: int | None,
    early_stopping_patience: int | None,
    output_dir: str,
    figures_dir: str,
) -> None:
    """Evaluate a trained model on the test set."""
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits
    from eit_sim2real.data.noise import NoiseConfig, apply_noise_in_scaled_space
    from eit_sim2real.evaluate import (
        evaluate_model,
        load_model,
        run_gaussian_only_evaluation,
    )

    cfg = load_config(config)

    if gaussian_only:
        run_gaussian_only_evaluation(
            data_path=Path(cfg["data"]["path"]),
            seed=seed if seed is not None else cfg.get("seed", 42),
            epochs=epochs if epochs is not None else cfg["training"]["epochs"],
            early_stopping_patience=(
                early_stopping_patience
                if early_stopping_patience is not None
                else cfg["training"]["early_stopping_patience"]
            ),
            output_dir=Path(output_dir),
            figures_dir=Path(figures_dir),
        )
        click.echo("Gaussian-only evaluation complete.")
        click.echo(f"Results: {Path(output_dir) / 'gaussian_only_evaluation.json'}")
        click.echo(f"Figure: {Path(figures_dir) / 'gaussian_only_evaluation.png'}")
        return

    if not model_path:
        raise click.UsageError(
            "--model-path is required unless --gaussian-only is used."
        )

    X, y = load_mat_dataset(cfg["data"]["path"], use_noisy=False)
    splits = prepare_splits(X, y, random_state=cfg.get("seed", 42))

    X_test = splits.X_test
    if noise:
        rng = np.random.default_rng(cfg.get("seed", 42))
        X_test = apply_noise_in_scaled_space(
            X_test,
            splits.scaler,
            NoiseConfig(),
            rng=rng,
        )

    model = load_model(Path(model_path), n_features=splits.X_test.shape[1])
    results = evaluate_model(model, X_test, splits.y_test)

    click.echo(f"Accuracy: {results['accuracy']:.4f}")
    click.echo(f"F1 (macro): {results['f1_macro']:.4f}")
    click.echo(results["report"])
