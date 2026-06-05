"""Evaluate commands."""

import click


@click.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model.",
)
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option("--noise/--no-noise", default=False, help="Apply noise to test data.")
def evaluate(model_path: str, config: str | None, noise: bool) -> None:
    """Evaluate a trained model on the test set."""
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.data import load_mat_dataset, prepare_splits
    from eit_sim2real.evaluate import evaluate_model, load_model

    cfg = load_config(config)
    X, y = load_mat_dataset(cfg["data"]["path"])
    splits = prepare_splits(X, y, random_state=cfg.get("seed", 42))

    model = load_model(Path(model_path), n_features=splits.X_test.shape[1])
    results = evaluate_model(model, splits.X_test, splits.y_test)

    click.echo(f"Accuracy: {results['accuracy']:.4f}")
    click.echo(f"F1 (macro): {results['f1_macro']:.4f}")
    click.echo(results["report"])
