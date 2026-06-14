"""Experiment commands."""

import click


@click.group()
def experiments() -> None:
    """Run experiment pipelines."""


@experiments.command("run-all")
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results",
    help="Results output directory.",
)
def run_all(config: str | None, output_dir: str) -> None:
    """Run the full Model x Dataset x Condition experiment grid."""
    from eit_sim2real.experiments.grid import main as grid_main

    grid_main()


@experiments.command("additional")
def additional() -> None:
    """Run additional memorisation experiments (fixed-bias, different-draw)."""
    from eit_sim2real.experiments.additional import main as additional_main

    additional_main()


@experiments.command("ablation")
@click.option(
    "--model",
    type=click.Choice(["cnn1d", "svm", "random_forest", "mlp"]),
    default="cnn1d",
    help="Model to use.",
)
@click.option(
    "--n-seeds", type=int, default=5, help="Number of independent seeds (default: 5)."
)
@click.option("--seed", type=int, default=42, help="Base random seed.")
@click.option("--epochs", type=int, default=200, help="Max CNN training epochs.")
@click.option(
    "--skip-all-configs", is_flag=True, help="Skip exhaustive subset/order ablation."
)
@click.option(
    "--no-severity-sweep", is_flag=True, help="Skip per-component severity sweep."
)
def ablation(
    model: str,
    n_seeds: int,
    seed: int,
    epochs: int,
    skip_all_configs: bool,
    no_severity_sweep: bool,
) -> None:
    """Run noise component ablation study (5-seed by default)."""
    import time
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.experiments.ablation import generate_ablation_report, run_ablation

    cfg = load_config()
    data_path = Path(cfg["data"]["path"])
    output_dir = Path("results/reports")
    figures_dir = Path("results/figures/ablation")

    start_time = time.time()
    study = run_ablation(
        data_path=data_path,
        model_name=model,
        seed=seed,
        n_seeds=n_seeds,
        run_all_configs=not skip_all_configs,
        run_severity_sweep=not no_severity_sweep,
        epochs=epochs,
        figures_dir=figures_dir,
        output_dir=output_dir,
    )
    total_time = time.time() - start_time

    study.save(Path("results/tables/ablation_results.csv"))
    generate_ablation_report(study, output_dir, total_time, model)


@experiments.command("hyperopt")
def hyperopt() -> None:
    """Run hyperparameter optimisation grid search."""
    from eit_sim2real.experiments.hyperopt import main as hyperopt_main

    hyperopt_main()


@experiments.command("architecture-sweep")
def architecture_sweep() -> None:
    """Run architecture depth/width sweep."""
    from eit_sim2real.experiments.architecture_sweep import main as sweep_main

    sweep_main()
