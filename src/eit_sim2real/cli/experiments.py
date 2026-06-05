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
def ablation() -> None:
    """Run noise component ablation study."""
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.experiments.ablation import run_ablation

    cfg = load_config()
    run_ablation(data_path=Path(cfg["data"]["path"]))


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
