"""Experiment commands."""

import click


@click.group()
def experiments() -> None:
    """Run experiment pipelines."""


@experiments.command("run-all")
@click.option(
    "--datasets",
    type=str,
    default="raw,pca,lda",
    show_default=True,
    help="Comma-separated list of datasets (raw,pca,lda).",
)
@click.option(
    "--seeds",
    type=int,
    default=5,
    show_default=True,
    help="Number of random seeds for uncertainty estimation.",
)
@click.option(
    "--epochs",
    type=int,
    default=200,
    show_default=True,
    help="Max CNN training epochs.",
)
@click.option(
    "--early-stopping-patience",
    type=int,
    default=40,
    show_default=True,
    help="CNN early stopping patience.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="results/reports",
    show_default=True,
    help="Directory for report and tables.",
)
@click.option(
    "--figures-dir",
    type=click.Path(file_okay=False),
    default="results/figures",
    show_default=True,
    help="Directory for figure outputs.",
)
@click.option(
    "--skip-ablation",
    is_flag=True,
    help="Skip the ablation study.",
)
@click.option(
    "--skip-extended",
    is_flag=True,
    help="Skip extended experiments (stats, ensemble, t-SNE, etc.).",
)
@click.option(
    "--skip-additional",
    is_flag=True,
    help="Skip additional memorisation experiments (fixed-bias, different-draw).",
)
def run_all(
    datasets: str,
    seeds: int,
    epochs: int,
    early_stopping_patience: int,
    output_dir: str,
    figures_dir: str,
    skip_ablation: bool,
    skip_extended: bool,
    skip_additional: bool,
) -> None:
    """Run the full Model x Dataset x Condition experiment grid.

    Also triggers the ablation study, extended experiments (statistical
    tests, learning curves, ensembles, calibration, ...) and the
    additional memorisation experiments (fixed-bias, different-draw),
    each over the same seed sequence so that all reported numbers share
    the same statistical protocol.
    """
    from eit_sim2real.experiments.grid import main as grid_main

    argv: list[str] = [
        "--datasets",
        *[d.strip() for d in datasets.split(",") if d.strip()],
        "--seeds",
        str(seeds),
        "--epochs",
        str(epochs),
        "--early-stopping-patience",
        str(early_stopping_patience),
        "--output-dir",
        output_dir,
        "--figures-dir",
        figures_dir,
    ]
    if skip_ablation:
        argv.append("--skip-ablation")
    if skip_extended:
        argv.append("--skip-extended")
    if skip_additional:
        argv.append("--skip-additional")

    grid_main(argv)


@experiments.command("additional")
@click.option(
    "--seeds",
    type=int,
    default=5,
    show_default=True,
    help="Number of random seeds (matches the headline 5-seed protocol).",
)
@click.option(
    "--epochs",
    type=int,
    default=200,
    show_default=True,
    help="Max CNN training epochs (used by the fixed-bias retrain).",
)
@click.option(
    "--early-stopping-patience",
    type=int,
    default=40,
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="results/reports/additional",
    show_default=True,
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False),
    default="results/models",
    show_default=True,
    help="Directory containing the headline cnn1d_noisy_best.pt checkpoint.",
)
def additional(
    seeds: int,
    epochs: int,
    early_stopping_patience: int,
    output_dir: str,
    models_dir: str,
) -> None:
    """Run the additional memorisation experiments under the 5-seed protocol.

    Includes:
      * Fixed-bias augmentation: noise sampled once per training instance and
        held fixed across epochs (mimics persistent per-device characteristics).
      * Different-draw evaluation: re-evaluates the noisy-trained CNN on test
        sets generated from independent noise realisations.
    """
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.experiments.additional import run_additional_experiments

    cfg = load_config()
    seed_list = list(range(int(cfg.get("seed", 42)), int(cfg.get("seed", 42)) + seeds))
    data_path = Path(cfg["data"]["path"])

    run_additional_experiments(
        data_path=data_path,
        seeds=seed_list,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        output_dir=Path(output_dir),
        models_dir=Path(models_dir),
    )


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
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Override path to the .mat dataset (defaults to data.path in config).",
)
@click.option(
    "--dev-fraction",
    type=float,
    default=0.1,
    show_default=True,
    help="Fraction of training data to use for the sweep.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="results",
    show_default=True,
    help="Directory in which architecture_sweep.csv is written.",
)
@click.option("--seed", type=int, default=42, show_default=True)
def architecture_sweep(
    data_path: str | None, dev_fraction: float, output_dir: str, seed: int
) -> None:
    """Run architecture depth sweep over {2,3,4,5} conv blocks (uses GPU if available)."""
    from pathlib import Path

    from eit_sim2real.experiments.architecture_sweep import run as sweep_run

    sweep_run(
        data_path=Path(data_path) if data_path else None,
        dev_fraction=dev_fraction,
        output_dir=Path(output_dir),
        seed=seed,
    )


@experiments.command("mesh-refinement")
@click.option(
    "--baseline-dataset",
    type=click.Path(exists=True, dir_okay=False),
    default="data/eit_dataset.mat",
    show_default=True,
    help="Production training dataset (coarse mesh). Used to refit the scaler "
    "the saved models expect.",
)
@click.option(
    "--fine-mesh-dataset",
    type=click.Path(dir_okay=False),
    default="data/eit_dataset_mesh_f.mat",
    show_default=True,
    help="Test set generated at an alternative mesh refinement by "
    "matlab/generate_mesh_refinement_testset.m.",
)
@click.option(
    "--noisy-model",
    type=click.Path(exists=True, dir_okay=False),
    default="results/models/cnn1d_noisy_best.pt",
    show_default=True,
    help="Noisy-trained CNN checkpoint.",
)
@click.option(
    "--clean-model",
    type=click.Path(dir_okay=False),
    default="results/models/cnn1d_clean_best.pt",
    show_default=True,
    help="Clean-trained CNN checkpoint (control). Pass an empty string to skip.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="results/additional_experiments/mesh_refinement",
    show_default=True,
    help="Where to write results.json.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed (must match the seed used to train the saved models).",
)
def mesh_refinement(
    baseline_dataset: str,
    fine_mesh_dataset: str,
    noisy_model: str,
    clean_model: str,
    output_dir: str,
    seed: int,
) -> None:
    """Cross-mesh evaluation of production CNNs (Dissertation Section 5.2).

    Evaluates the trained CNN(s) on a test set generated at a finer
    forward-model mesh refinement than the production training data,
    quantifying whether the noise-augmentation result transfers across
    discretisation fidelity.

    Prerequisites:
      1. Production training run has produced results/models/cnn1d_noisy_best.pt
         (and optionally cnn1d_clean_best.pt).
      2. Fine-mesh test set has been generated by running, in MATLAB:
            cd matlab
            generate_mesh_refinement_testset()
    """
    import logging

    from eit_sim2real.experiments.mesh_refinement import main as mesh_main

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    mesh_main(
        baseline_dataset=baseline_dataset,
        fine_mesh_dataset=fine_mesh_dataset,
        noisy_model=noisy_model,
        clean_model=clean_model if clean_model else None,
        output_dir=output_dir,
        seed=seed,
    )
