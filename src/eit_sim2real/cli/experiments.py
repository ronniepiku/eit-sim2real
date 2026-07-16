"""Experiment commands."""

import click


@click.group()
def experiments() -> None:
    """Run experiment pipelines."""


@experiments.command("run-all")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML.",
)
@click.option(
    "--skip-grid",
    is_flag=True,
    help="Skip full experiment grid.",
)
@click.option(
    "--skip-ablation",
    is_flag=True,
    help="Skip ablation study.",
)
@click.option(
    "--skip-hyperopt",
    is_flag=True,
    help="Skip hyperparameter optimization (grid-search).",
)
@click.option(
    "--skip-architecture-sweep",
    is_flag=True,
    help="Skip architecture sweep.",
)
@click.option(
    "--skip-extended",
    is_flag=True,
    help="Skip extended experiments (statistical tests, dataset size, etc.).",
)
@click.option(
    "--skip-additional",
    is_flag=True,
    help="Skip additional memorisation experiments.",
)
@click.option(
    "--include-mesh-refinement",
    is_flag=True,
    help="Include mesh refinement study (requires fine-mesh dataset).",
)
@click.option(
    "--skip-consistency-gate",
    is_flag=True,
    help="Skip cross-report benchmark consistency validation.",
)
def run_all(
    config: str | None,
    skip_grid: bool,
    skip_ablation: bool,
    skip_hyperopt: bool,
    skip_architecture_sweep: bool,
    skip_extended: bool,
    skip_additional: bool,
    include_mesh_refinement: bool,
    skip_consistency_gate: bool,
) -> None:
    """Run all experiments in the project.\n\n
    By default, runs: grid, ablation, hyperopt, architecture-sweep, and extended.
    Optionally includes: additional and mesh-refinement experiments.
    Each can be individually disabled using --skip-* flags.
    """
    import logging
    import sys
    import time

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("MASTER EXPERIMENT RUNNER - ALL EXPERIMENTS")
    logger.info("=" * 70)

    start_time = time.time()
    experiments_run = []
    experiments_failed = []

    # 1. Full experiment grid
    if not skip_grid:
        logger.info("\n" + "=" * 70)
        logger.info("1. GRID EXPERIMENTS - Model x Dataset x Condition")
        logger.info("=" * 70)
        try:
            old_argv = sys.argv
            sys.argv = [sys.argv[0]]
            if skip_consistency_gate:
                sys.argv.append("--skip-consistency-gate")
            from eit_sim2real.experiments.grid import main as grid_main

            grid_main()
            experiments_run.append("grid")
        except Exception:
            logger.exception("Grid experiments failed")
            experiments_failed.append("grid")
        finally:
            sys.argv = old_argv

    # 2. Ablation study
    if not skip_ablation:
        logger.info("\n" + "=" * 70)
        logger.info("2. ABLATION STUDY - Noise Component Analysis")
        logger.info("=" * 70)
        try:
            import click as click_module

            from eit_sim2real.cli.experiments import ablation as ablation_cmd

            ctx = click_module.Context(ablation_cmd)
            ctx.invoke(
                ablation_cmd,
                model="cnn1d",
                n_seeds=5,
                seed=42,
                epochs=200,
                skip_all_configs=False,
                no_severity_sweep=False,
            )
            experiments_run.append("ablation")
        except Exception:
            logger.exception("Ablation study failed")
            experiments_failed.append("ablation")

    # 3. Hyperparameter optimization (grid-search)
    if not skip_hyperopt:
        logger.info("\n" + "=" * 70)
        logger.info("3. HYPERPARAMETER OPTIMIZATION - Grid Search")
        logger.info("=" * 70)
        try:
            old_argv = sys.argv
            sys.argv = [sys.argv[0], "--mode", "grid-search"]
            from eit_sim2real.experiments.hyperopt import main as hyperopt_main

            hyperopt_main()
            sys.argv = old_argv
            experiments_run.append("hyperopt")
        except Exception:
            logger.exception("Hyperparameter optimization failed")
            experiments_failed.append("hyperopt")
        finally:
            sys.argv = old_argv

    # 4. Architecture sweep
    if not skip_architecture_sweep:
        logger.info("\n" + "=" * 70)
        logger.info("4. ARCHITECTURE SWEEP - Depth Search")
        logger.info("=" * 70)
        try:
            old_argv = sys.argv
            sys.argv = [sys.argv[0], "--mode", "arch-sweep"]
            from eit_sim2real.experiments.hyperopt import main as hyperopt_main

            hyperopt_main()
            sys.argv = old_argv
            experiments_run.append("architecture-sweep")
        except Exception:
            logger.exception("Architecture sweep failed")
            experiments_failed.append("architecture-sweep")
        finally:
            sys.argv = old_argv

    # 5. Extended experiments
    if not skip_extended:
        logger.info("\n" + "=" * 70)
        logger.info("5. EXTENDED EXPERIMENTS - Analysis")
        logger.info("=" * 70)
        try:
            import click as click_module

            from eit_sim2real.cli.experiments import extended_cmd

            ctx = click_module.Context(extended_cmd)
            ctx.invoke(
                extended_cmd,
                config=config,
                seeds=5,
                seed=42,
                epochs=200,
                early_stopping_patience=40,
                output_dir="results/reports",
                figures_dir="results/figures",
            )
            experiments_run.append("extended")
        except Exception:
            logger.exception("Extended experiments failed")
            experiments_failed.append("extended")

    # 6. Additional experiments
    if not skip_additional:
        logger.info("\n" + "=" * 70)
        logger.info("6. ADDITIONAL EXPERIMENTS - Memorisation Studies")
        logger.info("=" * 70)
        try:
            from eit_sim2real.experiments.additional import main as additional_main

            additional_main([])
            experiments_run.append("additional")
        except Exception:
            logger.exception("Additional experiments failed")
            experiments_failed.append("additional")

    # 7. Mesh refinement (optional)
    if include_mesh_refinement:
        logger.info("\n" + "=" * 70)
        logger.info("7. MESH REFINEMENT - Cross-Mesh Evaluation")
        logger.info("=" * 70)
        try:
            from eit_sim2real.experiments.mesh_refinement import main as mesh_main

            mesh_main(
                baseline_dataset="data/eit_dataset.mat",
                fine_mesh_dataset="data/eit_dataset_mesh_f.mat",
                noisy_model="results/models/cnn1d_noisy_best.pt",
                clean_model="results/models/cnn1d_clean_best.pt",
                output_dir="results/additional_experiments/mesh_refinement",
                seed=42,
            )
            experiments_run.append("mesh-refinement")
        except Exception:
            logger.exception("Mesh refinement failed")
            experiments_failed.append("mesh-refinement")

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"Completed: {', '.join(experiments_run) if experiments_run else 'none'}"
    )
    if experiments_failed:
        logger.error(f"Failed: {', '.join(experiments_failed)}")
        sys.exit(1)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info("=" * 70)


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


@experiments.command("additional")
def additional() -> None:
    """Run additional memorisation experiments (fixed-bias, different-draw)."""
    from eit_sim2real.experiments.additional import main as additional_main

    additional_main([])


@experiments.command("architecture-sweep")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML (matches train.py parameter).",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Override path to the .mat dataset (defaults to data.path in config).",
)
@click.option(
    "--noise/--no-noise",
    default=True,
    show_default=True,
    help="Train with noise augmentation (matches train.py parameter).",
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
    default=None,
    show_default=True,
    help="Directory for results (default: results/architecture_sweep).",
)
@click.option("--seed", type=int, default=42, show_default=True)
def architecture_sweep(
    config: str | None,
    data_path: str | None,
    noise: bool,
    dev_fraction: float,
    output_dir: str | None,
    seed: int,
) -> None:
    """Run architecture depth sweep over {2,3,4,5} conv blocks (uses GPU if available).

    NOTE: This is now integrated into the unified hyperopt module.
    This command runs hyperopt with --mode=arch-sweep for convenience.
    Parameters match train.py: --config, --noise/--no-noise, --epochs, --output-dir.
    """
    import sys as sys_module

    from eit_sim2real.experiments.hyperopt import main as hyperopt_main

    # Construct equivalent arguments for unified hyperopt
    sys_argv = ["--mode", "arch-sweep"]
    if config:
        sys_argv.extend(["--config", config])
    if data_path:
        sys_argv.extend(["--data-path", data_path])
    if not noise:  # Only add flag if noise is False (to match train.py --no-noise)
        sys_argv.append("--no-noise")
    if dev_fraction != 0.1:
        sys_argv.extend(["--dev-fraction", str(dev_fraction)])
    if output_dir:
        sys_argv.extend(["--output-dir", output_dir])
    sys_argv.extend(["--seed", str(seed)])

    # Parse and execute unified hyperopt
    old_argv = sys_module.argv
    try:
        sys_module.argv = [""] + sys_argv
        from eit_sim2real.experiments.hyperopt import main as hyperopt_main

        hyperopt_main()
    finally:
        sys_module.argv = old_argv


@experiments.command("extended")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML.",
)
@click.option(
    "--seeds",
    type=int,
    default=5,
    show_default=True,
    help="Number of independent seeds for statistical testing.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Base random seed.",
)
@click.option("--epochs", type=int, default=200, help="Max CNN training epochs.")
@click.option(
    "--early-stopping-patience",
    type=int,
    default=40,
    help="Early stopping patience.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/reports",
    help="Results output directory.",
)
@click.option(
    "--figures-dir",
    type=click.Path(),
    default="results/figures",
    help="Figures output directory.",
)
def extended_cmd(
    config: str | None,
    seeds: int,
    seed: int,
    epochs: int,
    early_stopping_patience: int,
    output_dir: str,
    figures_dir: str,
) -> None:
    """Run extended analysis experiments.

    Includes: statistical tests, dataset size effects, noise-type severity sweep,
    Gaussian-only evaluation, confidence calibration, per-class robustness,
    noise parameter sensitivity, and hyperparameter sensitivity.
    """
    import logging
    import time
    from pathlib import Path

    from eit_sim2real.configs import load_config
    from eit_sim2real.evaluate import (
        run_calibration_analysis,
        run_gaussian_only_evaluation,
        run_noise_parameter_sensitivity,
        run_per_class_robustness,
    )
    from eit_sim2real.experiments.grid import (
        run_dataset_size_experiment,
        run_noise_type_severity_sweep,
        run_statistical_tests,
    )
    from eit_sim2real.experiments.hyperopt import run_hyperparameter_sensitivity

    logger = logging.getLogger(__name__)

    cfg = load_config(config) if config else load_config()
    data_path = Path(cfg["data"]["path"])
    output_path = Path(output_dir)
    figures_path = Path(figures_dir)

    logger.info("=" * 70)
    logger.info("EXTENDED EXPERIMENTS")
    logger.info("=" * 70)

    start_time = time.time()
    all_results = {}

    # 1. Statistical testing
    logger.info("\nRunning: Statistical Tests")
    seed_list = list(range(seed, seed + seeds))
    stat_results = run_statistical_tests(
        data_path, seed_list, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["statistical_tests"] = stat_results

    # 2. Dataset size effects
    logger.info("\nRunning: Dataset Size Experiment")
    size_results = run_dataset_size_experiment(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["dataset_size"] = size_results

    # 3. Noise-type severity sweep
    logger.info("\nRunning: Noise-Type Severity Sweep")
    severity_results = run_noise_type_severity_sweep(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["noise_type_severity"] = severity_results

    # 4. Gaussian-only evaluation
    logger.info("\nRunning: Gaussian-Only Evaluation")
    gaussian_results = run_gaussian_only_evaluation(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["gaussian_only"] = gaussian_results

    # 5. Confidence calibration
    logger.info("\nRunning: Confidence Calibration Analysis")
    calibration_results = run_calibration_analysis(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["calibration"] = calibration_results

    # 6. Per-class robustness
    logger.info("\nRunning: Per-Class Robustness")
    per_class_results = run_per_class_robustness(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["per_class_robustness"] = per_class_results

    # 7. Noise parameter sensitivity
    logger.info("\nRunning: Noise Parameter Sensitivity")
    noise_param_results = run_noise_parameter_sensitivity(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["noise_parameter_sensitivity"] = noise_param_results

    # 8. Hyperparameter sensitivity
    logger.info("\nRunning: Hyperparameter Sensitivity")
    hp_results = run_hyperparameter_sensitivity(
        data_path, seed, epochs, early_stopping_patience, output_path, figures_path
    )
    all_results["hyperparameter_sensitivity"] = hp_results

    # Generate report
    runtime = time.time() - start_time
    # TODO: Implement extended report generation (was in deleted extended.py)
    # generate_extended_report(...)

    logger.info("\n" + "=" * 70)
    logger.info(f"Extended experiments complete — {runtime / 60:.1f} minutes total")
    logger.info("=" * 70)


@experiments.command("hyperopt")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML (matches train.py parameter).",
)
@click.option(
    "--mode",
    type=click.Choice(["arch-sweep", "grid-search"]),
    default="grid-search",
    show_default=True,
    help="Search mode: 'arch-sweep' for focused depth search, "
    "'grid-search' for comprehensive hyperparameter tuning.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Override path to the .mat dataset (defaults to data.path in config).",
)
@click.option(
    "--noise/--no-noise",
    default=True,
    show_default=True,
    help="Train with noise augmentation (matches train.py parameter).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for results (default: results/architecture_sweep or "
    "results/hyperparameter_optimisation based on mode).",
)
@click.option(
    "--n-folds",
    type=int,
    default=3,
    show_default=True,
    help="Number of CV folds (grid-search only).",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override epochs (default: from training.epochs in config for grid-search).",
)
@click.option(
    "--objective",
    type=click.Choice(["noisy_f1", "noisy_accuracy", "robustness"]),
    default=None,
    help="Grid-search optimisation objective (default: from config).",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume grid search from checkpoint (grid-search only).",
)
@click.option(
    "--final-only",
    is_flag=True,
    help="Skip search, train final model from existing results (grid-search only).",
)
@click.option("--seed", type=int, default=None, show_default=False)
def hyperopt(
    config: str | None,
    mode: str,
    data_path: str | None,
    noise: bool,
    output_dir: str | None,
    n_folds: int,
    epochs: int | None,
    objective: str | None,
    resume: bool,
    final_only: bool,
    seed: int | None,
) -> None:
    """Run unified hyperparameter optimisation (architecture search + grid search).

    Two modes available:
      - arch-sweep: Quick focused depth search with dev subset (single split).
      - grid-search: Comprehensive hyperparameter tuning with k-fold CV (default).

    Parameters match train.py: --config, --noise/--no-noise, --epochs, --output-dir.
    """
    import sys as sys_module

    from eit_sim2real.experiments.hyperopt import main as hyperopt_main

    # Construct command-line arguments for unified hyperopt
    sys_argv = ["--mode", mode]
    if config:
        sys_argv.extend(["--config", config])
    if data_path:
        sys_argv.extend(["--data-path", data_path])
    if not noise:  # Only add flag if noise is False (to match train.py --no-noise)
        sys_argv.append("--no-noise")
    if output_dir:
        sys_argv.extend(["--output-dir", output_dir])
    if n_folds != 3:
        sys_argv.extend(["--n-folds", str(n_folds)])
    if epochs is not None:
        sys_argv.extend(["--epochs", str(epochs)])
    if objective is not None:
        sys_argv.extend(["--objective", objective])
    if resume:
        sys_argv.append("--resume")
    if final_only:
        sys_argv.append("--final-only")
    if seed is not None:
        sys_argv.extend(["--seed", str(seed)])

    # Call unified hyperopt main with sys.argv manipulation
    old_argv = sys_module.argv
    try:
        sys_module.argv = [""] + sys_argv
        hyperopt_main()
    finally:
        sys_module.argv = old_argv


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
    from eit_sim2real.experiments.mesh_refinement import main as mesh_main

    mesh_main(
        baseline_dataset=baseline_dataset,
        fine_mesh_dataset=fine_mesh_dataset,
        noisy_model=noisy_model,
        clean_model=clean_model if clean_model else None,
        output_dir=output_dir,
        seed=seed,
    )
