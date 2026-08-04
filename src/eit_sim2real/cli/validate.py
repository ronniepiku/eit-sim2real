"""Dataset validation command."""

import click


@click.command("validate-dataset")
@click.option("--config", type=click.Path(exists=True), help="Path to config YAML.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/dataset_validation",
    help="Output directory.",
)
def validate_dataset(config: str | None, output_dir: str) -> None:
    """Run dataset validation and generate report."""
    import sys

    from eit_sim2real.validate_dataset import main as validate_main

    # The underlying module parses sys.argv, so forward the Click options.
    # It reads paths/seed from its own defaults and does not take a config
    # file, so --config is accepted for interface consistency but unused.
    _ = config
    argv = [sys.argv[0], "--output-dir", output_dir]

    old_argv = sys.argv
    sys.argv = argv
    try:
        validate_main()
    finally:
        sys.argv = old_argv
