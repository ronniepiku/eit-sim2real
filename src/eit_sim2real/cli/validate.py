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
    from eit_sim2real.validate_dataset import main as validate_main

    validate_main()
