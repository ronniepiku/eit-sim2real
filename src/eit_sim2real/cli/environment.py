"""Environment logging command."""

import click


@click.command("log-environment")
@click.option(
    "--output",
    type=click.Path(),
    default="results/environment.json",
    help="Output JSON path.",
)
def log_environment(output: str) -> None:
    """Log current environment details to JSON."""
    from eit_sim2real.log_environment import main as log_main

    log_main(["--output", output])
