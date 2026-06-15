"""Click CLI for the eit-sim2real package."""

import logging

import click

from eit_sim2real.cli.environment import log_environment
from eit_sim2real.cli.evaluate import evaluate
from eit_sim2real.cli.experiments import experiments
from eit_sim2real.cli.train import train
from eit_sim2real.cli.validate import validate_dataset


def _configure_logging() -> None:
    """Configure root logging once so device banners and progress reach the terminal."""
    root = logging.getLogger()
    if root.handlers:
        # Logging already configured (e.g. by a subcommand or test harness).
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(package_name="eit-sim2real")
def cli() -> None:
    """EIT Sim-to-Real: touch classification with noise augmentation."""
    _configure_logging()


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(experiments)
cli.add_command(validate_dataset)
cli.add_command(log_environment)
