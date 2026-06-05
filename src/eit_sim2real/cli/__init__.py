"""Click CLI for the eit-sim2real package."""

import click

from eit_sim2real.cli.environment import log_environment
from eit_sim2real.cli.evaluate import evaluate
from eit_sim2real.cli.experiments import experiments
from eit_sim2real.cli.train import train
from eit_sim2real.cli.validate import validate_dataset


@click.group()
@click.version_option(package_name="eit-sim2real")
def cli() -> None:
    """EIT Sim-to-Real: touch classification with noise augmentation."""


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(experiments)
cli.add_command(validate_dataset)
cli.add_command(log_environment)
