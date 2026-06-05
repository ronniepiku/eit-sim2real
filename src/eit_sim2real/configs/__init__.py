"""Configuration loading utilities.

Provides a single function to load the project YAML config with all
hyperparameters centralised in ``config.yaml``.
"""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Path to config file.  Falls back to the default
              ``src/eit_sim2real/configs/config.yaml`` shipped with the package.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    return cfg


__all__ = ["load_config"]
