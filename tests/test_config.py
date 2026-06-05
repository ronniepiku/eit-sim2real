"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from eit_sim2real.configs import load_config


class TestLoadConfig:
    """Tests for config.yaml loading."""

    def test_loads_default(self) -> None:
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "training" in cfg
        assert "seed" in cfg

    def test_has_required_keys(self) -> None:
        cfg = load_config()
        assert cfg["data"]["path"] == "data/eit_dataset.mat"
        assert cfg["data"]["scaler"] == "robust"
        assert cfg["training"]["epochs"] > 0
        assert cfg["training"]["batch_size"] > 0
        assert cfg["training"]["learning_rate"] > 0

    def test_noise_augmentation_config(self) -> None:
        cfg = load_config()
        assert "noise_augmentation" in cfg
        assert "snr_db" in cfg["noise_augmentation"]
        assert "adc_bits" in cfg["noise_augmentation"]

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))
