"""Unit tests for the Python-side EIT noise model."""

import numpy as np
import pytest

from eit_sim2real.data.noise import (
    NoiseConfig,
    apply_noise,
    apply_noise_batch_vectorised,
)


@pytest.fixture
def clean_signal() -> np.ndarray:
    """Generate a synthetic clean EIT measurement batch."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 208)).astype(np.float32)


class TestNoiseConfig:
    """Tests for NoiseConfig construction and helpers."""

    def test_defaults_all_enabled(self) -> None:
        cfg = NoiseConfig()
        assert cfg.enabled
        assert cfg.gaussian_enabled
        assert cfg.contact_impedance_enabled
        assert cfg.electrode_bias_enabled
        assert cfg.quantisation_enabled

    def test_all_off(self) -> None:
        cfg = NoiseConfig.all_off()
        assert not cfg.enabled

    def test_only_gaussian(self) -> None:
        cfg = NoiseConfig.only("gaussian")
        assert cfg.gaussian_enabled
        assert not cfg.contact_impedance_enabled
        assert not cfg.electrode_bias_enabled
        assert not cfg.quantisation_enabled

    def test_without_gaussian(self) -> None:
        cfg = NoiseConfig.without("gaussian")
        assert not cfg.gaussian_enabled
        assert cfg.contact_impedance_enabled
        assert cfg.electrode_bias_enabled
        assert cfg.quantisation_enabled

    def test_only_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown component"):
            NoiseConfig.only("invalid")

    def test_component_flags(self) -> None:
        cfg = NoiseConfig.only("quantisation")
        flags = cfg.component_flags()
        assert flags == {
            "gaussian": False,
            "contact_impedance": False,
            "electrode_bias": False,
            "quantisation": True,
        }

    def test_resolved_order_uses_default_for_enabled_components(self) -> None:
        cfg = NoiseConfig()
        assert cfg.resolved_component_order() == (
            "gaussian",
            "contact_impedance",
            "electrode_bias",
            "quantisation",
        )

    def test_resolved_order_filters_disabled_components(self) -> None:
        cfg = NoiseConfig.only("electrode_bias")
        cfg.component_order = ("quantisation", "electrode_bias", "gaussian")
        assert cfg.resolved_component_order() == ("electrode_bias",)

    def test_resolved_order_appends_missing_enabled_components(self) -> None:
        cfg = NoiseConfig(
            gaussian_enabled=True,
            contact_impedance_enabled=True,
            electrode_bias_enabled=False,
            quantisation_enabled=False,
            component_order=("contact_impedance",),
        )
        assert cfg.resolved_component_order() == ("contact_impedance", "gaussian")


class TestApplyNoise:
    """Tests for noise application functions."""

    def test_disabled_returns_copy(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig.all_off()
        result = apply_noise(clean_signal, cfg)
        np.testing.assert_array_equal(result, clean_signal)
        # Ensure it's a copy, not same buffer
        assert result is not clean_signal

    def test_enabled_modifies_signal(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig()
        result = apply_noise(clean_signal, cfg, rng=np.random.default_rng(0))
        assert not np.allclose(result, clean_signal)

    def test_output_shape_preserved(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig()
        result = apply_noise(clean_signal, cfg, rng=np.random.default_rng(0))
        assert result.shape == clean_signal.shape

    def test_reproducibility(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig()
        r1 = apply_noise(clean_signal, cfg, rng=np.random.default_rng(123))
        r2 = apply_noise(clean_signal, cfg, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(r1, r2)

    def test_severity_increases_noise(self, clean_signal: np.ndarray) -> None:
        cfg_low = NoiseConfig(severity=0.5)
        cfg_high = NoiseConfig(severity=2.0)
        r_low = apply_noise(clean_signal, cfg_low, rng=np.random.default_rng(0))
        r_high = apply_noise(clean_signal, cfg_high, rng=np.random.default_rng(0))
        # Higher severity should produce larger deviations from clean
        diff_low = np.abs(r_low - clean_signal).mean()
        diff_high = np.abs(r_high - clean_signal).mean()
        assert diff_high > diff_low


class TestApplyNoiseBatchVectorised:
    """Tests for the vectorised noise application."""

    def test_output_shape(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig()
        result = apply_noise_batch_vectorised(
            clean_signal, cfg, rng=np.random.default_rng(0)
        )
        assert result.shape == clean_signal.shape

    def test_disabled_returns_copy(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig.all_off()
        result = apply_noise_batch_vectorised(clean_signal, cfg)
        np.testing.assert_array_equal(result, clean_signal)

    def test_produces_noise(self, clean_signal: np.ndarray) -> None:
        cfg = NoiseConfig()
        result = apply_noise_batch_vectorised(
            clean_signal, cfg, rng=np.random.default_rng(0)
        )
        assert not np.allclose(result, clean_signal)


# ---------------------------------------------------------------------------
# Extra coverage added during the codebase audit
# ---------------------------------------------------------------------------


class TestAssertions:
    """The 4-component noise model requires n_features % n_electrodes == 0."""

    def test_apply_noise_rejects_non_divisible(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 207)).astype(np.float32)
        cfg = NoiseConfig()
        with pytest.raises(ValueError, match="integer multiple"):
            apply_noise(X, cfg, rng=rng)

    def test_apply_noise_batch_rejects_non_divisible(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 207)).astype(np.float32)
        cfg = NoiseConfig()
        with pytest.raises(ValueError, match="integer multiple"):
            apply_noise_batch_vectorised(X, cfg, rng=rng)


class TestEquivalence:
    """Reference (per-sample) and vectorised paths produce comparable
    overall noise magnitudes (independent RNG draws — distributions match)."""

    def test_quantisation_only_matches_in_magnitude(
        self, clean_signal: np.ndarray
    ) -> None:
        cfg = NoiseConfig.only("quantisation")
        ref = apply_noise(clean_signal, cfg, rng=np.random.default_rng(7))
        vec = apply_noise_batch_vectorised(
            clean_signal, cfg, rng=np.random.default_rng(7)
        )
        assert ref.shape == vec.shape
        ref_diff = np.abs(ref - clean_signal).mean()
        vec_diff = np.abs(vec - clean_signal).mean()
        assert np.isclose(ref_diff, vec_diff, rtol=0.3, atol=1e-6)


class TestFromYamlAndConfig:
    """``from_yaml`` and ``from_config_dict`` honour every field."""

    def test_from_yaml_round_trip(self, tmp_path) -> None:
        import yaml

        path = tmp_path / "noise.yaml"
        cfg_dict = {
            "enabled": True,
            "gaussian": {
                "enabled": True,
                "snr_db": 35.0,
                "noise_floor": 5e-5,
            },
            "contact_impedance": {
                "enabled": True,
                "std_percent": 12.5,
                "n_electrodes": 16,
            },
            "electrode_bias": {"enabled": False, "max_bias": 0.05},
            "quantisation": {
                "enabled": True,
                "adc_bits": 14,
                "voltage_range": 2.0,
            },
            "severity": 1.7,
            "component_order": [
                "quantisation",
                "gaussian",
                "contact_impedance",
                "electrode_bias",
            ],
        }
        path.write_text(yaml.safe_dump(cfg_dict))

        cfg = NoiseConfig.from_yaml(path)
        assert cfg.snr_db == 35.0
        assert cfg.noise_floor == 5e-5
        assert cfg.contact_impedance_std_percent == 12.5
        assert cfg.adc_bits == 14
        assert cfg.voltage_range == 2.0
        assert not cfg.electrode_bias_enabled
        assert cfg.severity == 1.7
        assert cfg.component_order[0] == "quantisation"

    def test_from_config_dict_uses_project_schema(self) -> None:
        cfg = NoiseConfig.from_config_dict(
            {
                "enabled": True,
                "snr_db": 30.0,
                "noise_floor": 1e-3,
                "contact_impedance_std_percent": 5.0,
                "max_bias": 0.01,
                "adc_bits": 12,
                "voltage_range": 0.5,
                "n_electrodes": 16,
                "severity": 2.3,
            }
        )
        assert cfg.snr_db == 30.0
        assert cfg.severity == 2.3
        assert cfg.adc_bits == 12
