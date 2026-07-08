"""Python-side EIT noise model matching the MATLAB add_noise.m implementation.

Enables on-the-fly noise augmentation during training without requiring
separate MATLAB-generated datasets for each ablation configuration.
Supports per-component toggling for ablation studies and severity scaling
for robustness evaluation.

The noise model has 4 independently-configurable components:
1. Gaussian measurement noise (SNR-based, signal-dependent)
2. Electrode contact impedance variation (per-electrode, multiplicative)
3. Electrode positioning bias (per-electrode, additive)
4. Quantisation noise (ADC resolution)

References:
    [1] Adler & Lionheart (2006) - SNR 40-80 dB for lab EIT
    [2] Vilhunen et al. (2002) - 5-20% contact impedance variation
    [3] Kolehmainen et al. (1997) - 1-2 mm electrode positioning errors
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_COMPONENT_ORDER = (
    "gaussian",
    "contact_impedance",
    "electrode_bias",
    "quantisation",
)


@dataclass
class NoiseConfig:
    """Configuration for the 4-component EIT noise model.

    Mirrors the structure of matlab/configs/noise_params.yaml.
    Each component can be independently enabled/disabled for ablation.
    """

    # Master toggle
    enabled: bool = True

    # Gaussian measurement noise
    gaussian_enabled: bool = True
    snr_db: float = 40.0
    noise_floor: float = 1e-4

    # Electrode contact impedance variation
    contact_impedance_enabled: bool = True
    contact_impedance_std_percent: float = 10.0
    n_electrodes: int = 16

    # Electrode positioning bias
    electrode_bias_enabled: bool = True
    max_bias: float = 0.02

    # Quantisation noise
    quantisation_enabled: bool = True
    adc_bits: int = 16
    voltage_range: float = 1.0

    # Severity multiplier (scales all noise magnitudes)
    severity: float = 1.0

    # Component application order.
    # Any enabled component omitted from this tuple is appended using the
    # default order in DEFAULT_COMPONENT_ORDER.
    component_order: tuple[str, ...] = DEFAULT_COMPONENT_ORDER

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NoiseConfig":
        """Load noise configuration from YAML file."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            enabled=raw.get("enabled", True),
            gaussian_enabled=raw.get("gaussian", {}).get("enabled", True),
            snr_db=raw.get("gaussian", {}).get("snr_db", 40.0),
            noise_floor=raw.get("gaussian", {}).get("noise_floor", 1e-4),
            contact_impedance_enabled=raw.get("contact_impedance", {}).get(
                "enabled", True
            ),
            contact_impedance_std_percent=raw.get("contact_impedance", {}).get(
                "std_percent", 10.0
            ),
            n_electrodes=raw.get("contact_impedance", {}).get("n_electrodes", 16),
            electrode_bias_enabled=raw.get("electrode_bias", {}).get("enabled", True),
            max_bias=raw.get("electrode_bias", {}).get("max_bias", 0.02),
            quantisation_enabled=raw.get("quantisation", {}).get("enabled", True),
            adc_bits=raw.get("quantisation", {}).get("adc_bits", 16),
            voltage_range=raw.get("quantisation", {}).get("voltage_range", 1.0),
            severity=raw.get("severity", 1.0),
            component_order=tuple(raw.get("component_order", DEFAULT_COMPONENT_ORDER)),
        )

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "NoiseConfig":
        """Build NoiseConfig from the project's flattened config schema."""
        return cls(
            enabled=cfg.get("enabled", True),
            gaussian_enabled=cfg.get("gaussian_enabled", True),
            snr_db=cfg.get("snr_db", 40.0),
            noise_floor=cfg.get("noise_floor", 1e-4),
            contact_impedance_enabled=cfg.get("contact_impedance_enabled", True),
            contact_impedance_std_percent=cfg.get(
                "contact_impedance_std_percent", 10.0
            ),
            n_electrodes=cfg.get("n_electrodes", 16),
            electrode_bias_enabled=cfg.get("electrode_bias_enabled", True),
            max_bias=cfg.get("max_bias", 0.02),
            quantisation_enabled=cfg.get("quantisation_enabled", True),
            adc_bits=cfg.get("adc_bits", 16),
            voltage_range=cfg.get("voltage_range", 1.0),
            severity=cfg.get("severity", 1.0),
            component_order=tuple(cfg.get("component_order", DEFAULT_COMPONENT_ORDER)),
        )

    @classmethod
    def all_off(cls) -> "NoiseConfig":
        """Return a config with all noise components disabled."""
        return cls(enabled=False)

    @classmethod
    def only(cls, component: str) -> "NoiseConfig":
        """Return a config with only the specified component enabled.

        Args:
            component: One of 'gaussian', 'contact_impedance',
                      'electrode_bias', 'quantisation'.
        """
        cfg = cls(
            enabled=True,
            gaussian_enabled=False,
            contact_impedance_enabled=False,
            electrode_bias_enabled=False,
            quantisation_enabled=False,
        )
        if component == "gaussian":
            cfg.gaussian_enabled = True
        elif component == "contact_impedance":
            cfg.contact_impedance_enabled = True
        elif component == "electrode_bias":
            cfg.electrode_bias_enabled = True
        elif component == "quantisation":
            cfg.quantisation_enabled = True
        else:
            raise ValueError(f"Unknown component: {component}")
        return cfg

    @classmethod
    def without(cls, component: str) -> "NoiseConfig":
        """Return a config with all components EXCEPT the specified one.

        Args:
            component: One of 'gaussian', 'contact_impedance',
                      'electrode_bias', 'quantisation'.
        """
        cfg = cls(enabled=True)
        if component == "gaussian":
            cfg.gaussian_enabled = False
        elif component == "contact_impedance":
            cfg.contact_impedance_enabled = False
        elif component == "electrode_bias":
            cfg.electrode_bias_enabled = False
        elif component == "quantisation":
            cfg.quantisation_enabled = False
        else:
            raise ValueError(f"Unknown component: {component}")
        return cfg

    def component_flags(self) -> dict[str, bool]:
        """Return a dict of component name → enabled status."""
        return {
            "gaussian": self.gaussian_enabled,
            "contact_impedance": self.contact_impedance_enabled,
            "electrode_bias": self.electrode_bias_enabled,
            "quantisation": self.quantisation_enabled,
        }

    def active_components(self) -> list[str]:
        """Return enabled components in canonical default order."""
        flags = self.component_flags()
        return [comp for comp in DEFAULT_COMPONENT_ORDER if flags[comp]]

    def resolved_component_order(self) -> tuple[str, ...]:
        """Resolve the effective component order for enabled components."""
        active = self.active_components()
        if not active:
            return ()

        # Keep only enabled components from requested order, then append any
        # missing enabled components in canonical order for stability.
        requested = [comp for comp in self.component_order if comp in active]
        missing = [comp for comp in active if comp not in requested]
        return tuple(requested + missing)


def apply_noise(
    X: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the 4-component noise model to a batch of EIT measurements.

    This is the reference implementation with a per-sample loop for clarity.
    For training-time augmentation where speed matters, use
    :func:`apply_noise_batch_vectorised` instead.

    Matches the MATLAB add_noise.m implementation by default (same component
    defaults and same default order). The order can be overridden via
    ``NoiseConfig.component_order``.
    - Per-electrode structure for impedance and bias
    - Signal-dependent Gaussian noise scaling

    Args:
        X: Input measurements, shape (n_samples, n_features).
           These should be CLEAN (pre-normalisation) voltage difference vectors.
        config: Noise configuration specifying which components are active.
        rng: NumPy random generator for reproducibility. If None, creates one.

    Returns:
        Noisy measurements, same shape as X.
    """
    if not config.enabled:
        return X.copy()

    if rng is None:
        rng = np.random.default_rng()

    X_noisy = X.copy().astype(np.float64)
    n_samples, n_meas = X_noisy.shape
    severity = config.severity

    n_elec = config.n_electrodes
    if n_meas % n_elec != 0:
        raise ValueError(
            f"n_features ({n_meas}) must be an integer multiple of "
            f"n_electrodes ({n_elec})"
        )
    meas_per_elec = n_meas / n_elec

    for component in config.resolved_component_order():
        if component == "gaussian":
            effective_snr = (
                config.snr_db + 20 * np.log10(1.0 / severity)
                if severity > 0
                else config.snr_db
            )
            for i in range(n_samples):
                noise = rng.standard_normal(n_meas)
                signal_power = np.linalg.norm(X_noisy[i])
                if signal_power > 0:
                    scale = (
                        signal_power
                        / np.linalg.norm(noise)
                        * 10 ** (-effective_snr / 20)
                    )
                else:
                    # Noise floor for zero-signal (no-contact class)
                    scale = config.noise_floor * n_meas / np.linalg.norm(noise)
                X_noisy[i] += scale * noise
        elif component == "contact_impedance":
            effective_std = (config.contact_impedance_std_percent / 100.0) * severity
            for i in range(n_samples):
                elec_factors = np.exp(effective_std * rng.standard_normal(n_elec))
                impedance_vec = np.repeat(elec_factors, int(np.round(meas_per_elec)))
                if len(impedance_vec) > n_meas:
                    impedance_vec = impedance_vec[:n_meas]
                elif len(impedance_vec) < n_meas:
                    impedance_vec = np.pad(
                        impedance_vec,
                        (0, n_meas - len(impedance_vec)),
                        constant_values=1.0,
                    )
                X_noisy[i] *= impedance_vec
        elif component == "electrode_bias":
            effective_bias = config.max_bias * severity
            for i in range(n_samples):
                elec_bias = effective_bias * (2 * rng.random(n_elec) - 1)
                bias_vec = np.repeat(elec_bias, int(np.round(meas_per_elec)))
                if len(bias_vec) > n_meas:
                    bias_vec = bias_vec[:n_meas]
                elif len(bias_vec) < n_meas:
                    bias_vec = np.pad(
                        bias_vec,
                        (0, n_meas - len(bias_vec)),
                        constant_values=0.0,
                    )
                X_noisy[i] += bias_vec
        elif component == "quantisation":
            lsb = config.voltage_range / (2**config.adc_bits)
            effective_lsb = lsb * severity
            quant_noise = (rng.random((n_samples, n_meas)) - 0.5) * effective_lsb
            X_noisy += quant_noise
        else:
            raise ValueError(f"Unknown noise component in order: {component}")

    return X_noisy.astype(np.float32)


def apply_noise_batch_vectorised(
    X: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vectorised batch noise application (faster for large batches).

    Same physics as apply_noise but uses batch operations where possible.
    Use this for training-time augmentation where speed matters.

    Args:
        X: Input measurements, shape (n_samples, n_features).
        config: Noise configuration.
        rng: NumPy random generator.

    Returns:
        Noisy measurements, same shape as X.
    """
    if not config.enabled:
        return X.copy()

    if rng is None:
        rng = np.random.default_rng()

    X_noisy = X.copy().astype(np.float64)
    n_samples, n_meas = X_noisy.shape
    severity = config.severity
    n_elec = config.n_electrodes
    if n_meas % n_elec != 0:
        raise ValueError(
            f"n_features ({n_meas}) must be an integer multiple of "
            f"n_electrodes ({n_elec})"
        )
    meas_per_elec = int(np.round(n_meas / n_elec))

    for component in config.resolved_component_order():
        if component == "gaussian":
            effective_snr = (
                config.snr_db + 20 * np.log10(1.0 / severity)
                if severity > 0
                else config.snr_db
            )
            noise = rng.standard_normal((n_samples, n_meas))
            signal_norms = np.linalg.norm(X_noisy, axis=1, keepdims=True)
            noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)
            scale = np.where(
                signal_norms > 0,
                signal_norms / noise_norms * 10 ** (-effective_snr / 20),
                config.noise_floor * n_meas / noise_norms,
            )
            X_noisy += scale * noise
        elif component == "contact_impedance":
            effective_std = (config.contact_impedance_std_percent / 100.0) * severity
            elec_factors = np.exp(
                effective_std * rng.standard_normal((n_samples, n_elec))
            )
            impedance_matrix = np.repeat(elec_factors, meas_per_elec, axis=1)[
                :, :n_meas
            ]
            if impedance_matrix.shape[1] < n_meas:
                pad_width = n_meas - impedance_matrix.shape[1]
                impedance_matrix = np.pad(
                    impedance_matrix, ((0, 0), (0, pad_width)), constant_values=1.0
                )
            X_noisy *= impedance_matrix
        elif component == "electrode_bias":
            effective_bias = config.max_bias * severity
            elec_bias = effective_bias * (2 * rng.random((n_samples, n_elec)) - 1)
            bias_matrix = np.repeat(elec_bias, meas_per_elec, axis=1)[:, :n_meas]
            if bias_matrix.shape[1] < n_meas:
                pad_width = n_meas - bias_matrix.shape[1]
                bias_matrix = np.pad(
                    bias_matrix, ((0, 0), (0, pad_width)), constant_values=0.0
                )
            X_noisy += bias_matrix
        elif component == "quantisation":
            lsb = config.voltage_range / (2**config.adc_bits)
            effective_lsb = lsb * severity
            X_noisy += (rng.random((n_samples, n_meas)) - 0.5) * effective_lsb
        else:
            raise ValueError(f"Unknown noise component in order: {component}")

    return X_noisy.astype(np.float32)


def apply_noise_in_scaled_space(
    X_scaled: np.ndarray,
    scaler: object,
    config: NoiseConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply physics noise in raw measurement space, then re-scale.

    This helper is for pipelines that keep model inputs in a scaled feature
    space but want the noise model to operate on the underlying clean voltage
    vectors, consistent with the MATLAB generation path and methodology.

    Args:
        X_scaled: Input features already transformed by ``scaler``.
        scaler: Fitted scaler with ``inverse_transform`` and ``transform``.
        config: Noise configuration.
        rng: NumPy random generator for reproducibility.

    Returns:
        Noisy features transformed back into the same scaled feature space.
    """
    X_raw = scaler.inverse_transform(X_scaled)  # type: ignore[attr-defined]
    X_noisy_raw = apply_noise_batch_vectorised(X_raw, config, rng=rng)
    return scaler.transform(X_noisy_raw)  # type: ignore[attr-defined]
