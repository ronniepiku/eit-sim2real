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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NoiseConfig":
        """Load noise configuration from YAML file."""
        with open(path) as f:
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


def apply_noise(
    X: np.ndarray,
    config: NoiseConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the 4-component noise model to a batch of EIT measurements.

    Matches the MATLAB add_noise.m implementation with identical physics:
    - Fixed application order: Gaussian → Contact Impedance → Bias → Quantisation
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

    # --- 1. Gaussian measurement noise (SNR-based) ---
    if config.gaussian_enabled:
        # Severity scales the effective noise: lower effective SNR
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
                    signal_power / np.linalg.norm(noise) * 10 ** (-effective_snr / 20)
                )
            else:
                # Noise floor for zero-signal (no-contact class)
                scale = config.noise_floor * n_meas / np.linalg.norm(noise)
            X_noisy[i] += scale * noise

    # --- 2. Electrode contact impedance variation (per-electrode, multiplicative) ---
    if config.contact_impedance_enabled:
        n_elec = config.n_electrodes
        meas_per_elec = n_meas / n_elec
        effective_std = (config.contact_impedance_std_percent / 100.0) * severity

        for i in range(n_samples):
            # Log-normal per-electrode factors
            elec_factors = np.exp(effective_std * rng.standard_normal(n_elec))
            # Map to measurements (block structure)
            impedance_vec = np.repeat(elec_factors, int(np.round(meas_per_elec)))
            # Trim/pad to match n_meas
            if len(impedance_vec) > n_meas:
                impedance_vec = impedance_vec[:n_meas]
            elif len(impedance_vec) < n_meas:
                impedance_vec = np.pad(
                    impedance_vec, (0, n_meas - len(impedance_vec)), constant_values=1.0
                )
            X_noisy[i] *= impedance_vec

    # --- 3. Electrode positioning bias (per-electrode, additive) ---
    if config.electrode_bias_enabled:
        n_elec = config.n_electrodes
        meas_per_elec = n_meas / n_elec
        effective_bias = config.max_bias * severity

        for i in range(n_samples):
            elec_bias = effective_bias * (2 * rng.random(n_elec) - 1)
            bias_vec = np.repeat(elec_bias, int(np.round(meas_per_elec)))
            if len(bias_vec) > n_meas:
                bias_vec = bias_vec[:n_meas]
            elif len(bias_vec) < n_meas:
                bias_vec = np.pad(
                    bias_vec, (0, n_meas - len(bias_vec)), constant_values=0.0
                )
            X_noisy[i] += bias_vec

    # --- 4. Quantisation noise (ADC) ---
    if config.quantisation_enabled:
        lsb = config.voltage_range / (2**config.adc_bits)
        effective_lsb = lsb * severity
        quant_noise = (rng.random((n_samples, n_meas)) - 0.5) * effective_lsb
        X_noisy += quant_noise

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
    meas_per_elec = int(np.round(n_meas / n_elec))

    # --- 1. Gaussian noise ---
    if config.gaussian_enabled:
        effective_snr = (
            config.snr_db + 20 * np.log10(1.0 / severity)
            if severity > 0
            else config.snr_db
        )
        noise = rng.standard_normal((n_samples, n_meas))
        signal_norms = np.linalg.norm(X_noisy, axis=1, keepdims=True)
        noise_norms = np.linalg.norm(noise, axis=1, keepdims=True)

        # Signal-dependent scaling
        scale = np.where(
            signal_norms > 0,
            signal_norms / noise_norms * 10 ** (-effective_snr / 20),
            config.noise_floor * n_meas / noise_norms,
        )
        X_noisy += scale * noise

    # --- 2. Contact impedance (per-electrode, multiplicative) ---
    if config.contact_impedance_enabled:
        effective_std = (config.contact_impedance_std_percent / 100.0) * severity
        elec_factors = np.exp(effective_std * rng.standard_normal((n_samples, n_elec)))
        # Expand to measurement vector via block repetition
        impedance_matrix = np.repeat(elec_factors, meas_per_elec, axis=1)[:, :n_meas]
        # Pad if needed
        if impedance_matrix.shape[1] < n_meas:
            pad_width = n_meas - impedance_matrix.shape[1]
            impedance_matrix = np.pad(
                impedance_matrix, ((0, 0), (0, pad_width)), constant_values=1.0
            )
        X_noisy *= impedance_matrix

    # --- 3. Electrode bias (per-electrode, additive) ---
    if config.electrode_bias_enabled:
        effective_bias = config.max_bias * severity
        elec_bias = effective_bias * (2 * rng.random((n_samples, n_elec)) - 1)
        bias_matrix = np.repeat(elec_bias, meas_per_elec, axis=1)[:, :n_meas]
        if bias_matrix.shape[1] < n_meas:
            pad_width = n_meas - bias_matrix.shape[1]
            bias_matrix = np.pad(
                bias_matrix, ((0, 0), (0, pad_width)), constant_values=0.0
            )
        X_noisy += bias_matrix

    # --- 4. Quantisation noise ---
    if config.quantisation_enabled:
        lsb = config.voltage_range / (2**config.adc_bits)
        effective_lsb = lsb * severity
        X_noisy += (rng.random((n_samples, n_meas)) - 0.5) * effective_lsb

    return X_noisy.astype(np.float32)
