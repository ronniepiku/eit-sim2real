"""Project-wide constants — single source of truth.

All modules should import constants from here rather than defining locally.
"""

from typing import Final

NUM_CLASSES: int = 5

CLASS_NAMES: list[str] = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed contact",
]

NOISE_COMPONENTS: list[str] = [
    "gaussian",
    "contact_impedance",
    "electrode_bias",
    "quantisation",
]

COMPONENT_LABELS: dict[str, str] = {
    "gaussian": "Gaussian",
    "contact_impedance": "Contact Impedance",
    "electrode_bias": "Electrode Bias",
    "quantisation": "Quantisation",
}

# ── Headline training hyperparameters ────────────────────────────────
# These are the dissertation's reported regularisation settings for the
# noisy/augmented/mixed conditions. Centralised here so that the CLI
# (`eit train cnn`), the experiment grid, and the extended/additional
# experiments all share a single source of truth.
NOISY_CNN_PARAMS: Final[dict[str, float]] = {
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "label_smoothing": 0.05,
}

MIXED_CNN_PARAMS: Final[dict[str, float]] = {
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "label_smoothing": 0.05,
    "clean_ratio": 0.3,
}

# Severity multiplier range used by online noise augmentation
# (sampled per batch).
DEFAULT_SEVERITY_RANGE: Final[tuple[float, float]] = (0.5, 2.0)

# Default headline seed sequence — five seeds, [42..46]. This is the
# protocol reported in the dissertation results tables.
DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 43, 44, 45, 46)
