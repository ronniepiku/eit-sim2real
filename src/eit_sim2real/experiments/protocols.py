"""Shared experiment protocol definitions.

This module centralizes training-hyperparameter presets that must stay
identical across pipelines when results are compared directly.
"""

NOISY_CNN_PARAMS = {
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "label_smoothing": 0.05,
}

MIXED_CNN_PARAMS = {
    "weight_decay": 1e-3,
    "dropout": 0.4,
    "label_smoothing": 0.05,
    "clean_ratio": 0.3,
}
