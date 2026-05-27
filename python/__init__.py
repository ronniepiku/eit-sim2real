"""EIT Touch Classification - Python ML Pipeline.

This package implements the machine learning pipeline for classifying
touch types from simulated EIT voltage measurements.
"""

__version__ = "0.1.0"

NUM_CLASSES = 5

CLASS_NAMES: list[str] = [
    "No contact",
    "Light touch",
    "Firm press",
    "Point contact",
    "Distributed contact",
]
