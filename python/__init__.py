"""EIT Touch Classification - Python ML Pipeline.

This package implements the machine learning pipeline for classifying
touch types from simulated EIT voltage measurements.
"""

__version__ = "0.1.0"

CLASS_NAMES = {
    1: "No contact",
    2: "Light touch",
    3: "Firm press",
    4: "Point contact",
    5: "Distributed contact",
}
NUM_CLASSES = 5
