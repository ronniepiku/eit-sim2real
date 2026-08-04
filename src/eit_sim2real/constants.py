"""Project-wide constants"""

from pathlib import Path

# Repository root, resolved from this file's location
# (src/eit_sim2real/constants.py -> parents[2] == repo root).
# Used so scripts resolve data/ and results/ correctly regardless of the
# current working directory. Assumes an editable install, which is how the
# project is installed (see docs/SETUP.md).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

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
