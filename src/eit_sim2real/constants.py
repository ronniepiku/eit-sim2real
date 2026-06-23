"""Project-wide constants"""

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
