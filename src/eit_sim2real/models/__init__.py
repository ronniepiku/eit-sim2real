"""ML model architectures for EIT touch classification."""

from eit_sim2real.models.baselines import get_baseline, train_baseline
from eit_sim2real.models.cnn1d import EITConv1D

__all__ = ["EITConv1D", "get_baseline", "train_baseline"]
