"""Baseline classifiers for EIT touch classification.

Implements SVM, Random Forest, and MLP baselines using scikit-learn.
All models follow a consistent interface for fair comparison.
"""

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def create_svm(random_state: int = 42) -> CalibratedClassifierCV:
    """Create an SVM classifier with RBF kernel.

    Hyperparameters chosen based on typical EIT classification literature.
    Uses CalibratedClassifierCV to provide predict_proba support.
    """
    svc = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        decision_function_shape="ovr",
        random_state=random_state,
    )
    return CalibratedClassifierCV(svc, ensemble=False)


def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create a Random Forest classifier.

    500 trees provides good performance without excessive compute.
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def create_mlp(random_state: int = 42) -> MLPClassifier:
    """Create a Multi-Layer Perceptron classifier.

    Two hidden layers with 128 units each, matching the CNN's FC capacity.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )


BASELINE_MODELS: dict[str, Any] = {
    "svm": create_svm,
    "random_forest": create_random_forest,
    "mlp": create_mlp,
}


def get_baseline(name: str, random_state: int = 42) -> Any:
    """Get a baseline model by name.

    Args:
        name: One of 'svm', 'random_forest', 'mlp'.
        random_state: Random seed.

    Returns:
        Unfitted sklearn classifier.

    Raises:
        ValueError: If name is not a valid baseline.
    """
    if name not in BASELINE_MODELS:
        valid = ", ".join(BASELINE_MODELS.keys())
        raise ValueError(f"Unknown baseline '{name}'. Valid options: {valid}")
    return BASELINE_MODELS[name](random_state=random_state)


def train_baseline(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Train a baseline model.

    Args:
        model: Unfitted sklearn classifier.
        X_train: Training features (n_samples, n_features).
        y_train: Training labels (n_samples,).

    Returns:
        Fitted model.
    """
    model.fit(X_train, y_train)
    return model
