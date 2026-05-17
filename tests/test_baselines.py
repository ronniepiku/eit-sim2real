"""Unit tests for baseline classifiers."""

import numpy as np
import pytest
from models.baselines import (
    create_mlp,
    create_random_forest,
    create_svm,
    get_baseline,
    train_baseline,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate simple synthetic data for testing."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 50
    n_classes = 5

    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)

    # Make data somewhat separable
    for c in range(n_classes):
        mask = y == c
        X[mask, c * 10 : (c + 1) * 10] += 2.0

    return X, y


class TestBaselineCreation:
    """Tests for baseline model creation."""

    def test_create_svm(self) -> None:
        model = create_svm()
        assert model.kernel == "rbf"
        assert model.C == 10.0

    def test_create_random_forest(self) -> None:
        model = create_random_forest()
        assert model.n_estimators == 500

    def test_create_mlp(self) -> None:
        model = create_mlp()
        assert model.hidden_layer_sizes == (128, 128)

    def test_get_baseline_valid(self) -> None:
        for name in ["svm", "random_forest", "mlp"]:
            model = get_baseline(name)
            assert model is not None

    def test_get_baseline_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown baseline"):
            get_baseline("invalid_model")


class TestBaselineTraining:
    """Tests for baseline model training and prediction."""

    def test_train_random_forest(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_data
        model = create_random_forest()
        model = train_baseline(model, X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        # Should achieve reasonable training accuracy on separable data
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.5

    def test_train_svm(self, synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_data
        model = create_svm()
        model = train_baseline(model, X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_train_mlp(self, synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_data
        model = create_mlp()
        model = train_baseline(model, X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape
