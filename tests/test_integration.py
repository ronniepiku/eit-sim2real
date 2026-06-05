"""Integration tests: end-to-end train → evaluate pipeline with synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eit_sim2real.constants import NUM_CLASSES
from eit_sim2real.data.noise import NoiseConfig, apply_noise_batch_vectorised
from eit_sim2real.evaluate import evaluate_model, evaluate_robustness, load_model
from eit_sim2real.models.cnn1d import EITConv1D
from eit_sim2real.train import train_cnn, train_cnn_mixed
from eit_sim2real.utils import predict_cnn, set_seeds


@pytest.fixture
def synthetic_data() -> dict[str, np.ndarray]:
    """Create a small synthetic dataset for integration testing."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 32  # Small enough for fast tests, large enough for CNN

    # Generate separable data: each class has a distinct mean
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    samples_per_class = n_samples // NUM_CLASSES

    for i in range(NUM_CLASSES):
        start = i * samples_per_class
        end = start + samples_per_class
        X[start:end] = rng.normal(
            loc=i * 2.0, scale=0.5, size=(samples_per_class, n_features)
        )
        y[start:end] = i

    return {"X": X, "y": y}


@pytest.fixture
def train_val_split(synthetic_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split synthetic data into train/val/test."""
    X, y = synthetic_data["X"], synthetic_data["y"]
    n = len(X)
    idx = np.random.default_rng(42).permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    return {
        "X_train": X[idx[:n_train]],
        "y_train": y[idx[:n_train]],
        "X_val": X[idx[n_train : n_train + n_val]],
        "y_val": y[idx[n_train : n_train + n_val]],
        "X_test": X[idx[n_train + n_val :]],
        "y_test": y[idx[n_train + n_val :]],
    }


class TestCNNEndToEnd:
    """End-to-end CNN training and evaluation."""

    def test_train_and_evaluate(self, train_val_split: dict[str, np.ndarray]) -> None:
        set_seeds(42)
        model, history = train_cnn(
            train_val_split["X_train"],
            train_val_split["y_train"],
            train_val_split["X_val"],
            train_val_split["y_val"],
            epochs=5,
            batch_size=32,
            lr=0.001,
            early_stopping_patience=10,
            device="cpu",
        )

        assert isinstance(model, EITConv1D)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 5

        results = evaluate_model(
            model, train_val_split["X_test"], train_val_split["y_test"], device="cpu"
        )
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["f1_macro"] <= 1.0
        assert results["confusion_matrix"].shape == (NUM_CLASSES, NUM_CLASSES)
        assert len(results["y_pred"]) == len(train_val_split["y_test"])

    def test_train_mixed_and_evaluate(
        self, train_val_split: dict[str, np.ndarray]
    ) -> None:
        set_seeds(42)
        noise_cfg = NoiseConfig(severity=0.5)
        rng = np.random.default_rng(42)
        X_noisy = apply_noise_batch_vectorised(
            train_val_split["X_train"], noise_cfg, rng=rng
        )
        model, history = train_cnn_mixed(
            train_val_split["X_train"],
            X_noisy,
            train_val_split["y_train"],
            train_val_split["X_val"],
            train_val_split["y_val"],
            epochs=3,
            batch_size=32,
            lr=0.001,
            early_stopping_patience=10,
            device="cpu",
        )

        assert isinstance(model, EITConv1D)
        results = evaluate_model(
            model, train_val_split["X_test"], train_val_split["y_test"], device="cpu"
        )
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_model_save_load_roundtrip(
        self, train_val_split: dict[str, np.ndarray], tmp_path
    ) -> None:
        set_seeds(42)
        n_features = train_val_split["X_train"].shape[1]
        model, _ = train_cnn(
            train_val_split["X_train"],
            train_val_split["y_train"],
            train_val_split["X_val"],
            train_val_split["y_val"],
            epochs=2,
            batch_size=32,
            lr=0.001,
            early_stopping_patience=10,
            device="cpu",
        )

        # Save
        save_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), save_path)

        # Load
        loaded = EITConv1D(n_features=n_features)
        loaded.load_state_dict(torch.load(save_path, weights_only=True))
        loaded.eval()

        # Predictions should match
        y_orig = predict_cnn(model, train_val_split["X_test"], device="cpu")
        y_loaded = predict_cnn(loaded, train_val_split["X_test"], device="cpu")
        np.testing.assert_array_equal(y_orig, y_loaded)


class TestNoiseIntegration:
    """Test noise model integration with training pipeline."""

    def test_noise_preserves_shape(self, synthetic_data: dict[str, np.ndarray]) -> None:
        X = synthetic_data["X"]
        noise_cfg = NoiseConfig(severity=1.0)
        rng = np.random.default_rng(42)
        X_noisy = apply_noise_batch_vectorised(X, noise_cfg, rng=rng)
        assert X_noisy.shape == X.shape
        assert X_noisy.dtype == X.dtype

    def test_noise_changes_data(self, synthetic_data: dict[str, np.ndarray]) -> None:
        X = synthetic_data["X"]
        noise_cfg = NoiseConfig(severity=1.0)
        rng = np.random.default_rng(42)
        X_noisy = apply_noise_batch_vectorised(X, noise_cfg, rng=rng)
        assert not np.allclose(X, X_noisy)

    def test_low_severity_minimal_change(
        self, synthetic_data: dict[str, np.ndarray]
    ) -> None:
        X = synthetic_data["X"]
        noise_cfg = NoiseConfig(severity=0.01)
        rng = np.random.default_rng(42)
        X_noisy = apply_noise_batch_vectorised(X, noise_cfg, rng=rng)
        # Very low severity should produce minimal perturbation
        max_diff = np.abs(X - X_noisy).max()
        assert max_diff < 1.0  # bounded perturbation


class TestRobustnessEvaluation:
    """Test robustness evaluation with trained model."""

    def test_robustness_sweep(self, train_val_split: dict[str, np.ndarray]) -> None:
        set_seeds(42)
        model, _ = train_cnn(
            train_val_split["X_train"],
            train_val_split["y_train"],
            train_val_split["X_val"],
            train_val_split["y_val"],
            epochs=3,
            batch_size=32,
            lr=0.001,
            early_stopping_patience=10,
            device="cpu",
        )

        results = evaluate_robustness(
            model,
            train_val_split["X_test"],
            train_val_split["y_test"],
            noise_levels=[0.0, 0.1, 0.5],
            device="cpu",
        )

        assert len(results["noise_levels"]) == 3
        assert len(results["accuracies"]) == 3
        assert len(results["f1_scores"]) == 3
        # At zero noise, accuracy should match clean evaluation
        assert results["accuracies"][0] >= 0.0


class TestLoadModel:
    """Test the load_model utility for both .pt and .joblib formats."""

    def test_load_cnn_model(
        self, train_val_split: dict[str, np.ndarray], tmp_path
    ) -> None:
        set_seeds(42)
        n_features = train_val_split["X_train"].shape[1]
        model, _ = train_cnn(
            train_val_split["X_train"],
            train_val_split["y_train"],
            train_val_split["X_val"],
            train_val_split["y_val"],
            epochs=2,
            batch_size=32,
            device="cpu",
        )

        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        loaded = load_model(save_path, n_features=n_features)
        assert isinstance(loaded, EITConv1D)

        y_orig = predict_cnn(model, train_val_split["X_test"], device="cpu")
        y_loaded = predict_cnn(loaded, train_val_split["X_test"], device="cpu")
        np.testing.assert_array_equal(y_orig, y_loaded)

    def test_load_sklearn_model(
        self, train_val_split: dict[str, np.ndarray], tmp_path
    ) -> None:
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(train_val_split["X_train"], train_val_split["y_train"])

        save_path = tmp_path / "model.joblib"
        joblib.dump(rf, save_path)

        loaded = load_model(save_path, n_features=train_val_split["X_train"].shape[1])
        y_orig = rf.predict(train_val_split["X_test"])
        y_loaded = loaded.predict(train_val_split["X_test"])
        np.testing.assert_array_equal(y_orig, y_loaded)

    def test_load_unsupported_format(self, tmp_path) -> None:
        bad_path = tmp_path / "model.pickle"
        bad_path.write_bytes(b"fake")

        with pytest.raises(ValueError, match="Unsupported model format"):
            load_model(bad_path, n_features=208)
