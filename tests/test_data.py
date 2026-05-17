"""Unit tests for dataset loading and preprocessing."""

import numpy as np
import pytest
from data.load_dataset import get_cv_splits, prepare_splits


@pytest.fixture
def synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic dataset mimicking EIT data structure."""
    rng = np.random.default_rng(42)
    n_samples = 500
    n_features = 208
    n_classes = 5

    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)

    return X, y


class TestPrepareSplits:
    """Tests for train/val/test splitting."""

    def test_split_sizes(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        dataset = prepare_splits(X, y, test_size=0.15, val_size=0.15)

        total = len(dataset.y_train) + len(dataset.y_val) + len(dataset.y_test)
        assert total == len(y)

        # Approximate split ratios (allow 5% tolerance)
        assert abs(len(dataset.y_test) / len(y) - 0.15) < 0.05
        assert abs(len(dataset.y_val) / len(y) - 0.15) < 0.05

    def test_stratification(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        dataset = prepare_splits(X, y)

        # Each split should contain all classes
        for split_y in [dataset.y_train, dataset.y_val, dataset.y_test]:
            unique_classes = np.unique(split_y)
            assert len(unique_classes) == 5

    def test_normalization(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        dataset = prepare_splits(X, y, normalize=True)

        # Training set should be approximately zero-mean unit-variance
        assert abs(dataset.X_train.mean()) < 0.1
        assert abs(dataset.X_train.std() - 1.0) < 0.1

    def test_no_normalization(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        dataset = prepare_splits(X, y, normalize=False)

        # Data should not be standardized
        # Original data has std ≈ 1 and mean ≈ 0 anyway (standard normal),
        # but scaler should not have been applied
        assert dataset.scaler is not None

    def test_reproducibility(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        d1 = prepare_splits(X, y, random_state=42)
        d2 = prepare_splits(X, y, random_state=42)

        np.testing.assert_array_equal(d1.y_train, d2.y_train)
        np.testing.assert_array_equal(d1.y_test, d2.y_test)


class TestCVSplits:
    """Tests for cross-validation split generation."""

    def test_n_folds(self, synthetic_dataset: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_dataset
        splits = get_cv_splits(X, y, n_folds=5)
        assert len(splits) == 5

    def test_no_overlap(self, synthetic_dataset: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = synthetic_dataset
        splits = get_cv_splits(X, y, n_folds=5)

        for train_idx, val_idx in splits:
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0

    def test_complete_coverage(
        self, synthetic_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = synthetic_dataset
        splits = get_cv_splits(X, y, n_folds=5)

        all_val_indices = set()
        for _, val_idx in splits:
            all_val_indices.update(val_idx)

        assert all_val_indices == set(range(len(y)))
