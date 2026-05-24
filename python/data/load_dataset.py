"""Dataset loading and preprocessing utilities for EIT touch classification."""

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


class EITDataset(NamedTuple):
    """Container for train/validation/test splits."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_mat_dataset(
    data_path: Path | str,
    use_noisy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load EIT dataset from MATLAB .mat file.

    Supports both original dataset format and cleaned dataset format:
    - Original: dataset_X_clean/dataset_X_noisy, dataset_y, dataset_X (fallback)
    - Cleaned: X_clean/X_noisy, y (from EDA notebook)

    Args:
        data_path: Path to the .mat file.
        use_noisy: If True, load noisy measurements; else load clean.

    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    try:
        mat = sio.loadmat(str(data_path))
    except NotImplementedError:
        with h5py.File(data_path, "r") as f:
            key = "dataset_X_noisy" if use_noisy else "dataset_X_clean"
            X = np.array(f[key], dtype=np.float32)
            y = np.array(f["dataset_y"], dtype=np.int64).ravel()
    else:
        # Try original dataset format first (dataset_X_noisy/dataset_X_clean)
        key = "dataset_X_noisy" if use_noisy else "dataset_X_clean"
        if key not in mat:
            # Fall back to generic dataset_X
            key = "dataset_X"
        if key not in mat:
            # Fall back to cleaned dataset format (X_noisy/X_clean)
            key = "X_noisy" if use_noisy else "X_clean"

        if key not in mat:
            available_keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                f"Could not find dataset keys in {data_path}. "
                f"Tried: {['dataset_X_noisy', 'dataset_X_clean', 'dataset_X', 'X_noisy', 'X_clean']} "
                f"Found: {available_keys}"
            )

        X = np.array(mat[key], dtype=np.float32)

        # Try original label format first (dataset_y)
        y_key = "dataset_y"
        if y_key not in mat:
            # Fall back to cleaned dataset format (y)
            y_key = "y"

        if y_key not in mat:
            raise KeyError(
                f"Could not find labels in {data_path}. Tried: ['dataset_y', 'y']"
            )

        y = np.array(mat[y_key], dtype=np.int64).ravel()

    # Check if labels are already 0-indexed (cleaned dataset format)
    # Original format is 1-indexed, cleaned is already 0-indexed
    if y.min() >= 1:
        # Convert from 1-indexed (MATLAB) to 0-indexed (PyTorch) labels
        y = y - 1

    # Ensure X rows match y length
    if X.shape[0] != y.shape[0]:
        if X.shape[1] == y.shape[0]:
            X = X.T
        else:
            raise ValueError(
                f"Unexpected dataset shape: X={X.shape}, y={y.shape}. "
                "Expected n_samples rows in X."
            )

    if y.min() < 0 or y.max() >= 5:
        raise ValueError(
            f"Labels out of range: min={y.min()}, max={y.max()}. "
            "Expected 0-indexed class labels (0-4 for 5 classes)."
        )

    return X, y


def prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    normalize: bool = True,
) -> EITDataset:
    """Split data into train/val/test and optionally normalize.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Label vector (n_samples,).
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        random_state: Random seed for reproducibility.
        normalize: Whether to apply StandardScaler.

    Returns:
        EITDataset with all splits and fitted scaler.
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_trainval,
    )

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    if normalize:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler.fit(X_train)

    return EITDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
    )


def get_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified k-fold cross-validation indices.

    Args:
        X: Feature matrix.
        y: Label vector.
        n_folds: Number of folds.
        random_state: Random seed.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [(train_idx, val_idx) for train_idx, val_idx in skf.split(X, y)]
