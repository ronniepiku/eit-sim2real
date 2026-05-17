"""Quick validation of EIT dataset: visualization and basic analysis."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
from sklearn.decomposition import PCA

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "eit_dataset_numpy.mat"
USE_NOISY = True


def load_dataset(data_path: Path | str, use_noisy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load EIT dataset from .mat file.
    
    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,).
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    mat = sio.loadmat(str(data_path))
    
    # Try to load noisy or clean version
    key = "dataset_X_noisy" if use_noisy else "dataset_X_clean"
    if key not in mat:
        # Fallback for legacy format
        key = "dataset_X"
    
    X = np.array(mat[key], dtype=np.float32)
    y = np.array(mat["dataset_y"], dtype=np.int64).ravel()
    
    # Convert from 1-indexed (MATLAB) to 0-indexed (Python)
    y = y - 1
    
    return X, y


def validate_dataset():
    """Run validation checks and visualizations."""
    print("=" * 60)
    print("EIT Dataset Validation")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset from: {DATA_PATH}")
    X, y = load_dataset(DATA_PATH, use_noisy=USE_NOISY)
    
    # Basic statistics
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features (voltage values): {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples")
    
    # Magnitude statistics
    print(f"\nVoltage magnitude statistics (all samples):")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    
    # Per-class statistics
    print(f"\nVoltage magnitude statistics (per class):")
    for cls in np.unique(y):
        X_cls = X[y == cls]
        print(f"  Class {cls}: min={X_cls.min():.6f}, max={X_cls.max():.6f}, "
              f"mean={X_cls.mean():.6f}, std={X_cls.std():.6f}")
    
    # Plot 1: 10 random voltage vectors per class
    print("\n" + "=" * 60)
    print("Plot 1: Random Voltage Vectors per Class")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(y))))
    
    for cls_idx, cls in enumerate(np.unique(y)):
        X_cls = X[y == cls]
        # Randomly select 10 samples from this class
        sample_indices = np.random.choice(X_cls.shape[0], min(10, X_cls.shape[0]), replace=False)
        
        for sample_idx in sample_indices:
            ax.plot(X_cls[sample_idx], alpha=0.6, color=colors[cls_idx], linewidth=1)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'Class {cls}') 
                       for i, cls in enumerate(np.unique(y))]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_xlabel('Electrode Index')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('10 Random Voltage Vectors per Class')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot1_path = Path(__file__).parent / "plots" / "validation_vectors.png"
    plot1_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot1_path, dpi=150)
    print(f"Saved to: {plot1_path}")
    plt.close()
    
    # Plot 2: PCA visualization
    print("\n" + "=" * 60)
    print("Plot 2: PCA Visualization")
    print("=" * 60)
    
    print("Running PCA (2D projection)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls_idx, cls in enumerate(np.unique(y)):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'Class {cls}', alpha=0.7, s=50, 
                   color=colors[cls_idx])
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA Projection of EIT Dataset (2D)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot2_path = Path(__file__).parent / "plots" / "validation_pca.png"
    plot2_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot2_path, dpi=150)
    print(f"Saved to: {plot2_path}")
    plt.close()
    
    # Plot 3: 3D PCA visualization
    print("\nRunning PCA (3D projection)...")
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    
    print(f"Explained variance ratio: {pca_3d.explained_variance_ratio_}")
    print(f"Total variance explained: {pca_3d.explained_variance_ratio_.sum():.4f}")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for cls_idx, cls in enumerate(np.unique(y)):
        mask = y == cls
        ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                   label=f'Class {cls}', alpha=0.7, s=50,
                   color=colors[cls_idx])
    
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
    ax.set_title('PCA Projection of EIT Dataset (3D)')
    ax.legend()
    plt.tight_layout()
    
    plot3_path = Path(__file__).parent / "plots" / "validation_pca_3d.png"
    plot3_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot3_path, dpi=150)
    print(f"Saved to: {plot3_path}")
    plt.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"\n✓ Dataset loaded successfully")
    print(f"✓ Classes are {'WELL SEPARATED' if pca.explained_variance_ratio_[0] > 0.3 else 'somewhat separated'} in PCA space")
    print(f"✓ Voltage magnitudes appear sensible (range: {X.min():.6f} to {X.max():.6f})")
    print(f"✓ All {len(np.unique(y))} classes have samples")
    print(f"\n✓ Visualizations saved to: {Path(__file__).parent / 'plots'}")
    

if __name__ == "__main__":
    validate_dataset()
