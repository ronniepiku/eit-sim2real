"""1D Convolutional Neural Network for EIT touch classification.

Architecture: 3x (Conv1D -> BatchNorm -> ReLU -> MaxPool) -> AdaptivePool -> FC -> Output
Designed for 1D voltage difference vectors (~208 for 16-electrode adjacent).

Uses adaptive average pooling to handle arbitrary input lengths without
requiring divisibility by 2^(num_layers).
"""

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """Single convolutional block: Conv1D -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EITConv1D(nn.Module):
    """1D-CNN classifier for EIT voltage measurements.

    Input shape: (batch_size, n_measurements)
    Output shape: (batch_size, n_classes)

    Uses ``nn.AdaptiveAvgPool1d(1)`` after the convolutional backbone so that
    the classifier head works with any input length, regardless of whether
    ``n_features`` is divisible by ``2 ** len(channels)``.

    Args:
        n_features: Length of input voltage vector (e.g., 208).
        n_classes: Number of touch classes (default: 5).
        channels: List of channel sizes for conv blocks (default: [32, 64, 128]).
        fc_dim: Hidden dimension of fully-connected layer (default: 128).
        dropout: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 5,
        channels: list[int] | None = None,
        fc_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_features = n_features

        if channels is None:
            channels = [32, 64, 128]

        # Validate that input is large enough for the pooling operations
        # Each Conv1DBlock has MaxPool1d(kernel_size=2), which halves the sequence length.
        # With len(channels) blocks, the minimum input size is 2^len(channels).
        min_features = 2 ** len(channels)
        if n_features < min_features:
            raise ValueError(
                f"Input feature size ({n_features}) is too small for CNN architecture. "
                f"The architecture has {len(channels)} pooling layers (2x reduction each), "
                f"requiring at least {min_features} input features. "
                f"\nRecommendation:\n"
                f"  • For CNN: Use eit_cleaned.mat (22 features) ✓\n"
                f"  • For shallow models (SVM/RF/MLP):\n"
                f"    - eit_cleaned.mat (22 features)\n"
                f"    - eit_cleaned_pca.mat (7 features)\n"
                f"    - eit_cleaned_umap.mat (7 features)\n"
                f"    - eit_cleaned_lda.mat (4 features)\n"
            )

        # Build convolutional backbone
        conv_layers: list[nn.Module] = []
        in_ch = 1  # Single-channel input (voltage vector as 1D signal)
        for out_ch in channels:
            conv_layers.append(Conv1DBlock(in_ch, out_ch))
            in_ch = out_ch

        self.conv_backbone = nn.Sequential(*conv_layers)

        # Adaptive pooling collapses the temporal dimension to 1,
        # making the architecture independent of n_features divisibility.
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_features).

        Returns:
            Logits of shape (batch_size, n_classes).
        """
        # Reshape: (batch, features) -> (batch, 1, features) for Conv1D
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_backbone(x)
        x = self.pool(x)  # (batch, channels[-1], 1)
        x = x.squeeze(-1)  # (batch, channels[-1])
        return self.classifier(x)
