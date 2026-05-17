"""Unit tests for EIT 1D-CNN model architecture."""

import pytest
import torch
from models.cnn1d import Conv1DBlock, EITConv1D


class TestConv1DBlock:
    """Tests for the Conv1DBlock module."""

    def test_output_shape(self) -> None:
        block = Conv1DBlock(in_channels=1, out_channels=32, kernel_size=5, pool_size=2)
        x = torch.randn(4, 1, 208)
        out = block(x)
        assert out.shape == (4, 32, 104)  # MaxPool(2) halves length

    def test_different_kernel_sizes(self) -> None:
        for k in [3, 5, 7]:
            block = Conv1DBlock(in_channels=1, out_channels=16, kernel_size=k)
            x = torch.randn(2, 1, 100)
            out = block(x)
            assert out.shape[0] == 2
            assert out.shape[1] == 16


class TestEITConv1D:
    """Tests for the full EITConv1D classifier."""

    @pytest.fixture
    def model(self) -> EITConv1D:
        return EITConv1D(n_features=208, n_classes=5)

    def test_output_shape(self, model: EITConv1D) -> None:
        x = torch.randn(8, 208)
        out = model(x)
        assert out.shape == (8, 5)

    def test_single_sample(self, model: EITConv1D) -> None:
        x = torch.randn(1, 208)
        out = model(x)
        assert out.shape == (1, 5)

    def test_output_is_logits(self, model: EITConv1D) -> None:
        """Output should be raw logits (not probabilities)."""
        x = torch.randn(4, 208)
        out = model(x)
        # Logits can be negative
        assert out.min().item() < 0 or out.max().item() > 1

    def test_gradient_flow(self, model: EITConv1D) -> None:
        """Ensure gradients flow through the entire model."""
        x = torch.randn(4, 208, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_custom_channels(self) -> None:
        model = EITConv1D(n_features=208, n_classes=5, channels=[16, 32])
        x = torch.randn(4, 208)
        out = model(x)
        assert out.shape == (4, 5)

    def test_different_feature_lengths(self) -> None:
        """Model should work with any input length (adaptive pooling)."""
        for n_features in [100, 104, 173, 208, 256, 300]:
            model = EITConv1D(n_features=n_features, n_classes=5)
            x = torch.randn(2, n_features)
            out = model(x)
            assert out.shape == (2, 5)
