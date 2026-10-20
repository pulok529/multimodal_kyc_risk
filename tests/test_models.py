"""Unit tests verifying forward pass tensor shapes and architectures across all models."""

from __future__ import annotations

import pytest
import torch

from kyc_risk.models import (
    FusionModel,
    ImageEncoder,
    ImageOnlyModel,
    TabularEncoder,
    TabularOnlyModel,
)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_image_encoder_forward_shape(batch_size: int) -> None:
    """Verify ImageEncoder produces 64-dimensional feature embeddings."""
    encoder = ImageEncoder()
    x_img = torch.randn(batch_size, 1, 96, 96)
    feats = encoder(x_img)
    assert feats.shape == (batch_size, 64)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_tabular_encoder_forward_shape(batch_size: int) -> None:
    """Verify TabularEncoder produces 32-dimensional feature embeddings."""
    encoder = TabularEncoder(in_features=9)
    x_tab = torch.randn(batch_size, 9)
    feats = encoder(x_tab)
    assert feats.shape == (batch_size, 32)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_fusion_model_forward_shape(batch_size: int) -> None:
    """Verify multimodal FusionModel produces 1D logit tensor of shape (batch_size,)."""
    model = FusionModel(tabular_dim=9)
    x_img = torch.randn(batch_size, 1, 96, 96)
    x_tab = torch.randn(batch_size, 9)
    logits = model(x_img, x_tab)
    assert logits.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_image_only_model_forward_shape(batch_size: int) -> None:
    """Verify ImageOnlyModel produces 1D logit tensor with single or dual arguments."""
    model = ImageOnlyModel()
    x_img = torch.randn(batch_size, 1, 96, 96)
    logits_single = model(x_img)
    assert logits_single.shape == (batch_size,)

    x_tab = torch.randn(batch_size, 9)
    logits_dual = model(x_img, x_tab)
    assert logits_dual.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_tabular_only_model_forward_shape(batch_size: int) -> None:
    """Verify TabularOnlyModel produces 1D logit tensor with single or dual arguments."""
    model = TabularOnlyModel(tabular_dim=9)
    x_tab = torch.randn(batch_size, 9)
    logits_single = model(x_tab)
    assert logits_single.shape == (batch_size,)

    x_img = torch.randn(batch_size, 1, 96, 96)
    logits_dual = model(x_img, x_tab)
    assert logits_dual.shape == (batch_size,)
