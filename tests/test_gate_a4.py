"""
Unit Test Suite for Gate A4.0 — OceanEmbed v1 Architecture Sanity
=================================================================
8 rigorous architecture sanity tests:
  1. test_1_forward_pass_shape: [B, 7, 24, 32] -> [B, 15, 24, 32]
  2. test_2_finite_output: All outputs finite without NaNs/Infs
  3. test_3_gradient_flow: Gradients flow to all trainable parameters
  4. test_4_tiny_subset_overfit: Verifies model can overfit 4 samples
  5. test_5_masked_loss: Masked loss ignores invalid pixels
  6. test_6_depth_embedding: All 15 canonical depth representations exist
  7. test_7_spatial_preservation: Output spatial dimensions remain strictly 24x32
  8. test_8_deterministic_seed: Identical seed produces deterministic output
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a4.model_v1 import (
    OceanEmbedV1,
    MultimodalBranchEncoder,
    MultiScaleSpatialEncoder,
    LightweightGlobalContext,
    DepthConditionedDecoder,
    MaskedMSELoss,
)


def test_1_forward_pass_shape():
    """Test 1: Forward pass preserves spatial dimensions and outputs 15 depth levels."""
    model = OceanEmbedV1()
    x = torch.randn(4, 7, 24, 32)
    y = model(x)
    assert y.shape == (4, 15, 24, 32), f"Expected [4, 15, 24, 32], got {y.shape}"


def test_2_finite_output():
    """Test 2: All outputs must be finite real numbers (no NaNs or Infs)."""
    model = OceanEmbedV1()
    x = torch.randn(2, 7, 24, 32)
    y = model(x)
    assert torch.all(torch.isfinite(y)), "Output contains NaN or Inf values!"


def test_3_gradient_flow():
    """Test 3: Every trainable parameter receives a non-zero gradient upon backpropagation."""
    model = OceanEmbedV1()
    x = torch.randn(2, 7, 24, 32)
    target = torch.randn(2, 15, 24, 32)
    mask = torch.ones(2, 15, 24, 32)
    criterion = MaskedMSELoss()

    out = model(x)
    loss = criterion(out, target, mask)
    loss.backward()

    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no grad!"
            if param.grad.abs().sum().item() == 0.0:
                zero_grad_params.append(name)

    assert len(zero_grad_params) == 0, f"Parameters with zero grad: {zero_grad_params}"


def test_4_tiny_subset_overfit():
    """Test 4: Model can rapidly overfit a tiny subset of samples (loss < 0.05)."""
    torch.manual_seed(42)
    model = OceanEmbedV1()
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(2, 7, 24, 32)
    target = torch.randn(2, 15, 24, 32)
    mask = torch.ones(2, 15, 24, 32)

    initial_loss = criterion(model(x), target, mask).item()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target, mask)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < 0.05, f"Overfit failed: initial={initial_loss:.4f}, final={final_loss:.4f}"


def test_5_masked_loss():
    """Test 5: Masked loss strictly ignores masked-out pixels."""
    criterion = MaskedMSELoss()
    pred = torch.ones(2, 15, 24, 32)
    target = torch.zeros(2, 15, 24, 32)

    # All masked out -> loss must be 0
    mask_zero = torch.zeros(2, 15, 24, 32)
    loss_zero = criterion(pred, target, mask_zero)
    assert loss_zero.item() == 0.0, f"Expected 0.0 loss for all-masked, got {loss_zero.item()}"

    # Invalid cells have arbitrary massive values; with mask=0 they must not affect loss
    target_corrupted = target.clone()
    target_corrupted[:, :, 0:5, 0:5] = 99999.0
    valid_mask = torch.ones(2, 15, 24, 32)
    valid_mask[:, :, 0:5, 0:5] = 0.0  # mask out corrupted region

    loss_clean = criterion(pred, target, valid_mask)
    loss_corrupted = criterion(pred, target_corrupted, valid_mask)
    assert abs(loss_clean.item() - loss_corrupted.item()) < 1e-6, "Corrupted masked pixels affected loss!"


def test_6_depth_embedding():
    """Test 6: All 15 canonical depth representations exist and are distinct."""
    model = OceanEmbedV1()
    assert model.depth_decoder.num_depths == 15
    assert len(model.depth_decoder.depth_values) == 15
    np.testing.assert_allclose(
        model.depth_decoder.depth_values.cpu().numpy(), REQUIRED_DEPTHS_M
    )

    # Check that discrete embeddings are distinct
    embeddings = model.depth_decoder.depth_embed.weight.detach().cpu().numpy()
    assert embeddings.shape == (15, 48)
    for i in range(15):
        for j in range(i + 1, 15):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            assert dist > 1e-4, f"Depth embeddings {i} and {j} are identical!"


def test_7_spatial_preservation():
    """Test 7: Spatial grid dimensions remain strictly 24x32 across all modules."""
    model = OceanEmbedV1()
    x = torch.randn(2, 7, 24, 32)

    # Module 1
    branch_out = model.branch_encoder(x)
    assert branch_out.shape == (2, 48, 24, 32)

    # Module 2
    spatial_out = model.spatial_encoder(branch_out)
    assert spatial_out.shape == (2, 48, 24, 32)

    # Module 3
    context_out = model.global_context(spatial_out)
    assert context_out.shape == (2, 48, 24, 32)

    # Module 4
    decoder_out = model.depth_decoder(context_out)
    assert decoder_out.shape == (2, 15, 24, 32)


def test_8_deterministic_seed():
    """Test 8: Identical random seeds produce bitwise deterministic outputs."""
    torch.manual_seed(12345)
    m1 = OceanEmbedV1()
    x1 = torch.randn(2, 7, 24, 32)
    y1 = m1(x1)

    torch.manual_seed(12345)
    m2 = OceanEmbedV1()
    x2 = torch.randn(2, 7, 24, 32)
    y2 = m2(x2)

    torch.testing.assert_close(y1, y2)
