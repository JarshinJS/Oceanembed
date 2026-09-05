"""
Gate A4.2 Unit Tests: OceanEmbed v2-Local and Surface Refinement Pathway.
========================================================================
Tests:
  1. Forward pass output shape [B, 15, 24, 32]
  2. Finite values (no NaNs or Infs)
  3. Gradient flow across all branches and modules
  4. Surface pathway isolation: depths 4..14 are bitwise identical when surface pathway weights change
  5. Tiny subset overfit (loss strictly decreases)
  6. Deterministic output under identical seed
  7. Masked loss behavior over land/bathymetry
  8. Parameter count sanity check
  9. Controlled toggle of surface refinement pathway (use_surface_refinement=False)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.gate_a4.model_v1 import MaskedMSELoss
from src.gate_a4.model_v2 import OceanEmbedV2Local, SurfaceRefinementPathway


@pytest.fixture
def dummy_inputs():
    torch.manual_seed(42)
    B, C, H, W = 4, 7, 24, 32
    x = torch.randn(B, C, H, W)
    climatology = torch.randn(15, H, W)
    y_target = torch.randn(B, 15, H, W)
    mask = torch.ones(B, 15, H, W)
    mask[:, :, :4, :4] = 0.0  # simulate land/masked region
    return {"x": x, "clim": climatology, "y": y_target, "mask": mask}


def test_1_v2_forward_shape(dummy_inputs):
    """Output tensor must have shape [B, 15, 24, 32]."""
    model = OceanEmbedV2Local()
    model.eval()
    with torch.no_grad():
        out = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])
    assert out.shape == (4, 15, 24, 32), f"Expected [4, 15, 24, 32], got {out.shape}"


def test_2_v2_finite_output(dummy_inputs):
    """Output must contain only finite numbers."""
    model = OceanEmbedV2Local()
    model.eval()
    with torch.no_grad():
        out = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values!"


def test_3_v2_gradient_flow(dummy_inputs):
    """Gradients must propagate through all modules including surface refinement pathway."""
    model = OceanEmbedV2Local()
    model.train()
    criterion = MaskedMSELoss()

    out = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])
    loss = criterion(out, dummy_inputs["y"], dummy_inputs["mask"])
    loss.backward()

    # Check gradients in each key module
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Param {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Param {name} has NaN gradient!"


def test_4_v2_surface_pathway_isolation(dummy_inputs):
    """
    CRITICAL ARCHITECTURAL GUARANTEE:
    Modifying surface refinement weights MUST NOT alter depths 4..14 (30m to 1000m).
    """
    model = OceanEmbedV2Local()
    model.eval()

    with torch.no_grad():
        out_orig = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])

    # Perturb the weights of the surface refinement pathway
    with torch.no_grad():
        for p in model.surface_refinement.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    with torch.no_grad():
        out_perturbed = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])

    # Upper levels (0, 1, 2, 3 -> depths 0m, 5m, 10m, 20m) should be modified
    diff_upper = (out_perturbed[:, :4, :, :] - out_orig[:, :4, :, :]).abs().max()
    assert diff_upper > 1e-4, "Surface pathway perturbation did not affect upper depths!"

    # Deeper levels (4..14 -> depths 30m to 1000m) must remain strictly identical
    diff_lower = (out_perturbed[:, 4:, :, :] - out_orig[:, 4:, :, :]).abs().max()
    assert diff_lower < 1e-6, f"Deeper levels were modified by surface pathway! Max diff: {diff_lower}"


def test_5_v2_tiny_subset_overfit(dummy_inputs):
    """Model must strictly overfit a tiny 2-batch subset."""
    torch.manual_seed(42)
    model = OceanEmbedV2Local()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = MaskedMSELoss()

    x = dummy_inputs["x"][:2]
    y = dummy_inputs["y"][:2]
    m = dummy_inputs["mask"][:2]
    clim = dummy_inputs["clim"]

    initial_loss = None
    final_loss = None

    for step in range(30):
        optimizer.zero_grad()
        pred = model(x, climatology=clim)
        loss = criterion(pred, y, m)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 29:
            final_loss = loss.item()

    assert final_loss < initial_loss * 0.2, (
        f"Overfit failed: Initial loss={initial_loss:.4f}, Final loss={final_loss:.4f}"
    )


def test_6_v2_deterministic_seed(dummy_inputs):
    """Identical seeds must yield bitwise identical outputs."""
    torch.manual_seed(123)
    m1 = OceanEmbedV2Local()
    with torch.no_grad():
        out1 = m1(dummy_inputs["x"], climatology=dummy_inputs["clim"])

    torch.manual_seed(123)
    m2 = OceanEmbedV2Local()
    with torch.no_grad():
        out2 = m2(dummy_inputs["x"], climatology=dummy_inputs["clim"])

    torch.testing.assert_close(out1, out2)


def test_7_v2_masked_loss():
    """MaskedMSELoss ignores invalid pixels and computes mean solely on valid pixels."""
    criterion = MaskedMSELoss()
    pred = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
    target = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

    loss = criterion(pred, target, mask)
    assert loss.item() == pytest.approx(0.0)

    # Error only in masked area
    pred_corrupt = torch.tensor([[[[10.0, 20.0], [999.0, 999.0]]]])
    loss_masked = criterion(pred_corrupt, target, mask)
    assert loss_masked.item() == pytest.approx(0.0)


def test_8_v2_parameter_count():
    """Parameter count must remain lightweight and well below heavy transformer scales."""
    model = OceanEmbedV2Local()
    params = model.get_parameter_count()
    assert 90_000 <= params <= 135_000, f"Unexpected parameter count: {params}"


def test_9_v2_without_surface_refinement(dummy_inputs):
    """Model works seamlessly when surface refinement is toggled off."""
    model = OceanEmbedV2Local(use_surface_refinement=False)
    model.eval()
    with torch.no_grad():
        out = model(dummy_inputs["x"], climatology=dummy_inputs["clim"])
    assert out.shape == (4, 15, 24, 32)
    assert torch.isfinite(out).all()
