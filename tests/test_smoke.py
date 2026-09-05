"""
Smoke test — verify imports, canonical shapes, and a single forward pass.
Run with:  python -m tests.test_smoke
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.constants import REQUIRED_DEPTHS_M, INPUT_CHANNELS, OUTPUT_DEPTHS
from src.train import load_config
from src.data import SyntheticOceanDataset
from src.model import SatEmbedNet


def test_forward_pass():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    
    # Verify configuration adheres to canonical PS specifications
    assert len(cfg["surface_variables"]) == INPUT_CHANNELS == 7, (
        f"Expected {INPUT_CHANNELS} input channels, got {len(cfg['surface_variables'])}"
    )
    assert len(cfg["target"]["depths"]) == OUTPUT_DEPTHS == 15, (
        f"Expected {OUTPUT_DEPTHS} output depths, got {len(cfg['target']['depths'])}"
    )
    assert cfg["target"]["depths"] == REQUIRED_DEPTHS_M, (
        f"Depths in config {cfg['target']['depths']} do not match REQUIRED_DEPTHS_M {REQUIRED_DEPTHS_M}"
    )

    ds = SyntheticOceanDataset(cfg, split="test")
    x, y = ds[0]
    assert x.shape[0] == 7, f"Expected 7 channels, got {x.shape[0]}"
    assert y.shape[0] == 15, f"Expected 15 depths, got {y.shape[0]}"

    model = SatEmbedNet(cfg)
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    assert "embedding" in out
    assert "temp_profile" in out
    assert out["embedding"].shape == (1, cfg["model"]["embedding_dim"])
    assert out["temp_profile"].shape == (1, 15)
    print("[PASS] Forward pass OK - input shape:", x.unsqueeze(0).shape, 
          "embedding shape:", out["embedding"].shape, 
          "profile shape:", out["temp_profile"].shape)


if __name__ == "__main__":
    test_forward_pass()
