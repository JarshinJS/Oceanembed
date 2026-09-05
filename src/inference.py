"""
OceanEmbed — Inference Module
==============================
Run the trained model on a single surface field (or batch) and
get subsurface temperature predictions + embeddings.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import torch

from src.train import load_config, load_model


def predict(model, x: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    """Run inference on a batch of surface images.

    Args:
        model: trained SatEmbedNet
        x:     tensor of shape (B, C, H, W)
        device: torch device

    Returns:
        dict with 'temp_profile' (B, n_depths) and 'embedding' (B, embed_dim)
    """
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    return {
        "temp_profile": out["temp_profile"].cpu().numpy(),
        "embedding": out["embedding"].cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Run OceanEmbed inference")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="number of synthetic samples to predict")
    args = parser.parse_args()

    model, cfg, device = load_model(args.checkpoint)

    # demo with synthetic data
    from src.data import SyntheticOceanDataset
    ds = SyntheticOceanDataset(cfg, split="test")
    idxs = np.random.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    results = []
    for idx in idxs:
        x, _ = ds[int(idx)]
        x = x.unsqueeze(0)  # add batch dim
        pred = predict(model, x, device)
        results.append({
            "sample_index": int(idx),
            "temp_profile_degC": pred["temp_profile"][0].tolist(),
            "embedding_dim": pred["embedding"].shape[1],
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
