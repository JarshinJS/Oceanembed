"""
OceanEmbed — Evaluation Module
==============================
Standalone evaluation and metric reporting for the trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SyntheticOceanDataset
from src.model import SatEmbedNet
from src.train import load_config, evaluate


def load_model(ckpt_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt["cfg"]
    model = SatEmbedNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, device


def full_report(model, cfg, device, depths) -> Dict:
    """Evaluate on train/val/test and return a combined report."""
    report = {}
    for split in ("train", "val", "test"):
        ds = SyntheticOceanDataset(cfg, split=split)
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
        m = evaluate(model, loader, device, depths=depths)
        report[split] = m
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    model, cfg, device = load_model(args.checkpoint)
    depths = cfg["target"]["depths"]
    report = full_report(model, cfg, device, depths)

    print("\n========== OceanEmbed Evaluation Report ==========")
    for split in ("train", "val", "test"):
        m = report[split]
        print(f"\n  [{split.upper()}]  n={len(SyntheticOceanDataset(cfg, split=split))}")
        print(f"    RMSE = {m['rmse_mean']:.3f} °C   Corr = {m['corr_mean']:.3f}   Bias = {m['bias_mean']:.3f} °C")
        print("    Per-depth RMSE:")
        for z, r in zip(depths, m["rmse_per_depth"]):
            print(f"      {z:5d} m : {r:.3f} °C")

    out = Path("outputs/eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
