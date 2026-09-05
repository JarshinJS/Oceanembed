"""
OceanEmbed — Visualization Module
==================================
Produces plots to demonstrate the model's predictions:

  1. Predicted vs. true temperature profiles (sample comparison)
  2. Per-depth RMSE bar chart
  3. Embedding space (PCA 2-D scatter, coloured by SST)
  4. Training loss curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import SyntheticOceanDataset
from src.model import SatEmbedNet
from src.train import load_config, evaluate, load_model


def plot_profiles(model, cfg, device, outdir: Path):
    ds = SyntheticOceanDataset(cfg, split="test")
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = model(x)["temp_profile"].cpu().numpy()
    y = y.cpu().numpy()
    depths = cfg["target"]["depths"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= pred.shape[0]:
            break
        ax.plot(y[i], depths, "b-o", label="True", markersize=4)
        ax.plot(pred[i], depths, "r--x", label="Predicted", markersize=4)
        ax.invert_yaxis()
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Predicted vs True Subsurface Temperature Profiles", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / "profile_comparison.png", dpi=150)
    plt.close()


def plot_rmse(model, cfg, device, outdir: Path):
    ds = SyntheticOceanDataset(cfg, split="test")
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    m = evaluate(model, loader, device)
    depths = cfg["target"]["depths"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(depths)), m["rmse_per_depth"], color="steelblue", edgecolor="navy")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths], rotation=45)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Per-Depth RMSE on Test Set")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "rmse_per_depth.png", dpi=150)
    plt.close()


def plot_embeddings(model, cfg, device, outdir: Path):
    ds = SyntheticOceanDataset(cfg, split="test")
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    embs, ssts = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            emb = model.get_embedding(x).cpu().numpy()
            embs.append(emb)
            ssts.append(x[:, 0].mean(dim=(1, 2)).cpu().numpy())
    embs = np.concatenate(embs)
    ssts = np.concatenate(ssts)

    # PCA to 2-D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=ssts, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(sc, label="Mean SST (°C)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Satellite Embedding Space (PCA)")
    plt.tight_layout()
    plt.savefig(outdir / "embedding_pca.png", dpi=150)
    plt.close()


def plot_loss_curves(history_path: str, outdir: Path):
    with open(history_path) as f:
        data = json.load(f)
    history = data.get("history", [])

    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, "b-", label="Train Loss")
    ax.plot(epochs, val, "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curves.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate OceanEmbed plots")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--outdir", default="outputs/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, cfg, device = load_model(args.checkpoint)
    plot_profiles(model, cfg, device, outdir)
    plot_rmse(model, cfg, device, outdir)
    plot_embeddings(model, cfg, device, outdir)

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        plot_loss_curves(str(metrics_path), outdir)

    print(f"Figures saved to {outdir}")


if __name__ == "__main__":
    main()
