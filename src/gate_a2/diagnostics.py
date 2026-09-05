"""
Gate A2 Visual Diagnostics Generator
====================================
Generates publication-quality diagnostic figures for Gate A2:
  1. Training & Validation Loss Curves
  2. Evaluation RMSE by Depth (Reference Climatology vs Simple CNN)
  3. Spatial Prediction Maps across key depths (0, 50, 100, 200, 500, 1000 m)
  4. Spatial Error Maps across key depths
  5. Vertical Profile Comparison at representative locations

All plots strictly labeled: 'GLORYS reanalysis-derived reference'.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M

DIAG_DIR = Path("Dataset/gate_a2_baseline/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("Dataset/gate_a2_baseline/metrics/baseline_results.json")
PRED_NC_PATH = Path("Dataset/gate_a2_baseline/predictions/eval_predictions_cnn.nc")


def plot_loss_curves(results: dict):
    """Plot training and validation loss curves."""
    history = results["training_configuration"]["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(epochs, history["train_loss"], label="Training Loss (Masked MSE)", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], label="Internal Validation Loss (Days 16–19)", color="#ff7f0e", lw=2)
    plt.axvline(
        x=results["training_configuration"]["best_epoch"],
        color="green",
        linestyle="--",
        label=f"Best Checkpoint (Epoch {results['training_configuration']['best_epoch']})",
    )
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Normalized Masked MSE)", fontsize=11)
    plt.title("Gate A2: SimpleSpatialCNN Training & Internal Validation Loss", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = DIAG_DIR / "loss_curve.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved loss curve to {out_file}")


def plot_rmse_by_depth(results: dict):
    """Plot per-depth RMSE comparing Reference Baseline vs Simple CNN."""
    depth_comp = results["per_depth_comparison"]
    depths = [d["depth_m"] for d in depth_comp]
    ref_rmse = [d["reference_rmse"] for d in depth_comp]
    cnn_rmse = [d["cnn_rmse"] for d in depth_comp]

    plt.figure(figsize=(7, 8), dpi=200)
    plt.plot(ref_rmse, depths, "o-", label="Reference Climatology (Baseline 0)", color="#d62728", lw=2, markersize=6)
    plt.plot(cnn_rmse, depths, "s-", label="Simple Spatial CNN (Baseline 1)", color="#1f77b4", lw=2, markersize=6)

    # Highlight thermocline zone (50 - 200m)
    plt.axhspan(50, 200, color="orange", alpha=0.15, label="Thermocline Zone (50–200 m)")

    plt.gca().invert_yaxis()
    plt.xlabel("Evaluation RMSE (°C) vs GLORYS Reference", fontsize=11)
    plt.ylabel("Depth (meters)", fontsize=11)
    plt.title("Gate A2 Evaluation RMSE by Depth\n(Target: GLORYS reanalysis-derived reference)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = DIAG_DIR / "rmse_by_depth.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved RMSE by depth to {out_file}")


def plot_spatial_predictions_and_errors(ds: xr.Dataset):
    """Plot spatial prediction maps and error maps at selected key depths."""
    selected_depths = [0, 50, 100, 200, 500, 1000]
    time_idx = 0  # First evaluation day (2020-01-21)
    date_str = str(ds.time.values[time_idx])[:10]

    # 1. Spatial Predictions Comparison (Target vs CNN)
    fig, axes = plt.subplots(len(selected_depths), 3, figsize=(12, 18), dpi=180)
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    for i, d in enumerate(selected_depths):
        tgt_slice = ds["target_reference"].sel(depth=d).isel(time=time_idx).values
        pred_slice = ds["pred_cnn"].sel(depth=d).isel(time=time_idx).values
        mask_slice = ds["mask_target"].sel(depth=d).isel(time=time_idx).values

        tgt_masked = np.where(mask_slice == 1.0, tgt_slice, np.nan)
        pred_masked = np.where(mask_slice == 1.0, pred_slice, np.nan)
        err_masked = pred_masked - tgt_masked

        # Common color scale for temperatures at depth d
        vmin = np.nanmin(tgt_masked)
        vmax = np.nanmax(tgt_masked)

        # Target Reference
        im0 = axes[i, 0].imshow(tgt_masked, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Target Ref {d}m ({date_str})", fontsize=9)
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # CNN Prediction
        im1 = axes[i, 1].imshow(pred_masked, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"CNN Prediction {d}m", fontsize=9)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Error
        err_lim = max(0.5, np.nanmax(np.abs(err_masked)))
        im2 = axes[i, 2].imshow(err_masked, origin="lower", cmap="bwr", vmin=-err_lim, vmax=err_lim)
        axes[i, 2].set_title(f"CNN Error (Pred - Ref) {d}m", fontsize=9)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Gate A2 Spatial Predictions vs GLORYS Reanalysis-Derived Reference\n(Central Bay of Bengal, 2020-01-21)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    out_file = DIAG_DIR / "spatial_predictions_depths.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved spatial predictions to {out_file}")


def plot_vertical_profiles(ds: xr.Dataset):
    """Plot vertical temperature profiles at representative valid central ocean locations."""
    time_idx = 0
    date_str = str(ds.time.values[time_idx])[:10]

    # Sample representative locations (central Bay of Bengal interior ocean)
    locations = [
        {"name": "Central South (13.6°N, 89.1°E)", "lat_idx": 6, "lon_idx": 16},
        {"name": "Central Core (15.1°N, 89.1°E)", "lat_idx": 12, "lon_idx": 16},
        {"name": "Central North (16.6°N, 89.1°E)", "lat_idx": 18, "lon_idx": 16},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), dpi=200, sharey=True)
    depths = ds.depth.values

    for ax, loc in zip(axes, locations):
        li, lj = loc["lat_idx"], loc["lon_idx"]
        tgt_prof = ds["target_reference"].isel(time=time_idx, latitude=li, longitude=lj).values
        cnn_prof = ds["pred_cnn"].isel(time=time_idx, latitude=li, longitude=lj).values
        ref_prof = ds["pred_ref"].isel(time=time_idx, latitude=li, longitude=lj).values

        ax.plot(tgt_prof, depths, "k-", lw=2.5, label="GLORYS Reference")
        ax.plot(cnn_prof, depths, "b--", lw=2, label="Simple Spatial CNN")
        ax.plot(ref_prof, depths, "r:", lw=1.8, label="Training Climatology")

        ax.set_title(loc["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("Depth (meters)", fontsize=11)
    axes[0].invert_yaxis()

    fig.suptitle(
        f"Gate A2 Subsurface Vertical Temperature Profiles ({date_str})\n(Target: GLORYS reanalysis-derived reference)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_file = DIAG_DIR / "vertical_profiles.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved vertical profiles to {out_file}")


def generate_all_diagnostics():
    """Generate all diagnostic figures."""
    with open(METRICS_PATH, "r") as f:
        results = json.load(f)

    ds_preds = xr.open_dataset(PRED_NC_PATH)

    plot_loss_curves(results)
    plot_rmse_by_depth(results)
    plot_spatial_predictions_and_errors(ds_preds)
    plot_vertical_profiles(ds_preds)

    ds_preds.close()
    print("All diagnostic figures generated successfully.")


if __name__ == "__main__":
    generate_all_diagnostics()
