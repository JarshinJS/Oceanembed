"""
Gate A4.0 — Publication-Quality Diagnostic Visualizations for OceanEmbed v1
===========================================================================
Generates 6 rigorous diagnostic figures in Dataset/gate_a4_oceanembed/figures/:
  1. oceanembed_v1_training_history.png: Train vs. Val loss curve with best epoch marked.
  2. depth_wise_rmse_comparison.png: Climatology vs. A3 Simple CNN vs. OceanEmbed v1 across all 15 depths.
  3. depth_wise_mae_comparison.png: Vertical MAE profile across all 15 depths.
  4. spatial_map_100m_representative.png: 5-panel spatial map at 100m on representative test day (2020-03-24).
  5. spatial_map_200m_representative.png: 5-panel spatial map at 200m on representative test day (2020-03-24).
  6. temporal_daily_rmse_trajectory.png: Day-by-day test trajectory (March 16–31, 2020).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M

FIGURES_DIR = Path("Dataset/gate_a4_oceanembed/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = Path("Dataset/gate_a4_oceanembed/metrics/oceanembed_v1_results.json")
DEPTH_CSV = Path("Dataset/gate_a4_oceanembed/metrics/oceanembed_v1_depth_metrics.csv")
TEMP_CSV = Path("Dataset/gate_a4_oceanembed/metrics/oceanembed_v1_temporal_metrics.csv")
PRED_NC = Path("Dataset/gate_a4_oceanembed/predictions/oceanembed_v1_test_predictions.nc")
A3_PRED_NC = Path("Dataset/gate_a3_temporal/predictions/eval_predictions_a3.nc")
INPUTS_NC = Path("Dataset/gate_a3_temporal/synchronized/oceanembed_inputs_7ch_91d.nc")


def plot_training_history(results: dict):
    """Plot 1: Training and Validation Loss history."""
    history = results["training_configuration"]["history"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    best_ep = results["training_configuration"]["best_epoch"]

    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epochs, train_loss, label="Train Loss (Masked MSE)", color="#1f77b4", lw=2)
    ax.plot(epochs, val_loss, label="Val Loss (Days 61–75)", color="#ff7f0e", lw=2)
    ax.axvline(best_ep, color="#2ca02c", linestyle="--", label=f"Best Val Epoch ({best_ep})")

    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Masked MSE Loss (°C²)", fontsize=11, fontweight="bold")
    ax.set_title("OceanEmbed v1 — Training & Validation Convergence", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    out_path = FIGURES_DIR / "oceanembed_v1_training_history.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_depth_wise_rmse(depth_df: pd.DataFrame):
    """Plot 2: Vertical RMSE Profile across all 15 depths."""
    depths = depth_df["depth_m"].values
    clim_rmse = depth_df["climatology_rmse"].values
    a3_rmse = depth_df["a3_cnn_rmse"].values
    oe_rmse = depth_df["oceanembed_v1_rmse"].values

    fig, ax = plt.subplots(figsize=(7, 9), dpi=300)
    ax.plot(clim_rmse, depths, "o-", color="#d62728", lw=2, label="Baseline 0: Training Climatology")
    ax.plot(a3_rmse, depths, "s--", color="#1f77b4", lw=2, label="Baseline 1: A3 Simple CNN")
    ax.plot(oe_rmse, depths, "D-", color="#2ca02c", lw=2.5, label="OceanEmbed v1 (Ours)")

    # Highlight thermocline region
    ax.axhspan(50, 200, color="#ffff99", alpha=0.3, label="Thermocline Region (50–200 m)")

    ax.set_yscale("symlog", linthresh=50)
    ax.invert_yaxis()
    ax.set_yticks(depths)
    ax.set_yticklabels([f"{int(d)}m" for d in depths])

    ax.set_xlabel("Test RMSE (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax.set_title("Subsurface Temperature RMSE Profile (Central Bay of Bengal)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    out_path = FIGURES_DIR / "depth_wise_rmse_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_depth_wise_mae(depth_df: pd.DataFrame):
    """Plot 3: Vertical MAE Profile across all 15 depths."""
    depths = depth_df["depth_m"].values
    clim_mae = depth_df["climatology_mae"].values
    a3_mae = depth_df["a3_cnn_mae"].values
    oe_mae = depth_df["oceanembed_v1_mae"].values

    fig, ax = plt.subplots(figsize=(7, 9), dpi=300)
    ax.plot(clim_mae, depths, "o-", color="#d62728", lw=2, label="Baseline 0: Climatology")
    ax.plot(a3_mae, depths, "s--", color="#1f77b4", lw=2, label="Baseline 1: A3 Simple CNN")
    ax.plot(oe_mae, depths, "D-", color="#2ca02c", lw=2.5, label="OceanEmbed v1 (Ours)")

    ax.axhspan(50, 200, color="#ffff99", alpha=0.3, label="Thermocline (50–200 m)")

    ax.set_yscale("symlog", linthresh=50)
    ax.invert_yaxis()
    ax.set_yticks(depths)
    ax.set_yticklabels([f"{int(d)}m" for d in depths])

    ax.set_xlabel("Test MAE (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax.set_title("Subsurface Temperature MAE Profile (Central Bay of Bengal)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    out_path = FIGURES_DIR / "depth_wise_mae_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_spatial_maps(depth_target_m: int = 100, date_str: str = "2020-03-24"):
    """
    Plot 4/5: 5-panel spatial map comparison at specified depth on representative date.
      Panel 1: SST Surface Input Context
      Panel 2: GLORYS Reanalysis-derived Reference Target
      Panel 3: A3 Simple CNN Prediction
      Panel 4: OceanEmbed v1 Prediction
      Panel 5: OceanEmbed v1 Absolute Error Field (|Pred - Ref|)
    """
    if not (PRED_NC.exists() and A3_PRED_NC.exists() and INPUTS_NC.exists()):
        print("Missing NetCDF files for spatial mapping, skipping.")
        return

    ds_oe = xr.open_dataset(PRED_NC)
    ds_a3 = xr.open_dataset(A3_PRED_NC)
    ds_in = xr.open_dataset(INPUTS_NC)

    # Slice date
    oe_sub = ds_oe.sel(time=date_str, depth=depth_target_m)
    a3_sub = ds_a3.sel(time=date_str, depth=depth_target_m)
    in_sub = ds_in.sel(time=date_str)

    sst = in_sub.sst.values
    ref = oe_sub.target_temperature.values
    oe_pred = oe_sub.predicted_temperature.values
    a3_pred = a3_sub.pred_cnn.values
    mask = oe_sub.mask.values

    # Mask invalid
    sst_m = np.where(mask == 1.0, sst, np.nan)
    ref_m = np.where(mask == 1.0, ref, np.nan)
    oe_m = np.where(mask == 1.0, oe_pred, np.nan)
    a3_m = np.where(mask == 1.0, a3_pred, np.nan)
    err_m = np.where(mask == 1.0, np.abs(oe_pred - ref), np.nan)

    lats = ds_oe.latitude.values
    lons = ds_oe.longitude.values

    ds_oe.close()
    ds_a3.close()
    ds_in.close()

    # Temperature color limits
    t_min = min(np.nanmin(ref_m), np.nanmin(oe_m), np.nanmin(a3_m))
    t_max = max(np.nanmax(ref_m), np.nanmax(oe_m), np.nanmax(a3_m))

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), dpi=300)

    # 1. SST
    im0 = axes[0].pcolormesh(lons, lats, sst_m, cmap="inferno", shading="auto")
    axes[0].set_title(f"Surface SST Input\n({date_str})", fontsize=10, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.15, label="SST (°C)")

    # 2. Reference Target
    im1 = axes[1].pcolormesh(lons, lats, ref_m, cmap="viridis", vmin=t_min, vmax=t_max, shading="auto")
    axes[1].set_title(f"GLORYS Reference Target\n({depth_target_m}m | {date_str})", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.15, label="Temp (°C)")

    # 3. A3 Simple CNN
    im2 = axes[2].pcolormesh(lons, lats, a3_m, cmap="viridis", vmin=t_min, vmax=t_max, shading="auto")
    axes[2].set_title(f"A3 Simple CNN Prediction\n({depth_target_m}m)", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.15, label="Temp (°C)")

    # 4. OceanEmbed v1
    im3 = axes[3].pcolormesh(lons, lats, oe_m, cmap="viridis", vmin=t_min, vmax=t_max, shading="auto")
    axes[3].set_title(f"OceanEmbed v1 Prediction\n({depth_target_m}m)", fontsize=10, fontweight="bold")
    fig.colorbar(im3, ax=axes[3], orientation="horizontal", pad=0.15, label="Temp (°C)")

    # 5. OceanEmbed v1 Error
    im4 = axes[4].pcolormesh(lons, lats, err_m, cmap="Reds", vmin=0, vmax=2.5, shading="auto")
    axes[4].set_title(f"OceanEmbed v1 |Error|\n({depth_target_m}m)", fontsize=10, fontweight="bold")
    fig.colorbar(im4, ax=axes[4], orientation="horizontal", pad=0.15, label="|Error| (°C)")

    for ax in axes:
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    out_path = FIGURES_DIR / f"spatial_map_{depth_target_m}m_representative.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_temporal_trajectory(temp_df: pd.DataFrame):
    """Plot 6: Day-by-Day Temporal RMSE Trajectory across the 16 test days."""
    dates = pd.to_datetime(temp_df["date"]).dt.strftime("%m-%d").values
    clim_rmse = temp_df["climatology_rmse"].values
    a3_rmse = temp_df["a3_cnn_rmse"].values
    oe_rmse = temp_df["oceanembed_v1_rmse"].values

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(dates, clim_rmse, "o-", color="#d62728", lw=2, label="Baseline 0: Climatology")
    ax.plot(dates, a3_rmse, "s--", color="#1f77b4", lw=2, label="Baseline 1: A3 Simple CNN")
    ax.plot(dates, oe_rmse, "D-", color="#2ca02c", lw=2.5, label="OceanEmbed v1 (Ours)")

    ax.set_xlabel("Test Date (2020)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Overall Test RMSE (°C)", fontsize=11, fontweight="bold")
    ax.set_title("Chronological Generalization Across 16-Day Test Holdout", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.xticks(rotation=45)

    out_path = FIGURES_DIR / "temporal_daily_rmse_trajectory.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)
    depth_df = pd.read_csv(DEPTH_CSV)
    temp_df = pd.read_csv(TEMP_CSV)

    plot_training_history(results)
    plot_depth_wise_rmse(depth_df)
    plot_depth_wise_mae(depth_df)
    plot_spatial_maps(depth_target_m=100, date_str="2020-03-24")
    plot_spatial_maps(depth_target_m=200, date_str="2020-03-24")
    plot_temporal_trajectory(temp_df)


if __name__ == "__main__":
    main()
