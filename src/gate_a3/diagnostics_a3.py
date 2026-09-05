"""
Gate A3 Visual Diagnostics Generator
====================================
Generates the 8 publication-quality diagnostic figures required for Gate A3:
  1. fig1_loss_curves.png: Training & validation loss curves for the spatial CNN over epochs
  2. fig2_overall_rmse_comparison.png: Overall RMSE comparison with 95% bootstrap CI
  3. fig3_depth_rmse_profile.png: Vertical profile of RMSE across all 15 depths (0-1000m)
  4. fig4_depth_relative_improvement.png: Relative RMSE reduction (%) vs Depth across zones
  5. fig5_spatial_error_maps.png: Spatial error maps across 24x32 grid at key depths (10m, 100m, 500m)
  6. fig6_ablation_comparison.png: Bar chart comparing all ablation configurations & shuffled control
  7. fig7_temporal_rmse_timeline.png: Daily RMSE time-series across 31 test days (March 2020)
  8. fig8_scatter_density_thermocline.png: Scatter/hexbin plot of predicted vs reference temperature at 100m

All figures strictly labeled: 'GLORYS reanalysis-derived reference'.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DIAG_DIR = Path("Dataset/gate_a3_temporal/diagnostics")
DIAG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("Dataset/gate_a3_temporal/metrics/gate_a3_results.json")
PRED_NC_PATH = Path("Dataset/gate_a3_temporal/predictions/eval_predictions_a3.nc")


def plot_fig1_loss_curves(results: dict):
    """Fig 1: Training & validation loss curves."""
    history = results["training_configuration"]["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = results["training_configuration"]["best_epoch"]

    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(epochs, history["train_loss"], label="Train Loss (Days 1–48, Masked MSE)", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], label="Internal Val Loss (Days 49–60)", color="#ff7f0e", lw=2)
    plt.axvline(x=best_ep, color="#2ca02c", linestyle="--", lw=1.8, label=f"Best Checkpoint (Epoch {best_ep})")

    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Normalized Masked MSE Loss", fontsize=11)
    plt.title("Gate A3: SimpleSpatialCNN Training & Internal Validation Loss\n(60-day Jan–Feb 2020 Training Window)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = DIAG_DIR / "fig1_loss_curves.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig2_overall_rmse_comparison(results: dict):
    """Fig 2: Overall RMSE comparison with 95% bootstrap CI."""
    b0 = results["baseline_0_climatology"]
    b1 = results["baseline_1_cnn"]

    models = ["Reference Climatology\n(Baseline 0)", "Simple Spatial CNN\n(Baseline 1)"]
    rmses = [b0["overall_rmse"], b1["overall_rmse"]]
    ci_lowers = [b0["overall_rmse_ci_95"][0], b1["overall_rmse_ci_95"][0]]
    ci_uppers = [b0["overall_rmse_ci_95"][1], b1["overall_rmse_ci_95"][1]]
    yerr = [[rmses[0] - ci_lowers[0], rmses[1] - ci_lowers[1]],
            [ci_uppers[0] - rmses[0], ci_uppers[1] - rmses[1]]]

    colors = ["#d62728", "#1f77b4"]

    plt.figure(figsize=(7, 6), dpi=200)
    bars = plt.bar(models, rmses, yerr=yerr, capsize=8, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    plt.ylabel("Evaluation RMSE (°C)", fontsize=11)
    plt.title(f"Gate A3 Overall Evaluation RMSE (March 2020, 31 Days)\nReduction: {results['overall_relative_rmse_reduction_pct']:.2f}% | Target: GLORYS reanalysis-derived reference", fontsize=11, fontweight="bold")
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)

    for bar, rmse, ci_l, ci_u in zip(bars, rmses, ci_lowers, ci_uppers):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{rmse:.4f} °C\n[95% CI: {ci_l:.3f}, {ci_u:.3f}]",
                 ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_file = DIAG_DIR / "fig2_overall_rmse_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig3_depth_rmse_profile(results: dict):
    """Fig 3: Vertical profile of RMSE across all 15 depths (0-1000m)."""
    depth_comp = results["per_depth_comparison"]
    depths = [d["depth_m"] for d in depth_comp]
    ref_rmse = [d["reference_rmse"] for d in depth_comp]
    cnn_rmse = [d["cnn_rmse"] for d in depth_comp]

    plt.figure(figsize=(7, 8), dpi=200)
    plt.plot(ref_rmse, depths, "o-", label="Reference Climatology (Baseline 0)", color="#d62728", lw=2, markersize=6)
    plt.plot(cnn_rmse, depths, "s-", label="Simple Spatial CNN (Baseline 1)", color="#1f77b4", lw=2, markersize=6)

    # Highlight 500m specifically
    idx_500 = depths.index(500)
    plt.scatter([cnn_rmse[idx_500]], [500], color="#2ca02c", s=120, zorder=5, edgecolors="black", label=f"500 m Depth Check ({cnn_rmse[idx_500]:.3f} °C)")

    # Highlight thermocline zone
    plt.axhspan(50, 200, color="orange", alpha=0.15, label="Thermocline Zone (50–200 m)")

    plt.gca().invert_yaxis()
    plt.xlabel("Evaluation RMSE (°C) vs GLORYS Reference", fontsize=11)
    plt.ylabel("Depth (meters)", fontsize=11)
    plt.title("Gate A3 Evaluation RMSE Vertical Profile (March 2020)\n(Target: GLORYS reanalysis-derived reference)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9, loc="lower right")
    plt.tight_layout()

    out_file = DIAG_DIR / "fig3_depth_rmse_profile.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig4_depth_relative_improvement(results: dict):
    """Fig 4: Relative RMSE reduction (%) vs Depth."""
    depth_comp = results["per_depth_comparison"]
    depths = [d["depth_m"] for d in depth_comp]
    pcts = [d["relative_rmse_reduction_pct"] for d in depth_comp]

    plt.figure(figsize=(7, 8), dpi=200)
    plt.plot(pcts, depths, "d-", color="#2ca02c", lw=2.5, markersize=7, label="Relative Improvement (%)")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7)

    # Zones
    plt.axhspan(0, 30, color="#17becf", alpha=0.12, label="Mixed Layer (0–30 m)")
    plt.axhspan(50, 200, color="#ff7f0e", alpha=0.12, label="Thermocline (50–200 m)")
    plt.axhspan(300, 1000, color="#9467bd", alpha=0.12, label="Deep (300–1000 m)")

    plt.gca().invert_yaxis()
    plt.xlabel("Relative RMSE Reduction (% vs Reference Climatology)", fontsize=11)
    plt.ylabel("Depth (meters)", fontsize=11)
    plt.title("Gate A3 Relative Accuracy Gain by Depth (March 2020)\n(Target: GLORYS reanalysis-derived reference)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=9, loc="lower left")
    plt.tight_layout()

    out_file = DIAG_DIR / "fig4_depth_relative_improvement.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig5_spatial_error_maps(ds: xr.Dataset):
    """Fig 5: Spatial error maps at key depths (10m, 100m, 500m)."""
    selected_depths = [10, 100, 500]
    fig, axes = plt.subplots(len(selected_depths), 3, figsize=(13, 9), dpi=200)
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # Compute mean absolute error over the 31 test days
    mask = ds["mask_target"].values  # (time, depth, lat, lon)
    tgt = ds["target_reference"].values
    cnn = ds["pred_cnn"].values
    ref = ds["pred_ref"].values

    for i, d in enumerate(selected_depths):
        d_idx = list(ds.depth.values).index(d)
        m = mask[:, d_idx, :, :]
        t = tgt[:, d_idx, :, :]
        c = cnn[:, d_idx, :, :]
        r = ref[:, d_idx, :, :]

        err_cnn = np.abs(c - t)
        err_ref = np.abs(r - t)
        err_cnn[m == 0] = np.nan
        err_ref[m == 0] = np.nan

        mae_cnn = np.nanmean(err_cnn, axis=0)
        mae_ref = np.nanmean(err_ref, axis=0)
        diff_mae = mae_ref - mae_cnn  # Positive = CNN improves over Ref

        vmax = max(0.2, np.nanpercentile(mae_ref, 95))

        # Ref MAE
        im0 = axes[i, 0].imshow(mae_ref, origin="lower", cmap="YlOrRd", vmin=0, vmax=vmax)
        axes[i, 0].set_title(f"Ref Climatology MAE at {d}m (°C)", fontsize=10)
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # CNN MAE
        im1 = axes[i, 1].imshow(mae_cnn, origin="lower", cmap="YlOrRd", vmin=0, vmax=vmax)
        axes[i, 1].set_title(f"Simple Spatial CNN MAE at {d}m (°C)", fontsize=10)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Improvement
        diff_lim = max(0.1, np.nanmax(np.abs(diff_mae)))
        im2 = axes[i, 2].imshow(diff_mae, origin="lower", cmap="bwr", vmin=-diff_lim, vmax=diff_lim)
        axes[i, 2].set_title(f"Gain (Ref MAE - CNN MAE) at {d}m", fontsize=10)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Gate A3 Spatial Error Maps (March 2020 Mean Absolute Error)\n(Target: GLORYS reanalysis-derived reference)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out_file = DIAG_DIR / "fig5_spatial_error_maps.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig6_ablation_comparison(results: dict):
    """Fig 6: Bar chart comparing all ablation configurations & shuffled control."""
    ablations = results.get("ablations", {})
    labels = [
        "All 7 Channels",
        "SST Only (1ch)",
        "SST + SSS (2ch)",
        "SST+SSS+SSH (3ch)",
        "Surface Ocean (5ch)",
        "Shuffled Control"
    ]
    keys = ["all_7_channels", "sst_only", "sst_sss", "sst_sss_ssh", "surface_ocean_5ch", "shuffled_control"]
    rmses = [ablations.get(k, {}).get("overall_rmse", np.nan) for k in keys]
    colors = ["#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#98df8a", "#d62728"]

    plt.figure(figsize=(10, 6), dpi=200)
    bars = plt.bar(range(len(labels)), rmses, color=colors, edgecolor="black", width=0.6)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=10)
    plt.ylabel("Evaluation Overall RMSE (°C)", fontsize=11)
    plt.title("Gate A3 Input Channel Ablation & Shuffled Control Comparison\n(March 2020 Test Window | Target: GLORYS Reference)", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Reference climatology dashed line
    ref_rmse = results["baseline_0_climatology"]["overall_rmse"]
    plt.axhline(y=ref_rmse, color="#d62728", linestyle="--", label=f"Ref Climatology ({ref_rmse:.4f} °C)")

    for bar, rmse in zip(bars, rmses):
        if not np.isnan(rmse):
            plt.text(bar.get_x() + bar.get_width() / 2.0, rmse + 0.01, f"{rmse:.4f} °C",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    out_file = DIAG_DIR / "fig6_ablation_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig7_temporal_rmse_timeline(ds: xr.Dataset):
    """Fig 7: Daily RMSE time-series across 31 test days (March 2020)."""
    mask = ds["mask_target"].values
    tgt = ds["target_reference"].values
    cnn = ds["pred_cnn"].values
    ref = ds["pred_ref"].values

    n_days = len(ds.time)
    daily_cnn = []
    daily_ref = []

    for t in range(n_days):
        m = mask[t] == 1.0
        c_diff = (cnn[t][m] - tgt[t][m]) ** 2
        r_diff = (ref[t][m] - tgt[t][m]) ** 2
        daily_cnn.append(np.sqrt(np.mean(c_diff)))
        daily_ref.append(np.sqrt(np.mean(r_diff)))

    days = [str(d)[:10] for d in ds.time.values]

    plt.figure(figsize=(12, 5), dpi=200)
    plt.plot(range(n_days), daily_ref, "o--", color="#d62728", lw=2, markersize=5, label="Reference Climatology (Baseline 0)")
    plt.plot(range(n_days), daily_cnn, "s-", color="#1f77b4", lw=2.2, markersize=5, label="Simple Spatial CNN (Baseline 1)")

    plt.xticks(range(0, n_days, 2), [days[i] for i in range(0, n_days, 2)], rotation=35, ha="right", fontsize=9)
    plt.xlabel("Date (March 2020)", fontsize=11)
    plt.ylabel("Daily Subsurface RMSE (°C)", fontsize=11)
    plt.title("Gate A3 Daily Evaluation RMSE Across Pre-Monsoon Transition (March 1–31, 2020)\n(Target: GLORYS reanalysis-derived reference)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_file = DIAG_DIR / "fig7_temporal_rmse_timeline.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_fig8_scatter_density_thermocline(ds: xr.Dataset):
    """Fig 8: Scatter/hexbin plot of predicted vs reference temperature at 100m."""
    d_idx = list(ds.depth.values).index(100)
    mask = ds["mask_target"][:, d_idx, :, :].values == 1.0
    tgt_vals = ds["target_reference"][:, d_idx, :, :].values[mask]
    cnn_vals = ds["pred_cnn"][:, d_idx, :, :].values[mask]

    plt.figure(figsize=(7, 6.5), dpi=200)
    hb = plt.hexbin(tgt_vals, cnn_vals, gridsize=35, cmap="Blues", mincnt=1)
    plt.colorbar(hb, label="Point Density", fraction=0.046, pad=0.04)

    # 1:1 Line
    min_val = min(np.min(tgt_vals), np.min(cnn_vals))
    max_val = max(np.max(tgt_vals), np.max(cnn_vals))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="1:1 Perfect Prediction")

    # Correlation & R2
    corr = np.corrcoef(tgt_vals, cnn_vals)[0, 1]
    r2 = corr ** 2
    rmse = np.sqrt(np.mean((cnn_vals - tgt_vals) ** 2))

    plt.text(0.05, 0.90, f"$R^2$ = {r2:.3f}\nPearson $r$ = {corr:.3f}\nRMSE = {rmse:.3f} °C",
             transform=plt.gca().transAxes, fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))

    plt.xlabel("GLORYS Reference Temperature (°C)", fontsize=11)
    plt.ylabel("Simple Spatial CNN Prediction (°C)", fontsize=11)
    plt.title("Gate A3 Predicted vs Reference Temperature at 100 m Depth\n(March 2020 Held-Out Test Window)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()

    out_file = DIAG_DIR / "fig8_scatter_density_thermocline.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def generate_all_diagnostics():
    """Generate all 8 Gate A3 diagnostic figures."""
    with open(METRICS_PATH, "r") as f:
        results = json.load(f)

    ds_preds = xr.open_dataset(PRED_NC_PATH)

    plot_fig1_loss_curves(results)
    plot_fig2_overall_rmse_comparison(results)
    plot_fig3_depth_rmse_profile(results)
    plot_fig4_depth_relative_improvement(results)
    plot_fig5_spatial_error_maps(ds_preds)
    plot_fig6_ablation_comparison(results)
    plot_fig7_temporal_rmse_timeline(ds_preds)
    plot_fig8_scatter_density_thermocline(ds_preds)

    ds_preds.close()
    print("All 8 Gate A3 diagnostic figures generated successfully.")


if __name__ == "__main__":
    generate_all_diagnostics()
