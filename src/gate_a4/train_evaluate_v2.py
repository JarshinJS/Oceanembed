"""
Gate A4.2 — OceanEmbed v2-Local Training, Evaluation, and Diagnostics.
=====================================================================
Executes the OceanEmbed v2-Local training and evaluation under strict experimental
control against frozen A3.3 and A4.1 benchmarks:
  - Frozen A3 CNN: Overall 0.8602 C, Thermocline 1.1914 C
  - Frozen A4.1 OceanEmbed v1-Local: Overall 0.8460 C, Thermocline 0.9949 C

Saves:
  Dataset/gate_a4_2/
    models/oceanembed_v2_local_best.pt
    metrics/oceanembed_v2_results.json
    metrics/oceanembed_v2_depth_metrics.csv
    predictions/oceanembed_v2_test_predictions.nc
    figures/
    reports/GATE_A4_2_OCEANEMBED_V2_LOCAL.md
    configs/oceanembed_v2_local_config.json
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3
from src.gate_a4.model_v1 import MaskedMSELoss
from src.gate_a4.model_v2 import OceanEmbedV2Local
from src.gate_a4.train_evaluate_a4 import compute_training_climatology, evaluate_predictions, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gate_a4_2")

# Directories
GATE_DIR = Path("Dataset/gate_a4_2")
MODELS_DIR = GATE_DIR / "models"
METRICS_DIR = GATE_DIR / "metrics"
PREDS_DIR = GATE_DIR / "predictions"
FIGURES_DIR = GATE_DIR / "figures"
REPORTS_DIR = GATE_DIR / "reports"
CONFIGS_DIR = GATE_DIR / "configs"

for d in [MODELS_DIR, METRICS_DIR, PREDS_DIR, FIGURES_DIR, REPORTS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Frozen benchmarks
A3_RESULTS_JSON = Path("Dataset/gate_a3_temporal/metrics/gate_a3_results.json")
A4_1_RESULTS_JSON = Path("Dataset/gate_a4_ablation/metrics/a4_1_ablation_results.json")


def train_v2_model(
    model: OceanEmbedV2Local,
    train_ds: SynchronizedOceanDatasetA3,
    val_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    device: torch.device,
    save_path: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train OceanEmbed v2-Local with identical protocol."""
    set_seed(seed)
    logger.info("Training OceanEmbed v2-Local on %s...", device)

    clim_tensor = torch.from_numpy(clim_mean).to(device)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        t_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y_raw = batch["y_raw"].to(device)
            m_tgt = batch["mask_y"].to(device)

            optimizer.zero_grad()
            pred = model(x, climatology=clim_tensor)
            loss = criterion(pred, y_raw, m_tgt)
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())

        mean_train = float(np.mean(t_losses))

        # Validation
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y_raw = batch["y_raw"].to(device)
                m_tgt = batch["mask_y"].to(device)
                pred = model(x, climatology=clim_tensor)
                loss = criterion(pred, y_raw, m_tgt)
                v_losses.append(loss.item())

        mean_val = float(np.mean(v_losses))
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(mean_val)

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["lr"].append(current_lr)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if ep % 10 == 0 or ep == epochs:
            logger.info(
                "Epoch %2d/%2d | Train Loss: %.6f | Val Loss: %.6f (Best: %.6f at ep %d) | LR: %.2e",
                ep, epochs, mean_train, mean_val, best_val_loss, best_epoch, current_lr
            )

    duration = round(time.time() - start_time, 2)
    logger.info("Training completed in %.2fs. Restoring best weights from epoch %d.", duration, best_epoch)
    model.load_state_dict(best_weights)

    # Save model weights
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": best_weights,
            "best_val_loss": best_val_loss,
            "config": {
                "in_channels": model.in_channels,
                "num_depths": model.num_depths,
                "latent_dim": model.latent_dim,
                "use_residual": model.use_residual,
                "use_surface_refinement": model.use_surface_refinement,
                "upper_depths_count": model.upper_depths_count,
            },
            "parameters": model.get_parameter_count(),
        },
        save_path,
    )
    logger.info("Saved best model weights to %s", save_path)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "duration_seconds": duration,
        "parameters": model.get_parameter_count(),
        "history": history,
    }


def evaluate_model(
    model: OceanEmbedV2Local,
    test_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model predictions across test dataset."""
    model.eval()
    clim_tensor = torch.from_numpy(clim_mean).to(device)
    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    with torch.no_grad():
        batch = next(iter(loader))
        x = batch["x"].to(device)
        y_raw = batch["y_raw"].cpu().numpy()
        mask_y = batch["mask_y"].cpu().numpy()
        pred = model(x, climatology=clim_tensor).cpu().numpy()

    return pred, y_raw, mask_y


def save_test_predictions_netcdf(
    preds: np.ndarray,
    trues: np.ndarray,
    clim_mean: np.ndarray,
    masks: np.ndarray,
    test_dates: List[str],
    save_path: Path,
):
    """Save test period predictions, targets, climatology, and errors to NetCDF."""
    logger.info("Saving test predictions NetCDF to %s...", save_path)
    N, D, H, W = preds.shape

    # Canonical coordinates
    lats = np.linspace(12.125, 17.875, H)
    lons = np.linspace(85.125, 92.875, W)
    depths = np.array(REQUIRED_DEPTHS_M, dtype=np.float32)
    clim_expanded = np.repeat(clim_mean[np.newaxis, :, :, :], N, axis=0)

    ds = xr.Dataset(
        data_vars={
            "temperature_pred": (["time", "depth", "lat", "lon"], preds.astype(np.float32)),
            "temperature_target": (["time", "depth", "lat", "lon"], trues.astype(np.float32)),
            "temperature_clim": (["time", "depth", "lat", "lon"], clim_expanded.astype(np.float32)),
            "temperature_error": (["time", "depth", "lat", "lon"], (preds - trues).astype(np.float32)),
            "ocean_mask": (["time", "depth", "lat", "lon"], masks.astype(np.float32)),
        },
        coords={
            "time": pd.to_datetime(test_dates),
            "depth": depths,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "OceanEmbed v2-Local Test Period Predictions",
            "gate": "A4.2",
            "model": "OceanEmbed v2-Local (Lightweight Learned Surface Refinement)",
            "target_reference": "GLORYS12V1 Reanalysis-Derived Reference",
            "domain": "Bay of Bengal (12-18N, 85-93E)",
            "test_period": f"{test_dates[0]} to {test_dates[-1]}",
        },
    )
    ds.to_netcdf(save_path)
    logger.info("Successfully wrote NetCDF predictions: %s", save_path)


def generate_diagnostic_plots(
    v2_metrics: Dict[str, Any],
    v1_local_metrics: Dict[str, Any],
    a3_metrics: Dict[str, Any],
    clim_metrics: Dict[str, Any],
    preds: np.ndarray,
    trues: np.ndarray,
    masks: np.ndarray,
    clim_mean: np.ndarray,
    test_dates: List[str],
    figures_dir: Path,
):
    """Generate all 5 required diagnostic visualizations."""
    logger.info("Generating diagnostic plots in %s...", figures_dir)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Depth-Wise RMSE Comparison Profile
    depths = REQUIRED_DEPTHS_M
    v2_depth_rmse = [d["rmse"] for d in v2_metrics["per_depth"]]
    v1_depth_rmse = [d["rmse"] for d in v1_local_metrics["per_depth"]]
    a3_depth_rmse = [d["rmse"] for d in a3_metrics["per_depth"]]
    clim_depth_rmse = [d["rmse"] for d in clim_metrics["per_depth"]]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(clim_depth_rmse, depths, "k--", label=f"Baseline 0: Climatology ({clim_metrics['overall_rmse']:.4f} °C)", linewidth=1.5, alpha=0.8)
    ax.plot(a3_depth_rmse, depths, "C0-o", label=f"Baseline 1: Frozen A3 CNN ({a3_metrics['overall_rmse']:.4f} °C)", linewidth=2.0)
    ax.plot(v1_depth_rmse, depths, "C1--s", label=f"Frozen A4.1 v1-Local ({v1_local_metrics['overall_rmse']:.4f} °C)", linewidth=2.0)
    ax.plot(v2_depth_rmse, depths, "C2-D", label=f"A4.2 OceanEmbed v2-Local ({v2_metrics['overall_rmse']:.4f} °C)", linewidth=2.5)

    # Highlight thermocline band
    ax.axhspan(50, 200, color="orange", alpha=0.15, label="Thermocline Band (50-200m)")
    # Highlight upper ocean band
    ax.axhspan(0, 20, color="blue", alpha=0.10, label="Upper-Ocean Band (0-20m)")

    ax.set_ylim(1050, -20)
    ax.set_xlabel("RMSE (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
    ax.set_title("Gate A4.2: Depth-Wise RMSE Comparison Profile\n(Held-out Test Period: 2020-03-16 to 2020-03-31)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(figures_dir / "depth_wise_rmse_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Overall & Thermocline RMSE Grouped Bar Comparison
    models = ["Climatology", "Frozen A3 CNN", "Frozen v1-Local", "A4.2 v2-Local"]
    overall_vals = [
        clim_metrics["overall_rmse"],
        a3_metrics["overall_rmse"],
        v1_local_metrics["overall_rmse"],
        v2_metrics["overall_rmse"],
    ]
    tc_vals = [
        clim_metrics["thermocline_rmse"],
        a3_metrics["thermocline_rmse"],
        v1_local_metrics["thermocline_rmse"],
        v2_metrics["thermocline_rmse"],
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, overall_vals, width, label="Overall Column RMSE (0-1000m)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, tc_vals, width, label="Thermocline RMSE (50-200m)", color="#e07a5f")

    ax.set_ylabel("RMSE (°C)", fontsize=12, fontweight="bold")
    ax.set_title("Gate A4.2: Overall vs Thermocline RMSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.4)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    fig.savefig(figures_dir / "overall_and_thermocline_rmse_comparison.png", dpi=200)
    plt.close(fig)

    # 3. Upper-Ocean (0m, 5m, 10m, 20m) Focused RMSE Comparison
    upper_depths = [0, 5, 10, 20]
    upper_idx = [0, 1, 2, 3]

    a3_upper = [a3_metrics["per_depth"][i]["rmse"] for i in upper_idx]
    v1_upper = [v1_local_metrics["per_depth"][i]["rmse"] for i in upper_idx]
    v2_upper = [v2_metrics["per_depth"][i]["rmse"] for i in upper_idx]

    xu = np.arange(len(upper_depths))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))
    r1 = ax.bar(xu - w, a3_upper, w, label="Frozen A3 CNN", color="#457b9d")
    r2 = ax.bar(xu, v1_upper, w, label="Frozen A4.1 v1-Local", color="#e76f51")
    r3 = ax.bar(xu + w, v2_upper, w, label="A4.2 v2-Local (Refined)", color="#2a9d8f")

    ax.set_ylabel("RMSE (°C)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Depth Level", fontsize=12, fontweight="bold")
    ax.set_title("Upper-Ocean RMSE Comparison (0m, 5m, 10m, 20m)\nResolving the Surface Error Gap", fontsize=13, fontweight="bold")
    ax.set_xticks(xu)
    ax.set_xticklabels([f"{d} m" for d in upper_depths], fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.45)

    autolabel(r1)
    autolabel(r2)
    autolabel(r3)
    plt.tight_layout()
    fig.savefig(figures_dir / "upper_ocean_rmse_comparison.png", dpi=200)
    plt.close(fig)

    # 4 & 5. Representative Spatial Maps at 0m and 100m for a documented mid-test date
    # Date: 2020-03-24 (day index 8 in 16-day test window)
    sample_idx = 8
    sample_date = test_dates[sample_idx]
    logger.info("Using representative sample date: %s (index %d)", sample_date, sample_idx)

    # 4. 0m Surface Maps
    d0_target = trues[sample_idx, 0]
    d0_pred = preds[sample_idx, 0]
    d0_clim = clim_mean[0]
    d0_error = d0_pred - d0_target
    d0_mask = masks[sample_idx, 0]

    d0_target_m = np.where(d0_mask == 1, d0_target, np.nan)
    d0_pred_m = np.where(d0_mask == 1, d0_pred, np.nan)
    d0_clim_m = np.where(d0_mask == 1, d0_clim, np.nan)
    d0_error_m = np.where(d0_mask == 1, d0_error, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    vmin_0 = np.nanmin([d0_target_m, d0_pred_m])
    vmax_0 = np.nanmax([d0_target_m, d0_pred_m])

    im0 = axes[0].imshow(d0_target_m, cmap="Spectral_r", vmin=vmin_0, vmax=vmax_0)
    axes[0].set_title(f"Target Reference (0m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="°C")

    im1 = axes[1].imshow(d0_pred_m, cmap="Spectral_r", vmin=vmin_0, vmax=vmax_0)
    axes[1].set_title(f"OceanEmbed v2-Local (0m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="°C")

    im2 = axes[2].imshow(d0_clim_m, cmap="Spectral_r", vmin=vmin_0, vmax=vmax_0)
    axes[2].set_title(f"Training Climatology (0m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="°C")

    err_abs = np.nanmax(np.abs(d0_error_m))
    im3 = axes[3].imshow(d0_error_m, cmap="coolwarm", vmin=-max(1.5, err_abs), vmax=max(1.5, err_abs))
    axes[3].set_title(f"Prediction Error (Pred - Target)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="°C Error")

    for ax_i in axes:
        ax_i.axis("off")
    plt.tight_layout()
    fig.savefig(figures_dir / "representative_maps_0m.png", dpi=200)
    plt.close(fig)

    # 5. 100m Core Thermocline Maps
    d100_idx = REQUIRED_DEPTHS_M.index(100)
    d100_target = trues[sample_idx, d100_idx]
    d100_pred = preds[sample_idx, d100_idx]
    d100_clim = clim_mean[d100_idx]
    d100_error = d100_pred - d100_target
    d100_mask = masks[sample_idx, d100_idx]

    d100_target_m = np.where(d100_mask == 1, d100_target, np.nan)
    d100_pred_m = np.where(d100_mask == 1, d100_pred, np.nan)
    d100_clim_m = np.where(d100_mask == 1, d100_clim, np.nan)
    d100_error_m = np.where(d100_mask == 1, d100_error, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    vmin_100 = np.nanmin([d100_target_m, d100_pred_m])
    vmax_100 = np.nanmax([d100_target_m, d100_pred_m])

    im0 = axes[0].imshow(d100_target_m, cmap="Spectral_r", vmin=vmin_100, vmax=vmax_100)
    axes[0].set_title(f"Target Reference (100m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="°C")

    im1 = axes[1].imshow(d100_pred_m, cmap="Spectral_r", vmin=vmin_100, vmax=vmax_100)
    axes[1].set_title(f"OceanEmbed v2-Local (100m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="°C")

    im2 = axes[2].imshow(d100_clim_m, cmap="Spectral_r", vmin=vmin_100, vmax=vmax_100)
    axes[2].set_title(f"Training Climatology (100m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="°C")

    err100_abs = np.nanmax(np.abs(d100_error_m))
    im3 = axes[3].imshow(d100_error_m, cmap="coolwarm", vmin=-max(2.0, err100_abs), vmax=max(2.0, err100_abs))
    axes[3].set_title(f"Thermocline Error (100m)\n{sample_date}", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="°C Error")

    for ax_i in axes:
        ax_i.axis("off")
    plt.tight_layout()
    fig.savefig(figures_dir / "representative_maps_100m.png", dpi=200)
    plt.close(fig)
    logger.info("All diagnostic figures successfully rendered.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load Frozen Benchmarks
    with open(A4_1_RESULTS_JSON, "r") as f:
        a4_1_data = json.load(f)
    v1_local_metrics = a4_1_data["ablations"]["without_global_context"]["metrics"]
    a3_cnn_metrics = a4_1_data["benchmarks"]["a3_cnn"]
    clim_metrics = a4_1_data["benchmarks"]["climatology"]

    # 2. Load Datasets
    logger.info("Loading synchronized datasets for Train, Val, and Test splits...")
    train_ds = SynchronizedOceanDatasetA3(split="train")
    val_ds = SynchronizedOceanDatasetA3(split="val")
    test_ds = SynchronizedOceanDatasetA3(split="test")
    test_dates = test_ds.dates

    # 3. Compute Training Climatology (strictly days 1-60)
    clim_mean, clim_std = compute_training_climatology(train_ds)

    # 4. Instantiate OceanEmbed v2-Local
    model = OceanEmbedV2Local(
        in_channels=7,
        num_depths=15,
        branch_dim=16,
        latent_dim=48,
        use_residual=True,
        use_surface_refinement=True,
        upper_depths_count=4,
    ).to(device)

    # Save configuration
    config = {
        "model_name": "OceanEmbed v2-Local",
        "gate": "A4.2",
        "in_channels": 7,
        "num_depths": 15,
        "latent_dim": 48,
        "branch_dim": 16,
        "use_residual": True,
        "use_surface_refinement": True,
        "upper_depths_count": 4,
        "surface_in_channels": 3,
        "parameters": model.get_parameter_count(),
        "train_split": "2020-01-01 to 2020-02-29 (60 days)",
        "val_split": "2020-03-01 to 2020-03-15 (15 days)",
        "test_split": "2020-03-16 to 2020-03-31 (16 days)",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "epochs": 50,
        "seed": 42,
    }
    with open(CONFIGS_DIR / "oceanembed_v2_local_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 5. Train Model
    weights_path = MODELS_DIR / "oceanembed_v2_local_best.pt"
    train_info = train_v2_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        clim_mean=clim_mean,
        device=device,
        save_path=weights_path,
        epochs=50,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
    )

    # 6. Evaluate Model on Test Split
    logger.info("Evaluating OceanEmbed v2-Local on 16 held-out test snapshots...")
    preds, trues, masks = evaluate_model(model, test_ds, clim_mean, device)
    v2_metrics = evaluate_predictions(preds, trues, masks, test_dates)

    # 7. Compute Relative Differences
    diff_overall_vs_v1_local = round(float(v2_metrics["overall_rmse"] - v1_local_metrics["overall_rmse"]), 4)
    diff_tc_vs_v1_local = round(float(v2_metrics["thermocline_rmse"] - v1_local_metrics["thermocline_rmse"]), 4)
    diff_overall_vs_a3 = round(float(v2_metrics["overall_rmse"] - a3_cnn_metrics["overall_rmse"]), 4)
    diff_tc_vs_a3 = round(float(v2_metrics["thermocline_rmse"] - a3_cnn_metrics["thermocline_rmse"]), 4)

    # Upper ocean metrics (0, 5, 10, 20m)
    upper_depths_metrics = []
    for d_m in [0, 5, 10, 20]:
        idx = REQUIRED_DEPTHS_M.index(d_m)
        v2_d = v2_metrics["per_depth"][idx]
        v1_d = v1_local_metrics["per_depth"][idx]
        a3_d = a3_cnn_metrics["per_depth"][idx]
        upper_depths_metrics.append({
            "depth_m": d_m,
            "v2_rmse": v2_d["rmse"],
            "v1_local_rmse": v1_d["rmse"],
            "a3_cnn_rmse": a3_d["rmse"],
            "v2_improvement_over_v1_local_pct": round((v1_d["rmse"] - v2_d["rmse"]) / v1_d["rmse"] * 100.0, 2),
        })

    results = {
        "gate": "A4.2",
        "objective": "OceanEmbed v2-Local Refinement (Surface Pathway Integration)",
        "target_nomenclature": "GLORYS reanalysis-derived reference",
        "parameters": model.get_parameter_count(),
        "training_info": train_info,
        "metrics": v2_metrics,
        "benchmarks": {
            "climatology": clim_metrics,
            "a3_cnn": a3_cnn_metrics,
            "oceanembed_v1_local": v1_local_metrics,
        },
        "upper_ocean_summary": upper_depths_metrics,
        "comparison": {
            "delta_overall_rmse_vs_v1_local": diff_overall_vs_v1_local,
            "delta_thermocline_rmse_vs_v1_local": diff_tc_vs_v1_local,
            "delta_overall_rmse_vs_a3": diff_overall_vs_a3,
            "delta_thermocline_rmse_vs_a3": diff_tc_vs_a3,
            "thermocline_preserved": bool(v2_metrics["thermocline_rmse"] <= 1.00),
        },
    }

    # Save results JSON
    with open(METRICS_DIR / "oceanembed_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics JSON to %s", METRICS_DIR / "oceanembed_v2_results.json")

    # Save per-depth CSV
    depth_rows = []
    for d in v2_metrics["per_depth"]:
        idx = d["depth_index"]
        depth_rows.append({
            "depth_m": d["depth_m"],
            "v2_rmse": d["rmse"],
            "v2_mae": d["mae"],
            "v2_bias": d["bias"],
            "v2_correlation": d["correlation"],
            "v1_local_rmse": v1_local_metrics["per_depth"][idx]["rmse"],
            "a3_cnn_rmse": a3_cnn_metrics["per_depth"][idx]["rmse"],
            "climatology_rmse": clim_metrics["per_depth"][idx]["rmse"],
        })
    df_depth = pd.DataFrame(depth_rows)
    df_depth.to_csv(METRICS_DIR / "oceanembed_v2_depth_metrics.csv", index=False)
    logger.info("Saved depth metrics CSV to %s", METRICS_DIR / "oceanembed_v2_depth_metrics.csv")

    # 8. Save Test Predictions NetCDF
    save_test_predictions_netcdf(
        preds=preds,
        trues=trues,
        clim_mean=clim_mean,
        masks=masks,
        test_dates=test_dates,
        save_path=PREDS_DIR / "oceanembed_v2_test_predictions.nc",
    )

    # 9. Generate Diagnostic Figures
    generate_diagnostic_plots(
        v2_metrics=v2_metrics,
        v1_local_metrics=v1_local_metrics,
        a3_metrics=a3_cnn_metrics,
        clim_metrics=clim_metrics,
        preds=preds,
        trues=trues,
        masks=masks,
        clim_mean=clim_mean,
        test_dates=test_dates,
        figures_dir=FIGURES_DIR,
    )

    logger.info("Gate A4.2 Execution Finished Successfully.")


if __name__ == "__main__":
    main()
