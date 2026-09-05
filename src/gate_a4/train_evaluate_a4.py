"""
Gate A4.0 — OceanEmbed v1 Training & Comprehensive Evaluation Pipeline
======================================================================
Trains OceanEmbed v1 on the frozen Gate A3 chronological sequence (Q1 2020)
and evaluates on the untouched 16-day held-out test sequence against:
  - Baseline 0: Training Spatial Climatology
  - Baseline 1: Gate A3 Simple Spatial CNN

Chronological Split Contract:
  - TRAIN: Days 0–59 (2020-01-01 to 2020-02-29, 60 days)
  - VAL: Days 60–74 (2020-03-01 to 2020-03-15, 15 days)
  - TEST: Days 75–90 (2020-03-16 to 2020-03-31, 16 days)
"""

import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M, CANONICAL_SURFACE_VARIABLES
from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3
from src.gate_a4.model_v1 import OceanEmbedV1, MaskedMSELoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gate_a4")

# Output directory paths
A4_DIR = Path("Dataset/gate_a4_oceanembed")
MODELS_DIR = A4_DIR / "models"
METRICS_DIR = A4_DIR / "metrics"
PRED_DIR = A4_DIR / "predictions"
CONFIGS_DIR = A4_DIR / "configs"
FIGURES_DIR = A4_DIR / "figures"
REPORTS_DIR = A4_DIR / "reports"

for d in [MODELS_DIR, METRICS_DIR, PRED_DIR, CONFIGS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

A3_RESULTS_JSON = Path("Dataset/gate_a3_temporal/metrics/gate_a3_results.json")


def set_seed(seed: int = 42):
    """Set deterministic random seeds across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_training_climatology(train_ds: SynchronizedOceanDatasetA3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial climatology mean field [15, 24, 32] strictly on TRAIN (Days 0–59).
    Never accesses validation or test samples.
    """
    targets, masks = [], []
    for item in train_ds:
        targets.append(item["y_raw"].numpy())
        masks.append(item["mask_y"].numpy())

    targets = np.stack(targets, axis=0)  # [60, 15, 24, 32]
    masks = np.stack(masks, axis=0)      # [60, 15, 24, 32]

    # Mean over training days where valid
    targets_masked = np.where(masks == 1.0, targets, np.nan)
    clim_mean = np.nanmean(targets_masked, axis=0)  # [15, 24, 32]
    clim_mask = np.where(~np.isnan(clim_mean), 1.0, 0.0).astype(np.float32)
    clim_mean = np.nan_to_num(clim_mean, nan=0.0).astype(np.float32)

    return clim_mean, clim_mask


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    dates: List[str],
    depths: List[float] = REQUIRED_DEPTHS_M,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute overall, depth-wise, thermocline, and temporal metrics on test samples.
    preds, targets, masks: [N, 15, 24, 32] in physical Celsius (°C).
    """
    N = len(dates)
    assert preds.shape == targets.shape == masks.shape

    # 1. Overall Metrics
    valid_idx = masks == 1.0
    diff = preds[valid_idx] - targets[valid_idx]
    overall_rmse = float(np.sqrt(np.mean(diff ** 2)))
    overall_mae = float(np.mean(np.abs(diff)))
    overall_bias = float(np.mean(diff))

    p_valid = preds[valid_idx]
    t_valid = targets[valid_idx]
    p_cent = p_valid - np.mean(p_valid)
    t_cent = t_valid - np.mean(t_valid)
    denom = np.sqrt(np.sum(p_cent ** 2) * np.sum(t_cent ** 2))
    overall_corr = float(np.sum(p_cent * t_cent) / denom) if denom > 1e-8 else 0.0

    # 2. Thermocline (50–200 m) Metrics: indices 5 to 10 inclusive
    tc_depth_indices = [5, 6, 7, 8, 9, 10]  # [50, 75, 100, 125, 150, 200] m
    tc_mask = masks[:, tc_depth_indices, :, :] == 1.0
    tc_diff = preds[:, tc_depth_indices, :, :][tc_mask] - targets[:, tc_depth_indices, :, :][tc_mask]
    tc_rmse = float(np.sqrt(np.mean(tc_diff ** 2)))
    tc_mae = float(np.mean(np.abs(tc_diff)))
    tc_bias = float(np.mean(tc_diff))

    # 3. Depth-Wise Metrics (all 15 depths)
    depth_metrics = []
    for d_idx, depth_m in enumerate(depths):
        d_mask = masks[:, d_idx, :, :] == 1.0
        if np.sum(d_mask) > 0:
            d_p = preds[:, d_idx, :, :][d_mask]
            d_t = targets[:, d_idx, :, :][d_mask]
            d_diff = d_p - d_t
            d_rmse = float(np.sqrt(np.mean(d_diff ** 2)))
            d_mae = float(np.mean(np.abs(d_diff)))
            d_bias = float(np.mean(d_diff))

            dp_cent = d_p - np.mean(d_p)
            dt_cent = d_t - np.mean(d_t)
            d_denom = np.sqrt(np.sum(dp_cent ** 2) * np.sum(dt_cent ** 2))
            d_corr = float(np.sum(dp_cent * dt_cent) / d_denom) if d_denom > 1e-8 else 0.0
        else:
            d_rmse, d_mae, d_bias, d_corr = float("nan"), float("nan"), float("nan"), float("nan")

        depth_metrics.append({
            "depth_index": d_idx,
            "depth_m": depth_m,
            "rmse": round(d_rmse, 4),
            "mae": round(d_mae, 4),
            "bias": round(d_bias, 4),
            "correlation": round(d_corr, 4),
        })

    # 4. Temporal Daily Metrics
    daily_metrics = []
    daily_overall_rmses = []
    daily_overall_maes = []
    daily_tc_rmses = []

    for t in range(N):
        t_mask = masks[t] == 1.0
        t_diff = preds[t][t_mask] - targets[t][t_mask]
        day_rmse = float(np.sqrt(np.mean(t_diff ** 2)))
        day_mae = float(np.mean(np.abs(t_diff)))
        day_bias = float(np.mean(t_diff))

        daily_metrics.append({
            "date": dates[t],
            "rmse": round(day_rmse, 4),
            "mae": round(day_mae, 4),
            "bias": round(day_bias, 4),
        })

        daily_overall_rmses.append(day_rmse)
        daily_overall_maes.append(day_mae)

        t_tc_mask = masks[t, tc_depth_indices, :, :] == 1.0
        t_tc_diff = preds[t, tc_depth_indices, :, :][t_tc_mask] - targets[t, tc_depth_indices, :, :][t_tc_mask]
        daily_tc_rmses.append(float(np.sqrt(np.mean(t_tc_diff ** 2))))

    # 5. Bootstrap Resampling (over 16 daily temporal units)
    rng = np.random.default_rng(seed)
    bs_rmses, bs_maes, bs_tc_rmses = [], [], []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(N, size=N, replace=True)
        bs_rmses.append(np.mean([daily_overall_rmses[i] for i in boot_idx]))
        bs_maes.append(np.mean([daily_overall_maes[i] for i in boot_idx]))
        bs_tc_rmses.append(np.mean([daily_tc_rmses[i] for i in boot_idx]))

    ci_rmse = [round(float(np.percentile(bs_rmses, 2.5)), 4), round(float(np.percentile(bs_rmses, 97.5)), 4)]
    ci_mae = [round(float(np.percentile(bs_maes, 2.5)), 4), round(float(np.percentile(bs_maes, 97.5)), 4)]
    ci_tc_rmse = [round(float(np.percentile(bs_tc_rmses, 2.5)), 4), round(float(np.percentile(bs_tc_rmses, 97.5)), 4)]

    return {
        "overall_rmse": round(overall_rmse, 4),
        "overall_mae": round(overall_mae, 4),
        "overall_bias": round(overall_bias, 4),
        "overall_correlation": round(overall_corr, 4),
        "thermocline_rmse": round(tc_rmse, 4),
        "thermocline_mae": round(tc_mae, 4),
        "thermocline_bias": round(tc_bias, 4),
        "per_depth": depth_metrics,
        "temporal_daily": daily_metrics,
        "overall_rmse_ci_95": ci_rmse,
        "overall_mae_ci_95": ci_mae,
        "thermocline_rmse_ci_95": ci_tc_rmse,
    }


def train_oceanembed_v1(
    train_ds: SynchronizedOceanDatasetA3,
    val_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    use_residual: bool = True,
) -> Tuple[OceanEmbedV1, Dict[str, Any]]:
    """
    Train OceanEmbed v1 with model selection based on validation loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedV1(
        in_channels=7,
        num_depths=15,
        branch_dim=16,
        latent_dim=48,
        use_residual=use_residual,
    ).to(device)

    clim_tensor = torch.from_numpy(clim_mean).to(device)  # [15, 24, 32]
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}

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
        scheduler.step(mean_val)

        history["train_loss"].append(round(mean_train, 6))
        history["val_loss"].append(round(mean_val, 6))

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    duration = time.time() - start_time
    model.load_state_dict(best_weights)
    model.eval()

    train_cfg = {
        "model_name": "OceanEmbed_v1",
        "total_parameters": model.get_parameter_count(),
        "optimizer": "Adam",
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss_function": "MaskedMSELoss",
        "weight_decay": weight_decay,
        "random_seed": 42,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "training_duration_seconds": round(duration, 2),
        "use_residual": use_residual,
        "history": history,
    }

    return model, train_cfg


def evaluate_model_on_test(
    model: OceanEmbedV1,
    test_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run inference on test samples and measure latency per sample.
    Returns: preds, trues, masks: [16, 15, 24, 32] and latency_ms.
    """
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    clim_tensor = torch.from_numpy(clim_mean).to(device)
    preds, trues, masks = [], [], []

    model.eval()
    latencies = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            t0 = time.perf_counter()
            out = model(x, climatology=clim_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            preds.append(out.cpu().numpy()[0])
            trues.append(batch["y_raw"].numpy()[0])
            masks.append(batch["mask_y"].numpy()[0])

    preds = np.stack(preds, axis=0)
    trues = np.stack(trues, axis=0)
    masks = np.stack(masks, axis=0)
    mean_latency_ms = float(np.mean(latencies))

    return preds, trues, masks, round(mean_latency_ms, 2)


def run_gate_a4_evaluation():
    """Execute complete Gate A4.0 training, benchmarking, and artifact generation."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing Gate A4.0 on compute device: {device}")

    # 1. Load Datasets
    train_ds = SynchronizedOceanDatasetA3(split="train")
    val_ds = SynchronizedOceanDatasetA3(split="val")
    test_ds = SynchronizedOceanDatasetA3(split="test")
    test_dates = test_ds.dates

    logger.info(f"Datasets loaded: Train={len(train_ds)}d, Val={len(val_ds)}d, Test={len(test_ds)}d")

    # 2. Compute Training Climatology (strictly on Days 0–59)
    clim_mean, clim_mask = compute_training_climatology(train_ds)

    # 3. Evaluate Baseline 0 (Climatology) on Test
    b0_preds = np.repeat(clim_mean[np.newaxis, ...], len(test_ds), axis=0)
    b0_trues = np.stack([item["y_raw"].numpy() for item in test_ds], axis=0)
    b0_masks = np.stack([item["mask_y"].numpy() for item in test_ds], axis=0)
    b0_metrics = evaluate_predictions(b0_preds, b0_trues, b0_masks, test_dates)

    logger.info(
        f"Baseline 0 (Climatology): RMSE={b0_metrics['overall_rmse']:.4f} °C, "
        f"Thermocline RMSE={b0_metrics['thermocline_rmse']:.4f} °C"
    )

    # 4. Load Baseline 1 (A3 Simple CNN) Benchmark Results
    if not A3_RESULTS_JSON.exists():
        raise FileNotFoundError(f"Missing Gate A3 benchmark results: {A3_RESULTS_JSON}")
    with open(A3_RESULTS_JSON, "r") as f:
        a3_results = json.load(f)
    a3_cnn_metrics = a3_results["baseline_1_cnn"]

    logger.info(
        f"Baseline 1 (A3 Simple CNN): RMSE={a3_cnn_metrics['overall_rmse']:.4f} °C, "
        f"Thermocline RMSE={a3_cnn_metrics['thermocline_rmse']:.4f} °C"
    )

    # 5. Train OceanEmbed v1
    logger.info("--- Training OceanEmbed v1 Architecture ---")
    model, train_cfg = train_oceanembed_v1(
        train_ds=train_ds,
        val_ds=val_ds,
        clim_mean=clim_mean,
        epochs=50,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        use_residual=True,
    )

    # Save model checkpoint
    model_ckpt_path = MODELS_DIR / "oceanembed_v1_best.pt"
    torch.save(model.state_dict(), model_ckpt_path)
    logger.info(f"Saved model checkpoint to {model_ckpt_path}")

    # 6. Evaluate OceanEmbed v1 on Test Sequence
    oe_preds, oe_trues, oe_masks, latency_ms = evaluate_model_on_test(
        model=model,
        test_ds=test_ds,
        clim_mean=clim_mean,
        device=device,
    )
    oe_metrics = evaluate_predictions(oe_preds, oe_trues, oe_masks, test_dates)
    oe_metrics["inference_latency_ms_per_sample"] = latency_ms

    logger.info(
        f"OceanEmbed v1: Overall RMSE={oe_metrics['overall_rmse']:.4f} °C, "
        f"Thermocline RMSE={oe_metrics['thermocline_rmse']:.4f} °C, "
        f"Inference Latency={latency_ms:.2f} ms/sample"
    )

    # 7. Save NetCDF Predictions
    pred_nc_path = PRED_DIR / "oceanembed_v1_test_predictions.nc"
    time_coords = pd.date_range(start="2020-03-16", end="2020-03-31", freq="D")
    ds_preds = xr.Dataset(
        data_vars={
            "predicted_temperature": (("time", "depth", "latitude", "longitude"), oe_preds),
            "target_temperature": (("time", "depth", "latitude", "longitude"), oe_trues),
            "mask": (("time", "depth", "latitude", "longitude"), oe_masks),
            "climatology_temperature": (("depth", "latitude", "longitude"), clim_mean),
        },
        coords={
            "time": time_coords,
            "depth": REQUIRED_DEPTHS_M,
            "latitude": test_ds.latitudes,
            "longitude": test_ds.longitudes,
        },
        attrs={
            "model": "OceanEmbed v1",
            "parameters": model.get_parameter_count(),
            "target_nomenclature": "GLORYS reanalysis-derived reference",
            "test_window": f"{test_dates[0]} to {test_dates[-1]}",
        },
    )
    ds_preds.to_netcdf(pred_nc_path)
    ds_preds.close()
    logger.info(f"Saved evaluation predictions to {pred_nc_path}")

    # 8. Depth-Wise Comparison Table
    depth_comparison = []
    for b0_d, a3_d, oe_d in zip(b0_metrics["per_depth"], a3_cnn_metrics["per_depth"], oe_metrics["per_depth"]):
        d_m = b0_d["depth_m"]
        r0 = b0_d["rmse"]
        r_a3 = a3_d["rmse"]
        r_oe = oe_d["rmse"]

        pct_vs_clim = float((r0 - r_oe) / r0 * 100)
        pct_vs_a3 = float((r_a3 - r_oe) / r_a3 * 100)

        depth_comparison.append({
            "depth_m": d_m,
            "climatology_rmse": r0,
            "a3_cnn_rmse": r_a3,
            "oceanembed_v1_rmse": r_oe,
            "improvement_vs_clim_pct": round(pct_vs_clim, 2),
            "improvement_vs_a3_pct": round(pct_vs_a3, 2),
            "climatology_mae": b0_d["mae"],
            "a3_cnn_mae": a3_d["mae"],
            "oceanembed_v1_mae": oe_d["mae"],
            "climatology_bias": b0_d["bias"],
            "a3_cnn_bias": a3_d["bias"],
            "oceanembed_v1_bias": oe_d["bias"],
            "oceanembed_v1_corr": oe_d["correlation"],
        })

    depth_df = pd.DataFrame(depth_comparison)
    depth_csv_path = METRICS_DIR / "oceanembed_v1_depth_metrics.csv"
    depth_df.to_csv(depth_csv_path, index=False)
    logger.info(f"Saved depth metrics CSV to {depth_csv_path}")

    # 9. Temporal Metrics Comparison Table
    temporal_comparison = []
    for b0_t, a3_t, oe_t in zip(b0_metrics["temporal_daily"], a3_cnn_metrics["temporal_daily"], oe_metrics["temporal_daily"]):
        date_str = b0_t["date"]
        temporal_comparison.append({
            "date": date_str,
            "climatology_rmse": b0_t["rmse"],
            "a3_cnn_rmse": a3_t["rmse"],
            "oceanembed_v1_rmse": oe_t["rmse"],
            "oceanembed_v1_mae": oe_t["mae"],
            "oceanembed_v1_bias": oe_t["bias"],
        })

    temp_df = pd.DataFrame(temporal_comparison)
    temp_csv_path = METRICS_DIR / "oceanembed_v1_temporal_metrics.csv"
    temp_df.to_csv(temp_csv_path, index=False)
    logger.info(f"Saved temporal metrics CSV to {temp_csv_path}")

    # 10. Assemble Full Results
    rel_rmse_vs_clim = float((b0_metrics["overall_rmse"] - oe_metrics["overall_rmse"]) / b0_metrics["overall_rmse"] * 100)
    rel_rmse_vs_a3 = float((a3_cnn_metrics["overall_rmse"] - oe_metrics["overall_rmse"]) / a3_cnn_metrics["overall_rmse"] * 100)
    rel_tc_vs_clim = float((b0_metrics["thermocline_rmse"] - oe_metrics["thermocline_rmse"]) / b0_metrics["thermocline_rmse"] * 100)
    rel_tc_vs_a3 = float((a3_cnn_metrics["thermocline_rmse"] - oe_metrics["thermocline_rmse"]) / a3_cnn_metrics["thermocline_rmse"] * 100)

    results = {
        "gate": "A4.0",
        "model_name": "OceanEmbed v1",
        "target_nomenclature": "GLORYS reanalysis-derived reference",
        "date_ranges": {
            "train": "2020-01-01 to 2020-02-29 (60 days)",
            "validation": "2020-03-01 to 2020-03-15 (15 days)",
            "test": f"{test_dates[0]} to {test_dates[-1]} (16 days)",
        },
        "training_configuration": train_cfg,
        "baseline_0_climatology": b0_metrics,
        "baseline_1_a3_cnn": a3_cnn_metrics,
        "oceanembed_v1": oe_metrics,
        "relative_improvements": {
            "overall_rmse_reduction_vs_clim_pct": round(rel_rmse_vs_clim, 2),
            "overall_rmse_reduction_vs_a3_cnn_pct": round(rel_rmse_vs_a3, 2),
            "thermocline_rmse_reduction_vs_clim_pct": round(rel_tc_vs_clim, 2),
            "thermocline_rmse_reduction_vs_a3_cnn_pct": round(rel_tc_vs_a3, 2),
        },
        "depth_wise_comparison": depth_comparison,
        "temporal_comparison": temporal_comparison,
    }

    results_json_path = METRICS_DIR / "oceanembed_v1_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved full results JSON to {results_json_path}")

    # Save Config JSON
    config_json_path = CONFIGS_DIR / "oceanembed_v1_config.json"
    with open(config_json_path, "w") as f:
        json.dump(train_cfg, f, indent=2)
    logger.info(f"Saved config JSON to {config_json_path}")

    return results


if __name__ == "__main__":
    run_gate_a4_evaluation()
