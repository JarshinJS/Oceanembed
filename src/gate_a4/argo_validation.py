"""
Gate A4.3 — Independent ARGO In Situ Validation of Frozen OceanEmbed v1-Local.
==============================================================================
Evaluates the frozen OceanEmbed v1-Local model against independent, un-gridded
ARGO float observations in the Bay of Bengal (12–18N, 85–93E) over the held-out
test period (2020-03-16 to 2020-03-31).

Protocol:
  1. Acquire regional ARGO subset from IFREMER ERDDAP
  2. Quality control filtering (QC=1,2, physical range check, min vertical levels)
  3. Load frozen OceanEmbed v1-Local model (seed 42, verify GLORYS metrics: 0.8460 / 0.9949 C)
  4. Perform daily temporal matching and 2D bilinear spatial collocation
  5. Linearly interpolate model 15 canonical depths onto exact ARGO in situ observation depths
  6. Calculate profile-level metrics, bootstrap 95% CIs over Profile IDs, and vertical regime metrics
  7. Compare against Reference Climatology and GLORYS12V1 reference fields
  8. Generate diagnostic figures, NetCDF predictions, and structured metrics.
"""

import io
import json
import logging
import math
import os
from pathlib import Path
import ssl
import time
from typing import Dict, Any, List, Tuple, Optional
import urllib.request

import certifi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3
from src.gate_a4.model_v1 import OceanEmbedV1, MaskedMSELoss
from src.gate_a4.train_evaluate_a4 import compute_training_climatology, evaluate_predictions, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gate_a4_3")

# Directories
GATE_DIR = Path("Dataset/gate_a4_3")
MODELS_DIR = GATE_DIR / "models"
DATA_DIR = GATE_DIR / "data"
METRICS_DIR = GATE_DIR / "metrics"
PREDS_DIR = GATE_DIR / "predictions"
FIGURES_DIR = GATE_DIR / "figures"
REPORTS_DIR = GATE_DIR / "reports"

for d in [MODELS_DIR, DATA_DIR, METRICS_DIR, PREDS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Canonical Grid
LATS = np.linspace(12.125, 17.875, 24)
LONS = np.linspace(85.125, 92.875, 32)
DEPTHS_CANONICAL = np.array(REQUIRED_DEPTHS_M, dtype=np.float32)


def fetch_regional_argo_data(save_path: Path) -> pd.DataFrame:
    """
    Fetch independent ARGO observations from IFREMER ERDDAP server.
    Bounding box: 12-18N, 85-93E, 2020-03-16 to 2020-03-31.
    """
    if save_path.exists() and save_path.stat().st_size > 1000:
        logger.info("Loading cached ARGO data from %s...", save_path)
        df_raw = pd.read_csv(save_path, skiprows=[1])  # skip units row
        return df_raw

    url = (
        "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        "platform_number,time,latitude,longitude,pres,temp,pres_qc,temp_qc&"
        "time%3E=2020-03-16T00:00:00Z&time%3C=2020-03-31T23:59:59Z&"
        "latitude%3E=12&latitude%3C=18&longitude%3E=85&longitude%3C=93"
    )
    logger.info("Fetching regional ARGO subset from IFREMER ERDDAP: %s", url)

    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (OceanEmbed ARGO Validation)"})

    with urllib.request.urlopen(req, context=ctx, timeout=45) as res:
        raw_bytes = res.read()

    logger.info("Downloaded %d bytes. Saving to %s...", len(raw_bytes), save_path)
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    df_raw = pd.read_csv(io.BytesIO(raw_bytes), skiprows=[1])
    return df_raw


def quality_control_argo(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply standard ARGO Quality Control:
      - Filter pres_qc and temp_qc in ['1', '2'] (good / probably good)
      - Physical validity: 0 <= pres <= 1050 dbar, 0 <= temp <= 35 C
      - Drop duplicates & NaNs
      - Minimum profile depth count >= 5
    """
    total_raw_obs = len(df_raw)
    raw_profiles = df_raw.groupby(["platform_number", "time"]).ngroups

    # Drop NaNs
    df = df_raw.dropna(subset=["platform_number", "time", "latitude", "longitude", "pres", "temp"]).copy()

    # Cast types
    df["pres"] = pd.to_numeric(df["pres"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["pres_qc"] = df["pres_qc"].astype(str).str.strip()
    df["temp_qc"] = df["temp_qc"].astype(str).str.strip()

    # QC flag check
    qc_mask = df["pres_qc"].isin(["1", "2", "1.0", "2.0"]) & df["temp_qc"].isin(["1", "2", "1.0", "2.0"])
    df = df[qc_mask].copy()

    # Physical range check
    range_mask = (df["pres"] >= 0.0) & (df["pres"] <= 1050.0) & (df["temp"] >= 0.0) & (df["temp"] <= 35.0)
    df = df[range_mask].copy()

    # Filter profiles with >= 5 vertical observations
    profile_counts = df.groupby(["platform_number", "time"])["pres"].count()
    valid_profiles = profile_counts[profile_counts >= 5].index
    df = df.set_index(["platform_number", "time"]).loc[valid_profiles].reset_index()

    passed_obs = len(df)
    passed_profiles = df.groupby(["platform_number", "time"]).ngroups

    qc_summary = {
        "raw_observations_count": total_raw_obs,
        "raw_profiles_count": raw_profiles,
        "passed_qc_observations_count": passed_obs,
        "passed_qc_profiles_count": passed_profiles,
        "rejected_observations_count": total_raw_obs - passed_obs,
        "rejected_profiles_count": raw_profiles - passed_profiles,
        "qc_flags_allowed": ["1 (Good)", "2 (Probably Good)"],
        "pressure_range_dbar": [0.0, 1050.0],
        "temperature_range_celsius": [0.0, 35.0],
        "minimum_points_per_profile": 5,
    }
    logger.info(
        "QC Complete: %d/%d observations passed (%.1f%%) across %d/%d profiles.",
        passed_obs, total_raw_obs, (passed_obs / total_raw_obs) * 100.0, passed_profiles, raw_profiles
    )
    return df, qc_summary


def load_or_train_frozen_v1_local(
    device: torch.device,
    save_path: Path,
) -> Tuple[OceanEmbedV1, np.ndarray, np.ndarray, SynchronizedOceanDatasetA3]:
    """
    Ensure the frozen OceanEmbed v1-Local model is available and verified
    against the A4.1 frozen benchmark (Overall 0.8460 C, Thermocline 0.9949 C).
    """
    train_ds = SynchronizedOceanDatasetA3(split="train")
    val_ds = SynchronizedOceanDatasetA3(split="val")
    test_ds = SynchronizedOceanDatasetA3(split="test")
    test_dates = test_ds.dates

    clim_mean, clim_std = compute_training_climatology(train_ds)

    if save_path.exists():
        logger.info("Loading existing frozen OceanEmbed v1-Local checkpoint from %s...", save_path)
        ckpt = torch.load(save_path, map_location=device)
        model = OceanEmbedV1(
            use_residual=True,
            use_depth_conditioning=True,
            use_global_context=False,
            use_multimodal_branches=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.info("Training frozen OceanEmbed v1-Local with exact seed 42 protocol to generate frozen weights...")
        from src.gate_a4.ablation_study import train_ablation_model
        kwargs = {
            "use_residual": True,
            "use_depth_conditioning": True,
            "use_global_context": False,
            "use_multimodal_branches": True,
        }
        model, train_info = train_ablation_model(
            config_name="without_global_context",
            kwargs=kwargs,
            train_ds=train_ds,
            val_ds=val_ds,
            clim_mean=clim_mean,
            device=device,
            epochs=50,
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            seed=42,
        )
        torch.save({"model_state_dict": model.state_dict(), "train_info": train_info}, save_path)
        logger.info("Saved frozen weights to %s.", save_path)

    # Verify against frozen A4.1 GLORYS benchmark
    model.eval()
    clim_tensor = torch.from_numpy(clim_mean).to(device)
    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    with torch.no_grad():
        b = next(iter(loader))
        preds = model(b["x"].to(device), climatology=clim_tensor).cpu().numpy()
        trues = b["y_raw"].cpu().numpy()
        masks = b["mask_y"].cpu().numpy()

    metrics = evaluate_predictions(preds, trues, masks, test_dates)
    logger.info(
        "Verification against GLORYS reference: Overall RMSE = %.4f °C, Thermocline RMSE = %.4f °C",
        metrics["overall_rmse"], metrics["thermocline_rmse"]
    )
    assert abs(metrics["overall_rmse"] - 0.8460) < 0.005, f"Overall RMSE mismatch: {metrics['overall_rmse']} vs 0.8460"
    assert abs(metrics["thermocline_rmse"] - 0.9949) < 0.005, f"Thermocline RMSE mismatch: {metrics['thermocline_rmse']} vs 0.9949"
    logger.info("Frozen OceanEmbed v1-Local benchmark verification: 100% MATCH.")

    return model, preds, trues, masks, clim_mean, test_dates


def bilinear_interpolate_field(
    field_15: np.ndarray,  # [15, 24, 32]
    mask_15: np.ndarray,   # [15, 24, 32]
    lat: float,
    lon: float,
) -> Optional[np.ndarray]:
    """
    Bilinearly interpolate a 15-depth vertical field to (lat, lon).
    Returns None if any of the 4 surrounding cells is masked (land/bathymetry).
    """
    # Grid boundaries
    lat_min, lat_max = LATS[0], LATS[-1]
    lon_min, lon_max = LONS[0], LONS[-1]

    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return None

    # Find bounding indices
    d_lat = LATS[1] - LATS[0]
    d_lon = LONS[1] - LONS[0]

    i0 = int(math.floor((lat - lat_min) / d_lat))
    j0 = int(math.floor((lon - lon_min) / d_lon))

    i0 = min(max(i0, 0), len(LATS) - 2)
    j0 = min(max(j0, 0), len(LONS) - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # Check ocean mask at all 4 surrounding cells for all 15 depths
    # If any cell is masked at surface, reject collocation
    if (mask_15[0, i0, j0] == 0 or mask_15[0, i1, j0] == 0 or
        mask_15[0, i0, j1] == 0 or mask_15[0, i1, j1] == 0):
        return None

    # Bilinear weights
    u = (lat - LATS[i0]) / d_lat
    v = (lon - LONS[j0]) / d_lon

    # Interpolate for each depth
    prof = (
        (1 - u) * (1 - v) * field_15[:, i0, j0] +
        u * (1 - v) * field_15[:, i1, j0] +
        (1 - u) * v * field_15[:, i0, j1] +
        u * v * field_15[:, i1, j1]
    )
    return prof


def collocate_argo_profiles(
    df_argo: pd.DataFrame,
    preds_test: np.ndarray,  # [16, 15, 24, 32]
    trues_test: np.ndarray,  # [16, 15, 24, 32]
    masks_test: np.ndarray,  # [16, 15, 24, 32]
    clim_mean: np.ndarray,   # [15, 24, 32]
    test_dates: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Collocate ARGO profiles with OceanEmbed predictions:
      - Temporal: match same calendar day
      - Spatial: 2D bilinear interpolation from canonical 0.25 deg grid
      - Vertical: interpolate 15 canonical depths to exact ARGO measurement depths
    """
    logger.info("Beginning ARGO collocation across test period (%s to %s)...", test_dates[0], test_dates[-1])

    date_to_idx = {d: i for i, d in enumerate(test_dates)}
    profile_groups = df_argo.groupby(["platform_number", "time"])

    paired_obs_list = []
    profile_summary_list = []
    rejected_reasons = {"out_of_test_dates": 0, "out_of_grid_bounds": 0, "masked_surrounding_cells": 0}

    collocated_profiles_count = 0
    total_profiles = len(profile_groups)

    for (platform, time_str), group in profile_groups:
        date_str = pd.to_datetime(time_str).strftime("%Y-%m-%d")
        lat = float(group["latitude"].iloc[0])
        lon = float(group["longitude"].iloc[0])

        # 1. Temporal match
        if date_str not in date_to_idx:
            rejected_reasons["out_of_test_dates"] += 1
            continue

        t_idx = date_to_idx[date_str]
        preds_t = preds_test[t_idx]
        trues_t = trues_test[t_idx]
        masks_t = masks_test[t_idx]

        # 2. Spatial collocation
        pred_colloc = bilinear_interpolate_field(preds_t, masks_t, lat, lon)
        if pred_colloc is None:
            # Check why
            if not (LATS[0] <= lat <= LATS[-1] and LONS[0] <= lon <= LONS[-1]):
                rejected_reasons["out_of_grid_bounds"] += 1
            else:
                rejected_reasons["masked_surrounding_cells"] += 1
            continue

        true_colloc = bilinear_interpolate_field(trues_t, masks_t, lat, lon)
        clim_colloc = bilinear_interpolate_field(clim_mean, masks_t, lat, lon)

        # 3. Vertical interpolation to exact ARGO observation depths
        prof_id = f"{platform}_{pd.to_datetime(time_str).strftime('%Y%m%dT%H%M')}"
        collocated_profiles_count += 1

        prof_obs_errors_model = []
        prof_obs_errors_clim = []
        prof_obs_errors_glorys = []

        for _, row in group.iterrows():
            z_obs = float(row["pres"])  # 1 dbar approx 1 m
            t_obs = float(row["temp"])

            if z_obs < 0.0 or z_obs > 1000.0:
                continue

            # Linear vertical interpolation of model, climatology, and GLORYS to z_obs
            t_pred_interp = float(np.interp(z_obs, DEPTHS_CANONICAL, pred_colloc))
            t_clim_interp = float(np.interp(z_obs, DEPTHS_CANONICAL, clim_colloc))

            valid_g = np.isfinite(true_colloc)
            if valid_g.sum() >= 2:
                t_glorys_interp = float(np.interp(z_obs, DEPTHS_CANONICAL[valid_g], true_colloc[valid_g]))
            else:
                t_glorys_interp = np.nan

            err_model = t_pred_interp - t_obs
            err_clim = t_clim_interp - t_obs
            err_glorys = t_glorys_interp - t_obs if np.isfinite(t_glorys_interp) else np.nan

            prof_obs_errors_model.append(err_model)
            prof_obs_errors_clim.append(err_clim)
            prof_obs_errors_glorys.append(err_glorys)

            paired_obs_list.append({
                "profile_id": prof_id,
                "platform_number": platform,
                "timestamp": time_str,
                "date": date_str,
                "latitude": lat,
                "longitude": lon,
                "depth_m": z_obs,
                "argo_temp": t_obs,
                "model_pred_temp": t_pred_interp,
                "climatology_temp": t_clim_interp,
                "glorys_ref_temp": t_glorys_interp,
                "error_model": err_model,
                "error_climatology": err_clim,
                "error_glorys": err_glorys,
            })

        # Profile-level metrics
        if prof_obs_errors_model:
            e_m = np.array(prof_obs_errors_model)
            e_c = np.array(prof_obs_errors_clim)
            e_g = np.array(prof_obs_errors_glorys)
            e_g_val = e_g[np.isfinite(e_g)]

            profile_summary_list.append({
                "profile_id": prof_id,
                "platform_number": platform,
                "timestamp": time_str,
                "date": date_str,
                "latitude": lat,
                "longitude": lon,
                "n_obs": len(e_m),
                "model_rmse": float(np.sqrt(np.mean(e_m**2))),
                "model_mae": float(np.mean(np.abs(e_m))),
                "model_bias": float(np.mean(e_m)),
                "clim_rmse": float(np.sqrt(np.mean(e_c**2))),
                "clim_mae": float(np.mean(np.abs(e_c))),
                "clim_bias": float(np.mean(e_c)),
                "glorys_rmse": float(np.sqrt(np.mean(e_g_val**2))) if len(e_g_val) > 0 else None,
                "glorys_mae": float(np.mean(np.abs(e_g_val))) if len(e_g_val) > 0 else None,
                "glorys_bias": float(np.mean(e_g_val)) if len(e_g_val) > 0 else None,
            })

    df_paired = pd.DataFrame(paired_obs_list)
    df_profiles = pd.DataFrame(profile_summary_list)

    collocation_info = {
        "total_qc_profiles": total_profiles,
        "collocated_profiles_count": collocated_profiles_count,
        "rejected_profiles_count": total_profiles - collocated_profiles_count,
        "rejection_reasons": rejected_reasons,
        "paired_observations_count": len(df_paired),
    }
    logger.info(
        "Collocation Complete: %d/%d profiles collocated (%d paired observations).",
        collocated_profiles_count, total_profiles, len(df_paired)
    )
    return df_paired, df_profiles, collocation_info


def compute_bootstrap_profile_cis(
    df_profiles: pd.DataFrame,
    metric_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval resampled over Profile IDs."""
    np.random.seed(seed)
    vals = df_profiles[metric_col].dropna().values
    N = len(vals)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(vals, size=N, replace=True)
        boot_means.append(np.mean(sample))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return float(np.mean(vals)), ci_lower, ci_upper


def compute_comprehensive_metrics(
    df_paired: pd.DataFrame,
    df_profiles: pd.DataFrame,
) -> Dict[str, Any]:
    """Calculate overall, regime-wise, and depth-wise validation metrics."""
    logger.info("Computing validation metrics across vertical ocean regimes...")

    def get_subset_stats(sub: pd.DataFrame) -> Dict[str, Any]:
        if len(sub) == 0:
            return {"rmse": None, "mae": None, "bias": None, "correlation": None, "n_obs": 0}
        e = sub["error_model"].values
        ec = sub["error_climatology"].values
        eg = sub["error_glorys"].values
        y_obs = sub["argo_temp"].values
        y_pred = sub["model_pred_temp"].values

        # Pearson correlation
        corr = float(np.corrcoef(y_obs, y_pred)[0, 1]) if len(y_obs) > 2 and np.std(y_obs) > 1e-6 else 0.0

        eg_val = eg[np.isfinite(eg)]
        glorys_rmse = float(np.sqrt(np.mean(eg_val**2))) if len(eg_val) > 0 else None
        glorys_mae = float(np.mean(np.abs(eg_val))) if len(eg_val) > 0 else None
        glorys_bias = float(np.mean(eg_val)) if len(eg_val) > 0 else None

        return {
            "model_rmse": float(np.sqrt(np.mean(e**2))),
            "model_mae": float(np.mean(np.abs(e))),
            "model_bias": float(np.mean(e)),
            "model_correlation": round(corr, 4),
            "clim_rmse": float(np.sqrt(np.mean(ec**2))),
            "clim_mae": float(np.mean(np.abs(ec))),
            "clim_bias": float(np.mean(ec)),
            "glorys_rmse": glorys_rmse,
            "glorys_mae": glorys_mae,
            "glorys_bias": glorys_bias,
            "n_obs": len(sub),
            "n_profiles": int(sub["profile_id"].nunique()),
        }

    # 1. Vertical Regimes
    overall_stats = get_subset_stats(df_paired)
    surface_stats = get_subset_stats(df_paired[(df_paired["depth_m"] >= 0) & (df_paired["depth_m"] <= 20)])
    thermocline_stats = get_subset_stats(df_paired[(df_paired["depth_m"] >= 50) & (df_paired["depth_m"] <= 200)])
    deep_stats = get_subset_stats(df_paired[df_paired["depth_m"] > 200])

    # 2. Bootstrap CIs over Profile IDs
    m_rmse_mean, m_rmse_l, m_rmse_u = compute_bootstrap_profile_cis(df_profiles, "model_rmse")
    m_mae_mean, m_mae_l, m_mae_u = compute_bootstrap_profile_cis(df_profiles, "model_mae")
    m_bias_mean, m_bias_l, m_bias_u = compute_bootstrap_profile_cis(df_profiles, "model_bias")

    c_rmse_mean, c_rmse_l, c_rmse_u = compute_bootstrap_profile_cis(df_profiles, "clim_rmse")
    g_rmse_mean, g_rmse_l, g_rmse_u = compute_bootstrap_profile_cis(df_profiles, "glorys_rmse")

    # 3. Canonical Depth-Wise Binned Metrics
    depth_bins_config = [
        {"depth_m": 0, "tol": 3.0},
        {"depth_m": 5, "tol": 3.0},
        {"depth_m": 10, "tol": 4.0},
        {"depth_m": 20, "tol": 5.0},
        {"depth_m": 30, "tol": 5.0},
        {"depth_m": 50, "tol": 7.5},
        {"depth_m": 75, "tol": 10.0},
        {"depth_m": 100, "tol": 12.5},
        {"depth_m": 125, "tol": 12.5},
        {"depth_m": 150, "tol": 15.0},
        {"depth_m": 200, "tol": 20.0},
        {"depth_m": 300, "tol": 35.0},
        {"depth_m": 500, "tol": 50.0},
        {"depth_m": 700, "tol": 75.0},
        {"depth_m": 1000, "tol": 100.0},
    ]

    depth_wise_list = []
    for cfg in depth_bins_config:
        d = cfg["depth_m"]
        tol = cfg["tol"]
        sub_d = df_paired[(df_paired["depth_m"] >= d - tol) & (df_paired["depth_m"] <= d + tol)]

        if len(sub_d) > 0:
            e_d = sub_d["error_model"].values
            c_d = sub_d["error_climatology"].values
            g_d = sub_d["error_glorys"].values
            g_d_val = g_d[np.isfinite(g_d)]

            depth_wise_list.append({
                "canonical_depth_m": d,
                "tolerance_m": tol,
                "n_observations": len(sub_d),
                "n_profiles": int(sub_d["profile_id"].nunique()),
                "model_rmse": round(float(np.sqrt(np.mean(e_d**2))), 4),
                "model_mae": round(float(np.mean(np.abs(e_d))), 4),
                "model_bias": round(float(np.mean(e_d)), 4),
                "clim_rmse": round(float(np.sqrt(np.mean(c_d**2))), 4),
                "glorys_rmse": round(float(np.sqrt(np.mean(g_d_val**2))), 4) if len(g_d_val) > 0 else None,
            })

    results = {
        "summary": {
            "total_profiles": len(df_profiles),
            "total_observations": len(df_paired),
            "profile_rmse_mean": round(m_rmse_mean, 4),
            "profile_rmse_median": round(float(df_profiles["model_rmse"].median()), 4),
            "profile_rmse_std": round(float(df_profiles["model_rmse"].std()), 4),
            "profile_rmse_95_ci": [round(m_rmse_l, 4), round(m_rmse_u, 4)],
            "profile_mae_mean": round(m_mae_mean, 4),
            "profile_mae_95_ci": [round(m_mae_l, 4), round(m_mae_u, 4)],
            "profile_bias_mean": round(m_bias_mean, 4),
            "profile_bias_95_ci": [round(m_bias_l, 4), round(m_bias_u, 4)],
        },
        "pooled_overall_column": overall_stats,
        "pooled_surface_0_20m": surface_stats,
        "pooled_thermocline_50_200m": thermocline_stats,
        "pooled_deep_ocean_gt_200m": deep_stats,
        "benchmarks_profile_rmse": {
            "climatology_mean": round(c_rmse_mean, 4),
            "climatology_95_ci": [round(c_rmse_l, 4), round(c_rmse_u, 4)],
            "glorys_mean": round(g_rmse_mean, 4),
            "glorys_95_ci": [round(g_rmse_l, 4), round(g_rmse_u, 4)],
        },
        "depth_wise_metrics": depth_wise_list,
    }
    return results


def generate_argo_diagnostic_plots(
    df_paired: pd.DataFrame,
    df_profiles: pd.DataFrame,
    metrics_summary: Dict[str, Any],
    figures_dir: Path,
):
    """Generate all 5 diagnostic visualizations."""
    logger.info("Rendering diagnostic plots in %s...", figures_dir)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. ARGO Locations Map
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(
        df_profiles["longitude"],
        df_profiles["latitude"],
        c=pd.to_datetime(df_profiles["date"]).dt.day,
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidth=1.2,
        alpha=0.9,
    )
    cb = plt.colorbar(sc, ax=ax, label="Day of March 2020")
    ax.set_xlim(84.5, 93.5)
    ax.set_ylim(11.5, 18.5)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Gate A4.3: Independent ARGO Float Profile Locations\nBay of Bengal ({len(df_profiles)} Collocated Profiles, 2020-03-16 to 2020-03-31)",
        fontsize=12, fontweight="bold",
    )
    # Highlight canonical domain box
    ax.plot([85.125, 92.875, 92.875, 85.125, 85.125], [12.125, 12.125, 17.875, 17.875, 12.125], "r--", linewidth=1.5, label="OceanEmbed Canonical Domain")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "argo_locations.png", dpi=200)
    plt.close(fig)

    # 2. Representative Profile Comparisons (4 panels: varying depths and locations)
    selected_profiles = df_profiles.sort_values("model_rmse").iloc[[0, len(df_profiles)//3, 2*len(df_profiles)//3, -1]]["profile_id"].values
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.5), sharey=True)

    for i, pid in enumerate(selected_profiles):
        ax = axes[i]
        sub = df_paired[df_paired["profile_id"] == pid].sort_values("depth_m")
        p_info = df_profiles[df_profiles["profile_id"] == pid].iloc[0]

        ax.plot(sub["argo_temp"], sub["depth_m"], "ko-", label="ARGO In Situ Obs", markersize=3.5, linewidth=1.5)
        ax.plot(sub["model_pred_temp"], sub["depth_m"], "b-", label="OceanEmbed v1-Local", linewidth=2.0)
        ax.plot(sub["climatology_temp"], sub["depth_m"], "k--", label="Training Climatology", linewidth=1.2, alpha=0.7)
        ax.plot(sub["glorys_ref_temp"], sub["depth_m"], "g:", label="GLORYS Reference", linewidth=1.5, alpha=0.8)

        # Highlight thermocline
        ax.axhspan(50, 200, color="orange", alpha=0.15)
        ax.set_ylim(1020, -10)
        ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{p_info['date']} | Float {p_info['platform_number']}\n({p_info['latitude']:.2f}°N, {p_info['longitude']:.2f}°E)\nRMSE: {p_info['model_rmse']:.3f} °C",
            fontsize=10, fontweight="bold",
        )
        if i == 0:
            ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Representative Vertical Temperature Profiles: In Situ ARGO vs OceanEmbed v1-Local", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(figures_dir / "argo_profile_comparison.png", dpi=200)
    plt.close(fig)

    # 3. Depth-Wise RMSE Profile
    dw = pd.DataFrame(metrics_summary["depth_wise_metrics"])
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(dw["model_rmse"], dw["canonical_depth_m"], "b-o", label="OceanEmbed v1-Local", linewidth=2.2, markersize=5)
    ax.plot(dw["clim_rmse"], dw["canonical_depth_m"], "k--", label="Training Climatology", linewidth=1.6, alpha=0.8)
    ax.plot(dw["glorys_rmse"], dw["canonical_depth_m"], "g:", label="GLORYS Reanalysis Reference", linewidth=1.8, alpha=0.8)

    ax.axhspan(50, 200, color="orange", alpha=0.15, label="Thermocline Band (50–200m)")
    ax.axhspan(0, 20, color="blue", alpha=0.10, label="Mixed Layer Band (0–20m)")

    ax.set_ylim(1050, -20)
    ax.set_xlabel("RMSE (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
    ax.set_title("Gate A4.3: Vertical RMSE vs Independent ARGO Floats", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "argo_depth_rmse.png", dpi=200)
    plt.close(fig)

    # 4. Depth-Wise Bias Profile
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(dw["model_bias"], dw["canonical_depth_m"], "b-s", label="OceanEmbed v1-Local Bias", linewidth=2.2, markersize=5)
    ax.axvline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.axhspan(50, 200, color="orange", alpha=0.15, label="Thermocline Band (50–200m)")
    ax.axhspan(0, 20, color="blue", alpha=0.10, label="Mixed Layer Band (0–20m)")

    ax.set_ylim(1050, -20)
    ax.set_xlabel("Mean Bias (°C, Pred - ARGO)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
    ax.set_title("Gate A4.3: Vertical Systematic Bias vs ARGO Floats", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "argo_depth_bias.png", dpi=200)
    plt.close(fig)

    # 5. Scatter: ARGO In Situ vs OceanEmbed Predicted Temperature
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    regimes = [
        (df_paired["depth_m"] <= 20, "Mixed Layer (0-20m)", "#e63946", 0.8),
        ((df_paired["depth_m"] > 20) & (df_paired["depth_m"] <= 200), "Thermocline (20-200m)", "#f4a261", 0.7),
        (df_paired["depth_m"] > 200, "Deep Ocean (>200m)", "#2a9d8f", 0.6),
    ]
    for mask_r, label, color, alpha in regimes:
        sub_r = df_paired[mask_r]
        ax.scatter(sub_r["argo_temp"], sub_r["model_pred_temp"], label=f"{label} (N={len(sub_r)})", color=color, alpha=alpha, s=20, edgecolors="none")

    min_t = min(df_paired["argo_temp"].min(), df_paired["model_pred_temp"].min())
    max_t = max(df_paired["argo_temp"].max(), df_paired["model_pred_temp"].max())
    ax.plot([min_t, max_t], [min_t, max_t], "k--", linewidth=1.5, label="1:1 Perfect Line")

    ax.set_xlabel("Observed ARGO In Situ Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("OceanEmbed v1-Local Predicted Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"ARGO In Situ vs OceanEmbed Predicted Temperature\n(N={len(df_paired)} paired points, Pearson r={metrics_summary['pooled_overall_column']['model_correlation']:.4f})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "argo_scatter.png", dpi=200)
    plt.close(fig)

    logger.info("All diagnostic figures successfully rendered.")


def save_collocated_predictions_netcdf(
    df_paired: pd.DataFrame,
    save_path: Path,
):
    """Save collocated point observations and predictions to NetCDF."""
    logger.info("Saving collocated predictions NetCDF to %s...", save_path)
    ds = xr.Dataset(
        data_vars={
            "argo_temp": (["obs_index"], df_paired["argo_temp"].values.astype(np.float32)),
            "model_pred_temp": (["obs_index"], df_paired["model_pred_temp"].values.astype(np.float32)),
            "climatology_temp": (["obs_index"], df_paired["climatology_temp"].values.astype(np.float32)),
            "glorys_ref_temp": (["obs_index"], df_paired["glorys_ref_temp"].values.astype(np.float32)),
            "depth_m": (["obs_index"], df_paired["depth_m"].values.astype(np.float32)),
            "latitude": (["obs_index"], df_paired["latitude"].values.astype(np.float32)),
            "longitude": (["obs_index"], df_paired["longitude"].values.astype(np.float32)),
        },
        coords={
            "obs_index": np.arange(len(df_paired)),
        },
        attrs={
            "title": "OceanEmbed v1-Local Independent ARGO Collocated Validation Dataset",
            "gate": "A4.3",
            "model": "OceanEmbed v1-Local (Frozen Champion)",
            "domain": "Bay of Bengal (12-18N, 85-93E)",
            "validation_period": "2020-03-16 to 2020-03-31",
            "source": "IFREMER ARGO GDAC ERDDAP",
        },
    )
    ds.to_netcdf(save_path)
    logger.info("NetCDF predictions written to %s.", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Executing Gate A4.3 on device: %s", device)

    # 1. Acquire ARGO Data
    raw_argo_csv = DATA_DIR / "argo_raw_bob_20200316_20200331.csv"
    df_raw = fetch_regional_argo_data(raw_argo_csv)

    # 2. Quality Control
    df_qc, qc_summary = quality_control_argo(df_raw)

    # 3. Load / Verify Frozen OceanEmbed v1-Local
    model_path = MODELS_DIR / "oceanembed_v1_local_frozen.pt"
    model, preds_test, trues_test, masks_test, clim_mean, test_dates = load_or_train_frozen_v1_local(device, model_path)

    # 4. Collocation
    df_paired, df_profiles, colloc_info = collocate_argo_profiles(
        df_argo=df_qc,
        preds_test=preds_test,
        trues_test=trues_test,
        masks_test=masks_test,
        clim_mean=clim_mean,
        test_dates=test_dates,
    )

    # 5. Compute Comprehensive Metrics
    metrics_summary = compute_comprehensive_metrics(df_paired, df_profiles)

    # Save metrics JSON
    results = {
        "gate": "A4.3",
        "objective": "Independent ARGO In Situ Float Validation of Frozen OceanEmbed v1-Local",
        "frozen_model": {
            "name": "OceanEmbed v1-Local",
            "checkpoint": str(model_path),
            "parameters": model.get_parameter_count(),
            "frozen_glorys_overall_rmse": 0.8460,
            "frozen_glorys_thermocline_rmse": 0.9949,
        },
        "qc_summary": qc_summary,
        "collocation_summary": colloc_info,
        "metrics": metrics_summary,
    }

    with open(METRICS_DIR / "argo_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics JSON to %s", METRICS_DIR / "argo_validation_results.json")

    # Save CSVs
    df_profiles.to_csv(METRICS_DIR / "argo_profile_metrics.csv", index=False)
    df_depth_wise = pd.DataFrame(metrics_summary["depth_wise_metrics"])
    df_depth_wise.to_csv(METRICS_DIR / "argo_depth_metrics.csv", index=False)
    logger.info("Saved profile metrics and depth metrics CSVs.")

    # 6. Save Collocated NetCDF
    save_collocated_predictions_netcdf(df_paired, PREDS_DIR / "argo_collocated_predictions.nc")

    # 7. Render Diagnostic Figures
    generate_argo_diagnostic_plots(df_paired, df_profiles, metrics_summary, FIGURES_DIR)

    logger.info("Gate A4.3 ARGO Validation Finished Successfully.")


if __name__ == "__main__":
    main()
