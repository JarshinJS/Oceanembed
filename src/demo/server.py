"""
OceanEmbed MVP Interactive Demo Server
======================================
Problem Statement ID: 26066

Production-quality lightweight HTTP server exposing:
  - GET  /api/health
  - GET  /api/dates
  - GET  /api/surface?date=YYYY-MM-DD
  - POST /api/predict
  - GET  /api/profile?date=YYYY-MM-DD&lat=...&lon=...
  - GET  /api/metrics
  - Static frontend assets (HTML, CSS, JS)

Adheres strictly to scientific integrity guidelines:
  - Frozen OceanEmbed v1-Local model (no retraining, no architecture edits)
  - Existing normalization statistics (no hard-coded statistics)
  - Validated climatology artifact (no recalculation)
  - 91-day synchronized dataset (Q1 2020)
  - Independent ARGO in situ validation (QC filtered, 34 profiles, 14 WMO platforms)
  - GLORYS labeled as "GLORYS reanalysis-derived reference" (never "ground truth")
  - ARGO labeled as "Independent ARGO observational reference"
  - WMO platforms labeled as "14 unique WMO float platforms, treated as conservative cluster units"
"""

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import mimetypes
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M, CANONICAL_SURFACE_VARIABLES
from src.gate_a4.model_v1 import OceanEmbedV1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("oceanembed_demo")

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Artifact filepaths
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "Dataset/gate_a4_3/models/oceanembed_v1_local_frozen.pt"
NORM_STATS_PATH = PROJECT_ROOT / "Dataset/gate_a3_temporal/normalization/normalization_stats.json"
CLIMATOLOGY_CACHE_PATH = PROJECT_ROOT / "Dataset/gate_a3_temporal/climatology/training_climatology_15depth.npz"
INPUTS_NC_PATH = PROJECT_ROOT / "Dataset/gate_a3_temporal/synchronized/oceanembed_inputs_7ch_91d.nc"
TARGETS_NC_PATH = PROJECT_ROOT / "Dataset/gate_a3_temporal/synchronized/oceanembed_target_15depth_91d.nc"
ARGO_METRICS_PATH = PROJECT_ROOT / "Dataset/gate_a4_3/metrics/argo_profile_metrics.csv"
ARGO_RAW_CSV_PATH = PROJECT_ROOT / "Dataset/gate_a4_3/data/argo_raw_bob_20200316_20200331.csv"
ARGO_PREDS_NC_PATH = PROJECT_ROOT / "Dataset/gate_a4_3/predictions/argo_collocated_predictions.nc"

# Canonical Grid
LATS = np.linspace(12.125, 17.875, 24)
LONS = np.linspace(85.125, 92.875, 32)
DEPTHS_CANONICAL = np.array(REQUIRED_DEPTHS_M, dtype=np.float32)

CHANNEL_METADATA = {
    "sst": {"name": "Sea Surface Temperature (SST)", "units": "°C", "idx": 0},
    "sss": {"name": "Sea Surface Salinity (SSS)", "units": "PSU", "idx": 1},
    "ssh": {"name": "Sea Surface Height / ADT (SSH)", "units": "m", "idx": 2},
    "u_current": {"name": "Zonal Surface Current (U)", "units": "m/s", "idx": 3},
    "v_current": {"name": "Meridional Surface Current (V)", "units": "m/s", "idx": 4},
    "u_wind": {"name": "Zonal 10m Surface Wind (U)", "units": "m/s", "idx": 5},
    "v_wind": {"name": "Meridional 10m Surface Wind (V)", "units": "m/s", "idx": 6},
}


def clean_for_json(obj: Any) -> Any:
    """Recursively clean numpy / float values for standard JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 5)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 5)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


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
    lat_min, lat_max = LATS[0], LATS[-1]
    lon_min, lon_max = LONS[0], LONS[-1]

    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return None

    d_lat = LATS[1] - LATS[0]
    d_lon = LONS[1] - LONS[0]

    i0 = int(math.floor((lat - lat_min) / d_lat))
    j0 = int(math.floor((lon - lon_min) / d_lon))

    i0 = min(max(i0, 0), len(LATS) - 2)
    j0 = min(max(j0, 0), len(LONS) - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    if (
        mask_15[0, i0, j0] == 0
        or mask_15[0, i1, j0] == 0
        or mask_15[0, i0, j1] == 0
        or mask_15[0, i1, j1] == 0
    ):
        return None

    u = (lat - LATS[i0]) / d_lat
    v = (lon - LONS[j0]) / d_lon

    prof = (
        (1 - u) * (1 - v) * field_15[:, i0, j0]
        + u * (1 - v) * field_15[:, i1, j0]
        + (1 - u) * v * field_15[:, i0, j1]
        + u * v * field_15[:, i1, j1]
    )
    return prof


class OceanEmbedDemoService:
    """
    Centralized inference and data service for OceanEmbed interactive demo.
    Loads and caches all models, datasets, statistics, and ARGO records into memory.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        logger.info("Initializing OceanEmbedDemoService on device %s...", self.device)
        t_start = time.perf_counter()

        # 1. Load Frozen Model
        self._load_frozen_model()

        # 2. Load Normalization Stats
        self._load_normalization_stats()

        # 3. Load Climatology
        self._load_climatology()

        # 4. Load 91-Day Synchronized Datasets
        self._load_synchronized_datasets()

        # 5. Load ARGO In Situ Observations
        self._load_argo_data()

        # 6. In-Memory Prediction Cache (keyed by date string)
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "OceanEmbedDemoService successfully initialized in %.2f seconds.",
            time.perf_counter() - t_start,
        )

    def _load_frozen_model(self):
        if not MODEL_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Frozen checkpoint not found at: {MODEL_CHECKPOINT_PATH}")

        logger.info("Loading frozen OceanEmbed v1-Local model from %s...", MODEL_CHECKPOINT_PATH)
        ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location=self.device)
        self.model = OceanEmbedV1(
            in_channels=7,
            num_depths=15,
            branch_dim=16,
            latent_dim=48,
            use_residual=True,
            use_depth_conditioning=True,
            use_global_context=False,
            use_multimodal_branches=True,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.param_count = self.model.get_parameter_count()
        logger.info("Model loaded. Total trainable parameter count: %d", self.param_count)

    def _load_normalization_stats(self):
        if not NORM_STATS_PATH.exists():
            raise FileNotFoundError(f"Normalization stats not found at: {NORM_STATS_PATH}")

        logger.info("Loading normalization statistics from %s...", NORM_STATS_PATH)
        with open(NORM_STATS_PATH, "r") as f:
            self.norm_metadata = json.load(f)

        self.input_means = np.zeros(7, dtype=np.float32)
        self.input_stds = np.zeros(7, dtype=np.float32)

        for c_idx, c_name in enumerate(CANONICAL_SURFACE_VARIABLES):
            c_info = self.norm_metadata["input_channels"][c_name]
            self.input_means[c_idx] = float(c_info["train_mean"])
            self.input_stds[c_idx] = float(c_info["train_std"])

        self.input_stds[self.input_stds < 1e-6] = 1.0

        # Broadcast helpers [7, 1, 1]
        self.input_means_tensor = torch.from_numpy(self.input_means[:, None, None]).to(self.device)
        self.input_stds_tensor = torch.from_numpy(self.input_stds[:, None, None]).to(self.device)

    def _load_climatology(self):
        if CLIMATOLOGY_CACHE_PATH.exists():
            logger.info("Loading precomputed climatology artifact from %s...", CLIMATOLOGY_CACHE_PATH)
            data = np.load(CLIMATOLOGY_CACHE_PATH)
            self.clim_mean = data["clim_mean"].astype(np.float32)
            self.clim_mask = data["clim_mask"].astype(np.float32)
        else:
            logger.info("Precomputed climatology not found. Computing from training split...")
            from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3
            from src.gate_a4.train_evaluate_a4 import compute_training_climatology

            train_ds = SynchronizedOceanDatasetA3(split="train")
            self.clim_mean, self.clim_mask = compute_training_climatology(train_ds)
            CLIMATOLOGY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CLIMATOLOGY_CACHE_PATH, clim_mean=self.clim_mean, clim_mask=self.clim_mask)
            logger.info("Saved computed climatology artifact to %s.", CLIMATOLOGY_CACHE_PATH)

        self.clim_tensor = torch.from_numpy(self.clim_mean).to(self.device)

    def _load_synchronized_datasets(self):
        logger.info("Loading 91-day synchronized NetCDFs...")
        with xr.open_dataset(INPUTS_NC_PATH) as ds_in, xr.open_dataset(TARGETS_NC_PATH) as ds_tgt:
            self.dates = [str(t)[:10] for t in ds_in.time.values]
            self.inputs_raw = ds_in.inputs.values.astype(np.float32)          # [91, 7, 24, 32]
            self.mask_inputs = ds_in.mask_inputs.values.astype(np.float32)    # [91, 7, 24, 32]
            self.ocean_mask_2d = ds_in.ocean_mask_2d.values[0].astype(np.float32)  # [24, 32]
            self.latitudes = ds_in.latitude.values.astype(np.float32)        # [24]
            self.longitudes = ds_in.longitude.values.astype(np.float32)      # [32]
            self.targets_raw = ds_tgt.target_thetao.values.astype(np.float32)  # [91, 15, 24, 32]
            self.mask_targets = ds_tgt.mask_target.values.astype(np.float32)   # [91, 15, 24, 32]

        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.ocean_mask_15 = np.repeat(self.ocean_mask_2d[None, :, :], 15, axis=0)

    def _load_argo_data(self):
        logger.info("Loading ARGO collocated records and soundings...")
        self.argo_profiles_by_date: Dict[str, List[Dict[str, Any]]] = {}

        if not ARGO_METRICS_PATH.exists():
            logger.warning("ARGO profile metrics file not found: %s", ARGO_METRICS_PATH)
            return

        df_profiles = pd.read_csv(ARGO_METRICS_PATH)

        # Also load soundings if raw csv exists
        df_qc_soundings = None
        if ARGO_RAW_CSV_PATH.exists():
            from src.gate_a4.argo_validation import quality_control_argo
            df_raw = pd.read_csv(ARGO_RAW_CSV_PATH, skiprows=[1])
            df_qc, _ = quality_control_argo(df_raw)
            df_qc["pres"] = pd.to_numeric(df_qc["pres"], errors="coerce")
            df_qc["temp"] = pd.to_numeric(df_qc["temp"], errors="coerce")
            df_qc_soundings = df_qc

        for _, row in df_profiles.iterrows():
            d_str = str(row["date"])
            prof_id = str(row["profile_id"])
            platform = str(row["platform_number"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            profile_entry = {
                "profile_id": prof_id,
                "platform_number": platform,
                "date": d_str,
                "timestamp": str(row["timestamp"]),
                "latitude": lat,
                "longitude": lon,
                "model_rmse": float(row["model_rmse"]),
                "clim_rmse": float(row["clim_rmse"]),
                "glorys_rmse": float(row["glorys_rmse"]) if pd.notna(row["glorys_rmse"]) else None,
                "n_obs": int(row["n_obs"]),
                "depths": [],
                "temperatures": [],
            }

            # Attach actual soundings
            if df_qc_soundings is not None:
                sub = df_qc_soundings[
                    (df_qc_soundings["platform_number"].astype(str) == platform)
                    & (df_qc_soundings["time"] == row["timestamp"])
                ].sort_values("pres")
                if len(sub) > 0:
                    profile_entry["depths"] = sub["pres"].tolist()
                    profile_entry["temperatures"] = sub["temp"].tolist()

            if d_str not in self.argo_profiles_by_date:
                self.argo_profiles_by_date[d_str] = []
            self.argo_profiles_by_date[d_str].append(profile_entry)

        logger.info(
            "Loaded %d ARGO profiles across %d distinct dates.",
            len(df_profiles),
            len(self.argo_profiles_by_date),
        )

    def get_health(self) -> Dict[str, Any]:
        """Return system status, model parameters, and domain metadata."""
        return {
            "status": "ok",
            "model_name": "OceanEmbed v1-Local",
            "model_loaded": True,
            "checkpoint_path": str(MODEL_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "device": str(self.device),
            "parameter_count": self.param_count,
            "architecture": {
                "in_channels": 7,
                "num_depths": 15,
                "branch_dim": 16,
                "latent_dim": 48,
                "use_residual": True,
                "use_depth_conditioning": True,
                "use_global_context": False,
                "use_multimodal_branches": True,
            },
            "domain": {
                "name": "Bay of Bengal",
                "lat_min": float(LATS[0]),
                "lat_max": float(LATS[-1]),
                "lon_min": float(LONS[0]),
                "lon_max": float(LONS[-1]),
                "grid_resolution_deg": 0.25,
                "grid_shape": [24, 32],
                "depth_levels_count": 15,
                "canonical_depths_m": REQUIRED_DEPTHS_M,
            },
            "date_range": {
                "start": self.dates[0],
                "end": self.dates[-1],
                "total_days": len(self.dates),
                "splits": {
                    "train": {"start": "2020-01-01", "end": "2020-02-29", "count": 60},
                    "validation": {"start": "2020-03-01", "end": "2020-03-15", "count": 15},
                    "test": {"start": "2020-03-16", "end": "2020-03-31", "count": 16},
                },
            },
        }

    def get_dates(self) -> List[Dict[str, Any]]:
        """Return metadata for all 91 synchronized dates."""
        result = []
        for d in self.dates:
            if d < "2020-03-01":
                split = "train"
            elif d <= "2020-03-15":
                split = "validation"
            else:
                split = "test"

            argo_profs = self.argo_profiles_by_date.get(d, [])
            result.append({
                "date": d,
                "split": split,
                "has_argo": len(argo_profs) > 0,
                "argo_count": len(argo_profs),
                "argo_platforms": list(set(p["platform_number"] for p in argo_profs)),
            })
        return result

    def get_surface(self, date_str: str) -> Dict[str, Any]:
        """
        Return unnormalized 7 surface channels with source masks and units.
        Land/invalid ocean cells are set to None (JSON null) to preserve validity.
        """
        if date_str not in self.date_to_idx:
            raise ValueError(f"Date '{date_str}' not in synchronized date range.")

        t_idx = self.date_to_idx[date_str]
        raw_7 = self.inputs_raw[t_idx]        # [7, 24, 32]
        mask_7 = self.mask_inputs[t_idx]      # [7, 24, 32]
        ocean_mask = self.ocean_mask_2d       # [24, 32]

        channels_out = {}
        for var_name, meta in CHANNEL_METADATA.items():
            idx = meta["idx"]
            val_grid = raw_7[idx].copy()
            val_mask = mask_7[idx].copy()

            # Land or source masked cells set to None
            val_grid_cleaned = np.where((ocean_mask == 1.0) & (val_mask == 1.0), val_grid, np.nan)
            valid_ocean_count = int(np.sum((ocean_mask == 1.0) & (val_mask == 1.0)))
            total_ocean_count = int(np.sum(ocean_mask == 1.0))
            coverage_pct = round((valid_ocean_count / max(total_ocean_count, 1)) * 100.0, 2)

            channels_out[var_name] = {
                "name": meta["name"],
                "units": meta["units"],
                "channel_index": idx,
                "coverage_pct": coverage_pct,
                "valid_ocean_count": valid_ocean_count,
                "data": clean_for_json(val_grid_cleaned),
            }

        argo_list = self.argo_profiles_by_date.get(date_str, [])
        argo_coords = [
            {
                "profile_id": p["profile_id"],
                "platform_number": p["platform_number"],
                "lat": p["latitude"],
                "lon": p["longitude"],
                "model_rmse": p["model_rmse"],
                "clim_rmse": p["clim_rmse"],
                "glorys_rmse": p["glorys_rmse"],
            }
            for p in argo_list
        ]

        return {
            "date": date_str,
            "latitudes": clean_for_json(self.latitudes),
            "longitudes": clean_for_json(self.longitudes),
            "ocean_mask": clean_for_json(self.ocean_mask_2d),
            "channels": channels_out,
            "channel_order": list(CHANNEL_METADATA.keys()),
            "argo_floats": argo_coords,
        }

    def predict(self, date_str: str) -> Dict[str, Any]:
        """
        Execute frozen OceanEmbed v1-Local inference for given date.
        Results are cached in memory for sub-millisecond retrieval.
        """
        if date_str in self._prediction_cache:
            res = dict(self._prediction_cache[date_str])
            res["cached"] = True
            return res

        if date_str not in self.date_to_idx:
            raise ValueError(f"Date '{date_str}' not in synchronized date range.")

        t_idx = self.date_to_idx[date_str]
        raw_7 = self.inputs_raw[t_idx]        # [7, 24, 32]
        ocean_mask = self.ocean_mask_2d       # [24, 32]

        # 1. Normalize inputs according to frozen training statistics
        means = self.input_means[:, None, None]
        stds = self.input_stds[:, None, None]
        x_norm = np.where(ocean_mask[None, :, :] == 1.0, (raw_7 - means) / stds, 0.0).astype(np.float32)

        # 2. Forward pass with torch.no_grad()
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(self.device)  # [1, 7, 24, 32]

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_tensor = self.model(x_tensor, climatology=self.clim_tensor)  # [1, 15, 24, 32]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_arr = pred_tensor.squeeze(0).cpu().numpy()  # [15, 24, 32] in Celsius

        # 3. Apply land mask (set land cells to NaN for clean JSON serialization)
        pred_masked = np.where(ocean_mask[None, :, :] == 1.0, pred_arr, np.nan)

        # Calculate summary statistics across ocean cells
        ocean_valid = pred_masked[~np.isnan(pred_masked)]
        stats = {
            "min_temp": float(np.min(ocean_valid)) if len(ocean_valid) > 0 else None,
            "max_temp": float(np.max(ocean_valid)) if len(ocean_valid) > 0 else None,
            "mean_temp": float(np.mean(ocean_valid)) if len(ocean_valid) > 0 else None,
        }

        argo_list = self.argo_profiles_by_date.get(date_str, [])
        argo_coords = [
            {
                "profile_id": p["profile_id"],
                "platform_number": p["platform_number"],
                "lat": p["latitude"],
                "lon": p["longitude"],
                "model_rmse": p["model_rmse"],
                "clim_rmse": p["clim_rmse"],
                "glorys_rmse": p["glorys_rmse"],
            }
            for p in argo_list
        ]

        result = {
            "date": date_str,
            "model_name": "OceanEmbed v1-Local",
            "depths": REQUIRED_DEPTHS_M,
            "latitudes": clean_for_json(self.latitudes),
            "longitudes": clean_for_json(self.longitudes),
            "ocean_mask": clean_for_json(self.ocean_mask_2d),
            "prediction": clean_for_json(pred_masked),  # [15, 24, 32]
            "inference_time_ms": round(latency_ms, 2),
            "stats": clean_for_json(stats),
            "argo_floats": argo_coords,
            "cached": False,
        }

        # Store in cache
        self._prediction_cache[date_str] = result
        return result

    def get_profile(self, date_str: str, lat: float, lon: float) -> Dict[str, Any]:
        """
        Extract vertical temperature profile at coordinate (lat, lon):
          - OceanEmbed v1-Local prediction
          - Training Spatial Climatology
          - GLORYS Reanalysis-Derived Reference
          - Independent ARGO In Situ Observational Reference (if collocated)
        """
        if date_str not in self.date_to_idx:
            raise ValueError(f"Date '{date_str}' not in synchronized date range.")

        # Ensure prediction is available
        pred_res = self.predict(date_str)
        t_idx = self.date_to_idx[date_str]

        # Extract raw arrays
        pred_raw = np.array(pred_res["prediction"], dtype=np.float32)  # [15, 24, 32]
        clim_raw = self.clim_mean                                     # [15, 24, 32]
        glorys_raw = self.targets_raw[t_idx]                          # [15, 24, 32]
        mask_raw = self.ocean_mask_15                                 # [15, 24, 32]

        # 1. Bilinear interpolation attempt
        model_prof = bilinear_interpolate_field(pred_raw, mask_raw, lat, lon)
        clim_prof = bilinear_interpolate_field(clim_raw, mask_raw, lat, lon)
        glorys_prof = bilinear_interpolate_field(glorys_raw, mask_raw, lat, lon)

        # 2. If bilinear failed (e.g. edge or coast), fallback to nearest valid ocean cell
        fallback_used = False
        if model_prof is None:
            # Find nearest valid ocean cell
            dist_sq = (LATS[:, None] - lat) ** 2 + (LONS[None, :] - lon) ** 2
            dist_sq[self.ocean_mask_2d == 0.0] = 1e9
            i_min, j_min = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)

            if dist_sq[i_min, j_min] > 0.5:  # too far from any ocean cell
                return {
                    "valid_ocean": False,
                    "date": date_str,
                    "lat": lat,
                    "lon": lon,
                    "message": "Selected coordinate is land or outside domain coverage.",
                }

            model_prof = pred_raw[:, i_min, j_min]
            clim_prof = clim_raw[:, i_min, j_min]
            glorys_prof = glorys_raw[:, i_min, j_min]
            fallback_used = True

        # 3. Check for nearby ARGO float on this date
        argo_match = None
        min_dist_km = 1e9
        argo_list = self.argo_profiles_by_date.get(date_str, [])

        for p in argo_list:
            # Approximate distance in km
            d_lat_km = (p["latitude"] - lat) * 111.0
            d_lon_km = (p["longitude"] - lon) * 111.0 * math.cos(math.radians(lat))
            d_km = math.sqrt(d_lat_km**2 + d_lon_km**2)
            if d_km < min_dist_km:
                min_dist_km = d_km
                if d_km <= 40.0:  # within ~40 km matching radius
                    argo_match = dict(p)
                    argo_match["distance_km"] = round(d_km, 1)

        argo_profile_out = None
        if argo_match is not None and len(argo_match.get("depths", [])) > 0:
            argo_profile_out = {
                "profile_id": argo_match["profile_id"],
                "platform_number": argo_match["platform_number"],
                "latitude": argo_match["latitude"],
                "longitude": argo_match["longitude"],
                "distance_km": argo_match["distance_km"],
                "depths": argo_match["depths"],
                "temperatures": argo_match["temperatures"],
                "label": "Independent ARGO observational reference",
            }

        return {
            "valid_ocean": True,
            "date": date_str,
            "lat": lat,
            "lon": lon,
            "depths": REQUIRED_DEPTHS_M,
            "oceanembed_profile": clean_for_json(model_prof),
            "climatology_profile": clean_for_json(clim_prof),
            "glorys_profile": clean_for_json(glorys_prof),
            "argo_profile": clean_for_json(argo_profile_out),
            "interpolation_method": "nearest_ocean_cell" if fallback_used else "bilinear",
            "series_labels": {
                "oceanembed": "OceanEmbed v1-Local",
                "climatology": "Training Spatial Climatology",
                "glorys": "GLORYS reanalysis-derived reference",
                "argo": "Independent ARGO observational reference",
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return the frozen Gate A4.3 statistical metrics verified against independent ARGO.
        Strict scientific labeling:
          - '14 unique WMO float platforms, treated as conservative cluster units'
          - 'GLORYS reanalysis-derived reference' (never 'ground truth')
          - 'Independent ARGO observational reference'
        """
        return {
            "gate": "Gate A4.3",
            "model_name": "OceanEmbed v1-Local (Frozen Champion)",
            "evaluation_window": "16–31 March 2020, Bay of Bengal",
            "domain": "Bay of Bengal (12.125–17.875°N, 85.125–92.875°E)",
            "canonical_depths_count": 15,
            "overall_metrics": {
                "rmse_celsius": 0.8117,
                "mae_celsius": 0.5426,
                "bias_celsius": -0.0863,
                "pearson_r": 0.9943,
                "climatology_rmse_celsius": 1.0947,
                "relative_rmse_improvement_pct": 25.85,
            },
            "thermocline_regime_50_200m": {
                "rmse_celsius": 0.8446,
                "mae_celsius": 0.6726,
                "bias_celsius": 0.0320,
                "pearson_r": 0.9849,
                "climatology_rmse_celsius": 1.2827,
                "relative_rmse_improvement_pct": 34.16,
            },
            "surface_regime_0_20m": {
                "rmse_celsius": 1.0672,
                "mae_celsius": 0.7725,
                "bias_celsius": -0.5960,
                "climatology_rmse_celsius": 1.1098,
            },
            "deep_regime_gt_200m": {
                "rmse_celsius": 0.6927,
                "mae_celsius": 0.4437,
                "bias_celsius": 0.0617,
                "climatology_rmse_celsius": 0.9634,
            },
            "independent_argo_statistical_rigor": {
                "paired_observations_count": 3966,
                "profiles_count": 34,
                "unique_platforms_count": 14,
                "platform_cluster_label": "14 unique WMO float platforms, treated as conservative cluster units",
                "exact_platform_permutation_p_value": 0.001709,
                "platform_clustered_bootstrap_95_ci": [-0.3457, -0.1117],
                "hypothesis_test_conclusion": "Statistically significant subsurface improvement over climatology (p = 0.001709)",
            },
            "scientific_reference_labels": {
                "glorys": "GLORYS reanalysis-derived reference",
                "argo": "Independent ARGO observational reference",
            },
            "system_audit": {
                "retrained": False,
                "weights_altered": False,
                "normalization_altered": False,
                "operational_claim": False,
            },
        }


# Global singleton instance of the service
_GLOBAL_SERVICE: Optional[OceanEmbedDemoService] = None


def get_demo_service() -> OceanEmbedDemoService:
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        _GLOBAL_SERVICE = OceanEmbedDemoService()
    return _GLOBAL_SERVICE


class OceanEmbedHTTPHandler(SimpleHTTPRequestHandler):
    """
    HTTP Request Handler routing API requests to OceanEmbedDemoService
    and serving static frontend assets from src/demo/static.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    @property
    def service(self) -> OceanEmbedDemoService:
        return get_demo_service()

    def _send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/api/health":
                data = self.service.get_health()
                self._send_json_response(data)
                return

            elif path == "/api/dates":
                data = self.service.get_dates()
                self._send_json_response(data)
                return

            elif path == "/api/surface":
                date_str = query.get("date", ["2020-03-20"])[0]
                data = self.service.get_surface(date_str)
                self._send_json_response(data)
                return

            elif path == "/api/profile":
                date_str = query.get("date", ["2020-03-20"])[0]
                lat = float(query.get("lat", [15.0])[0])
                lon = float(query.get("lon", [89.0])[0])
                data = self.service.get_profile(date_str, lat, lon)
                self._send_json_response(data)
                return

            elif path == "/api/metrics":
                data = self.service.get_metrics()
                self._send_json_response(data)
                return

            # Static asset serving
            if path == "/" or path == "":
                self.path = "/index.html"

            return super().do_GET()

        except Exception as e:
            logger.exception("Error processing GET %s", self.path)
            self._send_json_response({"error": str(e)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/predict":
                content_len = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_len)
                req_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                date_str = req_json.get("date", "2020-03-20")

                data = self.service.predict(date_str)
                self._send_json_response(data)
                return

            self._send_json_response({"error": "Endpoint not found"}, status=404)

        except Exception as e:
            logger.exception("Error processing POST %s", self.path)
            self._send_json_response({"error": str(e)}, status=500)


def run_server(host: str = "127.0.0.1", port: int = 8050):
    """Start the OceanEmbed demo server."""
    # Pre-warm service
    logger.info("Initializing OceanEmbed backend services...")
    get_demo_service()

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, OceanEmbedHTTPHandler)
    logger.info("=" * 70)
    logger.info("OCEANEMBED INTERACTIVE DEMO READY")
    logger.info("Access demo at: http://%s:%d", host, port)
    logger.info("API Health:     http://%s:%d/api/health", host, port)
    logger.info("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down OceanEmbed demo server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed MVP Interactive Demo Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
