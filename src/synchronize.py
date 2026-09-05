"""
OceanEmbed — Gate A1.5 Canonical 7-Channel Synchronization Engine
=================================================================
Synchronizes the 7 satellite observation channels and the 15-depth GLORYS target
to the canonical 0.25° cell-centered grid (24 x 32) over the Bay of Bengal pilot box
(12.0–18.0°N, 85.0–93.0°E) for the 30-day January 2020 window (2020-01-01 to 2020-01-30).

Channels:
  0: SST (OSTIA, Kelvin -> Celsius, regridded 0.05° -> 0.25°)
  1: SSS (Multi-Obs SMAP/SMOS, depth squeezed, regridded 0.125° -> 0.25°)
  2: SSH (DUACS ADT, natively 0.25° cell-center aligned)
  3: Surface Current U (OSCAR v2.0, transposed (time, lon, lat) -> (time, lat, lon), regridded to cell centers)
  4: Surface Current V (OSCAR v2.0, transposed (time, lon, lat) -> (time, lat, lon), regridded to cell centers)
  5: Surface Wind U (CCMP v3.1, daily vector mean, natively 0.25° cell-center aligned)
  6: Surface Wind V (CCMP v3.1, daily vector mean, natively 0.25° cell-center aligned)

Target:
  GLORYS thetao at 15 canonical depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m),
  spatially regridded from 0.0833° to the canonical 0.25° grid.

Outputs:
  - Dataset/gate_a1_pilot/synchronized/oceanembed_inputs_7ch_30d.nc
  - Dataset/gate_a1_pilot/synchronized/oceanembed_target_15depth_30d.nc
  - Dataset/gate_a1_pilot/synchronized/synchronization_metadata.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from src.constants import (
    REQUIRED_DEPTHS_M,
    INPUT_CHANNELS,
    CANONICAL_SURFACE_VARIABLES,
    CHANNEL_NAME_MAP,
    GATE_A1_PILOT_BOX,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("Dataset/gate_a1_pilot/synchronized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pilot Temporal Coordinates
PILOT_DATES = pd.date_range("2020-01-01", "2020-01-30", freq="D")
NUM_DAYS = len(PILOT_DATES)

# Canonical Spatial Coordinates (24 x 32 cell-center 0.25° grid)
CANONICAL_LATS = np.array(GATE_A1_PILOT_BOX["lat_centers"], dtype=np.float32)
CANONICAL_LONS = np.array(GATE_A1_PILOT_BOX["lon_centers"], dtype=np.float32)
GRID_H = len(CANONICAL_LATS)  # 24
GRID_W = len(CANONICAL_LONS)  # 32

# Training Split for Leakage-Free Normalization (first 20 days: Jan 1 to Jan 20)
TRAIN_DAYS = 20
VAL_TEST_DAYS = 10


def regrid_2d_field(da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    """
    Bilinearly interpolate a DataArray with (latitude, longitude) dimensions to target coordinates.
    Standardizes coordinate names to 'latitude' and 'longitude'.
    """
    # Rename coords if necessary
    rename_dict = {}
    for c in da.coords:
        if c in ["lat", "LAT", "Latitude"]:
            rename_dict[c] = "latitude"
        elif c in ["lon", "LON", "Longitude"]:
            rename_dict[c] = "longitude"
    if rename_dict:
        da = da.rename(rename_dict)

    regridded = da.interp(latitude=target_lat, longitude=target_lon, method="linear")
    return regridded


def process_channel_0_sst(path: Path) -> xr.DataArray:
    """OSTIA SST: Kelvin -> Celsius, regrid 0.05° -> 0.25°."""
    logger.info("Processing Channel 0: OSTIA SST...")
    ds = xr.open_dataset(path)
    sst_k = ds["analysed_sst"]
    # Unit conversion Kelvin -> Celsius
    sst_c = sst_k - 273.15
    sst_c.attrs["units"] = "degrees_C"
    sst_c.attrs["long_name"] = "Sea Surface Temperature (Celsius)"

    regridded = regrid_2d_field(sst_c, CANONICAL_LATS, CANONICAL_LONS)
    ds.close()
    return regridded


def process_channel_1_sss(path: Path) -> xr.DataArray:
    """Multi-Obs SSS: squeeze depth, regrid 0.125° -> 0.25°."""
    logger.info("Processing Channel 1: Multi-Obs SMAP/SMOS SSS...")
    ds = xr.open_dataset(path)
    sss = ds["sos"]
    if "depth" in sss.dims:
        sss = sss.squeeze("depth", drop=True)
    sss.attrs["units"] = "PSU"
    sss.attrs["long_name"] = "Sea Surface Salinity (PSU)"

    regridded = regrid_2d_field(sss, CANONICAL_LATS, CANONICAL_LONS)
    ds.close()
    return regridded


def process_channel_2_ssh(path: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    """DUACS Altimetry: select ADT as canonical SSH, preserve SLA."""
    logger.info("Processing Channel 2: DUACS Altimetry (ADT canonical SSH)...")
    ds = xr.open_dataset(path)
    adt = ds["adt"]
    sla = ds["sla"]
    adt.attrs["units"] = "m"
    adt.attrs["long_name"] = "Absolute Dynamic Topography (Sea Surface Height, m)"
    sla.attrs["units"] = "m"
    sla.attrs["long_name"] = "Sea Level Anomaly (m)"

    # Verify already on canonical coordinates
    adt_aligned = regrid_2d_field(adt, CANONICAL_LATS, CANONICAL_LONS)
    sla_aligned = regrid_2d_field(sla, CANONICAL_LATS, CANONICAL_LONS)
    ds.close()
    return adt_aligned, sla_aligned


def process_channels_3_4_currents(path: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    """OSCAR Surface Currents: transpose (time, lon, lat) -> (time, lat, lon), regrid to cell centers."""
    logger.info("Processing Channels 3 & 4: OSCAR Surface Currents (U and V)...")
    ds = xr.open_dataset(path)
    u = ds["u"]
    v = ds["v"]

    # Transpose dimensions if lon precedes lat
    if u.dims == ("time", "lon", "lat"):
        u = u.transpose("time", "lat", "lon")
        v = v.transpose("time", "lat", "lon")

    u.attrs["units"] = "m s-1"
    u.attrs["long_name"] = "Zonal surface current velocity (m/s)"
    v.attrs["units"] = "m s-1"
    v.attrs["long_name"] = "Meridional surface current velocity (m/s)"

    # Interpolate from 25 x 33 node-registered grid to 24 x 32 cell centers
    u_regridded = regrid_2d_field(u, CANONICAL_LATS, CANONICAL_LONS)
    v_regridded = regrid_2d_field(v, CANONICAL_LATS, CANONICAL_LONS)
    ds.close()
    return u_regridded, v_regridded


def process_channels_5_6_winds(path: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    """CCMP Winds: daily vector-averaged U and V, align to canonical coordinates."""
    logger.info("Processing Channels 5 & 6: CCMP 10m Vector Winds (U and V)...")
    ds = xr.open_dataset(path)
    uwnd = ds["uwnd"]
    vwnd = ds["vwnd"]
    uwnd.attrs["units"] = "m s-1"
    uwnd.attrs["long_name"] = "10m zonal wind component (m/s)"
    vwnd.attrs["units"] = "m s-1"
    vwnd.attrs["long_name"] = "10m meridional wind component (m/s)"

    u_aligned = regrid_2d_field(uwnd, CANONICAL_LATS, CANONICAL_LONS)
    v_aligned = regrid_2d_field(vwnd, CANONICAL_LATS, CANONICAL_LONS)
    ds.close()
    return u_aligned, v_aligned


def process_target_glorys_thetao(path: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    """GLORYS target: spatially regrid 15 canonical depths from 0.0833° to 0.25°."""
    logger.info("Processing GLORYS 15-depth target (thetao)...")
    ds = xr.open_dataset(path)
    thetao = ds["thetao"]
    mask_native = ds["mask_thetao"] if "mask_thetao" in ds else xr.where(np.isnan(thetao), 0.0, 1.0)

    # Verify depth levels
    depths = thetao.depth.values
    if not np.allclose(depths, REQUIRED_DEPTHS_M):
        raise ValueError(f"Target depths {depths} do not match required depths {REQUIRED_DEPTHS_M}")

    # Spatial regridding preserving time and depth dimensions
    thetao_regridded = regrid_2d_field(thetao, CANONICAL_LATS, CANONICAL_LONS)
    mask_regridded = regrid_2d_field(mask_native, CANONICAL_LATS, CANONICAL_LONS)
    # Threshold regridded mask at 0.5 for clean binary ocean definition
    binary_mask = xr.where((mask_regridded >= 0.5) & (~np.isnan(thetao_regridded)), 1.0, 0.0)

    thetao_regridded.attrs["units"] = "degrees_C"
    thetao_regridded.attrs["long_name"] = "GLORYS Potential Temperature (Celsius)"
    ds.close()
    return thetao_regridded, binary_mask


def compute_leakage_free_statistics(
    x_array: np.ndarray,
    mask_x: np.ndarray,
    y_array: np.ndarray,
    mask_y: np.ndarray,
    train_slice: slice,
) -> Dict[str, Any]:
    """
    Compute normalization mean and standard deviation strictly using the training period.
    """
    logger.info(f"Computing leakage-free normalization statistics on training window (first {TRAIN_DAYS} days)...")
    stats = {
        "training_window": {
            "start_day": 0,
            "end_day": TRAIN_DAYS - 1,
            "start_date": str(PILOT_DATES[0])[:10],
            "end_date": str(PILOT_DATES[TRAIN_DAYS - 1])[:10],
            "total_train_days": TRAIN_DAYS,
        },
        "evaluation_window": {
            "start_day": TRAIN_DAYS,
            "end_day": NUM_DAYS - 1,
            "start_date": str(PILOT_DATES[TRAIN_DAYS])[:10],
            "end_date": str(PILOT_DATES[-1])[:10],
            "total_val_test_days": VAL_TEST_DAYS,
        },
        "input_channels": {},
        "target_depths": {},
    }

    # Input channel statistics
    for c_idx, c_name in enumerate(CANONICAL_SURFACE_VARIABLES):
        train_vals = x_array[train_slice, c_idx, :, :]
        train_mask = mask_x[train_slice, c_idx, :, :]
        valid_points = train_vals[train_mask == 1.0]

        all_vals = x_array[:, c_idx, :, :]
        all_mask = mask_x[:, c_idx, :, :]
        all_valid_points = all_vals[all_mask == 1.0]

        train_mean = float(np.nanmean(valid_points))
        train_std = float(np.nanstd(valid_points))
        all_mean = float(np.nanmean(all_valid_points))
        all_std = float(np.nanstd(all_valid_points))

        stats["input_channels"][c_name] = {
            "channel_index": c_idx,
            "train_mean": round(train_mean, 5),
            "train_std": round(train_std, 5),
            "full_sample_mean": round(all_mean, 5),
            "full_sample_std": round(all_std, 5),
            "diff_mean_pct": round(abs(train_mean - all_mean) / (abs(all_mean) + 1e-6) * 100, 3),
            "units": (
                "degrees_C" if c_idx == 0
                else "PSU" if c_idx == 1
                else "m" if c_idx == 2
                else "m s-1"
            ),
        }

    # Target depth statistics
    for d_idx, depth_m in enumerate(REQUIRED_DEPTHS_M):
        train_tgt = y_array[train_slice, d_idx, :, :]
        train_tgt_mask = mask_y[train_slice, d_idx, :, :]
        valid_tgt = train_tgt[train_tgt_mask == 1.0]

        all_tgt = y_array[:, d_idx, :, :]
        all_tgt_mask = mask_y[:, d_idx, :, :]
        all_valid_tgt = all_tgt[all_tgt_mask == 1.0]

        train_mean = float(np.nanmean(valid_tgt))
        train_std = float(np.nanstd(valid_tgt))
        all_mean = float(np.nanmean(all_valid_tgt))
        all_std = float(np.nanstd(all_valid_tgt))

        stats["target_depths"][f"depth_{depth_m}m"] = {
            "depth_index": d_idx,
            "depth_meters": depth_m,
            "train_mean": round(train_mean, 5),
            "train_std": round(train_std, 5),
            "full_sample_mean": round(all_mean, 5),
            "full_sample_std": round(all_std, 5),
            "units": "degrees_C",
        }

    return stats


def synchronize_all() -> Tuple[Path, Path, Path]:
    """Execute complete synchronization pipeline and export canonical NetCDF products."""
    sat_dir = Path("Dataset/gate_a1_pilot/satellite")
    target_path = Path("Dataset/gate_a1_pilot/glorys_pilot_30d_15depths_preprocessed.nc")

    # 1. Process all 7 channels
    da_sst = process_channel_0_sst(sat_dir / "ostia_sst_pilot_30d.nc")
    da_sss = process_channel_1_sss(sat_dir / "multi_obs_sss_pilot_30d.nc")
    da_adt, da_sla = process_channel_2_ssh(sat_dir / "duacs_ssh_pilot_30d.nc")
    da_u_curr, da_v_curr = process_channels_3_4_currents(sat_dir / "oscar_currents_pilot_30d.nc")
    da_u_wind, da_v_wind = process_channels_5_6_winds(sat_dir / "ccmp_wind_pilot_30d.nc")

    # 2. Process Target
    da_thetao, da_target_mask = process_target_glorys_thetao(target_path)

    # 3. Stack into Canonical Input Tensor X: [time, channel, latitude, longitude]
    channels = [
        da_sst,
        da_sss,
        da_adt,
        da_u_curr,
        da_v_curr,
        da_u_wind,
        da_v_wind,
    ]

    # Convert each DataArray to aligned numpy array [30, 24, 32]
    x_list = []
    for da in channels:
        arr = da.values
        if arr.shape != (NUM_DAYS, GRID_H, GRID_W):
            raise ValueError(f"Channel array shape {arr.shape} != expected {(NUM_DAYS, GRID_H, GRID_W)}")
        x_list.append(arr)

    # Stack along channel axis: [30, 7, 24, 32]
    X = np.stack(x_list, axis=1).astype(np.float32)
    # Mask: 1.0 where finite, 0.0 where NaN
    mask_X = np.where(~np.isnan(X), 1.0, 0.0).astype(np.float32)

    # Target Tensor Y: [30, 15, 24, 32]
    Y = da_thetao.values.astype(np.float32)
    mask_Y = da_target_mask.values.astype(np.float32)

    # 2D Common Ocean Mask: valid across all 7 channels and target surface layer over all 30 days
    common_surface_valid = np.all(mask_X == 1.0, axis=(0, 1)) & np.all(mask_Y[:, 0, :, :] == 1.0, axis=0)
    common_ocean_mask_2d = np.where(common_surface_valid, 1.0, 0.0).astype(np.float32)
    valid_ocean_points = int(np.sum(common_ocean_mask_2d))
    total_grid_points = GRID_H * GRID_W
    valid_ocean_pct = round((valid_ocean_points / total_grid_points) * 100, 2)
    logger.info(f"Canonical 2D Ocean Mask: {valid_ocean_points}/{total_grid_points} cells ({valid_ocean_pct}%)")

    # 4. Compute Leakage-Free Normalization Statistics
    train_slice = slice(0, TRAIN_DAYS)
    stats = compute_leakage_free_statistics(X, mask_X, Y, mask_Y, train_slice)

    # 5. Build Inputs NetCDF
    logger.info("Building inputs NetCDF (Dataset/gate_a1_pilot/synchronized/oceanembed_inputs_7ch_30d.nc)...")
    time_coords = PILOT_DATES
    channel_indices = list(range(INPUT_CHANNELS))

    ds_inputs = xr.Dataset(
        data_vars={
            "inputs": (("time", "channel", "latitude", "longitude"), X),
            "mask_inputs": (("time", "channel", "latitude", "longitude"), mask_X),
            "ocean_mask_2d": (("latitude", "longitude"), common_ocean_mask_2d),
            "sst": (("time", "latitude", "longitude"), X[:, 0, :, :]),
            "sss": (("time", "latitude", "longitude"), X[:, 1, :, :]),
            "ssh_adt": (("time", "latitude", "longitude"), X[:, 2, :, :]),
            "ssh_sla": (("time", "latitude", "longitude"), da_sla.values.astype(np.float32)),
            "u_current": (("time", "latitude", "longitude"), X[:, 3, :, :]),
            "v_current": (("time", "latitude", "longitude"), X[:, 4, :, :]),
            "u_wind": (("time", "latitude", "longitude"), X[:, 5, :, :]),
            "v_wind": (("time", "latitude", "longitude"), X[:, 6, :, :]),
        },
        coords={
            "time": time_coords,
            "channel": channel_indices,
            "channel_name": ("channel", CANONICAL_SURFACE_VARIABLES),
            "latitude": CANONICAL_LATS,
            "longitude": CANONICAL_LONS,
        },
        attrs={
            "title": "OceanEmbed Synchronized 7-Channel Satellite Surface Inputs",
            "pilot_domain": "Bay of Bengal (12-18N, 85-93E)",
            "grid_resolution": 0.25,
            "grid_registration": "cell_center",
            "grid_dimensions": f"{GRID_H} x {GRID_W}",
            "time_range": f"{PILOT_DATES[0].strftime('%Y-%m-%d')} to {PILOT_DATES[-1].strftime('%Y-%m-%d')}",
            "channel_ordering": "0:sst, 1:sss, 2:ssh_adt, 3:u_current, 4:v_current, 5:u_wind, 6:v_wind",
            "units": "sst: degC, sss: PSU, ssh_adt: m, u_current: m/s, v_current: m/s, u_wind: m/s, v_wind: m/s",
        },
    )

    inputs_path = OUTPUT_DIR / "oceanembed_inputs_7ch_30d.nc"
    ds_inputs.to_netcdf(inputs_path)
    logger.info(f"Saved synchronized inputs to {inputs_path} ({inputs_path.stat().st_size / 1024:.1f} KB)")
    ds_inputs.close()

    # 6. Build Target NetCDF
    logger.info("Building target NetCDF (Dataset/gate_a1_pilot/synchronized/oceanembed_target_15depth_30d.nc)...")
    ds_target = xr.Dataset(
        data_vars={
            "target_thetao": (("time", "depth", "latitude", "longitude"), Y),
            "mask_target": (("time", "depth", "latitude", "longitude"), mask_Y),
            "ocean_mask_2d": (("latitude", "longitude"), common_ocean_mask_2d),
        },
        coords={
            "time": time_coords,
            "depth": np.array(REQUIRED_DEPTHS_M, dtype=np.float32),
            "latitude": CANONICAL_LATS,
            "longitude": CANONICAL_LONS,
        },
        attrs={
            "title": "OceanEmbed Synchronized 15-Depth GLORYS Potential Temperature Target",
            "pilot_domain": "Bay of Bengal (12-18N, 85-93E)",
            "grid_resolution": 0.25,
            "grid_registration": "cell_center",
            "grid_dimensions": f"{GRID_H} x {GRID_W}",
            "time_range": f"{PILOT_DATES[0].strftime('%Y-%m-%d')} to {PILOT_DATES[-1].strftime('%Y-%m-%d')}",
            "canonical_depths_m": str(REQUIRED_DEPTHS_M),
            "units": "degrees_C",
            "source": "GLORYS12V1 Daily Reanalysis (vertically interpolated to 15 canonical depths, regridded to 0.25 deg)",
        },
    )

    target_out_path = OUTPUT_DIR / "oceanembed_target_15depth_30d.nc"
    ds_target.to_netcdf(target_out_path)
    logger.info(f"Saved synchronized target to {target_out_path} ({target_out_path.stat().st_size / 1024:.1f} KB)")
    ds_target.close()

    # 7. Write Full Metadata JSON
    metadata = {
        "gate": "A1.5",
        "title": "OceanEmbed Canonical 7-Channel & 15-Depth Synchronized Pilot Dataset",
        "spatial_grid": {
            "domain": "Bay of Bengal",
            "registration": "cell_center",
            "resolution_deg": 0.25,
            "lat_min_bound": 12.0,
            "lat_max_bound": 18.0,
            "lon_min_bound": 85.0,
            "lon_max_bound": 93.0,
            "grid_height": GRID_H,
            "grid_width": GRID_W,
            "latitude_centers": CANONICAL_LATS.tolist(),
            "longitude_centers": CANONICAL_LONS.tolist(),
            "valid_ocean_cells": valid_ocean_points,
            "total_cells": total_grid_points,
            "valid_ocean_pct": valid_ocean_pct,
        },
        "temporal_alignment": {
            "start_date": str(PILOT_DATES[0])[:10],
            "end_date": str(PILOT_DATES[-1])[:10],
            "num_days": NUM_DAYS,
            "cadence": "1D",
            "duplicate_dates": False,
        },
        "input_tensor": {
            "file": str(inputs_path),
            "dimensions": ["time", "channel", "latitude", "longitude"],
            "shape": [NUM_DAYS, INPUT_CHANNELS, GRID_H, GRID_W],
            "channels": [
                {
                    "index": 0,
                    "name": "sst",
                    "long_name": "Sea Surface Temperature",
                    "source": "OSTIA L4 (Met Office)",
                    "native_res": "0.05 deg",
                    "units": "degrees_C",
                    "transformations": ["Kelvin to Celsius (T - 273.15)", "Bilinear regridding to 0.25 deg"],
                },
                {
                    "index": 1,
                    "name": "sss",
                    "long_name": "Sea Surface Salinity",
                    "source": "Multi-Obs SMAP/SMOS L4",
                    "native_res": "0.125 deg",
                    "units": "PSU",
                    "transformations": ["Squeezed depth dimension", "Bilinear regridding to 0.25 deg"],
                },
                {
                    "index": 2,
                    "name": "ssh_adt",
                    "long_name": "Absolute Dynamic Topography (SSH)",
                    "source": "DUACS Two-Sat L4",
                    "native_res": "0.25 deg cell-center",
                    "units": "m",
                    "transformations": ["Direct coordinate alignment"],
                },
                {
                    "index": 3,
                    "name": "u_current",
                    "long_name": "Zonal Surface Current Velocity",
                    "source": "OSCAR v2.0 L4",
                    "native_res": "0.25 deg node-registered",
                    "units": "m s-1",
                    "transformations": ["Transposed (time, lon, lat) -> (time, lat, lon)", "Interior bilinear interpolation to cell-center"],
                },
                {
                    "index": 4,
                    "name": "v_current",
                    "long_name": "Meridional Surface Current Velocity",
                    "source": "OSCAR v2.0 L4",
                    "native_res": "0.25 deg node-registered",
                    "units": "m s-1",
                    "transformations": ["Transposed (time, lon, lat) -> (time, lat, lon)", "Interior bilinear interpolation to cell-center"],
                },
                {
                    "index": 5,
                    "name": "u_wind",
                    "long_name": "10m Zonal Wind Component",
                    "source": "CCMP v3.1 L4",
                    "native_res": "0.25 deg cell-center",
                    "units": "m s-1",
                    "transformations": ["Daily vector-averaged from 6-hourly", "Direct coordinate alignment"],
                },
                {
                    "index": 6,
                    "name": "v_wind",
                    "long_name": "10m Meridional Wind Component",
                    "source": "CCMP v3.1 L4",
                    "native_res": "0.25 deg cell-center",
                    "units": "m s-1",
                    "transformations": ["Daily vector-averaged from 6-hourly", "Direct coordinate alignment"],
                },
            ],
        },
        "target_tensor": {
            "file": str(target_out_path),
            "variable": "thetao",
            "long_name": "GLORYS Potential Temperature",
            "source": "GLORYS12V1 Daily Reanalysis",
            "dimensions": ["time", "depth", "latitude", "longitude"],
            "shape": [NUM_DAYS, len(REQUIRED_DEPTHS_M), GRID_H, GRID_W],
            "depth_levels_m": REQUIRED_DEPTHS_M,
            "units": "degrees_C",
            "transformations": ["Vertical interpolation to 15 canonical depths", "Spatial bilinear regridding 0.0833 deg to 0.25 deg"],
        },
        "leakage_free_normalization": stats,
    }

    metadata_path = OUTPUT_DIR / "synchronization_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved synchronization metadata to {metadata_path}")

    return inputs_path, target_out_path, metadata_path


if __name__ == "__main__":
    synchronize_all()
