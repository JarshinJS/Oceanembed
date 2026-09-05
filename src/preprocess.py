"""
OceanEmbed — Vertical Interpolation & Preprocessing Pipeline
===========================================================
Implements PS-aligned vertical interpolation from native Copernicus GLORYS12V1
levels (36 levels, 0.494m to 1062.44m) to the exact 15 canonical SIH26066 target depths:
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m

Key Scientific & Engineering Rules:
  1. Coordinate Validation: Native depths must be monotonically increasing,
     uppermost level must be shallow (~0.494 m), deepest level must be >= 1000 m.
  2. 0 m Target Handling: Directly mapped to the nearest uppermost surface native level
     (index 0, approx 0.494 m). Upward extrapolation to 0.0 m is strictly forbidden.
  3. 5–700 m Targets: Vertically interpolated between immediate surrounding native levels.
  4. 1000 m Target: Interpolated between native levels 34 (902.3393 m) and 35 (1062.4399 m).
     Explicitly verified: 902.3393 m < 1000 m < 1062.4399 m.
  5. Mask & NaN Preservation: Any pixel with missing data or bathymetry cutoff in either
     bracketing level remains NaN. An explicit ocean validity mask is generated.
  6. Ocean coverage is accurately calculated per depth and never falsely claimed as 100%.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from src.constants import (
    REQUIRED_DEPTHS_M,
    OUTPUT_DEPTHS,
    GLORYS_SURFACE_LEVEL_APPROX_M,
)

logger = logging.getLogger(__name__)


def validate_native_depths(
    native_depths: np.ndarray,
    target_depths: List[int] = REQUIRED_DEPTHS_M,
) -> Dict[str, Union[float, int, bool, List[float]]]:
    """Validate that native GLORYS depths satisfy all prerequisites for interpolation.

    Prerequisites:
      - Array is 1D, non-empty, and strictly monotonically increasing.
      - Uppermost native depth is shallow (<= 2.0 m, typically ~0.494 m).
      - Deepest native depth is strictly >= 1000.0 m.
      - 1000 m target is strictly bracketed: z_lower < 1000.0 < z_upper.
    """
    depths = np.asarray(native_depths, dtype=np.float64)
    if depths.ndim != 1 or len(depths) == 0:
        raise ValueError(f"native_depths must be a non-empty 1D array. Got shape {depths.shape}")

    # Monotonicity check
    diffs = np.diff(depths)
    if np.any(diffs <= 0):
        raise ValueError("native_depths must be strictly monotonically increasing.")

    uppermost = float(depths[0])
    deepest = float(depths[-1])

    if uppermost > 2.0:
        raise ValueError(
            f"Uppermost native depth ({uppermost:.4f} m) is too deep for surface level mapping. "
            f"Expected <= 2.0 m (~0.494 m)."
        )

    if deepest < 1000.0:
        raise ValueError(
            f"Deepest native depth ({deepest:.4f} m) is less than required 1000 m target. "
            f"Extrapolation beyond deepest level is strictly forbidden."
        )

    # Check 1000 m bracketing specifically
    idx_below = np.where(depths <= 1000.0)[0][-1]
    idx_above = np.where(depths >= 1000.0)[0][0]
    z_below = float(depths[idx_below])
    z_above = float(depths[idx_above])

    if not (z_below < 1000.0 < z_above):
        # Unless exact hit
        if z_below != 1000.0 and z_above != 1000.0:
            raise ValueError(
                f"1000 m target depth is not strictly bracketed! Found z_below={z_below}, z_above={z_above}"
            )

    return {
        "valid": True,
        "n_levels": len(depths),
        "uppermost_m": uppermost,
        "deepest_m": deepest,
        "bracket_1000m_lower_idx": int(idx_below),
        "bracket_1000m_lower_m": z_below,
        "bracket_1000m_upper_idx": int(idx_above),
        "bracket_1000m_upper_m": z_above,
    }


def compute_interpolation_weights(
    native_depths: np.ndarray,
    target_depths: List[int] = REQUIRED_DEPTHS_M,
) -> List[Dict[str, Union[int, float, str]]]:
    """Compute exact bracketing level indices and linear interpolation weights for each target depth.

    For target depth 0 m:
      Mapped to nearest shallow native level (index 0). No upward extrapolation.
    For target depths > 0 m:
      Interpolated between z0 <= z <= z1 with alpha = (z - z0) / (z1 - z0).
      Formula: V(z) = (1 - alpha) * V(z0) + alpha * V(z1)
    """
    depths = np.asarray(native_depths, dtype=np.float64)
    weights = []

    for tz in target_depths:
        if tz == 0:
            weights.append({
                "target_depth_m": 0.0,
                "lower_idx": 0,
                "upper_idx": 0,
                "lower_depth_m": float(depths[0]),
                "upper_depth_m": float(depths[0]),
                "alpha": 0.0,
                "method": "nearest_surface_level",
            })
        else:
            idx_lower = int(np.where(depths <= tz)[0][-1])
            idx_upper = int(np.where(depths >= tz)[0][0])
            z0 = float(depths[idx_lower])
            z1 = float(depths[idx_upper])
            if z1 == z0:
                alpha = 0.0
            else:
                alpha = float((tz - z0) / (z1 - z0))

            weights.append({
                "target_depth_m": float(tz),
                "lower_idx": idx_lower,
                "upper_idx": idx_upper,
                "lower_depth_m": z0,
                "upper_depth_m": z1,
                "alpha": alpha,
                "method": "bracketed_linear_interpolation",
            })

    return weights


def interpolate_field_to_target_depths(
    data: np.ndarray,
    weights: List[Dict[str, Union[int, float, str]]],
    depth_axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a multi-dimensional array along its depth axis to target depths.

    Parameters:
      data: Input array (e.g. shape (time, native_depths, lat, lon)).
      weights: Precomputed interpolation weights from compute_interpolation_weights().
      depth_axis: Axis index corresponding to depth (default 1).

    Returns:
      interpolated_data: Array with depth dimension replaced by target depths (float32).
      ocean_mask: Boolean mask indicating valid ocean points (~isnan(interpolated_data)).
    """
    n_targets = len(weights)
    out_shape = list(data.shape)
    out_shape[depth_axis] = n_targets
    out = np.empty(out_shape, dtype=np.float32)

    # Move depth axis to front for easy slicing
    data_moved = np.moveaxis(data, depth_axis, 0)
    out_moved = np.moveaxis(out, depth_axis, 0)

    for k, w in enumerate(weights):
        if w["method"] == "nearest_surface_level":
            # Assign nearest surface level directly
            out_moved[k] = data_moved[w["lower_idx"]].astype(np.float32)
        else:
            i0 = w["lower_idx"]
            i1 = w["upper_idx"]
            alpha = w["alpha"]
            v0 = data_moved[i0].astype(np.float32)
            if i0 == i1 or alpha == 0.0:
                out_moved[k] = v0
            else:
                v1 = data_moved[i1].astype(np.float32)
                # Standard linear interpolation: (1 - alpha) * v0 + alpha * v1
                # If either v0 or v1 is NaN, the result is NaN naturally.
                out_moved[k] = (1.0 - alpha) * v0 + alpha * v1

    # Move axis back
    out = np.moveaxis(out_moved, 0, depth_axis)
    ocean_mask = ~np.isnan(out)

    return out, ocean_mask


def preprocess_glorys_dataset(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_depths: List[int] = REQUIRED_DEPTHS_M,
) -> Tuple[xr.Dataset, Dict]:
    """Full preprocessing pipeline for Copernicus GLORYS NetCDF files.

    Steps:
      1. Load NetCDF using xarray.
      2. Validate native depths against target depths.
      3. Compute interpolation weights for canonical 15 depths.
      4. Interpolate 3D variables (thetao, so, uo, vo) to 15 depths.
      5. Preserve 2D variable (zos) and surface variables.
      6. Calculate accurate ocean coverage and missing value metrics per depth.
      7. Construct new Dataset with canonical coordinates and metadata.
      8. Optionally save to NetCDF.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input NetCDF file not found: {input_path}")

    logger.info(f"Opening GLORYS dataset: {input_path}")
    ds = xr.open_dataset(input_path)

    # 1. Extract coordinates
    native_depths = ds["depth"].values.astype(np.float64)
    times = ds["time"].values
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    # 2. Validate native depths
    val_info = validate_native_depths(native_depths, target_depths)
    logger.info(
        f"Native depths validated: {val_info['n_levels']} levels, "
        f"surface={val_info['uppermost_m']:.4f}m, "
        f"deepest={val_info['deepest_m']:.4f}m. "
        f"1000m bracket: {val_info['bracket_1000m_lower_m']:.4f}m < 1000m < {val_info['bracket_1000m_upper_m']:.4f}m"
    )

    # 3. Compute weights
    weights = compute_interpolation_weights(native_depths, target_depths)

    # 4. Process variables
    target_depths_arr = np.array(target_depths, dtype=np.float32)
    processed_data_vars = {}
    coverage_report = {}

    # 3D Variables to interpolate
    vars_3d = [v for v in ["thetao", "so", "uo", "vo"] if v in ds.data_vars]
    for var_name in vars_3d:
        logger.info(f"Interpolating 3D variable '{var_name}' to 15 canonical depths...")
        raw_data = ds[var_name].values  # shape: (time, depth, lat, lon)
        interp_data, mask = interpolate_field_to_target_depths(raw_data, weights, depth_axis=1)

        # Store interpolated variable
        var_attrs = dict(ds[var_name].attrs)
        var_attrs["vertical_interpolation"] = (
            "Mapped to 15 canonical SIH26066 depths. Depth 0m uses nearest native surface level (~0.494m). "
            "Depths 5-1000m use bracketed linear vertical interpolation without extrapolation."
        )
        processed_data_vars[var_name] = (
            ["time", "depth", "latitude", "longitude"],
            interp_data,
            var_attrs,
        )

        # Store ocean mask
        mask_attrs = {
            "long_name": f"Ocean validity mask for {var_name}",
            "description": "True (1) where ocean data is valid; False (0) for land or sub-bottom cutoff.",
        }
        processed_data_vars[f"mask_{var_name}"] = (
            ["time", "depth", "latitude", "longitude"],
            mask.astype(np.uint8),
            mask_attrs,
        )

        # Compute per-depth statistics
        depth_stats = []
        for k, tz in enumerate(target_depths):
            slice_data = interp_data[:, k, :, :]
            slice_mask = mask[:, k, :, :]
            total_points = int(slice_data.size)
            valid_points = int(np.count_nonzero(slice_mask))
            nan_points = total_points - valid_points
            cov_pct = float(valid_points / total_points * 100.0)

            valid_vals = slice_data[slice_mask]
            min_val = float(np.min(valid_vals)) if valid_vals.size > 0 else np.nan
            max_val = float(np.max(valid_vals)) if valid_vals.size > 0 else np.nan
            mean_val = float(np.mean(valid_vals)) if valid_vals.size > 0 else np.nan

            depth_stats.append({
                "target_depth_m": tz,
                "valid_coverage_pct": round(cov_pct, 4),
                "valid_points": valid_points,
                "nan_points": nan_points,
                "total_points": total_points,
                "min": round(min_val, 4) if not np.isnan(min_val) else None,
                "max": round(max_val, 4) if not np.isnan(max_val) else None,
                "mean": round(mean_val, 4) if not np.isnan(mean_val) else None,
            })
        coverage_report[var_name] = depth_stats

    # 2D Variables (e.g. zos)
    if "zos" in ds.data_vars:
        logger.info("Processing 2D surface variable 'zos'...")
        zos_data = ds["zos"].values.astype(np.float32)
        zos_mask = ~np.isnan(zos_data)
        zos_attrs = dict(ds["zos"].attrs)
        processed_data_vars["zos"] = (
            ["time", "latitude", "longitude"],
            zos_data,
            zos_attrs,
        )
        processed_data_vars["mask_zos"] = (
            ["time", "latitude", "longitude"],
            zos_mask.astype(np.uint8),
            {"long_name": "Ocean validity mask for zos"},
        )

        valid_pts = int(np.count_nonzero(zos_mask))
        tot_pts = int(zos_data.size)
        valid_vals = zos_data[zos_mask]
        coverage_report["zos"] = {
            "valid_coverage_pct": round(float(valid_pts / tot_pts * 100.0), 4),
            "valid_points": valid_pts,
            "nan_points": tot_pts - valid_pts,
            "total_points": tot_pts,
            "min": round(float(np.min(valid_vals)), 4) if valid_vals.size > 0 else None,
            "max": round(float(np.max(valid_vals)), 4) if valid_vals.size > 0 else None,
            "mean": round(float(np.mean(valid_vals)), 4) if valid_vals.size > 0 else None,
        }

    # 5. Assemble new xarray Dataset
    coords = {
        "time": times,
        "depth": target_depths_arr,
        "latitude": lats,
        "longitude": lons,
    }

    global_attrs = dict(ds.attrs)
    global_attrs.update({
        "title": "OceanEmbed Preprocessed 15-Depth Ocean Physical Fields",
        "vertical_levels_canonical": str(target_depths),
        "vertical_levels_count": len(target_depths),
        "source_native_levels_count": len(native_depths),
        "sih26066_alignment": "Problem Statement SIH26066 Authoritative 15 Target Depths",
        "zero_meter_handling": "Mapped directly to uppermost native level (~0.494 m). Zero upward extrapolation.",
        "thousand_meter_handling": (
            f"Linearly interpolated between native levels {val_info['bracket_1000m_lower_m']:.4f}m "
            f"and {val_info['bracket_1000m_upper_m']:.4f}m. Zero downward extrapolation."
        ),
    })

    out_ds = xr.Dataset(data_vars=processed_data_vars, coords=coords, attrs=global_attrs)

    # 6. Save if output path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving preprocessed dataset to: {output_path}")
        out_ds.to_netcdf(output_path)

    summary_report = {
        "input_file": str(input_path),
        "output_file": str(output_path) if output_path else None,
        "input_dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "output_dimensions": {k: int(v) for k, v in out_ds.sizes.items()},
        "native_depths": [round(float(d), 4) for d in native_depths],
        "target_depths": list(target_depths),
        "validation": val_info,
        "weights": weights,
        "coverage": coverage_report,
    }

    return out_ds, summary_report


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Preprocessing & Vertical Interpolation")
    parser.add_argument(
        "--input",
        type=str,
        default="Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn_1100m.nc",
        help="Path to downloaded GLORYS NetCDF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Dataset/gate_a1_pilot/glorys_pilot_30d_15depths_preprocessed.nc",
        help="Path to save preprocessed NetCDF file",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="outputs/gate_a1_3_preprocessing_metadata.json",
        help="Path to save preprocessing JSON metadata",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 70)
    print("OCEANEMBED — PS-ALIGNED 15-DEPTH PREPROCESSING STAGE")
    print("=" * 70)

    out_ds, summary = preprocess_glorys_dataset(args.input, args.output)

    print(f"\n[SUCCESS] Preprocessing completed.")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Input Dimensions:  {summary['input_dimensions']}")
    print(f"  Output Dimensions: {summary['output_dimensions']}")
    print(f"\nThetao Ocean Coverage across 15 Canonical Depths:")
    for row in summary["coverage"]["thetao"]:
        print(f"  Depth {row['target_depth_m']:4.0f} m: {row['valid_coverage_pct']:6.2f}% valid ocean "
              f"({row['valid_points']:,} valid, {row['nan_points']:,} NaN)")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetadata report saved to: {report_path}")


if __name__ == "__main__":
    main()
