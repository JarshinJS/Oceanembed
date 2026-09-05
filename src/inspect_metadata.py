"""
OceanEmbed — NetCDF File-Level Metadata Inspector
=================================================
Inspects and reports empirical metadata from actual downloaded NetCDF files.
Enforces the Phase-A scientific contract requirement:
"Do not assume metadata from documentation alone."

Reports:
  - Dimensions and coordinate names
  - Coordinate ordering and ranges
  - Vertical depth values and surface level definition
  - Time coordinate values and calendar convention
  - Variable names, long names, standard names
  - Physical units and _FillValue / missing_value
  - Missing-value count and measured ocean coverage
  - Valid physical value ranges (min, max, mean)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

TARGET_VARIABLES = ["thetao", "so", "uo", "vo", "zos", "u10", "v10"]


def inspect_netcdf_metadata(file_path: str | Path) -> Dict[str, Any]:
    """Inspects a downloaded NetCDF file and extracts empirical metadata."""
    if xr is None:
        raise ImportError("xarray is required for metadata inspection.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    report: Dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "dimensions": {},
        "coordinates": {},
        "variables": {},
        "global_attributes": {},
    }

    with xr.open_dataset(path) as ds:
        # Dimensions
        for dim_name, dim_size in ds.dims.items():
            report["dimensions"][str(dim_name)] = int(dim_size)

        # Coordinates
        for coord_name, coord_var in ds.coords.items():
            c_data = coord_var.values
            c_info: Dict[str, Any] = {
                "dtype": str(coord_var.dtype),
                "shape": list(coord_var.shape),
                "units": str(coord_var.attrs.get("units", "N/A")),
            }
            if np.issubdtype(c_data.dtype, np.number):
                c_info["min"] = float(np.nanmin(c_data))
                c_info["max"] = float(np.nanmax(c_data))
                c_info["is_ascending"] = bool(np.all(np.diff(c_data) > 0)) if len(c_data) > 1 else True
            elif np.issubdtype(c_data.dtype, np.datetime64):
                c_info["min_time"] = str(c_data.min())
                c_info["max_time"] = str(c_data.max())
                c_info["step_count"] = len(c_data)

            if "depth" in str(coord_name).lower():
                c_info["depth_levels"] = [round(float(d), 4) for d in c_data[:15]]
                c_info["uppermost_depth_m"] = float(c_data[0]) if len(c_data) > 0 else None

            report["coordinates"][str(coord_name)] = c_info

        # Inspect Variables
        for var_name, data_var in ds.data_vars.items():
            v_data = data_var.values
            total_elements = v_data.size
            nan_count = int(np.isnan(v_data).sum())
            valid_count = total_elements - nan_count
            ocean_coverage = float(valid_count / total_elements) if total_elements > 0 else 0.0

            var_info: Dict[str, Any] = {
                "dimensions": [str(d) for d in data_var.dims],
                "shape": list(data_var.shape),
                "dtype": str(data_var.dtype),
                "units": str(data_var.attrs.get("units", "N/A")),
                "standard_name": str(data_var.attrs.get("standard_name", "N/A")),
                "long_name": str(data_var.attrs.get("long_name", "N/A")),
                "_FillValue": str(data_var.attrs.get("_FillValue", data_var.encoding.get("_FillValue", "N/A"))),
                "total_elements": total_elements,
                "missing_elements": nan_count,
                "valid_elements": valid_count,
                "ocean_coverage_ratio": round(ocean_coverage, 4),
                "ocean_coverage_pct": round(ocean_coverage * 100.0, 2),
            }

            if valid_count > 0:
                var_info["valid_min"] = float(np.nanmin(v_data))
                var_info["valid_max"] = float(np.nanmax(v_data))
                var_info["valid_mean"] = float(np.nanmean(v_data))

            report["variables"][str(var_name)] = var_info

        # Global attributes
        for attr_key, attr_val in ds.attrs.items():
            report["global_attributes"][str(attr_key)] = str(attr_val)

    return report


def print_inspection_report(report: Dict[str, Any]) -> None:
    """Pretty-prints the empirical inspection report to console."""
    print("=" * 70)
    print(f"NETCDF METADATA INSPECTION: {report['file_name']}")
    print(f"File Size: {report['file_size_mb']} MB ({report['file_size_bytes']} bytes)")
    print("=" * 70)

    print("\n--- DIMENSIONS ---")
    for d, s in report["dimensions"].items():
        print(f"  {d:15s}: {s}")

    print("\n--- COORDINATES ---")
    for c, info in report["coordinates"].items():
        print(f"  {c:15s}: shape={info['shape']}, dtype={info['dtype']}, units={info['units']}")
        if "min" in info and "max" in info:
            print(f"                   range=[{info['min']:.4f}, {info['max']:.4f}], ascending={info.get('is_ascending')}")
        if "uppermost_depth_m" in info:
            print(f"                   uppermost level = {info['uppermost_depth_m']:.4f} m")

    print("\n--- VARIABLES ---")
    for v, info in report["variables"].items():
        print(f"  {v:15s} ({info.get('standard_name', 'N/A')})")
        print(f"    Dimensions   : {info['dimensions']}")
        print(f"    Units        : {info['units']}")
        print(f"    _FillValue   : {info['_FillValue']}")
        print(f"    Ocean coverage: {info['ocean_coverage_pct']}% ({info['valid_elements']}/{info['total_elements']} valid)")
        if "valid_min" in info:
            print(f"    Valid range  : [{info['valid_min']:.3f}, {info['valid_max']:.3f}], mean={info['valid_mean']:.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect empirical NetCDF metadata.")
    parser.add_argument("--file", required=True, help="Path to NetCDF file to inspect")
    args = parser.parse_args()

    try:
        report = inspect_netcdf_metadata(args.file)
        print_inspection_report(report)
    except Exception as e:
        print(f"Error inspecting metadata: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
