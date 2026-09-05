"""
Inspection script for Gate A1.4 Acquired Satellite Datasets
===========================================================
Performs deep NetCDF inspection across all five acquired satellite datasets:
  1. OSTIA SST
  2. Multi-Obs SMAP/SMOS SSS
  3. DUACS SSH/SLA
  4. OSCAR U/V surface currents
  5. CCMP U/V 10m winds
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def inspect_all():
    files = {
        "OSTIA SST": "Dataset/gate_a1_pilot/satellite/ostia_sst_pilot_30d.nc",
        "Multi-Obs SSS": "Dataset/gate_a1_pilot/satellite/multi_obs_sss_pilot_30d.nc",
        "DUACS SSH": "Dataset/gate_a1_pilot/satellite/duacs_ssh_pilot_30d.nc",
        "OSCAR Currents": "Dataset/gate_a1_pilot/satellite/oscar_currents_pilot_30d.nc",
        "CCMP Winds": "Dataset/gate_a1_pilot/satellite/ccmp_wind_pilot_30d.nc",
    }

    results = {}

    for name, path_str in files.items():
        p = Path(path_str)
        res = {"name": name, "path": str(p), "exists": p.exists()}
        if not p.exists():
            results[name] = res
            continue
        res["file_size_bytes"] = p.stat().st_size
        res["file_size_kb"] = round(p.stat().st_size / 1024, 2)
        res["file_size_mb"] = round(p.stat().st_size / (1024 * 1024), 4)

        ds = xr.open_dataset(p)
        res["dims"] = dict(ds.sizes)
        res["data_vars"] = list(ds.data_vars.keys())
        res["coords"] = list(ds.coords.keys())

        # Time inspection
        time_coord = next((k for k in ["time", "TIME"] if k in ds.coords or k in ds.dims), None)
        if time_coord:
            t_vals = ds[time_coord].values
            res["num_time_steps"] = len(t_vals)
            # Format time strings
            t_strs = [str(t)[:19] for t in t_vals]
            res["first_time"] = t_strs[0]
            res["last_time"] = t_strs[-1]
            res["all_timestamps"] = t_strs
            res["has_duplicates"] = len(t_strs) != len(set(t_strs))

            # Check time diffs
            try:
                dt_index = pd.to_datetime([str(t)[:10] for t in t_vals])
                diffs = pd.Series(dt_index).diff().dropna()
                res["time_diffs_unique"] = [str(d) for d in diffs.unique()]
            except Exception as e:
                res["time_diff_error"] = str(e)

        # Lat/Lon inspection
        lat_name = next((k for k in ["lat", "latitude"] if k in ds.coords or k in ds.dims), None)
        lon_name = next((k for k in ["lon", "longitude"] if k in ds.coords or k in ds.dims), None)

        if lat_name:
            lat_vals = ds[lat_name].values
            res["lat_dim"] = lat_name
            res["lat_min"] = float(np.min(lat_vals))
            res["lat_max"] = float(np.max(lat_vals))
            res["lat_count"] = len(lat_vals)
            if len(lat_vals) > 1:
                res["lat_step"] = float(round(abs(float(np.diff(lat_vals).mean())), 4))

        if lon_name:
            lon_vals = ds[lon_name].values
            res["lon_dim"] = lon_name
            res["lon_min"] = float(np.min(lon_vals))
            res["lon_max"] = float(np.max(lon_vals))
            res["lon_count"] = len(lon_vals)
            if len(lon_vals) > 1:
                res["lon_step"] = float(round(abs(float(np.diff(lon_vals).mean())), 4))

        # Variables inspection
        var_details = {}
        for v in ds.data_vars:
            da = ds[v]
            vals = da.values
            attrs = da.attrs
            fill_val = attrs.get("_FillValue", attrs.get("missing_value", None))
            if isinstance(fill_val, np.number):
                fill_val = float(fill_val)
            valid_mask = ~np.isnan(vals)
            total_points = vals.size
            valid_points = int(np.sum(valid_mask))
            valid_pct = float(round((valid_points / total_points) * 100, 2)) if total_points > 0 else 0.0

            var_details[v] = {
                "dims": list(da.dims),
                "shape": list(da.shape),
                "dtype": str(da.dtype),
                "units": attrs.get("units", "unknown"),
                "long_name": str(attrs.get("long_name", attrs.get("standard_name", ""))),
                "fill_value": str(fill_val),
                "valid_points": valid_points,
                "total_points": total_points,
                "valid_percentage": valid_pct,
                "min": float(round(float(np.nanmin(vals)), 4)) if valid_points > 0 else None,
                "max": float(round(float(np.nanmax(vals)), 4)) if valid_points > 0 else None,
                "mean": float(round(float(np.nanmean(vals)), 4)) if valid_points > 0 else None,
            }
        res["variables"] = var_details
        ds.close()
        results[name] = res

    out_json = Path("Dataset/gate_a1_pilot/satellite/full_acquired_inspection.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Inspection written to {out_json}")
    return results


if __name__ == "__main__":
    inspect_all()
