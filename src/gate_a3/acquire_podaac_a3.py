"""
OceanEmbed — Gate A3 NASA PO.DAAC Acquisition Engine (Expanded Period)
======================================================================
Acquires OSCAR Surface Currents and CCMP 10m Vector Winds for the expanded
temporal period (2020-01-31 to 2020-03-31, 61 days) using Earthdata credentials.

Outputs:
  - Dataset/gate_a3_temporal/raw/oscar_currents_expanded_61d.nc
  - Dataset/gate_a3_temporal/raw/ccmp_wind_expanded_61d.nc
"""

import concurrent.futures
import io
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expanded 61-day date list (2020-01-31 to 2020-03-31)
DATES = [d.strftime("%Y%m%d") for d in pd.date_range("2020-01-31", "2020-03-31", freq="D")]

OUTPUT_DIR = Path("Dataset/gate_a3_temporal/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_OSCAR = OUTPUT_DIR / "temp_oscar"
TEMP_CCMP = OUTPUT_DIR / "temp_ccmp"
TEMP_OSCAR.mkdir(parents=True, exist_ok=True)
TEMP_CCMP.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 12.0, 18.0
LON_MIN, LON_MAX = 85.0, 93.0


def fetch_with_retry(url: str, max_retries: int = 4) -> bytes:
    """Download binary content with robust timeout and retry."""
    s = requests.Session()
    s.trust_env = True

    for attempt in range(1, max_retries + 1):
        try:
            r = s.get(url, timeout=(15, 60), stream=True)
            if r.status_code == 200:
                buf = io.BytesIO()
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        buf.write(chunk)
                return buf.getvalue()
            elif r.status_code in [404, 403]:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")
            else:
                logger.warning(f"HTTP {r.status_code} on attempt {attempt}/{max_retries} for {url}")
        except Exception as e:
            logger.warning(f"Error on attempt {attempt}/{max_retries} for {url}: {e}")

        time.sleep(2 * attempt)

    raise RuntimeError(f"Exhausted {max_retries} retries for {url}")


def acquire_oscar_day(date_str: str) -> Path:
    """Download, slice and cache one daily OSCAR file."""
    cache_file = TEMP_OSCAR / f"oscar_{date_str}.nc"
    if cache_file.exists() and cache_file.stat().st_size > 500:
        return cache_file

    url = f"https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/OSCAR_L4_OC_FINAL_V2.0/oscar_currents_final_{date_str}.nc"
    content = fetch_with_retry(url)

    ds = xr.open_dataset(io.BytesIO(content))
    ds = ds.swap_dims({"latitude": "lat", "longitude": "lon"})
    sub = ds[["u", "v"]].sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    ).load()
    ds.close()

    sub.to_netcdf(cache_file)
    logger.info(f"  [OSCAR] Cached {date_str} ({cache_file.stat().st_size / 1024:.1f} KB)")
    return cache_file


def acquire_ccmp_day(date_str: str) -> Path:
    """Download, slice and cache one daily CCMP file (daily vector average of 6-hourly steps)."""
    cache_file = TEMP_CCMP / f"ccmp_{date_str}.nc"
    if cache_file.exists() and cache_file.stat().st_size > 500:
        return cache_file

    url = f"https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/CCMP_WINDS_10M6HR_L4_V3.1/CCMP_Wind_Analysis_{date_str}_V03.1_L4.nc"
    content = fetch_with_retry(url)

    ds = xr.open_dataset(io.BytesIO(content))
    sub = ds[["uwnd", "vwnd"]].sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    ).load()
    ds.close()

    # Aggregate 6-hourly to daily vector mean
    daily_mean = sub.mean(dim="time", keep_attrs=True)
    daily_date = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    daily_mean = daily_mean.expand_dims(time=[daily_date])

    daily_mean.to_netcdf(cache_file)
    logger.info(f"  [CCMP] Cached {date_str} ({cache_file.stat().st_size / 1024:.1f} KB)")
    return cache_file


def acquire_all_oscar():
    logger.info(f"Starting OSCAR surface current acquisition ({len(DATES)} days)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(acquire_oscar_day, d): d for d in DATES}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    cached_files = [TEMP_OSCAR / f"oscar_{d}.nc" for d in sorted(DATES)]
    datasets = [xr.open_dataset(f) for f in cached_files]
    combined = xr.concat(datasets, dim="time")

    out_file = OUTPUT_DIR / "oscar_currents_expanded_61d.nc"
    combined.to_netcdf(out_file)
    for ds in datasets:
        ds.close()

    logger.info(f"[OSCAR] Combined 61-day dataset saved to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return out_file


def acquire_all_ccmp():
    logger.info(f"Starting CCMP surface wind acquisition ({len(DATES)} days)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(acquire_ccmp_day, d): d for d in DATES}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    cached_files = [TEMP_CCMP / f"ccmp_{d}.nc" for d in sorted(DATES)]
    datasets = [xr.open_dataset(f) for f in cached_files]
    combined = xr.concat(datasets, dim="time")

    out_file = OUTPUT_DIR / "ccmp_wind_expanded_61d.nc"
    combined.to_netcdf(out_file)
    for ds in datasets:
        ds.close()

    logger.info(f"[CCMP] Combined 61-day dataset saved to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return out_file


if __name__ == "__main__":
    t0 = time.time()
    acquire_all_oscar()
    acquire_all_ccmp()
    print(f"\n[SUCCESS] Both OSCAR and CCMP expanded 61-day datasets acquired in {time.time() - t0:.1f}s")
