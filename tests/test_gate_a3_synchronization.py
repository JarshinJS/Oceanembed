"""
Gate A3.1 Expanded Synchronization Test Suite
==============================================
Validates the synchronized NetCDF artifacts for Gate A3.1:
  1. Exact 61 daily timestamps (2020-01-31 to 2020-03-31)
  2. Exact 7 input channels with canonical names and order
  3. Exact 15 target depths matching REQUIRED_DEPTHS_M (including 1000m)
  4. Spatial grid dimensions (24 x 32) and coordinates
  5. Monotonic coordinate ordering (lat, lon, depth)
  6. Mask integrity (binary {0, 1}, finite data preservation)
  7. Ocean coverage consistency against Gate A1.5 pilot (~98%)
"""

import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from src.constants import (
    REQUIRED_DEPTHS_M,
    CANONICAL_SURFACE_VARIABLES,
    GATE_A1_PILOT_BOX,
)

CANONICAL_GRID_LAT_POINTS = GATE_A1_PILOT_BOX["grid_height"]
CANONICAL_GRID_LON_POINTS = GATE_A1_PILOT_BOX["grid_width"]
CANONICAL_GRID_LAT_MIN = GATE_A1_PILOT_BOX["lat_centers"][0]
CANONICAL_GRID_LAT_MAX = GATE_A1_PILOT_BOX["lat_centers"][-1]
CANONICAL_GRID_LON_MIN = GATE_A1_PILOT_BOX["lon_centers"][0]
CANONICAL_GRID_LON_MAX = GATE_A1_PILOT_BOX["lon_centers"][-1]

SYNCH_DIR = Path("Dataset/gate_a3_temporal/synchronized")
INPUT_NC = SYNCH_DIR / "oceanembed_inputs_7ch_a3.nc"
TARGET_NC = SYNCH_DIR / "oceanembed_target_15depth_a3.nc"
METADATA_JSON = SYNCH_DIR / "synchronization_metadata_a3.json"


@pytest.fixture(scope="module")
def a3_inputs():
    assert INPUT_NC.exists(), f"Missing input NetCDF: {INPUT_NC}"
    ds = xr.open_dataset(INPUT_NC)
    yield ds
    ds.close()


@pytest.fixture(scope="module")
def a3_target():
    assert TARGET_NC.exists(), f"Missing target NetCDF: {TARGET_NC}"
    ds = xr.open_dataset(TARGET_NC)
    yield ds
    ds.close()


@pytest.fixture(scope="module")
def a3_meta():
    assert METADATA_JSON.exists(), f"Missing metadata JSON: {METADATA_JSON}"
    with open(METADATA_JSON, "r") as f:
        return json.load(f)


def test_01_temporal_dimensions_and_continuity(a3_inputs, a3_target):
    """Verify exactly 61 continuous daily timestamps from 2020-01-31 to 2020-03-31."""
    assert len(a3_inputs.time) == 61
    assert len(a3_target.time) == 61

    in_times = [str(t)[:10] for t in a3_inputs.time.values]
    tgt_times = [str(t)[:10] for t in a3_target.time.values]

    assert in_times == tgt_times
    assert in_times[0] == "2020-01-31"
    assert in_times[-1] == "2020-03-31"

    diffs = np.diff(a3_inputs.time.values.astype("datetime64[D]"))
    assert np.all(diffs == np.timedelta64(1, "D")), "Timestamps are not strictly daily continuous!"


def test_02_spatial_grid_and_coordinates(a3_inputs, a3_target):
    """Verify 24 x 32 cell-centered 0.25 deg grid with ascending lat/lon."""
    assert len(a3_inputs.latitude) == CANONICAL_GRID_LAT_POINTS  # 24
    assert len(a3_inputs.longitude) == CANONICAL_GRID_LON_POINTS  # 32
    assert len(a3_target.latitude) == CANONICAL_GRID_LAT_POINTS
    assert len(a3_target.longitude) == CANONICAL_GRID_LON_POINTS

    # Monotonic ascending
    assert np.all(np.diff(a3_inputs.latitude.values) > 0)
    assert np.all(np.diff(a3_inputs.longitude.values) > 0)

    # Coordinate bounds
    np.testing.assert_allclose(a3_inputs.latitude.values[0], CANONICAL_GRID_LAT_MIN, atol=1e-4)
    np.testing.assert_allclose(a3_inputs.latitude.values[-1], CANONICAL_GRID_LAT_MAX, atol=1e-4)
    np.testing.assert_allclose(a3_inputs.longitude.values[0], CANONICAL_GRID_LON_MIN, atol=1e-4)
    np.testing.assert_allclose(a3_inputs.longitude.values[-1], CANONICAL_GRID_LON_MAX, atol=1e-4)


def test_03_canonical_seven_channels(a3_inputs):
    """Verify input tensor has shape [61, 7, 24, 32] and standard 7 channels."""
    assert a3_inputs["inputs"].shape == (61, 7, 24, 32)
    assert a3_inputs["mask_inputs"].shape == (61, 7, 24, 32)
    assert len(a3_inputs.channel) == 7

    channel_names = list(a3_inputs.channel_name.values)
    assert channel_names == CANONICAL_SURFACE_VARIABLES


def test_04_canonical_fifteen_target_depths(a3_target):
    """Verify target tensor has shape [61, 15, 24, 32] and exact 15 depths."""
    assert a3_target["target_thetao"].shape == (61, 15, 24, 32)
    assert a3_target["mask_target"].shape == (61, 15, 24, 32)
    assert len(a3_target.depth) == 15

    np.testing.assert_allclose(a3_target.depth.values, REQUIRED_DEPTHS_M, atol=1e-2)
    assert 1000.0 in a3_target.depth.values
    assert a3_target.depth.values[0] == 0.0


def test_05_ocean_mask_and_coverage_consistency(a3_inputs, a3_target, a3_meta):
    """Verify ocean masks are binary {0, 1} and coverage is consistent with Gate A1.5 (~98%)."""
    in_mask = a3_inputs.mask_inputs.values
    tgt_mask = a3_target.mask_target.values

    assert set(np.unique(in_mask)).issubset({0.0, 1.0})
    assert set(np.unique(tgt_mask)).issubset({0.0, 1.0})

    valid_ocean_pct = a3_meta["spatial_grid"]["valid_ocean_pct"]
    assert 95.0 <= valid_ocean_pct <= 100.0
    assert abs(valid_ocean_pct - 98.31) < 1.0, f"Coverage {valid_ocean_pct}% drifted significantly from A1.5 98.31%"


def test_06_finite_values_in_ocean(a3_inputs, a3_target):
    """Verify all ocean cells have finite values and no land cells contaminate predictions."""
    ocean_mask_2d = a3_inputs.ocean_mask_2d.values == 1.0

    # Every ocean cell across all 61 days must be finite in all 7 channels
    inputs_arr = a3_inputs["inputs"].values  # (61, 7, 24, 32)
    for c in range(7):
        ch_ocean_vals = inputs_arr[:, c, ocean_mask_2d]
        assert np.all(np.isfinite(ch_ocean_vals)), f"Found NaNs in ocean cells for channel {c}"

    # Target surface layer ocean cells must be finite
    target_arr = a3_target["target_thetao"].values  # (61, 15, 24, 32)
    assert np.all(np.isfinite(target_arr[:, 0, ocean_mask_2d])), "Found NaNs in target 0m ocean cells"
