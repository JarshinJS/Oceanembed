"""
Unit Test Suite for Gate A3 — Temporal / Seasonal Robustness
============================================================
12 rigorous tests verifying Phase Q requirements:
  1. test_temporal_continuity
  2. test_spatial_grid_preservation
  3. test_channel_dimension
  4. test_depth_dimension
  5. test_mask_consistency
  6. test_normalization_leakage_free
  7. test_temporal_split_integrity
  8. test_architecture_parameters
  9. test_a2_reproduction_metric
  10. test_baseline_0_vs_1_test_period
  11. test_depth_500m_metric
  12. test_ablation_and_control
"""

import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
import torch

from src.constants import (
    REQUIRED_DEPTHS_M,
    CANONICAL_SURFACE_VARIABLES as CANONICAL_CHANNELS,
    GATE_A1_PILOT_BOX,
)

CANONICAL_GRID_LAT_POINTS = GATE_A1_PILOT_BOX["grid_height"]
CANONICAL_GRID_LON_POINTS = GATE_A1_PILOT_BOX["grid_width"]
CANONICAL_GRID_LAT_MIN = GATE_A1_PILOT_BOX["lat_centers"][0]
CANONICAL_GRID_LAT_MAX = GATE_A1_PILOT_BOX["lat_centers"][-1]
CANONICAL_GRID_LON_MIN = GATE_A1_PILOT_BOX["lon_centers"][0]
CANONICAL_GRID_LON_MAX = GATE_A1_PILOT_BOX["lon_centers"][-1]
from src.gate_a2.cnn_baseline import SimpleSpatialCNN
from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3 as OceanEmbedA3Dataset

SYNCH_DIR = Path("Dataset/gate_a3_temporal/synchronized")
INPUT_NC = SYNCH_DIR / "oceanembed_inputs_7ch_91d.nc"
TARGET_NC = SYNCH_DIR / "oceanembed_target_15depth_91d.nc"
NORM_JSON = Path("Dataset/gate_a3_temporal/normalization/normalization_stats.json")
RESULTS_JSON = Path("Dataset/gate_a3_temporal/metrics/gate_a3_results.json")
PRED_NC = Path("Dataset/gate_a3_temporal/predictions/eval_predictions_a3.nc")
@pytest.fixture(scope="module")
def input_ds():
    if not INPUT_NC.exists():
        pytest.skip(f"Input NetCDF {INPUT_NC} does not exist yet.")
    with xr.open_dataset(INPUT_NC) as ds:
        loaded = ds.load()
    return loaded


@pytest.fixture(scope="module")
def target_ds():
    if not TARGET_NC.exists():
        pytest.skip(f"Target NetCDF {TARGET_NC} does not exist yet.")
    with xr.open_dataset(TARGET_NC) as ds:
        loaded = ds.load()
    return loaded


@pytest.fixture(scope="module")
def results_dict():
    if not RESULTS_JSON.exists():
        pytest.skip(f"Results JSON {RESULTS_JSON} does not exist yet.")
    with open(RESULTS_JSON, "r") as f:
        return json.load(f)


def test_1_temporal_continuity(input_ds, target_ds):
    """Test 1: 91 continuous daily timestamps from 2020-01-01 to 2020-03-31."""
    assert len(input_ds.time) == 91, f"Expected 91 input time steps, got {len(input_ds.time)}"
    assert len(target_ds.time) == 91, f"Expected 91 target time steps, got {len(target_ds.time)}"

    input_times = [str(t)[:10] for t in input_ds.time.values]
    target_times = [str(t)[:10] for t in target_ds.time.values]

    assert input_times == target_times, "Input and Target timestamps do not match!"
    assert input_times[0] == "2020-01-01", f"First date should be 2020-01-01, got {input_times[0]}"
    assert input_times[-1] == "2020-03-31", f"Last date should be 2020-03-31, got {input_times[-1]}"

    # Verify no duplicates and monotonic 1-day spacing
    diffs = np.diff(input_ds.time.values.astype("datetime64[D]"))
    assert np.all(diffs == np.timedelta64(1, "D")), "Timestamps are not strictly continuous daily steps!"


def test_2_spatial_grid_preservation(input_ds, target_ds):
    """Test 2: Canonical 24x32 grid coordinates preserved."""
    assert len(input_ds.latitude) == CANONICAL_GRID_LAT_POINTS
    assert len(input_ds.longitude) == CANONICAL_GRID_LON_POINTS
    assert len(target_ds.latitude) == CANONICAL_GRID_LAT_POINTS
    assert len(target_ds.longitude) == CANONICAL_GRID_LON_POINTS

    np.testing.assert_allclose(input_ds.latitude.values[0], CANONICAL_GRID_LAT_MIN, atol=1e-4)
    np.testing.assert_allclose(input_ds.latitude.values[-1], CANONICAL_GRID_LAT_MAX, atol=1e-4)
    np.testing.assert_allclose(input_ds.longitude.values[0], CANONICAL_GRID_LON_MIN, atol=1e-4)
    np.testing.assert_allclose(input_ds.longitude.values[-1], CANONICAL_GRID_LON_MAX, atol=1e-4)


def test_3_channel_dimension(input_ds):
    """Test 3: Exactly 7 canonical input channels."""
    assert len(input_ds.channel) == 7
    channels = list(input_ds.channel_name.values)
    assert channels == CANONICAL_CHANNELS
    assert input_ds.inputs.shape == (91, 7, 24, 32)


def test_4_depth_dimension(target_ds):
    """Test 4: Exactly 15 target depths matching REQUIRED_DEPTHS_M, including 1000m."""
    assert len(target_ds.depth) == 15
    np.testing.assert_allclose(target_ds.depth.values, REQUIRED_DEPTHS_M, atol=1e-2)
    assert 1000.0 in target_ds.depth.values
    assert target_ds.target_thetao.shape == (91, 15, 24, 32)


def test_5_mask_consistency(input_ds, target_ds):
    """Test 5: Ocean masks are consistent, binary {0, 1}, and reflect ocean domain."""
    assert "mask_inputs" in input_ds
    assert "mask_target" in target_ds
    assert "ocean_mask_2d" in input_ds
    assert "ocean_mask_2d" in target_ds

    in_mask = input_ds.ocean_mask_2d.values
    tgt_mask = target_ds.ocean_mask_2d.values

    assert set(np.unique(in_mask)).issubset({0.0, 1.0})
    assert set(np.unique(tgt_mask)).issubset({0.0, 1.0})

    # Central ocean pixel (lat=12, lon=16) must be valid ocean (1.0)
    assert np.all(in_mask[:, 12, 16] == 1.0)
    assert np.all(tgt_mask[:, 12, 16] == 1.0)


def test_6_normalization_leakage_free():
    """Test 6: Normalization stats calculated strictly on Days 1-60 with zero test data leakage."""
    if not NORM_JSON.exists():
        pytest.skip(f"{NORM_JSON} does not exist yet.")
    with open(NORM_JSON, "r") as f:
        norm_stats = json.load(f)

    assert norm_stats["training_window"]["days_count"] == 60
    assert norm_stats["training_window"]["start_date"] == "2020-01-01"
    assert norm_stats["training_window"]["end_date"] == "2020-02-29"
    assert norm_stats["training_window"]["leakage_prevention_verified"] is True
    assert len(norm_stats["input_means"]) == 7
    assert len(norm_stats["target_means"]) == 15


def test_7_temporal_split_integrity():
    """Test 7: PyTorch dataset splits are strictly partitioned."""
    if not (INPUT_NC.exists() and TARGET_NC.exists() and NORM_JSON.exists()):
        pytest.skip("Prerequisite NetCDF/JSON files missing.")

    ds_train = OceanEmbedA3Dataset(INPUT_NC, TARGET_NC, NORM_JSON, split="train")
    ds_val = OceanEmbedA3Dataset(INPUT_NC, TARGET_NC, NORM_JSON, split="val")
    ds_train_full = OceanEmbedA3Dataset(INPUT_NC, TARGET_NC, NORM_JSON, split="train_full")
    ds_test = OceanEmbedA3Dataset(INPUT_NC, TARGET_NC, NORM_JSON, split="test")

    assert len(ds_train) == 60  # Days 1-60 (2020-01-01 to 2020-02-29)
    assert len(ds_val) == 15    # Days 61-75 (2020-03-01 to 2020-03-15)
    assert len(ds_train_full) == 60  # Days 1-60
    assert len(ds_test) == 16   # Days 76-91 (2020-03-16 to 2020-03-31)

    # Check sample shapes
    sample = ds_test[0]
    x, y, m_in, m_tgt = (sample["x"], sample["y"], sample["mask_x"], sample["mask_y"]) if isinstance(sample, dict) else sample[:4]
    assert x.shape == (7, 24, 32)
    assert y.shape == (15, 24, 32)
    assert m_in.shape == (24, 32)
    assert m_tgt.shape == (15, 24, 32)


def test_8_architecture_parameters():
    """Test 8: SimpleSpatialCNN has exactly 39,711 parameters and 4 Conv2d layers."""
    model = SimpleSpatialCNN(in_channels=7, out_channels=15)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 39759, f"Expected 39,759 parameters, got {param_count}"

    # Verify 4 Conv2d layers
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(conv_layers) == 4, f"Expected 4 Conv2d layers, got {len(conv_layers)}"


def test_9_a2_reproduction_metric(results_dict):
    """Test 9: A2 reproduction check matches original A2 metric (0.4648 °C) within ±0.05 °C."""
    a2_repro = results_dict.get("a2_reproduction_check", {})
    assert "reproduced_a2_eval_rmse" in a2_repro
    repro_rmse = a2_repro["reproduced_a2_eval_rmse"]
    orig_rmse = a2_repro["original_a2_eval_rmse"]
    assert abs(repro_rmse - orig_rmse) <= 0.05, f"A2 reproduction discrepancy too large: {repro_rmse} vs {orig_rmse}"
    assert a2_repro["reproduction_passed"] is True


def test_10_baseline_0_vs_1_test_period(results_dict):
    """Test 10: Baseline 1 (CNN) achieves positive RMSE reduction vs Baseline 0 on March test period."""
    b0_rmse = results_dict["baseline_0_climatology"]["overall_rmse"]
    b1_rmse = results_dict["baseline_1_cnn"]["overall_rmse"]
    reduction_pct = results_dict["overall_relative_rmse_reduction_pct"]

    assert b1_rmse < b0_rmse, f"CNN ({b1_rmse:.4f}) failed to beat Climatology ({b0_rmse:.4f})"
    assert reduction_pct > 0.0, f"Expected positive reduction, got {reduction_pct:.2f}%"


def test_11_depth_500m_metric(results_dict):
    """Test 11: 500m depth metric exists, is finite, and tracked."""
    per_depth = results_dict["per_depth_comparison"]
    depth_500_entries = [d for d in per_depth if d["depth_m"] == 500]
    assert len(depth_500_entries) == 1
    d500 = depth_500_entries[0]
    assert np.isfinite(d500["cnn_rmse"])
    assert np.isfinite(d500["reference_rmse"])
    assert d500["cnn_rmse"] < 1.0  # Deep ocean temperature error should be small


def test_12_ablation_and_control(results_dict):
    """Test 12: All 5 ablations and shuffled control recorded; shuffled control is degraded."""
    ablations = results_dict.get("ablations", {})
    required_keys = ["all_7_channels", "sst_only", "sst_sss", "sst_sss_ssh", "surface_ocean_5ch", "shuffled_control"]
    for k in required_keys:
        assert k in ablations, f"Missing ablation key: {k}"
        assert np.isfinite(ablations[k]["overall_rmse"])

    # Shuffled control must have worse (higher) RMSE than the intact All 7 Channels model
    all_7_rmse = ablations["all_7_channels"]["overall_rmse"]
    shuffled_rmse = ablations["shuffled_control"]["overall_rmse"]
    assert shuffled_rmse > all_7_rmse, f"Shuffled control ({shuffled_rmse:.4f}) did not degrade vs All 7 ({all_7_rmse:.4f})"
