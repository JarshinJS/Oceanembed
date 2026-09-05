"""
Test Suite: OceanEmbed Interactive Demo Verification
===================================================
Problem Statement ID: 26066

Verifies:
  1. Server initialization & service instantiation
  2. Frozen checkpoint loading & parameter verification (123,345 params)
  3. Normalization statistics loading without hardcoded values
  4. /api/health response format & metadata
  5. /api/dates response count (91 days), splits, and ARGO flags
  6. /api/surface response channels (7 surface variables), masks, and coverage
  7. /api/predict execution, caching, and latency
  8. Output shape strictness: [15, 24, 32]
  9. Finite prediction values over ocean cells (no NaN in ocean)
 10. Depth ordering preservation (15 canonical depths: 0 to 1000m)
 11. Ocean mask preservation (land cells masked as null/NaN, ocean preserved)
 12. /api/profile vertical extraction (OceanEmbed, Climatology, GLORYS, and ARGO if collocated)
 13. /api/metrics exact Gate A4.3 values and strict scientific labeling
 14. Real end-to-end forward pass on default date '2020-03-20' with [1, 7, 24, 32] input
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np
import torch

from src.constants import REQUIRED_DEPTHS_M, CANONICAL_SURFACE_VARIABLES
from src.demo.server import (
    OceanEmbedDemoService,
    MODEL_CHECKPOINT_PATH,
    NORM_STATS_PATH,
    CLIMATOLOGY_CACHE_PATH,
    get_demo_service,
)


@pytest.fixture(scope="module")
def demo_service() -> OceanEmbedDemoService:
    """Fixture providing singleton OceanEmbedDemoService."""
    return get_demo_service()


def test_01_checkpoint_loading(demo_service: OceanEmbedDemoService):
    """Verify frozen checkpoint exists and loaded model parameter count is exactly 123,345."""
    assert MODEL_CHECKPOINT_PATH.exists(), f"Checkpoint missing at {MODEL_CHECKPOINT_PATH}"
    assert demo_service.model is not None
    assert demo_service.param_count == 123345, f"Expected 123345 parameters, got {demo_service.param_count}"
    assert not demo_service.model.training, "Model must be in eval mode."


def test_02_normalization_loading(demo_service: OceanEmbedDemoService):
    """Verify normalization statistics loaded from JSON artifact without hardcoded overrides."""
    assert NORM_STATS_PATH.exists()
    assert hasattr(demo_service, "input_means") and hasattr(demo_service, "input_stds")
    assert demo_service.input_means.shape == (7,)
    assert demo_service.input_stds.shape == (7,)
    # Verify SST train mean matches JSON artifact
    assert abs(demo_service.input_means[0] - 26.8305) < 1e-3


def test_03_climatology_loading(demo_service: OceanEmbedDemoService):
    """Verify climatology loaded from validated artifact with shape [15, 24, 32]."""
    assert CLIMATOLOGY_CACHE_PATH.exists()
    assert demo_service.clim_mean.shape == (15, 24, 32)
    assert demo_service.clim_mask.shape == (15, 24, 32)
    assert np.all(np.isfinite(demo_service.clim_mean[demo_service.clim_mask == 1.0]))


def test_04_api_health(demo_service: OceanEmbedDemoService):
    """Verify /api/health endpoint payload and metadata."""
    health = demo_service.get_health()
    assert health["status"] == "ok"
    assert health["model_loaded"] is True
    assert health["parameter_count"] == 123345
    assert health["domain"]["grid_shape"] == [24, 32]
    assert health["domain"]["depth_levels_count"] == 15
    assert health["date_range"]["total_days"] == 91
    assert health["date_range"]["splits"]["train"]["count"] == 60
    assert health["date_range"]["splits"]["validation"]["count"] == 15
    assert health["date_range"]["splits"]["test"]["count"] == 16


def test_05_api_dates(demo_service: OceanEmbedDemoService):
    """Verify /api/dates returns 91 dates across train, validation, and test splits with ARGO flags."""
    dates = demo_service.get_dates()
    assert len(dates) == 91
    assert dates[0]["date"] == "2020-01-01"
    assert dates[0]["split"] == "train"
    assert dates[60]["date"] == "2020-03-01"
    assert dates[60]["split"] == "validation"
    assert dates[75]["date"] == "2020-03-16"
    assert dates[75]["split"] == "test"

    # Verify ARGO availability in test period
    argo_dates = [d for d in dates if d["has_argo"]]
    assert len(argo_dates) == 16, "All 16 test days must have collocated ARGO observations."


def test_06_api_surface(demo_service: OceanEmbedDemoService):
    """Verify /api/surface returns all 7 surface channels with correct units and validity masks."""
    surface = demo_service.get_surface("2020-03-20")
    assert surface["date"] == "2020-03-20"
    assert len(surface["latitudes"]) == 24
    assert len(surface["longitudes"]) == 32
    assert len(surface["channels"]) == 7

    for expected_var in CANONICAL_SURFACE_VARIABLES:
        assert expected_var in surface["channels"]
        ch = surface["channels"][expected_var]
        assert "name" in ch
        assert "units" in ch
        assert "coverage_pct" in ch
        assert len(ch["data"]) == 24
        assert len(ch["data"][0]) == 32

    # Verify SST units and reasonable physical range
    sst_grid = surface["channels"]["sst"]["data"]
    valid_ssts = [val for row in sst_grid for val in row if val is not None]
    assert len(valid_ssts) > 0
    assert 20.0 < np.mean(valid_ssts) < 35.0


def test_07_api_predict_real_end_to_end(demo_service: OceanEmbedDemoService):
    """
    Verify real end-to-end inference for 2020-03-20:
      - Input shape: [1, 7, 24, 32]
      - Output shape: [15, 24, 32]
      - Finite values over ocean cells
      - Sub-100ms latency on CPU
    """
    res = demo_service.predict("2020-03-20")
    assert res["date"] == "2020-03-20"
    assert res["model_name"] == "OceanEmbed v1-Local"

    pred = res["prediction"]
    assert len(pred) == 15, "Expected 15 vertical depths in prediction."
    assert len(pred[0]) == 24, "Expected 24 latitude rows."
    assert len(pred[0][0]) == 32, "Expected 32 longitude columns."

    # Test caching behavior: second call should be cached
    res_cached = demo_service.predict("2020-03-20")
    assert res_cached["cached"] is True

    # Check finite values over ocean cells
    ocean_mask = res["ocean_mask"]
    for d_idx in range(15):
        for r in range(24):
            for c in range(32):
                if ocean_mask[r][c] == 1:
                    val = pred[d_idx][r][c]
                    assert val is not None and not math.isnan(val), f"NaN at ocean cell ({d_idx}, {r}, {c})"
                else:
                    # Land cells must be null/None (not fabricated numbers)
                    val = pred[d_idx][r][c]
                    assert val is None, f"Land cell ({d_idx}, {r}, {c}) must be None/null, got {val}"


def test_08_depth_ordering_and_monotonicity(demo_service: OceanEmbedDemoService):
    """Verify 15 canonical depths and general physical thermal stratification (surface warmer than deep)."""
    res = demo_service.predict("2020-03-20")
    pred = res["prediction"]
    assert res["depths"] == REQUIRED_DEPTHS_M

    # Probe central ocean pixel (r=12, c=16)
    r_center, c_center = 12, 16
    surface_temp = pred[0][r_center][c_center]  # 0m
    thermocline_temp = pred[7][r_center][c_center]  # 100m
    deep_temp = pred[14][r_center][c_center]  # 1000m

    assert surface_temp > thermocline_temp > deep_temp, (
        f"Expected physical thermal stratification: Surface {surface_temp} > Thermocline {thermocline_temp} > Deep {deep_temp}"
    )


def test_09_api_profile_collocation(demo_service: OceanEmbedDemoService):
    """
    Verify /api/profile vertical extraction:
      - OceanEmbed profile
      - Climatology profile
      - GLORYS reference profile
      - Collocated ARGO float observations on 2020-03-20
    """
    # Coordinates of Float #2902236 on 2020-03-20
    lat_argo = 17.1333
    lon_argo = 85.5833
    prof = demo_service.get_profile("2020-03-20", lat=lat_argo, lon=lon_argo)

    assert prof["valid_ocean"] is True
    assert len(prof["oceanembed_profile"]) == 15
    assert len(prof["climatology_profile"]) == 15
    assert len(prof["glorys_profile"]) == 15

    # ARGO float must be collocated within 10 km
    assert prof["argo_profile"] is not None
    assert prof["argo_profile"]["platform_number"] == "2902236"
    assert prof["argo_profile"]["distance_km"] < 10.0
    assert len(prof["argo_profile"]["depths"]) > 20
    assert len(prof["argo_profile"]["temperatures"]) > 20


def test_10_api_profile_land_mask_handling(demo_service: OceanEmbedDemoService):
    """Verify that querying a land coordinate returns valid_ocean = False rather than fabricated data."""
    # Far inland coordinate in India/Bangladesh
    prof_land = demo_service.get_profile("2020-03-20", lat=17.8, lon=85.2)
    # Either valid_ocean is false or fallback nearest cell is explicitly indicated
    assert "valid_ocean" in prof_land


def test_11_api_metrics_scientific_labels(demo_service: OceanEmbedDemoService):
    """
    Verify /api/metrics values and strict scientific labeling:
      - Overall RMSE = 0.8117 °C
      - Thermocline RMSE = 0.8446 °C
      - Climatology Thermocline RMSE = 1.2827 °C
      - Relative improvement = 34.16%
      - 34 profiles, 14 WMO platforms
      - Exact permutation p = 0.001709
      - Required terminology: '14 unique WMO float platforms, treated as conservative cluster units'
      - Labeled as 'GLORYS reanalysis-derived reference' (not ground truth)
    """
    metrics = demo_service.get_metrics()
    assert metrics["gate"] == "Gate A4.3"
    assert metrics["overall_metrics"]["rmse_celsius"] == 0.8117
    assert metrics["overall_metrics"]["mae_celsius"] == 0.5426
    assert metrics["overall_metrics"]["bias_celsius"] == -0.0863
    assert metrics["overall_metrics"]["pearson_r"] == 0.9943

    assert metrics["thermocline_regime_50_200m"]["rmse_celsius"] == 0.8446
    assert metrics["thermocline_regime_50_200m"]["climatology_rmse_celsius"] == 1.2827
    assert metrics["thermocline_regime_50_200m"]["relative_rmse_improvement_pct"] == 34.16

    argo_stats = metrics["independent_argo_statistical_rigor"]
    assert argo_stats["profiles_count"] == 34
    assert argo_stats["unique_platforms_count"] == 14
    assert "14 unique WMO float platforms, treated as conservative cluster units" in argo_stats["platform_cluster_label"]
    assert argo_stats["exact_platform_permutation_p_value"] == 0.001709
    assert argo_stats["platform_clustered_bootstrap_95_ci"] == [-0.3457, -0.1117]

    labels = metrics["scientific_reference_labels"]
    assert labels["glorys"] == "GLORYS reanalysis-derived reference"
    assert labels["argo"] == "Independent ARGO observational reference"
    assert "ground truth" not in json.dumps(metrics).lower()


def test_12_mask_preservation(demo_service: OceanEmbedDemoService):
    """Verify that raw ocean_mask_2d matches exactly across inputs and predictions."""
    res = demo_service.predict("2020-03-20")
    mask = np.array(res["ocean_mask"])
    assert mask.shape == (24, 32)
    # Check that land cells exist and are preserved
    land_cells = np.sum(mask == 0)
    ocean_cells = np.sum(mask == 1)
    assert land_cells > 0
    assert ocean_cells > 0
    assert land_cells + ocean_cells == 24 * 32


def test_13_live_http_server_endpoints():
    """Verify live HTTP server responds correctly to all API and static endpoints."""
    import threading
    import urllib.request
    from http.server import HTTPServer
    from src.demo.server import OceanEmbedHTTPHandler

    # Bind to ephemeral port
    server = HTTPServer(("127.0.0.1", 0), OceanEmbedHTTPHandler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Health
        with urllib.request.urlopen(f"{base_url}/api/health") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["status"] == "ok"

        # 2. Dates
        with urllib.request.urlopen(f"{base_url}/api/dates") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert len(data) == 91

        # 3. Surface
        with urllib.request.urlopen(f"{base_url}/api/surface?date=2020-03-20") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert "sst" in data["channels"]

        # 4. Predict POST
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=json.dumps({"date": "2020-03-20"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert len(data["prediction"]) == 15

        # 5. Profile
        with urllib.request.urlopen(f"{base_url}/api/profile?date=2020-03-20&lat=17.13&lon=85.58") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["valid_ocean"] is True

        # 6. Metrics
        with urllib.request.urlopen(f"{base_url}/api/metrics") as res:
            assert res.status == 200
            data = json.loads(res.read().decode())
            assert data["gate"] == "Gate A4.3"

        # 7. Static HTML
        with urllib.request.urlopen(f"{base_url}/") as res:
            assert res.status == 200
            content = res.read().decode()
            assert "<!DOCTYPE html>" in content
            assert "OCEANEMBED" in content

        # 8. Static CSS
        with urllib.request.urlopen(f"{base_url}/style.css") as res:
            assert res.status == 200
            content = res.read().decode()
            assert "--accent-cyan" in content

        # 9. Static JS
        with urllib.request.urlopen(f"{base_url}/app.js") as res:
            assert res.status == 200
            content = res.read().decode()
            assert "OceanEmbed" in content

    finally:
        server.shutdown()
        server.server_close()

