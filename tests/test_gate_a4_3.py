"""
Gate A4.3 Unit Tests: ARGO In Situ Validation and Collocation.
=============================================================
Tests:
  1. Bilinear spatial interpolation accuracy and boundary handling
  2. Masked land/coastal rejection during spatial collocation
  3. Quality control filtering (rejection of bad QC flags and invalid ranges)
  4. Monotonic vertical linear interpolation onto continuous ARGO depths
  5. Bootstrap confidence interval ordering (ci_lower <= mean <= ci_upper)
"""

import numpy as np
import pandas as pd
import pytest

from src.gate_a4.argo_validation import (
    bilinear_interpolate_field,
    quality_control_argo,
    compute_bootstrap_profile_cis,
    LATS,
    LONS,
)


def test_1_bilinear_interpolation_accuracy():
    """Bilinear interpolation exactly reproduces a linear field."""
    # Linear field: T = 10 + 2*lat + 3*lon
    H, W = len(LATS), len(LONS)
    field_15 = np.zeros((15, H, W), dtype=np.float32)
    mask_15 = np.ones((15, H, W), dtype=np.float32)

    lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")
    linear_plane = 10.0 + 2.0 * lat_grid + 3.0 * lon_grid
    for d in range(15):
        field_15[d] = linear_plane + d

    test_lat = 14.5
    test_lon = 88.25
    expected_t0 = 10.0 + 2.0 * test_lat + 3.0 * test_lon

    prof = bilinear_interpolate_field(field_15, mask_15, test_lat, test_lon)
    assert prof is not None, "Collocation failed for valid interior point!"
    assert prof[0] == pytest.approx(expected_t0, rel=1e-4)
    assert prof[5] == pytest.approx(expected_t0 + 5, rel=1e-4)


def test_2_bilinear_interpolation_boundary_and_mask():
    """Bilinear interpolation correctly rejects points outside grid or with masked neighbors."""
    H, W = len(LATS), len(LONS)
    field_15 = np.ones((15, H, W), dtype=np.float32)
    mask_15 = np.ones((15, H, W), dtype=np.float32)

    # Point outside latitude range
    out_prof = bilinear_interpolate_field(field_15, mask_15, 20.0, 88.0)
    assert out_prof is None, "Did not reject out-of-bounds latitude!"

    # Point outside longitude range
    out_prof_lon = bilinear_interpolate_field(field_15, mask_15, 14.0, 95.0)
    assert out_prof_lon is None, "Did not reject out-of-bounds longitude!"

    # Mask a cell near (14.5, 88.25)
    # i0 corresponds to floor of 14.5
    mask_15[0, 10, 12] = 0.0  # land cell
    lat_near = LATS[10] + 0.05
    lon_near = LONS[12] + 0.05
    masked_prof = bilinear_interpolate_field(field_15, mask_15, lat_near, lon_near)
    assert masked_prof is None, "Failed to reject collocation with masked neighbor!"


def test_3_argo_quality_control():
    """Quality control rejects bad flags and invalid physical ranges."""
    raw_data = {
        "platform_number": ["1", "1", "1", "1", "1", "2", "2"],
        "time": ["2020-03-20T00:00:00Z"] * 5 + ["2020-03-20T00:00:00Z"] * 2,
        "latitude": [14.0] * 5 + [15.0] * 2,
        "longitude": [88.0] * 5 + [89.0] * 2,
        "pres": [10.0, 50.0, 100.0, 200.0, 9999.0, 10.0, 50.0],  # 9999 is invalid
        "temp": [28.0, 26.0, 22.0, 15.0, 10.0, 28.0, 99.0],      # 99 is invalid temp
        "pres_qc": ["1", "1", "1", "1", "1", "1", "1"],
        "temp_qc": ["1", "1", "1", "1", "4", "1", "1"],          # flag 4 is bad
    }
    df_raw = pd.DataFrame(raw_data)
    df_qc, qc_summary = quality_control_argo(df_raw)

    # Profile 1 has 4 valid points (< 5 required), Profile 2 has only 1 valid point (< 5 required)
    # Both should be filtered out by min vertical levels
    assert qc_summary["passed_qc_profiles_count"] == 0, "Failed to filter profiles with < 5 valid observations!"


def test_4_vertical_interpolation_monotonicity():
    """Model canonical profile linear interpolation preserves physical gradient."""
    canonical_depths = np.array([0, 50, 100, 200, 1000], dtype=np.float32)
    canonical_temps = np.array([28.0, 26.0, 20.0, 15.0, 5.0], dtype=np.float32)

    argo_obs_depths = np.array([25.0, 75.0, 150.0, 500.0])
    interp_temps = np.interp(argo_obs_depths, canonical_depths, canonical_temps)

    assert interp_temps[0] == pytest.approx(27.0)
    assert interp_temps[1] == pytest.approx(23.0)
    # Monotonically decreasing
    assert (np.diff(interp_temps) < 0).all()


def test_5_bootstrap_confidence_interval():
    """Bootstrap CI properly encapsulates the mean."""
    df_dummy = pd.DataFrame({
        "model_rmse": [0.8, 0.85, 0.9, 0.95, 1.0, 0.88, 0.92, 0.86]
    })
    mean_val, ci_l, ci_u = compute_bootstrap_profile_cis(df_dummy, "model_rmse", n_boot=500)
    assert ci_l <= mean_val <= ci_u, f"Bootstrap CI ill-conditioned: [{ci_l}, {mean_val}, {ci_u}]"


def test_6_paired_inference_results():
    """Paired inference artifact exists and establishes statistical superiority."""
    import json
    from pathlib import Path

    json_path = Path("Dataset/gate_a4_3/metrics/argo_paired_inference.json")
    csv_path = Path("Dataset/gate_a4_3/metrics/argo_paired_inference.csv")
    assert json_path.exists(), "Missing argo_paired_inference.json"
    assert csv_path.exists(), "Missing argo_paired_inference.csv"

    with open(json_path) as f:
        res = json.load(f)

    paired = res["paired_profile_rmse_inference"]
    assert paired["mean_paired_difference_degC"] < 0.0
    assert paired["bootstrap_95_ci_degC"][0] < 0.0
    assert paired["bootstrap_95_ci_degC"][1] < 0.0
    assert paired["bootstrap_prob_difference_lt_0"] == 1.0
    assert paired["permutation_two_sided_p_value"] < 0.001
    assert paired["profiles_superior_count"] == 29
    assert paired["n_profiles"] == 34


def test_7_wmo_platform_reconciliation():
    """WMO platform counts are correctly resolved between raw and collocated sets."""
    import json
    from pathlib import Path

    json_path = Path("Dataset/gate_a4_3/metrics/argo_paired_inference.json")
    with open(json_path) as f:
        res = json.load(f)

    counts = res["sample_counts"]
    assert counts["raw_erddap_platforms_count"] == 16
    assert counts["collocated_platforms_count"] == 14
    assert counts["raw_erddap_profiles_count"] == 40
    assert counts["collocated_profiles_count"] == 34
    assert counts["boundary_excluded_profiles_count"] == 6
    assert set(counts["boundary_excluded_platforms"]) == {2902282, 2902770}


def test_8_depth_wise_and_paired_counts():
    """Observation counts in depth metrics match total and predictions NetCDF."""
    import json
    from pathlib import Path

    json_path = Path("Dataset/gate_a4_3/metrics/argo_paired_inference.json")
    with open(json_path) as f:
        res = json.load(f)

    assert res["sample_counts"]["total_paired_observations"] == 3966
    for d in res["depth_wise_count_verifications"]:
        assert d["verified"] is True
        assert d["actual_count"] == d["expected_count"]


def test_9_platform_clustered_inference():
    """Platform-clustered bootstrap and platform-level aggregation artifacts are valid."""
    import json
    from pathlib import Path

    json_path = Path("Dataset/gate_a4_3/metrics/argo_platform_clustered_inference.json")
    csv_path = Path("Dataset/gate_a4_3/metrics/argo_platform_clustered_inference.csv")
    assert json_path.exists(), "Missing argo_platform_clustered_inference.json"
    assert csv_path.exists(), "Missing argo_platform_clustered_inference.csv"

    with open(json_path) as f:
        res = json.load(f)

    # Preserved primary profile metrics
    prof = res["preserved_primary_profile_level_metrics"]
    assert prof["n_profiles"] == 34
    assert prof["mean_paired_difference_degC"] == pytest.approx(-0.2231, abs=1e-4)

    # Clustered bootstrap
    clust = res["platform_clustered_bootstrap_sensitivity"]
    assert clust["number_of_clusters"] == 14
    assert clust["clustered_bootstrap_mean_degC"] < 0.0
    assert clust["clustered_bootstrap_95_ci_degC"][0] < 0.0
    assert clust["clustered_bootstrap_95_ci_degC"][1] < 0.0
    assert clust["probability_mean_diff_lt_0"] == 1.0
    assert clust["bootstrap_samples_gte_0_count"] == 0

    # Conservative platform-level analysis
    plat = res["conservative_platform_level_analysis"]
    assert plat["n_platforms"] == 14
    assert plat["mean_platform_level_delta_degC"] < 0.0
    assert plat["platforms_favoring_oceanembed_count"] == 11
    assert plat["exact_permutation_two_sided_p_value"] < 0.01
    assert plat["exact_permutation_combinations_count"] == 16384


def test_10_depth_bin_non_exclusivity_audit():
    """Depth bins are non-mutually exclusive, overlapping subsets with 169 overlapping observations."""
    import xarray as xr
    import numpy as np
    from pathlib import Path

    nc_path = Path("Dataset/gate_a4_3/predictions/argo_collocated_predictions.nc")
    assert nc_path.exists(), "Missing argo_collocated_predictions.nc"

    ds = xr.open_dataset(nc_path)
    z_obs = ds["depth_m"].values
    total_obs = len(z_obs)
    assert total_obs == 3966

    depth_bins = [
        (0, 3.0),
        (5, 3.0),
        (10, 4.0),
        (20, 5.0),
        (30, 5.0),
        (50, 7.5),
        (75, 10.0),
        (100, 12.5),
        (125, 12.5),
        (150, 15.0),
        (200, 20.0),
        (300, 35.0),
        (500, 50.0),
        (700, 75.0),
        (1000, 100.0),
    ]

    bin_membership = np.zeros(total_obs, dtype=int)
    bin_counts = []
    for d, tol in depth_bins:
        mask = (z_obs >= d - tol) & (z_obs <= d + tol)
        bin_counts.append(int(np.sum(mask)))
        bin_membership[mask] += 1

    assert sum(bin_counts) == 3091
    assert np.sum(bin_membership == 0) == 1044
    assert np.sum(bin_membership == 1) == 2753
    assert np.sum(bin_membership == 2) == 169
    assert np.sum(bin_membership >= 3) == 0



