"""
Gate A4.3 Statistical Integrity & Reproducibility Patch.
Computes paired profile-level inference, verifies confidence intervals,
audits observation counts, and outputs structured artifacts.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr


def run_statistical_audit(
    base_dir: Path = Path("Dataset/gate_a4_3"),
    seed: int = 42,
    bootstrap_b: int = 10000,
    permutation_m: int = 100000,
):
    print("=== STARTING GATE A4.3 STATISTICAL INTEGRITY & REPRODUCIBILITY AUDIT ===")

    # 1. Load profile metrics
    prof_path = base_dir / "metrics" / "argo_profile_metrics.csv"
    if not prof_path.exists():
        raise FileNotFoundError(f"Missing {prof_path}")
    prof_df = pd.read_csv(prof_path)

    # 2. Verify platform & profile counts
    raw_path = base_dir / "data" / "argo_raw_bob_20200316_20200331.csv"
    raw_df = pd.read_csv(raw_path, skiprows=[1])
    raw_platforms = sorted([int(p) for p in raw_df["platform_number"].dropna().unique()])
    raw_profiles = (
        raw_df.groupby(["platform_number", "time"])
        .size()
        .reset_index(name="n_obs")
    )

    colloc_platforms = sorted([int(p) for p in prof_df["platform_number"].unique()])
    colloc_profiles_count = len(prof_df)

    excluded_platforms = sorted(list(set(raw_platforms) - set(colloc_platforms)))

    print(f"Raw ERDDAP float platforms count: {len(raw_platforms)}: {raw_platforms}")
    print(f"Raw profiles count: {len(raw_profiles)}")
    print(f"Collocated float platforms count: {len(colloc_platforms)}: {colloc_platforms}")
    print(f"Collocated profiles count: {colloc_profiles_count}")
    print(f"Entirely excluded platforms: {excluded_platforms}")

    # 3. Load prediction NetCDF and verify counts
    nc_path = base_dir / "predictions" / "argo_collocated_predictions.nc"
    ds = xr.open_dataset(nc_path)
    total_paired_obs = ds.sizes["obs_index"]
    assert (
        prof_df["n_obs"].sum() == total_paired_obs
    ), f"Count mismatch: {prof_df['n_obs'].sum()} vs {total_paired_obs}"

    # 4. Verify depth-wise counts
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    tolerances = [3.0, 3.0, 4.0, 5.0, 5.0, 7.5, 10.0, 12.5, 12.5, 15.0, 20.0, 35.0, 50.0, 75.0, 100.0]
    depth_csv_path = base_dir / "metrics" / "argo_depth_metrics.csv"
    depth_df = pd.read_csv(depth_csv_path)

    z_obs = ds["depth_m"].values
    depth_verifications = []
    for target_z, tol, row in zip(depths, tolerances, depth_df.itertuples()):
        mask = np.abs(z_obs - target_z) <= tol
        actual_count = int(np.sum(mask))
        match = actual_count == int(row.n_observations)
        depth_verifications.append(
            {
                "canonical_depth_m": target_z,
                "tolerance_m": tol,
                "expected_count": int(row.n_observations),
                "actual_count": actual_count,
                "verified": match,
            }
        )
        assert match, f"Depth mismatch at {target_z}m: {actual_count} vs {row.n_observations}"

    # 5. Paired profile-level inference
    # difference = OceanEmbed_RMSE - Climatology_RMSE
    prof_df["rmse_difference"] = prof_df["model_rmse"] - prof_df["clim_rmse"]
    prof_df["mae_difference"] = prof_df["model_mae"] - prof_df["clim_mae"]
    prof_df["bias_difference"] = prof_df["model_bias"] - prof_df["clim_bias"]
    prof_df["relative_rmse_reduction_pct"] = (
        (prof_df["clim_rmse"] - prof_df["model_rmse"]) / prof_df["clim_rmse"]
    ) * 100.0

    diff = prof_df["rmse_difference"].values
    n_profiles = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    median_diff = float(np.median(diff))
    n_superior = int(np.sum(diff < 0))

    # Bootstrap B = 10,000 resamples over profile IDs
    rng = np.random.default_rng(seed)
    boot_diff_means = np.zeros(bootstrap_b)
    for b in range(bootstrap_b):
        boot_sample = rng.choice(diff, size=n_profiles, replace=True)
        boot_diff_means[b] = np.mean(boot_sample)

    ci_lower = float(np.percentile(boot_diff_means, 2.5))
    ci_upper = float(np.percentile(boot_diff_means, 97.5))
    prob_diff_lt_0 = float(np.mean(boot_diff_means < 0))

    # Permutation / sign-flip test (M = 100,000)
    # Null hypothesis H0: E[diff] = 0 (symmetric under sign flip)
    obs_test_stat = abs(mean_diff)
    perm_means = np.zeros(permutation_m)
    for m in range(permutation_m):
        signs = rng.choice([-1.0, 1.0], size=n_profiles)
        perm_means[m] = np.mean(diff * signs)

    perm_two_sided_p = float(np.mean(np.abs(perm_means) >= obs_test_stat))
    perm_one_sided_p = float(np.mean(perm_means <= mean_diff))

    # Paired Student t and Wilcoxon signed rank
    t_stat, t_pval = stats.ttest_rel(prof_df["model_rmse"], prof_df["clim_rmse"])
    w_stat, w_pval = stats.wilcoxon(prof_df["model_rmse"], prof_df["clim_rmse"])

    # 6. Marginal Confidence Intervals over profile metrics (B = 10,000)
    def compute_boot_ci(vals: np.ndarray):
        means = np.zeros(bootstrap_b)
        for b in range(bootstrap_b):
            means[b] = np.mean(rng.choice(vals, size=len(vals), replace=True))
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    m_rmse_ci = compute_boot_ci(prof_df["model_rmse"].values)
    m_mae_ci = compute_boot_ci(prof_df["model_mae"].values)
    m_bias_ci = compute_boot_ci(prof_df["model_bias"].values)

    c_rmse_ci = compute_boot_ci(prof_df["clim_rmse"].values)
    c_mae_ci = compute_boot_ci(prof_df["clim_mae"].values)
    c_bias_ci = compute_boot_ci(prof_df["clim_bias"].values)

    g_rmse_ci = compute_boot_ci(prof_df["glorys_rmse"].values)
    g_mae_ci = compute_boot_ci(prof_df["glorys_mae"].values)
    g_bias_ci = compute_boot_ci(prof_df["glorys_bias"].values)

    # 7. Write argo_paired_inference.csv
    paired_csv_path = base_dir / "metrics" / "argo_paired_inference.csv"
    cols_order = [
        "profile_id",
        "platform_number",
        "date",
        "timestamp",
        "latitude",
        "longitude",
        "n_obs",
        "model_rmse",
        "clim_rmse",
        "rmse_difference",
        "relative_rmse_reduction_pct",
        "model_mae",
        "clim_mae",
        "mae_difference",
        "model_bias",
        "clim_bias",
        "bias_difference",
        "glorys_rmse",
        "glorys_mae",
        "glorys_bias",
    ]
    prof_df[cols_order].to_csv(paired_csv_path, index=False)
    print(f"Wrote paired inference CSV to {paired_csv_path}")

    # 8. Write argo_paired_inference.json
    inference_results = {
        "audit_metadata": {
            "patch_gate": "A4.3 Statistical Integrity & Reproducibility Patch",
            "model_evaluated": "OceanEmbed v1-Local (Frozen Champion from A4.1)",
            "observational_reference": "Independent In Situ ARGO Profiling Floats",
            "reanalysis_reference": "GLORYS12V1 Reanalysis-Derived Reference",
            "random_seed": seed,
            "bootstrap_replicates_B": bootstrap_b,
            "permutation_samples_M": permutation_m,
        },
        "sample_counts": {
            "raw_erddap_platforms_count": len(raw_platforms),
            "raw_erddap_platforms_list": raw_platforms,
            "raw_erddap_profiles_count": len(raw_profiles),
            "collocated_platforms_count": len(colloc_platforms),
            "collocated_platforms_list": colloc_platforms,
            "collocated_profiles_count": colloc_profiles_count,
            "boundary_excluded_profiles_count": len(raw_profiles) - colloc_profiles_count,
            "boundary_excluded_platforms": excluded_platforms,
            "boundary_partially_excluded_platforms": [2902772],
            "total_paired_observations": int(total_paired_obs),
        },
        "paired_profile_rmse_inference": {
            "metric": "OceanEmbed_RMSE - Climatology_RMSE",
            "n_profiles": n_profiles,
            "mean_paired_difference_degC": round(mean_diff, 4),
            "std_paired_difference_degC": round(std_diff, 4),
            "median_paired_difference_degC": round(median_diff, 4),
            "profiles_superior_count": n_superior,
            "profiles_superior_fraction": round(n_superior / n_profiles, 4),
            "bootstrap_b": bootstrap_b,
            "bootstrap_mean_degC": round(float(np.mean(boot_diff_means)), 4),
            "bootstrap_95_ci_degC": [round(ci_lower, 4), round(ci_upper, 4)],
            "bootstrap_prob_difference_lt_0": prob_diff_lt_0,
            "permutation_m": permutation_m,
            "permutation_two_sided_p_value": perm_two_sided_p,
            "permutation_one_sided_p_value": perm_one_sided_p,
            "paired_student_t_test": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": float(f"{t_pval:.4e}"),
            },
            "wilcoxon_signed_rank_test": {
                "w_statistic": round(float(w_stat), 4),
                "p_value": float(f"{w_pval:.4e}"),
            },
        },
        "profile_level_metrics_with_bootstrap_95_ci": {
            "model": {
                "rmse_mean": round(float(np.mean(prof_df["model_rmse"])), 4),
                "rmse_95_ci": [round(m_rmse_ci[0], 4), round(m_rmse_ci[1], 4)],
                "mae_mean": round(float(np.mean(prof_df["model_mae"])), 4),
                "mae_95_ci": [round(m_mae_ci[0], 4), round(m_mae_ci[1], 4)],
                "bias_mean": round(float(np.mean(prof_df["model_bias"])), 4),
                "bias_95_ci": [round(m_bias_ci[0], 4), round(m_bias_ci[1], 4)],
            },
            "climatology": {
                "rmse_mean": round(float(np.mean(prof_df["clim_rmse"])), 4),
                "rmse_95_ci": [round(c_rmse_ci[0], 4), round(c_rmse_ci[1], 4)],
                "mae_mean": round(float(np.mean(prof_df["clim_mae"])), 4),
                "mae_95_ci": [round(c_mae_ci[0], 4), round(c_mae_ci[1], 4)],
                "bias_mean": round(float(np.mean(prof_df["clim_bias"])), 4),
                "bias_95_ci": [round(c_bias_ci[0], 4), round(c_bias_ci[1], 4)],
            },
            "glorys_reference": {
                "rmse_mean": round(float(np.mean(prof_df["glorys_rmse"])), 4),
                "rmse_95_ci": [round(g_rmse_ci[0], 4), round(g_rmse_ci[1], 4)],
                "mae_mean": round(float(np.mean(prof_df["glorys_mae"])), 4),
                "mae_95_ci": [round(g_mae_ci[0], 4), round(g_mae_ci[1], 4)],
                "bias_mean": round(float(np.mean(prof_df["glorys_bias"])), 4),
                "bias_95_ci": [round(g_bias_ci[0], 4), round(g_bias_ci[1], 4)],
            },
        },
        "depth_wise_count_verifications": depth_verifications,
    }

    paired_json_path = base_dir / "metrics" / "argo_paired_inference.json"
    with open(paired_json_path, "w") as f:
        json.dump(inference_results, f, indent=2)
    print(f"Wrote paired inference JSON to {paired_json_path}")

    print("=== AUDIT AND PATCH COMPUTATION COMPLETE ===")
    return inference_results


if __name__ == "__main__":
    run_statistical_audit()
