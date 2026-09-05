"""
Unit Tests for PS-Aligned 15-Depth Vertical Interpolation Pipeline
================================================================
Enforces:
  1. Exact target-depth ordering ([0, 5, 10, ..., 1000] m).
  2. Exactly 15 output depths.
  3. 1000 m interpolation bracket (902.3393 m < 1000 m < 1062.4399 m).
  4. 0 m nearest-level handling (~0.494 m, zero upward extrapolation).
  5. NaN / ocean validity mask preservation.
"""

import unittest
from pathlib import Path
import numpy as np

from src.constants import (
    REQUIRED_DEPTHS_M,
    OUTPUT_DEPTHS,
    GLORYS_SURFACE_LEVEL_APPROX_M,
)
from src.preprocess import (
    validate_native_depths,
    compute_interpolation_weights,
    interpolate_field_to_target_depths,
    preprocess_glorys_dataset,
)

# Reference 36 native GLORYS depths (0.494 m to 1062.44 m)
NATIVE_36_DEPTHS = np.array([
    0.494025, 1.541375, 2.645669, 3.819495, 5.078224,
    6.440614, 7.92956, 9.572997, 11.405, 13.46714,
    15.81007, 18.49556, 21.59882, 25.21141, 29.44473,
    34.43415, 40.34405, 47.37369, 55.76429, 65.80727,
    77.85385, 92.32607, 109.7293, 130.666, 155.8507,
    186.1256, 222.4752, 266.0403, 318.1274, 380.213,
    453.9377, 541.0889, 643.5668, 763.3331, 902.3393,
    1062.4399
], dtype=np.float64)


class TestVerticalInterpolation(unittest.TestCase):
    """Test suite verifying vertical interpolation requirements for SIH26066."""

    def test_01_exact_target_depth_ordering(self):
        """Verify target depth list has exact canonical ordering and values."""
        canonical = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.assertEqual(REQUIRED_DEPTHS_M, canonical)
        self.assertEqual(len(REQUIRED_DEPTHS_M), 15)
        self.assertEqual(OUTPUT_DEPTHS, 15)

        # Ensure strictly ascending
        diffs = np.diff(REQUIRED_DEPTHS_M)
        self.assertTrue(np.all(diffs > 0), "Target depths must be strictly increasing.")

        # Ensure disallowed non-canonical depths are absent
        self.assertNotIn(400, REQUIRED_DEPTHS_M)
        self.assertNotIn(750, REQUIRED_DEPTHS_M)

    def test_02_fifteen_output_depths(self):
        """Verify interpolation produces exactly 15 output depths."""
        weights = compute_interpolation_weights(NATIVE_36_DEPTHS, REQUIRED_DEPTHS_M)
        self.assertEqual(len(weights), 15)

        # Synthetic 4D field: (time=2, depth=36, lat=4, lon=5)
        data = np.ones((2, 36, 4, 5), dtype=np.float32)
        interp, mask = interpolate_field_to_target_depths(data, weights, depth_axis=1)

        self.assertEqual(interp.shape, (2, 15, 4, 5))
        self.assertEqual(mask.shape, (2, 15, 4, 5))

    def test_03_1000m_interpolation_bracket(self):
        """Verify 1000 m target is strictly bracketed: 902.3393 m < 1000 m < 1062.4399 m."""
        val = validate_native_depths(NATIVE_36_DEPTHS, REQUIRED_DEPTHS_M)
        self.assertTrue(val["valid"])
        self.assertEqual(val["bracket_1000m_lower_idx"], 34)
        self.assertEqual(val["bracket_1000m_upper_idx"], 35)

        z_low = val["bracket_1000m_lower_m"]
        z_high = val["bracket_1000m_upper_m"]
        self.assertAlmostEqual(z_low, 902.3393, places=3)
        self.assertAlmostEqual(z_high, 1062.4399, places=3)
        self.assertTrue(z_low < 1000.0 < z_high, "1000 m is NOT strictly bracketed!")

        # Check calculated alpha weight
        weights = compute_interpolation_weights(NATIVE_36_DEPTHS, REQUIRED_DEPTHS_M)
        w_1000 = weights[-1]
        self.assertEqual(w_1000["target_depth_m"], 1000.0)
        self.assertEqual(w_1000["lower_idx"], 34)
        self.assertEqual(w_1000["upper_idx"], 35)

        expected_alpha = (1000.0 - z_low) / (z_high - z_low)
        self.assertAlmostEqual(w_1000["alpha"], expected_alpha, places=5)
        self.assertAlmostEqual(w_1000["alpha"], 0.609996, places=4)

        # Synthetic linear profile: T(z) = 30 - 0.02 * z
        # At 1000 m: T = 30 - 20 = 10.0
        synthetic_profile = 30.0 - 0.02 * NATIVE_36_DEPTHS
        data = synthetic_profile[None, :, None, None].astype(np.float32)  # (1, 36, 1, 1)
        interp, _ = interpolate_field_to_target_depths(data, weights, depth_axis=1)

        t_1000_interp = float(interp[0, 14, 0, 0])
        self.assertAlmostEqual(t_1000_interp, 10.0, places=4)

    def test_04_0m_nearest_level_handling(self):
        """Verify 0 m target uses nearest native surface level (~0.494 m) without upward extrapolation."""
        weights = compute_interpolation_weights(NATIVE_36_DEPTHS, REQUIRED_DEPTHS_M)
        w_0 = weights[0]

        self.assertEqual(w_0["target_depth_m"], 0.0)
        self.assertEqual(w_0["lower_idx"], 0)
        self.assertEqual(w_0["upper_idx"], 0)
        self.assertEqual(w_0["alpha"], 0.0)
        self.assertEqual(w_0["method"], "nearest_surface_level")

        # Test with arbitrary surface value: T(0.494m) = 28.5
        data = np.zeros((1, 36, 1, 1), dtype=np.float32)
        data[0, 0, 0, 0] = 28.5
        data[0, 1, 0, 0] = 27.0  # slope should NOT be extrapolated upward

        interp, _ = interpolate_field_to_target_depths(data, weights, depth_axis=1)
        self.assertEqual(float(interp[0, 0, 0, 0]), 28.5)

    def test_05_nan_and_mask_preservation(self):
        """Verify that NaNs are strictly preserved and ocean validity mask is exact."""
        weights = compute_interpolation_weights(NATIVE_36_DEPTHS, REQUIRED_DEPTHS_M)

        # Create synthetic column with intentional NaNs
        # Pixel (0, 0): all valid
        # Pixel (0, 1): land (all NaN)
        # Pixel (1, 0): shallow bathymetry (NaN below index 20, ~77m)
        data = np.ones((1, 36, 2, 2), dtype=np.float32)
        data[:, :, 0, 1] = np.nan  # land column
        data[:, 21:, 1, 0] = np.nan  # shallow seafloor below 77m

        interp, mask = interpolate_field_to_target_depths(data, weights, depth_axis=1)

        # 1. Pixel (0, 0) should be 100% valid at all 15 depths
        self.assertTrue(np.all(~np.isnan(interp[0, :, 0, 0])))
        self.assertTrue(np.all(mask[0, :, 0, 0]))

        # 2. Pixel (0, 1) should be 100% NaN and mask False at all 15 depths
        self.assertTrue(np.all(np.isnan(interp[0, :, 0, 1])))
        self.assertFalse(np.any(mask[0, :, 0, 1]))

        # 3. Pixel (1, 0) should be valid for shallow levels (0-75m) and NaN for deep levels (>=100m)
        # Target index 6 is 75m (native levels 19 & 20, both valid) -> valid
        self.assertFalse(np.isnan(interp[0, 6, 1, 0]))
        self.assertTrue(mask[0, 6, 1, 0])

        # Target index 7 is 100m (native levels 21 & 22, level 21 is NaN) -> must be NaN
        self.assertTrue(np.isnan(interp[0, 7, 1, 0]))
        self.assertFalse(mask[0, 7, 1, 0])

    def test_06_real_pilot_nc_preprocessing(self):
        """Verify preprocessing on actual pilot NetCDF file if present."""
        pilot_nc = Path("Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn_1100m.nc")
        if not pilot_nc.exists():
            self.skipTest(f"Pilot NetCDF not found: {pilot_nc}")

        out_ds, summary = preprocess_glorys_dataset(pilot_nc)

        self.assertEqual(summary["input_dimensions"]["depth"], 36)
        self.assertEqual(summary["output_dimensions"]["depth"], 15)
        self.assertEqual(summary["target_depths"], REQUIRED_DEPTHS_M)

        # Check coverage is realistic (< 100% ocean coverage due to land / shallow margins)
        thetao_cov = summary["coverage"]["thetao"]
        for row in thetao_cov:
            cov = row["valid_coverage_pct"]
            self.assertGreater(cov, 90.0, f"Depth {row['target_depth_m']} coverage too low: {cov}%")
            self.assertLess(cov, 100.0, f"Depth {row['target_depth_m']} coverage falsely 100%: {cov}%")


if __name__ == "__main__":
    unittest.main()
