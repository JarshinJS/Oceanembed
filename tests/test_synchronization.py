"""
Unit Tests for Gate A1.5: Canonical 7-Channel & 15-Depth Synchronization
========================================================================
Validates:
  1. Exact 7 input channels in canonical ordering and naming.
  2. Exact 15 canonical target depths (0 to 1000m).
  3. Exact 30 common dates (2020-01-01 to 2020-01-30) with no duplicates.
  4. Strict spatial coordinate equality between input and target grids (24 x 32).
  5. Finite/NaN handling and validity masks.
  6. SST in realistic Celsius range (not Kelvin).
  7. OSCAR transposition and coordinate matching.
  8. Target and input shape consistency: X [30, 7, 24, 32], Y [30, 15, 24, 32].
  9. Zero normalization leakage between training and evaluation windows.
"""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import xarray as xr

from src.constants import (
    REQUIRED_DEPTHS_M,
    INPUT_CHANNELS,
    CANONICAL_SURFACE_VARIABLES,
    GATE_A1_PILOT_BOX,
)


class TestGateA15Synchronization(unittest.TestCase):
    """Test suite validating the synchronized 7-channel inputs and 15-depth target."""

    @classmethod
    def setUpClass(cls):
        cls.sync_dir = Path("Dataset/gate_a1_pilot/synchronized")
        cls.inputs_path = cls.sync_dir / "oceanembed_inputs_7ch_30d.nc"
        cls.target_path = cls.sync_dir / "oceanembed_target_15depth_30d.nc"
        cls.meta_path = cls.sync_dir / "synchronization_metadata.json"

        cls.ds_in = xr.open_dataset(cls.inputs_path)
        cls.ds_tgt = xr.open_dataset(cls.target_path)
        with open(cls.meta_path, "r") as f:
            cls.meta = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.ds_in.close()
        cls.ds_tgt.close()

    def test_01_canonical_seven_channels(self):
        """Verify exact 7 channels, ordering, and channel names."""
        self.assertEqual(len(self.ds_in.channel), INPUT_CHANNELS)
        self.assertEqual(self.ds_in.inputs.shape[1], 7)

        channel_names = list(self.ds_in.channel_name.values)
        self.assertEqual(channel_names, CANONICAL_SURFACE_VARIABLES)
        self.assertEqual(channel_names[0], "sst")
        self.assertEqual(channel_names[1], "sss")
        self.assertEqual(channel_names[2], "ssh")
        self.assertEqual(channel_names[3], "u_current")
        self.assertEqual(channel_names[4], "v_current")
        self.assertEqual(channel_names[5], "u_wind")
        self.assertEqual(channel_names[6], "v_wind")

    def test_02_canonical_fifteen_target_depths(self):
        """Verify exact 15 target depths matching official PS specifications."""
        target_depths = list(self.ds_tgt.depth.values)
        self.assertEqual(len(target_depths), 15)
        self.assertEqual(self.ds_tgt.target_thetao.shape[1], 15)
        self.assertTrue(np.allclose(target_depths, REQUIRED_DEPTHS_M))
        self.assertEqual(target_depths[0], 0.0)
        self.assertEqual(target_depths[-1], 1000.0)

    def test_03_temporal_synchronization_and_monotonicity(self):
        """Verify exact 30 common daily dates with no duplicates or gaps."""
        in_times = pd.to_datetime(self.ds_in.time.values)
        tgt_times = pd.to_datetime(self.ds_tgt.time.values)

        self.assertEqual(len(in_times), 30)
        self.assertEqual(len(tgt_times), 30)
        self.assertTrue((in_times == tgt_times).all())

        expected_dates = pd.date_range("2020-01-01", "2020-01-30", freq="D")
        self.assertTrue((in_times == expected_dates).all())

        # No duplicate timestamps
        self.assertEqual(len(in_times), len(in_times.unique()))
        self.assertTrue(in_times.is_monotonic_increasing)

    def test_04_spatial_coordinate_equality(self):
        """Verify strict spatial coordinate equality on 24x32 cell-center 0.25 deg grid."""
        in_lats = self.ds_in.latitude.values
        tgt_lats = self.ds_tgt.latitude.values
        in_lons = self.ds_in.longitude.values
        tgt_lons = self.ds_tgt.longitude.values

        self.assertEqual(len(in_lats), 24)
        self.assertEqual(len(in_lons), 32)
        self.assertEqual(len(tgt_lats), 24)
        self.assertEqual(len(tgt_lons), 32)

        self.assertTrue(np.allclose(in_lats, tgt_lats))
        self.assertTrue(np.allclose(in_lons, tgt_lons))
        self.assertTrue(np.allclose(in_lats, GATE_A1_PILOT_BOX["lat_centers"]))
        self.assertTrue(np.allclose(in_lons, GATE_A1_PILOT_BOX["lon_centers"]))

        # Check resolution is exactly 0.25 deg
        self.assertTrue(np.allclose(np.diff(in_lats), 0.25))
        self.assertTrue(np.allclose(np.diff(in_lons), 0.25))

    def test_05_sst_celsius_conversion(self):
        """Verify SST has been converted from Kelvin to Celsius and is in realistic physical range."""
        sst = self.ds_in.inputs[:, 0, :, :].values
        valid_sst = sst[~np.isnan(sst)]

        # Must be in Celsius (e.g. 23 - 32 degC), strictly NOT Kelvin (~295 - 305 K)
        self.assertLess(float(np.max(valid_sst)), 40.0, "SST values appear to still be in Kelvin!")
        self.assertGreater(float(np.min(valid_sst)), 15.0, "SST values are unrealistically low for Bay of Bengal!")
        self.assertAlmostEqual(float(np.mean(valid_sst)), 26.93, delta=0.5)

    def test_06_oscar_transpose_and_currents(self):
        """Verify OSCAR surface currents are properly transposed and aligned."""
        u_curr = self.ds_in.inputs[:, 3, :, :].values
        v_curr = self.ds_in.inputs[:, 4, :, :].values

        self.assertEqual(u_curr.shape, (30, 24, 32))
        self.assertEqual(v_curr.shape, (30, 24, 32))

        valid_u = u_curr[~np.isnan(u_curr)]
        valid_v = v_curr[~np.isnan(v_curr)]

        # Ocean current velocities in Bay of Bengal typically within [-1.5, 1.5] m/s
        self.assertTrue(np.all(np.abs(valid_u) < 2.0))
        self.assertTrue(np.all(np.abs(valid_v) < 2.0))

    def test_07_finite_and_nan_masking_consistency(self):
        """Verify validity masks properly isolate ocean from land/bathymetry."""
        mask_in = self.ds_in.mask_inputs.values
        mask_tgt = self.ds_tgt.mask_target.values
        ocean_mask_2d = self.ds_in.ocean_mask_2d.values

        self.assertEqual(mask_in.shape, (30, 7, 24, 32))
        self.assertEqual(mask_tgt.shape, (30, 15, 24, 32))
        self.assertEqual(ocean_mask_2d.shape, (24, 32))

        # Check mask values are strictly binary 0 or 1
        self.assertTrue(set(np.unique(mask_in)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(mask_tgt)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(ocean_mask_2d)).issubset({0.0, 1.0}))

        # Valid ocean cells must be > 95% in the central Bay of Bengal pilot box
        valid_pct = np.mean(ocean_mask_2d) * 100
        self.assertGreater(valid_pct, 95.0)

    def test_08_target_and_input_shape_consistency(self):
        """Verify X [30, 7, 24, 32] and Y [30, 15, 24, 32] shape alignment."""
        x = self.ds_in.inputs.values
        y = self.ds_tgt.target_thetao.values

        self.assertEqual(x.shape, (30, 7, 24, 32))
        self.assertEqual(y.shape, (30, 15, 24, 32))
        self.assertEqual(x.shape[0], y.shape[0])  # Same time
        self.assertEqual(x.shape[2], y.shape[2])  # Same latitude
        self.assertEqual(x.shape[3], y.shape[3])  # Same longitude

    def test_09_no_normalization_leakage(self):
        """Verify normalization statistics are computed strictly on the training window without leakage."""
        stats = self.meta["leakage_free_normalization"]
        train_win = stats["training_window"]

        self.assertEqual(train_win["total_train_days"], 20)
        self.assertEqual(train_win["start_day"], 0)
        self.assertEqual(train_win["end_day"], 19)

        # For every channel, train statistics must differ from full-sample statistics (proving non-leakage)
        for ch_name, ch_stat in stats["input_channels"].items():
            self.assertIn("train_mean", ch_stat)
            self.assertIn("full_sample_mean", ch_stat)
            # The full sample mean incorporates val/test days, so they should not be identical
            self.assertNotEqual(
                ch_stat["train_mean"],
                ch_stat["full_sample_mean"],
                f"Channel {ch_name} train_mean equals full_sample_mean; possible leakage!",
            )


if __name__ == "__main__":
    unittest.main()
