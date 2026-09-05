"""
Unit Tests for Gate A1.4: 7-Channel Surface Input Data Audit
===========================================================
Validates:
  1. Exact 7 required channels and canonical ordering.
  2. Primary source mapping per official SIH26066 specification.
  3. Accurate classification of primary satellite channels as BLOCKED on disk.
  4. Accurate classification of GLORYS surface layers as REANALYSIS_FALLBACK.
  5. Enforcement that GLORYS does not contain atmospheric wind variables.
  6. Target analysis domain and grid specifications.
"""

import unittest
from pathlib import Path
import numpy as np

from src.constants import (
    CANONICAL_SURFACE_VARIABLES,
    INPUT_CHANNELS,
    CHANNEL_NAME_MAP,
    GATE_A1_PILOT_BOX,
)
from src.surface_audit import (
    SURFACE_SPECIFICATIONS,
    audit_local_dataset_inventory,
    evaluate_channel_availability,
)


class TestSurfaceInputAudit(unittest.TestCase):
    """Test suite validating the 7-channel surface input audit requirements."""

    def test_01_canonical_seven_channels(self):
        """Verify exact 7 channels, ordering, and names."""
        expected = [
            "sst",
            "sss",
            "ssh",
            "u_current",
            "v_current",
            "u_wind",
            "v_wind",
        ]
        self.assertEqual(CANONICAL_SURFACE_VARIABLES, expected)
        self.assertEqual(len(CANONICAL_SURFACE_VARIABLES), 7)
        self.assertEqual(INPUT_CHANNELS, 7)
        self.assertEqual(len(SURFACE_SPECIFICATIONS), 7)

        for idx, name in enumerate(expected):
            self.assertIn(idx, CHANNEL_NAME_MAP)
            self.assertIn(name, SURFACE_SPECIFICATIONS)
            self.assertEqual(SURFACE_SPECIFICATIONS[name]["channel_index"], idx)

    def test_02_primary_source_hierarchy(self):
        """Verify primary sources adhere to official SIH26066 specifications."""
        # SST -> OSTIA
        self.assertIn("OSTIA", SURFACE_SPECIFICATIONS["sst"]["primary_source"])
        # SSS -> SMAP / SMOS
        self.assertTrue(
            "SMAP" in SURFACE_SPECIFICATIONS["sss"]["primary_source"]
            or "SMOS" in SURFACE_SPECIFICATIONS["sss"]["primary_source"]
        )
        # SSH -> DUACS
        self.assertIn("DUACS", SURFACE_SPECIFICATIONS["ssh"]["primary_source"])
        # Currents -> OSCAR
        self.assertIn("OSCAR", SURFACE_SPECIFICATIONS["u_current"]["primary_source"])
        self.assertIn("OSCAR", SURFACE_SPECIFICATIONS["v_current"]["primary_source"])
        # Winds -> CCMP / ASCAT
        self.assertTrue(
            "CCMP" in SURFACE_SPECIFICATIONS["u_wind"]["primary_source"]
            or "ASCAT" in SURFACE_SPECIFICATIONS["u_wind"]["primary_source"]
        )

    def test_03_target_domain_specifications(self):
        """Verify target domain is the pilot box 12-18N, 85-93E at 0.25 deg."""
        self.assertEqual(GATE_A1_PILOT_BOX["lat_min"], 12.0)
        self.assertEqual(GATE_A1_PILOT_BOX["lat_max"], 18.0)
        self.assertEqual(GATE_A1_PILOT_BOX["lon_min"], 85.0)
        self.assertEqual(GATE_A1_PILOT_BOX["lon_max"], 93.0)
        self.assertEqual(GATE_A1_PILOT_BOX["grid_resolution"], 0.25)

    def test_04_audit_evaluates_local_inventory_accurately(self):
        """Verify audit programmatically scans Dataset/ and detects temporal discordance."""
        inventory = audit_local_dataset_inventory("Dataset")
        self.assertGreater(len(inventory), 0, "No files found in Dataset/ directory")

        audit = evaluate_channel_availability(inventory)

        # In SATELLITE_OBSERVATION mode, all 7 channels must be BLOCKED on current disk
        for ch_name, ch_res in audit["channel_audit"].items():
            self.assertEqual(
                ch_res["primary_status"],
                "BLOCKED",
                f"Channel {ch_name} primary status should be BLOCKED due to temporal discordance or missing file",
            )
            self.assertGreater(
                len(ch_res["blocking_reasons"]),
                0,
                f"Channel {ch_name} must provide explicit blocking justification",
            )

        self.assertEqual(audit["summary"]["primary_satellite_available"], 0)
        self.assertEqual(audit["summary"]["primary_satellite_blocked"], 7)

    def test_05_reanalysis_fallback_classification(self):
        """Verify GLORYS provides 5 ocean channels as FALLBACK but cannot provide winds."""
        inventory = audit_local_dataset_inventory("Dataset")
        audit = evaluate_channel_availability(inventory)

        channel_results = audit["channel_audit"]

        # 5 Hydrodynamic channels available from GLORYS preprocessed dataset
        ocean_channels = ["sst", "sss", "ssh", "u_current", "v_current"]
        for ch in ocean_channels:
            self.assertEqual(
                channel_results[ch]["fallback_status"],
                "AVAILABLE",
                f"Ocean channel {ch} should be AVAILABLE under REANALYSIS_FALLBACK",
            )
            # Crucial: Must be documented as FALLBACK, NOT primary satellite
            self.assertEqual(channel_results[ch]["primary_status"], "BLOCKED")

        # 2 Wind channels must be strictly BLOCKED in GLORYS
        wind_channels = ["u_wind", "v_wind"]
        for ch in wind_channels:
            self.assertEqual(
                channel_results[ch]["fallback_status"],
                "BLOCKED",
                f"Wind channel {ch} must be BLOCKED in GLORYS (ocean-only model)",
            )

        self.assertEqual(audit["summary"]["fallback_reanalysis_available"], 5)
        self.assertEqual(audit["summary"]["fallback_reanalysis_blocked"], 2)


if __name__ == "__main__":
    unittest.main()
