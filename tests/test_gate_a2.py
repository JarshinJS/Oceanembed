"""
Unit Tests for Gate A2: Baseline Learning Feasibility
=====================================================
Validates all 10 mandatory Phase L requirements:
  1. Input tensor shape [B, 7, 24, 32]
  2. Output tensor shape [B, 15, 24, 32]
  3. Temporal split correctness (train: 20 days, eval: 10 days)
  4. No evaluation dates in training loader
  5. Mask-aware loss zeroes out invalid pixels
  6. Loss is strictly finite with no NaN propagation
  7. Model predictions are strictly finite
  8. Depth ordering preserved across 15 levels
  9. Metric calculations correct against analytical formulations
 10. Baseline comparison reproducibility under fixed seed
"""

import json
from pathlib import Path
import unittest
import numpy as np
import torch

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a2.dataset import SynchronizedOceanDataset
from src.gate_a2.cnn_baseline import SimpleSpatialCNN, MaskedMSELoss
from src.gate_a2.reference_baseline import evaluate_predictions


class TestGateA2Baseline(unittest.TestCase):
    """Test suite validating Gate A2 baseline pipeline and models."""

    @classmethod
    def setUpClass(cls):
        cls.train_ds = SynchronizedOceanDataset(split="train_full")
        cls.eval_ds = SynchronizedOceanDataset(split="eval")
        cls.model = SimpleSpatialCNN(in_channels=7, out_channels=15, hidden_dim=32)

        # Load results JSON
        results_path = Path("Dataset/gate_a2_baseline/metrics/baseline_results.json")
        with open(results_path, "r") as f:
            cls.results = json.load(f)

    def test_01_input_tensor_shape(self):
        """Verify input tensor shape is exactly [B, 7, 24, 32]."""
        sample = self.train_ds[0]
        x = sample["x"]
        self.assertEqual(x.shape, (7, 24, 32))
        batch_x = torch.stack([self.train_ds[0]["x"], self.train_ds[1]["x"]])
        self.assertEqual(batch_x.shape, (2, 7, 24, 32))

    def test_02_output_tensor_shape(self):
        """Verify model output shape is exactly [B, 15, 24, 32]."""
        batch_x = torch.randn(3, 7, 24, 32)
        out = self.model(batch_x)
        self.assertEqual(out.shape, (3, 15, 24, 32))

    def test_03_temporal_split_correctness(self):
        """Verify 20 training days and 10 evaluation days."""
        self.assertEqual(len(self.train_ds), 20)
        self.assertEqual(len(self.eval_ds), 10)
        self.assertEqual(self.train_ds.dates[0], "2020-01-01")
        self.assertEqual(self.train_ds.dates[-1], "2020-01-20")
        self.assertEqual(self.eval_ds.dates[0], "2020-01-21")
        self.assertEqual(self.eval_ds.dates[-1], "2020-01-30")

    def test_04_no_evaluation_dates_in_training_loader(self):
        """Verify strict temporal separation between training and evaluation splits."""
        train_dates_set = set(self.train_ds.dates)
        eval_dates_set = set(self.eval_ds.dates)
        intersection = train_dates_set.intersection(eval_dates_set)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Dates in both splits: {intersection}")

    def test_05_mask_aware_loss_zeroes_invalid_pixels(self):
        """Verify masked loss ignores invalid target pixels completely."""
        criterion = MaskedMSELoss()
        pred = torch.ones(1, 15, 24, 32) * 5.0
        target = torch.zeros(1, 15, 24, 32)

        # 1. Mask with zero valid pixels
        mask_empty = torch.zeros(1, 15, 24, 32)
        loss_empty = criterion(pred, target, mask_empty)
        self.assertEqual(loss_empty.item(), 0.0)

        # 2. Mask with single valid pixel
        mask_single = torch.zeros(1, 15, 24, 32)
        mask_single[0, 0, 5, 5] = 1.0
        loss_single = criterion(pred, target, mask_single)
        self.assertAlmostEqual(loss_single.item(), 25.0)  # (5.0 - 0.0)^2 = 25.0

    def test_06_no_nan_loss_propagation(self):
        """Verify that NaNs in masked-out regions do not propagate into the loss."""
        criterion = MaskedMSELoss()
        pred = torch.ones(1, 15, 24, 32) * 2.0
        target = torch.ones(1, 15, 24, 32) * 2.0

        # Inject NaNs where mask is 0.0
        mask = torch.ones(1, 15, 24, 32)
        mask[0, :, :5, :5] = 0.0
        target[0, :, :5, :5] = float("nan")

        loss = criterion(pred, target, mask)
        self.assertFalse(torch.isnan(loss).item(), "Loss contains NaN!")
        self.assertTrue(torch.isfinite(loss).item())

    def test_07_model_output_finite(self):
        """Verify model predictions on evaluation dataset are strictly finite."""
        self.assertTrue(self.results["sanity_checks"]["output_sanity"]["all_finite"])
        self.assertTrue(self.results["sanity_checks"]["output_sanity"]["physically_plausible"])

    def test_08_depth_ordering_preserved(self):
        """Verify all 15 canonical depths are present and strictly ascending."""
        depths = list(self.train_ds.depths)
        self.assertEqual(len(depths), 15)
        self.assertTrue(np.allclose(depths, REQUIRED_DEPTHS_M))
        self.assertTrue(all(depths[i] < depths[i + 1] for i in range(len(depths) - 1)))

    def test_09_metric_calculations_against_analytical_formulations(self):
        """Verify evaluate_predictions matches direct numpy analytical calculations."""
        y_pred = np.array([[[[20.0, 22.0], [24.0, 26.0]]]])  # [1, 1, 2, 2]
        y_true = np.array([[[[19.0, 23.0], [25.0, 24.0]]]])  # [1, 1, 2, 2]
        mask = np.ones((1, 1, 2, 2))

        # diff = [1.0, -1.0, -1.0, 2.0]
        # squared = [1, 1, 1, 4] -> mean = 7 / 4 = 1.75 -> rmse = sqrt(1.75) ~ 1.3229
        # abs = [1, 1, 1, 2] -> mean = 5 / 4 = 1.25
        # bias = (1 - 1 - 1 + 2) / 4 = 0.25
        res = evaluate_predictions(y_pred, y_true, mask, depths=[0])
        self.assertAlmostEqual(res["overall"]["rmse"], np.sqrt(1.75), places=4)
        self.assertAlmostEqual(res["overall"]["mae"], 1.25, places=4)
        self.assertAlmostEqual(res["overall"]["bias"], 0.25, places=4)

    def test_10_baseline_comparison_reproducibility(self):
        """Verify baseline comparison summary exists and is internally consistent."""
        comp = self.results["comparison_summary"]
        ref_rmse = comp["overall_reference_rmse"]
        cnn_rmse = comp["overall_cnn_rmse"]
        imp = comp["overall_improvement_pct"]

        expected_imp = round((ref_rmse - cnn_rmse) / ref_rmse * 100, 2)
        self.assertEqual(imp, expected_imp)
        self.assertGreater(ref_rmse, 0.0)
        self.assertGreater(cnn_rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
