"""
Gate A2 Baseline 0: Non-Neural Training Reference Climatology
============================================================
Constructs the non-neural spatial reference prediction derived strictly
from the training window (days 1–20, 2020-01-01 to 2020-01-20).

For each depth and location (d, i, j):
    prediction(d, i, j) = training-period mean temperature at that depth/location

Evaluates performance on the held-out evaluation window (days 21–30).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a2.dataset import SynchronizedOceanDataset


def compute_training_climatology(train_ds: SynchronizedOceanDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute training climatology mean field [15, 24, 32] and valid mask [15, 24, 32].
    Strictly uses training samples (days 0–19).
    """
    # Collect all raw target arrays from train_ds
    targets = []
    masks = []
    for item in train_ds:
        targets.append(item["y_raw"].numpy())
        masks.append(item["mask_y"].numpy())

    # [N_train, 15, 24, 32]
    Y_stack = np.stack(targets, axis=0)
    M_stack = np.stack(masks, axis=0)

    # Sum across time
    counts = np.sum(M_stack, axis=0)  # [15, 24, 32]
    masked_sum = np.sum(np.where(M_stack == 1.0, Y_stack, 0.0), axis=0)

    # Compute mean where valid
    clim_mean = np.zeros_like(counts, dtype=np.float32)
    valid_mask = counts > 0
    clim_mean[valid_mask] = masked_sum[valid_mask] / counts[valid_mask]

    return clim_mean, valid_mask.astype(np.float32)


def evaluate_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    depths: list = REQUIRED_DEPTHS_M,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics:
      - Overall RMSE, MAE, Mean Bias, Pearson Correlation
      - Per-depth metrics (for all 15 depths)
      - Thermocline metrics (50–200m)
      - Daily metrics
    
    Inputs:
      y_pred: [N_eval, 15, 24, 32]
      y_true: [N_eval, 15, 24, 32]
      mask:   [N_eval, 15, 24, 32] (1 where valid, 0 where invalid)
    """
    N_eval, n_depths, H, W = y_pred.shape
    valid_bool = mask == 1.0

    # Overall metrics
    pred_valid = y_pred[valid_bool]
    true_valid = y_true[valid_bool]

    diff = pred_valid - true_valid
    overall_rmse = float(np.sqrt(np.mean(diff ** 2)))
    overall_mae = float(np.mean(np.abs(diff)))
    overall_bias = float(np.mean(diff))

    if len(pred_valid) > 1 and np.std(pred_valid) > 1e-6 and np.std(true_valid) > 1e-6:
        corr, _ = pearsonr(pred_valid, true_valid)
        overall_corr = float(corr)
    else:
        overall_corr = 0.0

    # Per-depth metrics
    depth_results = {}
    thermocline_depths = [50, 75, 100, 125, 150, 200]
    thermocline_diffs = []
    thermocline_preds = []
    thermocline_trues = []

    for d_idx, depth_m in enumerate(depths):
        d_mask = valid_bool[:, d_idx, :, :]
        d_pred = y_pred[:, d_idx, :, :][d_mask]
        d_true = y_true[:, d_idx, :, :][d_mask]

        if len(d_pred) > 0:
            d_diff = d_pred - d_true
            d_rmse = float(np.sqrt(np.mean(d_diff ** 2)))
            d_mae = float(np.mean(np.abs(d_diff)))
            d_bias = float(np.mean(d_diff))

            if len(d_pred) > 1 and np.std(d_pred) > 1e-6 and np.std(d_true) > 1e-6:
                d_corr, _ = pearsonr(d_pred, d_true)
                d_corr = float(d_corr)
            else:
                d_corr = 0.0

            if depth_m in thermocline_depths:
                thermocline_diffs.extend(d_diff.tolist())
                thermocline_preds.extend(d_pred.tolist())
                thermocline_trues.extend(d_true.tolist())
        else:
            d_rmse = d_mae = d_bias = d_corr = 0.0

        depth_results[f"depth_{depth_m}m"] = {
            "depth_index": d_idx,
            "depth_meters": depth_m,
            "rmse": round(d_rmse, 4),
            "mae": round(d_mae, 4),
            "bias": round(d_bias, 4),
            "correlation": round(d_corr, 4),
            "valid_points": int(len(d_pred)),
        }

    # Thermocline summary (50 - 200 m)
    if thermocline_diffs:
        tc_diff_arr = np.array(thermocline_diffs)
        tc_rmse = float(np.sqrt(np.mean(tc_diff_arr ** 2)))
        tc_mae = float(np.mean(np.abs(tc_diff_arr)))
        tc_bias = float(np.mean(tc_diff_arr))
        if len(thermocline_preds) > 1 and np.std(thermocline_preds) > 1e-6:
            tc_corr, _ = pearsonr(thermocline_preds, thermocline_trues)
            tc_corr = float(tc_corr)
        else:
            tc_corr = 0.0
    else:
        tc_rmse = tc_mae = tc_bias = tc_corr = 0.0

    thermocline_results = {
        "depths_meters": thermocline_depths,
        "rmse": round(tc_rmse, 4),
        "mae": round(tc_mae, 4),
        "bias": round(tc_bias, 4),
        "correlation": round(tc_corr, 4),
    }

    # Daily metrics
    daily_results = {}
    for t_idx in range(N_eval):
        t_mask = valid_bool[t_idx]
        t_diff = y_pred[t_idx][t_mask] - y_true[t_idx][t_mask]
        t_rmse = float(np.sqrt(np.mean(t_diff ** 2)))
        t_mae = float(np.mean(np.abs(t_diff)))
        t_bias = float(np.mean(t_diff))
        daily_results[f"day_{t_idx}"] = {
            "rmse": round(t_rmse, 4),
            "mae": round(t_mae, 4),
            "bias": round(t_bias, 4),
        }

    return {
        "overall": {
            "rmse": round(overall_rmse, 4),
            "mae": round(overall_mae, 4),
            "bias": round(overall_bias, 4),
            "correlation": round(overall_corr, 4),
            "total_valid_eval_points": int(len(pred_valid)),
        },
        "per_depth": depth_results,
        "thermocline_50_200m": thermocline_results,
        "daily": daily_results,
    }


def run_reference_baseline() -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute Reference Baseline 0 and return prediction array and evaluation dictionary."""
    train_ds = SynchronizedOceanDataset(split="train_full")
    eval_ds = SynchronizedOceanDataset(split="eval")

    # 1. Compute training climatology
    clim_mean, clim_mask = compute_training_climatology(train_ds)

    # 2. Broadcast prediction across evaluation days
    N_eval = len(eval_ds)
    y_pred_clim = np.broadcast_to(clim_mean, (N_eval, *clim_mean.shape)).copy()

    # Collect ground truth from eval_ds
    y_true_list = []
    mask_list = []
    for item in eval_ds:
        y_true_list.append(item["y_raw"].numpy())
        mask_list.append(item["mask_y"].numpy())

    y_true = np.stack(y_true_list, axis=0)
    mask_eval = np.stack(mask_list, axis=0)

    # 3. Evaluate
    metrics = evaluate_predictions(y_pred_clim, y_true, mask_eval)
    return y_pred_clim, metrics
