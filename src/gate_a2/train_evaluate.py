"""
Gate A2 Training & Evaluation Execution Engine
==============================================
Orchestrates training discipline, evaluation metrics computation,
reference baseline comparison, sanity checks, and artifact serialization.
"""

import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xarray as xr

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a2.dataset import SynchronizedOceanDataset
from src.gate_a2.reference_baseline import run_reference_baseline, evaluate_predictions
from src.gate_a2.cnn_baseline import SimpleSpatialCNN, MaskedMSELoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path("Dataset/gate_a2_baseline")
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
PRED_DIR = BASE_DIR / "predictions"
DIAG_DIR = BASE_DIR / "diagnostics"
REPORTS_DIR = BASE_DIR / "reports"

for d in [MODELS_DIR, METRICS_DIR, PRED_DIR, DIAG_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int = 42):
    """Ensure strict experimental reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cnn_model(
    epochs: int = 120,
    lr: float = 1e-3,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[SimpleSpatialCNN, Dict[str, Any]]:
    """
    Train SimpleSpatialCNN strictly using the training window (days 0–19).
    Internal validation on days 16–19 for checkpoint selection.
    """
    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training SimpleSpatialCNN on device: {device}")

    train_ds = SynchronizedOceanDataset(split="train")
    val_ds = SynchronizedOceanDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleSpatialCNN(in_channels=7, out_channels=15, hidden_dim=32).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SimpleSpatialCNN Parameter Count: {param_count:,}")

    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights_path = MODELS_DIR / "simple_spatial_cnn_best.pt"
    train_history = {"train_loss": [], "val_loss": [], "lr": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask_y = batch["mask_y"].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y, mask_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mask_y = batch["mask_y"].to(device)

                out = model(x)
                loss = criterion(out, y, mask_y)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        train_history["train_loss"].append(round(avg_train_loss, 6))
        train_history["val_loss"].append(round(avg_val_loss, 6))
        train_history["lr"].append(current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "train_loss": avg_train_loss,
                    "param_count": param_count,
                    "seed": seed,
                },
                best_weights_path,
            )

        if epoch % 20 == 0 or epoch == epochs:
            logger.info(
                f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.6f}"
            )

    train_duration = round(time.time() - t0, 2)
    logger.info(f"Training completed in {train_duration}s. Best Val Loss: {best_val_loss:.5f}")

    # Reload best weights
    best_ckpt = torch.load(best_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    train_info = {
        "epochs_trained": epochs,
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": round(best_val_loss, 6),
        "parameter_count": param_count,
        "seed": seed,
        "optimizer": "AdamW (lr=1e-3, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR",
        "batch_size": batch_size,
        "device": str(device),
        "training_duration_s": train_duration,
        "history": train_history,
    }

    return model, train_info


def evaluate_cnn_model(
    model: SimpleSpatialCNN,
    eval_ds: SynchronizedOceanDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate CNN model on the held-out evaluation window (days 20–29).
    Returns (pred_celsius, true_celsius, mask, metrics_dict).
    """
    device = next(model.parameters()).device
    model.eval()

    preds_norm = []
    trues_celsius = []
    masks = []

    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device)
            mask_y = batch["mask_y"]
            y_raw = batch["y_raw"]

            out = model(x)
            preds_norm.append(out.cpu().numpy()[0])
            trues_celsius.append(y_raw.numpy()[0])
            masks.append(mask_y.numpy()[0])

    preds_norm_arr = np.stack(preds_norm, axis=0)   # [10, 15, 24, 32]
    trues_celsius_arr = np.stack(trues_celsius, axis=0) # [10, 15, 24, 32]
    masks_arr = np.stack(masks, axis=0)             # [10, 15, 24, 32]

    # Convert normalized predictions back to physical Celsius
    preds_celsius = eval_ds.unnormalize_target(preds_norm_arr)

    metrics = evaluate_predictions(preds_celsius, trues_celsius_arr, masks_arr)
    return preds_celsius, trues_celsius_arr, masks_arr, metrics


def run_sanity_checks(eval_ds: SynchronizedOceanDataset, trained_model: SimpleSpatialCNN) -> Dict[str, Any]:
    """Execute all mandatory scientific sanity checks."""
    logger.info("Executing mandatory sanity checks...")
    device = next(trained_model.parameters()).device
    results = {}

    # 1. Overfit Sanity Test on 2 samples
    train_ds = SynchronizedOceanDataset(split="train")
    subset_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    overfit_batch = next(iter(subset_loader))
    ox = overfit_batch["x"].to(device)
    oy = overfit_batch["y"].to(device)
    omask = overfit_batch["mask_y"].to(device)

    overfit_model = SimpleSpatialCNN(7, 15, 32).to(device)
    opt = torch.optim.Adam(overfit_model.parameters(), lr=5e-3)
    crit = MaskedMSELoss()

    initial_loss = crit(overfit_model(ox), oy, omask).item()
    for _ in range(80):
        opt.zero_grad()
        loss = crit(overfit_model(ox), oy, omask)
        loss.backward()
        opt.step()
    final_loss = crit(overfit_model(ox), oy, omask).item()
    loss_reduction_pct = (initial_loss - final_loss) / max(initial_loss, 1e-6) * 100

    results["overfit_test"] = {
        "initial_loss": round(initial_loss, 5),
        "final_loss": round(final_loss, 5),
        "loss_reduction_pct": round(loss_reduction_pct, 2),
        "passed": bool(loss_reduction_pct > 80.0),
    }

    # 2. Shuffled-Input Control
    # Disrupt spatial structure by randomly permuting grid cells along spatial dimensions
    shuffled_preds = []
    trues = []
    masks = []
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    trained_model.eval()

    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].clone()
            # Shuffle spatial coordinates per channel
            for c in range(7):
                flat = x[0, c].view(-1)
                perm = torch.randperm(flat.size(0))
                x[0, c] = flat[perm].view(24, 32)

            out = trained_model(x.to(device))
            shuffled_preds.append(out.cpu().numpy()[0])
            trues.append(batch["y_raw"].numpy()[0])
            masks.append(batch["mask_y"].numpy()[0])

    shuffled_preds_arr = eval_ds.unnormalize_target(np.stack(shuffled_preds, axis=0))
    trues_arr = np.stack(trues, axis=0)
    masks_arr = np.stack(masks, axis=0)
    shuffled_metrics = evaluate_predictions(shuffled_preds_arr, trues_arr, masks_arr)

    results["shuffled_input_control"] = {
        "shuffled_rmse": shuffled_metrics["overall"]["rmse"],
        "shuffled_mae": shuffled_metrics["overall"]["mae"],
        "shuffled_correlation": shuffled_metrics["overall"]["correlation"],
    }

    # 3. Dynamic Variance Check (verify output is not a static constant field)
    # Variance across time for each valid ocean pixel
    with torch.no_grad():
        norm_preds = []
        for batch in eval_loader:
            out = trained_model(batch["x"].to(device))
            norm_preds.append(out.cpu().numpy()[0])
    norm_preds_arr = eval_ds.unnormalize_target(np.stack(norm_preds, axis=0))
    temporal_std = np.std(norm_preds_arr, axis=0)  # [15, 24, 32]
    ocean_2d = eval_ds.ocean_mask_2d
    valid_temporal_std = temporal_std[:, ocean_2d == 1.0]
    mean_dynamic_variation = float(np.mean(valid_temporal_std))

    results["dynamic_variation"] = {
        "mean_temporal_std_degC": round(mean_dynamic_variation, 4),
        "is_dynamic": bool(mean_dynamic_variation > 0.001),
    }

    # 4. Physical Plausibility & Finite Check
    is_finite = bool(np.all(np.isfinite(norm_preds_arr[:, :, ocean_2d == 1.0])))
    min_temp = float(np.min(norm_preds_arr[:, :, ocean_2d == 1.0]))
    max_temp = float(np.max(norm_preds_arr[:, :, ocean_2d == 1.0]))
    plausible = bool(3.0 <= min_temp and max_temp <= 33.0)

    results["output_sanity"] = {
        "all_finite": is_finite,
        "min_temperature_degC": round(min_temp, 2),
        "max_temperature_degC": round(max_temp, 2),
        "physically_plausible": plausible,
    }

    return results


def run_full_gate_a2_experiment():
    """Run full Gate A2 pipeline and generate all experiment artifacts."""
    logger.info("==================================================")
    logger.info("STARTING GATE A2: BASELINE LEARNING FEASIBILITY")
    logger.info("==================================================")

    # 1. Non-Neural Reference Baseline 0
    logger.info("--- Phase C: Baseline 0 (Reference Climatology) ---")
    y_pred_ref, ref_metrics = run_reference_baseline()
    logger.info(f"Baseline 0 Reference Overall RMSE: {ref_metrics['overall']['rmse']:.4f} °C")

    # 2. Simple Spatial CNN Baseline 1
    logger.info("--- Phase D & F: Baseline 1 (Simple Spatial CNN Training) ---")
    model, train_info = train_cnn_model(epochs=120, lr=1e-3, batch_size=4, seed=42)

    # 3. Evaluate CNN Baseline
    logger.info("--- Phase G: Baseline 1 Evaluation ---")
    eval_ds = SynchronizedOceanDataset(split="eval")
    y_pred_cnn, y_true_eval, mask_eval, cnn_metrics = evaluate_cnn_model(model, eval_ds)
    logger.info(f"Baseline 1 CNN Overall RMSE: {cnn_metrics['overall']['rmse']:.4f} °C")

    # 4. Compare Baselines
    logger.info("--- Phase H: Baseline Comparison ---")
    ref_rmse = ref_metrics["overall"]["rmse"]
    cnn_rmse = cnn_metrics["overall"]["rmse"]
    overall_imp_pct = round((ref_rmse - cnn_rmse) / ref_rmse * 100, 2)
    logger.info(f"Overall Improvement: {overall_imp_pct}% (Ref: {ref_rmse:.4f} °C -> CNN: {cnn_rmse:.4f} °C)")

    # Thermocline comparison
    ref_tc_rmse = ref_metrics["thermocline_50_200m"]["rmse"]
    cnn_tc_rmse = cnn_metrics["thermocline_50_200m"]["rmse"]
    tc_imp_pct = round((ref_tc_rmse - cnn_tc_rmse) / ref_tc_rmse * 100, 2)
    logger.info(f"Thermocline (50-200m) Improvement: {tc_imp_pct}% (Ref: {ref_tc_rmse:.4f} °C -> CNN: {cnn_tc_rmse:.4f} °C)")

    # Per-depth comparison table
    depth_comparison = []
    for d_m in REQUIRED_DEPTHS_M:
        key = f"depth_{d_m}m"
        r_d = ref_metrics["per_depth"][key]["rmse"]
        c_d = cnn_metrics["per_depth"][key]["rmse"]
        imp = round((r_d - c_d) / (r_d + 1e-6) * 100, 2)
        depth_comparison.append(
            {
                "depth_m": d_m,
                "reference_rmse": r_d,
                "cnn_rmse": c_d,
                "improvement_pct": imp,
                "reference_mae": ref_metrics["per_depth"][key]["mae"],
                "cnn_mae": cnn_metrics["per_depth"][key]["mae"],
                "reference_bias": ref_metrics["per_depth"][key]["bias"],
                "cnn_bias": cnn_metrics["per_depth"][key]["bias"],
                "reference_corr": ref_metrics["per_depth"][key]["correlation"],
                "cnn_corr": cnn_metrics["per_depth"][key]["correlation"],
            }
        )

    # Export depth comparison CSV
    df_depth = pd.DataFrame(depth_comparison)
    csv_path = METRICS_DIR / "depth_metrics.csv"
    df_depth.to_csv(csv_path, index=False)
    logger.info(f"Saved depth metrics to {csv_path}")

    # 5. Sanity Checks
    sanity_results = run_sanity_checks(eval_ds, model)

    # 6. Save Complete Metrics JSON
    results_json = {
        "gate": "A2",
        "title": "Gate A2 Baseline Learning Feasibility Metrics",
        "training_configuration": train_info,
        "baseline_0_reference": ref_metrics,
        "baseline_1_cnn": cnn_metrics,
        "comparison_summary": {
            "overall_reference_rmse": ref_rmse,
            "overall_cnn_rmse": cnn_rmse,
            "overall_improvement_pct": overall_imp_pct,
            "thermocline_reference_rmse": ref_tc_rmse,
            "thermocline_cnn_rmse": cnn_tc_rmse,
            "thermocline_improvement_pct": tc_imp_pct,
        },
        "per_depth_comparison": depth_comparison,
        "sanity_checks": sanity_results,
    }

    json_path = METRICS_DIR / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved baseline results JSON to {json_path}")

    # 7. Save Evaluation Predictions NetCDF
    eval_dates = pd.to_datetime(eval_ds.dates)
    ds_preds = xr.Dataset(
        data_vars={
            "pred_cnn": (("time", "depth", "latitude", "longitude"), y_pred_cnn),
            "pred_ref": (("time", "depth", "latitude", "longitude"), y_pred_ref),
            "target_reference": (("time", "depth", "latitude", "longitude"), y_true_eval),
            "error_cnn": (("time", "depth", "latitude", "longitude"), y_pred_cnn - y_true_eval),
            "error_ref": (("time", "depth", "latitude", "longitude"), y_pred_ref - y_true_eval),
            "mask_target": (("time", "depth", "latitude", "longitude"), mask_eval),
        },
        coords={
            "time": eval_dates,
            "depth": eval_ds.depths,
            "latitude": eval_ds.latitudes,
            "longitude": eval_ds.longitudes,
        },
        attrs={
            "title": "Gate A2 Evaluation Window Predictions (2020-01-21 to 2020-01-30)",
            "target_description": "GLORYS reanalysis-derived reference potential temperature",
            "units": "degrees_C",
        },
    )
    nc_preds_path = PRED_DIR / "eval_predictions_cnn.nc"
    ds_preds.to_netcdf(nc_preds_path)
    logger.info(f"Saved evaluation predictions NetCDF to {nc_preds_path}")
    ds_preds.close()

    return results_json


if __name__ == "__main__":
    run_full_gate_a2_experiment()
