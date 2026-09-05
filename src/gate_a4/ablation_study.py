"""
Gate A4.1 — Controlled OceanEmbed v1 Ablation Study
===================================================
Systematically evaluates the contribution of individual OceanEmbed v1 components
by varying EXACTLY ONE component at a time against frozen A3.3 and A4.0 benchmarks:

Experiments:
  1. Full OceanEmbed v1
  2. OceanEmbed WITHOUT depth conditioning (use_depth_conditioning=False)
  3. OceanEmbed WITHOUT global context (use_global_context=False)
  4. OceanEmbed WITHOUT multimodal branch fusion (use_multimodal_branches=False)
  5. OceanEmbed direct-temperature formulation (use_residual=False)

Outputs:
  Dataset/gate_a4_ablation/
    metrics/a4_1_ablation_results.json
    metrics/a4_1_depth_metrics.csv
    figures/
    reports/GATE_A4_1_ABLATION.md
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a3.dataset_a3 import SynchronizedOceanDatasetA3
from src.gate_a4.model_v1 import OceanEmbedV1, MaskedMSELoss
from src.gate_a4.train_evaluate_a4 import compute_training_climatology, evaluate_predictions, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gate_a4_ablation")

ABLATION_DIR = Path("Dataset/gate_a4_ablation")
METRICS_DIR = ABLATION_DIR / "metrics"
FIGURES_DIR = ABLATION_DIR / "figures"
REPORTS_DIR = ABLATION_DIR / "reports"

for d in [METRICS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

A3_RESULTS_JSON = Path("Dataset/gate_a3_temporal/metrics/gate_a3_results.json")
A4_RESULTS_JSON = Path("Dataset/gate_a4_oceanembed/metrics/oceanembed_v1_results.json")


def train_ablation_model(
    config_name: str,
    kwargs: Dict[str, Any],
    train_ds: SynchronizedOceanDatasetA3,
    val_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[OceanEmbedV1, Dict[str, Any]]:
    """Train single ablation configuration with identical hyperparameter protocol."""
    set_seed(seed)
    logger.info(f"--- Training Ablation: {config_name} ---")

    use_residual = kwargs.get("use_residual", True)
    model = OceanEmbedV1(**kwargs).to(device)

    clim_tensor = torch.from_numpy(clim_mean).to(device)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y_raw = batch["y_raw"].to(device)
            m_tgt = batch["mask_y"].to(device)

            optimizer.zero_grad()
            pred = model(x, climatology=clim_tensor if use_residual else None)
            loss = criterion(pred, y_raw, m_tgt)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y_raw = batch["y_raw"].to(device)
                m_tgt = batch["mask_y"].to(device)
                pred = model(x, climatology=clim_tensor if use_residual else None)
                loss = criterion(pred, y_raw, m_tgt)
                v_losses.append(loss.item())

        mean_val = float(np.mean(v_losses))
        scheduler.step(mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    duration = time.time() - start_time
    model.load_state_dict(best_weights)
    model.eval()

    info = {
        "config_name": config_name,
        "parameters": model.get_parameter_count(),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "duration_seconds": round(duration, 2),
    }
    logger.info(
        f"Completed {config_name}: Best Epoch={best_epoch}, "
        f"Val Loss={best_val_loss:.4f}, Duration={duration:.1f}s, Params={model.get_parameter_count():,}"
    )
    return model, info


def evaluate_ablation_model(
    model: OceanEmbedV1,
    test_ds: SynchronizedOceanDatasetA3,
    clim_mean: np.ndarray,
    device: torch.device,
    use_residual: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference over 16 test days."""
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    clim_tensor = torch.from_numpy(clim_mean).to(device)
    preds, trues, masks = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x, climatology=clim_tensor if use_residual else None)
            preds.append(out.cpu().numpy()[0])
            trues.append(batch["y_raw"].numpy()[0])
            masks.append(batch["mask_y"].numpy()[0])

    return np.stack(preds, axis=0), np.stack(trues, axis=0), np.stack(masks, axis=0)


def run_ablation_study():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing Gate A4.1 Ablation Study on device: {device}")

    # 1. Load Datasets
    train_ds = SynchronizedOceanDatasetA3(split="train")
    val_ds = SynchronizedOceanDatasetA3(split="val")
    test_ds = SynchronizedOceanDatasetA3(split="test")
    test_dates = test_ds.dates

    # 2. Climatology Baseline (Days 0–59)
    clim_mean, clim_mask = compute_training_climatology(train_ds)
    b0_preds = np.repeat(clim_mean[np.newaxis, ...], len(test_ds), axis=0)
    b0_trues = np.stack([item["y_raw"].numpy() for item in test_ds], axis=0)
    b0_masks = np.stack([item["mask_y"].numpy() for item in test_ds], axis=0)
    b0_metrics = evaluate_predictions(b0_preds, b0_trues, b0_masks, test_dates)

    # 3. Load A3 CNN Benchmark
    with open(A3_RESULTS_JSON, "r") as f:
        a3_results = json.load(f)
    a3_cnn_metrics = a3_results["baseline_1_cnn"]

    # 4. Define Ablation Configurations
    ablation_definitions = {
        "full_oceanembed_v1": {
            "name": "Full OceanEmbed v1",
            "kwargs": {
                "use_residual": True,
                "use_depth_conditioning": True,
                "use_global_context": True,
                "use_multimodal_branches": True,
            },
            "description": "Complete architecture (Multimodal + Multi-scale + Attention + Depth-conditioning + Residual)",
        },
        "without_depth_conditioning": {
            "name": "OceanEmbed w/o Depth Conditioning",
            "kwargs": {
                "use_residual": True,
                "use_depth_conditioning": False,
                "use_global_context": True,
                "use_multimodal_branches": True,
            },
            "description": "Standard 15-channel output conv instead of depth embeddings and depth filter generator",
        },
        "without_global_context": {
            "name": "OceanEmbed w/o Global Context",
            "kwargs": {
                "use_residual": True,
                "use_depth_conditioning": True,
                "use_global_context": False,
                "use_multimodal_branches": True,
            },
            "description": "Bypasses non-local spatial self-attention block (Identity)",
        },
        "without_multimodal_branches": {
            "name": "OceanEmbed w/o Multimodal Branches",
            "kwargs": {
                "use_residual": True,
                "use_depth_conditioning": True,
                "use_global_context": True,
                "use_multimodal_branches": False,
            },
            "description": "Single monolithic 7->48 conv instead of 4 specialized modality branches",
        },
        "direct_temperature_formulation": {
            "name": "OceanEmbed Direct Temperature (w/o Residual)",
            "kwargs": {
                "use_residual": False,
                "use_depth_conditioning": True,
                "use_global_context": True,
                "use_multimodal_branches": True,
            },
            "description": "Directly regresses physical temperature instead of predicting anomaly over climatology",
        },
    }

    # 5. Train and Evaluate each Ablation
    ablation_results = {}
    all_depth_rows = []

    for key, defn in ablation_definitions.items():
        name = defn["name"]
        kwargs = defn["kwargs"]
        use_residual = kwargs.get("use_residual", True)

        model, train_info = train_ablation_model(
            config_name=name,
            kwargs=kwargs,
            train_ds=train_ds,
            val_ds=val_ds,
            clim_mean=clim_mean,
            device=device,
        )

        preds, trues, masks = evaluate_ablation_model(
            model=model,
            test_ds=test_ds,
            clim_mean=clim_mean,
            device=device,
            use_residual=use_residual,
        )

        metrics = evaluate_predictions(preds, trues, masks, test_dates)

        # Compute improvements relative to benchmarks
        diff_vs_full_rmse = float(metrics["overall_rmse"] - 0.9239)
        diff_vs_full_tc = float(metrics["thermocline_rmse"] - 1.0455)
        diff_vs_a3_tc = float(metrics["thermocline_rmse"] - a3_cnn_metrics["thermocline_rmse"])
        diff_vs_clim_tc = float(metrics["thermocline_rmse"] - b0_metrics["thermocline_rmse"])

        ablation_results[key] = {
            "name": name,
            "description": defn["description"],
            "parameters": train_info["parameters"],
            "training_info": train_info,
            "metrics": metrics,
            "comparison": {
                "delta_overall_rmse_vs_full": round(diff_vs_full_rmse, 4),
                "delta_thermocline_rmse_vs_full": round(diff_vs_full_tc, 4),
                "delta_thermocline_rmse_vs_a3": round(diff_vs_a3_tc, 4),
                "delta_thermocline_rmse_vs_clim": round(diff_vs_clim_tc, 4),
            },
        }

        # Collect per-depth rows
        for d in metrics["per_depth"]:
            all_depth_rows.append({
                "model_key": key,
                "model_name": name,
                "depth_m": d["depth_m"],
                "rmse": d["rmse"],
                "mae": d["mae"],
                "bias": d["bias"],
                "correlation": d["correlation"],
            })

    # 6. Save Depth Metrics CSV
    depth_df = pd.DataFrame(all_depth_rows)
    depth_csv_path = METRICS_DIR / "a4_1_depth_metrics.csv"
    depth_df.to_csv(depth_csv_path, index=False)
    logger.info(f"Saved depth metrics CSV to {depth_csv_path}")

    # 7. Compile Final Results JSON
    summary_table = []
    # Add benchmarks
    summary_table.append({
        "model": "Baseline 0: Climatology",
        "overall_rmse": b0_metrics["overall_rmse"],
        "overall_mae": b0_metrics["overall_mae"],
        "bias": b0_metrics["overall_bias"],
        "thermocline_rmse": b0_metrics["thermocline_rmse"],
        "thermocline_mae": b0_metrics["thermocline_mae"],
        "parameters": 0,
    })
    summary_table.append({
        "model": "Baseline 1: A3 Simple CNN",
        "overall_rmse": a3_cnn_metrics["overall_rmse"],
        "overall_mae": a3_cnn_metrics["overall_mae"],
        "bias": a3_cnn_metrics["overall_bias"],
        "thermocline_rmse": a3_cnn_metrics["thermocline_rmse"],
        "thermocline_mae": a3_cnn_metrics["thermocline_mae"],
        "parameters": 39759,
    })
    for k, v in ablation_results.items():
        m = v["metrics"]
        summary_table.append({
            "model": v["name"],
            "overall_rmse": m["overall_rmse"],
            "overall_mae": m["overall_mae"],
            "bias": m["overall_bias"],
            "thermocline_rmse": m["thermocline_rmse"],
            "thermocline_mae": m["thermocline_mae"],
            "parameters": v["parameters"],
        })

    final_payload = {
        "gate": "A4.1",
        "objective": "Controlled OceanEmbed v1 Component Ablation Study",
        "target_nomenclature": "GLORYS reanalysis-derived reference",
        "summary_table": summary_table,
        "benchmarks": {
            "climatology": b0_metrics,
            "a3_cnn": a3_cnn_metrics,
        },
        "ablations": ablation_results,
    }

    results_json_path = METRICS_DIR / "a4_1_ablation_results.json"
    with open(results_json_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Saved ablation results JSON to {results_json_path}")

    # 8. Generate Visualizations
    plot_ablation_figures(final_payload, depth_df)

    return final_payload


def plot_ablation_figures(payload: Dict[str, Any], depth_df: pd.DataFrame):
    """Generate 4 diagnostic figures for Gate A4.1."""
    summary = payload["summary_table"]
    models = [s["model"] for s in summary]
    tc_rmses = [s["thermocline_rmse"] for s in summary]
    ov_rmses = [s["overall_rmse"] for s in summary]

    # Clean short names for plotting
    short_names = [
        "Climatology",
        "A3 CNN",
        "Full OceanEmbed v1",
        "w/o Depth Cond",
        "w/o Global Context",
        "w/o Modality Branches",
        "Direct Temp (w/o Res)",
    ]

    # 1. Thermocline RMSE Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    bars = ax.bar(short_names, tc_rmses, color=colors, alpha=0.85, edgecolor="black")

    ax.axhline(payload["benchmarks"]["climatology"]["thermocline_rmse"], color="#d62728", linestyle="--", label="Climatology Ref (1.1295 °C)")
    ax.axhline(payload["benchmarks"]["a3_cnn"]["thermocline_rmse"], color="#1f77b4", linestyle=":", label="A3 Simple CNN Ref (1.1914 °C)")

    for bar, val in zip(bars, tc_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Thermocline (50–200m) RMSE (°C)", fontsize=11, fontweight="bold")
    ax.set_title("OceanEmbed v1 Component Ablation — Thermocline RMSE", fontsize=12, fontweight="bold")
    ax.set_ylim(0.8, 1.4)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.legend(frameon=True, fontsize=10)

    p1 = FIGURES_DIR / "ablation_thermocline_rmse_comparison.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Overall RMSE Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.bar(short_names, ov_rmses, color=colors, alpha=0.85, edgecolor="black")

    ax.axhline(payload["benchmarks"]["climatology"]["overall_rmse"], color="#d62728", linestyle="--", label="Climatology Ref (1.0590 °C)")
    ax.axhline(payload["benchmarks"]["a3_cnn"]["overall_rmse"], color="#1f77b4", linestyle=":", label="A3 Simple CNN Ref (0.8602 °C)")

    for bar, val in zip(bars, ov_rmses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Overall Test RMSE (°C)", fontsize=11, fontweight="bold")
    ax.set_title("OceanEmbed v1 Component Ablation — Overall Test RMSE", fontsize=12, fontweight="bold")
    ax.set_ylim(0.6, 1.3)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.legend(frameon=True, fontsize=10)

    p2 = FIGURES_DIR / "ablation_overall_rmse_comparison.png"
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Depth-Wise Profiles
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    depths = REQUIRED_DEPTHS_M

    # Plot Climatology & A3 CNN
    clim_depths = [d["rmse"] for d in payload["benchmarks"]["climatology"]["per_depth"]]
    a3_depths = [d["rmse"] for d in payload["benchmarks"]["a3_cnn"]["per_depth"]]
    ax.plot(clim_depths, depths, "o-", color="#d62728", lw=2, label="Climatology")
    ax.plot(a3_depths, depths, "s--", color="#1f77b4", lw=2, label="A3 Simple CNN")

    model_keys = [
        ("full_oceanembed_v1", "Full OceanEmbed v1", "#2ca02c", "D-", 2.5),
        ("without_depth_conditioning", "w/o Depth Cond", "#9467bd", "^-.", 1.8),
        ("without_global_context", "w/o Global Context", "#8c564b", "v-.", 1.8),
        ("without_multimodal_branches", "w/o Modality Branches", "#e377c2", "x-.", 1.8),
        ("direct_temperature_formulation", "Direct Temp (w/o Res)", "#7f7f7f", "*:", 1.8),
    ]

    for m_key, m_lbl, col, ls, lw in model_keys:
        m_depth_df = depth_df[depth_df["model_key"] == m_key]
        m_rmse = m_depth_df["rmse"].values
        ax.plot(m_rmse, depths, ls, color=col, lw=lw, label=m_lbl)

    ax.axhspan(50, 200, color="#ffff99", alpha=0.25, label="Thermocline (50–200m)")
    ax.set_yscale("symlog", linthresh=50)
    ax.invert_yaxis()
    ax.set_yticks(depths)
    ax.set_yticklabels([f"{int(d)}m" for d in depths])

    ax.set_xlabel("Test RMSE (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax.set_title("Depth-Wise RMSE Across Controlled Ablations", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    p3 = FIGURES_DIR / "ablation_depth_wise_profiles.png"
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")


if __name__ == "__main__":
    run_ablation_study()
