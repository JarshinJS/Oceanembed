# OceanEmbed — MVP (SIH26066 Aligned)

**Estimating 3D subsurface ocean temperature (0–1000 m) from 7 surface fields using a learned satellite embedding.**

This repository contains the reference implementation for **SIH26066**: *"Estimation of 3D subsurface ocean temperature profiles from satellite-derived surface observations using deep representation learning"*.

The pipeline demonstrates the complete end-to-end workflow: multi-sensor data ingestion → 7-channel convolutional satellite embedding (`SatEmbedNet`) → 15-depth subsurface temperature regression → validation against reanalysis training supervision and independent observational benchmarks.

---

## 1. Core Specifications & Immutable Contract

- **Official PS Domain:** North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N}, 45^\circ\text{E} - 105^\circ\text{E}$), covering the Arabian Sea and Bay of Bengal.
- **MVP / Pilot Subset:** Bay of Bengal ($5^\circ\text{N} - 25^\circ\text{N}, 80^\circ\text{E} - 100^\circ\text{E}$), designated strictly as a **Phase A feasibility pilot**.
- **Canonical Input Tensor:** `[B, 7, H, W]` (7 surface parameter channels).
- **Canonical Output Tensor:**
  - `[B, 15]` in **Profile Regression Mode** (Phase A feasibility pilot).
  - `[B, 15, H, W]` in **Dense 3D Field Mode** (future full reconstruction).
- **Canonical 15 Depth Levels:**
  ```
  0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
  ```
- **Training Supervision:** Copernicus GLORYS12V1 (reanalysis-derived training target).
- **Observational Benchmark:** In-situ ARGO float profiles / RAMA moorings (independent observational validation).

---

## 2. Canonical 7 Input Channels

The model ingests 7 surface parameters in fixed canonical order:

| Channel | Variable | Physical Parameter | Official Primary Product (SIH26066) | Approved Fallback | Units |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **0** | `sst` | Sea Surface Temperature | **OSTIA** (0.05°, CMEMS) | NOAA OISST v2.1 | $^\circ\text{C}$ / $\text{K}$ |
| **1** | `sss` | Sea Surface Salinity | **SMAP / SMOS** (0.125°, CMEMS/PO.DAAC) | CMEMS SSS L4 / GLORYS `so` | $\text{PSU}$ |
| **2** | `ssh` / `sla` | Sea Level Anomaly / Sea Surface Height | **DUACS** (0.25°, CMEMS) | GLORYS `zos` | $\text{m}$ |
| **3** | `u_current` | Surface Current Zonal Velocity | **OSCAR** (0.25°, PO.DAAC) | GLORYS `uo` ($z=0$) | $\text{m/s}$ |
| **4** | `v_current` | Surface Current Meridional Velocity | **OSCAR** (0.25°, PO.DAAC) | GLORYS `vo` ($z=0$) | $\text{m/s}$ |
| **5** | `u_wind` | 10m Surface Wind Zonal Velocity | **ASCAT-L2 Coastal / CCMP** (0.25°, EUMETSAT/RSS) | ECMWF ERA5 `u10` | $\text{m/s}$ |
| **6** | `v_wind` | 10m Surface Wind Meridional Velocity | **ASCAT-L2 Coastal / CCMP** (0.25°, EUMETSAT/RSS) | ECMWF ERA5 `v10` | $\text{m/s}$ |

---

## 3. Data Source Modes

To maintain scientific integrity, experiments are categorized into three explicit modes:

1. **`SATELLITE_OBSERVATION`**: Pure satellite observation inputs (final Problem Statement requirement).
2. **`REANALYSIS_FALLBACK`**: Surface layers extracted from reanalysis (GLORYS12V1). Used strictly as an **engineering mode** for pipeline smoke testing; does **not** constitute the final satellite-only PS demonstration.
3. **`MIXED`**: Combined satellite observations and atmospheric reanalysis winds.

---

## 4. Project Structure

```
oceanembed/
├── configs/
│   └── default.yaml          # official domain, pilot region, 7 channels, 15 depths
├── docs/
│   ├── OFFICIAL_SOURCE_AUDIT.md # Primary source audit based on authoritative SIH26066 PDF
│   ├── EXPERIMENT_CONTRACT.md # Formal scientific contract (Gate A1 vs Gate A2)
│   ├── PS_ALIGNMENT.md        # Canonical SIH26066 specification document
│   ├── IMPLEMENTATION_PLAN.md # Phased roadmap (Phase A, Phase B, Phase C)
│   └── DATA_AUDIT.md          # Multi-sensor data audit & storage inventory
├── src/
│   ├── __init__.py
│   ├── constants.py          # Centralized REQUIRED_DEPTHS_M, domains, channel maps
│   ├── data.py               # 7-channel SyntheticOceanDataset + GLORYS ingestion stub
│   ├── model.py              # SatEmbedNet (7-channel CNN encoder + 15-depth regressor)
│   ├── train.py              # Training loop, early stopping, checkpointing
│   ├── evaluate.py           # Per-depth RMSE / correlation / bias reporting
│   ├── visualize.py          # Profile plots, RMSE bar chart, embedding PCA
│   └── inference.py          # Single/batch prediction + embedding export
├── data/                     # NetCDF storage directory (currently 0 B, no large downloads)
├── outputs/                  # Model checkpoints, evaluation metrics, figures
├── notebooks/                # Interactive exploration
├── tests/
│   └── test_smoke.py         # Shape verification & forward pass smoke test
├── requirements.txt
├── run.sh                    # Verification script
└── README.md
```

---

## 5. Model Architecture & Output Modes

```
Input [B, 7, H, W]
  │
  ├─ ConvBlock(32)  → MaxPool
  ├─ ConvBlock(64)  → MaxPool
  ├─ ConvBlock(128) → MaxPool
  ├─ AdaptiveAvgPool → Flatten
  ├─ Linear(128)          ← satellite embedding (latent vector)
  └─ MLP → T(z) × 15      ← 15-level subsurface temperature profile
```

The embedding layer is designed to learn a compact 128-dimensional latent representation of the multi-sensor surface state, and will be evaluated against baseline linear and empirical models (climatology, linear regression, GEM).

> [!NOTE]
> **Output Mode Distinction:** The current model operates in **Profile Regression Mode** `[B, 7, H, W] -> [B, 15]`, predicting domain-average or point vertical profiles for Phase-A feasibility testing. It does **not** claim equivalence to a dense 3D spatial reconstruction `[B, 15, H, W]`, which will be evaluated in subsequent phases.

---

## 6. Quick Verification

To verify the updated 7-channel, 15-depth architecture without triggering any large data downloads:

```bash
# Run the forward pass smoke test
python -m tests.test_smoke
```

Output:
```
[PASS] Forward pass OK - input shape: torch.Size([1, 7, 81, 81]) embedding shape: torch.Size([1, 128]) profile shape: torch.Size([1, 15])
```

---

## 7. Phased Development Roadmap

- **Phase A: 30–60 Day Feasibility Pilot**
  - Small slice (~12–25 MB) of real data over a central Bay of Bengal pilot box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$).
  - Evaluates data harmonization, 7-channel regridding, and model feasibility against baseline climatology and linear persistence.
- **Phase B: Larger Temporal Validation**
  - 1–2 contiguous years over the regional Bay of Bengal ($5^\circ - 25^\circ\text{N}, 80^\circ - 100^\circ\text{E}$).
  - Evaluates cross-seasonal generalization and comparison with collocated ARGO float profiles.
- **Phase C: North Indian Ocean Basin Demonstration**
  - Multi-year evaluation across the full official domain ($5^\circ - 30^\circ\text{N}, 45^\circ - 105^\circ\text{E}$), encompassing both the Arabian Sea and Bay of Bengal.

---

## 8. References

- Su et al. (2022) — *DORS*: ConvLSTM for global 3D ocean temperature (R²≈0.99, RMSE≈0.34 °C).
- Chae et al. (2026) — *TS-Cast*: Uncertainty-aware DL for NW Pacific (RMSE <1 °C, upper 500 m).
- Sun et al. (2026) — *CSSP-ConvLSTM*: CNN+Transformer hybrid (R²≈0.981, RMSE≈0.456 °C).
