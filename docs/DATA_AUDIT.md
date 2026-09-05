# OceanEmbed: Data Audit & Requirements Specification (SIH26066 Aligned)

## 1. Audit Executive Summary

- **Repository Root:** `c:\Users\Jarsh\Downloads\oceanembed_mvp\oceanembed`
- **Current Data Status:** The `data/` directory is currently **empty** (contains only `.gitkeep`, 0 bytes). No large dataset downloads have been initiated.
- **Current Operational Mode:** PyTorch pipeline verified with `SyntheticOceanDataset` (`src/data.py`), generating canonical 7-channel surface fields `[B, 7, H, W]` and matching 15-depth target profiles `[B, 15]`.
- **Stub Status:** `GLORYSDataset` in `src/data.py` is an un-implemented stub awaiting real NetCDF ingestion.

---

## 2. Hardware & Runtime Environment

- **OS / Platform:** Windows 11 / Windows PowerShell
- **Python Version:** Python 3.12.7
- **PyTorch & CUDA:** PyTorch `2.11.0+cu128`, CUDA Available: `True` (Driver 592.82, CUDA 13.1 compatibility)
- **Active GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM / ~6 GB)
- **Scientific Stack Installed:** `numpy 2.5.2`, `pandas 3.0.5`, `xarray 2026.7.0`, `netCDF4 1.7.4`, `scipy 1.18.1`, `scikit-learn 1.9.0`, `matplotlib 3.11.1`

---

## 3. Data Requirements & Channel Specification

> [!NOTE]
> **Historical Supersession Notice:**
> - Earlier drafting referenced a 5-channel surface input and depth levels containing 400 m and 750 m.
> - **These are superseded:** Official SIH26066 requires strictly **7 surface input channels** and the **15 canonical depth levels** `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m (with 125 m and 700 m replacing the incorrect 400 m and 750 m).
> - **Official PS Domain:** $5^\circ - 30^\circ\text{N}, 45^\circ - 105^\circ\text{E}$ (North Indian Ocean). The Bay of Bengal region is strictly an **MVP / pilot feasibility subset**.
> - **GLORYS Terminology:** GLORYS12V1 is strictly designated as **"reanalysis-derived training target"** / **"training supervision"**, not "ground truth". Independent **in-situ ARGO profiles** serve as the independent observational validation pathway.
> - **Data Source Mode:** Stated strictly as `REANALYSIS_FALLBACK` for Phase A engineering convenience. GLORYS-derived surface variables are NOT equivalent to the final satellite-observation input configuration.

### Canonical 7 Surface Input Channels:

| Channel Index | Channel Name | Parameter Name | Target Spatial Grid | Official Primary Source (SIH26066) | Approved Fallback Source | Temporal Resolution | Status in Repo |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| **0** | `sst` | Sea Surface Temperature ($^\circ\text{C}$ / $\text{K}$) | 0.25° | **OSTIA** (0.05°, CMEMS) | NOAA OISST v2.1 (0.25°, NOAA) | Daily | Synthetic Ready / Awaiting NetCDF |
| **1** | `sss` | Sea Surface Salinity ($\text{PSU}$) | 0.25° | **SMAP / SMOS** (0.125°, CMEMS/PO.DAAC) | CMEMS SSS L4 / GLORYS `so` ($z=0$) | Daily | Synthetic Ready / Awaiting NetCDF |
| **2** | `ssh` / `sla` | Sea Surface Height / Sea Level Anomaly ($\text{m}$) | 0.25° | **DUACS** (0.25°, CMEMS) | GLORYS `zos` | Daily | Synthetic Ready / Awaiting NetCDF |
| **3** | `u_current` | Surface Current Zonal Velocity ($\text{m/s}$) | 0.25° | **OSCAR** (0.25°, PO.DAAC) | GLORYS `uo` ($z=0$) | Daily | Synthetic Ready / Awaiting NetCDF |
| **4** | `v_current` | Surface Current Meridional Velocity ($\text{m/s}$) | 0.25° | **OSCAR** (0.25°, PO.DAAC) | GLORYS `vo` ($z=0$) | Daily | Synthetic Ready / Awaiting NetCDF |
| **5** | `u_wind` | 10m Surface Wind Zonal Velocity ($\text{m/s}$) | 0.25° | **ASCAT-L2 Coastal / CCMP** (0.25°, EUMETSAT/RSS) | ECMWF ERA5 `u10` | Daily | Synthetic Ready / Awaiting NetCDF |
| **6** | `v_wind` | 10m Surface Wind Meridional Velocity ($\text{m/s}$) | 0.25° | **ASCAT-L2 Coastal / CCMP** (0.25°, EUMETSAT/RSS) | ECMWF ERA5 `v10` | Daily | Synthetic Ready / Awaiting NetCDF |

### Training Target & Validation Pathways:

| Role | Dataset | Official Designation | Variable & Vertical Levels | Spatial Resolution | Temporal Resolution | Status in Repo |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Training Target** | **Copernicus GLORYS12V1** (`GLOBAL_MULTIYEAR_PHY_001_030`) | **PRIMARY** | `thetao` at 15 canonical depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m.<br/>*Rule: 0m mapped to uppermost level (~0.494m); depths > 0m vertically interpolated without unintended extrapolation.* | 1/12° native (regridded to 0.25°) | Daily | Synthetic Ready / Stub Defined / No Large Download Yet |
| **Observational Validation** | **INCOIS LAS — Gridded ARGO** (Indian Ocean analysis) | **PRIMARY** | Temperature profiles $T(z)$ at discrete sensor levels (matched to nearest grid point). Fallback: Coriolis GDAC ARGO. | Point / 1.0°/0.5°/0.25° gridded | 10-day / monthly | Planned for Phase B / C Validation |

---

## 4. Current Dataset Storage & File Sizes

```
c:\Users\Jarsh\Downloads\oceanembed_mvp\oceanembed\data
└── .gitkeep (0 B)
```
- **Total Real Data on Disk:** 0.0 MB (Policy enforced: no unapproved large downloads).
- **Total Outputs on Disk:**
  - `outputs/best_model.pt`: 529,653 bytes (~517 KB) — Checkpoint from synthetic pre-check.

---

## 5. Phased Experimental Protocol & Gating

### Gate A1: Real-Data Pipeline Smoke Test (30 Daily Samples)
- **Purpose:** Engineering pipeline smoke test. Demonstrates NetCDF ingestion, coordinate validation, regridding, masking, normalization, 7-channel tensor creation, 15-depth target creation, and GPU forward/backward pass.
- **Important Scientific Caveat:** Gate A1 is **not** presented as sufficient evidence of model generalization or statistical superiority.
- **Data Source Mode:** Declared as `REANALYSIS_FALLBACK`.
- **Scope:**
  - **Temporal:** 30 daily steps (2020-05-01 to 2020-05-30, pre-monsoon stratification transition).
  - **Spatial:** Focused Bay of Bengal pilot box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$), selected to minimize expected coastal/land-mask contamination.
  - **Vertical:** Exactly the 15 canonical depth levels `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m (with 0m mapped to uppermost level ~0.494m).
  - **Total Estimated Slice Size:** **~12 MB to 20 MB**.
- **Pre-download Requirement:** Actual downloaded-file metadata must be empirically verified using `src/inspect_metadata.py` before downstream ingestion.
- **Acceptance Gate:** Must pass all 14 tests in `tests/test_a1_acceptance.py`.

### Gate A2: Preliminary ML Feasibility Experiment (>=90 Daily Samples)
- **Purpose:** Preliminary ML feasibility and baseline benchmarking across a broader seasonal window (e.g. May 1 to July 31, 2020).
- **Evaluation Protocol:**
  - Strict temporal holdout (train on early 70%, val on middle 15%, test on late 15%).
  - Benchmark against Climatology, SST-only Linear Persistence, and Multi-input Linear Regression.
  - Primary metrics: RMSE, Bias, and Pearson Correlation reported overall and in the 50–200m thermocline (recording a 20% thermocline RMSE reduction as an optional stretch target only).

### Phase B: Larger Temporal Validation
- **Scope:** 1–2 contiguous years over the regional Bay of Bengal ($5^\circ - 25^\circ\text{N}, 80^\circ - 100^\circ\text{E}$).
- **Purpose:** Assess cross-seasonal generalization (monsoon reversal dynamics) and perform independent observational validation against collocated ARGO float profiles.

### Phase C: Full North Indian Ocean Demonstration
- **Scope:** Multi-year dataset spanning the full official PS domain ($5^\circ - 30^\circ\text{N}, 45^\circ - 105^\circ\text{E}$).
- **Purpose:** Demonstration across distinct hydrodynamic regimes (Arabian Sea high-salinity upwelling vs. Bay of Bengal freshwater stratification).
