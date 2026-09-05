# OceanEmbed

**Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

[![SIH 2026](https://img.shields.io/badge/SIH%202026-Problem%20Statement%2026066-blue.svg)](https://www.sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Validation](https://img.shields.io/badge/Gate%20A4.3-ARGO%20Frozen%20(p%3D0.0017)-green.svg)](reports/GATE_A4_3_ARGO_VALIDATION.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title & Overview

**OceanEmbed** is a deep learning framework designed to reconstruct dense 3D subsurface ocean temperature fields across 15 vertical levels (from the sea surface down to 1000 meters) directly from multi-sensor satellite surface observations.

By integrating specialized multimodal input encoders, multi-scale spatial convolutional representations, and depth-conditioned feature modulation, OceanEmbed accurately models the non-linear coupling between surface signatures (temperature, salinity, sea surface height, currents, and winds) and internal vertical ocean thermal structure.

---

## 2. Problem Statement

- **Competition:** Smart India Hackathon (SIH) 2026
- **Problem Statement ID:** 26066
- **Title:** *Estimation of 3D subsurface ocean temperature profiles from satellite-derived surface observations using deep representation learning*
- **Objective:** Bridge the observational gap between ubiquitous, high-resolution satellite remote sensing of the ocean surface and sparse, in situ subsurface oceanographic measurements.

---

## 3. What OceanEmbed Does

1. **Multimodal Surface Ingestion:** Ingests synchronized 2D fields of Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Absolute Dynamic Topography (SSH/ADT), surface current vectors ($U, V$), and 10m surface wind vectors ($U, V$).
2. **Specialized Branch Embedding:** Processes distinct physical dynamics through isolated convolutional branches (Thermodynamics, Sea Level, Currents, Winds) before spatial feature fusion.
3. **Multi-Scale Spatial Modeling:** Captures mesoscale and sub-mesoscale ocean eddies using multi-scale dilated convolutions (dilation rates 1, 2, 4).
4. **Depth-Conditioned Vertical Decoding:** Projects latent features to 15 canonical depth levels using explicit Depth FiLM conditioning.
5. **Residual Climatology Formulation:** Formulates prediction as a residual anomaly relative to local training spatial climatology:
   $$T(z, y, x) = T_{\text{climatology}}(z, y, x) + \Delta T(z, y, x)$$
6. **Dense 3D Output:** Outputs a complete physical temperature cube $[15, 24, 32]$ in degrees Celsius ($^\circ\text{C}$).

---

## 4. Official Problem Statement Scope vs. Current Validated PoC Scope

To maintain rigorous scientific clarity, the domain boundaries are explicitly separated:

| Parameter | Official PS Scope (SIH26066) | Current Validated PoC (This Repository) |
| :--- | :--- | :--- |
| **Domain** | Full North Indian Ocean | Central Bay of Bengal Pilot Box |
| **Latitude Range** | $5.0^\circ\text{N} - 30.0^\circ\text{N}$ | $12.125^\circ\text{N} - 17.875^\circ\text{N}$ ($24$ cell centers) |
| **Longitude Range** | $45.0^\circ\text{E} - 105.0^\circ\text{E}$ (Arabian Sea + Bay of Bengal) | $85.125^\circ\text{E} - 92.875^\circ\text{E}$ ($32$ cell centers) |
| **Grid Resolution** | $0.25^\circ$ cell-centered | $0.25^\circ$ cell-centered ($24 \times 32$ grid) |
| **Vertical Levels** | 15 canonical depths ($0\text{--}1000\text{ m}$) | 15 canonical depths ($0\text{--}1000\text{ m}$) |
| **Status** | Full Target Objective | **Fully Validated Proof-of-Concept (Gates A1–A4)** |

> [!IMPORTANT]
> The current validated Proof-of-Concept demonstrates the architecture, data pipeline, and statistical validation in the Bay of Bengal pilot box. It does **not** claim pan-basin generalization across the Arabian Sea or full North Indian Ocean without expanded regional training.

---

## 5. Core Specifications

- **Input Tensor Shape:** $[B, 7, 24, 32]$
- **Output Tensor Shape:** $[B, 15, 24, 32]$ (Dense 3D temperature field)
- **Vertical Resolution:** 15 canonical depth levels:
  ```
  0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
  ```
- **Spatial Grid:** $24 \times 32$ cells at $0.25^\circ$ resolution (Central Bay of Bengal)
- **Temporal Window:** 91 consecutive calendar days (Q1 2020: January 1, 2020 to March 31, 2020)
- **Chronological Split Contract:**
  - **Train Period:** 2020-01-01 to 2020-02-29 (60 days, Days 0–59)
  - **Validation Period:** 2020-03-01 to 2020-03-15 (15 days, Days 60–74)
  - **Test Period (Held-Out):** 2020-03-16 to 2020-03-31 (16 days, Days 75–90)

---

## 6. Seven Input Channels

The model ingests 7 physical surface variables in strict canonical order:

| Channel | Identifier | Physical Parameter | Native Resolution | Physical Units |
| :---: | :--- | :--- | :---: | :---: |
| **0** | `sst` | Sea Surface Temperature | $0.05^\circ$ L4 | $^\circ\text{C}$ |
| **1** | `sss` | Sea Surface Salinity | $0.125^\circ$ L4 | $\text{PSU}$ |
| **2** | `ssh` | Sea Surface Height / Absolute Dynamic Topography (ADT) | $0.25^\circ$ L4 | $\text{m}$ |
| **3** | `u_current` | Zonal Surface Current Velocity ($U$) | $0.25^\circ$ L4 | $\text{m/s}$ |
| **4** | `v_current` | Meridional Surface Current Velocity ($V$) | $0.25^\circ$ L4 | $\text{m/s}$ |
| **5** | `u_wind` | 10m Zonal Neutral Wind Velocity ($U$) | $0.25^\circ$ L4 | $\text{m/s}$ |
| **6** | `v_wind` | 10m Meridional Neutral Wind Velocity ($V$) | $0.25^\circ$ L4 | $\text{m/s}$ |

All variables are regridded to the canonical $0.25^\circ$ cell-centered grid and normalized using statistics computed strictly on the training partition (`Dataset/gate_a3_temporal/normalization/normalization_stats.json`).

---

## 7. Model Architecture

The production model is **OceanEmbed v1-Local** (`Dataset/gate_a4_3/models/oceanembed_v1_local_frozen.pt`), established as champion through Gate A4.1 ablation testing.

```
Input Surface Tensor: [B, 7, 24, 32]
  │
  ├── Multimodal Specialized Branches (branch_dim = 16)
  │     ├── Thermodynamics Branch [SST, SSS]    → Conv2D 3x3 → GELU
  │     ├── Sea-Level Branch      [SSH/ADT]     → Conv2D 3x3 → GELU
  │     ├── Currents Branch       [U, V Curr]   → Conv2D 3x3 → GELU
  │     └── Winds Branch          [U, V Wind]   → Conv2D 3x3 → GELU
  │
  ├── Feature Fusion: Concat & 1x1 Projection → Latent [B, 48, 24, 32]
  │
  ├── Multi-Scale Spatial Trunk (dilation = 1, 2, 4)
  │     ├── Dilated Conv2D (rate=1) + BatchNorm + GELU
  │     ├── Dilated Conv2D (rate=2) + BatchNorm + GELU
  │     └── Dilated Conv2D (rate=4) + BatchNorm + GELU
  │
  ├── Depth-Conditioned Decoder (FiLM Modulation)
  │     ├── Discrete Depth Embeddings: Embedding(15, 48)
  │     ├── Continuous Physical Projections: Linear(1, 48) on log1p(depth)
  │     └── Modulated 15-Level Projection → Temperature Anomaly ΔT: [B, 15, 24, 32]
  │
  └── Residual Climatology Fusion:
        Output T(z) = Climatology(z) + ΔT(z) → [B, 15, 24, 32] (°C)
```

- **Total Trainable Parameters:** Exactly **123,345** parameters.
- **Controlled Ablation Finding (Gate A4.1):** Global self-attention was systematically removed (`use_global_context=False`). Local convolutional inductive bias yielded lower thermocline RMSE ($0.9949^\circ\text{C}$ vs $1.0455^\circ\text{C}$) and superior physical consistency on mesoscale ocean fronts.

---

## 8. Data Pipeline

1. **Acquisition & Slicing:** Multi-source satellite and reanalysis fields downloaded and cropped to the pilot bounding box ($12.0^\circ - 18.0^\circ\text{N}, 85.0^\circ - 93.0^\circ\text{E}$).
2. **Bilinear Collocation:** Regridded to canonical cell centers:
   - Latitude: $12.125^\circ\text{N} + i \times 0.25^\circ$ ($i = 0 \dots 23$)
   - Longitude: $85.125^\circ\text{E} + j \times 0.25^\circ$ ($j = 0 \dots 31$)
3. **Zero-Leakage Normalization:** Mean and standard deviation computed exclusively on training samples (Days 0–59). Stored in `normalization_stats.json`.
4. **Mask Preservation:** 2D ocean masks ($68,614$ valid ocean pixels across 91 days; $1,274$ land/bathymetry pixels) are strictly preserved. Land cells are never filled with fabricated values.

---

## 9. Training & Validation Methodology

- **Training Supervision:** GLORYS12V1 reanalysis-derived training target (`Dataset/gate_a3_temporal/synchronized/oceanembed_target_15depth_91d.nc`).
- **Independent Benchmark:** IFREMER ERDDAP in situ ARGO float soundings. ARGO observations are held completely untouched during training and model selection.
- **Loss Function:** Masked Mean Squared Error (computes loss strictly over valid ocean pixels).
- **Optimization:** AdamW optimizer, initial learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$, cosine annealing schedule over 50 epochs, random seed 42.

---

## 10. Gate A4.3 Validated Benchmark Results

Evaluated against **34 independent in situ ARGO float profiles** across the held-out test window (March 16–31, 2020):

### Performance by Vertical Ocean Regime

| Ocean Regime | Depth Range | Climatology RMSE | OceanEmbed v1-Local RMSE | Relative Error Reduction | Pearson Correlation ($r$) | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Column** | **$0\text{--}1000\text{ m}$** | **$1.0947\text{ }^\circ\text{C}$** | **$0.8117\text{ }^\circ\text{C}$** | **$-25.85\%$** | **$0.9943$** | **$-0.0863\text{ }^\circ\text{C}$** |
| **Thermocline Regime** | **$50\text{--}200\text{ m}$** | **$1.2827\text{ }^\circ\text{C}$** | **$0.8446\text{ }^\circ\text{C}$** | **$-34.16\%$** | **$0.9849$** | **$+0.0320\text{ }^\circ\text{C}$** |
| **Surface Regime** | $0\text{--}20\text{ m}$ | $1.1098\text{ }^\circ\text{C}$ | $1.0672\text{ }^\circ\text{C}$ | $-3.84\%$ | $0.8423$ | $-0.5960\text{ }^\circ\text{C}$ |
| **Deep Ocean** | $> 200\text{ m}$ | $0.9634\text{ }^\circ\text{C}$ | $0.6927\text{ }^\circ\text{C}$ | $-28.10\%$ | $0.9958$ | $+0.0617\text{ }^\circ\text{C}$ |

### Statistical Independence & Clustering Rigor
- **Paired In Situ Soundings:** $3,966$ observation points.
- **Float Population:** $34$ collocated profiles nested within **14 unique WMO float platforms, treated as conservative cluster units**.
- **Platform-Clustered Permutation Test:** **$p = 0.001709$** (statistically significant subsurface improvement over climatology).
- **Platform-Clustered Bootstrap 95% CI:** **$[-0.3457, -0.1117]\text{ }^\circ\text{C}$** paired RMSE reduction.

---

## 11. Interactive Proof-of-Concept Demo

The repository includes an interactive ocean-intelligence web application:

- **Backend:** `src/demo/server.py` (lightweight Python `ThreadingHTTPServer`, zero external web frameworks).
- **Frontend:** `src/demo/static/` (vanilla HTML5, CSS3, JavaScript).
- **Interactive Capabilities:**
  - Date navigation across all 91 days (Train, Validation, Test) with ARGO float presence indicators.
  - Depth selection across all 15 canonical levels ($0\text{--}1000\text{ m}$).
  - 7 surface satellite input maps with physical units and coverage percentages.
  - 2D subsurface predicted temperature field with calibrated Celsius colorbar.
  - Click-to-probe coordinates on map with interactive vertical temperature profiles.
  - Multi-series vertical profile visualization comparing:
    - OceanEmbed v1-Local
    - Training Spatial Climatology
    - GLORYS reanalysis-derived reference
    - Independent ARGO observational reference
  - Data quality indicators and Gate A4.3 statistical audit card.

---

## 12. Repository Structure

```
oceanembed/
├── configs/
│   └── default.yaml                # Canonical domain and experiment configurations
├── docs/
│   ├── ACTUAL_DATA_INVENTORY.md     # Multi-sensor file and variable inventory
│   ├── DATA_AUDIT.md                # Data quality and grid alignment audit
│   ├── EXPERIMENT_CONTRACT.md       # Gate experimental contracts
│   ├── OFFICIAL_SOURCE_AUDIT.md     # SIH26066 problem statement audit
│   └── PS_ALIGNMENT.md              # Alignment with official problem requirements
├── figures/                         # Verified benchmark plots & profiles
├── metrics/                         # Machine-readable evaluation JSONs and CSVs
├── reports/
│   ├── GATE_A4_1_ABLATION.md        # Controlled architecture ablation report
│   ├── GATE_A4_2_OCEANEMBED_V2_LOCAL.md # Local refinement report
│   ├── GATE_A4_3_ARGO_VALIDATION.md # Independent ARGO validation report
│   ├── GATE_A4_3_STATISTICAL_AUDIT.md # Paired bootstrap and permutation report
│   ├── GATE_A4_3_PLATFORM_INDEPENDENCE_AUDIT.md # Platform clustering audit
│   ├── GATE_A4_3_FINAL_FREEZE_AUDIT.md # Final pre-freeze verification report
│   └── gate_demo_implementation_report.md # Demo implementation report
├── src/
│   ├── constants.py                # Depths, domain coordinates, variable channels
│   ├── model.py                    # Legacy model definition (reference)
│   ├── demo/
│   │   ├── README.md               # Demo operations guide
│   │   ├── server.py               # Standalone Python HTTP backend
│   │   └── static/
│   │       ├── index.html          # Web UI layout
│   │       ├── style.css           # Oceanic scientific UI stylesheet
│   │       └── app.js              # Canvas rendering and profile plotting
│   ├── gate_a2/                    # 2D Spatial CNN and climatology baseline
│   ├── gate_a3/                    # 91-day temporal sequence synchronization
│   └── gate_a4/
│       ├── model_v1.py             # OceanEmbed v1 model architecture
│       ├── ablation_study.py       # Gate A4.1 controlled ablation harness
│       ├── train_evaluate_a4.py    # Training and evaluation routines
│       ├── argo_validation.py      # Independent ARGO collocation pipeline
│       └── platform_independence_audit.py # Conservative cluster statistics
├── tests/
│   ├── test_demo.py                # 13 comprehensive demo and live HTTP tests
│   ├── test_gate_a4_3.py           # ARGO validation pipeline tests
│   ├── test_smoke.py               # Model forward pass smoke tests
│   └── test_synchronization.py     # Multi-sensor grid synchronization tests
├── requirements.txt                # Production and research dependencies
├── run.sh                          # Legacy baseline verification script
├── .gitignore                      # Safe Git exclusions (weights, NetCDFs, caches)
└── README.md                       # Master project documentation
```

---

## 13. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/oceanembed.git
   cd oceanembed
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 14. Running Tests

Execute the automated verification suite:

```bash
# Run all demo, model, and server tests (13 tests):
pytest tests/test_demo.py -v

# Run ARGO validation and statistical integrity tests:
pytest tests/test_gate_a4_3.py -v

# Run smoke tests:
pytest tests/test_smoke.py -v
```

---

## 15. Running the Demo

Launch the interactive demo server:

```bash
python -m src.demo.server --port 8050
```

Open your browser and navigate to:
```
http://127.0.0.1:8050
```

- **Health Check:** `http://127.0.0.1:8050/api/health`
- **CPU Inference Latency:** $\sim 46\text{ ms}$ (Direct forward pass); $< 0.5\text{ ms}$ (Cached)

---

## 16. Scientific Integrity & Terminology Standards

This repository strictly adheres to physical oceanography conventions and SIH 2026 evaluation rules:

1. **GLORYS Reference Designation:** Copernicus GLORYS12V1 is a numerical reanalysis product assimilating observational data; it is designated as **"GLORYS reanalysis-derived reference"** and is **never** referred to as "ground truth".
2. **ARGO Designation:** In situ ARGO float soundings represent independent, un-gridded measurements and are designated as **"Independent ARGO observational reference"**.
3. **Platform Clustering:** Float statistics explicitly cite **"14 unique WMO float platforms, treated as conservative cluster units"** to prevent pseudo-replication.
4. **Land Mask Preservation:** Coastal and land cells are masked as `null` or NaN. Missing values are never filled with synthetic numbers.

---

## 17. Current Limitations

- **Regional Bounding:** Model verification is strictly regional (Central Bay of Bengal). Generalization to the Western Arabian Sea, equatorial currents, or global ocean basins requires regional retraining.
- **Temporal Bounding:** Validated over Q1 2020. Inter-annual variability (e.g., strong Indian Ocean Dipole or El Niño events) is not yet captured.
- **Surface Regime Cold Bias:** In the uppermost $0\text{--}20\text{ m}$ mixed layer, a systematic cold bias of $-0.596\text{ }^\circ\text{C}$ remains relative to in situ floats.
- **Deep Ocean Convergence:** Below $200\text{ m}$, the model converges toward climatology due to weak surface-to-deep physical coupling.
- **Operational Boundary:** OceanEmbed is a diagnostic subsurface reconstruction framework. It is **not** an operational circulation forecast model and does **not** provide cyclone trajectory predictions.

---

## 18. Future Work

- **Pan-Indian Ocean Scaling:** Extend temporal synchronization to multi-year coverage ($2018\text{--}2024$) across the complete North Indian Ocean ($5^\circ - 30^\circ\text{N}, 45^\circ - 105^\circ\text{E}$).
- **Physical Mixed-Layer Constraints:** Incorporate mixed-layer depth (MLD) physical loss penalties to eliminate the surface cold bias.
- **Live Satellite Streaming:** Connect real-time Copernicus Marine (CMEMS) and NASA PO.DAAC automated ingestion pipelines for near-real-time subsurface diagnostic maps.

---

## 19. References

1. **Su et al. (2022)** — *DORS: Deep learning for Ocean temperature and salinity Reconstruction System*. Journal of Geophysical Research: Oceans.
2. **Chae et al. (2026)** — *TS-Cast: Uncertainty-aware deep learning for 3D ocean temperature forecasting*. Remote Sensing of Environment.
3. **Jean-Michel et al. (2021)** — *The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12V1 Reanalysis*. Frontiers in Earth Science.
4. **Wong et al. (2020)** — *Argo Data 1999–2019: Two Million Temperature-Salinity Profiles and Velocity Observations From a Global Active Array*. Frontiers in Marine Science.
