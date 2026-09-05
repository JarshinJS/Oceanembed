# OceanEmbed MVP Interactive Demo
**Problem Statement ID: 26066 — SIH 2026**

Interactive, production-quality Proof-of-Concept demonstration of the frozen **OceanEmbed v1-Local** deep learning framework for subsurface ocean temperature reconstruction from multimodal satellite observations in the Bay of Bengal.

---

## 1. Quick Start

### Prerequisites
- Python 3.10+ (tested on Python 3.12.7)
- Virtual environment with project dependencies:
  ```powershell
  .venv\Scripts\activate
  ```

### Launch the Demo Server
Run the lightweight backend HTTP server:
```powershell
python -m src.demo.server --port 8050
```
Or simply:
```powershell
python src/demo/server.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8050
```

---

## 2. System Architecture

The demo uses a zero-external-framework architecture consisting of:
1. **Backend (`src/demo/server.py`)**:
   - Python standard library `ThreadingHTTPServer` (concurrent, lightweight, robust).
   - Singleton `OceanEmbedDemoService` caching the frozen model, normalization parameters, climatology, and synchronized datasets in memory.
   - Sub-50 ms CPU inference latency per day; sub-millisecond on in-memory cache hit.
2. **Frontend (`src/demo/static/`)**:
   - `index.html`: Clean, semantic HTML5 structure tailored to physical oceanography workflows.
   - `style.css`: Modern scientific ocean-intelligence dark theme with glassmorphism, responsive cards, and distinct land masking.
   - `app.js`: Vanilla JavaScript managing date sliders, depth buttons, 2D canvas maps, and vertical profile SVG/canvas charts.

---

## 3. Scientific Model & Artifacts

| Component | Path / Value | Description |
| :--- | :--- | :--- |
| **Frozen Model** | `Dataset/gate_a4_3/models/oceanembed_v1_local_frozen.pt` | Gate A4.3 champion model (123,345 parameters, seed 42) |
| **Architecture** | `OceanEmbedV1` | Residual formulation + Depth FiLM conditioning + Multimodal specialized branches (No global self-attention) |
| **Normalization** | `Dataset/gate_a3_temporal/normalization/normalization_stats.json` | Training split leakage-free statistics |
| **Climatology** | `Dataset/gate_a3_temporal/climatology/training_climatology_15depth.npz` | Training set spatial mean field $[15, 24, 32]$ |
| **Input NetCDF** | `Dataset/gate_a3_temporal/synchronized/oceanembed_inputs_7ch_91d.nc` | 91-day Q1 2020 synchronized 7-channel satellite observations |
| **Target NetCDF** | `Dataset/gate_a3_temporal/synchronized/oceanembed_target_15depth_91d.nc` | 91-day 15-depth GLORYS reanalysis-derived reference |
| **In Situ ARGO** | `Dataset/gate_a4_3/metrics/argo_profile_metrics.csv` | 34 collocated profiles across 14 unique WMO float platforms |

---

## 4. API Endpoints

### `GET /api/health`
Returns application status, parameter counts, domain bounds, and split metadata.
```json
{
  "status": "ok",
  "model_name": "OceanEmbed v1-Local",
  "model_loaded": true,
  "parameter_count": 123345,
  "domain": {
    "name": "Bay of Bengal",
    "grid_shape": [24, 32],
    "depth_levels_count": 15
  }
}
```

### `GET /api/dates`
Returns all 91 synchronized dates with chronological split flags and ARGO in situ availability:
- **TRAIN:** 2020-01-01 through 2020-02-29 (60 days)
- **VAL:** 2020-03-01 through 2020-03-15 (15 days)
- **TEST:** 2020-03-16 through 2020-03-31 (16 days; 100% have collocated ARGO floats)

### `GET /api/surface?date=YYYY-MM-DD`
Returns unnormalized physical values for all 7 surface channels (SST, SSS, SSH/ADT, Current U/V, Wind U/V) with raw validity masks. Land and source-masked cells are rendered as `null` to preserve scientific integrity.

### `POST /api/predict`
Executes frozen model inference for the requested date.
- **Input:** `{"date": "2020-03-20"}`
- **Process:** Normalizes 7-channel input $\to$ Tensor $[1, 7, 24, 32]$ $\to$ Model forward pass with `torch.no_grad()` $\to$ Returns $[15, 24, 32]$ in °C.
- **Output:** Prediction array, ocean mask, coordinates, and latency.

### `GET /api/profile?date=YYYY-MM-DD&lat=...&lon=...`
Extracts vertical profile at the given coordinate:
1. OceanEmbed v1-Local prediction
2. Training Spatial Climatology
3. GLORYS reanalysis-derived reference
4. Independent ARGO observational reference (if collocated within ~40 km)

### `GET /api/metrics`
Returns verified Gate A4.3 statistical metrics:
- Overall RMSE: **0.8117 °C**
- Thermocline RMSE (50–200m): **0.8446 °C** (vs Climatology 1.2827 °C, **-34.16%**)
- Exact platform permutation: **$p = 0.001709$**
- 95% Platform-clustered bootstrap CI: **$[-0.3457, -0.1117]\text{ °C}$**

---

## 5. Scientific Terminology & Integrity Rules

In compliance with physical oceanography and Gate A4.3 validation standards:
1. **GLORYS** is explicitly designated as **"GLORYS reanalysis-derived reference"** (never "ground truth").
2. **ARGO** is designated as **"Independent ARGO observational reference"**.
3. Float statistics reference **"14 unique WMO float platforms, treated as conservative cluster units"** (never "14 independent floats").
4. Land and bathymetric masks are strictly preserved; missing values are never fabricated or filled with synthetic numbers.

---

## 6. Known Scientific Limitations

- **Regional & Temporal Scope:** Model validation is strictly bounded to the Central Bay of Bengal ($12.125\text{--}17.875^\circ\text{N}, 85.125\text{--}92.875^\circ\text{E}$) during Q1 2020.
- **Surface Regime Bias:** A systematic cold bias of $-0.596\text{ °C}$ exists in the uppermost $0\text{--}20\text{ m}$ mixed layer relative to in situ floats.
- **Deep Ocean Regimes (>200m):** Below the thermocline, performance converges toward climatology due to weak surface thermal coupling.
- **Operational Status:** This is an interactive Proof-of-Concept research prototype. It does NOT replace operational numerical circulation models and does NOT provide cyclone forecasts.
