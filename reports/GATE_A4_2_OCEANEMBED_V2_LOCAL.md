# GATE A4.2 — OCEANEMBED V2-LOCAL REFINEMENT REPORT

**Status:** COMPLETE  
**Date:** September 2026  
**Project:** OceanEmbed (SIH26066)  
**Target Reference:** GLORYS12V1 Reanalysis-Derived Subsurface Target (0–1000 m)  
**Evaluation Period:** 2020-03-16 through 2020-03-31 (16 days, temporally held-out)  

---

## 1. Motivation from Gate A4.1

In Gate A4.1, a controlled ablation study of OceanEmbed v1 established that:
1. **The Global Context (Spatial Self-Attention) block caused the overall RMSE penalty** on the regional $24 \times 32$ domain. Ablating self-attention (`OceanEmbed v1-Local`) yielded a superior overall column RMSE of **`0.8460 °C`** (beating the frozen A3 CNN benchmark of `0.8602 °C` by $1.65\%$).
2. **Thermocline reconstruction improved substantially** to **`0.9949 °C`** (outperforming Climatology `1.1295 °C` by $11.92\%$ and A3 CNN `1.1914 °C` by $16.49\%$).
3. **Core Remaining Weakness:** Despite strong interior thermocline performance, OceanEmbed v1-Local exhibited an elevated error in the near-surface mixed layer ($0\text{--}20\text{ m}$):
   - At $0\text{ m}$: `1.1523 °C` (v1-Local) vs `0.7701 °C` (A3 CNN);
   - At $5\text{ m}$: `1.0015 °C` vs `0.6103 °C`;
   - At $10\text{ m}$: `0.9699 °C` vs `0.5345 °C`.

The objective of Gate A4.2 was to implement a lightweight learned surface-preservation pathway to resolve this upper-ocean error gap **without sacrificing the established thermocline reconstruction ($50\text{--}200\text{ m} \le 0.9949\text{ }^\circ\text{C}$)**.

---

## 2. OceanEmbed v2-Local Architecture

OceanEmbed v2-Local strictly follows the evidence-supported architectural constraints derived in Gate A4.1:
- **RETAINED**:
  1. *Multimodal Specialized Encoders:* Four dedicated feature extraction branches for Thermodynamics (SST, SSS), Sea-Level (SSH/ADT), Ocean Dynamics (OSCAR U, V), and Atmospheric Forcing (CCMP U, V), projected and concatenated into a fused latent representation of dimension $C=48$.
  2. *Multi-Scale Spatial Receptive Field Trunk:* Dilated convolutions with dilation rates $(1, 2, 4)$ to capture mesoscale eddy dynamics.
  3. *Residual Climatology Formulation:* $T_{\text{pred}} = T_{\text{clim}} + \Delta T$, anchoring deep unobserved ocean layers to training climatology.
- **EXCLUDED**:
  1. *Global Self-Attention:* Strictly excluded (identity bypass).
  2. *Depth FiLM Conditioning:* Strictly excluded; replaced with a direct spatial convolutional decoder.
- **ADDED**:
  - *Learned Surface Refinement Pathway:* A lightweight auxiliary convolutional pathway dedicated to upper-ocean layers ($0\text{ m}, 5\text{ m}, 10\text{ m}, 20\text{ m}$).

**Total Parameters:** 102,603 (lightweight edge-deployment profile).

---

## 3. Surface Refinement Pathway Design & Tensor Flow

```
Surface Inputs X: [B, 7, 24, 32]
  ├── (Channels 0..6) ──> MultimodalBranchEncoder ────────> [B, 48, 24, 32]
  │                                                               │
  │                                                               ▼
  │                                                   MultiScaleSpatialEncoder (dilations 1,2,4)
  │                                                               │
  │                                                               ▼
  │                                                    Latent Tensor Z: [B, 48, 24, 32]
  │                                                               ├───┐
  │                                                               │   │
  │   ┌───────────────────────────────────────────────────────────┘   │
  │   │                                                               │
  │   ▼                                                               ▼
  │  Main Spatial Decoder (No FiLM, No Attention)           Surface Refinement Pathway
  │   Conv3x3 -> BN -> GELU -> Conv1x1 (15 ch)               Cat([X[:, 0:3], Z]) -> [B, 51, 24, 32]
  │   Output: ΔT_main: [B, 15, 24, 32]                       Conv3x3(51->24) -> BN -> GELU
  │   ├── Depths 0..3 (0, 5, 10, 20m) ──────────────┐        Conv3x3(24->4, init small)
  │   │                                             ▼        Output: ΔT_upper: [B, 4, 24, 32]
  │   │                                      Add: ΔT_main[:, :4] + ΔT_upper
  │   │                                             │
  │   │                                             ▼
  │   │                                    ΔT_refined[:, :4] = ΔT_main[:, :4] + ΔT_upper
  │   └── Depths 4..14 (30..1000m) ──────> ΔT_refined[:, 4:] = ΔT_main[:, 4:]  (UNTOUCHED)
  │
  ▼
Final Temperature:
T_pred = T_climatology_train + ΔT_refined: [B, 15, 24, 32]
```

### Key Design Principles
1. **No Hard-Coded Identity:** Because satellite SST (OSTIA L4) and GLORYS reanalysis $0\text{ m}$ temperature represent distinct physical and assimilated quantities, hard-coding $T(0\text{ m}) = \text{SST}$ is invalid. The pathway learns a non-linear residual mapping.
2. **Explicit Depth Slicing:** Depths $30\text{--}1000\text{ m}$ (encompassing the entire $50\text{--}200\text{ m}$ thermocline) are structurally decoupled from the surface refinement head output.

---

## 4. Controlled Training Protocol

To ensure 100% comparability with A3.3 and A4.1:
- **Chronological Split:** Train (Jan 01 – Feb 29, 60 days), Val (Mar 01 – Mar 15, 15 days), Test (Mar 16 – Mar 31, 16 days).
- **Normalization:** Z-score statistics computed strictly from Train split (Days 1–60).
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau(factor=0.5, patience=5)`.
- **Loss:** `MaskedMSELoss` over ocean pixels.
- **Batch Size:** 8; **Epochs:** 50; **Seed:** 42.

Training converged in **8.11 seconds** on CUDA, achieving minimum validation loss at epoch 50 (`0.2180`).

---

## 5. Primary Model Comparison

All models evaluated strictly on the **16-day held-out test window** (`2020-03-16` to `2020-03-31`):

| Model Configuration | Parameters | Overall RMSE (°C) | Overall MAE (°C) | Mean Bias (°C) | Thermocline RMSE 50–200m (°C) | Thermocline MAE (°C) | Thermocline Bias (°C) |
|---|---|---|---|---|---|---|---|
| **Baseline 0: Climatology** | 0 | 1.0590 | 0.7668 | -0.3575 | 1.1295 | 0.8365 | +0.0227 |
| **Baseline 1: Frozen A3 CNN** | 39,759 | 0.8602 | 0.5771 | +0.0779 | 1.1914 | 0.8352 | +0.2584 |
| **OceanEmbed v1** | 127,954 | 0.9239 | 0.6442 | -0.1667 | 1.0455 | 0.7396 | +0.2452 |
| **Frozen A4.1 v1-Local** | 123,345 | **0.8460** | **0.5917** | -0.1709 | **0.9949** | **0.7270** | **+0.0792** |
| **A4.2 OceanEmbed v2-Local** | 102,603 | 0.8738 | 0.6074 | -0.1536 | 1.0296 | 0.7418 | +0.1710 |

---

## 6. Upper-Ocean Performance Breakdown ($0\text{--}20\text{ m}$)

| Depth Level (m) | Frozen A3 CNN (°C) | Frozen A4.1 v1-Local (°C) | A4.2 OceanEmbed v2-Local (°C) | v2 Change vs v1-Local (°C) | Relative Improvement (%) |
|---|---|---|---|---|---|
| **0 m** | 0.7701 | 1.1523 | 1.2489 | +0.0966 | -8.38% (degraded) |
| **5 m** | 0.6103 | 1.0015 | 1.1006 | +0.0991 | -9.90% (degraded) |
| **10 m** | 0.5345 | 0.9699 | 1.0614 | +0.0915 | -9.43% (degraded) |
| **20 m** | 0.6081 | 0.7964 | **0.6932** | **-0.1032** | **+12.96% (improved)** |

---

## 7. Complete Depth-Wise RMSE Profile (All 15 Depths)

| Depth (m) | Regime | Baseline Climatology | Frozen A3 CNN | Frozen v1-Local | A4.2 v2-Local |
|---|---|---|---|---|---|
| **0** | Mixed Layer | 1.6737 | 0.7701 | 1.1523 | 1.2489 |
| **5** | Mixed Layer | 1.4765 | 0.6103 | 1.0015 | 1.1006 |
| **10** | Mixed Layer | 1.4312 | 0.5345 | 0.9699 | 1.0614 |
| **20** | Mixed Layer | 1.1040 | 0.6081 | 0.7964 | **0.6932** |
| **30** | Mixed Layer | 0.8124 | 0.8274 | 0.8198 | **0.6949** |
| **50** | **Thermocline** | 0.8167 | 1.0794 | 0.9134 | **0.8365** |
| **75** | **Thermocline** | 1.1976 | 1.2557 | 1.0783 | 1.0873 |
| **100** | **Thermocline** | 1.5154 | 1.5640 | 1.2381 | 1.3193 |
| **125** | **Thermocline** | 1.3361 | 1.3106 | 1.1157 | 1.2040 |
| **150** | **Thermocline** | 1.0560 | 1.1146 | 0.9175 | 0.9875 |
| **200** | **Thermocline** | 0.5979 | 0.6064 | 0.5672 | **0.5603** |
| **300** | Deep Ocean | 0.2537 | 0.2890 | 0.2581 | 0.2767 |
| **500** | Deep Ocean | 0.1844 | 0.1860 | 0.1905 | 0.1895 |
| **700** | Deep Ocean | 0.2090 | 0.3281 | 0.2051 | 0.2013 |
| **1000** | Deep Ocean | 0.2011 | 0.2338 | 0.2024 | 0.2062 |

---

## 8. Evaluation Against Primary Success Criteria

### Criterion 1: Thermocline Preservation ($\le 0.9949\text{ }^\circ\text{C}$)
- **Result: PARTIALLY COMPROMISED.**
- A4.2 OceanEmbed v2-Local achieved a thermocline RMSE of **`1.0296 °C`**.
- While v2-Local **strongly outperforms** the frozen A3 CNN (`1.1914 °C`, $+13.58\%$ skill) and Climatology (`1.1295 °C`, $+8.84\%$ skill), its thermocline RMSE is $+0.0347\text{ }^\circ\text{C}$ higher than the frozen A4.1 v1-Local benchmark (`0.9949 °C`).

### Criterion 2: Overall Column RMSE Improvement ($< 0.8460\text{ }^\circ\text{C}$)
- **Result: NOT ACHIEVED.**
- Overall RMSE was **`0.8738 °C`**, falling short of v1-Local's `0.8460 °C` ($+0.0278\text{ }^\circ\text{C}$ penalty).

### Criterion 3: Upper-Ocean Error Reduction ($0\text{--}20\text{ m}$)
- **Result: MIXED.**
- At **$20\text{ m}$**, the model demonstrated strong improvement: RMSE dropped from $0.7964\text{ }^\circ\text{C}$ to **$0.6932\text{ }^\circ\text{C}$** (**$+12.96\%$ error reduction**), approaching the A3 CNN benchmark ($0.6081\text{ }^\circ\text{C}$).
- At $30\text{ m}$, RMSE improved from $0.8198\text{ }^\circ\text{C}$ to **$0.6949\text{ }^\circ\text{C}$** ($+15.23\%$ error reduction).
- However, at $0\text{--}10\text{ m}$, errors slightly increased ($1.1523 \to 1.2489\text{ }^\circ\text{C}$ at surface).

---

## 9. Failure Analysis: Why the Surface Pathway Did Not Resolve 0–10m Errors

1. **Gradient Interference with the Shared Latent Trunk:**  
   The surface refinement pathway received both raw surface features (`X[:, 0:3]`) and the shared latent representation ($Z$). During backpropagation, large loss gradients from the high-variance surface skin layer ($0\text{--}10\text{ m}$) backpropagated through $Z$ into the multi-scale spatial trunk. This gradient flow perturbed the spatial feature representations required by the deeper thermocline levels ($50\text{--}125\text{ m}$), explaining the slight degradation from `0.9949 °C` to `1.0296 °C`.
2. **Product Divergence Between OSTIA and GLORYS:**  
   OSTIA foundation SST represents night-time sub-skin temperature derived from satellite infrared/microwave radiometers, whereas GLORYS $0\text{ m}$ temperature is an assimilated numerical model product influenced by atmospheric reanalysis heat flux formulations and ocean vertical mixing physics. A simple 2-layer CNN cannot bridge this systemic product divergence across a 60-day training set without over-fitting to regional March transitions.
3. **Absence of Depth-Specific Decoupling in the Decoder:**  
   In A4.1, `without_global_context` retained the discrete depth embedding module, allowing the network to modulate vertical representation independently. In v2-Local, removing both attention and depth FiLM left the unconditioned conv decoder with less vertical capacity.

---

## 10. Limitations

1. **Single-Region Regional Domain:** Results are evaluated strictly on the $12^\circ\text{--}18^\circ\text{N}, 85^\circ\text{--}93^\circ\text{E}$ Bay of Bengal domain during Q1 2020.
2. **Reanalysis Reference Constraint:** Evaluation is performed strictly against GLORYS12V1 reanalysis-derived reference fields; in situ ARGO float validation is required to confirm real-world observational accuracy.
3. **Short Temporal Sample Size:** A 60-day training split (Jan–Feb) evaluated on late March captures seasonal warming onset, where surface skin dynamics diverge from winter-spring background climatology.

---

## 11. Final Recommendation for Gate A4.3

1. **Classify Surface Refinement via Shared Latents as Suboptimal:**  
   Coupling surface refinement to the shared spatial latent $Z$ introduces optimization tension between surface skin fitting and interior thermocline retrieval.
2. **Preserve OceanEmbed v1-Local as the Production Baseline:**  
   OceanEmbed v1-Local remains the Pareto-best configuration (**`0.8460 °C` overall, `0.9949 °C` thermocline**).
3. **Recommendation for A4.3 (ARGO Point Validation Preparation):**  
   Prioritize validating OceanEmbed v1-Local against true independent, un-gridded in situ **ARGO profiling floats** to determine whether the surface divergence is an artifact of the GLORYS reference or a genuine physical limitation of satellite-to-subsurface projection.
