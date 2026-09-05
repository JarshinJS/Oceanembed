# GATE A4.3 — INDEPENDENT ARGO IN SITU VALIDATION REPORT

**Status:** COMPLETE & PASSED  
**Date:** September 2026  
**Project:** OceanEmbed (SIH26066)  
**Evaluated Model:** Frozen OceanEmbed v1-Local (Champion Architecture from Gate A4.1)  
**Observational Reference:** Independent in-situ observational reference: Un-gridded ARGO Profiling Floats (IFREMER GDAC ERDDAP)  
**Development Reanalysis Target:** Reanalysis-derived reference: GLORYS12V1 (0.083° gridded target)  
**Validation Domain:** Bay of Bengal ($12^\circ\text{--}18^\circ\text{N}, 85^\circ\text{--}93^\circ\text{E}$)  
**Validation Period:** 2020-03-16 through 2020-03-31 (16 days, temporally held-out)  

---

## 1. Validation Objective

Gate A4.3 serves as the independent in-situ observational validation for the OceanEmbed framework. Throughout Gates A2, A3, and A4.0–A4.2, neural network models were developed and evaluated against the **GLORYS12V1 numerical reanalysis product** (serving as the development training target and reanalysis-derived reference). 

The primary scientific objective of Gate A4.3 is to determine whether the predictive skill demonstrated against the GLORYS reanalysis reference—specifically the **thermocline reconstruction improvement** ($0.9949\text{ }^\circ\text{C}$ RMSE in Gate A4.1)—**transfers to independent in situ physical oceanographic observations** collected by autonomous ARGO profiling floats drifting in the Bay of Bengal.

---

## 2. Frozen Champion Model Definition

In accordance with strict verification rules:
- **Model:** `OceanEmbed v1-Local` (from Gate A4.1).
- **Architecture:**
  - Multimodal specialized input branches (Thermodynamics, Sea-Level, Currents, Winds);
  - Multi-scale spatial convolutional trunk (dilations 1, 2, 4);
  - Global context / self-attention bypassed (`use_global_context = False`);
  - Depth FiLM conditioning active (`use_depth_conditioning = True`);
  - Residual climatology formulation ($T = T_{\text{clim}} + \Delta T$).
- **Parameters:** 123,345.
- **Weights:** Strictly frozen from Gate A4.1 training protocol (seed 42). Verified on GLORYS test set: Overall RMSE = `0.8460 °C`, Thermocline RMSE = `0.9949 °C`.
- **Strict Prohibition Adherence:** No retraining, no fine-tuning on ARGO, no hyperparameter adjustment, and no model selection using ARGO observations.

---

## 3. ARGO Dataset & Data Independence Statement

### Acquisition Source
- **Provider:** International ARGO Program via IFREMER GDAC ERDDAP Server (`erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv`).
- **Geographic Bounding Box:** $12.0^\circ\text{N} \le \text{Latitude} \le 18.0^\circ\text{N}$, $85.0^\circ\text{E} \le \text{Longitude} \le 93.0^\circ\text{E}$.
- **Temporal Bounding Box:** `2020-03-16T00:00:00Z` to `2020-03-31T23:59:59Z`.
- **Platforms Identified:** 16 unique WMO float platforms in raw bounding-box query (40 profiles total): `2902230`, `2902233`, `2902235`, `2902236`, `2902264`, `2902278`, `2902279`, `2902280`, `2902282`, `2902283`, `2902596`, `2902766`, `2902768`, `2902769`, `2902770`, `2902772`.
- **Data Volume:** 398 KB raw text/CSV (no global dataset download required).

### Data Independence Statement
ARGO float observations were **completely untouched and isolated** throughout the entire development lifecycle:
- ARGO was NOT used to train model weights, compute normalization statistics, or calculate training climatologies.
- ARGO was NOT used in architectural search, ablation decisions, or loss formulation.
- ARGO serves strictly as an independent in-situ observational reference.

---

## 4. Quality Control & Collocation Methodology

### Quality Control (QC) Protocol
1. **Flag Filtering:** Observations filtered strictly for ARGO Quality Control flags `temp_qc` $\in \{1, 2\}$ and `pres_qc` $\in \{1, 2\}$ (1 = Good Data, 2 = Probably Good Data).
2. **Physical Validity Bounds:** Pressure $0 \le p \le 1050\text{ dbar}$ ($1\text{ dbar} \approx 0.993\text{ m}$); Temperature $0.0^\circ\text{C} \le T \le 35.0^\circ\text{C}$.
3. **Completeness Constraint:** Profiles with fewer than 5 valid vertical observation depths were rejected.

### Collocation Methodology
- **Temporal Collocation:** Each ARGO observation timestamp was matched to the corresponding daily OceanEmbed prediction field on the same UTC calendar day (temporal tolerance: $\pm 12\text{ hours}$).
- **Spatial Collocation:** 2D Bilinear Interpolation from the canonical $0.25^\circ$ ($24 \times 32$) cell-centered grid to the float's exact (lat, lon).
  - *Coastline & Mask Constraint:* Collocation was marked invalid and rejected if any of the four surrounding grid cells was masked (land or shallow bathymetry).
- **Vertical Interpolation:** Primary comparison was conducted at the **exact observed in situ ARGO depths** by linearly interpolating the model's 15 canonical depth predictions $[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$ to the ARGO sensor depths ($z \in [0, 1000]\text{ m}$).

---

## 5. Sample Counts & Collocation Summary

| Metric / Stage | Count | Notes |
|---|---|---|
| **Raw ARGO Observations** | 5,922 | Downloaded from IFREMER ERDDAP |
| **Raw Profiles Discovered** | 40 | 16 unique WMO platforms across bounding box |
| **Passed QC Observations** | 4,401 | 74.3% retention (bad flags / raw depth > 1050 dbar removed) |
| **Passed QC Profiles** | 40 | 100% of profiles met $\ge 5$ vertical points requirement |
| **Successfully Collocated Profiles** | **34** | **14 unique WMO platforms** successfully collocated inside canonical grid |
| **Rejected Profiles** | 6 | Rejection Reason: Float drifted outside canonical grid bounds: 2 platforms entirely excluded (`2902282` [3 profiles], `2902770` [2 profiles]) + 1 profile from platform `2902772` |
| **Total Paired In Situ Observations** | **3,966** | Valid, depth-matched pairs ($0\text{--}1000\text{ m}$) |

---

## 6. Primary Validation Results Across Vertical Regimes

The table below summarizes performance against independent ARGO in situ observations across vertical ocean regimes:

| Vertical Regime | Paired Observations | Number of Profiles | Model RMSE (°C) | Model MAE (°C) | Model Bias (°C) | Pearson Correlation ($r$) | Climatology RMSE (°C) | Climatology Bias (°C) | Relative RMSE Reduction vs Clim (%) | GLORYS Ref RMSE vs ARGO (°C) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Overall Column ($0\text{--}1000\text{ m}$)** | 3,966 | 34 | **0.8117** | **0.5426** | **-0.0863** | **0.9943** | 1.0947 | -0.1490 | **+25.86%** | 0.5776 |
| **Surface Mixed Layer ($0\text{--}20\text{ m}$)** | 417 | 34 | **1.0672** | **0.8555** | **-0.8188** | **0.5363** | 1.4975 | -1.3470 | **+28.74%** | 0.2855 |
| **Thermocline Band ($50\text{--}200\text{ m}$)** | 1,821 | 34 | **0.8446** | **0.6726** | **+0.0320** | **0.9849** | 1.2827 | +0.0383 | **+34.16%** | 0.6883 |
| **Deep Ocean ($>200\text{ m}$)** | 1,364 | 34 | **0.6927** | **0.2708** | **-0.1177** | **0.9549** | 0.6874 | -0.0737 | -0.77% | 0.5127 |

---

## 7. Paired Profile-Level Inference & Bootstrap Uncertainty Analysis

To rigorously evaluate predictive skill without inferring significance from non-overlapping marginal confidence intervals, paired profile-level inference was performed between OceanEmbed v1-Local profile RMSE and Training Climatology profile RMSE across all 34 collocated profiles.

### Paired Profile Difference:
For each profile $i \in \{1, \dots, 34\}$:
$$\Delta_i = \text{OceanEmbed\_RMSE}_i - \text{Climatology\_RMSE}_i$$

- **Mean Paired RMSE Difference ($\bar{\Delta}$):** **`-0.2231 °C`**
- **Standard Deviation of Difference:** `0.2229 °C`
- **Median Paired Difference:** **`-0.1898 °C`**
- **Profile Win Rate:** **29 of 34 profiles (85.29%)** exhibit lower RMSE for OceanEmbed v1-Local than climatology.
- **95% Bootstrap Confidence Interval over Profile IDs ($B = 10,000$):** **`[-0.2972, -0.1499] °C`**
- **Bootstrap Empirical Probability $P(\bar{\Delta}^* < 0)$:** **`1.000`** (10,000 of 10,000 resamples negative).
- **Paired Sign-Flip Permutation Test ($M = 100,000$):** **`p < 0.00001`** (two-sided permutation $p = 0.0 / 100,000$).
- **Paired Parametric Student's $t$-test:** $t = -5.8358, p = 1.565 \times 10^{-6}$.
- **Paired Non-Parametric Wilcoxon Signed-Rank Test:** $W = 40.0, p = 1.009 \times 10^{-6}$.

> [!NOTE]
> Statistical significance is demonstrated via paired resampling and permutation tests over profile differences. Non-overlapping marginal confidence intervals between models are not used as proof of significance.

### Profile-Level Marginal Uncertainty Summary ($B = 10,000$ Bootstrap Resamples):
- **OceanEmbed v1-Local Profile RMSE:** `0.7624 °C` (95% CI: `[0.6734, 0.8607] °C`)
- **OceanEmbed v1-Local Profile MAE:** `0.5471 °C` (95% CI: `[0.4922, 0.6019] °C`)
- **OceanEmbed v1-Local Profile Bias:** `-0.2072 °C` (95% CI: `[-0.3194, -0.0904] °C`)
- **Training Climatology Profile RMSE:** `0.9854 °C` (95% CI: `[0.8843, 1.0938] °C`)
- **Training Climatology Profile MAE:** `0.6933 °C` (95% CI: `[0.6263, 0.7659] °C`)
- **Training Climatology Profile Bias:** `-0.3170 °C` (95% CI: `[-0.4802, -0.1428] °C`)
- **GLORYS Reference Profile RMSE vs ARGO:** `0.4886 °C` (95% CI: `[0.4285, 0.5574] °C`)

---

## 8. Complete Depth-Wise Profile Analysis Against In Situ ARGO

Calculated using explicit tolerance brackets around each canonical depth level:

| Canonical Depth (m) | Depth Tolerance (m) | ARGO Obs Count ($N$) | ARGO Profiles ($N$) | OceanEmbed v1-Local RMSE (°C) | OceanEmbed MAE (°C) | OceanEmbed Bias (°C) | Reference Climatology RMSE (°C) | GLORYS Reference RMSE (°C) | Relative Skill vs Clim (%) |
|---|---|---|---|---|---|---|---|---|---|
| **0** | $\pm 3.0$ | 71 | 21 | **1.1104** | 0.9468 | -0.9416 | 1.6353 | 0.2834 | **+32.10%** |
| **5** | $\pm 3.0$ | 180 | 34 | **1.2761** | 1.0749 | -1.0717 | 1.6906 | 0.3053 | **+24.52%** |
| **10** | $\pm 4.0$ | 186 | 34 | **1.1254** | 0.9122 | -0.9008 | 1.5468 | 0.2825 | **+27.24%** |
| **20** | $\pm 5.0$ | 132 | 33 | **0.6302** | 0.4854 | -0.1922 | 0.8924 | 0.2836 | **+29.38%** |
| **30** | $\pm 5.0$ | 131 | 32 | **0.5934** | 0.4842 | +0.1414 | 0.5843 | 0.3611 | -1.56% |
| **50** | $\pm 7.5$ | 194 | 34 | **0.7926** | 0.6083 | +0.4416 | 0.8053 | 0.5241 | **+1.58%** |
| **75** | $\pm 10.0$ | 257 | 34 | **0.7181** | 0.5402 | -0.2326 | 1.1931 | 0.6608 | **+39.81%** |
| **100** | $\pm 12.5$ | 295 | 34 | **1.1043** | 0.9264 | -0.2636 | 1.8163 | 0.8143 | **+39.20%** |
| **125** | $\pm 12.5$ | 314 | 34 | **1.0453** | 0.8682 | +0.0295 | 1.5615 | 0.7943 | **+33.06%** |
| **150** | $\pm 15.0$ | 377 | 34 | **0.8187** | 0.6856 | +0.1310 | 1.2105 | 0.6821 | **+32.37%** |
| **200** | $\pm 20.0$ | 336 | 34 | **0.5155** | 0.4167 | +0.2813 | 0.6540 | 0.4589 | **+21.18%** |
| **300** | $\pm 35.0$ | 84 | 34 | **0.2456** | 0.1888 | -0.0266 | 0.2134 | 0.1773 | -15.09% |
| **500** | $\pm 50.0$ | 154 | 34 | **0.8791** | 0.2694 | -0.2117 | 0.8628 | 0.3461 | -1.89% |
| **700** | $\pm 75.0$ | 228 | 34 | **0.8401** | 0.3054 | -0.2886 | 0.8213 | 0.5746 | -2.29% |
| **1000** | $\pm 100.0$ | 152 | 34 | **0.7226** | 0.2590 | -0.2380 | 0.6936 | 0.8011 | -4.18% |

---

## 9. Critical Findings & Evaluation Questions

### A. Does OceanEmbed retain positive predictive skill against real in situ observations?
**YES.**  
Against 3,966 independent in situ ARGO observations across 34 profiles, OceanEmbed v1-Local achieved an overall column RMSE of **`0.8117 °C`** compared to Reference Climatology's **`1.0947 °C`**, representing a **`25.86%` error reduction** and a Pearson correlation of **`r = 0.9943`**.

### B. Does the thermocline improvement demonstrated against GLORYS transfer to ARGO?
**YES.**  
In the thermocline band ($50\text{--}200\text{ m}$), OceanEmbed v1-Local achieved:
- **Thermocline RMSE:** **`0.8446 °C`** vs Climatology's **`1.2827 °C`** (a **`34.16%` error reduction**);
- **Systematic Bias:** **`+0.0320 °C`** (virtually unbiased across the entire thermocline);
- **Pearson Correlation:** **`r = 0.9849`**.

At the core pycnocline depths ($75\text{ m}, 100\text{ m}, 125\text{ m}$), OceanEmbed outperformed climatology by **`39.81%`**, **`39.20%`**, and **`33.06%`** respectively. This demonstrates that the physical relationship learned by OceanEmbed—mapping multi-scale sea surface height, currents, and thermodynamics to subsurface thermal anomalies—transfers to real-world subsurface temperature structure within the evaluated regional and temporal validation window.

### C. Does the near-surface error seen against GLORYS persist against ARGO?
**YES.**  
In the upper mixed layer ($0\text{--}20\text{ m}$), OceanEmbed v1-Local exhibited an RMSE of **`1.0672 °C`** with a systematic cold bias of **`-0.8188 °C`**.
- Notably, Reference Climatology had an even larger error (**`1.4975 °C`** RMSE, Bias `-1.3470 °C`), confirming that OceanEmbed is **`28.74%` better than climatology at the surface**.
- In contrast, GLORYS reanalysis achieves `0.2855 °C` RMSE at the surface because GLORYS explicitly assimilates satellite SST and in situ observations into its bulk mixed-layer physics equations.

### D. Comparison with the GLORYS Reanalysis Reference
- **GLORYS vs ARGO:** GLORYS achieves an overall RMSE of **`0.5776 °C`** and thermocline RMSE of **`0.6883 °C`** against ARGO. This is expected because GLORYS12V1 is an operational numerical reanalysis assimilating millions of in situ and satellite observations via a 3D-Var assimilation engine coupled to the NEMO primitive equation ocean model.
- **OceanEmbed vs GLORYS:** OceanEmbed operates as a pure **satellite-to-subsurface projection model** without assimilating in situ profilers or running 3D fluid dynamic time-stepping. Achieving **`0.8446 °C`** in the thermocline against raw physical float data indicates that satellite surface embeddings carry meaningful subsurface baroclinic information.

---

## 10. Failure Analysis

1. **Upper-Ocean Cold Bias in Late March:**  
   The model exhibits a cold bias of $-0.94\text{ }^\circ\text{C}$ at $0\text{ m}$ and $-1.07\text{ }^\circ\text{C}$ at $5\text{ m}$. This occurs because late March in the Bay of Bengal experiences intense solar radiation and pre-monsoon surface stratification, causing rapid heating in the top 10 meters. Because training climatology was calculated from January–February (winter cooling / moderate spring transition), the climatological anchor was cooler than late March in situ observations.
2. **Deep Ocean ($>200\text{ m}$) Residual Convergence:**  
   Below $200\text{ m}$, temperature variations are small ($< 0.5\text{ }^\circ\text{C}$ temporal standard deviation). OceanEmbed's error (`0.6927 °C`) aligns closely with climatology (`0.6874 °C`), confirming that deep isothermal waters are dominated by background climatological hydrography rather than instantaneous surface wind/eddy forcing.
3. **Regional Float Boundary Drift:**  
   Out of 40 raw QC-passed profiles, 6 profiles drifted outside the canonical $12.125^\circ\text{--}17.875^\circ\text{N}, 85.125^\circ\text{--}92.875^\circ\text{E}$ bounding box into coastal waters or northern latitudes, correctly excluded by spatial bilinear domain bounds.

---

## 11. Limitations

1. **Temporal Sample Window & Platform Clustering:** Evaluated across a 16-day late-March test window (34 collocated profiles from 14 unique WMO float platforms, treated as conservative cluster units). While the paired profile-level RMSE reduction is supported by both unclustered and platform-clustered bootstrap testing (clustered 95% CI [-0.3457, -0.1117] °C, exact platform-level permutation $p = 0.0017$), seasonal monsoon transitions (e.g., June–September southwest monsoon) remain to be evaluated in future work.
2. **Geographic Scope:** Validated specifically in the Bay of Bengal tropical basin. Global generalization requires broader multi-basin training.
3. **Observational Density:** With 34 profiles in 16 days across 14 floats, temporal float revisit frequency is approximately once per 10 days per float.

---

## 12. Evaluator Verdict

**GATE A4.3 STATUS: PASSED.**

These results support the hypothesis that multimodal surface observations contain information useful for reconstructing subsurface temperature structure, including the thermocline, within the evaluated regional and temporal validation window.

---

## 13. Persisted Artifacts

- **Statistical Audit Report:** `Dataset/gate_a4_3/reports/GATE_A4_3_STATISTICAL_AUDIT.md`
- **Paired Inference JSON:** `Dataset/gate_a4_3/metrics/argo_paired_inference.json`
- **Paired Inference CSV:** `Dataset/gate_a4_3/metrics/argo_paired_inference.csv`
- **Validation Metrics JSON:** `Dataset/gate_a4_3/metrics/argo_validation_results.json`
- **Profile-Level Metrics CSV:** `Dataset/gate_a4_3/metrics/argo_profile_metrics.csv`
- **Depth-Wise Metrics CSV:** `Dataset/gate_a4_3/metrics/argo_depth_metrics.csv`
- **NetCDF Collocated Predictions:** `Dataset/gate_a4_3/predictions/argo_collocated_predictions.nc`
- **Diagnostic Figures:**
  - `Dataset/gate_a4_3/figures/argo_locations.png`
  - `Dataset/gate_a4_3/figures/argo_profile_comparison.png`
  - `Dataset/gate_a4_3/figures/argo_depth_rmse.png`
  - `Dataset/gate_a4_3/figures/argo_depth_bias.png`
  - `Dataset/gate_a4_3/figures/argo_scatter.png`
