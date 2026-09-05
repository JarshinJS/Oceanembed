# GATE A4.3 — FINAL DEPTH-BIN INDEPENDENCE AND FREEZE AUDIT

**Audit Date:** September 2026  
**Status:** PASS WITH LIMITATION  
**Project:** OceanEmbed (SIH26066)  
**Evaluated Model:** Frozen OceanEmbed v1-Local (Champion Architecture from Gate A4.1)  
**Evaluated Artifacts:**
- Prediction NetCDF: `Dataset/gate_a4_3/predictions/argo_collocated_predictions.nc`
- Profile Metrics: `Dataset/gate_a4_3/metrics/argo_profile_metrics.csv`
- Depth Metrics: `Dataset/gate_a4_3/metrics/argo_depth_metrics.csv`
- Platform Clustered Metrics: `Dataset/gate_a4_3/metrics/argo_platform_clustered_inference.json` & `.csv`
- ARGO Validation Report: `Dataset/gate_a4_3/reports/GATE_A4_3_ARGO_VALIDATION.md`
- Statistical Audit Report: `Dataset/gate_a4_3/reports/GATE_A4_3_STATISTICAL_AUDIT.md`
- Platform Independence Audit: `Dataset/gate_a4_3/reports/GATE_A4_3_PLATFORM_INDEPENDENCE_AUDIT.md`

---

## 1. Executive Summary

This final audit completes the pre-freeze evaluation of Gate A4.3, with specific focus on:
1. Auditing the canonical-depth assignment rule and testing for mutual exclusivity across depth tolerance brackets;
2. Confirming that all key sample counts, profile metrics, clustered bootstrap intervals, and platform-level permutation tests remain intact without modification;
3. Ensuring strict compliance with scientific phrasing guidelines;
4. Establishing the definitive freeze status of Gate A4.3.

**FINAL AUDIT VERDICT: PASS WITH LIMITATION.**  
Gate A4.3 is scientifically locked and approved for **FREEZE**.

---

## 2. Canonical Depth Assignment Audit

### The Reported Depth Tolerance Brackets:
The 15 canonical depth bins are evaluated using nominal target depths and half-width tolerances $[d - \text{tol}, d + \text{tol}]$:

| Canonical Depth ($d$) | Tolerance ($\text{tol}$) | Depth Window Range (m) | Overlap with Adjacent Windows? |
|---|---|---|---|
| **0 m** | $\pm 3.0\text{ m}$ | $[0.0, 3.0]$ | Overlaps with 5 m window on $[2.0, 3.0\text{ m}]$ |
| **5 m** | $\pm 3.0\text{ m}$ | $[2.0, 8.0]$ | Overlaps with 0 m on $[2.0, 3.0\text{ m}]$; with 10 m on $[6.0, 8.0\text{ m}]$ |
| **10 m** | $\pm 4.0\text{ m}$ | $[6.0, 14.0]$ | Overlaps with 5 m on $[6.0, 8.0\text{ m}]$ |
| **20 m** | $\pm 5.0\text{ m}$ | $[15.0, 25.0]$ | Boundary intersection with 30 m window at $z = 25.0\text{ m}$ |
| **30 m** | $\pm 5.0\text{ m}$ | $[25.0, 35.0]$ | Boundary intersection with 20 m window at $z = 25.0\text{ m}$ |
| **50 m** | $\pm 7.5\text{ m}$ | $[42.5, 57.5]$ | Disjoint (gap $[35.0, 42.5\text{ m}]$) |
| **75 m** | $\pm 10.0\text{ m}$ | $[65.0, 85.0]$ | Disjoint (gap $[57.5, 65.0\text{ m}]$) |
| **100 m** | $\pm 12.5\text{ m}$ | $[87.5, 112.5]$ | Boundary intersection with 125 m window at $z = 112.5\text{ m}$ |
| **125 m** | $\pm 12.5\text{ m}$ | $[112.5, 137.5]$ | Overlaps with 150 m window on $[135.0, 137.5\text{ m}]$ |
| **150 m** | $\pm 15.0\text{ m}$ | $[135.0, 165.0]$ | Overlaps with 125 m window on $[135.0, 137.5\text{ m}]$ |
| **200 m** | $\pm 20.0\text{ m}$ | $[180.0, 220.0]$ | Disjoint (gap $[165.0, 180.0\text{ m}]$) |
| **300 m** | $\pm 35.0\text{ m}$ | $[265.0, 335.0]$ | Disjoint (gap $[220.0, 265.0\text{ m}]$) |
| **500 m** | $\pm 50.0\text{ m}$ | $[450.0, 550.0]$ | Disjoint (gap $[335.0, 450.0\text{ m}]$) |
| **700 m** | $\pm 75.0\text{ m}$ | $[625.0, 775.0]$ | Disjoint (gap $[550.0, 625.0\text{ m}]$) |
| **1000 m** | $\pm 100.0\text{ m}$ | $[900.0, 1100.0]$ | Disjoint (gap $[775.0, 900.0\text{ m}]$) |

---

### Actual Implementation Rule in Code:
Inspection of `src/gate_a4/argo_validation.py` (line 518) reveals the exact filter logic:
```python
sub_d = df_paired[(df_paired["depth_m"] >= d - tol) & (df_paired["depth_m"] <= d + tol)]
```
Each canonical depth level is evaluated as an **independent window query**.

### Exclusivity Audit Findings:
1. **Are observations mutually exclusive?**  
   **NO.** Assignment is **non-mutually exclusive**.
2. **Was nearest-canonical-depth partitioning used?**  
   **NO.** Any observation whose sensor depth falls within $[d - \text{tol}, d + \text{tol}]$ is included in that depth bin's evaluation.
3. **Observation Membership Breakdown:**  
   Across the **3,966** total collocated observations:
   - **Belongs to exactly 1 bin:** **2,753 observations (69.42%)**
   - **Belongs to 2 overlapping bins:** **169 observations (4.26%)**
   - **Belongs to 0 bins (in inter-bin gaps):** **1,044 observations (26.32%)**
   - **Belongs to $\ge 3$ bins:** **0 observations (0.00%)**
4. **Summability of Depth-Wise Counts:**  
   The sum of `n_observations` across the 15 depth rows in `argo_depth_metrics.csv` is **3,091** (not 3,966).
   - **2,922 unique observations** are represented across the 15 depth bins.
   - **169 observations** are evaluated in two adjacent depth bins.
   - **1,044 observations** fell between discrete bracket gaps and are evaluated in the continuous regime metrics (Overall column, Surface, Thermocline, Deep) rather than in nominal depth rows.

> [!IMPORTANT]
> **Methodological Clarification & Limitation:**  
> The 15 canonical depth rows in `argo_depth_metrics.csv` **must be interpreted as localized, overlapping descriptive slices**, rather than a mutually exclusive partition of the water column. The primary column-wide performance is established by the continuous vertical regime metrics (Overall column $0\text{--}1000\text{ m}$, $N = 3,966$).

---

## 2. Preservation of Verified Sample Counts & Baseline Metrics

All primary metrics and sample counts established in Gate A4.3 are preserved:

| Verified Parameter | Verified Value | Artifact Source |
|---|---|---|
| **Total Paired Observations** | **3,966** | `argo_collocated_predictions.nc` |
| **Collocated Vertical Profiles** | **34** | `argo_profile_metrics.csv` |
| **Raw ERDDAP Platforms** | **16** | `argo_raw_bob_20200316_20200331.csv` |
| **Collocated Float Platforms** | **14 unique platforms** (treated as conservative cluster units) | `argo_profile_metrics.csv` |
| **Boundary-Excluded Profiles** | **6** (platforms `2902282` [3], `2902770` [2], `2902772` [1]) | Quality Control & Grid Mask Bounds |
| **Overall Column Model RMSE** | **`0.8117 °C`** | `argo_validation_results.json` |
| **Overall Column Climatology RMSE** | **`1.0947 °C`** | `argo_validation_results.json` |
| **Relative RMSE Reduction vs Clim** | **`+25.86%`** | `argo_validation_results.json` |
| **Thermocline Band Model RMSE** | **`0.8446 °C`** (Clim: `1.2827 °C`, -34.16%) | `argo_validation_results.json` |
| **Surface Mixed Layer Model RMSE** | **`1.0672 °C`** (Clim: `1.4975 °C`, -28.74%) | `argo_validation_results.json` |
| **Deep Ocean Model RMSE** | **`0.6927 °C`** (Clim: `0.6874 °C`, -0.77%) | `argo_validation_results.json` |

---

## 4. Platform-Level & Clustered Inference Confirmation

Both unclustered profile-level, platform-clustered bootstrap, and platform-level aggregated analyses are confirmed:

1. **Profile-Level Paired Difference ($N = 34$):**  
   - $\bar{\Delta} = -0.2231\text{ }^\circ\text{C}$ (Standard Deviation: $0.2229\text{ }^\circ\text{C}$)
   - 95% Bootstrap CI ($B = 10,000$): `[-0.2972, -0.1499] °C`
   - Sign-flip permutation test ($M = 100,000$): **`p < 1e-5`** ($p = \frac{0 + 1}{100000 + 1} \approx 9.999 \times 10^{-6}$)
   - Paired Student's $t$-test: $t = -5.8358, p = 1.565 \times 10^{-6}$
   - Wilcoxon signed-rank test: $W = 40.0, p = 1.009 \times 10^{-6}$

2. **Platform-Clustered Bootstrap Sensitivity ($B = 10,000$, Clusters = 14):**  
   - Clustered Bootstrap Mean: **`-0.2236 °C`**
   - Clustered 95% Bootstrap CI: **`[-0.3457, -0.1117] °C`**
   - $P(\bar{\Delta}^* < 0)$: **`1.000000`** (0 of 10,000 resamples $\ge 0$)

3. **Conservative Platform-Level Aggregated Analysis ($N = 14$):**  
   - Mean Platform-Level Difference: **`-0.2140 °C`**
   - Median Platform-Level Difference: **`-0.2006 °C`**
   - Platform Win Rate: **11 of 14 platforms (78.57%)** favor OceanEmbed v1-Local
   - Exact Platform Permutation Test ($2^{14} = 16,384$ combinations): **`p = 0.001709`**
   - Platform-level $t$-test: $t = -3.9726, p = 0.001592$
   - Platform-level Wilcoxon test: $W = 8.0, p = 0.003052$

---

## 5. Scientific Phrasing & Behavioral Guideline Compliance

An audit of all reports confirms that:
- The phrasing **"14 unique WMO float platforms, treated as conservative cluster units"** has replaced unqualified descriptions of float independence.
- Mechanistic claims regarding unmeasured **"isopycnal displacement"** have been excised; descriptions focus on reconstructed thermal structure.
- Overstrong terms ("ground truth", "decisively", "proves", "verified", "operational validation") have been removed or replaced with neutral, defensible scientific phrasing.
- GLORYS is designated as **"reanalysis-derived reference"** and ARGO as **"independent in-situ observational reference"**.
- The overall evaluative verdict adheres strictly to the required hypothesis-testing standard:
  > *"These results support the hypothesis that multimodal surface observations contain information useful for reconstructing subsurface temperature structure, including the thermocline, within the evaluated regional and temporal validation window."*

---

## 6. Documented Limitations

1. **Temporal Sample Window:** Evaluated across a 16-day late-March test window (34 collocated profiles from 14 unique WMO float platforms, treated as conservative cluster units). Seasonal monsoon transitions remain to be evaluated in future work.
2. **Geographic Scope:** Validated specifically within the Bay of Bengal tropical basin ($12^\circ\text{--}18^\circ\text{N}, 85^\circ\text{--}93^\circ\text{E}$).
3. **Non-Mutually Exclusive Depth Slices:** Exactly 169 observations (4.26%) fall into overlapping tolerance brackets of adjacent canonical depth levels, and 1,044 observations fall in bracket gaps. The canonical depth table is an overlapping descriptive slice, not a partition.
4. **Finite Float Trajectories:** While clustered and exact permutation tests demonstrate statistical significance ($p = 0.0017$), float distribution is bounded by the trajectories of 14 physical floats.

---

## 7. Final Recommendation & Freeze Verdict

**GATE A4.3 STATUS: PASS WITH LIMITATION**

All code, metrics, prediction NetCDF files, clustered inference artifacts, depth-bin properties, and technical reports are fully reconciled, reproducible, and internally consistent.

**Gate A4.3 is declared officially and permanently FROZEN.**
