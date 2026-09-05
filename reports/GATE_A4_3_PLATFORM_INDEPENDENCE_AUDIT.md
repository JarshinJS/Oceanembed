# GATE A4.3 — PLATFORM INDEPENDENCE SENSITIVITY AUDIT REPORT

**Audit Date:** September 2026  
**Status:** PASS WITH LIMITATION  
**Project:** OceanEmbed (SIH26066)  
**Evaluated Model:** Frozen OceanEmbed v1-Local (Champion Architecture from Gate A4.1)  
**Evaluated Artifacts:**
- Profile Metrics: `Dataset/gate_a4_3/metrics/argo_profile_metrics.csv`
- Clustered Inference Artifacts: `Dataset/gate_a4_3/metrics/argo_platform_clustered_inference.csv` & `.json`
- Collocated Predictions NetCDF: `Dataset/gate_a4_3/predictions/argo_collocated_predictions.nc`

---

## 1. Executive Summary & Audit Context

The independent observational validation of OceanEmbed v1-Local in Gate A4.3 contains **34 collocated vertical profiles** sampled from autonomous profiling floats drifting in the Bay of Bengal between 2020-03-16 and 2020-03-31.

### The Statistical Independence Issue:
The 34 collocated profiles are **nested within 14 unique WMO float platforms**. Because multiple profiles were collected by the same physical drifting float separated by ~5 to 10 days, successive profiles from a single float are not statistically independent observations.

Treating all 34 profiles as mutually independent could potentially underestimate sampling variance due to intra-platform spatial and instrumental autocorrelation. 

To resolve this issue, this sensitivity audit performs:
1. **Platform-Clustered Bootstrap Resampling ($B = 10,000$):** Resampling at the platform cluster level to account for intra-float correlation.
2. **Conservative Platform-Level Paired Analysis ($N = 14$):** Collapsing all soundings to platform-level mean differences and computing exact permutation and non-parametric tests.
3. **Preservation of Baseline Profile Descriptive Metrics:** Maintaining the primary $N = 34$ benchmark metrics without disruption.

---

## 2. Preserved Primary Profile-Level Results ($N = 34$)

The descriptive profile-level results established in Gate A4.3 are preserved:

$$\Delta_i = \text{OceanEmbed\_RMSE}_i - \text{Climatology\_RMSE}_i \quad (i = 1, \dots, 34)$$

- **Total Collocated Profiles:** $N = 34$ (spanning 14 WMO platforms, 3,966 paired depth soundings)
- **Mean Paired RMSE Difference ($\bar{\Delta}$):** **`-0.2231 °C`**
- **Standard Deviation of Difference:** `0.2229 °C`
- **Median Paired Difference:** **`-0.1898 °C`**
- **Profile Win Rate:** **29 of 34 profiles (85.29%)** exhibit lower RMSE for OceanEmbed v1-Local than climatology
- **Unclustered 95% Bootstrap CI ($B = 10,000$):** `[-0.2972, -0.1499] °C`
- **Permutation Test ($M = 100,000$):** **`p < 1e-5`** (0 of 100,000 permutations as extreme as observed; with $+1$ continuity correction: $p = \frac{0 + 1}{100000 + 1} \approx 9.999 \times 10^{-6}$)
- **Parametric Paired $t$-test:** $t = -5.8358, p = 1.565 \times 10^{-6}$
- **Wilcoxon Signed-Rank Test:** $W = 40.0, p = 1.009 \times 10^{-6}$

---

## 3. Platform-Clustered Bootstrap Sensitivity Analysis

### Resampling Methodology:
- **Cluster Variable:** WMO `platform_number` ($K = 14$ clusters).
- **Procedure:** In each bootstrap iteration $b \in \{1, \dots, 10000\}$, sample 14 platforms with replacement from the 14 available platforms. When a platform is drawn, **all of its collocated profiles are included** in the bootstrap sample.
- **Statistic:** Mean paired profile RMSE difference $\bar{\Delta}^{*(b)}$ across all profiles in the clustered sample.
- **Replicates:** $B = 10,000$ (seed = 42).

### Clustered Bootstrap Results:

| Metric | Unclustered Profile Bootstrap | Platform-Clustered Bootstrap | Difference / Impact |
|---|---|---|---|
| **Sampling Unit** | 34 Individual Profiles | 14 Platform Clusters | Accounts for intra-float correlation |
| **Bootstrap Mean ($\bar{\Delta}^*$)** | `-0.2232 °C` | **`-0.2236 °C`** | Virtually identical central tendency |
| **95% Bootstrap CI** | `[-0.2972, -0.1499] °C` | **`[-0.3457, -0.1117] °C`** | 61% wider CI reflecting cluster variance |
| **Probability $\bar{\Delta}^* < 0$** | `1.000000` | **`1.000000`** | 100% of resamples favor model |
| **Bootstrap Samples $\ge 0$** | 0 / 10,000 | **0 / 10,000** | Zero samples crossed zero |

**Finding:** While accounting for intra-platform clustering broadens the uncertainty interval from `[-0.2972, -0.1499] °C` to `[-0.3457, -0.1117] °C`, the entire 95% confidence interval remains **strictly negative and separated from zero**. Clustered inference completely confirms profile-level superiority.

---

## 4. Conservative Platform-Level Paired Analysis ($N = 14$)

To eliminate any potential inflation of degrees of freedom, the paired difference was aggregated to the platform level by computing the mean error difference across all profiles for each physical float:

$$\Delta_{\text{plat}, k} = \frac{1}{m_k} \sum_{j=1}^{m_k} \Delta_{k, j} \quad (k = 1, \dots, 14)$$

### Platform-Level Results:
- **Effective Sample Size:** $N = 14$ unique WMO float platforms, treated as conservative cluster units.
- **Mean Platform-Level Difference:** **`-0.2140 °C`**
- **Median Platform-Level Difference:** **`-0.2006 °C`**
- **Standard Deviation:** `0.2014 °C`
- **Platform Win Count:** **11 of 14 platforms (78.57%)** favor OceanEmbed v1-Local over Climatology.
- **Exact Platform-Level Sign-Flip Permutation Test ($2^{14} = 16,384$ combinations):**
  - Two-sided exact $p$-value: **`p = 0.001709`**
  - One-sided exact $p$-value: **`p = 0.000854`**
- **Platform-Level One-Sample $t$-test ($N = 14$):** $t = -3.9726$, **`p = 0.001592`**.
- **Platform-Level Wilcoxon Signed-Rank Test ($N = 14$):** $W = 8.0$, **`p = 0.003052`**.

Even under this conservative platform-level aggregation where each float has exactly one vote, OceanEmbed achieves statistically significant superior skill ($p < 0.002$).

---

## 5. Complete WMO Platform Breakdown Table

Verification of all 34 profiles correctly assigned across the 14 WMO platforms:

| WMO Platform | Profile Count ($m_k$) | Float Mean $\Delta$ (°C) | Float Median $\Delta$ (°C) | Model RMSE (°C) | Clim RMSE (°C) | Model Favored? |
|---|---|---|---|---|---|---|
| **2902230** | 3 | **`-0.5429`** | `-0.5523` | 0.6558 | 1.1988 | **YES** |
| **2902233** | 3 | **`-0.1437`** | `-0.1146` | 0.6383 | 0.7820 | **YES** |
| **2902235** | 1 | **`-0.0097`** | `-0.0097` | 1.6091 | 1.6188 | **YES** |
| **2902236** | 3 | `+0.0555` | `+0.0037` | 1.0552 | 0.9997 | NO |
| **2902264** | 3 | **`-0.4451`** | `-0.4748` | 0.7064 | 1.1516 | **YES** |
| **2902278** | 3 | **`-0.1485`** | `-0.2023` | 0.6104 | 0.7589 | **YES** |
| **2902279** | 3 | **`-0.1211`** | `-0.1403` | 0.9503 | 1.0714 | **YES** |
| **2902280** | 4 | `+0.0032` | `-0.0158` | 0.7393 | 0.7361 | NO |
| **2902283** | 3 | **`-0.5578`** | `-0.5505` | 0.9655 | 1.5233 | **YES** |
| **2902596** | 1 | **`-0.3095`** | `-0.3095` | 0.4517 | 0.7612 | **YES** |
| **2902766** | 2 | **`-0.2675`** | `-0.2675` | 0.4295 | 0.6969 | **YES** |
| **2902768** | 2 | **`-0.2702`** | `-0.2702` | 0.6822 | 0.9524 | **YES** |
| **2902769** | 2 | **`-0.2526`** | `-0.2526` | 0.4737 | 0.7263 | **YES** |
| **2902772** | 1 | `+0.0144` | `+0.0144` | 0.9860 | 0.9716 | NO |
| **TOTAL / AVG** | **34** | **`-0.2140`** | **`-0.2006`** | **0.7824** | **0.9964** | **11 / 14 (78.6%)** |

---

## 6. Synthesis of Statistical Inferences

| Analysis Level | Nominal $N$ | Test Type | Observed Statistic | 95% Confidence Interval | $p$-value | Conclusion |
|---|---|---|---|---|---|---|
| **Profile-Level** | 34 profiles | Student's $t$ | $\bar{\Delta} = -0.2231\text{ }^\circ\text{C}$ | `[-0.2972, -0.1499]` (boot) | $1.56 \times 10^{-6}$ | Strongly Significant |
| **Profile-Level** | 34 profiles | Wilcoxon | $W = 40.0$ | — | $1.01 \times 10^{-6}$ | Strongly Significant |
| **Profile-Level** | 34 profiles | Permutation ($100\text{k}$) | $\bar{\Delta} = -0.2231\text{ }^\circ\text{C}$ | — | **`p < 1e-5`** | Strongly Significant |
| **Platform-Clustered** | 14 clusters (34 prof) | Clustered Boot ($10\text{k}$) | $\bar{\Delta}^* = -0.2236\text{ }^\circ\text{C}$ | **`[-0.3457, -0.1117]`** | $P(\bar{\Delta}^* \ge 0) = 0$ | Robust Superiority |
| **Platform-Level** | 14 platforms | Student's $t$ | $\bar{\Delta}_{\text{plat}} = -0.2140\text{ }^\circ\text{C}$ | `[-0.3303, -0.0977]` (param) | **`p = 0.001592`** | Statistically Significant |
| **Platform-Level** | 14 platforms | Wilcoxon | $W = 8.0$ | — | **`p = 0.003052`** | Statistically Significant |
| **Platform-Level** | 14 platforms | Exact Permutation ($16\text{k}$) | $\bar{\Delta}_{\text{plat}} = -0.2140\text{ }^\circ\text{C}$ | — | **`p = 0.001709`** | Statistically Significant |

---

## 7. Statistical Concerns & Limitations

1. **Finite Platform Sample ($N = 14$):** While 14 platforms are sufficient for exact permutation testing ($p = 0.0017$), float distribution is non-uniform across the basin.
2. **Temporal Revisit Clustering:** Platforms `2902230`, `2902233`, `2902236`, `2902264`, `2902278`, `2902279`, `2902280`, `2902283` each contributed 3 to 4 profiles over the 16-day window. Clustered bootstrap explicitly accommodates this, but regional generalization to other seasons or oceans cannot be asserted.
3. **Mechanistic Claims Removed:** References to inferred "isopycnal displacement" have been removed as unmeasured internal physical mechanisms; the empirical finding is strictly defined as subsurface temperature reconstruction.

---

## 8. Final Audit Verdict

**VERDICT: PASS WITH LIMITATION**

### Preferred Scientific Conclusion:
> *"These results support the hypothesis that multimodal surface observations contain information useful for reconstructing subsurface temperature structure, including the thermocline, within the evaluated regional and temporal validation window."*
