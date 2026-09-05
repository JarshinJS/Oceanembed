# GATE A1.3 — PS-Aligned 15-Depth Preprocessing Verification Report

**Document Version:** 1.0  
**Date:** 2026-09-04  
**Project:** OceanEmbed MVP (SIH26066)  
**Status:** **`GATE A1.3 PASSED — 15-DEPTH PREPROCESSING VERIFIED`**  

---

## 1. Executive Summary

Following the acquisition and inspection of the corrected Copernicus GLORYS reanalysis pilot NetCDF (`Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn_1100m.nc`), the official Problem Statement (SIH26066) **15-Depth Vertical Preprocessing Stage** has been implemented and verified.

All native 36 vertical levels ($0.4940\text{ m} \to 1062.4399\text{ m}$) were transformed into the exact authoritative 15 target depths:
$$\mathbf{Z}_{\text{target}} = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$$

### Key Compliance Points:
1. **0 m Target:** Mapped directly to the uppermost valid native surface level ($z_0 = 0.4940\text{ m}$). Blind upward extrapolation to 0.0 m is **strictly prevented** ($\alpha = 0.0$).
2. **5 m to 700 m Targets:** Vertically interpolated strictly between the immediate lower and upper native levels.
3. **1000 m Target:** Strictly bracketed and linearly interpolated between native levels 34 ($902.3393\text{ m}$) and 35 ($1062.4399\text{ m}$) with $\alpha \approx 0.6100$. Extrapolation is **zero**.
4. **NaN & Mask Preservation:** Missing values and land pixels in either bracketing level propagate as NaNs, and an explicit binary ocean validity mask (`mask_*`) is produced for every variable and depth.
5. **No 100% Coverage Claims:** Valid ocean coverage reflects true bathymetry in the Bay of Bengal pilot box, gently decreasing from **99.34%** at the surface to **96.62%** at 1000 m.
6. **Integrity & Constraints:** No model training, no architecture modification, and no atmospheric wind channels were introduced.

---

## 2. Dataset Dimensions & Files

| Dataset Stage | File Path | File Size | Dimensions |
| :--- | :--- | :--- | :--- |
| **Input (Corrected Pilot)** | `Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn_1100m.nc` | 58.787 MB | `time: 30`, `depth: 36`, `latitude: 73`, `longitude: 97` |
| **Output (Preprocessed)** | `Dataset/gate_a1_pilot/glorys_pilot_30d_15depths_preprocessed.nc` | 61.823 MB | `time: 30`, `depth: 15`, `latitude: 73`, `longitude: 97` |
| **Metadata Record** | `outputs/gate_a1_3_preprocessing_metadata.json` | 21.274 KB | Full weight & depth metadata record |

---

## 3. Depth Coordinate Verification

### 3.1. Native Depth List (36 Levels)
```
Level  0:    0.4940 m     Level 12:   21.5988 m     Level 24:  155.8507 m
Level  1:    1.5414 m     Level 13:   25.2114 m     Level 25:  186.1256 m
Level  2:    2.6457 m     Level 14:   29.4447 m     Level 26:  222.4752 m
Level  3:    3.8195 m     Level 15:   34.4342 m     Level 27:  266.0403 m
Level  4:    5.0782 m     Level 16:   40.3441 m     Level 28:  318.1274 m
Level  5:    6.4406 m     Level 17:   47.3737 m     Level 29:  380.2130 m
Level  6:    7.9296 m     Level 18:   55.7643 m     Level 30:  453.9377 m
Level  7:    9.5730 m     Level 19:   65.8073 m     Level 31:  541.0889 m
Level  8:   11.4050 m     Level 20:   77.8539 m     Level 32:  643.5668 m
Level  9:   13.4671 m     Level 21:   92.3261 m     Level 33:  763.3331 m
Level 10:   15.8101 m     Level 22:  109.7293 m     Level 34:  902.3393 m
Level 11:   18.4956 m     Level 23:  130.6660 m     Level 35: 1062.4399 m
```

### 3.2. Authoritative Target Depth List (15 Levels)
```
[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
```

---

## 4. Interpolation Mechanics & Bracketing Weights

For target depths $z > 0$, the bracketing native levels $z_0 \le z \le z_1$ satisfy $z_0 < z_1$. The linear interpolation weight is:
$$\alpha = \frac{z - z_0}{z_1 - z_0}$$
$$V(z) = (1 - \alpha) \cdot V(z_0) + \alpha \cdot V(z_1)$$

For target depth $z = 0$, $V(0) = V(z_0 = 0.4940\text{ m})$ directly ($\alpha = 0$).

| Target Depth | Lower Level Index ($z_0$) | Lower Depth ($z_0$) | Upper Level Index ($z_1$) | Upper Depth ($z_1$) | Weight $\alpha$ | Transformation Method |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0 m** | 0 | 0.4940 m | 0 | 0.4940 m | 0.000000 | `nearest_surface_level` (No upward extrapolation) |
| **5 m** | 3 | 3.8195 m | 4 | 5.0782 m | 0.937855 | `bracketed_linear_interpolation` |
| **10 m** | 7 | 9.5730 m | 8 | 11.4050 m | 0.233080 | `bracketed_linear_interpolation` |
| **20 m** | 11 | 18.4956 m | 12 | 21.5988 m | 0.484795 | `bracketed_linear_interpolation` |
| **30 m** | 14 | 29.4447 m | 15 | 34.4342 m | 0.111289 | `bracketed_linear_interpolation` |
| **50 m** | 17 | 47.3737 m | 18 | 55.7643 m | 0.312999 | `bracketed_linear_interpolation` |
| **75 m** | 19 | 65.8073 m | 20 | 77.8539 m | 0.763076 | `bracketed_linear_interpolation` |
| **100 m** | 21 | 92.3261 m | 22 | 109.7293 m | 0.440947 | `bracketed_linear_interpolation` |
| **125 m** | 22 | 109.7293 m | 23 | 130.6660 m | 0.729396 | `bracketed_linear_interpolation` |
| **150 m** | 23 | 130.6660 m | 24 | 155.8507 m | 0.767708 | `bracketed_linear_interpolation` |
| **200 m** | 25 | 186.1256 m | 26 | 222.4752 m | 0.381695 | `bracketed_linear_interpolation` |
| **300 m** | 27 | 266.0403 m | 28 | 318.1274 m | 0.652003 | `bracketed_linear_interpolation` |
| **500 m** | 30 | 453.9377 m | 31 | 541.0889 m | 0.528532 | `bracketed_linear_interpolation` |
| **700 m** | 32 | 643.5668 m | 33 | 763.3331 m | 0.471194 | `bracketed_linear_interpolation` |
| **1000 m** | 34 | 902.3393 m | 35 | 1062.4399 m | 0.609996 | `bracketed_linear_interpolation` |

### 1000 m Bracketing Verification
- Lower native level: $z_{34} = 902.339294\text{ m}$
- Upper native level: $z_{35} = 1062.439941\text{ m}$
- **Verification:** $902.3393\text{ m} < 1000.0\text{ m} < 1062.4399\text{ m}$ $\rightarrow$ **TRUE**
- **Interior weight:** $\alpha = \frac{1000 - 902.339294}{1062.439941 - 902.339294} = 0.6099957 \in (0, 1)$ $\rightarrow$ **TRUE**

---

## 5. Actual Ocean Coverage & Missing-Value Statistics

Per time step, the spatial grid has $73 \times 97 = 7,081$ horizontal grid points.  
Across the 30 daily time steps, each depth level has $30 \times 7,081 = 212,430$ total space-time pixels.

### Detailed Per-Depth Physical Statistics for `thetao` (Potential Temperature, °C):

| Target Depth | Valid Points | NaN Points | Total Points | Ocean Coverage (%) | Min (°C) | Max (°C) | Mean (°C) | Physical Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0 m** | 211,020 | 1,410 | 212,430 | **99.3363%** | 23.2142 | 28.9514 | 26.9070 | Surface Mixed Layer |
| **5 m** | 211,020 | 1,410 | 212,430 | **99.3363%** | 23.3765 | 29.0120 | 26.8741 | Upper Mixed Layer |
| **10 m** | 210,840 | 1,590 | 212,430 | **99.2515%** | 23.7131 | 29.1911 | 26.8984 | Mixed Layer Base |
| **20 m** | 210,420 | 2,010 | 212,430 | **99.0538%** | 24.0368 | 29.4062 | 27.1556 | Sub-surface Warm Core |
| **30 m** | 209,880 | 2,550 | 212,430 | **98.7996%** | 24.8619 | 29.5098 | 27.4731 | Barrier Layer Top |
| **50 m** | 209,010 | 3,420 | 212,430 | **98.3901%** | 25.6328 | 29.0872 | 27.6549 | Upper Thermocline |
| **75 m** | 208,380 | 4,050 | 212,430 | **98.0935%** | 21.1430 | 27.9835 | 25.8740 | Core Thermocline |
| **100 m** | 207,570 | 4,860 | 212,430 | **97.7122%** | 18.3138 | 25.7345 | 22.3764 | Steep Gradient Zone |
| **125 m** | 207,420 | 5,010 | 212,430 | **97.6416%** | 15.5937 | 22.5092 | 19.2235 | Deep Thermocline |
| **150 m** | 207,420 | 5,010 | 212,430 | **97.6416%** | 14.1954 | 19.8093 | 16.7679 | Lower Thermocline |
| **200 m** | 207,240 | 5,190 | 212,430 | **97.5568%** | 12.3914 | 15.9721 | 13.9470 | Intermediate Water |
| **300 m** | 207,120 | 5,310 | 212,430 | **97.5004%** | 10.7952 | 12.5298 | 11.5602 | Intermediate Water |
| **500 m** | 206,760 | 5,670 | 212,430 | **97.3309%** | 9.4012 | 10.5324 | 9.8927 | Deep Water |
| **700 m** | 206,250 | 6,180 | 212,430 | **97.0908%** | 7.8437 | 9.2162 | 8.4234 | Deep Water |
| **1000 m** | 205,260 | 7,170 | 212,430 | **96.6248%** | 5.8826 | 7.2236 | 6.6332 | Abyssal Transition |

### Key Physical Observations:
1. **Physical Ocean Stratification:** The mean temperature profile accurately exhibits classic tropical Indian Ocean stratification: a surface mixed layer at $\sim 26.9^\circ\text{C}$, the core thermocline dropping sharply between $50\text{ m}$ ($27.65^\circ\text{C}$) and $150\text{ m}$ ($16.77^\circ\text{C}$), cooling gradually to $6.63^\circ\text{C}$ at $1000\text{ m}$.
2. **Bathymetric Dropout:** Ocean coverage decreases monotonically from **$99.34\%$** at the surface to **$96.62\%$** at $1000\text{ m}$. This gradual loss of $\sim 2.72\%$ coverage correctly reflects the shoaling of the continental shelf and bathymetric topography along the eastern/northern margins of the Bay of Bengal pilot domain ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$).
3. **No Unphysical Fabrication:** No imputation, spatial filling, or extrapolation was applied. Pixels where the seafloor is shallower than target depth remain exact NaNs with a mask value of $0$.

---

## 6. Automated Acceptance & Unit Test Results

The unit test suite was implemented in [test_vertical_interpolation.py](file:///c:/Users/Jarsh/Downloads/oceanembed_mvp/oceanembed/tests/test_vertical_interpolation.py) and executed alongside the full test harness:

```bash
python -m pytest tests/
```

### Test Suite Execution Output:
```
tests/test_a1_acceptance.py ..............                               [ 66%]
tests/test_smoke.py .                                                    [ 71%]
tests/test_vertical_interpolation.py ......                              [100%]

======================== 21 passed, 1 warning in 7.79s ========================
```

### Specific Verifications in `test_vertical_interpolation.py`:
- `test_01_exact_target_depth_ordering`: Passed. Canonical 15 depths strictly verified.
- `test_02_fifteen_output_depths`: Passed. Dimension size verified across 4D fields.
- `test_03_1000m_interpolation_bracket`: Passed. $902.3393\text{ m} < 1000\text{ m} < 1062.4399\text{ m}$ condition and synthetic linear temperature reconstruction tested.
- `test_04_0m_nearest_level_handling`: Passed. Exact mapping to surface native level without upward extrapolation verified.
- `test_05_nan_and_mask_preservation`: Passed. Controlled pixel test with land columns, surface NaNs, and bathymetric cutoffs verified.
- `test_06_real_pilot_nc_preprocessing`: Passed. Real NetCDF dimensions, depths, and coverage thresholds validated.

---

## 7. Status & Next Step

The 15-depth vertical preprocessing stage is fully verified, mathematically sound, grounded in actual physical data, and backed by automated tests.

### Current Status:
# **`GATE A1.3 PASSED — PREPROCESSING VERIFIED`**

Execution is stopped as instructed. Awaiting user review and authorization to proceed to subsequent stages.
