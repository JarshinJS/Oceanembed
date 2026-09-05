# GATE A1.2 — GLORYS 1000 m Target Depth Resolution Report

**Document Version:** 1.0  
**Date:** 2026-09-04  
**Project:** OceanEmbed MVP (SIH26066)  
**Status:** **`READY_FOR_CORRECTED_DOWNLOAD`**  

---

## Executive Summary

During Gate A1 verification, inspection of the downloaded file `Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn.nc` revealed 35 vertical levels spanning **0.4940 m to 902.3393 m**. Because SIH26066 requires canonical target depths up to **1000 m** (`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m`), producing 1000 m from this file would necessitate unphysical extrapolation beyond the deepest available level (902.3393 m).

In accordance with strict Gate A1.2 rules:
1. **Preprocessing was NOT modified.** No extrapolation or fabricated data was created.
2. The existing file `Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn.nc` is preserved intact and cataloged as **`A1-OCEAN-ATTEMPT-1`** (Status: *BLOCKED — deepest native level 902.3393 m*).
3. Live dataset metadata from the Copernicus Marine catalogue (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, version `202311`, part `default`) was audited and confirmed to contain all **50 native NEMO vertical levels** down to **5727.92 m**.
4. The truncation at 902.3393 m occurred because the Copernicus Marine API subsetting service uses the default coordinate selection filter `inside` ($z \le z_{\max}$). When $z_{\max} = 1000.0\text{ m}$ was requested, the next deeper native level ($1062.4399\text{ m}$) was excluded because $1062.44 > 1000.0$.
5. A dry-run query with `--maximum-depth 1100.0` was executed and successfully validated. It retrieves **36 native vertical levels** ($0.4940\text{ m} \to 1062.4399\text{ m}$), cleanly bracketing the required 1000 m target depth between Level 34 ($902.3393\text{ m}$) and Level 35 ($1062.4399\text{ m}$).

---

## 1. Inventory & Preservation of Attempt 1

| Property | Value |
| :--- | :--- |
| **Internal Designation** | `A1-OCEAN-ATTEMPT-1` |
| **File Path** | `Dataset/gate_a1_pilot/glorys_pilot_30d_fullcolumn.nc` |
| **File Size** | 59,943,126 bytes (~57.17 MB) |
| **Dimensions** | `time=30`, `depth=35`, `latitude=73`, `longitude=97` |
| **Native Depth Range** | `0.494025 m` to `902.3393 m` |
| **Preservation Status** | **PRESERVED INTACT (NOT DELETED)** |
| **Audit Status** | **`BLOCKED — deepest native level 902.3393 m`** |

---

## 2. Root Cause Analysis: Why the Current File Stops at 902.3393 m

The Copernicus Marine CLI subsetting tool (`copernicusmarine subset`) implements a coordinate filtering parameter:

```
--coordinates-selection-method [inside|strict-inside|nearest|outside]
```

The default option is **`inside`**, which specifies:
> *"the selection retrieved will be inside the requested range."*

When a request specifies `--minimum-depth 0.0 --maximum-depth 1000.0` (or short form `-z 0 -Z 1000`), the API evaluates all discrete native grid depth coordinates $z_k$ against the condition:
$$0.0 \le z_k \le 1000.0$$

In the native GLORYS12V1 vertical grid:
- **Level index 34:** $z_{34} = 902.339294\text{ m} \le 1000.0\text{ m}$ $\rightarrow$ **INCLUDED**
- **Level index 35:** $z_{35} = 1062.439941\text{ m} > 1000.0\text{ m}$ $\rightarrow$ **EXCLUDED**

Because $1062.4399\text{ m}$ strictly exceeds $1000.0\text{ m}$, the subsetting service truncated the returned coordinate slice at index 34. The returned NetCDF file consequently contained exactly **35 vertical levels** (indices $0 \dots 34$), leaving the bottom coordinate at $902.3393\text{ m}$.

---

## 3. Native GLORYS12V1 Vertical Depth Levels

Querying the Copernicus Marine metadata catalogue directly via `copernicusmarine.describe(dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m')` confirmed that the product contains the full standard **50 vertical levels** of the NEMO ocean model.

Below is the complete table of native depth coordinates in the dataset:

| Level Index | Native Depth (m) | Status in Attempt 1 | Status in Corrected Request |
| :---: | :---: | :---: | :---: |
| 0 | 0.4940 | Included | Included |
| 1 | 1.5414 | Included | Included |
| 2 | 2.6457 | Included | Included |
| 3 | 3.8195 | Included | Included |
| 4 | 5.0782 | Included | Included |
| 5 | 6.4406 | Included | Included |
| 6 | 7.9296 | Included | Included |
| 7 | 9.5730 | Included | Included |
| 8 | 11.4050 | Included | Included |
| 9 | 13.4671 | Included | Included |
| 10 | 15.8101 | Included | Included |
| 11 | 18.4956 | Included | Included |
| 12 | 21.5988 | Included | Included |
| 13 | 25.2114 | Included | Included |
| 14 | 29.4447 | Included | Included |
| 15 | 34.4342 | Included | Included |
| 16 | 40.3441 | Included | Included |
| 17 | 47.3737 | Included | Included |
| 18 | 55.7643 | Included | Included |
| 19 | 65.8073 | Included | Included |
| 20 | 77.8539 | Included | Included |
| 21 | 92.3261 | Included | Included |
| 22 | 109.7293 | Included | Included |
| 23 | 130.6660 | Included | Included |
| 24 | 155.8507 | Included | Included |
| 25 | 186.1256 | Included | Included |
| 26 | 222.4752 | Included | Included |
| 27 | 266.0403 | Included | Included |
| 28 | 318.1274 | Included | Included |
| 29 | 380.2130 | Included | Included |
| 30 | 453.9377 | Included | Included |
| 31 | 541.0889 | Included | Included |
| 32 | 643.5668 | Included | Included |
| 33 | 763.3331 | Included | Included |
| **34** | **902.3393** | **Included (Last level)** | **Included (Lower Bracket)** |
| **---** | **1000.0000** | **MISSING (Extrapolation required)** | **INTERIOR INTERPOLATION TARGET** |
| **35** | **1062.4399** | **Excluded** | **Included (Upper Bracket)** |
| 36 | 1245.2910 | Excluded | Excluded (Unneeded depth) |
| 37 | 1452.2510 | Excluded | Excluded |
| 38 | 1684.2841 | Excluded | Excluded |
| 39 | 1941.8929 | Excluded | Excluded |
| 40 | 2225.0779 | Excluded | Excluded |
| 41 | 2533.3359 | Excluded | Excluded |
| 42 | 2865.7029 | Excluded | Excluded |
| 43 | 3220.8201 | Excluded | Excluded |
| 44 | 3597.0320 | Excluded | Excluded |
| 45 | 3992.4839 | Excluded | Excluded |
| 46 | 4405.2241 | Excluded | Excluded |
| 47 | 4833.2910 | Excluded | Excluded |
| 48 | 5274.7842 | Excluded | Excluded |
| 49 | 5727.9170 | Excluded | Excluded |

---

## 4. Dataset Identifier, Version, and Service Part

The current dataset configuration is entirely correct and does not need to be altered:
- **Product Title:** Global Ocean Physics Reanalysis (`GLOBAL_MULTIYEAR_PHY_001_030`)
- **Dataset ID:** `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Dataset Version:** `202311`
- **Dataset Part:** `default`
- **DOI:** `10.48670/moi-00021`

The product natively provides all depth levels required. Only the subset request boundary needs adjustment to incorporate the level above 1000 m.

---

## 5. Verification of 1000 m Interpolation Mechanics

With native levels 0 through 35 present:

### Bracketing Levels for 1000 m
- **Native level below 1000 m (shallower):** $z_0 = 902.339294\text{ m}$ (Level index 34)
- **Native level above 1000 m (deeper):** $z_1 = 1062.439941\text{ m}$ (Level index 35)

### Linear Interpolation Weights
For any vertical profile $V(z)$ (temperature $\theta$, salinity $S$, horizontal velocity $u$, $v$):
$$\alpha = \frac{z_{\text{target}} - z_0}{z_1 - z_0} = \frac{1000.0 - 902.339294}{1062.439941 - 902.339294} = \frac{97.660706}{160.100647} \approx 0.6099957$$

$$V(1000) = (1 - \alpha) \cdot V(902.3393) + \alpha \cdot V(1062.4399) \approx 0.3900043 \cdot V_{34} + 0.6099957 \cdot V_{35}$$

**Guaranteed property:** Because $902.3393 < 1000.0 < 1062.4399$, $0 < \alpha < 1$. This is a **strict interior linear interpolation**. Zero extrapolation or numerical fabrication is performed.

All other 14 canonical depths (`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700] m`) are likewise strictly bracketed by native levels between index 0 ($0.4940\text{ m}$) and index 33 ($763.3331\text{ m}$).

---

## 6. Corrected Subset Command & Dry-Run Validation

Setting `--maximum-depth 1100.0` (or `--maximum-depth 1065.0`) instructs Copernicus Marine to retrieve all coordinates up to 1100 m, which incorporates native level 35 ($1062.4399\text{ m}$) while safely excluding unnecessary deeper levels ($1245.29\text{ m} \dots 5727.92\text{ m}$).

### Validated Command (PowerShell)

```powershell
copernicusmarine subset `
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m `
  --variable thetao --variable so --variable zos --variable uo --variable vo `
  --start-datetime 2020-01-01T00:00:00 `
  --end-datetime 2020-01-30T23:59:59 `
  --minimum-longitude 85.0 --maximum-longitude 93.0 `
  --minimum-latitude 12.0 --maximum-latitude 18.0 `
  --minimum-depth 0.0 --maximum-depth 1100.0 `
  --output-directory Dataset/gate_a1_pilot `
  --output-filename glorys_pilot_30d_15depths.nc
```

### Dry-Run Output Confirmation (Executed on System)
```json
{
  "file_path": "Dataset\\gate_a1_pilot\\glorys_pilot_30d_15depths.nc",
  "output_directory": "Dataset\\gate_a1_pilot",
  "filename": "glorys_pilot_30d_15depths.nc",
  "file_size": "72.99 MB",
  "variables": ["thetao", "so", "zos", "uo", "vo"],
  "coordinates_extent": [
    {"coordinate_id": "longitude", "minimum": 85.0, "maximum": 93.0, "unit": "degrees_east"},
    {"coordinate_id": "latitude", "minimum": 12.0, "maximum": 18.0, "unit": "degrees_north"},
    {"coordinate_id": "time", "minimum": "2020-01-01T00:00:00+00:00", "maximum": "2020-01-30T00:00:00+00:00", "unit": "iso8601"},
    {"coordinate_id": "depth", "minimum": 0.49402499198913574, "maximum": 1062.43994140625, "unit": "m"}
  ],
  "status": "001",
  "message": "The request was run with the dry-run option. No data was downloaded."
}
```

---

## 7. Expected File Attributes & Resource Profile

| Attribute | Expected Value |
| :--- | :--- |
| **Output File** | `Dataset/gate_a1_pilot/glorys_pilot_30d_15depths.nc` |
| **Estimated File Size** | **72.99 MB** (modest, well within standard storage/RAM budgets) |
| **Time Steps** | 30 daily means (2020-01-01 to 2020-01-30) |
| **Depth Levels** | **36 native levels** ($0.4940\text{ m} \to 1062.4399\text{ m}$) |
| **Latitude Points** | 73 points ($12.0^\circ\text{N} \to 18.0^\circ\text{N}$, $\Delta = 0.0833^\circ$) |
| **Longitude Points** | 97 points ($85.0^\circ\text{E} \to 93.0^\circ\text{E}$, $\Delta = 0.0833^\circ$) |
| **Target Depths Supported** | All 15 canonical depths: $[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$ |

---

## 8. Final Status and Recommendation

### Recommendation
The corrected command resolves the depth truncation issue completely by requesting depth up to 1100.0 m, which includes native level 35 (1062.4399 m). This satisfies the strict scientific requirement that 1000 m be computed strictly via interior interpolation without extrapolation. 

### Final Status:
# **`READY_FOR_CORRECTED_DOWNLOAD`**

*Execution is halted pending user approval.*
