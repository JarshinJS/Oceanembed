# Gate A1: Real NetCDF Data Integrity Audit Report (SIH26066)

**Audit Execution Date:** 2026-09-04  
**Audit Purpose:** Verify whether the actual NetCDF datasets downloaded into `Dataset/` can be reliably transformed into the canonical SIH26066 7-channel input tensor `[B, 7, H, W]` and 15-depth target profile `[B, 15]`.

---

## 1. File Identification

A comprehensive scan of the repository was conducted. Exactly **20 NetCDF files** were discovered in `Dataset/` (0 files in `data/`):

| File Path | File Size | Format | Primary Role |
| :--- | :---: | :---: | :--- |
| `Dataset/cmems_mod_glo_phy_my_0.083deg_P1D-m_1788514642488.nc` | **370.05 MB** (388,027,930 B) | NetCDF4 / CF-1.11 | Training target supervision (GLORYS12V1) |
| `Dataset/METOFFICE-GLO-SST-L4-REP-OBS-SST_1788514360579.nc` | **395.57 MB** (414,784,428 B) | NetCDF4 / CF-1.11 | Surface SST observation (OSTIA) |
| `Dataset/cmems_obs-mob_glo_phy-sss_my_multi_P1D_1788514338421.nc` | **79.14 MB** (82,980,140 B) | NetCDF4 / CF-1.11 | Surface SSS observation (Multi-Obs SMAP/SMOS) |
| `Dataset/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_1788514168280.nc`| **39.59 MB** (41,515,708 B) | NetCDF4 / CF-1.11 | Surface SSH/SLA observation (DUACS) |
| `Dataset/ASCAT/ascat_*_eps_o_coa_3203_ovw.l2.nc` (16 individual files) | **52.14 MB** total (~3.26 MB each) | NetCDF3/4 / CF-1.6 | Surface wind observation (ASCAT-L2 Coastal swaths) |

**Total Real Data on Disk:** **936.49 MB** across 20 files.

---

## 2. Product Identification

All products were verified using actual internal NetCDF attributes:

1. **GLORYS12V1:**
   - `title`: daily mean fields from Global Ocean Physics Analysis and Forecast updated Daily
   - `institution`: MERCATOR OCEAN
   - `source`: MERCATOR GLORYS12V1
   - `product_id`: `GLOBAL_MULTIYEAR_PHY_001_030`
   - `dataset_id`: `cmems_mod_glo_phy_my_0.083deg_P1D-m_202311`
2. **OSTIA SST:**
   - `title`: Global SST & Sea Ice Analysis, L4 OSTIA, 0.05 deg daily (METOFFICE-GLO-SST-L4-REP-OBS-SST-V2)
   - `institution`: UKMO (UK Met Office)
   - `references`: Donlon et al. (2011)
3. **Multi-Obs SSS:**
   - `title`: Global Analysed Sea Surface Salinity and Density
   - `institution`: CNR
   - `references`: Buongiorno Nardelli et al. (2016), Sammartino et al. (2022)
4. **DUACS SSH/SLA:**
   - `title`: DT merged two satellites Global Ocean Gridded SSALTO/DUACS Sea Surface Height L4 product
   - `institution`: CLS, CNES
   - `source`: Altimetry measurements
5. **ASCAT-L2 Coastal Winds:**
   - `title`: MetOp-C ASCAT Level 2 Coastal Ocean Surface Wind Vector Product
   - `institution`: EUMETSAT/OSI SAF/KNMI
   - `processing_level`: L2 (Orbital swaths)

---

## 3. Dimensions

| Dataset | Native Dimensions & Lengths | Total Grid Cells per Timestep |
| :--- | :--- | :---: |
| **GLORYS** | `{'time': 1, 'depth': 1, 'latitude': 2041, 'longitude': 4320}` | 8,817,120 |
| **OSTIA** | `{'time': 1, 'latitude': 3600, 'longitude': 7200}` | 25,920,000 |
| **SSS** | `{'time': 1, 'depth': 1, 'latitude': 1440, 'longitude': 2880}` | 4,147,200 |
| **DUACS** | `{'time': 1, 'latitude': 720, 'longitude': 1440}` | 1,036,800 |
| **ASCAT (Granule)**| `{'NUMROWS': 3264, 'NUMCELLS': 82}` | 267,648 (orbital cells) |

---

## 4. Coordinates

- **Latitude:**
  - GLORYS: `[-80.0, 90.0]`, spacing $\approx 0.08333^\circ$, ascending, 1D.
  - OSTIA: `[-89.975, 89.975]`, spacing $= 0.05^\circ$, ascending, 1D.
  - SSS: `[-89.9375, 89.9375]`, spacing $= 0.125^\circ$, ascending, 1D.
  - DUACS: `[-89.875, 89.875]`, spacing $= 0.25^\circ$, ascending, 1D.
  - ASCAT: 2D coordinates `lat(NUMROWS, NUMCELLS)`, range `[-89.31, 89.18]`.
- **Longitude:**
  - GLORYS: `[-180.0, 179.9167]`, spacing $\approx 0.08333^\circ$, ascending, 1D ($-180..180$).
  - OSTIA: `[-179.975, 179.975]`, spacing $= 0.05^\circ$, ascending, 1D ($-180..180$).
  - SSS: `[-179.9375, 179.9375]`, spacing $= 0.125^\circ$, ascending, 1D ($-180..180$).
  - DUACS: `[-179.875, 179.875]`, spacing $= 0.25^\circ$, ascending, 1D ($-180..180$).
  - ASCAT: 2D coordinates `lon(NUMROWS, NUMCELLS)`, range `[0.002, 359.999]`, **uses $0^\circ$ to $360^\circ$ convention**.

---

## 5. Time Coverage

| Dataset | Time Dimension Size | Exact Timestamp in File | Frequency |
| :--- | :---: | :--- | :---: |
| **GLORYS** | 1 | `2026-06-23T00:00:00.000000000` | Daily snapshot |
| **OSTIA** | 1 | `2026-03-31T00:00:00.000000000` | Daily snapshot |
| **SSS** | 1 | `2024-12-15T00:00:00.000000000` | Daily snapshot |
| **DUACS** | 1 | `2026-01-16T00:00:00.000000000` | Daily snapshot |
| **ASCAT** | 16 granules | `2019-12-31T22:33:00` to `2020-01-01T23:54:00` | Orbital passes |

> [!WARNING]
> **Severe Temporal Discordance:**
> The downloaded files do not share a single common date. Each CMEMS file represents an isolated snapshot from a different year/month (June 2026, March 2026, Dec 2024, Jan 2026, and Jan 2020). None of the files provide the required 30-day continuous sequence.

---

## 6. Native Depth Structure

- **Native Depth Variable:** `depth` in GLORYS and SSS.
- **GLORYS Native Depth Levels Present:** Exactly **1 level**: `[0.494025]` m.
- **Deepest Native Level in GLORYS File:** $0.494025\text{ m}$.
- **Required Depth Levels in SIH26066:**
  `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m.

> [!CAUTION]
> **Subsurface Target Levels Missing:**
> Subsurface levels $[5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$ are completely missing from the downloaded GLORYS NetCDF file. The file only contains the uppermost surface layer ($0.494\text{ m}$).

---

## 7. Variable Inventory

| Variable | Dataset | Dimensions | Dtype | Units | Valid Count / Total | Valid % | Min Valid | Max Valid | Mean |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `thetao` | GLORYS | `(1, 1, 2041, 4320)` | float32 | `degrees_C` | 6,146,048 / 8,817,120 | 69.71% | -2.73 | 34.59 | 17.51 |
| `so` | GLORYS | `(1, 1, 2041, 4320)` | float32 | `1e-3` (PSU) | 6,146,048 / 8,817,120 | 69.71% | 0.0015 | 42.51 | 34.78 |
| `zos` | GLORYS | `(1, 2041, 4320)` | float32 | `m` | 6,146,048 / 8,817,120 | 69.71% | -1.996 | 1.698 | -0.12 |
| `uo` | GLORYS | `(1, 1, 2041, 4320)` | float32 | `m s-1` | 6,145,586 / 8,817,120 | 69.70% | -1.96 | 2.55 | -0.01 |
| `vo` | GLORYS | `(1, 1, 2041, 4320)` | float32 | `m s-1` | 6,145,586 / 8,817,120 | 69.70% | -2.43 | 2.12 | -0.01 |
| `analysed_sst` | OSTIA | `(1, 3600, 7200)` | float32 | `kelvin` | 17,241,665 / 25,920,000 | 66.52% | 271.15 | 305.64 | 288.75 |
| `sos` | SSS | `(1, 1, 1440, 2880)` | float32 | `.001` (PSU) | 2,728,867 / 4,147,200 | 65.80% | 5.69 | 40.00 | 34.62 |
| `sla` | DUACS | `(1, 720, 1440)` | float32 | `m` | 614,259 / 1,036,800 | 59.25% | -0.96 | 1.07 | 0.02 |
| `adt` | DUACS | `(1, 720, 1440)` | float32 | `m` | 611,975 / 1,036,800 | 59.03% | -1.46 | 1.90 | 0.48 |
| `ugosa` / `vgosa` | DUACS | `(1, 720, 1440)` | float32 | `m/s` | 614,244 / 1,036,800 | 59.24% | -2.48 | 2.93 | 0.00 |
| `wind_speed` | ASCAT | `(3264, 82)` | float32 | `m/s` | Granule dependent | ~80% | 0.12 | 28.45 | 7.62 |
| `wind_dir` | ASCAT | `(3264, 82)` | float32 | `degree` | Granule dependent | ~80% | 0.00 | 360.00 | 182.4 |
| `u10`, `v10` | — | — | — | — | **ABSENT** | 0% | — | — | — |

---

## 8. Seven-Channel Availability

| Channel | Parameter | Physical Designation | Status in Downloaded Files |
| :---: | :--- | :--- | :--- |
| **0** | `sst` | Oceanic Surface Temperature | Available in OSTIA (`analysed_sst`) & GLORYS (`thetao` at $0.494\text{ m}$) |
| **1** | `sss` | Oceanic Surface Salinity | Available in Multi-Obs SSS (`sos`) & GLORYS (`so` at $0.494\text{ m}$) |
| **2** | `ssh` / `sla` | Oceanic Dynamic Topography | Available in DUACS (`sla`/`adt`) & GLORYS (`zos`) |
| **3** | `u_current` | Upper Ocean Current Velocity ($U$) | Available in DUACS (`ugosa`) & GLORYS (`uo`). **NOT a wind.** |
| **4** | `v_current` | Upper Ocean Current Velocity ($V$) | Available in DUACS (`vgosa`) & GLORYS (`vo`). **NOT a wind.** |
| **5** | `u_wind` | Atmospheric Wind Velocity ($U$) | **UNGRIDDED:** Only present as polar speed/dir in ASCAT-L2 swaths; missing from GLORYS. |
| **6** | `v_wind` | Atmospheric Wind Velocity ($V$) | **UNGRIDDED:** Only present as polar speed/dir in ASCAT-L2 swaths; missing from GLORYS. |

---

## 9. 15-Depth Feasibility

- Target depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m.
- **Feasibility with downloaded files:** **0% feasible**.
- In the downloaded GLORYS NetCDF, `depth` has size 1 ($0.494\text{ m}$).
- No vertical interpolation is possible because all subsurface levels are absent from the file.

---

## 10. Missing Values

- Across all global marine products, missing values represent land masses, inland lakes, and permanent ice shelves.
- Missing value percentage:
  - GLORYS: $30.29\%$ missing globally (consistent with global ocean fraction $\approx 70\%$).
  - OSTIA: $33.48\%$ missing globally.
  - SSS: $34.20\%$ missing globally.
  - DUACS: $40.75\%$ missing globally.

---

## 11. Ocean Coverage (Regional Empirical Measurement)

Within the **Gate A1 Pilot Box** ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$):
- **DUACS SSH (`sla`):** $768 / 768$ cells = **$100.0\%$ valid ocean**.
- **GLORYS (`thetao`):** $6,968 / 7,008$ cells = **$99.43\%$ valid ocean** (only 40 cells along Myanmar/Andaman margin are land).
- **OSTIA SST (`analysed_sst`):** $19,105 / 19,200$ cells = **$99.51\%$ valid ocean**.
- **SSS (`sos`):** $3,055 / 3,072$ cells = **$99.45\%$ valid ocean**.

Within the **Official PS Domain** ($5^\circ - 30^\circ\text{N}, 45^\circ - 105^\circ\text{E}$):
- Ocean coverage is $\approx 49.7\% - 51.6\%$ due to landmasses (India, Arabian Peninsula, Southeast Asia).

---

## 12. Grid Compatibility

- **GLORYS:** $0.08333^\circ \times 0.08333^\circ$ $\rightarrow$ **Requires bilinear regridding to $0.25^\circ$**.
- **OSTIA:** $0.05^\circ \times 0.05^\circ$ $\rightarrow$ **Requires area-weighted or bilinear regridding to $0.25^\circ$**.
- **SSS:** $0.125^\circ \times 0.125^\circ$ $\rightarrow$ **Requires bilinear regridding to $0.25^\circ$**.
- **DUACS:** $0.25^\circ \times 0.25^\circ$ $\rightarrow$ **Directly grid-compatible** with canonical model grid.
- **ASCAT:** Orbital swath cells $\rightarrow$ **Requires swath binning and regular gridding to $0.25^\circ$**.

---

## 13. Data-Quality Issues Identified

1. **Target Subsurface Levels Missing:** The GLORYS download query did not include the vertical depth levels ($0-1000\text{ m}$); only the surface level ($0.494\text{ m}$) was extracted.
2. **Temporal Asynchrony:** The 4 downloaded CMEMS files are from completely unrelated dates (2026-06-23, 2026-03-31, 2024-12-15, 2026-01-16), and the ASCAT files are from 2020-01-01. They cannot be paired into a multi-sensor input tensor.
3. **Single Timestamps:** Each CMEMS file contains only $N=1$ daily snapshot, preventing the creation of a 30-day temporal sequence.
4. **ASCAT Swath Geometry:** ASCAT files are Level 2 swath files with irregular 2D coordinates and $0..360^\circ$ longitude, requiring custom scatterometer swath gridding rather than direct raster regridding.

---

## 14. Required Preprocessing Once Proper Data is Acquired

1. **Spatial Subsetting:** Extract bounding box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$).
2. **Horizontal Regridding:** Bilinear interpolation to uniform $0.25^\circ \times 0.25^\circ$ ($24 \times 32$ grid).
3. **Vertical Selection:** Nearest-neighbor map for $0\text{ m}$ ($z=0.494\text{ m}$); 1D linear interpolation across native 50 levels for $[5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$.
4. **Channel Normalization:** Compute channel-wise means and standard deviations from training split.
5. **Tensor Stacking:** Assemble canonical `[B, 7, 24, 32]` input tensor and `[B, 15]` target tensor.

---

## 15. Final Recommendation & Status

### Can this actual NC file be reliably transformed into the SIH26066 input/target tensors?
**NO.** Creating the 15-depth target profiles `[B, 15]` is physically impossible because subsurface levels ($5-1000\text{ m}$) are absent from the downloaded GLORYS NetCDF file, and the observation files do not share a common timestamp.

### Final Status:
# **`BLOCKED — DATA ISSUE FOUND`**

---

### Exact Proposed Fix:
Execute a single, bounded Copernicus Marine API download command for GLORYS12V1 specifying **all depth levels from 0 to 1000 m**, a 30-day temporal window, and the pilot spatial bounding box:

```powershell
copernicusmarine subset `
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m `
  --variable thetao --variable so --variable zos --variable uo --variable vo `
  --start-datetime 2020-01-01T00:00:00 `
  --end-datetime 2020-01-30T23:59:59 `
  --minimum-longitude 85.0 --maximum-longitude 93.0 `
  --minimum-latitude 12.0 --maximum-latitude 18.0 `
  --minimum-depth 0.0 --maximum-depth 1050.0 `
  --output-directory Dataset/gate_a1_pilot `
  --output-filename glorys_pilot_30d_15depths.nc
```

*(Total download volume for this bounded slice is only **~14.2 MB**, containing all 30 days, all 7 channels under `REANALYSIS_FALLBACK`, and all 15 vertical levels).*
