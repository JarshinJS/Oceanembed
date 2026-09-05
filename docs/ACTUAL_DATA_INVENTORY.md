# OceanEmbed: Actual Downloaded Data Inventory & Channel Audit

This document audits the physical NetCDF datasets downloaded into the repository (`Dataset/`), rigorously cross-referencing each required SIH26066 input channel and target depth against empirical file contents.

Audit Date: 2026-09-04
Files Location: `Dataset/`
Total NetCDF Files Found: 20 (4 global CMEMS NetCDF files + 16 ASCAT-L2 orbital swath files)

---

## 1. Canonical 7 Surface Input Channels vs. Actual Data

| Required Channel Index | Required Physical Parameter | Official SIH26066 Source | Actual Variable in Downloaded Files | Present? | Physical Source Dataset | Native Units | Native Dimensions & Shape | Disambiguation & Physical Verification |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **0** | **SST** (Sea Surface Temperature) | OSTIA | `analysed_sst` (OSTIA)<br>`thetao` ($z=0.494\text{m}$, GLORYS) | **YES** | `METOFFICE-GLO-SST-L4-REP-OBS-SST` / GLORYS12V1 | $\text{K}$ (OSTIA)<br>$^\circ\text{C}$ (GLORYS) | `(time: 1, lat: 3600, lon: 7200)`<br>`(time: 1, depth: 1, lat: 2041, lon: 4320)` | **VERIFIED SURFACE OBSERVATION.** Radiometric foundation temperature. |
| **1** | **SSS** (Sea Surface Salinity) | SMAP / SMOS | `sos` (CMEMS Multi-Obs)<br>`so` ($z=0.494\text{m}$, GLORYS) | **YES** | `cmems_obs-mob_glo_phy-sss_my_multi_P1D` / GLORYS12V1 | $\text{PSU}$ ($10^{-3}$) | `(time: 1, depth: 1, lat: 1440, lon: 2880)`<br>`(time: 1, depth: 1, lat: 2041, lon: 4320)` | **VERIFIED SURFACE OBSERVATION.** Microwave radiometer salinity analysis. |
| **2** | **SSH / SLA** (Sea Surface Height / Anomaly) | DUACS | `sla` (Sea Level Anomaly)<br>`adt` (Dynamic Topography)<br>`zos` (GLORYS SSH) | **YES** | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D` / GLORYS12V1 | $\text{m}$ | `(time: 1, lat: 720, lon: 1440)`<br>`(time: 1, lat: 2041, lon: 4320)` | **VERIFIED ALTIMETRY OBSERVATION.** Radar altimeter dynamic topography. |
| **3** | **Surface Current U** (Oceanic Zonal Velocity) | OSCAR / CMEMS | `uo` (GLORYS total ocean current)<br>`ugos` / `ugosa` (DUACS geostrophic) | **YES** (via GLORYS/DUACS)<br>*(OSCAR absent)* | `cmems_mod_glo_phy_my_0.083deg_P1D-m` / DUACS | $\text{m/s}$ | `(time: 1, depth: 1, lat: 2041, lon: 4320)`<br>`(time: 1, lat: 720, lon: 1440)` | **VERIFIED OCEAN CURRENT.** Physical velocity of the upper ocean water column. **NOT an atmospheric wind.** |
| **4** | **Surface Current V** (Oceanic Meridional Velocity) | OSCAR / CMEMS | `vo` (GLORYS total ocean current)<br>`vgos` / `vgosa` (DUACS geostrophic) | **YES** (via GLORYS/DUACS)<br>*(OSCAR absent)* | `cmems_mod_glo_phy_my_0.083deg_P1D-m` / DUACS | $\text{m/s}$ | `(time: 1, depth: 1, lat: 2041, lon: 4320)`<br>`(time: 1, lat: 720, lon: 1440)` | **VERIFIED OCEAN CURRENT.** Physical velocity of the upper ocean water column. **NOT an atmospheric wind.** |
| **5** | **Surface Wind U** (Atmospheric Zonal Wind) | ASCAT-L2 / CCMP | `wind_speed` + `wind_dir` (ASCAT-L2)<br>*(GLORYS `u10` is absent)* | **PARTIAL / UNGRIDDED** | `Dataset/ASCAT/*.nc` (MetOp-C ASCAT-L2) | $\text{m/s}$, degrees | `(NUMROWS: 3264, NUMCELLS: 82)` | **ATMOSPHERIC WIND.** Available only as ungridded orbital swath polar vectors; requires $u = -S \sin(\theta)$ decomposition and spatial binning to regular $0.25^\circ$. |
| **6** | **Surface Wind V** (Atmospheric Meridional Wind) | ASCAT-L2 / CCMP | `wind_speed` + `wind_dir` (ASCAT-L2)<br>*(GLORYS `v10` is absent)* | **PARTIAL / UNGRIDDED** | `Dataset/ASCAT/*.nc` (MetOp-C ASCAT-L2) | $\text{m/s}$, degrees | `(NUMROWS: 3264, NUMCELLS: 82)` | **ATMOSPHERIC WIND.** Available only as ungridded orbital swath polar vectors; requires $v = -S \cos(\theta)$ decomposition and spatial binning to regular $0.25^\circ$. |

---

## 2. Canonical 15 Subsurface Target Depths vs. Actual GLORYS File

Required Depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m.

| Target Depth | Required by SIH26066 | Present in Downloaded GLORYS NetCDF? | Actual Native Level in File | Extrapolation / Interpolation Status |
| :---: | :---: | :---: | :---: | :--- |
| **0 m** | **YES** | **YES** (Mapped to nearest level) | $0.494025\text{ m}$ | Mapped to uppermost cell center without upward extrapolation. |
| **5 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** Native subsurface levels were truncated in download. |
| **10 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **20 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **30 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **50 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **75 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **100 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **125 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **150 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **200 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **300 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **500 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **700 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |
| **1000 m** | **YES** | **NO — MISSING** | None | **UNAVAILABLE.** |

---

## 3. Temporal Consistency Across Downloaded Files

| Product | Filename | Time Dimension Size | Exact Timestamps in File | Temporal Harmonization Status |
| :--- | :--- | :---: | :--- | :--- |
| **GLORYS** | `cmems_mod_glo_phy_my_0.083deg_P1D-m_1788514642488.nc` | 1 | `2026-06-23T00:00:00` | **MUTUALLY DISCORDANT.** |
| **OSTIA SST** | `METOFFICE-GLO-SST-L4-REP-OBS-SST_1788514360579.nc` | 1 | `2026-03-31T00:00:00` | **MUTUALLY DISCORDANT.** |
| **Multi-Obs SSS**| `cmems_obs-mob_glo_phy-sss_my_multi_P1D_1788514338421.nc` | 1 | `2024-12-15T00:00:00` | **MUTUALLY DISCORDANT.** |
| **DUACS SSH** | `c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_1788514168280.nc` | 1 | `2026-01-16T00:00:00` | **MUTUALLY DISCORDANT.** |
| **ASCAT Winds** | `Dataset/ASCAT/*.nc` (16 orbital files) | 16 granules | `2019-12-31T22:33` to `2020-01-01T23:54` | **MUTUALLY DISCORDANT.** |

---

## 4. Summary Audit Verdict

- **Total Channels Available on a Shared Timestamp:** **0 of 7** (no single date is shared by all observation files).
- **Target Depths Available in GLORYS:** **1 of 15** ($0.494\text{ m}$ only).
- **Subsurface Depth Levels (5 to 1000 m):** **0% present**.
- **Audit Verdict:** **`BLOCKED — DATA ISSUE FOUND`**. Corrected download parameters must be executed to obtain full-column GLORYS profiles ($0-1000\text{ m}$) across a continuous temporal sequence.
