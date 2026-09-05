# OceanEmbed: Official Data Source Audit (SIH26066 Authoritative Specification)

This document establishes the official source availability audit for **SIH26066** based on the primary authoritative INCOIS Problem Statement document.

Every product explicitly designated in the official Problem Statement is audited below. Each source is categorized under strict scientific gating rules.

---

## 1. Primary vs. Fallback Product Hierarchy

| Ocean Parameter | Official Primary Source (SIH26066 Mandated) | Primary Status | Approved Fallback Option | Fallback Status | Justification / Role |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **SST** | **OSTIA** (0.05°, Daily) | **PRIMARY** | NOAA OISST v2.1 (0.25°, Daily) | **FALLBACK** | OSTIA is official high-res foundation SST; OISST is open HTTP/OPeNDAP fallback. |
| **SSS** | **SMAP / SMOS** (0.125°, Daily) | **PRIMARY** | CMEMS Multi-Year SSS L4 / GLORYS `so` ($z=0$) | **FALLBACK** | SMAP/SMOS 0.125° is official satellite SSS; CMEMS/GLORYS used if swath gaps occur. |
| **SSH / SLA** | **DUACS** (0.25°, Daily) | **PRIMARY** | GLORYS12V1 `zos` | **FALLBACK** | DUACS multi-altimeter is official SLA/ADT product; GLORYS `zos` is engineering fallback. |
| **Surface Current ($U, V$)** | **OSCAR** (0.25°, Daily) | **PRIMARY** | GLORYS12V1 `uo`, `vo` ($z=0$) / CMEMS Total Current L4 | **FALLBACK** | OSCAR is official satellite-derived ocean surface current (0–30m mixed layer). |
| **Surface Wind ($U, V$)** | **ASCAT-L2 Coastal / CCMP** (0.25°, Daily) | **PRIMARY** | ECMWF ERA5 $10\text{m}$ Reanalysis Winds | **FALLBACK** | ASCAT/CCMP are official satellite scatterometer winds; ERA5 is reanalysis fallback. |
| **Training Supervision** | **Copernicus GLORYS12V1** | **PRIMARY** | None (Singular reanalysis target) | — | 3D numerical reanalysis target ($0-1000\text{ m}$, 15 canonical depth levels). |
| **Observational Benchmark** | **INCOIS LAS — Gridded ARGO** | **PRIMARY** | Coriolis GDAC In-situ ARGO Float Profiles | **FALLBACK** | Independent in-situ observational validation pathway. |

---

## 2. Detailed Technical Audit of Official Sources

### 1. OSTIA — Operational Sea Surface Temperature and Sea Ice Analysis (SST)
- **Role & Priority:** **PRIMARY** (Official SIH26066 SST Source).
- **Official Name:** Global Ocean OSTIA Sea Surface Temperature and Sea Ice Reprocessed (or NRT).
- **Official Provider:** UK Met Office / Copernicus Marine Service (CMEMS).
- **Official URL:** `https://marine.copernicus.eu/`
- **Product Identifier:** `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011` (Dataset ID: `cmems_SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`)
- **Native Spatial Resolution:** $0.05^\circ \times 0.05^\circ$ ($\approx 6\text{ km}$ grid).
- **Native Temporal Resolution:** Daily ($1\text{-day}$ step, foundation SST free of diurnal warming).
- **Variable Names:**
  - `analysed_sst`: Foundation Sea Surface Temperature.
  - `analysis_error`: Estimated standard error of the SST analysis.
  - `sea_ice_fraction`: Sea ice concentration.
- **Physical Units:** `analysed_sst` in **Kelvin ($\text{K}$)** ($T_{^\circ\text{C}} = T_{\text{K}} - 273.15$).
- **Coordinate Conventions:**
  - Coordinates: `time` (standard Gregorian daily UTC), `lat` ($-89.975$ to $89.975$, ascending), `lon` ($-179.975$ to $179.975$, ascending).
  - Longitude convention: $-180^\circ$ to $+180^\circ$.
- **Depth Requirement:** Surface foundation ($z=0\text{ m}$).
- **Authentication:** Required (Free Copernicus Marine Service account).
- **Download Mechanism:** Copernicus Marine API / CLI (`copernicusmarine subset`), Python client, OPeNDAP.
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days):** $\approx 1.2\text{ MB}$ (regridded to $0.25^\circ$: $\approx 198\text{ KB}$).
- **Usability on This Machine:** **YES**, fully supported via Copernicus Marine API.
- **Known Limitations:** Heavy compute if downloading global $0.05^\circ$; strict regional spatial bounding box is required.
- **Metadata Status:** **VERIFIED** from CMEMS Product User Manual (PUM).

---

### 2. SMAP / SMOS — Satellite Sea Surface Salinity (SSS)
- **Role & Priority:** **PRIMARY** (Official SIH26066 SSS Source).
- **Official Name:** Global Ocean L4 Sea Surface Salinity (Multi-Observation Analysis) / RSS SMAP L3.
- **Official Provider:** Copernicus Marine Service (`MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013`) and NASA PO.DAAC / Remote Sensing Systems (RSS).
- **Official URL:** `https://marine.copernicus.eu/` / `https://podaac.jpl.nasa.gov/`
- **Product Identifier:**
  - CMEMS: `MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013` (Dataset: `cmems_obs-mob_glo_phy-sss_my_multi_P1D` at $0.125^\circ$).
  - NASA PO.DAAC: `SMAP_RSS_L3_SSS_SMI_DAILY_V5.0` ($0.25^\circ$).
- **Native Spatial Resolution:** $0.125^\circ \times 0.125^\circ$ ($\approx 12.5\text{ km}$, CMEMS optimal interpolation combining SMAP + SMOS).
- **Native Temporal Resolution:** Daily.
- **Variable Names:**
  - CMEMS: `sos` (Sea surface salinity) or `sss`.
  - PO.DAAC / RSS: `sss_smap` (70 km smoothed) or `sss_smap_40km`.
- **Physical Units:** Practical Salinity Units ($\text{PSU}$ / parts per thousand, $10^{-3}$).
- **Coordinate Conventions:**
  - Coordinates: `time` (daily UTC), `latitude` (ascending), `longitude` ($-180^\circ$ to $+180^\circ$, ascending).
- **Depth Requirement:** Surface ($z \approx 0-1\text{ cm}$ microwave skin depth).
- **Authentication:** Required (CMEMS account or NASA Earthdata login).
- **Download Mechanism:** Copernicus Marine API / NASA Earthdata `earthaccess` tool.
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days):** $\approx 0.8\text{ MB}$.
- **Usability on This Machine:** **YES**, fully accessible via CMEMS API.
- **Known Limitations:** Radio Frequency Interference (RFI) near coasts; pilot box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$) minimizes coastal RFI.
- **Metadata Status:** **VERIFIED** from CMEMS/PO.DAAC documentation.

---

### 3. DUACS — Altimeter Sea Surface Height / Sea Level Anomaly (SSH / SLA)
- **Role & Priority:** **PRIMARY** (Official SIH26066 Altimetry Source).
- **Official Name:** Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed.
- **Official Provider:** CNES / CLS / Copernicus Marine Service (CMEMS).
- **Official URL:** `https://marine.copernicus.eu/`
- **Product Identifier:** `SEALEVEL_GLO_PHY_L4_MY_008_047` (Dataset ID: `cmems_obs-sl_glo_phy-ssh_my_0.25deg_P1D`)
- **Native Spatial Resolution:** $0.25^\circ \times 0.25^\circ$ (or modern $0.125^\circ$).
- **Native Temporal Resolution:** Daily ($1\text{-day}$ step).
- **Variable Names:**
  - `sla`: Sea level anomaly relative to 1993–2012 mean sea surface ($\text{m}$).
  - `adt`: Absolute dynamic topography ($\text{m}$).
  - `ugosa`, `vgosa`: Geostrophic surface current velocity anomalies ($\text{m/s}$).
  - `ugos`, `vgos`: Absolute geostrophic surface current velocity ($\text{m/s}$).
- **Physical Units:** `sla` and `adt` in **meters ($\text{m}$)**.
- **Coordinate Conventions:**
  - Coordinates: `time` (days since 1950-01-01), `latitude` ($-89.875$ to $89.875$, ascending), `longitude` ($-179.875$ to $179.875$ or $0$ to $360$).
- **Depth Requirement:** Sea surface level ($z=0\text{ m}$).
- **Authentication:** Required (Copernicus Marine account).
- **Download Mechanism:** Copernicus Marine CLI / API / OPeNDAP.
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days):** $\approx 450\text{ KB}$.
- **Usability on This Machine:** **YES**, fast and lightweight.
- **Known Limitations:** Multi-satellite optimal interpolation has an effective temporal decorrelation scale of $\sim 10-15$ days; short-period high-frequency internal tides are filtered out.
- **Metadata Status:** **VERIFIED** from CMEMS user manuals.

---

### 4. OSCAR — Ocean Surface Current Analysis Real-time (Surface Currents)
- **Role & Priority:** **PRIMARY** (Official SIH26066 Surface Current Source).
- **Official Name:** Ocean Surface Current Analysis Real-time (OSCAR) Version 2.0.
- **Official Provider:** Earth & Space Research (ESR) / NASA PO.DAAC.
- **Official URL:** `https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0`
- **Product Identifier:** `OSCAR_L4_OC_FINAL_V2.0` (or `OSCAR_L4_OC_NRT_V2.0`).
- **Native Spatial Resolution:** $0.25^\circ \times 0.25^\circ$ grid.
- **Native Temporal Resolution:** Daily.
- **Variable Names:**
  - `u`: Zonal surface current velocity component ($\text{m/s}$, eastward).
  - `v`: Meridional surface current velocity component ($\text{m/s}$, northward).
  - `ug`: Zonal geostrophic velocity ($\text{m/s}$).
  - `vg`: Meridional geostrophic velocity ($\text{m/s}$).
- **Physical Units:** **Meters per second ($\text{m/s}$)**.
- **Coordinate Conventions:**
  - Coordinates: `time`, `latitude` ($-89.75$ to $89.75$, ascending), `longitude` ($0.25$ to $359.75$).
  - **CRITICAL NOTE ON LONGITUDE:** OSCAR uses a **$0^\circ - 360^\circ$ convention** (our pilot $85^\circ - 93^\circ\text{E}$ is in range $85.0$ to $93.0$, which maps identically).
- **Depth Requirement:** Top $30\text{ m}$ vertically integrated mixed-layer current.
- **Authentication:** Required (NASA Earthdata Login — free).
- **Download Mechanism:** NASA Earthdata OPeNDAP / HTTPS direct / PO.DAAC drive / `earthaccess` library.
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days):** $\approx 650\text{ KB}$.
- **Usability on This Machine:** **YES**, directly streamable via NASA Earthdata.
- **Known Limitations:** Derived from geostrophic balance and thermal wind equations; can underestimate high-frequency near-equatorial baroclinic waves.
- **Metadata Status:** **VERIFIED** from PO.DAAC dataset guide.

---

### 5. ASCAT-L2 Coastal / CCMP — Satellite Surface Winds
- **Role & Priority:** **PRIMARY** (Official SIH26066 Surface Wind Source).
- **Official Name:**
  - ASCAT: Advanced Scatterometer Level 2 Coastal Winds (EUMETSAT OSI SAF / KNMI).
  - CCMP: Cross-Calibrated Multi-Platform Gridded Surface Vector Winds Version 3.1 (Remote Sensing Systems).
- **Official Provider:** EUMETSAT OSI SAF / KNMI & NASA PO.DAAC / RSS / CMEMS.
- **Official URL:** `https://osi-saf.eumetsat.int/` / `https://podaac.jpl.nasa.gov/` / CMEMS `WIND_GLO_PHY_L4_MY_012_006`
- **Product Identifier:**
  - Gridded CCMP: `CCMP_V3.1_L4` (PO.DAAC) / `cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H` (CMEMS L4 daily-aggregated).
  - ASCAT Swath: `ASCATA-L2-Coastal` / `ASCATB-L2-Coastal`.
- **Native Spatial Resolution:** $0.25^\circ \times 0.25^\circ$ regular grid (for CCMP and CMEMS L4 ASCAT gridded analysis).
- **Native Temporal Resolution:** 6-hourly or Daily aggregated.
- **Variable Names:**
  - CCMP / CMEMS L4: `uwnd` (or `eastward_wind`), `vwind` (or `northward_wind`), `wind_speed`.
  - ASCAT L2 Swath: `wind_speed`, `wind_dir` on irregular wind vector cells (WVC).
- **Physical Units:** **Meters per second ($\text{m/s}$)** (equivalent $10\text{m}$ neutral stability wind).
- **Coordinate Conventions:**
  - Coordinates: `time`, `latitude` (ascending), `longitude` ($-180^\circ$ to $+180^\circ$).
- **Depth Requirement:** Atmospheric surface layer ($10\text{ m}$ height above sea level).
- **Authentication:** Required (CMEMS account or NASA Earthdata).
- **Download Mechanism:** Copernicus Marine API / NASA PO.DAAC OPeNDAP.
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days):** $\approx 700\text{ KB}$.
- **Usability on This Machine:** **YES**, available through CMEMS or PO.DAAC.
- **Known Limitations:** Raw ASCAT-L2 is an orbital swath (requires geospatial regridding and binning); CCMP or CMEMS L4 gridded ASCAT provides the gap-free $0.25^\circ$ regular analysis required by SIH26066.
- **Metadata Status:** **VERIFIED** from OSI SAF & RSS CCMP documentation.

---

### 6. Copernicus GLORYS12V1 — Global Ocean Physics Reanalysis (Training Target)
- **Role & Priority:** **PRIMARY** (Official SIH26066 Training Supervision Target).
- **Official Name:** Global Ocean Physics Reanalysis GLORYS12V1.
- **Official Provider:** Mercator Ocean International / Copernicus Marine Service (CMEMS).
- **Official URL:** `https://marine.copernicus.eu/`
- **Product Identifier:** `GLOBAL_MULTIYEAR_PHY_001_030` (Dataset ID: `cmems_mod_glo_phy_my_0.083deg_P1D-m`).
- **Native Spatial Resolution:** $1/12^\circ \approx 0.0833^\circ$ (horizontal) $\times$ 50 geopotential levels (vertical).
- **Native Temporal Resolution:** Daily-mean.
- **Variable Names:**
  - `thetao`: Sea water potential temperature ($^\circ\text{C}$).
  - `so`: Sea water salinity ($\text{PSU}$).
  - `zos`: Sea surface height above geoid ($\text{m}$).
  - `uo`, `vo`: Horizontal sea water velocity components ($\text{m/s}$).
- **Physical Units:** Potential temperature in **degrees Celsius ($^\circ\text{C}$)**.
- **Coordinate Conventions:**
  - Coordinates: `time` (daily UTC), `depth` (50 levels: $0.494\text{ m}$ to $5727.9\text{ m}$), `latitude` (ascending), `longitude` ($-180^\circ$ to $+180^\circ$).
- **Depth Levels for OceanEmbed:** Extracted and interpolated to canonical 15 depths:
  `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m`
  *(Target depth $0\text{ m}$ mapped to uppermost level $0.494\text{ m}$ without blind extrapolation).*
- **Authentication:** Required (Copernicus Marine free account).
- **Download Mechanism:** Copernicus Marine API (`copernicusmarine subset`).
- **Estimated Size for Pilot Box ($12^\circ - 18^\circ\text{N}, 85^\circ - 93^\circ\text{E}$, 30 days, 15 depths):** $\approx 14\text{ MB}$.
- **Usability on This Machine:** **YES**, ideal volume and format.
- **Metadata Status:** **VERIFIED** from CMEMS Product User Manual.

---

### 7. INCOIS LAS — Gridded ARGO (Observational Validation Benchmark)
- **Role & Priority:** **PRIMARY** (Official SIH26066 In-Situ Observational Validation).
- **Official Name:** INCOIS Gridded Argo Product for the Indian Ocean.
- **Official Provider:** Indian National Centre for Ocean Information Services (INCOIS), MoES, Govt. of India.
- **Official URL:** `https://las.incois.gov.in/` / `https://incois.gov.in/OOS/argo_products.jsp`
- **Product Identifier:** `INCOIS_Argo_Gridded_Monthly_IndianOcean` / `INCOIS_Argo_10Day_Analysis`.
- **Native Spatial Resolution:** $1.0^\circ \times 1.0^\circ$ (standard) or $0.5^\circ \times 0.5^\circ$ / $0.25^\circ$.
- **Native Temporal Resolution:** 10-day or Monthly gridded analysis (derived from irregular float surfacing cycles).
- **Variable Names:**
  - `temp` (or `temperature`): In-situ seawater temperature ($^\circ\text{C}$).
  - `sal` (or `salinity`): Practical salinity ($\text{PSU}$).
  - `depth` (or `level`): Discrete vertical levels ($0$ to $2000\text{ m}$).
- **Physical Units:** Temperature in **degrees Celsius ($^\circ\text{C}$)**.
- **Coordinate Conventions:**
  - Coordinates: `time`, `depth`, `latitude` ($30^\circ\text{S} - 30^\circ\text{N}$), `longitude` ($30^\circ\text{E} - 120^\circ\text{E}$).
- **Authentication:** None required (Open public access via INCOIS Live Access Server / OPeNDAP).
- **Download Mechanism:** INCOIS LAS web portal direct download / OPeNDAP / THREDDS server.
- **Estimated Size for Pilot Box:** $< 2\text{ MB}$.
- **Usability on This Machine:** **YES**, directly accessible.
- **Known Limitations:** In-situ Argo profiling floats cycle every 5 to 10 days; gridded products therefore represent smoothed 10-day or monthly statistical analyses rather than instantaneous daily snapshots. For high-frequency daily validation, collocated float point profiles ($T(z)$ CTD trajectories) serve as the point-to-grid verification pathway.
- **Metadata Status:** **DOCUMENTED BUT NOT FILE-VERIFIED** (Live Access Server requires network handshake).

---

## 3. Approved Fallback Sources

| Fallback Source | Replaces | Priority | Format | Auth | Rationale for Fallback Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **NOAA OISST v2.1** | OSTIA | **FALLBACK** | NetCDF4 | None (Open HTTPS) | Completely unauthenticated open OPeNDAP stream from NOAA PSL. Useful if CMEMS server downtime occurs. |
| **CMEMS SSS L4** | SMAP/SMOS | **FALLBACK** | NetCDF4 | CMEMS | Operational L4 reprocessed salinity combining multi-sensor feeds. |
| **GLORYS Surface Layers (`tos`, `sos`, `zos`, `uo`, `vo`)** | All 5 Surface Marine Variables | **FALLBACK (ENGINEERING MODE)** | NetCDF4 | CMEMS | Used strictly in `REANALYSIS_FALLBACK` mode for Gate A1 pipeline smoke testing. Eliminates cross-portal integration latency during initial ingestion verification. |
| **ECMWF ERA5 $10\text{m}$ Winds (`u10`, `v10`)** | ASCAT / CCMP | **FALLBACK** | NetCDF4 / GRIB | CDS / open | Gap-free hourly/daily atmospheric wind reanalysis. Used if scatterometer coastal swath gaps require complete spatial coverage. |
| **Coriolis GDAC ARGO Profiles** | INCOIS LAS | **FALLBACK** | NetCDF3/4 | Open FTP/HTTPS | Global raw CTD float point profiles (matched by nearest spatial-temporal neighbor) if INCOIS LAS server is under maintenance. |

---

## 4. Final Problem Statement Data Pipeline Architecture

```
========================================================================================
                      OFFICIAL SIH26066 TARGET PIPELINE ARCHITECTURE
========================================================================================

 [PRIMARY SURFACE OBSERVATIONS]
 ---------------------------------------------------------------------------------------
  OSTIA (0.05° SST)           ──┐
  SMAP / SMOS (0.125° SSS)    ──┤
  DUACS (0.25° SSH / SLA)     ──┼──> Quality Control & Coordinate Standardization
  OSCAR (0.25° Current U, V)  ──┤     (Ascending Lat/Lon, -180..180 Lon Convention)
  ASCAT / CCMP (0.25° Wind U,V)──┘
 ---------------------------------------------------------------------------------------
                                           │
                                           ▼
                                 Temporal Harmonization
                                (UTC Daily-Mean Aggregation)
                                           │
                                           ▼
                                   Spatial Regridding
                              (Bilinear Interpolation to
                               Common 0.25° x 0.25° Grid)
                                           │
                                           ▼
                            Channel Stacking & Normalization
                               (Train-Split Z-Score Fitted)
                                           │
                                           ▼
                             CANONICAL INPUT TENSOR [B, 7, H, W]
                              Channel 0: SST
                              Channel 1: SSS
                              Channel 2: SSH / SLA
                              Channel 3: Surface Current U (Oceanic)
                              Channel 4: Surface Current V (Oceanic)
                              Channel 5: Surface Wind U (Atmospheric)
                              Channel 6: Surface Wind V (Atmospheric)
                                           │
                                           ▼
                             =============================
                                     OceanEmbed
                                (CNN Latent Encoder +
                                   MLP Regressor Head)
                             =============================
                                           │
                                           ▼
                           CANONICAL OUTPUT TENSOR [B, 15]
                         Subsurface Temperature Profile T(z)
                        (0, 5, 10, 20, 30, 50, 75, 100, 125,
                         150, 200, 300, 500, 700, 1000 m)
                                   ▲               │
                                   │               ▼
            [TRAINING SUPERVISION] │      [INDEPENDENT VALIDATION]
            Copernicus GLORYS12V1 ─┘      INCOIS LAS Gridded ARGO
             Global Ocean Reanalysis       In-Situ Profiling Floats
             (15 Canonical Depths)         (Point-to-Profile Metrics)
========================================================================================
```

---

## 5. Candidate Pilot Strategies Comparison

| Evaluation Dimension | Strategy A: All 7 Official Surface Products + GLORYS Target | Strategy B: Controlled Ingestion with Reanalysis Fallback (Recommended for Gate A1) |
| :--- | :--- | :--- |
| **Surface Inputs** | OSTIA + SMAP/SMOS + DUACS + OSCAR + CCMP/ASCAT (5 distinct providers) | GLORYS12V1 Surface Layers (`tos`, `sos`, `zos`, `uo`, `vo`) + ERA5/CCMP Winds |
| **Target** | Copernicus GLORYS12V1 (15 Depths) | Copernicus GLORYS12V1 (15 Depths) |
| **Validation** | INCOIS LAS Gridded ARGO | INCOIS LAS Gridded ARGO (benchmarking) |
| **Data Source Mode** | `SATELLITE_OBSERVATION` | `REANALYSIS_FALLBACK` (Engineering Smoke Test) |
| **Data Availability** | High, but distributed across CMEMS, PO.DAAC, and RSS | 100% available via single Copernicus Marine API call |
| **Download Complexity** | High (5 disparate API clients, 3 different login keys) | Very Low (single authenticated API query) |
| **Data Volume (Pilot Box)**| $\sim 35\text{ MB} - 60\text{ MB}$ across all sources | $\sim 12\text{ MB} - 18\text{ MB}$ total |
| **Preprocessing Complexity**| High (reconciling 5 coordinate systems, calendars, masks) | Low (internally self-consistent spatial grid) |
| **PS Fidelity** | 100% (Strict satellite observation demonstration) | 90% (Uses exact PS channels & depths; surface reanalysis proxy) |
| **Scientific Credibility**| Complete observational validation | Appropriate as an initial engineering proof-of-concept |
| **Hackathon Feasibility** | Risk of download/authentication failure during initial run | Zero-risk immediate pipeline verification |

### Scientific Strategy Recommendation:
- **For Gate A1 (Real-Data Pipeline Smoke Test):** Execute **Strategy B (`REANALYSIS_FALLBACK`)**. It delivers the exact 7 channels and 15 depths required by the Problem Statement in a single self-consistent ~14 MB download, allowing immediate automated verification of coordinate ordering, regridding, masking, tensor shapes, and GPU compatibility without cross-portal API friction.
- **For Gate A2 / Phase B (Observational Demonstration):** Execute **Strategy A (`SATELLITE_OBSERVATION`)**, pulling OSTIA, SMAP, DUACS, OSCAR, and CCMP/ASCAT as independent observation-derived surface fields.

---

## 6. Critical Physical Variable Disambiguation

To prevent physical or conceptual confusion across model heads:

1. **Surface Ocean Currents ($U, V$) vs. Surface Atmospheric Winds ($U, V$):**
   - **Currents (`u_current`, `v_current`):** Physical velocity of the upper ocean water column ($0-30\text{ m}$ mixed layer). Magnitude typically $0.01 - 1.5\text{ m/s}$. Governed by geostrophic balance and oceanic Ekman drift.
   - **Winds (`u_wind`, `v_wind`):** Physical velocity of atmospheric air parcels at $10\text{ m}$ reference height above the sea surface. Magnitude typically $1.0 - 25.0\text{ m/s}$. Governed by atmospheric pressure gradients.
   - **Channels are strictly segregated:** Currents occupy channels 3 & 4; winds occupy channels 5 & 6.
2. **SST and SSS:**
   - Radiometric / thermal observations of the ocean skin and mixed layer ($^\circ\text{C}$ and $\text{PSU}$).
3. **SSH / SLA:**
   - Radar altimetry-derived dynamic topography representing vertically integrated heat/salt content (steric height, $\text{m}$).
4. **GLORYS vs. ARGO:**
   - **GLORYS12V1:** Reanalysis-derived training supervision. Numerical model assimilating observation data to produce 3D gridded targets.
   - **INCOIS LAS ARGO:** Independent observational validation benchmark. Genuine in-situ CTD profiles providing real-world physical ground truth.
