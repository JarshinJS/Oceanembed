"""
OceanEmbed — Constants and Official PS Specifications
=====================================================
Centralized configuration parameters and constants aligned with the
official SIH26066 Problem Statement requirements.
"""

from typing import Dict, List

# =====================================================================
# 1. DEPTH SPECIFICATIONS (Official SIH26066: 15 vertical levels)
# =====================================================================
# Canonical 15 depth levels:
REQUIRED_DEPTHS_M: List[int] = [
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
]
OUTPUT_DEPTHS: int = 15

# GLORYS Vertical Depth Mapping Transformation Rules:
# - Target depth = 0 m: Map directly to nearest valid uppermost surface
#   model level (approx 0.494 m in GLORYS12V1). Blind extrapolation to 0 m is forbidden.
# - Target depth > 0 m: Vertically interpolate from native GLORYS depth levels
#   without unconstrained extrapolation beyond deepest valid ocean level.
GLORYS_SURFACE_LEVEL_APPROX_M: float = 0.494025

# =====================================================================
# 2. SURFACE INPUT FIELDS (Official SIH26066: 7 surface variables)
# =====================================================================
# Canonical channel order:
# Channel 0: SST (Sea Surface Temperature)
# Channel 1: SSS (Sea Surface Salinity)
# Channel 2: SLA / SSH (Sea Level Anomaly / Sea Surface Height)
# Channel 3: Surface Current U (Zonal component)
# Channel 4: Surface Current V (Meridional component)
# Channel 5: Surface Wind U (Zonal 10m wind vector)
# Channel 6: Surface Wind V (Meridional 10m wind vector)
INPUT_CHANNELS: int = 7

CANONICAL_SURFACE_VARIABLES: List[str] = [
    "sst",
    "sss",
    "ssh",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
]

CHANNEL_NAME_MAP: Dict[int, str] = {
    0: "SST (Sea Surface Temperature)",
    1: "SSS (Sea Surface Salinity)",
    2: "SLA/SSH (Sea Level Anomaly / Sea Surface Height)",
    3: "Surface Current U (Zonal)",
    4: "Surface Current V (Meridional)",
    5: "Surface Wind U (Zonal 10m)",
    6: "Surface Wind V (Meridional 10m)",
}

# =====================================================================
# 3. DOMAIN SPECIFICATIONS (Official PS Domain vs MVP Pilot Subset)
# =====================================================================
OFFICIAL_PS_DOMAIN = {
    "name": "North Indian Ocean (Official PS Domain)",
    "lat_min": 5.0,
    "lat_max": 30.0,
    "lon_min": 45.0,
    "lon_max": 105.0,
    "grid_resolution": 0.25,  # degrees
    "description": "Full North Indian Ocean encompassing Arabian Sea and Bay of Bengal",
}

MVP_PILOT_DOMAIN = {
    "name": "Bay of Bengal (MVP / Pilot Subset)",
    "lat_min": 5.0,
    "lat_max": 25.0,
    "lon_min": 80.0,
    "lon_max": 100.0,
    "grid_resolution": 0.25,  # degrees
    "description": "Restricted Bay of Bengal subset for Phase A pipeline and model feasibility pilot",
}

# Focused Gate A1 Pilot Box (Canonical 0.25° cell-center grid)
GATE_A1_PILOT_BOX = {
    "name": "Central Bay of Bengal Pilot Box",
    "lat_min": 12.0,
    "lat_max": 18.0,
    "lon_min": 85.0,
    "lon_max": 93.0,
    "grid_resolution": 0.25,  # degrees
    "grid_height": 24,        # 6.0 deg / 0.25 deg = 24 cell centers
    "grid_width": 32,         # 8.0 deg / 0.25 deg = 32 cell centers
    "grid_registration": "cell_center",  # DUACS and CCMP native standard
    "lat_centers": [round(12.125 + i * 0.25, 4) for i in range(24)],
    "lon_centers": [round(85.125 + j * 0.25, 4) for j in range(32)],
    "description": "Canonical 24 x 32 cell-centered 0.25° grid covering 12-18N, 85-93E",
}

# =====================================================================
# 4. DATA SOURCE MODES (Scientific Contract Requirement)
# =====================================================================
DATA_SOURCE_MODES = {
    "SATELLITE_OBSERVATION": (
        "Pure satellite observation inputs (e.g. NOAA OISST, SMAP SSS, "
        "DUACS Altimetry, OSCAR Currents, ERA5/CCMP Winds). This mode is "
        "required for final Problem Statement demonstration."
    ),
    "REANALYSIS_FALLBACK": (
        "Surface fields extracted from ocean reanalysis (e.g. GLORYS12V1 "
        "surface layers tos, sos, zos, uo, vo). Used strictly for initial "
        "engineering convenience and pipeline proof. Does NOT constitute "
        "the final satellite-only PS demonstration."
    ),
    "MIXED": (
        "Combination of satellite observation and reanalysis boundary fields."
    ),
}

# =====================================================================
# 5. EXPERIMENT GATES (Scientific Reframing)
# =====================================================================
GATE_STAGES = {
    "A1": {
        "name": "Gate A1 — Real-data pipeline smoke test",
        "sample_count": 30,
        "purpose": (
            "Validate NetCDF ingestion, coordinate validation, regridding, "
            "masking, normalization, 7-channel tensor creation [B,7,H,W], "
            "15-depth target creation [B,15], and GPU forward/backward compatibility. "
            "NOT presented as evidence of model generalization."
        ),
    },
    "A2": {
        "name": "Gate A2 — Preliminary ML feasibility experiment",
        "sample_count": 90,  # recommended >= 90 days
        "purpose": (
            "Preliminary ML feasibility: baseline comparison (Climatology, Linear Regression), "
            "temporal holdout split, and preliminary depth-wise RMSE/correlation/bias evaluation."
        ),
    },
}
