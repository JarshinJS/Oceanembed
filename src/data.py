"""
OceanEmbed — Data Module (SIH26066 Aligned)
===========================================
Handles data loading, harmonisation, and synthetic generation.

Canonical Specifications:
  - 7 Surface Input Channels:
      Channel 0: SST (Sea Surface Temperature)
      Channel 1: SSS (Sea Surface Salinity)
      Channel 2: SLA / SSH (Sea Level Anomaly / Sea Surface Height)
      Channel 3: Surface Current U (Zonal component)
      Channel 4: Surface Current V (Meridional component)
      Channel 5: Surface Wind U (10m Zonal wind vector)
      Channel 6: Surface Wind V (10m Meridional wind vector)
  - Canonical Input Shape:  (B, 7, H, W)
  - Canonical Output Shape: (B, 15, H, W) for 3D field, or (B, 15) for profile.
  - 15 Canonical Depth Levels:
      [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
  - Training Supervision: Copernicus GLORYS12V1 (reanalysis-derived training target).
  - Observational Validation: In-situ ARGO float profiles / RAMA moorings.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from src.constants import (
    REQUIRED_DEPTHS_M,
    INPUT_CHANNELS,
    CANONICAL_SURFACE_VARIABLES,
    OFFICIAL_PS_DOMAIN,
    MVP_PILOT_DOMAIN,
)


# ------------------------------------------------------------------ #
#  Grid helpers
# ------------------------------------------------------------------ #
def build_grid(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return lat / lon arrays for the configured region."""
    r = cfg.get("region", MVP_PILOT_DOMAIN)
    res = r["grid_resolution"]
    lats = np.arange(r["lat_min"], r["lat_max"] + res / 2, res)
    lons = np.arange(r["lon_min"], r["lon_max"] + res / 2, res)
    return lats, lons


def n_days(cfg: dict) -> int:
    """Number of daily samples in the temporal window."""
    import pandas as pd

    dates = pd.date_range(
        cfg["temporal"]["start_date"],
        cfg["temporal"]["end_date"],
        freq=cfg["temporal"]["frequency"],
    )
    return len(dates)


# ------------------------------------------------------------------ #
#  Synthetic data generator
# ------------------------------------------------------------------ #
class SyntheticOceanDataset(Dataset):
    """Generates synthetic 7-channel surface fields and matching 15-depth
    subsurface temperature profiles for rapid verification and smoke testing.

    Physics-flavoured relationships:
      * SST drives a mixed-layer temperature that decays exponentially with depth.
      * SSH / SLA correlates with integrated upper-ocean heat content.
      * SSS adds a subtle halocline-temperature covariance.
      * Surface wind vectors (u_wind, v_wind) modulate mixed-layer depth (MLD).
      * Surface currents (u_current, v_current) reflect wind-driven Ekman transport
        and geostrophic flow.
    """

    def __init__(self, cfg: dict, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.lats, self.lons = build_grid(cfg)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        # Ensure canonical 15 depths are used
        self.depths: List[int] = cfg.get("target", {}).get("depths", REQUIRED_DEPTHS_M)
        self.n_depths = len(self.depths)

        # Canonical surface variables (7 channels)
        surface_vars = cfg.get("surface_variables", CANONICAL_SURFACE_VARIABLES)
        self.surface_variables = surface_vars
        self.n_channels = len(surface_vars)

        import pandas as pd

        self.dates = pd.date_range(
            cfg["temporal"]["start_date"],
            cfg["temporal"]["end_date"],
            freq=cfg["temporal"]["frequency"],
        )
        self._split_indices()
        self._generate()

    # -- splitting ---------------------------------------------------
    def _split_indices(self):
        """Split dates into train/val/test.
        
        Default mode is 'temporal' (contiguous chronological blocks), which
        protects against data leakage caused by strong day-to-day ocean
        autocorrelation.
        """
        n = len(self.dates)
        t = self.cfg.get("training", {})
        split_mode = t.get("split_mode", "temporal")

        if split_mode == "temporal":
            # Chronological separation: train (early) -> val (mid) -> test (late)
            idx = np.arange(n)
        else:
            # Random permutation (baseline / comparison only)
            rng = np.random.default_rng(t.get("seed", 42))
            idx = rng.permutation(n)

        n_tr = int(n * t.get("train_split", 0.7))
        n_va = int(n * t.get("val_split", 0.15))
        if self.split == "train":
            self.indices = idx[:n_tr]
        elif self.split == "val":
            self.indices = idx[n_tr : n_tr + n_va]
        else:
            self.indices = idx[n_tr + n_va :]

    # -- generation --------------------------------------------------
    def _generate(self):
        """Pre-generate 7 surface channels and 15 subsurface depths."""
        n = len(self.indices)
        H, W = self.n_lat, self.n_lon
        rng = np.random.default_rng(42)

        # Latitude-dependent baseline SST (warm equator -> cool northward)
        lat_rad = np.radians(self.lats)
        sst_base = 30.0 - 0.5 * (self.lats - 5.0)  # ~30 to 20 °C
        sst_base = np.broadcast_to(sst_base[:, None], (H, W))

        # Seasonal cycle (day-of-year)
        doy = np.array([d.dayofyear for d in self.dates[self.indices]])
        seasonal = 2.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)  # peak ~May/Jun
        seasonal = seasonal[:, None, None]  # (N, 1, 1)

        # Spatial noise
        noise = rng.normal(0, 0.4, size=(n, H, W))

        # Channel 0: SST (°C)
        self.sst = sst_base[None] + seasonal + noise

        # Channel 1: SSS (PSU)
        self.sss = 34.0 + 0.5 * np.sin(lat_rad)[:, None][None] + rng.normal(0, 0.1, (n, H, W))

        # Channel 2: SSH / SLA (m)
        self.ssh = 0.3 + 0.01 * (self.sst - 25.0) + rng.normal(0, 0.02, (n, H, W))

        # Channels 5 & 6: 10m Surface Wind Vectors (m/s)
        self.u_wind = 3.0 + 2.0 * np.sin(2 * np.pi * doy / 365.0)[:, None, None] + rng.normal(0, 1.2, (n, H, W))
        self.v_wind = 1.5 + 2.5 * np.cos(2 * np.pi * doy / 365.0)[:, None, None] + rng.normal(0, 1.2, (n, H, W))

        # Channels 3 & 4: Surface Current Vectors (m/s) (distinct from winds)
        # Driven partially by wind-drift and steric gradients
        self.u_current = 0.04 * self.u_wind + rng.normal(0, 0.05, (n, H, W))
        self.v_current = 0.04 * self.v_wind + rng.normal(0, 0.05, (n, H, W))

        # Mixed-layer depth modulated by wind stress
        wind_speed = np.sqrt(self.u_wind**2 + self.v_wind**2)
        mld = 35.0 + 5.0 * wind_speed
        mld = np.clip(mld, 20.0, 120.0)

        # Subsurface temperature profile at the 15 canonical depth levels:
        # T(z) = SST - (SST - T_deep) * (1 - exp(-z / mld))
        self.targets = np.zeros((n, self.n_depths), dtype=np.float32)
        for i, z in enumerate(self.depths):
            deep_t = 5.0  # deep ocean temperature (~1000m)
            t_z = self.sst - (self.sst - deep_t) * (1.0 - np.exp(-np.array(z) / mld))
            # Spatial-mean profile for this day
            self.targets[:, i] = t_z.mean(axis=(1, 2)).astype(np.float32)

    # -- torch interface --------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (surface_image, temp_profile).

        surface_image shape: (7, H, W) [Canonical 7 channels]
          Channel 0: SST
          Channel 1: SSS
          Channel 2: SLA / SSH
          Channel 3: Surface Current U
          Channel 4: Surface Current V
          Channel 5: Surface Wind U
          Channel 6: Surface Wind V
        temp_profile shape:  (15,) [Canonical 15 depths]
        """
        img = np.stack(
            [
                self.sst[idx],
                self.sss[idx],
                self.ssh[idx],
                self.u_current[idx],
                self.v_current[idx],
                self.u_wind[idx],
                self.v_wind[idx],
            ],
            axis=0,
        ).astype(np.float32)
        target = self.targets[idx]
        return torch.from_numpy(img), torch.from_numpy(target)


# ------------------------------------------------------------------ #
#  Real-data stub (GLORYS / Satellite Ingestion)
# ------------------------------------------------------------------ #
class GLORYSDataset(Dataset):
    """Dataset class for real Copernicus GLORYS12V1 reanalysis-derived training
    targets + satellite surface observations.

    Important terminology:
      - GLORYS12V1 is a "reanalysis-derived training target" or "training supervision",
        never "ground truth".
      - Observational validation pathway is provided by independent in-situ ARGO
        float temperature profiles and RAMA mooring arrays.

    Canonical 7 Input Channels (regridded to common 0.25° grid):
      - Channel 0: SST (NOAA OISST v2.1 or CMEMS L4 SST)
      - Channel 1: SSS (CMEMS Multi-Year SSS L4 or SMAP)
      - Channel 2: SLA / SSH (C3S / CMEMS DUACS Altimetry)
      - Channel 3: Surface Current U (OSCAR or CMEMS Total Surface Current L4)
      - Channel 4: Surface Current V (OSCAR or CMEMS Total Surface Current L4)
      - Channel 5: Surface Wind U (ECMWF ERA5 10m zonal wind vector)
      - Channel 6: Surface Wind V (ECMWF ERA5 10m meridional wind vector)

    Canonical 15 Target Depths:
      [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
    """

    def __init__(self, cfg: dict, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        raise NotImplementedError(
            "GLORYSDataset is a stub — implement NetCDF loading "
            "with xarray to use real data."
        )
