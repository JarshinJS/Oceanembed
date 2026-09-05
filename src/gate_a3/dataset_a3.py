"""
Gate A3 Dataset Loader (91-Day Temporal Sequence)
=================================================
Loads canonical 91-day synchronized NetCDFs and provides PyTorch Datasets
with chronological temporal holdout splits and leakage-free normalization.

Chronological Split Contract (SIH26066 Gate A3.3):
  - 'train': Days 0–59 (60 days: 2020-01-01 to 2020-02-29)
  - 'val': Days 60–74 (15 days: 2020-03-01 to 2020-03-15)
  - 'test': Days 75–90 (16 days: 2020-03-16 to 2020-03-31)
  - 'train_full': Days 0–59 (60 days: 2020-01-01 to 2020-02-29)
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from src.constants import CANONICAL_SURFACE_VARIABLES, REQUIRED_DEPTHS_M

_ARRAY_CACHE = {}


class SynchronizedOceanDatasetA3(Dataset):
    """
    PyTorch Dataset for Gate A3 91-day synchronized sequence (Q1 2020).
    """

    def __init__(
        self,
        *args,
        split: str = "train",
        data_dir: str = "Dataset/gate_a3_temporal/synchronized",
        norm_file: str = "Dataset/gate_a3_temporal/normalization/normalization_stats.json",
        active_channels: Optional[List[int]] = None,
        normalize_target: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Allow positional passing of split or file paths
        if args:
            if isinstance(args[0], str) and args[0] in ["train", "val", "test", "train_full"]:
                split = args[0]
            elif len(args) >= 3:
                norm_file = str(args[2])
                data_dir = str(Path(args[0]).parent)
        self.split = split
        self.normalize_target = normalize_target
        self.active_channels = active_channels if active_channels is not None else list(range(7))

        in_path = Path(data_dir) / "oceanembed_inputs_7ch_91d.nc"
        tgt_path = Path(data_dir) / "oceanembed_target_15depth_91d.nc"

        with open(norm_file, "r") as f:
            self.norm_metadata = json.load(f)

        # Load normalization stats from train period only
        norm_stats = self.norm_metadata
        if "input_means" in norm_stats:
            self.input_means = np.array(norm_stats["input_means"], dtype=np.float32)
            self.input_stds = np.array(norm_stats["input_stds"], dtype=np.float32)
        else:
            self.input_means = np.zeros(7, dtype=np.float32)
            self.input_stds = np.zeros(7, dtype=np.float32)
            for c_idx, c_name in enumerate(CANONICAL_SURFACE_VARIABLES):
                c_info = norm_stats["input_channels"][c_name]
                self.input_means[c_idx] = c_info["train_mean"]
                self.input_stds[c_idx] = c_info["train_std"]

        if "target_means" in norm_stats:
            self.target_means = np.array(norm_stats["target_means"], dtype=np.float32)
            self.target_stds = np.array(norm_stats["target_stds"], dtype=np.float32)
        else:
            self.target_means = np.zeros(15, dtype=np.float32)
            self.target_stds = np.zeros(15, dtype=np.float32)
            for d_idx, depth_m in enumerate(REQUIRED_DEPTHS_M):
                key = f"{float(depth_m)}m" if f"{float(depth_m)}m" in norm_stats["target_depths"] else f"{depth_m}m"
                if key not in norm_stats["target_depths"]:
                    key = f"depth_{depth_m}m"
                d_info = norm_stats["target_depths"][key]
                self.target_means[d_idx] = d_info["train_mean"]
                self.target_stds[d_idx] = d_info["train_std"]

        # Prevent zero-division
        self.input_stds[self.input_stds < 1e-6] = 1.0
        self.target_stds[self.target_stds < 1e-6] = 1.0

        # Global cache to prevent repeated NetCDF file opening on Windows
        cache_key = (str(in_path), str(tgt_path))
        if cache_key not in _ARRAY_CACHE:
            with xr.open_dataset(in_path) as ds_in, xr.open_dataset(tgt_path) as ds_tgt:
                _ARRAY_CACHE[cache_key] = {
                    "times": [str(t)[:10] for t in ds_in.time.values],
                    "inputs": ds_in.inputs.values.astype(np.float32),
                    "mask_inputs": ds_in.mask_inputs.values.astype(np.float32),
                    "targets": ds_tgt.target_thetao.values.astype(np.float32),
                    "mask_target": ds_tgt.mask_target.values.astype(np.float32),
                    "ocean_mask_2d": ds_in.ocean_mask_2d.values.astype(np.float32),
                    "latitudes": ds_in.latitude.values.astype(np.float32),
                    "longitudes": ds_in.longitude.values.astype(np.float32),
                    "depths": ds_tgt.depth.values.astype(np.float32),
                }

        cached = _ARRAY_CACHE[cache_key]
        self.full_times = cached["times"]
        self.full_inputs = cached["inputs"]          # [91, 7, 24, 32]
        self.full_mask_in = cached["mask_inputs"]    # [91, 7, 24, 32]
        self.full_targets = cached["targets"]        # [91, 15, 24, 32]
        self.full_mask_tgt = cached["mask_target"]   # [91, 15, 24, 32]
        self.ocean_mask_2d = cached["ocean_mask_2d"] # [24, 32]
        self.latitudes = cached["latitudes"]
        self.longitudes = cached["longitudes"]
        self.depths = cached["depths"]

        if split == "train":
            self.indices = list(range(0, 60))
        elif split == "val":
            self.indices = list(range(60, 75))
        elif split == "test":
            self.indices = list(range(75, 91))
        elif split == "train_full":
            self.indices = list(range(0, 60))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.dates = [self.full_times[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def unnormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized target array [..., 15, H, W] back to physical Celsius (°C)."""
        shape = y_norm.shape
        d_idx = shape.index(15) if 15 in shape else 1
        broadcast_dims = [1] * len(shape)
        broadcast_dims[d_idx] = 15

        means = self.target_means.reshape(broadcast_dims)
        stds = self.target_stds.reshape(broadcast_dims)
        return y_norm * stds + means

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        time_idx = self.indices[idx]
        date_str = self.full_times[time_idx]

        # Extract sample
        x_raw = self.full_inputs[time_idx].copy()        # [7, 24, 32]
        m_in = self.full_mask_in[time_idx].copy()        # [7, 24, 32]
        y_raw = self.full_targets[time_idx].copy()       # [15, 24, 32]
        m_tgt = self.full_mask_tgt[time_idx].copy()      # [15, 24, 32]

        # Subset active channels if ablated
        x_raw = x_raw[self.active_channels]
        m_in_sub = m_in[self.active_channels]
        means_sub = self.input_means[self.active_channels, None, None]
        stds_sub = self.input_stds[self.active_channels, None, None]

        # Normalize inputs: zero out invalid
        x_norm = np.where(m_in_sub == 1.0, (x_raw - means_sub) / stds_sub, 0.0).astype(np.float32)

        # Normalize targets
        if self.normalize_target:
            t_means = self.target_means[:, None, None]
            t_stds = self.target_stds[:, None, None]
            y_norm = np.where(m_tgt == 1.0, (y_raw - t_means) / t_stds, 0.0).astype(np.float32)
        else:
            y_norm = np.where(m_tgt == 1.0, y_raw, 0.0).astype(np.float32)

        # 2D surface mask across all active channels
        m_in_2d = np.all(m_in_sub == 1.0, axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x_norm),
            "y": torch.from_numpy(y_norm),
            "y_raw": torch.from_numpy(y_raw),
            "mask_x": torch.from_numpy(m_in_2d),
            "mask_y": torch.from_numpy(m_tgt),
            "ocean_mask_2d": torch.from_numpy(self.ocean_mask_2d),
            "date": date_str,
            "global_idx": time_idx,
        }
