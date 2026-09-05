"""
Gate A2 Dataset Loader and Preprocessing Pipeline
=================================================
Loads canonical A1.5 synchronized NetCDFs and constructs PyTorch datasets
with strict temporal holdout splits and leakage-free normalization.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from src.constants import CANONICAL_SURFACE_VARIABLES, REQUIRED_DEPTHS_M


class SynchronizedOceanDataset(Dataset):
    """
    PyTorch Dataset for Gate A2 synchronized surface observations and subsurface target.
    
    Splits:
      - 'train': days 0–15 (16 days, internal training)
      - 'val': days 16–19 (4 days, internal validation for checkpoint selection)
      - 'train_full': days 0–19 (all 20 training window days)
      - 'eval': days 20–29 (10 evaluation days, strictly held out)
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "Dataset/gate_a1_pilot/synchronized",
        normalize_target: bool = True,
    ):
        super().__init__()
        self.split = split
        self.normalize_target = normalize_target
        self.data_dir = Path(data_dir)

        in_path = self.data_dir / "oceanembed_inputs_7ch_30d.nc"
        tgt_path = self.data_dir / "oceanembed_target_15depth_30d.nc"
        meta_path = self.data_dir / "synchronization_metadata.json"

        if not in_path.exists() or not tgt_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Synchronized artifacts missing in {self.data_dir}")

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Extract leakage-free normalization statistics
        norm_stats = self.metadata["leakage_free_normalization"]
        self.input_means = np.zeros(7, dtype=np.float32)
        self.input_stds = np.zeros(7, dtype=np.float32)
        for c_idx, c_name in enumerate(CANONICAL_SURFACE_VARIABLES):
            c_info = norm_stats["input_channels"][c_name]
            self.input_means[c_idx] = c_info["train_mean"]
            self.input_stds[c_idx] = c_info["train_std"]

        self.target_means = np.zeros(15, dtype=np.float32)
        self.target_stds = np.zeros(15, dtype=np.float32)
        for d_idx, depth_m in enumerate(REQUIRED_DEPTHS_M):
            d_info = norm_stats["target_depths"][f"depth_{depth_m}m"]
            self.target_means[d_idx] = d_info["train_mean"]
            self.target_stds[d_idx] = d_info["train_std"]

        # Load complete arrays into memory
        ds_in = xr.open_dataset(in_path)
        ds_tgt = xr.open_dataset(tgt_path)

        self.full_times = [str(t)[:10] for t in ds_in.time.values]
        self.full_inputs = ds_in.inputs.values.astype(np.float32)          # [30, 7, 24, 32]
        self.full_mask_in = ds_in.mask_inputs.values.astype(np.float32)    # [30, 7, 24, 32]
        self.full_targets = ds_tgt.target_thetao.values.astype(np.float32) # [30, 15, 24, 32]
        self.full_mask_tgt = ds_tgt.mask_target.values.astype(np.float32)  # [30, 15, 24, 32]
        self.ocean_mask_2d = ds_in.ocean_mask_2d.values.astype(np.float32) # [24, 32]
        self.latitudes = ds_in.latitude.values.astype(np.float32)
        self.longitudes = ds_in.longitude.values.astype(np.float32)
        self.depths = ds_tgt.depth.values.astype(np.float32)

        ds_in.close()
        ds_tgt.close()

        # Define split indices
        if split == "train":
            self.indices = list(range(0, 16))
        elif split == "val":
            self.indices = list(range(16, 20))
        elif split == "train_full":
            self.indices = list(range(0, 20))
        elif split == "eval":
            self.indices = list(range(20, 30))
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'train', 'val', 'train_full', or 'eval'.")

        self.dates = [self.full_times[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def unnormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized target array [..., 15, H, W] back to physical Celsius (°C)."""
        shape = y_norm.shape
        # Ensure depth dimension matches
        means = self.target_means.reshape((1, 15, 1, 1) if len(shape) == 4 else (15, 1, 1))
        stds = self.target_stds.reshape((1, 15, 1, 1) if len(shape) == 4 else (15, 1, 1))
        return y_norm * stds + means

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.indices[idx]
        date_str = self.full_times[global_idx]

        x_raw = self.full_inputs[global_idx]     # [7, 24, 32]
        m_x = self.full_mask_in[global_idx]      # [7, 24, 32]
        y_raw = self.full_targets[global_idx]    # [15, 24, 32]
        m_y = self.full_mask_tgt[global_idx]     # [15, 24, 32]

        # Normalized input: zero out invalid points
        x_norm = np.zeros_like(x_raw)
        for c in range(7):
            valid_mask = m_x[c] == 1.0
            x_norm[c, valid_mask] = (x_raw[c, valid_mask] - self.input_means[c]) / (self.input_stds[c] + 1e-6)

        # Normalized target if requested
        if self.normalize_target:
            y_norm = np.zeros_like(y_raw)
            for d in range(15):
                valid_mask = m_y[d] == 1.0
                y_norm[d, valid_mask] = (y_raw[d, valid_mask] - self.target_means[d]) / (self.target_stds[d] + 1e-6)
        else:
            y_norm = np.where(m_y == 1.0, y_raw, 0.0)

        return {
            "x": torch.from_numpy(x_norm),
            "y": torch.from_numpy(y_norm),
            "y_raw": torch.from_numpy(y_raw),
            "mask_x": torch.from_numpy(m_x),
            "mask_y": torch.from_numpy(m_y),
            "ocean_mask_2d": torch.from_numpy(self.ocean_mask_2d),
            "date": date_str,
            "global_idx": global_idx,
        }
