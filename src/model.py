"""
OceanEmbed — Model Architecture (SIH26066 Aligned)
===================================================
SatEmbedNet: A convolutional encoder producing a learned "satellite embedding"
(latent representation of 7 surface input fields) followed by a regression
head that maps the embedding to a 15-level subsurface temperature profile.

Canonical Tensor Specifications:
--------------------------------
    Input Tensor:
        Shape: [B, 7, H, W]
        Channels (Canonical Order):
          Channel 0: SST (Sea Surface Temperature)
          Channel 1: SSS (Sea Surface Salinity)
          Channel 2: SLA / SSH (Sea Level Anomaly / Sea Surface Height)
          Channel 3: Surface Current U (Zonal velocity)
          Channel 4: Surface Current V (Meridional velocity)
          Channel 5: Surface Wind U (10m Zonal wind vector)
          Channel 6: Surface Wind V (10m Meridional wind vector)

    Output Tensor:
        Profile Regression Mode: [B, 15]
        Dense 3D Field Mode:     [B, 15, H, W]
        Target Depths (15 Canonical Levels):
          [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m

Architecture:
-------------
    Input  [B, 7, H, W]
      └─ ConvBlock × N  →  [B, C_n, H', W']
      └─ AdaptiveAvgPool → [B, C_n, 1, 1]
      └─ Flatten        → [B, C_n]
      └─ Linear          → embedding [B, embedding_dim]
      └─ [embedding head — extractable]
      └─ MLPRegressor    → T(z) [B, 15]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List

from src.constants import INPUT_CHANNELS, OUTPUT_DEPTHS


class ConvBlock(nn.Module):
    """Conv → BatchNorm → Activation → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float, act: str = "gelu"):
        super().__init__()
        pad = kernel // 2
        activation = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[act]
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SatEmbedNet(nn.Module):
    """CNN embedding encoder + temperature regression head.

    Accepts canonical 7-channel input [B, 7, H, W] and predicts
    15-level subsurface temperature profile [B, 15].
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg.get("model", {})
        surface_vars = cfg.get("surface_variables")
        n_channels = len(surface_vars) if surface_vars else mc.get("in_channels", INPUT_CHANNELS)

        target_depths = cfg.get("target", {}).get("depths")
        n_depths = len(target_depths) if target_depths else OUTPUT_DEPTHS

        embed_dim = mc.get("embedding_dim", 128)

        # --- encoder ---
        blocks: List[nn.Module] = []
        in_ch = n_channels
        for out_ch in mc.get("conv_channels", [32, 64, 128]):
            blocks.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    mc.get("conv_kernel", 3),
                    mc.get("dropout", 0.1),
                    mc.get("activation", "gelu"),
                )
            )
            blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # latent projection
        self.embed_proj = nn.Linear(in_ch, embed_dim)

        # --- regression head ---
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(mc.get("dropout", 0.1)),
            nn.Linear(embed_dim, n_depths),
        )

    # -- forward ----------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return dict with 'embedding' [B, embed_dim], 'temp_profile' [B, n_depths]."""
        feat = self.encoder(x)
        feat = self.pool(feat)
        feat = self.flatten(feat)
        embedding = self.embed_proj(feat)
        temp_profile = self.regressor(embedding)
        return {"embedding": embedding, "temp_profile": temp_profile}

    # -- convenience ------------------------------------------------
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the latent vector (for downstream tasks)."""
        feat = self.encoder(x)
        feat = self.pool(feat)
        feat = self.flatten(feat)
        return self.embed_proj(feat)
