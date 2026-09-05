"""
OceanEmbed v2-Local Architecture with Surface Refinement Pathway.
================================================================
Based on empirical findings from Gate A4.1:
  - KEEP: Multimodal specialized input encoders (Thermodynamics, Sea-Level, Currents, Winds)
  - KEEP: Multi-scale spatial convolutional trunk (dilations 1, 2, 4)
  - KEEP: Residual climatology formulation (T_pred = T_clim + ΔT)
  - EXCLUDE: Global self-attention (banned; over-smooths small grids)
  - EXCLUDE: Depth FiLM conditioning (banned; standard 15ch spatial conv decoder is cleaner)
  - ADD: Lightweight Learned Surface Refinement Pathway specifically targeting
         near-surface depths (0m, 5m, 10m, 20m) while leaving depths 30m-1000m
         completely anchored to the main spatial trunk.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import REQUIRED_DEPTHS_M
from src.gate_a4.model_v1 import (
    MultimodalBranchEncoder,
    MultiScaleSpatialEncoder,
    MaskedMSELoss,
)


class SurfaceRefinementPathway(nn.Module):
    """
    Lightweight learned upper-ocean residual correction module.
    Fuses raw surface thermodynamic & sea-level inputs (SST, SSS, SSH) with
    shared latent spatial representations to predict localized residual corrections
    for the upper 4 canonical depths (0m, 5m, 10m, 20m).

    Tensor Flow:
      Inputs:
        surface_inputs: [B, C_surf, H, W] (e.g., channels 0..2: SST, SSS, SSH)
        latent_features: [B, C_latent, H, W] (from multi-scale spatial trunk)
      Operations:
        Cat -> Conv3x3(51->24) -> BatchNorm -> GELU -> Conv3x3(24->4)
      Output:
        ΔT_upper: [B, 4, H, W]
    """

    def __init__(
        self,
        surface_in_channels: int = 3,
        latent_dim: int = 48,
        hidden_dim: int = 24,
        upper_depths_count: int = 4,
    ):
        super().__init__()
        self.upper_depths_count = upper_depths_count
        in_dim = surface_in_channels + latent_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, upper_depths_count, kernel_size=3, padding=1),
        )

        # Initialize final projection with small weights and zero bias
        # so at step 0 the model operates near the established v1-Local baseline
        final_conv = self.net[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-3)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(
        self,
        surface_inputs: torch.Tensor,
        latent_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        surface_inputs: [B, 3, H, W] (SST, SSS, SSH)
        latent_features: [B, 48, H, W]
        Returns:
          delta_upper: [B, 4, H, W]
        """
        fused = torch.cat([surface_inputs, latent_features], dim=1)
        return self.net(fused)


class OceanEmbedV2Local(nn.Module):
    """
    OceanEmbed v2-Local Architecture.

    Pipeline:
      1. MultimodalBranchEncoder: [B, 7, 24, 32] -> [B, 48, 24, 32]
      2. MultiScaleSpatialEncoder: [B, 48, 24, 32] -> [B, 48, 24, 32] (dilations 1, 2, 4)
      3. Main Spatial Decoder: [B, 48, 24, 32] -> [B, 15, 24, 32] (NO FiLM, NO Attention)
      4. SurfaceRefinementPathway:
         Takes X[:, :3] + Latent Z -> predicts ΔT_upper for depths 0..3 (0m, 5m, 10m, 20m)
         Depths 4..14 (30m to 1000m, including the 50-200m thermocline) are strictly untouched.
      5. Climatological Residual Addition: T_pred = T_clim + ΔT_refined
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_depths: int = 15,
        branch_dim: int = 16,
        latent_dim: int = 48,
        use_residual: bool = True,
        use_surface_refinement: bool = True,
        upper_depths_count: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths
        self.latent_dim = latent_dim
        self.use_residual = use_residual
        self.use_surface_refinement = use_surface_refinement
        self.upper_depths_count = upper_depths_count

        # 1. Multimodal Specialized Input Branches
        self.branch_encoder = MultimodalBranchEncoder(
            branch_dim=branch_dim,
            fused_dim=latent_dim,
        )

        # 2. Multi-Scale Spatial Convolutional Trunk
        self.spatial_encoder = MultiScaleSpatialEncoder(dim=latent_dim)

        # 3. Main Spatial Output Decoder (Standard 15-channel spatial conv, NO FiLM)
        self.main_decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, num_depths, kernel_size=1),
        )

        # 4. Learned Surface Refinement Pathway (for depths 0..upper_depths_count-1)
        if use_surface_refinement:
            # Inputs: Channels 0, 1, 2 (SST, SSS, SSH) + latent_dim (48) -> 51 channels
            self.surface_refinement = SurfaceRefinementPathway(
                surface_in_channels=3,
                latent_dim=latent_dim,
                hidden_dim=24,
                upper_depths_count=upper_depths_count,
            )
        else:
            self.surface_refinement = None

    def forward(
        self,
        x: torch.Tensor,
        climatology: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, 7, 24, 32]
        climatology: Optional [15, 24, 32] or [B, 15, 24, 32]
        Returns:
          y_pred: [B, 15, 24, 32]
        """
        B, C, H, W = x.shape

        # Step 1: Multimodal feature extraction
        fused = self.branch_encoder(x)  # [B, 48, H, W]

        # Step 2: Multi-scale spatial representation
        latent_z = self.spatial_encoder(fused)  # [B, 48, H, W]

        # Step 3: Main spatial decoding for all 15 depths
        anomaly_main = self.main_decoder(latent_z)  # [B, 15, H, W]

        # Step 4: Apply surface refinement pathway if active
        if self.use_surface_refinement and self.surface_refinement is not None:
            surface_inputs = x[:, 0:3, :, :]  # SST, SSS, SSH
            delta_upper = self.surface_refinement(surface_inputs, latent_z)  # [B, 4, H, W]

            # Refine upper depths, leave lower depths (30m - 1000m) completely untouched
            anomaly_refined = torch.cat(
                [
                    anomaly_main[:, :self.upper_depths_count, :, :] + delta_upper,
                    anomaly_main[:, self.upper_depths_count:, :, :],
                ],
                dim=1,
            )
        else:
            anomaly_refined = anomaly_main

        # Step 5: Residual formulation over training climatology
        if self.use_residual and climatology is not None:
            if climatology.dim() == 3:
                climatology = climatology.unsqueeze(0)
            return climatology + anomaly_refined

        return anomaly_refined

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
