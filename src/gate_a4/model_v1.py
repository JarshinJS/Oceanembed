"""
Gate A4.0: OceanEmbed v1 Model Architecture
===========================================
A lightweight multimodal encoder + multi-scale spatial context + depth-conditioned decoder
for reconstructing 15-depth subsurface ocean temperatures from 7 surface satellite observations.

Architecture Components:
  1. MultimodalBranchEncoder: Specialized conv branches for Thermodynamics, Sea-Level, Dynamics, Atmosphere.
  2. MultiScaleSpatialEncoder: Preserves 24x32 grid with local and mesoscale dilated receptive fields.
  3. LightweightGlobalContext: Single-head non-local spatial self-attention across HW=768 positions.
  4. DepthConditionedDecoder: Explicit 15-depth learnable + physical embeddings with shared spatial decoding.
  5. Residual Formulation: Prediction = T_climatology + learned anomaly ΔT.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import REQUIRED_DEPTHS_M


class MultimodalBranchEncoder(nn.Module):
    """
    Branch 1: Thermodynamics (SST, SSS)           -> 2 channels
    Branch 2: Sea-Level (SSH/ADT)                 -> 1 channel
    Branch 3: Ocean Dynamics (OSCAR U, V current) -> 2 channels
    Branch 4: Atmospheric (CCMP U, V wind)        -> 2 channels
    """

    def __init__(self, branch_dim: int = 16, fused_dim: int = 48):
        super().__init__()
        self.thermo_branch = nn.Sequential(
            nn.Conv2d(2, branch_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )
        self.sealevel_branch = nn.Sequential(
            nn.Conv2d(1, branch_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )
        self.dynamics_branch = nn.Sequential(
            nn.Conv2d(2, branch_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )
        self.atmosphere_branch = nn.Sequential(
            nn.Conv2d(2, branch_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_dim),
            nn.GELU(),
        )

        total_branch_channels = branch_dim * 4  # 64
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_branch_channels, fused_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 7, H, W]
        channel order:
          0: sst, 1: sss
          2: ssh_adt
          3: u_current, 4: v_current
          5: u_wind, 6: v_wind
        """
        thermo_feat = self.thermo_branch(x[:, 0:2])
        sealevel_feat = self.sealevel_branch(x[:, 2:3])
        dynamics_feat = self.dynamics_branch(x[:, 3:5])
        atmos_feat = self.atmosphere_branch(x[:, 5:7])

        concat_feat = torch.cat([thermo_feat, sealevel_feat, dynamics_feat, atmos_feat], dim=1)
        fused = self.fusion_conv(concat_feat)  # [B, fused_dim, H, W]
        return fused


class DilatedResidualBlock(nn.Module):
    """Multi-scale spatial block with parallel dilation rates 1 and 2 to capture mesoscale eddies."""

    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        self.conv_d1 = nn.Conv2d(dim, half_dim, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv_d2 = nn.Conv2d(dim, half_dim, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv_d1(x)
        h2 = self.conv_d2(x)
        h = torch.cat([h1, h2], dim=1)
        h = self.gelu(self.bn(h))
        h = self.proj(h)
        return x + h


class MultiScaleSpatialEncoder(nn.Module):
    """
    Compact multi-scale spatial encoder preserving 24x32 grid resolution.
    Captures both local thermal boundaries and mesoscale eddies (50-150 km).
    """

    def __init__(self, dim: int = 48):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.stage2 = DilatedResidualBlock(dim)
        self.stage3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = x + self.stage1(x)
        h2 = self.stage2(h1)
        h3 = h2 + self.stage3(h2)
        return h3


class LightweightGlobalContext(nn.Module):
    """
    Single-head non-local spatial self-attention block across HW=768 grid cells.
    Allows capturing basin-wide teleconnections (e.g. coastal Kelvin waves to interior eddies).
    With HW=768, attention map is 768x768 (<0.6M float elements), avoiding transformer bloat.
    """

    def __init__(self, dim: int = 48, key_dim: int = 24):
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim
        self.q_conv = nn.Conv2d(dim, key_dim, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(dim, key_dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.scale = 1.0 / math.sqrt(key_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W  # 768

        Q = self.q_conv(x).view(B, self.key_dim, N).permute(0, 2, 1)  # [B, N, key_dim]
        K = self.k_conv(x).view(B, self.key_dim, N)                   # [B, key_dim, N]
        V = self.v_conv(x).view(B, C, N)                              # [B, C, N]

        attn = torch.bmm(Q, K) * self.scale                           # [B, N, N]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)   # [B, C, H, W]
        return x + self.gamma * out


class DepthConditionedDecoder(nn.Module):
    """
    Depth-conditioned spatial decoder.
    Each of the 15 canonical depths has an explicit learnable embedding modulated
    by normalized physical depth values.
    Uses a shared spatial convolution head to ensure vertical representation coupling.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        num_depths: int = 15,
        depth_values: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_depths = num_depths
        if depth_values is None:
            depth_values = REQUIRED_DEPTHS_M
        self.register_buffer(
            "depth_values", torch.tensor(depth_values, dtype=torch.float32)
        )

        # 1. Discrete learnable embedding for each depth index
        self.depth_embed = nn.Embedding(num_depths, latent_dim)

        # 2. Continuous physical depth embedding (log1p depth projection)
        self.phys_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # 3. Depth-conditioning fusion MLP
        self.depth_fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
        )

        # 4. Spatial feature refinement
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )

        # 5. Depth-conditioned filter generator: produces [15, C] spatial decoding weights
        self.depth_filter_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.depth_bias_proj = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim, H, W]
        Returns: [B, 15, H, W]
        """
        B, C, H, W = z.shape
        device = z.device

        # 1. Spatial feature refinement
        z_refined = self.spatial_refine(z)  # [B, C, H, W]

        # 2. Explicit Depth Embeddings: [15, C]
        depth_indices = torch.arange(self.num_depths, device=device)
        discrete_emb = self.depth_embed(depth_indices)
        log_depths = torch.log1p(self.depth_values).unsqueeze(-1).to(device)
        phys_emb = self.phys_proj(log_depths)
        depth_features = self.depth_fusion(torch.cat([discrete_emb, phys_emb], dim=-1))  # [15, C]

        # 3. Generate depth-conditioned spatial decoding filters: [15, C] and biases: [1, 15, 1, 1]
        depth_filters = self.depth_filter_proj(depth_features)
        depth_biases = self.depth_bias_proj(depth_features).view(1, self.num_depths, 1, 1)

        # 4. Decode via depth-conditioned projection
        out = torch.einsum("dc,bchw->bdhw", depth_filters, z_refined) + depth_biases
        return out


class OceanEmbedV1(nn.Module):
    """
    OceanEmbed v1 Architecture with Controlled Ablation Support.
    Pipeline:
      1. MultimodalBranchEncoder (or Monolithic Conv if ablated)
      2. MultiScaleSpatialEncoder: [B, 48, 24, 32] -> [B, 48, 24, 32]
      3. LightweightGlobalContext (or Identity if ablated)
      4. DepthConditionedDecoder (or Standard 15ch Conv if ablated)
      5. Residual Formulation (or Direct Temperature if ablated)
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_depths: int = 15,
        branch_dim: int = 16,
        latent_dim: int = 48,
        use_residual: bool = True,
        use_depth_conditioning: bool = True,
        use_global_context: bool = True,
        use_multimodal_branches: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_depth_conditioning = use_depth_conditioning
        self.use_global_context = use_global_context
        self.use_multimodal_branches = use_multimodal_branches
        self.latent_dim = latent_dim
        self.num_depths = num_depths

        # Module 1: Multimodal Branch Encoder vs. Monolithic Single-Conv
        if use_multimodal_branches:
            self.branch_encoder = MultimodalBranchEncoder(branch_dim=branch_dim, fused_dim=latent_dim)
        else:
            self.branch_encoder = nn.Sequential(
                nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(latent_dim),
                nn.GELU(),
            )

        # Module 2: Multi-Scale Spatial Encoder
        self.spatial_encoder = MultiScaleSpatialEncoder(dim=latent_dim)

        # Module 3: Global Context vs. Identity
        if use_global_context:
            self.global_context = LightweightGlobalContext(dim=latent_dim, key_dim=latent_dim // 2)
        else:
            self.global_context = nn.Identity()

        # Module 4: Depth-Conditioned Decoder vs. Unconditioned 15ch Conv
        if use_depth_conditioning:
            self.depth_decoder = DepthConditionedDecoder(latent_dim=latent_dim, num_depths=num_depths)
        else:
            self.depth_decoder = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(latent_dim),
                nn.GELU(),
                nn.Conv2d(latent_dim, num_depths, kernel_size=1),
            )

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
        # 1. Multimodal branch encoding (or monolithic conv)
        fused = self.branch_encoder(x)

        # 2. Multi-scale spatial feature encoding
        spatial = self.spatial_encoder(fused)

        # 3. Global context (or identity)
        context = self.global_context(spatial)

        # 4. Depth-conditioned decoding (or standard 15ch conv)
        anomaly = self.depth_decoder(context)  # [B, 15, 24, 32]

        # 5. Residual formulation
        if self.use_residual and climatology is not None:
            if climatology.dim() == 3:
                climatology = climatology.unsqueeze(0)
            return climatology + anomaly
        return anomaly

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskedMSELoss(nn.Module):
    """
    Mask-aware Mean Squared Error loss.
    Computes loss strictly on valid target ocean pixels (mask == 1.0).
    Invalid pixels (land, bathymetry, or missing observations) contribute zero loss.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        depth_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred:   [B, 15, H, W]
        target: [B, 15, H, W]
        mask:   [B, 15, H, W] in {0.0, 1.0}
        depth_weights: Optional [15]
        """
        mask = (mask > 0.5).float()
        clean_target = torch.where(mask > 0.5, target, pred)
        diff = pred - clean_target
        squared_err = (diff ** 2) * mask

        if depth_weights is not None:
            w = depth_weights.view(1, -1, 1, 1).to(pred.device)
            squared_err = squared_err * w

        valid_sum = mask.sum()
        if valid_sum < 1.0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return squared_err.sum() / (valid_sum + self.eps)
