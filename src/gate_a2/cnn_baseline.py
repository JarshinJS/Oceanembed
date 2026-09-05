"""
Gate A2 Baseline 1: Simple Spatial CNN Architecture & Mask-Aware Loss
=====================================================================
A deliberately small, 4-layer spatial CNN directly mapping the 7 surface
channels [B, 7, 24, 32] to the 15-depth subsurface profile field [B, 15, 24, 32].

No transformers, attention mechanisms, diffusion models, or pretrained weights.
"""

import torch
import torch.nn as nn


class SimpleSpatialCNN(nn.Module):
    """
    Lightweight 4-layer spatial convolutional neural network.
    Maintains constant spatial resolution (24 x 32) without pooling or downsampling.
    
    Structure:
      Conv2d(7 -> 32, 3x3, pad 1) + BatchNorm + ReLU
      Conv2d(32 -> 64, 3x3, pad 1) + BatchNorm + ReLU
      Conv2d(64 -> 32, 3x3, pad 1) + BatchNorm + ReLU
      Conv2d(32 -> 15, 1x1)
    """

    def __init__(self, in_channels: int = 7, out_channels: int = 15, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: [B, 7, 24, 32]
        Output: [B, 15, 24, 32]
        """
        return self.net(x)


class MaskedMSELoss(nn.Module):
    """
    Mask-aware Mean Squared Error loss.
    Computes loss strictly on valid target ocean pixels (mask == 1.0).
    Invalid pixels (land, bathymetry, or missing data) contribute zero loss.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred:   [B, 15, H, W]
        target: [B, 15, H, W]
        mask:   [B, 15, H, W] in {0.0, 1.0}
        """
        # Ensure mask is binary float
        mask = (mask > 0.5).float()
        
        # Zero out invalid predictions/targets before difference to avoid NaN leakage
        clean_target = torch.where(mask > 0.5, target, pred)
        diff = pred - clean_target
        squared_err = (diff ** 2) * mask
        
        valid_sum = mask.sum()
        if valid_sum < 1.0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        loss = squared_err.sum() / (valid_sum + self.eps)
        return loss
