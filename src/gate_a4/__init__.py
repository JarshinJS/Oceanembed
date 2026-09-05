"""
Gate A4.0 — OceanEmbed v1 Architecture
======================================
"""

from .model_v1 import (
    OceanEmbedV1,
    MultimodalBranchEncoder,
    MultiScaleSpatialEncoder,
    LightweightGlobalContext,
    DepthConditionedDecoder,
    MaskedMSELoss,
)

__all__ = [
    "OceanEmbedV1",
    "MultimodalBranchEncoder",
    "MultiScaleSpatialEncoder",
    "LightweightGlobalContext",
    "DepthConditionedDecoder",
    "MaskedMSELoss",
]
