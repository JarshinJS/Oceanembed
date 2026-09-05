"""
OceanEmbed — package init
=========================
Exports canonical constants and configurations aligned with official SIH26066.
"""

from src.constants import (
    REQUIRED_DEPTHS_M,
    OUTPUT_DEPTHS,
    INPUT_CHANNELS,
    CANONICAL_SURFACE_VARIABLES,
    CHANNEL_NAME_MAP,
    OFFICIAL_PS_DOMAIN,
    MVP_PILOT_DOMAIN,
    DATA_SOURCE_MODES,
)

__all__ = [
    "REQUIRED_DEPTHS_M",
    "OUTPUT_DEPTHS",
    "INPUT_CHANNELS",
    "CANONICAL_SURFACE_VARIABLES",
    "CHANNEL_NAME_MAP",
    "OFFICIAL_PS_DOMAIN",
    "MVP_PILOT_DOMAIN",
    "DATA_SOURCE_MODES",
]
