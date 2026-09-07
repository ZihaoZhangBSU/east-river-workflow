"""Shared constants and plotting conventions."""

from __future__ import annotations

MODEL_COLORS = {
    "SNOTEL": "black",
    "iSnobal": "#1F77B4",
    "Noah-MP": "#D62728",
    "USGS": "black",
}

ELEVATION_BANDS = {
    "Lower": (None, 2896.0),
    "Middle": (2896.0, 3353.0),
    "Upper": (3353.0, None),
    "All": (None, None),
}

CFS_TO_M3_S = 0.028316846592
SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0
MM_TO_M = 1.0e-3
MJ_PER_J = 1.0e-6
