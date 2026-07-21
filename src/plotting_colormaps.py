"""Shared presentation colormaps."""

from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

VSD_CMAP = "mapgeog"


def register_mapgeog() -> None:
    """Register a blue→green→yellow→red→magenta mapgeog colormap."""
    if VSD_CMAP in matplotlib.colormaps:
        return
    cmap = LinearSegmentedColormap.from_list(
        VSD_CMAP,
        [
            (0.00, (0.0, 0.0, 0.15)),
            (0.125, (0.0, 0.0, 1.0)),
            (0.375, (0.0, 1.0, 0.0)),
            (0.625, (1.0, 1.0, 0.0)),
            (0.875, (1.0, 0.0, 0.0)),
            (1.00, (1.0, 1.0, 1.0)),
        ],
        N=256,
    )
    matplotlib.colormaps.register(cmap)


register_mapgeog()
