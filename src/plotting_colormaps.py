"""Shared presentation colormaps."""

from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

VSD_CMAP = "mapgeog"
VSD_CMAP_GRAY = "mapgeog_gray"

# Same knot positions as mapgeog (dark navy → white).
_MAPGEOG_STOPS = (0.00, 0.125, 0.375, 0.625, 0.875, 1.00)


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
    # Masked / out-of-ROI predictions use NaN; render like the dark low end.
    cmap = cmap.with_extremes(bad=(0.0, 0.0, 0.15))
    matplotlib.colormaps.register(cmap)


def register_mapgeog_gray() -> None:
    """Register a monotonic grayscale analog of mapgeog (dark → white).

    Knots match ``mapgeog``. This is **not** a luminance flatten of the
    rainbow (that LUT is non-monotonic at red).
    """
    if VSD_CMAP_GRAY in matplotlib.colormaps:
        return
    cmap = LinearSegmentedColormap.from_list(
        VSD_CMAP_GRAY,
        [(t, (t, t, t)) for t in _MAPGEOG_STOPS],
        N=256,
    )
    cmap = cmap.with_extremes(bad=(0.0, 0.0, 0.0))
    matplotlib.colormaps.register(cmap)


register_mapgeog()
register_mapgeog_gray()
