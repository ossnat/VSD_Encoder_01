"""Matplotlib helpers: VF panels with degree tick labels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def imshow_vf_degrees(
    ax: plt.Axes,
    rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    title: str = "",
    tick_deg: float = 1.0,
) -> None:
    """Show legacy 301×512 stimulus with **x/y in visual degrees**.

    Uses the same ``trans_cortex`` grid: ``m1[col]``, ``m2[row]`` (positive abs VF).
    Axis labels match catalog-style ticks: x = eccentricity right; y = upward
    (``y_catalog = -m2``).
    """
    x0, x1 = float(m1[0]), float(m1[-1])
    y0, y1 = float(m2[0]), float(m2[-1])
    # imshow row 0 at top → larger m2 at bottom; label y catalog (+up) on left.
    ax.imshow(
        rgb,
        extent=[x0, x1, -y1, -y0],
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (deg VF)")
    ax.set_ylabel("y (deg VF, + up)")
    ax.xaxis.set_major_locator(MultipleLocator(tick_deg))
    ax.yaxis.set_major_locator(MultipleLocator(tick_deg))
    ax.set_title(title, fontsize=11)
    ax.grid(False)


def imshow_cortical(
    ax: plt.Axes,
    rgb: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    x0, x1, y0, y1 = extent
    ax.imshow(
        rgb,
        extent=[x0, x1, y0, y1],
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
