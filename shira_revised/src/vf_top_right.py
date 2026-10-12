"""VF coordinates with fixation at the top-right; 1 image pixel grid = 1/ppd deg."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks


def detect_ppd_from_grid(
    gray: np.ndarray,
    *,
    prominence: float = 12.0,
    min_distance: int = 8,
) -> float | None:
    """Estimate pixels/deg from dark grid ticks along HM and VM from top-right."""
    h, w = gray.shape
    hm = gray[0, ::-1].astype(np.float64)
    peaks_h, _ = find_peaks(-hm, distance=min_distance, prominence=prominence)
    vm = gray[:, w - 1].astype(np.float64)
    peaks_v, _ = find_peaks(-vm, distance=min_distance, prominence=prominence)
    spacings = []
    if len(peaks_h) >= 2:
        d = np.diff(peaks_h[: min(12, len(peaks_h))])
        d = d[d > 3]
        if d.size:
            spacings.append(float(np.median(d)))
    if len(peaks_v) >= 2:
        d = np.diff(peaks_v[: min(12, len(peaks_v))])
        d = d[d > 3]
        if d.size:
            spacings.append(float(np.median(d)))
    if not spacings:
        return None
    return float(np.median(spacings))


def vf_grid_top_right(
    n_rows: int,
    n_cols: int,
    ppd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deg fields for legacy Schira (positive abs x,y fed to ``trans_cortex``).

    Origin (0, 0) at pixel (row=0, col=n_cols-1). +x left, +y down (rows).
    """
    cols = np.arange(n_cols, dtype=np.float64)
    rows = np.arange(n_rows, dtype=np.float64)
    x_deg = (n_cols - 1 - cols) / ppd
    y_deg = rows / ppd
    x_deg, y_deg = np.meshgrid(x_deg, y_deg, indexing="xy")
    return x_deg[0, :], y_deg[:, 0], x_deg, y_deg


def imshow_input_vf_top_right(
    ax,
    rgb: np.ndarray,
    ppd: float,
    *,
    title: str = "",
    tick_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Input image with x/y in visual degrees (fixation top-right)."""
    n_rows, n_cols = rgb.shape[:2]
    m1, m2, _, _ = vf_grid_top_right(n_rows, n_cols, ppd)
    x0, x1 = float(np.min(m1)), float(np.max(m1))
    y_down_max = float(np.max(m2))
    ax.imshow(
        rgb,
        extent=[x0, x1, -y_down_max, -float(m2[0])],
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xlabel("x (deg VF, HM from fixation)")
    ax.set_ylabel("y (deg VF, + up; VM down is negative)")
    ax.xaxis.set_major_locator(MultipleLocator(tick_deg))
    ax.yaxis.set_major_locator(MultipleLocator(tick_deg))
    ax.set_title(title, fontsize=11)
    return m1, m2


def folded_xy_at_vf(
    x_deg: float,
    y_deg: float,
    params,
) -> tuple[float, float]:
    from shira_revised.src.legacy_schira import forward_legacy_schira
    from shira_revised.src.legacy_warp import legacy_folded_cortical_xy

    w = forward_legacy_schira(x_deg, y_deg, params)
    cx, cy = legacy_folded_cortical_xy(w)
    return float(cx), float(cy)


def set_cortical_axes_vf_degrees(
    ax,
    params,
    *,
    x_max: float,
    y_max: float,
    y_ref: float = 4.0,
    x_ref: float = 1.5,
    tick_deg: float = 1.0,
) -> None:
    """Label cortical plot axes with VF degrees (same tick values as input).

    x tick marks: forward map ``(x, y_ref)`` → cortical x.
    y tick marks: forward map ``(x_ref, y)`` → cortical y (catalog +up labels).
    """
    xticks, xlabels = [], []
    x = 0.0
    while x <= x_max + 1e-9:
        cx, _ = folded_xy_at_vf(x, y_ref, params)
        if np.isfinite(cx):
            xticks.append(cx)
            xlabels.append(f"{int(x)}" if abs(x - round(x)) < 1e-9 else f"{x:g}")
        x += tick_deg

    yticks, ylabels = [], []
    y = 0.0
    while y <= y_max + 1e-9:
        _, cy = folded_xy_at_vf(x_ref, y, params)
        if np.isfinite(cy):
            yticks.append(cy)
            ylabels.append(f"{int(-y)}" if abs(y - round(y)) < 1e-9 else f"{-y:g}")
        y += tick_deg

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("x (deg VF)")
    ax.set_ylabel("y (deg VF, + up)")
