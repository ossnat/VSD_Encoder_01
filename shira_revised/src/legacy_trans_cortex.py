"""Faithful port of Amit ``trans_cortex.m`` (Schira model=2 monopole, polygon OM fill)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from shira_revised.src.legacy_schira import LegacySchiraParams, forward_legacy_schira


@dataclass(frozen=True)
class TransCortexResult:
    """Outputs matching MATLAB ``trans_cortex`` (before ``trans_and_registration``)."""

    om: np.ndarray  # uint8 (H, W, 3), unfilled = 0
    xcoord: np.ndarray  # float (n_rows, n_cols), 1-based OM column index
    ycoord: np.ndarray  # float (n_rows, n_cols), 1-based OM row index
    m1: np.ndarray  # deg per col
    m2: np.ndarray  # deg per row


def m1_m2_from_retino_mat(
    retino_mat: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same linspace as ``trans_cortex.m`` (abs corners → full ``IM`` size)."""
    m1 = np.linspace(
        abs(float(retino_mat[0, 0, 0])),
        abs(float(retino_mat[0, -1, 0])),
        n_cols,
        dtype=np.float64,
    )
    m2 = np.linspace(
        abs(float(retino_mat[0, 0, 1])),
        abs(float(retino_mat[-1, 0, 1])),
        n_rows,
        dtype=np.float64,
    )
    return m1, m2


def m1_m2_top_right_origin(
    n_rows: int,
    n_cols: int,
    ppd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """VF deg grid: (0,0) at top-right; +x left, +y down (legacy abs inputs)."""
    cols = np.arange(n_cols, dtype=np.float64)
    rows = np.arange(n_rows, dtype=np.float64)
    m1 = (n_cols - 1 - cols) / ppd
    m2 = rows / ppd
    return m1, m2


def _cortical_indices_from_w(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x1``, ``y1`` with global min-shift (MATLAB ``trans_cortex``)."""
    log_angle = np.abs(np.angle(w))
    log_rad = np.abs(w)
    x = log_rad * np.cos(log_angle)
    y = log_rad * np.sin(log_angle)
    xr = np.round(x * 1000.0)
    yr = np.round(y * 1000.0)
    ok = np.isfinite(xr) & np.isfinite(yr)
    if not np.any(ok):
        return xr, yr
    xmin = np.min(xr[ok])
    ymin = np.min(yr[ok])
    x1 = np.full_like(xr, np.nan)
    y1 = np.full_like(yr, np.nan)
    x1[ok] = xr[ok] - xmin + 1.0
    y1[ok] = yr[ok] - ymin + 1.0
    return x1, y1


def trans_cortex(
    im_rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    params: LegacySchiraParams,
    *,
    quick_mode: bool = False,
    bg: int = 0,
) -> TransCortexResult:
    """Run legacy Schira forward map + polygon fill into ``OM``.

    Parameters
    ----------
    im_rgb
        Stimulus ``(n_rows, n_cols, 3)`` uint8 or float.
    m1, m2
        VF degree axes (length ``n_cols`` / ``n_rows``), as in ``trans_cortex``.
    """
    im = np.asarray(im_rgb)
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {im.shape}")
    n_rows, n_cols = im.shape[:2]
    if len(m1) != n_cols or len(m2) != n_rows:
        raise ValueError("m1/m2 length must match image cols/rows")

    # Build per-pixel VF coords (same nested loop order as MATLAB).
    mat_x = np.zeros((n_rows, n_cols), dtype=np.float64)
    mat_y = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            mat_x[i, j] = m1[j]
            mat_y[i, j] = m2[i]

    w = forward_legacy_schira(mat_x, mat_y, params)
    x1, y1 = _cortical_indices_from_w(w)

    # xcoord( row, col ) = x1 at that pixel; MATLAB reshape(...)' .
    xcoord = x1.astype(np.float64)
    ycoord = y1.astype(np.float64)

    if quick_mode:
        om = np.zeros((1, 1, 3), dtype=np.uint8)
        return TransCortexResult(om=om, xcoord=xcoord, ycoord=ycoord, m1=m1, m2=m2)

    ok = np.isfinite(xcoord) & np.isfinite(ycoord)
    if not np.any(ok):
        om = np.full((1, 1, 3), bg, dtype=np.uint8)
        return TransCortexResult(om=om, xcoord=xcoord, ycoord=ycoord, m1=m1, m2=m2)

    om_h = int(np.nanmax(ycoord))
    om_w = int(np.nanmax(xcoord))
    om = np.full((om_h, om_w, 3), bg, dtype=np.uint8)

    im_u8 = im.astype(np.uint8) if im.dtype == np.uint8 else np.clip(im, 0, 255).astype(np.uint8)

    # Polygon fill: MATLAB outer j=col, inner k=row; IM(k,j,:).
    for j in range(n_cols - 1):
        for i in range(n_rows - 1):
            poly_x = np.array(
                [
                    xcoord[i, j],
                    xcoord[i + 1, j],
                    xcoord[i + 1, j + 1],
                    xcoord[i, j + 1],
                ],
                dtype=np.float64,
            )
            poly_y = np.array(
                [
                    ycoord[i, j],
                    ycoord[i + 1, j],
                    ycoord[i + 1, j + 1],
                    ycoord[i, j + 1],
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(poly_x)) or not np.all(np.isfinite(poly_y)):
                continue
            b1 = int(np.max(poly_y))
            b2 = int(np.max(poly_x))
            b3 = int(np.min(poly_y))
            b4 = int(np.min(poly_x))
            if b1 < b3 or b2 < b4:
                continue
            hh = b1 - b3 + 1
            ww = b2 - b4 + 1
            if hh <= 0 or ww <= 0 or hh * ww > 2_000_000:
                continue
            xs = np.arange(1, ww + 1, dtype=np.float64)
            ys = np.arange(1, hh + 1, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys, indexing="xy")
            px = poly_x - b4 + 1.0
            py = poly_y - b3 + 1.0
            path = Path(np.column_stack([px, py]))
            pts = np.column_stack([xx.ravel(), yy.ravel()])
            inside = path.contains_points(pts, radius=0.5).reshape(hh, ww)
            if not np.any(inside):
                continue
            color = im_u8[i, j]
            # OM row = bounds(3)-1 + roi_y, col = bounds(4)-1 + roi_x
            rows_om = (b3 - 1) + np.where(inside)[0]
            cols_om = (b4 - 1) + np.where(inside)[1]
            om[rows_om, cols_om] = color

    return TransCortexResult(om=om, xcoord=xcoord, ycoord=ycoord, m1=m1, m2=m2)
