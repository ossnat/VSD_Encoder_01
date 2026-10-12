"""VF degree grid for ``trans_cortex`` (``m1`` / ``m2`` linspace from ``retino_mat``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shira_revised.src.legacy_schira import LegacySchiraParams, forward_legacy_schira


def load_retino_mat_crop(path: Path | None = None) -> np.ndarray:
    """Load ``retino_mat_20_03_22.mat`` or rebuild from ``create_retino_map`` crop."""
    if path is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / ".."
        )
        path = (
            Path(
                "/Users/ossnat/GondaResearch/VSD_FM/legacy_code/Retinotopic model Joanna/Gandalf"
            )
            / "Results Data/20_03_2022 results/retino_mat_20_03_22.mat"
        )
    if path.is_file():
        from scipy.io import loadmat

        m = loadmat(path, squeeze_me=True)
        return np.asarray(m["retino_mat"], dtype=np.float64)

    return rebuild_retino_mat_crop()


def rebuild_retino_mat_crop(
    *,
    full_rows: int = 301,
    full_cols: int = 512,
    ppd: float = 35.0,
    cal: tuple[float, float] = (5.25, -1.3),
) -> np.ndarray:
    """Match ``optimization_joanna.m`` crop (x>0, y<0 in world coords)."""
    center = (cal[0] - 4.0, cal[1] - 1.0)
    mat = np.zeros((full_rows, full_cols, 2), dtype=np.float64)
    hx = round(full_cols / 2)
    hy = round(full_rows / 2)
    mat[hy, hx, :] = center
    xc = 0
    for j in range(hx, -1, -1):
        mat[:, j, 0] = (abs(center[0]) + xc / ppd) * -1
        xc += 1
    xc = 0
    for j in range(hx, hx * 2):
        if j >= full_cols:
            break
        mat[:, j, 0] = (abs(center[0]) - xc / ppd) * -1
        xc += 1
    yc = 0
    for i in range(hy, -1, -1):
        mat[i, :, 1] = (abs(center[1]) - yc / ppd) * -1
        yc += 1
    yc = 0
    for i in range(hy, hy * 2):
        if i >= full_rows:
            break
        mat[i, :, 1] = (abs(center[1]) + yc / ppd) * -1
        yc += 1

    keep_x = np.where(mat[0, :, 0] > 0)[0]
    keep_y = np.where(mat[:, 0, 1] < 0)[0]
    cropped = mat[np.ix_(keep_y, keep_x)]
    return cropped


def trans_cortex_vf_grid(
    retino_mat: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``m1`` (cols), ``m2`` (rows) and mesh ``x_deg, y_deg`` (positive abs VF)."""
    m1 = np.linspace(
        abs(float(retino_mat[0, 0, 0])),
        abs(float(retino_mat[0, -1, 0])),
        int(n_cols),
        dtype=np.float64,
    )
    m2 = np.linspace(
        abs(float(retino_mat[0, 0, 1])),
        abs(float(retino_mat[-1, 0, 1])),
        int(n_rows),
        dtype=np.float64,
    )
    x_deg, y_deg = np.meshgrid(m1, m2, indexing="xy")
    return m1, m2, x_deg, y_deg


def catalog_deg_from_legacy_grid(
    x_deg_abs: np.ndarray,
    y_deg_abs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map legacy positive-Q1 ``(x,y)`` to catalog LRQ (x>0, y<0)."""
    return np.asarray(x_deg_abs, dtype=np.float64), -np.asarray(y_deg_abs, dtype=np.float64)


def legacy_folded_cortical_xy(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``trans_cortex.m`` display coords before ``round(*1000)``."""
    la = np.abs(np.angle(w))
    lr = np.abs(w)
    return lr * np.cos(la), lr * np.sin(la)


def legacy_trans_cortex_om(
    rgb: np.ndarray,
    x_deg: np.ndarray,
    y_deg: np.ndarray,
    params: LegacySchiraParams,
    *,
    ink_threshold: float = 200.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Schira-only legacy ``OM`` via per-pixel splat (``xcoord``/``ycoord`` indices).

    Returns OM (H×W×3) and extent ``(x0, x1, y0, y1)`` in folded cortical units.
    """
    w = forward_legacy_schira(x_deg, y_deg, params)
    cx, cy = legacy_folded_cortical_xy(w)
    ok = np.isfinite(cx) & np.isfinite(cy)
    g = rgb[:, :, 0].astype(np.float64)
    ink = g >= ink_threshold
    ink &= ok

    if not np.any(ink):
        return np.full((8, 8, 3), 130, dtype=np.uint8), (0.0, 1.0, 0.0, 1.0)

    cx_i = cx[ink]
    cy_i = cy[ink]
    colors = rgb[ink].astype(np.float64)

    # Scale to integer OM like MATLAB (×1000, min-shift).
    sx = np.round(cx_i * 1000.0)
    sy = np.round(cy_i * 1000.0)
    sx = sx - sx.min() + 1
    sy = sy - sy.min() + 1
    xi = sx.astype(int) - 1
    yi = sy.astype(int) - 1
    h = int(yi.max()) + 1
    w_ = int(xi.max()) + 1
    om = np.full((h, w_, 3), 130, dtype=np.uint8)
    om[yi, xi] = colors.astype(np.uint8)

    ext = (
        float(cx_i.min()),
        float(cx_i.max()),
        float(cy_i.min()),
        float(cy_i.max()),
    )
    return om, ext


def repo_schira_warp_on_legacy_vf(
    rgb: np.ndarray,
    x_deg_abs: np.ndarray,
    y_deg_abs: np.ndarray,
    *,
    a: float = 0.72,
    alpha: float = 1.5,
    k: float = 1.0,
    grid: int = 480,
    ink_threshold: float = 200.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Current ``src.retinotopy`` Schira-only inverse warp on the **same** VF grid."""
    from src.retinotopy.schira import SchiraParams, inverse_schira, polar_to_cartesian

    params = SchiraParams(a=a, alpha=alpha, k=k, fa_combine="power")
    x_cat, y_cat = catalog_deg_from_legacy_grid(x_deg_abs, y_deg_abs)
    g = rgb[:, :, 0].astype(np.float64)
    ink = g >= ink_threshold

    # Forward cloud for bbox (repo convention).
    from src.retinotopy.schira import cartesian_to_polar, forward_schira

    ecc, theta = cartesian_to_polar(x_cat[ink], y_cat[ink])
    w_cloud = forward_schira(ecc, theta, params)
    ok = np.isfinite(w_cloud)
    if not np.any(ok):
        return np.full((grid, grid, 3), 130, dtype=np.uint8), (0.0, 1.0, 0.0, 1.0)
    u = w_cloud.real[ok]
    v = w_cloud.imag[ok]
    pad = 0.15 * max(u.max() - u.min(), v.max() - v.min(), 0.1)
    u0, u1 = float(u.min()) - pad, float(u.max()) + pad
    v0, v1 = float(v.min()) - pad, float(v.max()) + pad

    u_c = np.linspace(u0, u1, grid, dtype=np.float64)
    v_c = np.linspace(v1, v0, grid, dtype=np.float64)
    uu, vv = np.meshgrid(u_c, v_c, indexing="xy")
    w_grid = uu + 1j * vv
    ecc_g, theta_g = inverse_schira(w_grid, params, min_real_z=0.0)
    x_inv, y_inv = polar_to_cartesian(ecc_g, theta_g)

    n_rows, n_cols = rgb.shape[:2]
    m1 = x_deg_abs[0, :]
    m2 = y_deg_abs[:, 0]
    col_f = np.interp(x_inv.ravel(), m1, np.arange(n_cols, dtype=np.float64))
    row_f = np.interp(-y_inv.ravel(), m2, np.arange(n_rows, dtype=np.float64))

    from scipy.ndimage import map_coordinates

    warped = np.full((grid, grid, 3), 130, dtype=np.uint8)
    valid = (
        np.isfinite(col_f)
        & np.isfinite(row_f)
        & (col_f >= 0)
        & (col_f < n_cols)
        & (row_f >= 0)
        & (row_f < n_rows)
        & np.isfinite(ecc_g.ravel())
    )
    coords = np.vstack([row_f, col_f])
    for ch in range(3):
        plane = map_coordinates(
            rgb[:, :, ch].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=130.0,
        ).reshape(grid, grid)
        out = np.full((grid, grid), 130.0)
        mask = valid.reshape(grid, grid)
        out[mask] = plane[mask]
        warped[:, :, ch] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return warped, (u0, u1, v0, v1)
