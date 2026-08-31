"""Fit camera affine from Schira ``w`` ↔ VSD pixel landmark pairs.

Schira ``(a, α, k, fa)`` are held fixed. Landmarks are pairs
``(u, v) = (Re(w), Im(w))`` after forward Schira and ``(row, col)`` on the
VSD map. The similarity (origin, scale, rotation, optional flips) is solved
in closed form for each flip combination; the best RMSD wins.
"""

from __future__ import annotations

import numpy as np

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import SchiraParams, fovea_w


def fit_affine_from_w_pixels(
    w: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SchiraParams,
    *,
    try_flips: bool = True,
) -> tuple[CorticalAffine, dict]:
    """Least-squares similarity: Schira ``w`` → VSD ``(row, col)``.

    Matches ``CorticalAffine.w_to_pixel``: rotate/scale in model cortex, then
    optional axis flips in camera ``(dx, dy)``. For fixed flips the model is
    linear in ``(ox, oy, A, B)`` with ``A = ppu cos θ``, ``B = ppu sin θ``::

        dx0 = A u + B v
        dy0 = -B u + A v
        col = ox + s_u * dx0
        row = oy - s_v * dy0

    where ``s_u, s_v ∈ {+1, -1}`` encode ``flip_u`` / ``flip_v``, and
    ``(u, v)`` are offsets from the fovea in model cortex.
    """
    w = np.asarray(w, dtype=np.complex128).ravel()
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError("Need at least 2 landmark pairs to fit a similarity")
    if not (w.size == rows.size == cols.size):
        raise ValueError("w, rows, cols must have the same length")

    delta = w - fovea_w(params)
    u = np.real(delta)
    v = np.imag(delta)

    flip_opts = (
        [(False, False), (True, False), (False, True), (True, True)]
        if try_flips
        else [(False, False)]
    )

    best: tuple[CorticalAffine, dict] | None = None
    for flip_u, flip_v in flip_opts:
        s_u = -1.0 if flip_u else 1.0
        s_v = -1.0 if flip_v else 1.0
        # Unknowns x = [ox, oy, A, B]
        n = w.size
        M = np.zeros((2 * n, 4), dtype=np.float64)
        y = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            # col = ox + s_u*(A*u + B*v)
            M[2 * i] = [1.0, 0.0, s_u * u[i], s_u * v[i]]
            y[2 * i] = cols[i]
            # row = oy - s_v*(-B*u + A*v) = oy + s_v*(B*u - A*v)
            M[2 * i + 1] = [0.0, 1.0, -s_v * v[i], s_v * u[i]]
            y[2 * i + 1] = rows[i]

        sol, residuals, rank, _sv = np.linalg.lstsq(M, y, rcond=None)
        ox, oy, a_cos, b_sin = (float(x) for x in sol)
        ppu = float(np.hypot(a_cos, b_sin))
        if ppu < 1e-9:
            continue
        rotation_deg = float(np.rad2deg(np.arctan2(b_sin, a_cos)))
        affine = CorticalAffine(
            origin_x=ox,
            origin_y=oy,
            pixels_per_unit=ppu,
            rotation_deg=rotation_deg,
            flip_u=flip_u,
            flip_v=flip_v,
            needs_confirmation=True,
        )
        pred_r, pred_c = affine.w_to_pixel(w, params)
        err = np.hypot(pred_r - rows, pred_c - cols)
        rmsd = float(np.sqrt(np.mean(err**2)))
        meta = {
            "rmsd_px": rmsd,
            "n_landmarks": int(n),
            "flip_u": flip_u,
            "flip_v": flip_v,
            "residuals_px": err.tolist(),
            "rank": int(rank),
        }
        if best is None or rmsd < best[1]["rmsd_px"]:
            best = (affine, meta)

    if best is None:
        raise RuntimeError("Affine fit failed for all flip combinations")
    return best


def affine_to_mapping(affine: CorticalAffine) -> dict:
    """YAML-friendly dict matching ``configs/schira`` affine blocks."""
    return {
        "needs_confirmation": bool(affine.needs_confirmation),
        "origin_xy": [float(affine.origin_x), float(affine.origin_y)],
        "pixels_per_unit": float(affine.pixels_per_unit),
        "rotation_deg": float(affine.rotation_deg),
        "flip_u": bool(affine.flip_u),
        "flip_v": bool(affine.flip_v),
    }


def schira_uv_to_w(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Landmark ``(u, v)`` = ``(Re(w), Im(w))`` from Schira forward ink or clicks."""
    u_arr = np.asarray(u, dtype=np.float64).ravel()
    v_arr = np.asarray(v, dtype=np.float64).ravel()
    if u_arr.size != v_arr.size:
        raise ValueError("u and v must have the same length")
    return u_arr + 1j * v_arr


def landmark_mode_from_pairs(pairs: list[dict]) -> str:
    """``'schira_uv'`` or ``'stimulus_px'`` from the first pair keys."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    first = pairs[0]
    if "schira_u" in first and "schira_v" in first:
        return "schira_uv"
    if "stim_xy_px" in first:
        return "stimulus_px"
    raise ValueError(
        "Each pair needs schira_u/schira_v or stim_xy_px (+ vsd_col/vsd_row)"
    )


def landmarks_stimulus_to_w(
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    params: SchiraParams,
    render_cfg,
) -> np.ndarray:
    """Stimulus image clicks → Schira ``w`` (ink path: px → deg → polar → w)."""
    from src.retinotopy.schira import cartesian_to_polar, forward_schira
    from src.retinotopy.visual_field import image_px_to_cartesian_deg

    x_deg, y_deg = image_px_to_cartesian_deg(x_px, y_px, render_cfg)
    ecc, theta = cartesian_to_polar(x_deg, y_deg)
    return forward_schira(ecc, theta, params)


def landmarks_pairs_to_w(
    pairs: list[dict],
    *,
    params: SchiraParams,
    render_cfg,
) -> np.ndarray:
    """Build Schira ``w`` from saved landmark pairs (stimulus or Schira UV mode)."""
    mode = landmark_mode_from_pairs(pairs)
    if mode == "schira_uv":
        u = np.array([p["schira_u"] for p in pairs], dtype=np.float64)
        v = np.array([p["schira_v"] for p in pairs], dtype=np.float64)
        return schira_uv_to_w(u, v)
    x_px = np.array([p["stim_xy_px"][0] for p in pairs], dtype=np.float64)
    y_px = np.array([p["stim_xy_px"][1] for p in pairs], dtype=np.float64)
    return landmarks_stimulus_to_w(x_px, y_px, params=params, render_cfg=render_cfg)
