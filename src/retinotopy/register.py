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
        candidate = _affine_for_flips(u, v, rows, cols, params, flip_u, flip_v)
        if candidate is None:
            continue
        affine, meta = candidate
        if best is None or meta["rmsd_px"] < best[1]["rmsd_px"]:
            best = (affine, meta)

    if best is None:
        raise RuntimeError("Affine fit failed for all flip combinations")
    return best


def _affine_for_flips(
    u: np.ndarray,
    v: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SchiraParams,
    flip_u: bool,
    flip_v: bool,
) -> tuple[CorticalAffine, dict] | None:
    s_u = -1.0 if flip_u else 1.0
    s_v = -1.0 if flip_v else 1.0
    n = int(u.size)
    M = np.zeros((2 * n, 4), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    for i in range(n):
        M[2 * i] = [1.0, 0.0, s_u * u[i], s_u * v[i]]
        y[2 * i] = cols[i]
        M[2 * i + 1] = [0.0, 1.0, -s_v * v[i], s_v * u[i]]
        y[2 * i + 1] = rows[i]

    sol, _residuals, rank, _sv = np.linalg.lstsq(M, y, rcond=None)
    ox, oy, a_cos, b_sin = (float(x) for x in sol)
    ppu = float(np.hypot(a_cos, b_sin))
    if ppu < 1e-9:
        return None
    affine = CorticalAffine(
        origin_x=ox,
        origin_y=oy,
        pixels_per_unit=ppu,
        rotation_deg=float(np.rad2deg(np.arctan2(b_sin, a_cos))),
        flip_u=flip_u,
        flip_v=flip_v,
        needs_confirmation=True,
    )
    w = (u + 1j * v) + fovea_w(params)
    pred_r, pred_c = affine.w_to_pixel(w, params)
    err = np.hypot(pred_r - rows, pred_c - cols)
    rmsd = float(np.sqrt(np.mean(err**2)))
    meta = {
        "rmsd_px": rmsd,
        "n_landmarks": n,
        "flip_u": flip_u,
        "flip_v": flip_v,
        "residuals_px": err.tolist(),
        "rank": int(rank),
    }
    return affine, meta


def isotropic_scale_w_to_vsd(
    w: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SchiraParams,
    *,
    min_w_dist: float = 1e-6,
) -> tuple[float, dict]:
    """Median pairwise |ΔVSD| / |Δw| — one isotropic scale, no shear."""
    w = np.asarray(w, dtype=np.complex128).ravel()
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    if w.size != rows.size or w.size != cols.size:
        raise ValueError("w, rows, cols must have the same length")
    delta = w - fovea_w(params)
    u = np.real(delta)
    v = np.imag(delta)
    ratios: list[float] = []
    n = int(w.size)
    for i in range(n):
        for j in range(i + 1, n):
            dw = float(np.hypot(u[i] - u[j], v[i] - v[j]))
            dp = float(np.hypot(cols[i] - cols[j], rows[i] - rows[j]))
            if dw >= min_w_dist:
                ratios.append(dp / dw)
    if not ratios:
        raise ValueError("Need a landmark pair with nonzero w-distance")
    arr = np.asarray(ratios, dtype=np.float64)
    return float(np.median(arr)), {
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
    }


def fit_rigid_from_w_pixels(
    w: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SchiraParams,
    *,
    pixels_per_unit: float | None = None,
    try_flips: bool = True,
) -> tuple[CorticalAffine, dict]:
    """Session camera: frozen isotropic scale, then rotation + translation.

    Schira may change stimulus size/shape. This step does **not**: no extra
    stretch, no per-stimulus scale. ``pixels_per_unit`` is the w→pixel scale
    (median pairwise if omitted). Discrete axis flips are camera orientation.
    """
    w = np.asarray(w, dtype=np.complex128).ravel()
    rows = np.asarray(rows, dtype=np.float64).ravel()
    cols = np.asarray(cols, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError("Need at least 2 landmark pairs for a rigid map")
    if not (w.size == rows.size == cols.size):
        raise ValueError("w, rows, cols must have the same length")
    scale_meta: dict = {}
    if pixels_per_unit is None:
        ppu, scale_meta = isotropic_scale_w_to_vsd(w, rows, cols, params)
    else:
        ppu = float(pixels_per_unit)
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
        candidate = _rigid_for_flips(u, v, rows, cols, params, ppu, flip_u, flip_v)
        if candidate is None:
            continue
        affine, meta = candidate
        meta = {**meta, "scale": scale_meta, "pixels_per_unit": ppu}
        if best is None or meta["rmsd_px"] < best[1]["rmsd_px"]:
            best = (affine, meta)
    if best is None:
        raise RuntimeError("Rigid fit failed for all flip combinations")
    return best


def _rigid_for_flips(
    u: np.ndarray,
    v: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SchiraParams,
    pixels_per_unit: float,
    flip_u: bool,
    flip_v: bool,
) -> tuple[CorticalAffine, dict] | None:
    """Kabsch: ``Y ≈ t + F R (s X)`` with fixed ``s`` and det(R)=+1."""
    s_u = -1.0 if flip_u else 1.0
    s_v = -1.0 if flip_v else 1.0
    ppu = float(pixels_per_unit)
    if ppu <= 0:
        return None
    src = np.column_stack([u * ppu, v * ppu])
    cam = np.column_stack([cols, -rows])
    f_diag = np.array([s_u, s_v], dtype=np.float64)
    tgt = cam * f_diag
    src_mu = src.mean(axis=0)
    tgt_mu = tgt.mean(axis=0)
    h = (src - src_mu).T @ (tgt - tgt_mu)
    uu, _ss, vt = np.linalg.svd(h)
    rot = vt.T @ uu.T
    if np.linalg.det(rot) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rot = vt.T @ uu.T
    t_flipped = tgt_mu - rot @ src_mu
    t_cam = f_diag * t_flipped
    ox = float(t_cam[0])
    oy = float(-t_cam[1])
    rotation_deg = float(np.rad2deg(np.arctan2(rot[0, 1], rot[0, 0])))
    affine = CorticalAffine(
        origin_x=ox,
        origin_y=oy,
        pixels_per_unit=ppu,
        rotation_deg=rotation_deg,
        flip_u=flip_u,
        flip_v=flip_v,
        needs_confirmation=True,
    )
    w = (u + 1j * v) + fovea_w(params)
    pred_r, pred_c = affine.w_to_pixel(w, params)
    err = np.hypot(pred_r - rows, pred_c - cols)
    rmsd = float(np.sqrt(np.mean(err**2)))
    meta = {
        "rmsd_px": rmsd,
        "n_landmarks": int(u.size),
        "flip_u": flip_u,
        "flip_v": flip_v,
        "residuals_px": err.tolist(),
        "mode": "rigid",
    }
    return affine, meta


def fit_shift_from_w_pixel(
    w: complex,
    row: float,
    col: float,
    params: SchiraParams,
    *,
    pixels_per_unit: float,
    rotation_deg: float = 0.0,
    flip_u: bool = False,
    flip_v: bool = False,
) -> tuple[CorticalAffine, dict]:
    """Place one Schira ``w`` on one VSD pixel: translation only.

    Scale and rotation stay as given (one point cannot solve them). Origin is
    set so this landmark lands exactly.
    """
    ppu = float(pixels_per_unit)
    proto = CorticalAffine(
        origin_x=0.0,
        origin_y=0.0,
        pixels_per_unit=ppu,
        rotation_deg=float(rotation_deg),
        flip_u=bool(flip_u),
        flip_v=bool(flip_v),
        needs_confirmation=True,
    )
    r0, c0 = proto.w_to_pixel(np.array([w], dtype=np.complex128), params)
    ox = float(col) - float(c0[0])
    oy = float(row) - float(r0[0])
    affine = CorticalAffine(
        origin_x=ox,
        origin_y=oy,
        pixels_per_unit=ppu,
        rotation_deg=float(rotation_deg),
        flip_u=bool(flip_u),
        flip_v=bool(flip_v),
        needs_confirmation=True,
    )
    pr, pc = affine.w_to_pixel(np.array([w], dtype=np.complex128), params)
    err = float(np.hypot(float(pr[0]) - float(row), float(pc[0]) - float(col)))
    meta = {
        "rmsd_px": err,
        "n_landmarks": 1,
        "mode": "shift",
        "pixels_per_unit": ppu,
        "rotation_deg": float(rotation_deg),
        "residuals_px": [err],
    }
    return affine, meta


def vf_cartesian_px_to_w(
    x_px: np.ndarray | float,
    y_px: np.ndarray | float,
    params: SchiraParams,
    *,
    pixels_per_deg: float = 35.0,
) -> np.ndarray:
    """Stimulus-canvas pixels (35 px/deg, fovea at top-left) → Schira ``w``."""
    from src.retinotopy.schira import cartesian_to_polar, forward_schira

    x_deg = np.asarray(x_px, dtype=np.float64) / float(pixels_per_deg)
    y_deg = -np.asarray(y_px, dtype=np.float64) / float(pixels_per_deg)
    ecc, theta = cartesian_to_polar(x_deg, y_deg)
    return forward_schira(ecc, theta, params)


def affine_from_two_pixel_anchors(
    w_a: complex,
    w_b: complex,
    row_a: float,
    col_a: float,
    row_b: float,
    col_b: float,
    params: SchiraParams,
    *,
    check_w: complex | None = None,
    check_row: float | None = None,
    check_col: float | None = None,
) -> tuple[CorticalAffine, dict]:
    """Exact similarity: two Schira ``w`` points land on two VSD pixels.

    This is not a least-squares fit over extra landmarks. Both interpolants
    (with/without axis flips) hit the two anchors; if ``check_*`` is given,
    the reflection with the smaller check residual is kept.
    """
    w = np.array([w_a, w_b], dtype=np.complex128)
    rows = np.array([row_a, row_b], dtype=np.float64)
    cols = np.array([col_a, col_b], dtype=np.float64)
    have_check = check_w is not None and check_row is not None and check_col is not None
    delta = w - fovea_w(params)
    u = np.real(delta)
    v = np.imag(delta)

    best: tuple[CorticalAffine, dict] | None = None
    for flip_u, flip_v in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        candidate = _affine_for_flips(u, v, rows, cols, params, flip_u, flip_v)
        if candidate is None:
            continue
        affine, meta = candidate
        check_err = None
        if have_check:
            cr, cc = affine.w_to_pixel(
                np.array([check_w], dtype=np.complex128), params
            )
            check_err = float(
                np.hypot(
                    float(cr[0]) - float(check_row),
                    float(cc[0]) - float(check_col),
                )
            )
        scored = {
            **meta,
            "anchor_rmsd_px": float(meta["rmsd_px"]),
            "check_err_px": check_err,
        }
        if best is None:
            best = (affine, scored)
            continue
        if have_check:
            prev = best[1].get("check_err_px")
            if check_err is not None and (prev is None or check_err < prev - 1e-9):
                best = (affine, scored)
            continue
        if scored["anchor_rmsd_px"] < best[1]["anchor_rmsd_px"]:
            best = (affine, scored)

    if best is None:
        raise RuntimeError("Two-anchor placement failed")
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
