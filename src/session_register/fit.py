"""Fit a 2D session-camera transform from vessel-emphasized maps.

Primary backend (v1): **numpy FFT phase-correlation** for translation, plus a
bounded **rotation (and optional scale) grid**, scored with NCC or Mattes-style
mutual information. Warp uses ``scipy.ndimage.affine_transform`` via
:class:`~src.session_register.transform.SessionTransform`.

This is intentionally swappable: ``fit_transform`` takes images and returns a
``SessionTransform``. A SimpleITK Euler2D + Mattes MI backend can replace the
body later without changing YAML or the CLI.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.session_register.config import (
    RegistrationBoundsError,
    SessionRegisterConfig,
)
from src.session_register.transform import SessionTransform, default_center_xy


def border_mask(shape: tuple[int, int], border_px: int) -> np.ndarray:
    """True in the interior; ignore chamber-edge pixels."""
    h, w = int(shape[0]), int(shape[1])
    b = max(int(border_px), 0)
    mask = np.zeros((h, w), dtype=bool)
    if b * 2 >= h or b * 2 >= w:
        raise ValueError(
            f"mask_border_px={border_px} leaves an empty interior for shape {(h, w)}"
        )
    mask[b : h - b, b : w - b] = True
    return mask


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    wy = np.hanning(int(shape[0]))
    wx = np.hanning(int(shape[1]))
    return np.outer(wy, wx).astype(np.float64)


def phase_cross_correlation(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Shift ``(dy_row, dx_col)`` to apply to ``moving`` so it matches ``fixed``.

    Sub-pixel peak via 3-point parabolic interpolation on the phase-correlation
    surface. Border pixels can be Hann-windowed / zeroed via ``mask``.
    """
    a = np.asarray(fixed, dtype=np.float64)
    b = np.asarray(moving, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    window = _hann2d(a.shape)
    if mask is not None:
        window = window * np.asarray(mask, dtype=np.float64)
    a = (a - a.mean()) * window
    b = (b - b.mean()) * window
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conjugate(fb)
    mag = np.abs(cross)
    cross = np.divide(cross, mag, out=np.zeros_like(cross), where=mag > 1e-12)
    xcorr = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(xcorr)), xcorr.shape)
    shifts = []
    for axis, p in enumerate(peak):
        n = xcorr.shape[axis]
        def _val(i: int, ax: int = axis, pk=peak) -> float:
            idx = list(pk)
            idx[ax] = int(i) % n
            return float(xcorr[tuple(idx)])

        ym, y0, yp = _val(p - 1), _val(p), _val(p + 1)
        denom = ym - 2.0 * y0 + yp
        delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (ym - yp) / denom
        s = float(p) + float(delta)
        if s > n / 2.0:
            s -= float(n)
        shifts.append(s)
    return float(shifts[0]), float(shifts[1])


def ncc(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(fixed, dtype=np.float64)[mask]
    b = np.asarray(moving, dtype=np.float64)[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def mattes_mi(
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = 32,
) -> float:
    a = np.asarray(fixed, dtype=np.float64)[mask]
    b = np.asarray(moving, dtype=np.float64)[mask]
    if a.size == 0:
        return 0.0
    hist, _, _ = np.histogram2d(a, b, bins=int(bins))
    joint = hist.astype(np.float64)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(joint) - np.log(pa) - np.log(pb)
    nz = joint > 0
    return float(np.sum(joint[nz] * logp[nz]))


def _metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    key = str(name).strip().lower()
    if key in ("mattes_mi", "mi"):
        return mattes_mi
    if key == "ncc":
        return ncc
    raise ValueError(f"Unknown metric {name!r}")


def _angle_grid(center: float, half_width: float, step: float, lo: float, hi: float) -> np.ndarray:
    a0 = max(center - half_width, lo)
    a1 = min(center + half_width, hi)
    if a1 < a0:
        return np.array([float(np.clip(center, lo, hi))], dtype=np.float64)
    n = int(np.floor((a1 - a0) / step + 0.5)) + 1
    return np.linspace(a0, a1, max(n, 1), dtype=np.float64)


def _scale_grid(transform_type: str) -> np.ndarray:
    if transform_type == "similarity":
        return np.array([0.92, 0.96, 1.0, 1.04, 1.08], dtype=np.float64)
    return np.array([1.0], dtype=np.float64)


def _hits_wall(
    transform: SessionTransform,
    *,
    max_rotation_deg: float,
    max_translation_px: float,
    transform_type: str,
    atol_deg: float = 0.05,
    atol_px: float = 0.25,
) -> str | None:
    if transform_type != "translation":
        if abs(transform.rotation_deg) >= float(max_rotation_deg) - atol_deg:
            return (
                f"rotation {transform.rotation_deg:.3f}° hit max_rotation_deg="
                f"{max_rotation_deg:g}"
            )
    mag = transform.translation_magnitude_px
    if mag >= float(max_translation_px) - atol_px:
        return (
            f"translation {mag:.3f} px hit max_translation_px={max_translation_px:g}"
        )
    if abs(transform.dx_col) >= float(max_translation_px) - atol_px:
        return (
            f"dx {transform.dx_col:.3f} px hit max_translation_px={max_translation_px:g}"
        )
    if abs(transform.dy_row) >= float(max_translation_px) - atol_px:
        return (
            f"dy {transform.dy_row:.3f} px hit max_translation_px={max_translation_px:g}"
        )
    return None


def _eval_pose(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    rotation_deg: float,
    scale: float,
    translation_xy: tuple[float, float],
    center_xy: tuple[float, float],
    spatial_size: tuple[int, int],
    transform_type: str,
    mask: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> tuple[SessionTransform, float, np.ndarray]:
    transform = SessionTransform(
        rotation_deg=float(rotation_deg),
        translation_xy=(float(translation_xy[0]), float(translation_xy[1])),
        transform_type=transform_type,
        scale=float(scale),
        center_xy=center_xy,
        spatial_size=spatial_size,
    )
    warped = transform.apply_to_map(moving, inverse=False, cval=0.0, order=1)
    score = float(score_fn(fixed, warped, mask))
    return transform, score, warped


def fit_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    transform_type: str = "rigid",
    metric: str = "ncc",
    max_rotation_deg: float = 25.0,
    max_translation_px: float = 30.0,
    mask_border_px: int = 4,
    cfg: SessionRegisterConfig | None = None,
) -> tuple[SessionTransform, dict[str, Any]]:
    """Register ``moving`` onto ``fixed``. Both should already be vessel maps.

    Raises ``RegistrationBoundsError`` if the best pose sits on the search wall.
    """
    if cfg is not None:
        transform_type = cfg.transform_type_name
        metric = cfg.metric_name
        max_rotation_deg = cfg.max_rotation_deg
        max_translation_px = cfg.max_translation_px
        mask_border_px = cfg.mask_border_px

    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if fixed.shape != moving.shape:
        raise ValueError(f"fixed {fixed.shape} vs moving {moving.shape}")
    if fixed.ndim != 2:
        raise ValueError("fixed and moving must be 2D")

    transform_type = str(transform_type).strip().lower()
    spatial_size = (int(fixed.shape[0]), int(fixed.shape[1]))
    center_xy = default_center_xy(fixed.shape)
    mask = border_mask(fixed.shape, mask_border_px)
    score_fn = _metric_fn(metric)
    max_rot = abs(float(max_rotation_deg))
    max_t = abs(float(max_translation_px))

    if transform_type == "translation":
        angle_passes: list[np.ndarray] = [np.array([0.0])]
    else:
        coarse = np.arange(-max_rot, max_rot + 0.5, 1.0, dtype=np.float64)
        angle_passes = [coarse]

    scales = _scale_grid(transform_type)
    best: tuple[SessionTransform, float, np.ndarray] | None = None

    def consider(rot: float, scale: float) -> None:
        nonlocal best
        probe = SessionTransform(
            rotation_deg=float(rot),
            translation_xy=(0.0, 0.0),
            transform_type=transform_type,
            scale=float(scale),
            center_xy=center_xy,
            spatial_size=spatial_size,
        )
        rotated = probe.apply_to_map(moving, inverse=False, cval=0.0, order=1)
        dy, dx = phase_cross_correlation(fixed, rotated, mask=mask)
        cand, score, warped = _eval_pose(
            fixed,
            moving,
            rotation_deg=rot,
            scale=scale,
            translation_xy=(dx, dy),
            center_xy=center_xy,
            spatial_size=spatial_size,
            transform_type=transform_type,
            mask=mask,
            score_fn=score_fn,
        )
        if best is None or score > best[1]:
            best = (cand, score, warped)

    for scale in scales:
        for rot in angle_passes[0]:
            consider(float(rot), float(scale))

    if best is None:
        raise RegistrationBoundsError(
            "No rigid candidate inside translation bounds "
            f"(max_translation_px={max_t:g})"
        )

    if transform_type != "translation":
        fine = _angle_grid(best[0].rotation_deg, 2.0, 0.25, -max_rot, max_rot)
        finer = _angle_grid(best[0].rotation_deg, 0.6, 0.1, -max_rot, max_rot)
        scale_refine = (
            np.unique(
                np.clip(
                    np.array(
                        [
                            best[0].scale * 0.97,
                            best[0].scale,
                            best[0].scale * 1.03,
                        ],
                        dtype=np.float64,
                    ),
                    0.85,
                    1.15,
                )
            )
            if transform_type == "similarity"
            else np.array([1.0], dtype=np.float64)
        )
        for scale in scale_refine:
            for rot in np.unique(np.concatenate([fine, finer])):
                consider(float(rot), float(scale))

    assert best is not None
    transform, score, warped = best
    wall = _hits_wall(
        transform,
        max_rotation_deg=max_rot,
        max_translation_px=max_t,
        transform_type=transform_type,
    )
    leftover_dy, leftover_dx = phase_cross_correlation(fixed, warped, mask=mask)
    rmsd_px = float(np.hypot(leftover_dx, leftover_dy))
    intensity_rmse = float(np.sqrt(np.mean((fixed[mask] - warped[mask]) ** 2)))
    meta: dict[str, Any] = {
        "backend": "phasecorr_grid",
        "metric": "mattes_mi" if str(metric).lower() in ("mattes_mi", "mi") else "ncc",
        "metric_value": float(score),
        "rmsd_px": rmsd_px,
        "intensity_rmse": intensity_rmse,
        "n_mask_pixels": int(mask.sum()),
    }
    if wall is not None:
        raise RegistrationBoundsError(
            f"Registration hit search bounds ({wall}). "
            f"Got rotation={transform.rotation_deg:.3f}°, "
            f"translation_xy=[{transform.dx_col:.3f}, {transform.dy_row:.3f}], "
            f"metric={meta['metric']}={score:.4f}. "
            "This pair is likely a bad match or the bound is too tight."
        )
    meta["bounds_ok"] = True
    return transform, meta


def landmark_id_label(curve_id: int, station: int) -> str:
    """Human label: U0/L0 on the two rivers, J at a fork."""
    if int(curve_id) < 0:
        return "J"
    prefix = "U" if int(curve_id) == 0 else "L"
    return f"{prefix}{int(station)}"


def match_landmarks_by_id(
    fixed_rc: np.ndarray,
    fixed_ids: np.ndarray,
    moving_rc: np.ndarray,
    moving_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair landmarks that share the same ``(curve_id, station_k)`` label.

    Returns ``(fixed_matched, moving_matched, ids)``.  A label present in only
    one session is skipped; nearest-neighbour is not used.
    """
    fixed = np.asarray(fixed_rc, dtype=np.float64).reshape(-1, 2)
    moving = np.asarray(moving_rc, dtype=np.float64).reshape(-1, 2)
    fid = np.asarray(fixed_ids, dtype=np.int64).reshape(-1, 2) if np.asarray(fixed_ids).size else np.zeros((0, 2), dtype=np.int64)
    mid = np.asarray(moving_ids, dtype=np.int64).reshape(-1, 2) if np.asarray(moving_ids).size else np.zeros((0, 2), dtype=np.int64)
    if fixed.size == 0 or moving.size == 0 or fid.size == 0 or mid.size == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, np.zeros((0, 2), dtype=np.int64)
    fmap: dict[tuple[int, int], int] = {}
    for i, pair in enumerate(fid):
        fmap[(int(pair[0]), int(pair[1]))] = i
    keep_f: list[int] = []
    keep_m: list[int] = []
    keep_id: list[tuple[int, int]] = []
    used_f: set[int] = set()
    for j, pair in enumerate(mid):
        key = (int(pair[0]), int(pair[1]))
        i = fmap.get(key)
        if i is None or i in used_f:
            continue
        used_f.add(i)
        keep_f.append(i)
        keep_m.append(j)
        keep_id.append(key)
    if not keep_f:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, np.zeros((0, 2), dtype=np.int64)
    return (
        fixed[np.asarray(keep_f)],
        moving[np.asarray(keep_m)],
        np.asarray(keep_id, dtype=np.int64),
    )


def match_landmarks(
    fixed_rc: np.ndarray,
    moving_rc: np.ndarray,
    transform: SessionTransform,
    *,
    max_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy unique nearest-neighbour match after applying ``transform`` to moving.

    Returns ``(fixed_matched, moving_matched)`` as ``(K, 2)`` row/col arrays.
    """
    fixed = np.asarray(fixed_rc, dtype=np.float64).reshape(-1, 2)
    moving = np.asarray(moving_rc, dtype=np.float64).reshape(-1, 2)
    if fixed.size == 0 or moving.size == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    wr, wc = transform.apply_to_points(moving[:, 0], moving[:, 1], inverse=False)
    warped = np.stack([wr, wc], axis=1)
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(warped):
        d2 = np.sum((fixed - p) ** 2, axis=1)
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        if dist <= float(max_px):
            pairs.append((dist, i, j))
    pairs.sort()
    used_m: set[int] = set()
    used_f: set[int] = set()
    keep_m: list[int] = []
    keep_f: list[int] = []
    for _d, i, j in pairs:
        if i in used_m or j in used_f:
            continue
        used_m.add(i)
        used_f.add(j)
        keep_m.append(i)
        keep_f.append(j)
    if not keep_m:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    return fixed[np.asarray(keep_f)], moving[np.asarray(keep_m)]


def rigid_from_point_pairs(
    moving_rc: np.ndarray,
    fixed_rc: np.ndarray,
    *,
    center_xy: tuple[float, float],
    spatial_size: tuple[int, int],
) -> SessionTransform:
    """Kabsch rigid map: moving landmarks → fixed landmarks (no scale)."""
    src = np.asarray(moving_rc, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(fixed_rc, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 2 or src.shape != dst.shape:
        raise ValueError("need at least 2 matched landmark pairs")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    x = src - mu_s
    y = dst - mu_d
    h = x.T @ y
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if float(np.linalg.det(rot)) < 0:
        vt = vt.copy()
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    # p_fixed = R (p_moving - c) + c + t  ⇒  t = (μ_d - c) - R (μ_s - c)
    c = np.array([center_xy[1], center_xy[0]], dtype=np.float64)  # (row, col)
    t_yx = (mu_d - c) - rot @ (mu_s - c)
    theta = float(np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0])))
    return SessionTransform(
        rotation_deg=theta,
        translation_xy=(float(t_yx[1]), float(t_yx[0])),
        transform_type="rigid",
        scale=1.0,
        center_xy=center_xy,
        spatial_size=spatial_size,
    )


def landmark_rmsd_px(
    moving_rc: np.ndarray,
    fixed_rc: np.ndarray,
    transform: SessionTransform,
) -> float:
    wr, wc = transform.apply_to_points(
        np.asarray(moving_rc)[:, 0], np.asarray(moving_rc)[:, 1]
    )
    d = np.stack([wr, wc], axis=1) - np.asarray(fixed_rc, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
