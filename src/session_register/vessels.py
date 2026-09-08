"""Isolate static vasculature so later registration sees vessels, not cortex.

Default path: median-smooth, then a **black top-hat at vessel scale** so each
trunk is one dark basin (not two wall-edges). The centerline is the top-hat
peak / intensity trough inside that basin. Short fragments are dropped.
Search is the bottom half of an r≈40 disk (chamber ring stays outside).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.ndimage import (
    black_tophat,
    convolve,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    median_filter,
    white_tophat,
)
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from src.session_register.config import SessionRegisterConfig


def _disk(radius: float) -> np.ndarray:
    r = max(int(np.ceil(radius)), 1)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (float(radius) ** 2)


def _rescale01(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _percentile_clip(image: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    lo, hi = np.percentile(image, [float(percentiles[0]), float(percentiles[1])])
    if hi <= lo:
        return np.asarray(image, dtype=np.float64)
    return np.clip(image, lo, hi)


def chamber_mask(
    image: np.ndarray, *, border_px: int = 6, rim_px: int = 0
) -> np.ndarray:
    """Circular window inside the cranial chamber.

    ``border_px`` drops the square-frame corners. ``rim_px`` additionally
    peels an annulus so the bright/dark **chamber circle** is not treated
    as a vessel.
    """
    x = np.asarray(image, dtype=np.float64)
    height, width = x.shape
    b = max(int(border_px), 0)
    rim = max(int(rim_px), 0)
    rows, cols = np.ogrid[:height, :width]
    radius = np.hypot(rows - (height - 1) / 2.0, cols - (width - 1) / 2.0)
    radius_max = min(height, width) / 2.0 - b - rim
    mask = radius <= max(radius_max, 1.0)
    if b:
        mask[:b, :] = False
        mask[-b:, :] = False
        mask[:, :b] = False
        mask[:, -b:] = False
    return mask


def keep_n_longest_objects(
    mask: np.ndarray,
    n: int = 2,
    *,
    min_area: int = 40,
) -> np.ndarray:
    """Keep the ``n`` components with the most left–right span (vessel trunks)."""
    binary = np.asarray(mask, dtype=bool)
    lab = label(binary)
    scored: list[tuple[float, int]] = []
    for prop in regionprops(lab):
        if prop.area < min_area:
            continue
        _minr, minc, _maxr, maxc = prop.bbox
        width = maxc - minc
        score = float(prop.area) * (1.0 + width / 15.0) * (1.0 + float(prop.eccentricity))
        scored.append((score, int(prop.label)))
    scored.sort(reverse=True)
    keep = np.zeros_like(binary)
    for _score, lab_id in scored[: max(int(n), 0)]:
        keep |= lab == lab_id
    return keep


def _eroded_interior(chamber: np.ndarray, erode_px: int = 5) -> np.ndarray:
    """Shrink the circular window so the mask edge is not a false ridge."""
    if erode_px <= 0:
        return np.asarray(chamber, dtype=bool)
    return distance_transform_edt(chamber) >= int(erode_px)


def _bottom_disk_roi(
    shape: tuple[int, int],
    *,
    radius_px: float = 40.0,
    rim_erode_px: int = 0,
) -> np.ndarray:
    """Bottom half of a disk at the frame center (chamber interior, no outer circle)."""
    height, width = int(shape[0]), int(shape[1])
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rows, cols = np.ogrid[:height, :width]
    disk = np.hypot(rows - cy, cols - cx) <= float(radius_px)
    if rim_erode_px > 0:
        disk = distance_transform_edt(disk) >= int(rim_erode_px)
    return disk & (rows >= cy)


def _skip_roi_bottom(roi: np.ndarray, skip_px: int = 6) -> np.ndarray:
    """Drop the last rows of the ROI so the disk cut is not voted as a river."""
    out = np.asarray(roi, dtype=bool).copy()
    if skip_px <= 0:
        return out
    rows = np.where(out.any(axis=1))[0]
    if rows.size:
        ymax = int(rows.max())
        out[max(0, ymax - int(skip_px) + 1) :, :] = False
    return out


def _tophat_seed_rows(
    hat: np.ndarray,
    roi: np.ndarray,
    *,
    n: int = 2,
    min_sep: int = 12,
) -> list[int]:
    """Two trunk rows = strongest black-tophat peaks in the ROI (basin centers)."""
    height = hat.shape[0]
    profile = np.zeros(height, dtype=np.float64)
    for row in range(height):
        ok = roi[row]
        if not np.any(ok):
            continue
        profile[row] = float(np.percentile(hat[row, ok], 75))
    profile = gaussian_filter1d(profile, 1.2)
    work = profile.copy()
    work[~np.asarray(roi).any(axis=1)] = -1.0
    peaks: list[int] = []
    for _ in range(max(int(n), 0)):
        row = int(np.argmax(work))
        if work[row] <= 0:
            break
        peaks.append(row)
        work[max(0, row - min_sep) : row + min_sep + 1] = -1.0
    peaks.sort(key=lambda r: float(profile[r]), reverse=True)
    return peaks


def _trace_tophat_centerline(
    hat: np.ndarray,
    roi: np.ndarray,
    seed_row: int,
    *,
    occupied: np.ndarray | None = None,
    half_win: int = 6,
    min_sep: int = 8,
    min_len: int = 18,
) -> dict[int, int]:
    """Continuous centerline on one dark basin: strong tophat knots, small-gap fill only.

    Do **not** interpolate across long empty spans — that draws through cortex.
    """
    height, width = hat.shape
    knots: dict[int, tuple[int, float]] = {}
    for col in range(width):
        lo = max(0, int(seed_row) - int(half_win))
        hi = min(height, int(seed_row) + int(half_win) + 1)
        ok = roi[lo:hi, col]
        if not np.any(ok):
            continue
        sl = np.where(ok, hat[lo:hi, col], -np.inf)
        i = int(np.argmax(sl))
        val = float(sl[i])
        if not np.isfinite(val) or val <= 0:
            continue
        row = lo + i
        if occupied is not None and np.isfinite(occupied[col]):
            if abs(float(row) - float(occupied[col])) < min_sep:
                continue
        knots[int(col)] = (int(row), val)
    if len(knots) < int(min_len):
        return {}
    strengths = np.array([v for _r, v in knots.values()], dtype=np.float64)
    thr = float(np.percentile(strengths, 35))
    knots = {c: (r, v) for c, (r, v) in knots.items() if v >= thr}
    if len(knots) < int(min_len):
        return {}
    cols = np.array(sorted(knots), dtype=np.int64)
    # Split into runs where the column gap is small; keep the longest run.
    runs: list[list[int]] = [[int(cols[0])]]
    for c in cols[1:]:
        if int(c) - runs[-1][-1] <= 6:
            runs[-1].append(int(c))
        else:
            runs.append([int(c)])
    run = max(runs, key=len)
    if len(run) < int(min_len):
        return {}
    known = np.asarray(run, dtype=np.int64)
    y_known = np.asarray([knots[int(c)][0] for c in known], dtype=np.float64)
    out_cols = np.arange(int(known[0]), int(known[-1]) + 1)
    yv = np.interp(out_cols.astype(np.float64), known.astype(np.float64), y_known)
    if yv.size >= 5:
        yv = median_filter(yv, size=5)
    out: dict[int, int] = {}
    for col, y in zip(out_cols, yv):
        y0 = int(round(float(y)))
        lo = max(0, y0 - 2)
        hi = min(height, y0 + 3)
        ok = roi[lo:hi, int(col)]
        if np.any(ok):
            i = int(np.argmax(np.where(ok, hat[lo:hi, int(col)], -np.inf)))
            row = lo + i
        else:
            row = min(max(y0, 0), height - 1)
        if occupied is not None and np.isfinite(occupied[int(col)]):
            if abs(float(row) - float(occupied[int(col)])) < min_sep:
                continue
        out[int(col)] = int(row)
    return out if len(out) >= int(min_len) else {}


def _skeleton_junctions(skel: np.ndarray) -> np.ndarray:
    """(row, col) pixels where a 1 px skeleton has 3+ neighbors (vessel junctions)."""
    binary = np.asarray(skel, dtype=bool)
    if not np.any(binary):
        return np.zeros((0, 2), dtype=np.int64)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.int64)
    nhood = convolve(binary.astype(np.int64), kernel, mode="constant", cval=0)
    junct = binary & (nhood >= 13)
    rows, cols = np.where(junct)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([rows, cols], axis=1).astype(np.int64)


def _cluster_points(pts: np.ndarray, *, radius_px: int = 4) -> np.ndarray:
    """Collapse nearby junction pixels to one landmark each."""
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    left = [tuple(map(int, p)) for p in np.asarray(pts).reshape(-1, 2)]
    kept: list[tuple[int, int]] = []
    while left:
        r0, c0 = left.pop(0)
        cluster = [(r0, c0)]
        rest: list[tuple[int, int]] = []
        for r, c in left:
            if (r - r0) ** 2 + (c - c0) ** 2 <= radius_px * radius_px:
                cluster.append((r, c))
            else:
                rest.append((r, c))
        left = rest
        kept.append(
            (
                int(round(float(np.mean([p[0] for p in cluster])))),
                int(round(float(np.mean([p[1] for p in cluster])))),
            )
        )
    return np.asarray(kept, dtype=np.int64)


def _nudge_into_bed(
    image01: np.ndarray, roi: np.ndarray, row: int, col: int
) -> int:
    """If the trough is 2 px thick under a bright bank, sit one pixel deeper."""
    height = image01.shape[0]
    r = int(row)
    if r <= 0 or r + 1 >= height or not roi[r + 1, col]:
        return r
    above = float(image01[r - 1, col])
    here = float(image01[r, col])
    below = float(image01[r + 1, col])
    if above >= 0.55 and below <= 0.32 and below <= here + 0.20:
        return r + 1
    return r


def _is_column_local_min(
    image01: np.ndarray, roi: np.ndarray, row: int, col: int
) -> bool:
    """True if ``row`` is a dark trough (not the bright bank, not a slope)."""
    height = image01.shape[0]
    if row < 0 or row >= height or not roi[row, col]:
        return False
    v = float(image01[row, col])
    up = float(image01[row - 1, col]) if row > 0 else v + 1.0
    dn = float(image01[row + 1, col]) if row + 1 < height else v + 1.0
    return v <= up and v <= dn


def _pick_riverbed_row(
    image01: np.ndarray,
    roi: np.ndarray,
    col: int,
    y_prev: int,
    *,
    max_jump: int,
    occupied: np.ndarray | None,
    min_sep: int,
    seed_row: int | None = None,
    max_drift: int = 10,
) -> tuple[int | None, float]:
    """Stay on a local intensity minimum. Argmin-on-a-slope ratchets off the river."""
    height = image01.shape[0]
    lo = max(0, int(y_prev) - int(max_jump))
    hi = min(height, int(y_prev) + int(max_jump) + 1)
    candidates: list[int] = []
    for row in range(lo, hi):
        if not roi[row, col]:
            continue
        if occupied is not None and np.isfinite(occupied[col]):
            if abs(row - float(occupied[col])) < min_sep:
                continue
        if seed_row is not None and abs(row - int(seed_row)) > int(max_drift):
            continue
        candidates.append(row)
    if not candidates:
        return None, np.inf
    troughs = [r for r in candidates if _is_column_local_min(image01, roi, r, col)]
    if not troughs:
        return None, np.inf
    row = min(troughs, key=lambda r: float(image01[r, col]))
    row = _nudge_into_bed(image01, roi, row, col)
    return int(row), float(image01[row, col])


def _trace_near_seed(
    image01: np.ndarray,
    roi: np.ndarray,
    seed_row: int,
    *,
    occupied: np.ndarray | None = None,
    min_sep: int = 7,
    max_drift: int = 8,
) -> dict[int, int]:
    """Dark local-minima within ``max_drift`` of ``seed_row``, then fill gaps."""
    _height, width = image01.shape
    knots: dict[int, int] = {}
    darks: list[float] = []
    for col in range(width):
        rows = np.where(roi[:, col])[0]
        if rows.size < 3:
            continue
        troughs: list[int] = []
        for row in rows:
            if abs(int(row) - int(seed_row)) > int(max_drift):
                continue
            if occupied is not None and np.isfinite(occupied[col]):
                if abs(float(row) - float(occupied[col])) < min_sep:
                    continue
            if _is_column_local_min(image01, roi, int(row), col):
                troughs.append(int(row))
        if not troughs:
            continue
        row = min(troughs, key=lambda r: float(image01[r, col]))
        row = _nudge_into_bed(image01, roi, row, col)
        val = float(image01[row, col])
        if val > 0.55:
            continue
        knots[int(col)] = int(row)
        darks.append(val)
    if len(knots) < 8:
        return {}
    thr = float(np.percentile(darks, 75))
    knots = {c: r for c, r in knots.items() if float(image01[r, c]) <= thr}
    if len(knots) < 8:
        return {}
    return _fill_and_snap_riverbed(
        knots,
        image01,
        roi,
        occupied=occupied,
        min_sep=min_sep,
        seed_row=seed_row,
        max_drift=max_drift,
    )


def _fill_and_snap_riverbed(
    ys: dict[int, int],
    image01: np.ndarray,
    roi: np.ndarray,
    *,
    occupied: np.ndarray | None,
    min_sep: int,
    seed_row: int,
    max_drift: int,
) -> dict[int, int]:
    """Interpolate gaps, then snap each column onto a local dark trough."""
    if len(ys) < 6:
        return {}
    known = np.array(sorted(ys), dtype=np.int64)
    y_known = np.asarray([ys[int(c)] for c in known], dtype=np.float64)
    cols = np.arange(int(known[0]), int(known[-1]) + 1)
    yv = np.interp(cols.astype(np.float64), known.astype(np.float64), y_known)
    if yv.size >= 5:
        yv = median_filter(yv, size=7)
    raw: dict[int, int] = {}
    for col, y in zip(cols, yv):
        lo = max(0, int(seed_row) - int(max_drift))
        hi = min(image01.shape[0], int(seed_row) + int(max_drift) + 1)
        troughs: list[int] = []
        for row in range(lo, hi):
            if occupied is not None and np.isfinite(occupied[col]):
                if abs(float(row) - float(occupied[col])) < min_sep:
                    continue
            if _is_column_local_min(image01, roi, row, int(col)):
                troughs.append(row)
        if troughs:
            row = min(troughs, key=lambda r: float(image01[r, int(col)]))
            row = _nudge_into_bed(image01, roi, row, int(col))
            if float(image01[row, int(col)]) <= 0.55:
                raw[int(col)] = int(row)
                continue
        row, val = _pick_riverbed_row(
            image01,
            roi,
            int(col),
            int(round(float(y))),
            max_jump=2,
            occupied=occupied,
            min_sep=min_sep,
            seed_row=seed_row,
            max_drift=max_drift,
        )
        if row is None or not np.isfinite(val):
            continue
        raw[int(col)] = int(row)
    if len(raw) < 6:
        return {}
    known2 = np.array(sorted(raw), dtype=np.int64)
    y2 = np.asarray([raw[int(c)] for c in known2], dtype=np.float64)
    if y2.size >= 5:
        y2 = median_filter(y2, size=5)
    out: dict[int, int] = {}
    for col, y in zip(known2, y2):
        row, val = _pick_riverbed_row(
            image01,
            roi,
            int(col),
            int(round(float(y))),
            max_jump=2,
            occupied=occupied,
            min_sep=min_sep,
            seed_row=seed_row,
            max_drift=max_drift,
        )
        if row is None or not np.isfinite(val):
            continue
        out[int(col)] = _nudge_into_bed(image01, roi, int(row), int(col))
    return out


def _rasterize_polyline(
    ys: dict[int, int],
    interior: np.ndarray,
    *,
    thickness_px: int = 1,
) -> np.ndarray:
    """Paint an 8-connected 1 px polyline (no gaps between adjacent columns)."""
    height, width = interior.shape
    mask = np.zeros((height, width), dtype=bool)
    if len(ys) < 6:
        return mask
    known = np.array(sorted(ys), dtype=np.int64)
    y_known = np.asarray([ys[int(c)] for c in known], dtype=np.float64)
    cols = np.arange(int(known[0]), int(known[-1]) + 1)
    yv = np.interp(cols.astype(np.float64), known.astype(np.float64), y_known)
    if yv.size >= 5:
        yv = gaussian_filter1d(yv.astype(np.float64), 0.8)
    thick = max(int(thickness_px), 1)
    half = (thick - 1) / 2.0
    prev_r: int | None = None
    prev_c: int | None = None
    for col, y in zip(cols, yv):
        r_mid = int(round(float(y)))
        r_mid = min(max(r_mid, 0), height - 1)
        r0 = r_mid if thick <= 1 else int(np.floor(float(y) - half))
        r1 = r_mid if thick <= 1 else int(np.ceil(float(y) + half))
        if prev_r is not None and prev_c is not None and int(col) - prev_c <= 2:
            for rr, cc in _bresenham(prev_r, prev_c, r_mid, int(col)):
                if 0 <= rr < height and 0 <= cc < width and interior[rr, cc]:
                    mask[rr, cc] = True
        for rr in range(min(r0, r1), max(r0, r1) + 1):
            if 0 <= rr < height and 0 <= int(col) < width and interior[rr, int(col)]:
                mask[rr, int(col)] = True
        prev_r, prev_c = r_mid, int(col)
    return mask


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    dr = abs(int(r1) - int(r0))
    dc = abs(int(c1) - int(c0))
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    row, col = int(r0), int(c0)
    pts: list[tuple[int, int]] = []
    if dc >= dr:
        err = dc // 2
        for _ in range(dc + 1):
            pts.append((row, col))
            err -= dr
            if err < 0:
                row += sr
                err += dc
            col += sc
    else:
        err = dr // 2
        for _ in range(dr + 1):
            pts.append((row, col))
            err -= dc
            if err < 0:
                col += sc
                err += dr
            row += sr
    return pts


def _maps_from_mask(
    shape: tuple[int, int],
    vmask: np.ndarray,
    chamber: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> VesselMaps:
    display = np.ones(shape, dtype=np.float64)
    display[vmask] = 0.0
    fit = np.zeros(shape, dtype=np.float64)
    fit[vmask] = 1.0
    pts = (
        np.zeros((0, 2), dtype=np.int64)
        if landmarks is None
        else np.asarray(landmarks, dtype=np.int64).reshape(-1, 2)
    )
    return VesselMaps(
        display=display.astype(np.float32),
        fit=fit.astype(np.float32),
        vessel_mask=vmask,
        chamber_mask=chamber,
        landmarks=pts,
    )


@dataclass(frozen=True)
class VesselMaps:
    """Paired views of the same vessel isolation.

    ``display`` — near-white field, dark vessels (what you look at).
    ``fit`` — bright ridges on black (what a later rigid search uses).
    """

    display: np.ndarray
    fit: np.ndarray
    vessel_mask: np.ndarray
    chamber_mask: np.ndarray
    landmarks: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )


def extract_vessels(
    image: np.ndarray,
    *,
    cfg: SessionRegisterConfig | None = None,
    invert_intensity: bool = False,
    vessel_method: str = "meijering",
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    hard_clip_percentiles: tuple[float, float] | None = None,
    tophat_radius_px: float = 6.0,
    gaussian_sigma: float = 1.5,
    vessel_percentile: float = 75.0,
    background_sigma: float = 8.0,
    mask_border_px: int = 8,
    n_vessel_objects: int = 2,
    hysteresis_percentiles: tuple[float, float] = (74.0, 91.0),
    chamber_rim_px: int = 12,
    roi_radius_px: int = 40,
    vessel_trace_px: int = 1,
) -> VesselMaps:
    """Isolate dark vascular ridges; saturate everything else to white."""
    if cfg is not None:
        invert_intensity = cfg.invert_intensity
        vessel_method = cfg.vessel_method
        clip_percentiles = cfg.clip_percentiles
        hard_clip_percentiles = cfg.hard_clip_percentiles
        tophat_radius_px = cfg.tophat_radius_px
        gaussian_sigma = cfg.gaussian_sigma
        vessel_percentile = cfg.vessel_percentile
        background_sigma = cfg.background_sigma
        mask_border_px = max(int(cfg.mask_border_px), 6)
        n_vessel_objects = int(cfg.n_vessel_objects)
        chamber_rim_px = int(cfg.chamber_rim_px)
        roi_radius_px = int(cfg.roi_radius_px)
        vessel_trace_px = int(cfg.vessel_trace_px)

    raw = np.asarray(image, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {raw.shape}")
    method = str(vessel_method).strip().lower()
    if method in ("clip_tophat", "clip", "clahe_clip", "dark_tophat"):
        return _dark_tophat_maps(
            raw,
            invert_intensity=invert_intensity,
            method=method,
            clip_percentiles=clip_percentiles,
            hard_clip_percentiles=hard_clip_percentiles,
            tophat_radius_px=tophat_radius_px,
            gaussian_sigma=gaussian_sigma,
            vessel_percentile=vessel_percentile,
            background_sigma=background_sigma,
            mask_border_px=mask_border_px,
            n_vessel_objects=n_vessel_objects,
            chamber_rim_px=chamber_rim_px,
        )

    work = -raw if invert_intensity else raw
    chamber = chamber_mask(
        work, border_px=mask_border_px, rim_px=chamber_rim_px
    )
    image01 = _rescale01(_percentile_clip(work, (2.0, 98.0)))
    # Median kills salt-pepper; black top-hat at vessel radius sees each
    # trunk as one dark basin (not two wall-edges of a too-small Hessian).
    med = median_filter(image01, size=5)
    hat_r = max(float(tophat_radius_px), 5.0)
    hat = black_tophat(med, footprint=_disk(hat_r))
    search_roi = _skip_roi_bottom(
        _bottom_disk_roi(raw.shape, radius_px=roi_radius_px, rim_erode_px=4),
        skip_px=6,
    )
    seeds = _tophat_seed_rows(
        hat, search_roi, n=n_vessel_objects, min_sep=12
    )
    half = max(int(round(hat_r)), 5)
    vmask = np.zeros(raw.shape, dtype=bool)
    occupied = np.full(raw.shape[1], np.nan, dtype=np.float64)
    for seed in seeds:
        ys = _trace_tophat_centerline(
            hat,
            search_roi,
            int(seed),
            occupied=occupied,
            half_win=half,
            min_sep=8,
            min_len=18,
        )
        if len(ys) < 18:
            continue
        for col, row in ys.items():
            occupied[int(col)] = float(row)
        vmask |= _rasterize_polyline(ys, search_roi, thickness_px=vessel_trace_px)
    if np.any(vmask):
        vmask = keep_n_longest_objects(
            vmask, n=n_vessel_objects, min_area=18
        )
    hat_vals = hat[search_roi]
    landmarks = np.zeros((0, 2), dtype=np.int64)
    if hat_vals.size and float(np.max(hat_vals)) > 0:
        fat = (hat >= float(np.percentile(hat_vals, 80))) & search_roi
        landmarks = _skeleton_junctions(skeletonize(fat))
        if landmarks.size and np.any(vmask):
            near = []
            for row, col in landmarks:
                r0, r1 = max(0, int(row) - 4), min(vmask.shape[0], int(row) + 5)
                c0, c1 = max(0, int(col) - 4), min(vmask.shape[1], int(col) + 5)
                if np.any(vmask[r0:r1, c0:c1]):
                    near.append((int(row), int(col)))
            landmarks = (
                np.asarray(near, dtype=np.int64)
                if near
                else np.zeros((0, 2), dtype=np.int64)
            )
            landmarks = _cluster_points(landmarks, radius_px=4)
    return _maps_from_mask(raw.shape, vmask, chamber, landmarks=landmarks)


def _dark_tophat_maps(
    raw: np.ndarray,
    *,
    invert_intensity: bool,
    method: str,
    clip_percentiles: tuple[float, float],
    hard_clip_percentiles: tuple[float, float] | None,
    tophat_radius_px: float,
    gaussian_sigma: float,
    vessel_percentile: float,
    background_sigma: float,
    mask_border_px: int,
    n_vessel_objects: int,
    chamber_rim_px: int = 12,
) -> VesselMaps:
    work = -raw if invert_intensity else raw
    chamber = chamber_mask(
        work, border_px=mask_border_px, rim_px=chamber_rim_px
    )
    if method in ("clip_tophat", "clip"):
        x = _rescale01(-work if not invert_intensity else work)
        x = _rescale01(white_tophat(x, footprint=_disk(tophat_radius_px)))
        x = _rescale01(_percentile_clip(x, clip_percentiles))
        if hard_clip_percentiles is not None:
            x = _rescale01(_percentile_clip(x, hard_clip_percentiles))
        if gaussian_sigma and gaussian_sigma > 0:
            x = _rescale01(gaussian_filter(x, float(gaussian_sigma)))
        vmask = (x > np.percentile(x, 70)) & chamber
        vmask = keep_n_longest_objects(vmask, n=n_vessel_objects, min_area=40)
        display = np.ones_like(x)
        display[vmask] = 1.0 - x[vmask]
        fit = np.zeros_like(x)
        fit[vmask] = x[vmask]
        return VesselMaps(
            display=display.astype(np.float32),
            fit=fit.astype(np.float32),
            vessel_mask=vmask,
            chamber_mask=chamber,
        )

    ridge = np.maximum(black_tophat(work, footprint=_disk(tophat_radius_px)), 0.0)
    if background_sigma and background_sigma > 0:
        flat = work - gaussian_filter(work, float(background_sigma))
        ridge = np.maximum(
            ridge, np.maximum(black_tophat(flat, footprint=_disk(tophat_radius_px)), 0.0)
        )
    ridge[~chamber] = 0.0
    if gaussian_sigma and gaussian_sigma > 0:
        ridge = gaussian_filter(ridge, float(gaussian_sigma))
        ridge[~chamber] = 0.0
    rvals = ridge[chamber]
    if rvals.size == 0 or float(np.max(rvals)) <= 0:
        ones = np.ones(raw.shape, dtype=np.float32)
        zeros = np.zeros(raw.shape, dtype=np.float32)
        return VesselMaps(ones, zeros, np.zeros(raw.shape, dtype=bool), chamber)
    thr = float(np.percentile(rvals, vessel_percentile))
    vmask = keep_n_longest_objects(
        (ridge >= thr) & chamber, n=n_vessel_objects, min_area=40
    )
    peak = float(np.max(rvals))
    fit = np.zeros_like(ridge)
    fit[vmask] = (ridge[vmask] - thr) / max(peak - thr, 1e-9)
    display = np.ones_like(ridge)
    display[vmask] = 1.0 - np.clip(fit[vmask], 0.0, 1.0)
    return VesselMaps(
        display=display.astype(np.float32),
        fit=fit.astype(np.float32),
        vessel_mask=vmask,
        chamber_mask=chamber,
    )


def compute_stability_score_map(
    trial_stack: np.ndarray,
    *,
    cfg: SessionRegisterConfig | None = None,
    method: str = "dark_and_stable",
    smooth_sigma: float = 1.2,
    chamber_rim_px: int = 12,
    mask_border_px: int = 8,
    roi_row_min: int = 38,
    roi_row_max: int = 82,
    roi_radius_px: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the stability score shown in the gallery (yellow ridges).

    Returns ``(mean_map, score_display, chamber, roi)`` where ``score_display``
    is smoothed and masked — the same array used for the stability heatmap panel.
    No thresholding, skeletonization, or tracing.
    """
    if cfg is not None:
        chamber_rim_px = cfg.chamber_rim_px
        mask_border_px = max(int(cfg.mask_border_px), 6)
        smooth_sigma = cfg.gaussian_sigma if cfg.gaussian_sigma > 0 else smooth_sigma
        roi_radius_px = float(cfg.roi_radius_px)

    stack = np.asarray(trial_stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"trial_stack must be (N, H, W), got shape {stack.shape}")
    N, H, W = stack.shape
    if N < 2:
        raise ValueError(f"Need at least 2 trials, got N={N}")

    mean_map = stack.mean(axis=0)
    std_map = stack.std(axis=0)

    if method in ("baseline_stability", "dark_and_stable"):
        cv = std_map / (np.abs(mean_map) + 1e-9)
        cv_norm = (cv - cv.min()) / (cv.max() - cv.min() + 1e-9)
        score = 1.0 - cv_norm
        if method == "dark_and_stable":
            mean_norm = (mean_map - mean_map.min()) / (
                mean_map.max() - mean_map.min() + 1e-9
            )
            score = score * (1.0 - mean_norm)
    elif method == "inter_trial_corr":
        flat = stack.reshape(N, H * W)
        mu = flat.mean(axis=1, keepdims=True)
        flat_c = flat - mu
        norms = np.linalg.norm(flat_c, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        flat_n = flat_c / norms
        col_sum = flat_n.sum(axis=0)
        col_sq = (flat_n ** 2).sum(axis=0)
        n_pairs = N * (N - 1) / 2.0
        score = ((col_sum ** 2 - col_sq) / (2.0 * n_pairs)).reshape(H, W)
    else:
        raise ValueError(
            f"Unknown method {method!r}; "
            "use 'dark_and_stable', 'baseline_stability', or 'inter_trial_corr'"
        )

    from scipy.ndimage import gaussian_filter as _gf

    score_s = _gf(score, float(smooth_sigma)) if smooth_sigma > 0 else score.copy()
    chamber = chamber_mask(mean_map, border_px=mask_border_px, rim_px=chamber_rim_px)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rows_g, cols_g = np.ogrid[:H, :W]
    disk_roi = np.hypot(rows_g - cy, cols_g - cx) <= float(roi_radius_px)
    row_roi = np.zeros((H, W), dtype=bool)
    row_roi[roi_row_min:roi_row_max, :] = True
    roi = disk_roi & row_roi & chamber
    score_s = score_s.copy()
    score_s[~roi] = 0.0
    return (
        mean_map.astype(np.float32),
        score_s.astype(np.float32),
        chamber,
        roi,
    )


def _score_ridge_crests(
    score: np.ndarray,
    roi: np.ndarray,
    *,
    n_ridges: int = 2,
    min_sep: int = 8,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """1-px yellow-line geometry: dark troughs of the VSD mean inside ROI.

    The transparent yellow overlay sits on the dark vessel rivers.  Score-map
    local maxima sit a few pixels *above* those rivers, so landmarks must
    follow intensity troughs, not score peaks.
    """
    from scipy.signal import find_peaks

    roi = np.asarray(roi, dtype=bool)
    height, width = roi.shape
    if image is not None:
        profile_src = -np.asarray(image, dtype=np.float64)  # peaks of -I = dark troughs
    else:
        profile_src = np.asarray(score, dtype=np.float64)
    crest = np.zeros((height, width), dtype=bool)
    for col in range(width):
        ok = roi[:, col]
        if not np.any(ok):
            continue
        profile = np.where(ok, profile_src[:, col], -np.inf)
        finite = np.isfinite(profile) & ok
        if not np.any(finite):
            continue
        span = float(np.max(profile[finite]) - np.min(profile[finite]))
        prom = 0.12 * span if span > 0 else 0.0
        peaks, _ = find_peaks(
            profile, distance=max(int(min_sep), 2), prominence=max(prom, 0.0)
        )
        if peaks.size == 0:
            peaks, _ = find_peaks(profile, distance=max(int(min_sep), 2))
        if peaks.size == 0:
            continue
        # Prefer actually-dark troughs (below column median intensity)
        if image is not None:
            col_med = float(np.median(image[ok, col]))
            dark_ok = [int(p) for p in peaks if float(image[int(p), col]) <= col_med]
            if dark_ok:
                peaks = np.asarray(dark_ok, dtype=np.int64)
        order = np.argsort(profile[peaks])[::-1]
        for idx in order[: max(int(n_ridges), 1)]:
            row = int(peaks[idx])
            val = float(profile[row])
            if not (np.isfinite(val) and val > -np.inf):
                continue
            if row <= 0 or row >= height - 1:
                continue
            if not (bool(ok[row - 1]) and bool(ok[row]) and bool(ok[row + 1])):
                continue
            crest[row, col] = True
    return crest


def _link_column_peaks(
    crest: np.ndarray,
    *,
    n_ridges: int = 2,
    max_jump: int = 10,
) -> list[dict[int, int]]:
    """Connect per-column troughs into ``n_ridges`` left–right polylines."""
    _height, width = crest.shape
    peaks = [np.where(crest[:, col])[0] for col in range(width)]
    two = [c for c in range(width) if peaks[c].size >= max(int(n_ridges), 1)]
    if two:
        start_col = two[len(two) // 2]
    else:
        start_col = int(np.argmax([p.size for p in peaks]))
    start_rows = np.asarray(peaks[start_col], dtype=np.int64)
    if start_rows.size == 0:
        return []
    start_rows = np.sort(start_rows)[: max(int(n_ridges), 1)]
    tracks: list[dict[int, int]] = [{int(start_col): int(r)} for r in start_rows]

    def _step(src_col: int, dst_col: int) -> None:
        for track in tracks:
            if src_col not in track:
                continue
            r0 = track[src_col]
            cands = peaks[dst_col]
            if cands.size == 0:
                continue
            j = int(np.argmin(np.abs(cands.astype(np.int64) - r0)))
            row = int(cands[j])
            if abs(row - int(r0)) > max_jump:
                continue
            taken = {tr.get(dst_col) for tr in tracks}
            if row in taken:
                continue
            track[dst_col] = row

    for col in range(start_col + 1, width):
        _step(col - 1, col)
    for col in range(start_col - 1, -1, -1):
        _step(col + 1, col)

    filled: list[dict[int, int]] = []
    for track in tracks:
        if len(track) < 8:
            continue
        cols = np.array(sorted(track), dtype=np.int64)
        rows = np.array([track[int(c)] for c in cols], dtype=np.float64)
        full = np.arange(int(cols[0]), int(cols[-1]) + 1)
        yv = np.interp(full.astype(np.float64), cols.astype(np.float64), rows)
        filled.append({int(c): int(round(float(y))) for c, y in zip(full, yv)})
    return filled


def _tracks_from_column_troughs(
    crest: np.ndarray,
    *,
    min_sep: int = 6,
    min_len: int = 8,
) -> tuple[list[dict[int, int]], dict[int, int], dict[int, int]]:
    """Upper/lower rivers from two troughs per column (no greedy linking).

    Returns ``(tracks, detected_upper, detected_lower)``.  Detected dicts are
    only columns that actually had two troughs — not interpolated gaps.
    Stations must sit on those columns so a point cannot float off the river.
    """
    _height, width = crest.shape
    upper: dict[int, int] = {}
    lower: dict[int, int] = {}
    singles: dict[int, int] = {}
    for col in range(width):
        rows = np.sort(np.where(crest[:, col])[0])
        if rows.size == 0:
            continue
        if rows.size >= 2:
            r0, r1 = int(rows[0]), int(rows[-1])
            if abs(r1 - r0) < int(min_sep):
                singles[col] = int(round(0.5 * (r0 + r1)))
                continue
            upper[col] = min(r0, r1)
            lower[col] = max(r0, r1)
        else:
            singles[col] = int(rows[0])

    def _fill(track: dict[int, int]) -> dict[int, int]:
        if len(track) < 4:
            return {}
        cols = np.array(sorted(track), dtype=np.int64)
        rows = np.array([track[int(c)] for c in cols], dtype=np.float64)
        full = np.arange(int(cols[0]), int(cols[-1]) + 1)
        yv = np.interp(full.astype(np.float64), cols.astype(np.float64), rows)
        return {int(c): int(round(float(y))) for c, y in zip(full, yv)}

    def _reject_jumps(track: dict[int, int], max_dev: int = 8) -> dict[int, int]:
        if len(track) < 5:
            return dict(track)
        cols = np.array(sorted(track), dtype=np.int64)
        rows = np.array([track[int(c)] for c in cols], dtype=np.float64)
        odd = int(min(7, len(rows) if len(rows) % 2 == 1 else len(rows) - 1))
        odd = max(odd, 3)
        filt = median_filter(rows, size=odd)
        return {
            int(c): int(r)
            for c, r, f in zip(cols, rows, filt)
            if abs(float(r) - float(f)) <= float(max_dev)
        }

    def _clip_band(track: dict[int, int], halfwidth: float = 12.0) -> dict[int, int]:
        """Drop detections that jumped to the other river."""
        if len(track) < 4:
            return dict(track)
        med = float(np.median(list(track.values())))
        return {
            int(c): int(r)
            for c, r in track.items()
            if abs(float(r) - med) <= float(halfwidth)
        }

    upper = _clip_band(_reject_jumps(upper))
    lower = _clip_band(_reject_jumps(lower))
    upper_f = _fill(upper)
    lower_f = _fill(lower)
    for col, row in singles.items():
        d_u = abs(upper_f[col] - row) if col in upper_f else 1e9
        d_l = abs(lower_f[col] - row) if col in lower_f else 1e9
        if min(d_u, d_l) > 12:
            continue
        if d_u <= d_l:
            upper[col] = int(row)
        else:
            lower[col] = int(row)
    upper = _clip_band(_reject_jumps(upper))
    lower = _clip_band(_reject_jumps(lower))
    tracks = [t for t in (_fill(upper), _fill(lower)) if len(t) >= int(min_len)]
    return _order_tracks_upper_lower(tracks), dict(upper), dict(lower)


def _order_tracks_upper_lower(tracks: list[dict[int, int]]) -> list[dict[int, int]]:
    """Upper curve first (smaller mean row)."""
    return sorted(
        tracks,
        key=lambda t: float(np.mean(list(t.values()))) if t else 0.0,
    )


def _homologous_stations(
    tracks: list[dict[int, int]],
    *,
    n_landmarks: int,
    row_lo: int,
    row_hi: int,
    detected_upper: dict[int, int] | None = None,
    detected_lower: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the same left–right stations on upper/lower curves.

    Landmark id is ``(curve_id, station_k)`` with curve 0 = upper, 1 = lower.
    Prefer columns where **both** rivers were actually detected (not a
    gap-fill).  A fork (tracks within 3 px) is tagged ``(-1, 0)``.
    """
    tracks = _order_tracks_upper_lower(tracks)
    if not tracks:
        empty_p = np.zeros((0, 2), dtype=np.int64)
        empty_i = np.zeros((0, 2), dtype=np.int64)
        return empty_p, empty_i
    n_curves = len(tracks)
    n_per = max(2, int(n_landmarks) // max(n_curves, 1))
    pts: list[tuple[int, int]] = []
    ids: list[tuple[int, int]] = []

    row_of = list(tracks[:2]) if n_curves >= 2 else list(tracks[:1])
    overlap: list[int] = []
    if n_curves >= 2:
        overlap = sorted(set(tracks[0]) & set(tracks[1]))

    def _keep(row: int) -> bool:
        return int(row_lo) <= int(row) < int(row_hi)

    def _snap(cid: int, col: int, row: int) -> int:
        det = detected_upper if cid == 0 else detected_lower
        if det and col in det and abs(int(det[col]) - int(row)) <= 4:
            return int(det[col])
        return int(row)

    # Stay inside the overlap, off the disk edge (U0 on 100718a sat on a gap).
    edge = 0.18

    if len(overlap) >= 8:
        cols = np.asarray(overlap, dtype=np.int64)
        i0 = int(round(edge * (cols.size - 1)))
        i1 = int(round((1.0 - edge) * (cols.size - 1)))
        cols = cols[i0 : i1 + 1]
        station_idx = np.unique(
            np.linspace(0, cols.size - 1, n_per).round().astype(np.int64)
        )
        for k, ii in enumerate(station_idx):
            col = int(cols[int(ii)])
            pair: list[tuple[int, int, int, int]] = []
            for cid in range(min(2, len(row_of))):
                pair.append((cid, k, _snap(cid, col, int(row_of[cid][col])), col))
            if all(_keep(row) for _cid, _k, row, _col in pair) and abs(pair[0][2] - pair[1][2]) >= 8:
                for cid, kk, row, cc in pair:
                    pts.append((row, cc))
                    ids.append((cid, kk))
        gap = np.array(
            [abs(int(row_of[0][c]) - int(row_of[1][c])) for c in overlap if c in row_of[0] and c in row_of[1]],
            dtype=np.int64,
        )
        if gap.size:
            j = int(np.argmin(gap))
            if int(gap[j]) <= 3:
                col = int(overlap[j])
                row = int(round(0.5 * (row_of[0][col] + row_of[1][col])))
                if _keep(row):
                    pts.append((row, col))
                    ids.append((-1, 0))
    else:
        for cid, tr in enumerate(tracks):
            cols = np.asarray(sorted(tr), dtype=np.int64)
            if cols.size < 8:
                continue
            i0 = int(round(edge * (cols.size - 1)))
            i1 = int(round((1.0 - edge) * (cols.size - 1)))
            cols = cols[i0 : i1 + 1]
            station_idx = np.unique(
                np.linspace(0, cols.size - 1, n_per).round().astype(np.int64)
            )
            for k, ii in enumerate(station_idx):
                col = int(cols[int(ii)])
                row = int(tr[col])
                if _keep(row):
                    pts.append((row, col))
                    ids.append((cid, k))

    if not pts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pts, dtype=np.int64), np.asarray(ids, dtype=np.int64)


def _landmarks_from_polylines(
    tracks: list[dict[int, int]],
    intensity: np.ndarray,
    *,
    min_distance: int = 6,
    meet_px: int = 3,
    sample_spacing: int = 8,
    max_row: int | None = None,
) -> np.ndarray:
    """Landmarks on deep vessel curves: darkest points, spaced samples, junctions.

    Skip the chamber-bottom cut (``max_row``).  Extra spaced samples along each
    trough give better coverage for rigid registration without leaving the curve.
    """
    from scipy.signal import find_peaks

    image = np.asarray(intensity, dtype=np.float64)
    height, width = image.shape
    row_lim = int(max_row) if max_row is not None else height
    pts: list[tuple[int, int]] = []

    def _keep(row: int, col: int) -> bool:
        return 0 <= int(row) < row_lim and 0 <= int(col) < width

    dist = max(int(min_distance), 2)
    spacing = max(int(sample_spacing), 4)
    for ys in tracks:
        cols = np.array(sorted(ys), dtype=np.int64)
        rows = np.array([ys[int(c)] for c in cols], dtype=np.int64)
        if cols.size < 8:
            continue
        depth = np.array(
            [
                float(image[min(max(int(r), 0), height - 1), min(max(int(c), 0), width - 1)])
                for r, c in zip(rows, cols)
            ],
            dtype=np.float64,
        )
        deep, _ = find_peaks(-depth, distance=dist)
        idxs = set(int(i) for i in deep)
        idxs.add(int(np.argmin(depth)))
        n = int(cols.size)
        lo_i, hi_i = 3, max(n - 4, 3)
        for i in range(lo_i, hi_i + 1, spacing):
            idxs.add(int(i))
        for i in idxs:
            row, col = int(rows[i]), int(cols[i])
            if _keep(row, col):
                pts.append((row, col))
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            shared = set(a) & set(b)
            best: tuple[int, int, float] | None = None
            for col in shared:
                d = abs(int(a[col]) - int(b[col]))
                if d <= meet_px and (best is None or d < best[2]):
                    row = int(round(0.5 * (a[col] + b[col])))
                    best = (row, int(col), float(d))
            if best is not None and _keep(best[0], best[1]):
                pts.append((best[0], best[1]))
    if not pts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pts, dtype=np.int64)


def _spaced_deep_landmarks(
    troughs: np.ndarray,
    intensity: np.ndarray,
    *,
    n_target: int = 10,
    min_sep: int = 6,
    max_row: int | None = None,
) -> np.ndarray:
    """Darkest trough pixels, greedily spaced, excluding the chamber bottom."""
    mask = np.asarray(troughs, dtype=bool)
    image = np.asarray(intensity, dtype=np.float64)
    rr, cc = np.where(mask)
    if rr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if max_row is not None:
        keep = rr < int(max_row)
        rr, cc = rr[keep], cc[keep]
    if rr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    depth = image[rr, cc]
    order = np.argsort(depth)  # darkest first
    picked: list[tuple[int, int]] = []
    sep2 = int(min_sep) ** 2
    for i in order:
        row, col = int(rr[i]), int(cc[i])
        if any((row - pr) ** 2 + (col - pc) ** 2 < sep2 for pr, pc in picked):
            continue
        picked.append((row, col))
        if len(picked) >= int(n_target):
            break
    return np.asarray(picked, dtype=np.int64)


def _select_n_landmarks(
    pts: np.ndarray,
    intensity: np.ndarray,
    *,
    n: int,
    min_sep: int = 5,
) -> np.ndarray:
    """Keep up to ``n`` darkest, spatially spread landmarks."""
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pts = np.asarray(pts, dtype=np.int64).reshape(-1, 2)
    image = np.asarray(intensity, dtype=np.float64)
    h, w = image.shape
    rows = np.clip(pts[:, 0], 0, h - 1)
    cols = np.clip(pts[:, 1], 0, w - 1)
    depth = image[rows, cols]
    order = np.argsort(depth)
    picked: list[tuple[int, int]] = []
    sep2 = int(min_sep) ** 2
    for i in order:
        row, col = int(pts[i, 0]), int(pts[i, 1])
        if any((row - pr) ** 2 + (col - pc) ** 2 < sep2 for pr, pc in picked):
            continue
        picked.append((row, col))
        if len(picked) >= int(n):
            break
    return np.asarray(picked, dtype=np.int64)


def _snap_to_crest(pts: np.ndarray, crest: np.ndarray) -> np.ndarray:
    """Move each point onto the nearest 1-px crest pixel."""
    if pts.size == 0 or not np.any(crest):
        return np.zeros((0, 2), dtype=np.int64)
    cr, cc = np.where(crest)
    out: list[tuple[int, int]] = []
    for row, col in np.asarray(pts).reshape(-1, 2):
        d2 = (cr - int(row)) ** 2 + (cc - int(col)) ** 2
        i = int(np.argmin(d2))
        out.append((int(cr[i]), int(cc[i])))
    return np.asarray(out, dtype=np.int64)


@dataclass(frozen=True)
class StabilityRidgeResult:
    """Vessels = yellow stability ridges; landmarks sit on those crests."""

    mean_map: np.ndarray
    score_display: np.ndarray
    chamber_mask: np.ndarray
    roi: np.ndarray
    ridge_mask: np.ndarray
    landmarks: np.ndarray
    ridge_threshold: float
    method: str
    n_trials: int
    landmark_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )


def save_named_landmarks(
    path: Path,
    session: str,
    landmarks: np.ndarray,
    landmark_ids: np.ndarray,
) -> Path:
    """Write homologous stations so the anchor can be reused unchanged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(landmarks, dtype=np.int64).reshape(-1, 2)
    ids = np.asarray(landmark_ids, dtype=np.int64).reshape(-1, 2)
    payload = {
        "session": str(session),
        "landmarks": [
            {"id": [int(a), int(b)], "row": int(r), "col": int(c)}
            for (r, c), (a, b) in zip(pts, ids)
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_named_landmarks(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(path)
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("landmarks") or []
    if not rows:
        return None
    pts = np.array([[int(p["row"]), int(p["col"])] for p in rows], dtype=np.int64)
    ids = np.array([p["id"] for p in rows], dtype=np.int64).reshape(-1, 2)
    return pts, ids


def resolve_anchor_landmarks(
    result: StabilityRidgeResult,
    path: Path,
    session: str,
    *,
    overwrite: bool = False,
    write_if_missing: bool = True,
) -> tuple[StabilityRidgeResult, str]:
    """Load frozen stations if present; optionally write them on first run."""
    path = Path(path)
    loaded = None if overwrite else load_named_landmarks(path)
    if loaded is not None:
        pts, ids = loaded
        return replace(result, landmarks=pts, landmark_ids=ids), f"loaded {path}"
    if not write_if_missing:
        return result, f"extracted {session}"
    save_named_landmarks(path, session, result.landmarks, result.landmark_ids)
    return result, f"wrote {path}"


def stability_ridge_landmarks(
    trial_stack: np.ndarray,
    *,
    cfg: SessionRegisterConfig | None = None,
    method: str = "dark_and_stable",
    smooth_sigma: float = 1.2,
    ridge_percentile: float = 80.0,
    chamber_rim_px: int = 12,
    mask_border_px: int = 8,
    roi_row_min: int = 38,
    roi_row_max: int = 82,
    roi_radius_px: float = 40.0,
    junction_cluster_px: int = 4,
    n_ridges: int = 2,
    n_landmarks: int | None = None,
) -> StabilityRidgeResult:
    """Homologous stations on the two deep vessel curves (upper / lower).

    Points are labelled ``(curve_id, station_k)`` so another session can match
    the same landscape without nearest-neighbour guessing.  Chamber-bottom
    and ROI-edge pixels are dropped.
    """
    _ = ridge_percentile
    if cfg is not None:
        chamber_rim_px = cfg.chamber_rim_px
        mask_border_px = max(int(cfg.mask_border_px), 6)
        smooth_sigma = cfg.gaussian_sigma if cfg.gaussian_sigma > 0 else smooth_sigma
        roi_radius_px = float(cfg.roi_radius_px)
        n_ridges = int(cfg.n_vessel_objects)
        if n_landmarks is None:
            n_landmarks = int(cfg.n_landmarks)
    if n_landmarks is None:
        n_landmarks = 6

    stack = np.asarray(trial_stack, dtype=np.float64)
    n_trials = stack.shape[0]

    mean_map, score_s, chamber, roi = compute_stability_score_map(
        stack,
        method=method,
        smooth_sigma=smooth_sigma,
        chamber_rim_px=chamber_rim_px,
        mask_border_px=mask_border_px,
        roi_row_min=roi_row_min,
        roi_row_max=roi_row_max,
        roi_radius_px=roi_radius_px,
    )

    troughs = _score_ridge_crests(
        score_s, roi, n_ridges=n_ridges, min_sep=8, image=mean_map
    )
    from scipy.ndimage import binary_erosion as _erode

    interior = _erode(np.asarray(roi, dtype=bool), iterations=4)
    troughs_use = troughs & interior
    tracks, detected_upper, detected_lower = _tracks_from_column_troughs(
        troughs_use, min_sep=6
    )
    keep_row_lo = int(roi_row_min) + 4
    keep_row_hi = int(roi_row_max) - 4
    landmarks, landmark_ids = _homologous_stations(
        tracks,
        n_landmarks=int(n_landmarks),
        row_lo=keep_row_lo,
        row_hi=keep_row_hi,
        detected_upper=detected_upper,
        detected_lower=detected_lower,
    )
    _ = junction_cluster_px
    ridge_mask = np.zeros_like(troughs_use)
    for tr in tracks:
        ridge_mask |= _rasterize_polyline(tr, interior, thickness_px=1)
    if not np.any(ridge_mask):
        ridge_mask = troughs_use

    return StabilityRidgeResult(
        mean_map=mean_map,
        score_display=score_s,
        chamber_mask=chamber,
        roi=roi,
        ridge_mask=ridge_mask,
        landmarks=landmarks,
        landmark_ids=landmark_ids,
        ridge_threshold=0.0,
        method=method,
        n_trials=int(n_trials),
    )


def correlation_vessel_map(
    trial_stack: np.ndarray,
    *,
    cfg: SessionRegisterConfig | None = None,
    method: str = "dark_and_stable",
    smooth_sigma: float = 1.2,
    vessel_percentile: float = 80.0,
    n_vessel_objects: int = 2,
    chamber_rim_px: int = 12,
    mask_border_px: int = 8,
    roi_row_min: int = 38,
    roi_row_max: int = 82,
    roi_radius_px: float = 40.0,
    vessel_trace_px: int = 1,
    junction_percentile: float = 80.0,
    junction_cluster_px: int = 4,
    min_junction_vessel_dist: int = 5,
) -> tuple[np.ndarray, "VesselMaps", dict]:
    """Stability ridges = vessels; skeleton junctions on ridges = landmarks."""
    _ = (n_vessel_objects, vessel_trace_px, junction_percentile, min_junction_vessel_dist)
    result = stability_ridge_landmarks(
        trial_stack,
        cfg=cfg,
        method=method,
        smooth_sigma=smooth_sigma,
        ridge_percentile=vessel_percentile,
        chamber_rim_px=chamber_rim_px,
        mask_border_px=mask_border_px,
        roi_row_min=roi_row_min,
        roi_row_max=roi_row_max,
        roi_radius_px=roi_radius_px,
        junction_cluster_px=junction_cluster_px,
    )
    vessel_maps = _maps_from_mask(
        result.mean_map.shape,
        result.ridge_mask,
        result.chamber_mask,
        landmarks=result.landmarks,
    )
    debug = {
        "score": result.score_display,
        "smoothed": result.score_display,
        "chamber": result.chamber_mask,
        "roi": result.roi,
        "ridge_mask": result.ridge_mask,
        "threshold": result.ridge_threshold,
        "method": result.method,
        "n_trials": result.n_trials,
        "mean_map": result.mean_map,
    }
    return result.score_display, vessel_maps, debug


def emphasize_vessels(
    image: np.ndarray,
    *,
    invert_intensity: bool = False,
    vessel_method: str = "meijering",
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
    hard_clip_percentiles: tuple[float, float] | None = None,
    tophat_radius_px: float = 6.0,
    gaussian_sigma: float = 1.5,
    cfg: SessionRegisterConfig | None = None,
) -> np.ndarray:
    """White-field vessel map (dark ridges). Use ``extract_vessels`` for the fit image."""
    maps = extract_vessels(
        image,
        cfg=cfg,
        invert_intensity=invert_intensity,
        vessel_method=vessel_method,
        clip_percentiles=clip_percentiles,
        hard_clip_percentiles=hard_clip_percentiles,
        tophat_radius_px=tophat_radius_px,
        gaussian_sigma=gaussian_sigma,
    )
    return maps.display


def emphasize_vessels_from_config(
    image: np.ndarray, cfg: SessionRegisterConfig
) -> np.ndarray:
    return extract_vessels(image, cfg=cfg).display


def vessel_params_dict(cfg: SessionRegisterConfig) -> dict[str, Any]:
    return {
        "invert_intensity": cfg.invert_intensity,
        "vessel_method": cfg.vessel_method,
        "clip_percentiles": list(cfg.clip_percentiles),
        "hard_clip_percentiles": (
            None
            if cfg.hard_clip_percentiles is None
            else list(cfg.hard_clip_percentiles)
        ),
        "tophat_radius_px": cfg.tophat_radius_px,
        "gaussian_sigma": cfg.gaussian_sigma,
        "vessel_percentile": cfg.vessel_percentile,
        "background_sigma": cfg.background_sigma,
        "n_vessel_objects": cfg.n_vessel_objects,
        "n_landmarks": cfg.n_landmarks,
        "vessel_start_frame": cfg.vessel_start_frame,
        "vessel_end_frame": cfg.vessel_end_frame,
        "hysteresis_percentiles": list(cfg.hysteresis_percentiles),
        "chamber_rim_px": cfg.chamber_rim_px,
        "roi_radius_px": cfg.roi_radius_px,
        "vessel_trace_px": cfg.vessel_trace_px,
    }
