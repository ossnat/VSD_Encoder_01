"""Faithful port of Joanna ``trans_and_registration.m`` (2-point similarity → 100×100)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.transform import SimilarityTransform, warp

from shira_revised.src.legacy_schira import LegacySchiraParams
from shira_revised.src.legacy_trans_cortex import TransCortexResult, trans_cortex

BRAIN_SIZE = 100


@dataclass(frozen=True)
class RegistrationResult:
    """Chamber-registered stimulus (legacy ``out_image``) + OM intermediates."""

    out_image: np.ndarray  # (100, 100, 3) uint8, after low/left hem flips
    om: np.ndarray
    xcoord: np.ndarray
    ycoord: np.ndarray
    tform: SimilarityTransform
    src_om_xy: np.ndarray  # (2, 2) OM (col, row) of anchors (pre-flip space)
    dst_brain_xy: np.ndarray  # (2, 2) brain (col, row) used for fit (after lowH flip)


def check_coords(
    coords_temp: np.ndarray,
    xy: str,
    n_rows: int,
    n_cols: int,
    *,
    backwards: bool = False,
) -> np.ndarray:
    """Port of ``check_coords.m`` (``backwrds=0`` path used in registration)."""
    coords_temp = np.asarray(coords_temp, dtype=np.float64).ravel()
    out = np.zeros_like(coords_temp)
    if backwards:
        if xy == "x":
            out[:] = n_cols - coords_temp
        else:
            out[:] = coords_temp
        return out
    if xy == "x":
        for i, c in enumerate(coords_temp):
            out[i] = n_cols + c if c < 0 else c
    elif xy == "y":
        for i, c in enumerate(coords_temp):
            out[i] = abs(c) if c < 0 else n_rows - c
    else:
        raise ValueError("xy must be 'x' or 'y'")
    return out


def vf_pix_to_stim_xy(
    vf_row_pix: float,
    vf_col_pix: float,
    *,
    ppd: float,
    n_rows: int,
    n_cols: int,
    low_hemisphere: bool = True,
    left_hemisphere: bool = True,
) -> tuple[float, float]:
    """Map opt-point VF pixel → stimulus (col, row) indices used in ``trans_and_registration``.

    Mirrors ``header.m`` (deg + sign flips) then ``conv_pix_deg`` + ``check_coords``.
    """
    y_deg = float(vf_row_pix) / ppd
    x_deg = float(vf_col_pix) / ppd
    if low_hemisphere:
        y_deg = -abs(y_deg)
    if not left_hemisphere:
        x_deg = -abs(x_deg)
    x_pix = round(x_deg * ppd)
    y_pix = round(y_deg * ppd)
    x1 = float(check_coords([x_pix], "x", n_rows, n_cols)[0])
    y1 = float(check_coords([y_pix], "y", n_rows, n_cols)[0])
    return x1, y1


def _lookup_om_xy(
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    stim_col: float,
    stim_row: float,
) -> tuple[float, float]:
    """MATLAB 1-based ``[xcoord(row,col), ycoord(row,col)]`` → OM (col, row)."""
    r = int(round(stim_row)) - 1
    c = int(round(stim_col)) - 1
    if r < 0 or c < 0 or r >= xcoord.shape[0] or c >= xcoord.shape[1]:
        raise IndexError(
            f"anchor stim index (row={stim_row}, col={stim_col}) out of "
            f"xcoord shape {xcoord.shape}"
        )
    om_col = float(xcoord[r, c])
    om_row = float(ycoord[r, c])
    if not (np.isfinite(om_col) and np.isfinite(om_row)):
        raise ValueError(f"non-finite OM coords at stim ({stim_row},{stim_col})")
    return om_col, om_row


def estimate_nonreflective_similarity(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
) -> SimilarityTransform:
    """``fitgeotrans(..., 'nonreflectivesimilarity')`` for 2 points."""
    src = np.asarray(src_xy, dtype=np.float64).reshape(2, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(2, 2)
    # skimage ≥0.26 prefers from_estimate; fall back for older versions.
    if hasattr(SimilarityTransform, "from_estimate"):
        tform = SimilarityTransform.from_estimate(src, dst)
    else:
        tform = SimilarityTransform()
        if not tform.estimate(src, dst):
            raise RuntimeError("SimilarityTransform.estimate failed")
    return tform


def register_om_to_chamber(
    om: np.ndarray,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    *,
    vf_pix_anchors: list[tuple[float, float]],
    brain_anchors: list[tuple[float, float]],
    ppd: float = 35.0,
    n_stim_rows: int | None = None,
    n_stim_cols: int | None = None,
    low_hemisphere: bool = True,
    left_hemisphere: bool = True,
    brain_size: int = BRAIN_SIZE,
) -> RegistrationResult:
    """Register ``OM`` onto 100×100 chamber (legacy ``trans_and_registration`` body).

    Parameters
    ----------
    vf_pix_anchors
        ``[(row, col), ...]`` stimulus / opt_points VF pixels (1-based MATLAB).
    brain_anchors
        ``[(row, col), ...]`` VSD / BV pixels (1-based), matching each VF anchor.
    """
    if len(vf_pix_anchors) != 2 or len(brain_anchors) != 2:
        raise ValueError("need exactly 2 VF and 2 brain anchors")
    if n_stim_rows is None:
        n_stim_rows = int(xcoord.shape[0])
    if n_stim_cols is None:
        n_stim_cols = int(xcoord.shape[1])

    src_pts = []
    dst_pts = []
    for (vf_r, vf_c), (br, bc) in zip(vf_pix_anchors, brain_anchors):
        stim_col, stim_row = vf_pix_to_stim_xy(
            vf_r,
            vf_c,
            ppd=ppd,
            n_rows=n_stim_rows,
            n_cols=n_stim_cols,
            low_hemisphere=low_hemisphere,
            left_hemisphere=left_hemisphere,
        )
        om_col, om_row = _lookup_om_xy(xcoord, ycoord, stim_col, stim_row)
        src_pts.append([om_col, om_row])
        yb = float(br)
        xb = float(bc)
        if low_hemisphere:
            yb = brain_size - yb + 1.0
        if not left_hemisphere:
            xb = brain_size - xb + 1.0
        dst_pts.append([xb, yb])

    src_om_xy = np.asarray(src_pts, dtype=np.float64)
    dst_brain_xy = np.asarray(dst_pts, dtype=np.float64)
    tform = estimate_nonreflective_similarity(src_om_xy, dst_brain_xy)

    om_f = np.asarray(om, dtype=np.float64)
    if om_f.ndim == 2:
        om_f = om_f[..., None]
    channels = []
    for c in range(om_f.shape[2]):
        # MATLAB ``imwarp`` default is linear interpolation.
        warped = warp(
            om_f[:, :, c],
            inverse_map=tform.inverse,
            output_shape=(brain_size, brain_size),
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        channels.append(warped)
    out = np.stack(channels, axis=-1)
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)

    if low_hemisphere:
        out = np.flipud(out)
    if not left_hemisphere:
        out = np.fliplr(out)

    return RegistrationResult(
        out_image=out,
        om=om,
        xcoord=xcoord,
        ycoord=ycoord,
        tform=tform,
        src_om_xy=src_om_xy,
        dst_brain_xy=dst_brain_xy,
    )


def trans_and_register(
    im_rgb: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    params: LegacySchiraParams,
    *,
    vf_pix_anchors: list[tuple[float, float]],
    brain_anchors: list[tuple[float, float]],
    ppd: float = 35.0,
    low_hemisphere: bool = True,
    left_hemisphere: bool = True,
    quick_mode: bool = False,
    bg: int = 0,
) -> RegistrationResult:
    """``trans_cortex`` + chamber registration (full legacy path)."""
    tc: TransCortexResult = trans_cortex(
        im_rgb, m1, m2, params, quick_mode=quick_mode, bg=bg
    )
    if quick_mode:
        raise ValueError("quick_mode has no OM to register")
    return register_om_to_chamber(
        tc.om,
        tc.xcoord,
        tc.ycoord,
        vf_pix_anchors=vf_pix_anchors,
        brain_anchors=brain_anchors,
        ppd=ppd,
        n_stim_rows=im_rgb.shape[0],
        n_stim_cols=im_rgb.shape[1],
        low_hemisphere=low_hemisphere,
        left_hemisphere=left_hemisphere,
    )


def prepare_chamber_display(
    chamber: np.ndarray,
    *,
    darken: float = 0.52,
    contrast: float = 1.25,
    circular: bool = True,
) -> np.ndarray:
    """Dark circular chamber plate (thesis Fig. 12 style).

    Outside the chamber circle is black. Tissue is a mid/dark gray — a bit
    darker than the thesis scan, but vessels / V1–V2 anatomy stay visible.
    """
    ch = np.asarray(chamber)
    if ch.ndim == 3:
        g = ch[:, :, :3].astype(np.float64).mean(axis=2)
    else:
        g = ch.astype(np.float64)
    if g.max() <= 1.0:
        g = g * 255.0
    # mild contrast, then scale down (raw BV is bright ~195; thesis ~100–120)
    mid = float(np.median(g[g > 5])) if np.any(g > 5) else 128.0
    g = mid + (g - mid) * contrast
    g = np.clip(g * darken, 0, 255)
    h, w = g.shape
    rgb = np.stack([g, g, g], axis=-1)
    if circular:
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rad = min(h, w) / 2.0 - 0.5
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
        rgb = np.where(mask[..., None], rgb, 0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def om_points_to_chamber(
    om_xy: np.ndarray,
    tform: SimilarityTransform,
    *,
    low_hemisphere: bool = True,
    left_hemisphere: bool = True,
    brain_size: int = BRAIN_SIZE,
) -> np.ndarray:
    """Map OM ``(col, row)`` samples through registration → chamber ``(row, col)``."""
    from skimage.transform import matrix_transform

    src = np.asarray(om_xy, dtype=np.float64).reshape(-1, 2)
    dst = matrix_transform(src, tform.params)
    col0 = dst[:, 0] - 1.0
    row0 = dst[:, 1] - 1.0
    if low_hemisphere:
        row0 = (brain_size - 1) - row0
    if not left_hemisphere:
        col0 = (brain_size - 1) - col0
    return np.column_stack([row0, col0])


def blend_on_chamber(
    chamber: np.ndarray,
    registered: np.ndarray,
    *,
    ink_threshold: float = 200.0,
    ink_rgb: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.9,
    thin: bool = False,
    dark_chamber: bool = False,
) -> np.ndarray:
    """White-ink overlay of registered *lines* on BV (thesis Fig. 12 style).

    Stimulus OM includes gray fill (BG≈130); only bright ink (≥``ink_threshold``)
    is drawn so the chamber vessels stay visible under the grid.
    """
    from scipy.ndimage import binary_erosion

    if dark_chamber:
        base = prepare_chamber_display(chamber).astype(np.float64)
    else:
        ch = np.asarray(chamber)
        if ch.ndim == 2:
            base = np.stack([ch, ch, ch], axis=-1)
        elif ch.shape[2] == 4:
            base = ch[:, :, :3]
        else:
            base = ch[:, :, :3].copy()
        base = base.astype(np.float64)
        if base.max() <= 1.0:
            base = base * 255.0

    reg = np.asarray(registered)
    if reg.ndim == 2:
        ink = reg.astype(np.float64)
    else:
        ink = reg[:, :, 0].astype(np.float64)
    mask = ink >= ink_threshold
    if thin and np.any(mask):
        # Keep a 1-px skeleton so warped grid lines stay fine (not bold bands).
        mask = mask & ~binary_erosion(mask, iterations=1)
        if not np.any(mask):
            mask = ink >= ink_threshold

    out = base.copy()
    color = np.array(ink_rgb, dtype=np.float64)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _om_xy_at_vf_deg(
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    *,
    x_deg: float,
    y_deg: float,
    ppd: float,
) -> tuple[float, float]:
    """Lookup OM (col, row) at VF ``(x_deg, y_deg)`` with top-left fovea indexing."""
    # 1-based MATLAB row/col from degrees (lower field → positive row)
    col = int(round(abs(x_deg) * ppd)) + 1
    row = int(round(abs(y_deg) * ppd)) + 1
    col = min(max(col, 1), xcoord.shape[1])
    row = min(max(row, 1), xcoord.shape[0])
    return _lookup_om_xy(xcoord, ycoord, float(col), float(row))


def register_om_align_vm_to_lvm(
    om: np.ndarray,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    *,
    lvm_row: float = 36.0,
    vm_deg: tuple[float, float] = (0.6, 3.5),
    lvm_cols: tuple[float, float] = (18.0, 82.0),
    ppd: float = 35.0,
    low_hemisphere: bool = True,
    left_hemisphere: bool = True,
    brain_size: int = BRAIN_SIZE,
) -> RegistrationResult:
    """Register by pinning the transformed VF y-axis (VM) onto chamber LVM.

    Matches the Ayzenshtat / thesis Fig. 13a idea: predicted vertical meridian
    lies on the V1/V2 border (LVM). Two points on VF ``x=0`` at ``vm_deg``
    eccentricities map to ``(lvm_cols[i], lvm_row)`` on the final chamber
    (0-based row increasing downward).
    """
    y_near, y_far = vm_deg
    src = []
    for y in (y_near, y_far):
        src.append(_om_xy_at_vf_deg(xcoord, ycoord, x_deg=0.0, y_deg=y, ppd=ppd))
    src_om_xy = np.asarray(src, dtype=np.float64)

    # Destination in pre-flipud warp space (same convention as ``register_om_to_chamber``).
    dst = []
    for c_final in lvm_cols:
        r_final = float(lvm_row)  # 0-based desired row after flipud
        # After flipud: out[r] = warped[brain_size-1-r] → warped_row0 = brain_size-1-r_final
        # MATLAB-like 1-based dst used in fitgeotrans:
        if low_hemisphere:
            yb = (brain_size - 1 - r_final) + 1.0  # = brain_size - r_final
        else:
            yb = r_final + 1.0
        xb = float(c_final) + 1.0
        if not left_hemisphere:
            xb = brain_size - float(c_final)
        dst.append([xb, yb])
    dst_brain_xy = np.asarray(dst, dtype=np.float64)

    tform = estimate_nonreflective_similarity(src_om_xy, dst_brain_xy)
    om_f = np.asarray(om, dtype=np.float64)
    if om_f.ndim == 2:
        om_f = om_f[..., None]
    channels = []
    for c in range(om_f.shape[2]):
        warped = warp(
            om_f[:, :, c],
            inverse_map=tform.inverse,
            output_shape=(brain_size, brain_size),
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        channels.append(warped)
    out = np.clip(np.round(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)
    if low_hemisphere:
        out = np.flipud(out)
    if not left_hemisphere:
        out = np.fliplr(out)
    return RegistrationResult(
        out_image=out,
        om=om,
        xcoord=xcoord,
        ycoord=ycoord,
        tform=tform,
        src_om_xy=src_om_xy,
        dst_brain_xy=dst_brain_xy,
    )


def draw_dashed_polyline(
    rgb: np.ndarray,
    pts_rc: np.ndarray,
    *,
    color: tuple[int, int, int] = (220, 40, 40),
    dash: int = 3,
    gap: int = 2,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a dashed polyline on an RGB image. ``pts_rc`` is (N,2) as (row, col)."""
    out = np.asarray(rgb).copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    pts = np.asarray(pts_rc, dtype=np.float64)
    if pts.shape[0] < 2:
        return out
    # densify
    segs = []
    for i in range(len(pts) - 1):
        n = int(max(np.hypot(*(pts[i + 1] - pts[i])), 1))
        for t in np.linspace(0, 1, n + 1):
            segs.append((1 - t) * pts[i] + t * pts[i + 1])
    segs = np.asarray(segs)
    on = True
    count = 0
    for p in segs:
        count += 1
        if count > (dash if on else gap):
            on = not on
            count = 0
        if not on:
            continue
        r, c = int(round(p[0])), int(round(p[1]))
        for dr in range(-thickness, thickness + 1):
            for dc in range(-thickness, thickness + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = color
    return out


def lvm_horizontal_polyline(
    row: float,
    col0: float = 10.0,
    col1: float = 90.0,
    n: int = 40,
) -> np.ndarray:
    """Horizontal LVM polyline as (row, col) samples (thesis Fig. 13a style)."""
    cols = np.linspace(col0, col1, n)
    rows = np.full(n, row)
    return np.column_stack([rows, cols])
