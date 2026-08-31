"""Schira sampling of a visual stimulus onto cortical ``w`` or the VSDI grid.

Two stages (keep them separate):

1. **Schira only** — sample the stimulus onto a regular grid in model
   cortical coordinates ``w = u + i v`` (no camera registration).
2. **Registration** — map that ``w`` onto VSD camera pixels via
   :class:`~src.retinotopy.affine.CorticalAffine` (translation / rotation /
   scale). Phase 0 overlays use (2); Schira-shape checks use (1).

For each cortical sample, look up the visual-field location (inverse map)
and bilinear-sample the raw stimulus image.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, map_coordinates

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import (
    SchiraParams,
    cartesian_to_polar,
    forward_schira,
    inverse_schira,
    polar_to_cartesian,
)
from src.retinotopy.visual_field import (
    cartesian_deg_to_image_px,
    image_px_to_cartesian_deg,
    stimulus_ink_mask,
)
from src.stimuli.render import RenderConfig


def forward_ink_cloud_w(
    stimulus_rgb: np.ndarray,
    *,
    params: SchiraParams,
    render_cfg: RenderConfig | None = None,
    background_gray: int | None = None,
    threshold: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-map stimulus ink pixels to cortical ``w``.

    Returns
    -------
    u, v : float arrays
        ``Re(w)``, ``Im(w)`` for each ink pixel (NaNs dropped).
    w : complex array
        Same points as a complex vector.
    """
    cfg = render_cfg or RenderConfig()
    bg = int(cfg.background_gray if background_gray is None else background_gray)
    ink = stimulus_ink_mask(
        stimulus_rgb, background_gray=bg, threshold=threshold
    )
    yy, xx = np.where(ink)
    if yy.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty.astype(np.complex128)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), cfg
    )
    ecc, theta = cartesian_to_polar(x_deg, y_deg)
    w = forward_schira(ecc, theta, params)
    ok = np.isfinite(w.real) & np.isfinite(w.imag)
    w = w[ok]
    return w.real.copy(), w.imag.copy(), w


def _cortical_bbox_from_w(
    w: np.ndarray,
    *,
    margin_frac: float = 0.2,
    min_half_span: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned ``(u_min, u_max), (v_min, v_max)`` with margin."""
    u = np.asarray(w, dtype=np.complex128).real
    v = np.asarray(w, dtype=np.complex128).imag
    ok = np.isfinite(u) & np.isfinite(v)
    if not np.any(ok):
        raise ValueError("No finite cortical w samples for bbox")
    u_ok, v_ok = u[ok], v[ok]
    u0, u1 = float(u_ok.min()), float(u_ok.max())
    v0, v1 = float(v_ok.min()), float(v_ok.max())
    du = max(u1 - u0, min_half_span)
    dv = max(v1 - v0, min_half_span)
    # Keep a little area even for near-degenerate clouds.
    span = max(du, dv, min_half_span)
    mu = 0.5 * (u0 + u1)
    mv = 0.5 * (v0 + v1)
    pad = float(margin_frac) * span
    half = 0.5 * span + pad
    return (mu - half, mu + half), (mv - half, mv + half)


def sample_stimulus_onto_cortical_w(
    stimulus_rgb: np.ndarray,
    *,
    params: SchiraParams,
    render_cfg: RenderConfig | None = None,
    grid_size: int = 256,
    margin_frac: float = 0.25,
    max_ecc_deg: float | None = None,
    u_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Warp a stimulus onto a regular grid in model cortical ``w`` (no affine).

    Builds a square patch in ``(u, v) = (Re(w), Im(w))`` around the
    forward-mapped ink cloud (or uses the supplied ranges), then inverse-
    Schira samples the stimulus at each grid point.

    Returns
    -------
    warped_rgb : uint8 ``(G, G, 3)``
    valid : bool ``(G, G)``
    u_range, v_range : axis limits for plotting (``imshow`` extent)
    """
    cfg = render_cfg or RenderConfig()
    rgb = np.asarray(stimulus_rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 stimulus, got {rgb.shape}")
    if grid_size < 8:
        raise ValueError(f"grid_size must be >= 8, got {grid_size}")

    if u_range is None or v_range is None:
        _u, _v, w_cloud = forward_ink_cloud_w(rgb, params=params, render_cfg=cfg)
        if w_cloud.size == 0:
            # Empty ink: still return a small patch around the fovea.
            from src.retinotopy.schira import fovea_w

            wf = fovea_w(params)
            u_range = (float(wf.real) - 0.5, float(wf.real) + 0.5)
            v_range = (float(wf.imag) - 0.5, float(wf.imag) + 0.5)
        else:
            u_range, v_range = _cortical_bbox_from_w(
                w_cloud, margin_frac=margin_frac
            )

    u0, u1 = float(u_range[0]), float(u_range[1])
    v0, v1 = float(v_range[0]), float(v_range[1])
    # imshow extent uses pixel *edges*; sample at pixel centers.
    u_edges = np.linspace(u0, u1, grid_size + 1, dtype=np.float64)
    v_edges = np.linspace(v1, v0, grid_size + 1, dtype=np.float64)  # top→bottom
    u_c = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_c = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(u_c, v_c, indexing="xy")
    w = uu + 1j * vv

    ecc, polar = inverse_schira(w, params, min_real_z=0.0)
    x_deg, y_deg = polar_to_cartesian(ecc, polar)
    x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)

    hs, ws = rgb.shape[:2]
    bg = int(cfg.background_gray)
    ecc_ok = np.isfinite(ecc) & np.isfinite(polar)
    if max_ecc_deg is not None:
        ecc_ok &= ecc <= float(max_ecc_deg)
    in_canvas = (
        ecc_ok
        & (x_px >= -0.5)
        & (y_px >= -0.5)
        & (x_px < ws - 0.5)
        & (y_px < hs - 0.5)
    )

    warped = np.full((grid_size, grid_size, 3), bg, dtype=np.uint8)
    if not np.any(in_canvas):
        return warped, in_canvas, (u0, u1), (v0, v1)

    coords = np.vstack([y_px.ravel(), x_px.ravel()])
    for channel in range(3):
        sampled = map_coordinates(
            rgb[:, :, channel].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=float(bg),
        ).reshape(grid_size, grid_size)
        plane = np.full((grid_size, grid_size), float(bg))
        plane[in_canvas] = sampled[in_canvas]
        warped[:, :, channel] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
    return warped, in_canvas, (u0, u1), (v0, v1)


def cortical_w_grid(
    spatial_size: tuple[int, int],
    affine: CorticalAffine,
    params: SchiraParams,
) -> np.ndarray:
    """Complex ``w`` at every VSD pixel center, shape ``(H, W)``."""
    height, width = spatial_size
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    return affine.pixel_to_w(grid_r, grid_c, params)


def sample_stimulus_onto_vsd_grid(
    stimulus_rgb: np.ndarray,
    *,
    params: SchiraParams,
    affine: CorticalAffine,
    render_cfg: RenderConfig | None = None,
    spatial_size: tuple[int, int] = (100, 100),
    max_ecc_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a stimulus RGB image into VSD pixel space.

    Returns
    -------
    warped_rgb : uint8 ``(H, W, 3)``
        Background gray where the inverse map is undefined or outside the
        canvas.
    valid : bool ``(H, W)``
        True where the pixel samples a defined right-hemifield location
        inside the stimulus canvas.
    """
    cfg = render_cfg or RenderConfig()
    rgb = np.asarray(stimulus_rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 stimulus, got {rgb.shape}")
    height, width = spatial_size
    bg = int(cfg.background_gray)
    w = cortical_w_grid(spatial_size, affine, params)
    ecc, polar = inverse_schira(w, params, min_real_z=0.0)
    x_deg, y_deg = polar_to_cartesian(ecc, polar)
    x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)

    hs, ws = rgb.shape[:2]
    ecc_ok = np.isfinite(ecc) & np.isfinite(polar)
    if max_ecc_deg is not None:
        ecc_ok &= ecc <= float(max_ecc_deg)
    in_canvas = (
        ecc_ok
        & (x_px >= -0.5)
        & (y_px >= -0.5)
        & (x_px < ws - 0.5)
        & (y_px < hs - 0.5)
    )

    warped = np.full((height, width, 3), bg, dtype=np.uint8)
    if not np.any(in_canvas):
        return warped, in_canvas

    coords = np.vstack([y_px.ravel(), x_px.ravel()])
    for channel in range(3):
        sampled = map_coordinates(
            rgb[:, :, channel].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=float(bg),
        ).reshape(height, width)
        plane = np.full((height, width), float(bg))
        plane[in_canvas] = sampled[in_canvas]
        warped[:, :, channel] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
    return warped, in_canvas


def fill_stimulus_ink_holes(
    stimulus_rgb: np.ndarray,
    *,
    background_gray: int = 128,
    threshold: float = 8.0,
    fill_value: int = 0,
    thicken_px: int = 2,
) -> np.ndarray:
    """Thicken stroke glyphs and fill closed interiors for overlay warps.

    Catalog letter BMPs are thin outlines. After a log-polar warp those
    strokes stay ~1 px thick and look like curves even when the forward
    map has 2D area. Dilating ink (and filling closed holes) before
    sampling makes Phase 0 overlays show letter support as filled /
    thick patches. Does not invent ink far from the glyph.
    """
    from scipy.ndimage import binary_dilation

    rgb = np.asarray(stimulus_rgb).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 stimulus, got {rgb.shape}")
    ink = stimulus_ink_mask(rgb, background_gray=background_gray, threshold=threshold)
    if thicken_px > 0:
        ink = binary_dilation(ink, iterations=int(thicken_px))
    filled = binary_fill_holes(ink)
    paint = filled | ink
    if np.any(paint):
        rgb[paint] = (int(fill_value), int(fill_value), int(fill_value))
    return rgb


def warped_ink_mask(
    warped_rgb: np.ndarray,
    valid: np.ndarray,
    *,
    background_gray: int = 128,
    threshold: float = 8.0,
) -> np.ndarray:
    """Stimulus geometry on the VSD grid (True = draw in black).

    Intensity threshold on the warped RGB (filled support), not an
    edge/contour detector.
    """
    ink = stimulus_ink_mask(
        warped_rgb, background_gray=background_gray, threshold=threshold
    )
    return ink & np.asarray(valid, dtype=bool)


def forward_ink_mask_on_vsd(
    stimulus_rgb: np.ndarray,
    *,
    params: SchiraParams,
    affine: CorticalAffine,
    render_cfg: RenderConfig | None = None,
    spatial_size: tuple[int, int] = (100, 100),
    background_gray: int | None = None,
    threshold: float = 8.0,
) -> np.ndarray:
    """Rasterize forward Schira ink through the camera affine onto VSD pixels.

    Matches thesis Fig. 13 style: thin stroke outlines (no dilate / hole-fill /
    inverse sampling). Each catalog ink pixel is mapped VF → ``w`` → ``(row, col)``
    and stamped onto the nearest camera pixel.
    """
    cfg = render_cfg or RenderConfig()
    _u, _v, w = forward_ink_cloud_w(
        stimulus_rgb,
        params=params,
        render_cfg=cfg,
        background_gray=background_gray,
        threshold=threshold,
    )
    height, width = int(spatial_size[0]), int(spatial_size[1])
    ink = np.zeros((height, width), dtype=bool)
    if w.size == 0:
        return ink
    rows, cols = affine.w_to_pixel(w, params)
    rr = np.rint(np.asarray(rows, dtype=np.float64)).astype(np.int64)
    cc = np.rint(np.asarray(cols, dtype=np.float64)).astype(np.int64)
    ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    ink[rr[ok], cc[ok]] = True
    return ink
