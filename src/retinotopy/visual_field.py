"""Visual-field geometry matching ``src.stimuli.render``.

The encoding canvas is the **lower-right quadrant only**: fixation at the
top-left pixel, ``+x`` right of the vertical meridian, catalog ``+y`` up
(so downward targets have **negative** ``y_deg``). Image row grows
downward, hence ``y_px = -y_deg * pixels_per_deg``.
"""

from __future__ import annotations

import numpy as np

from src.retinotopy.schira import cartesian_to_polar
from src.stimuli.render import RenderConfig

SCHIRA_RENDER_UPSAMPLE = 3


def upsample_render_canvas(
    base_canvas: int,
    *,
    multiple: int = SCHIRA_RENDER_UPSAMPLE,
) -> int:
    """Denser Schira ink canvas: integer multiple of the YAML 35 px/deg canvas.

    Canonical field is 210 px = 6° at 35 px/deg. Default 3× → 630 px,
    so 1° = 105 px = 35×3. (672 was 224×3 from the old 37.33 px/deg canvas.)
    """
    return int(base_canvas) * int(multiple)


def cartesian_deg_to_image_px(
    x_deg: np.ndarray | float,
    y_deg: np.ndarray | float,
    cfg: RenderConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map visual degrees to stimulus-image pixel coordinates."""
    cfg = cfg or RenderConfig()
    x = np.asarray(x_deg, dtype=np.float64)
    y = np.asarray(y_deg, dtype=np.float64)
    x_px = x * cfg.pixels_per_deg
    y_px = -y * cfg.pixels_per_deg
    return x_px, y_px


def image_px_to_cartesian_deg(
    x_px: np.ndarray | float,
    y_px: np.ndarray | float,
    cfg: RenderConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`cartesian_deg_to_image_px`."""
    cfg = cfg or RenderConfig()
    x = np.asarray(x_px, dtype=np.float64)
    y = np.asarray(y_px, dtype=np.float64)
    x_deg = x / cfg.pixels_per_deg
    y_deg = -y / cfg.pixels_per_deg
    return x_deg, y_deg


def image_px_to_polar(
    x_px: np.ndarray | float,
    y_px: np.ndarray | float,
    cfg: RenderConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stimulus image pixels → polar visual field ``(E, θ)``.

    Lab canvas: ``1° = pixels_per_deg`` (35 px). Fixation is top-left.

    ::

        x_deg =  x_px / pixels_per_deg
        y_deg = -y_px / pixels_per_deg   # image row down, catalog +y up
        E     = hypot(x_deg, y_deg)      # eccentricity (deg)
        θ     = atan2(y_deg, x_deg)      # radians; HM = 0, below HM negative
    """
    x_deg, y_deg = image_px_to_cartesian_deg(x_px, y_px, cfg)
    return cartesian_to_polar(x_deg, y_deg)


def cartesian_grid_line_degrees(
    *,
    spacing_deg: float = 0.5,
    extent_deg: float = 6.0,
    include_zero: bool = True,
) -> np.ndarray:
    """Line loci in degrees on ``[0, extent)`` at a fixed spacing.

    ``extent`` itself is excluded: pixel ``extent * pixels_per_deg`` sits on
    the far edge of a ``canvas = extent * pixels_per_deg`` image.
    """
    start = 0.0 if include_zero else float(spacing_deg)
    vals = np.arange(start, float(extent_deg), float(spacing_deg), dtype=np.float64)
    return vals[vals < float(extent_deg)]


def stamp_horizontal_meridian(
    img: np.ndarray,
    cfg: RenderConfig | None = None,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
    width_px: int = 2,
) -> None:
    """HM on the top edge of the quadrant: VF ``(0, 0) → (extent, 0)``.

    Image row 0 is ``y_deg = 0`` (fixation at the top-left).
    """
    del cfg
    n = int(img.shape[0])
    w = max(int(width_px), 1)
    img[0 : min(w, n), :, :] = color


def render_cartesian_degree_grid(
    cfg: RenderConfig | None = None,
    *,
    spacing_deg: float = 0.5,
    line_width_px: int = 1,
    include_zero: bool = True,
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, dict]:
    """Gray quadrant with horizontal + vertical lines every ``spacing_deg``.

    Distinct from ``experiments/retinotopic_map/render_grid_stimulus.py``,
    which places five equal-gap lines (not 0.5° spacing).
    """
    cfg = cfg or RenderConfig()
    n = int(cfg.canvas_size)
    bg = int(cfg.background_gray)
    img = np.full((n, n, 3), bg, dtype=np.uint8)
    x_deg = cartesian_grid_line_degrees(
        spacing_deg=spacing_deg,
        extent_deg=float(cfg.quadrant_extent_deg),
        include_zero=include_zero,
    )
    # Lower-right quadrant: +x right, +y up → image y is negative y_deg.
    y_deg = -x_deg
    half = max(int(line_width_px) // 2, 0)
    width = max(int(line_width_px), 1)

    def _stamp_vertical(x_px: float) -> None:
        c0 = int(round(x_px)) - half
        c1 = c0 + width
        c0 = max(0, c0)
        c1 = min(n, c1)
        if c0 < c1:
            img[:, c0:c1, :] = color

    def _stamp_horizontal(y_px: float) -> None:
        r0 = int(round(y_px)) - half
        r1 = r0 + width
        r0 = max(0, r0)
        r1 = min(n, r1)
        if r0 < r1:
            img[r0:r1, :, :] = color

    x_px, _ = cartesian_deg_to_image_px(x_deg, np.zeros_like(x_deg), cfg)
    _, y_px = cartesian_deg_to_image_px(np.zeros_like(y_deg), y_deg, cfg)
    for xp in np.asarray(x_px, dtype=np.float64).ravel():
        _stamp_vertical(float(xp))
    for yp in np.asarray(y_px, dtype=np.float64).ravel():
        _stamp_horizontal(float(yp))

    stamp_horizontal_meridian(img, cfg, color=color, width_px=max(width, 2))

    meta = {
        "spacing_deg": float(spacing_deg),
        "pixels_per_deg": float(cfg.pixels_per_deg),
        "canvas_size": n,
        "quadrant_extent_deg": float(cfg.quadrant_extent_deg),
        "vertical_x_deg": x_deg.tolist(),
        "horizontal_y_deg": y_deg.tolist(),
        "vertical_x_px": [float(v) for v in np.asarray(x_px).ravel()],
        "horizontal_y_px": [float(v) for v in np.asarray(y_px).ravel()],
        "line_width_px": width,
    }
    return img, meta


def _stamp_polyline_px(
    img: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    n = img.shape[0]
    xs = np.rint(np.asarray(x_px, dtype=np.float64)).astype(np.int64)
    ys = np.rint(np.asarray(y_px, dtype=np.float64)).astype(np.int64)
    ok = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return
    half = max(int(width) // 2, 0)
    w = max(int(width), 1)
    if w == 1:
        img[ys, xs, :] = color
        return
    for dx in range(-half, -half + w):
        for dy in range(-half, -half + w):
            cc = xs + dx
            rr = ys + dy
            inside = (cc >= 0) & (cc < n) & (rr >= 0) & (rr < n)
            img[rr[inside], cc[inside], :] = color


def polar_grid_eccentricities_deg(
    *,
    spacing_deg: float = 0.5,
    extent_deg: float = 6.0,
) -> np.ndarray:
    """Iso-eccentricity rings on ``(0, extent]`` (skip E=0 at fixation)."""
    vals = np.arange(
        float(spacing_deg), float(extent_deg) + 1e-9, float(spacing_deg), dtype=np.float64
    )
    return vals[vals > 0.0]


def polar_grid_theta_deg(
    *,
    step_deg: float = 15.0,
    theta_min_deg: float = -90.0,
    theta_max_deg: float = 0.0,
) -> np.ndarray:
    """Iso-angle rays in the lower-right quadrant (HM=0°, VM=−90°)."""
    if step_deg <= 0:
        raise ValueError("theta step must be > 0")
    vals = np.arange(
        float(theta_max_deg), float(theta_min_deg) - 1e-9, -float(step_deg)
    )
    return vals.astype(np.float64)


def render_polar_degree_grid(
    cfg: RenderConfig | None = None,
    *,
    ecc_spacing_deg: float = 0.5,
    theta_step_deg: float = 15.0,
    line_width_px: int = 1,
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, dict]:
    """Lower-right quadrant polar grid: iso-E rings + iso-θ rays (thesis Fig. 12A).

    Fixation is top-left. ``θ = atan2(y_deg, x_deg)`` with HM = 0 and VM = −90°.
    """
    cfg = cfg or RenderConfig()
    n = int(cfg.canvas_size)
    bg = int(cfg.background_gray)
    img = np.full((n, n, 3), bg, dtype=np.uint8)
    extent = float(cfg.quadrant_extent_deg)
    eccs = polar_grid_eccentricities_deg(
        spacing_deg=ecc_spacing_deg, extent_deg=extent
    )
    thetas_deg = polar_grid_theta_deg(step_deg=theta_step_deg)
    width = max(int(line_width_px), 1)

    for ecc in eccs:
        n_samp = max(180, int(round(0.5 * np.pi * ecc * cfg.pixels_per_deg)))
        th = np.linspace(0.0, -0.5 * np.pi, n_samp, dtype=np.float64)
        x_deg = ecc * np.cos(th)
        y_deg = ecc * np.sin(th)
        x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)
        _stamp_polyline_px(img, x_px, y_px, color=color, width=width)

    for th_deg in thetas_deg:
        th = np.deg2rad(th_deg)
        cth, sth = float(np.cos(th)), float(np.sin(th))
        denom = max(abs(cth), abs(sth), 1e-12)
        e_max = extent / denom
        n_samp = max(80, int(round(e_max * cfg.pixels_per_deg)))
        ecc = np.linspace(0.0, e_max, n_samp, dtype=np.float64)
        x_deg = ecc * cth
        y_deg = ecc * sth
        x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)
        _stamp_polyline_px(img, x_px, y_px, color=color, width=width)

    stamp_horizontal_meridian(img, cfg, color=color, width_px=max(width, 2))

    meta = {
        "ecc_spacing_deg": float(ecc_spacing_deg),
        "theta_step_deg": float(theta_step_deg),
        "pixels_per_deg": float(cfg.pixels_per_deg),
        "canvas_size": n,
        "quadrant_extent_deg": extent,
        "eccentricity_deg": eccs.tolist(),
        "theta_deg": thetas_deg.tolist(),
        "line_width_px": width,
    }
    return img, meta


def stimulus_ink_mask(
    rgb: np.ndarray,
    *,
    background_gray: int = 128,
    threshold: float = 8.0,
) -> np.ndarray:
    """True where a rendered stimulus differs from the gray field."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        delta = np.abs(arr.astype(np.float64) - float(background_gray))
    else:
        delta = np.max(
            np.abs(arr.astype(np.float64) - float(background_gray)), axis=-1
        )
    return delta > float(threshold)


def overlay_ink(
    base_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    *,
    background_gray: int = 128,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Stamp overlay ink onto ``base_rgb`` (letter-on-grid compositing).

    Overlay pixels replace the base. If ``color`` is set, those pixels are
    painted that RGB instead of the overlay's own values (so a black BMP
    glyph can sit visibly on a black grid).
    """
    out = np.array(base_rgb, dtype=np.uint8, copy=True)
    ink = stimulus_ink_mask(overlay_rgb, background_gray=background_gray)
    if not np.any(ink):
        return out
    if color is None:
        out[ink] = np.asarray(overlay_rgb, dtype=np.uint8)[ink]
    else:
        out[ink] = np.asarray(color, dtype=np.uint8)
    return out
