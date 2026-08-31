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
    """Stimulus image pixels → ``(E, θ)`` with ``θ = atan2(y_deg, x_deg)`` rad."""
    x_deg, y_deg = image_px_to_cartesian_deg(x_px, y_px, cfg)
    return cartesian_to_polar(x_deg, y_deg)


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
