"""Render stimulus images for CNN encoding models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from src.stimuli.catalog import StimulusSpec


@dataclass(frozen=True)
class RenderConfig:
    canvas_size: int = 224
    # 224×224 input = 6° × 6° lower-right quadrant (fixation at top-left).
    pixels_per_deg: float = 224.0 / 6.0
    quadrant_extent_deg: float = 6.0
    background_gray: int = 128
    bar_length_deg: float = 0.3
    bar_width_px: int = 1
    contour_width_px: int = 1
    assume_size_is_diameter: bool = True
    draw_fixation: bool = False


def _color_rgb(name: str) -> tuple[int, int, int]:
    if name == "white":
        return (255, 255, 255)
    if name == "black":
        return (0, 0, 0)
    raise ValueError(f"Unsupported color: {name!r}")


def _deg_to_px(deg: float, cfg: RenderConfig) -> float:
    return deg * cfg.pixels_per_deg


def _deg_point_to_px(x_deg: float, y_deg: float, cfg: RenderConfig) -> tuple[float, float]:
    """
    Map visual-field degrees to quadrant pixels.

    Fixation is the top-left corner (0, 0). Catalog positions use +x right and
    negative y for downward; image y grows downward.
    """
    x_px = _deg_to_px(x_deg, cfg)
    y_px = _deg_to_px(-y_deg, cfg)
    return x_px, y_px


def _size_to_radius_px(size_deg: float, cfg: RenderConfig) -> float:
    diameter_deg = size_deg if cfg.assume_size_is_diameter else size_deg * 2.0
    radius_px = _deg_to_px(diameter_deg, cfg) / 2.0
    # Keep sub-pixel accuracy so e.g. 0.1 vs 0.05 deg remain distinguishable.
    return max(radius_px, 0.5)


def _draw_point(
    draw: ImageDraw.ImageDraw, spec: StimulusSpec, cfg: RenderConfig
) -> None:
    if spec.size_deg is None:
        raise ValueError(f"Point stimulus missing size: {spec.stimulus_text}")
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    radius_px = _size_to_radius_px(spec.size_deg, cfg)
    color = _color_rgb(spec.color)
    draw.ellipse(
        (x_px - radius_px, y_px - radius_px, x_px + radius_px, y_px + radius_px),
        fill=color,
        outline=color,
    )


def _draw_filled_circle(
    draw: ImageDraw.ImageDraw, spec: StimulusSpec, cfg: RenderConfig
) -> None:
    if spec.size_deg is None:
        raise ValueError(f"Circle stimulus missing size: {spec.stimulus_text}")
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    radius_px = _size_to_radius_px(spec.size_deg, cfg)
    color = _color_rgb(spec.color)
    draw.ellipse(
        (x_px - radius_px, y_px - radius_px, x_px + radius_px, y_px + radius_px),
        fill=color,
        outline=color,
    )


def _draw_circle_contour(
    draw: ImageDraw.ImageDraw, spec: StimulusSpec, cfg: RenderConfig
) -> None:
    if spec.size_deg is None:
        raise ValueError(f"Contour circle missing size: {spec.stimulus_text}")
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    radius_px = _size_to_radius_px(spec.size_deg, cfg)
    outline = _color_rgb(spec.color)
    width = cfg.contour_width_px
    draw.ellipse(
        (x_px - radius_px, y_px - radius_px, x_px + radius_px, y_px + radius_px),
        outline=outline,
        width=width,
    )


def _draw_triangle_contour(
    draw: ImageDraw.ImageDraw, spec: StimulusSpec, cfg: RenderConfig
) -> None:
    if spec.size_deg is None:
        raise ValueError(f"Triangle contour missing size: {spec.stimulus_text}")
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    radius_px = _size_to_radius_px(spec.size_deg, cfg)
    # Equilateral triangle inscribed in circle of radius radius_px.
    pts: list[tuple[float, float]] = []
    for angle_deg in (90, 210, 330):
        rad = math.radians(angle_deg)
        pts.append((x_px + radius_px * math.cos(rad), y_px - radius_px * math.sin(rad)))
    outline = _color_rgb(spec.color)
    closed = pts + [pts[0]]
    draw.line(closed, fill=outline, width=cfg.contour_width_px)


def _draw_bar(
    draw: ImageDraw.ImageDraw, spec: StimulusSpec, cfg: RenderConfig, *, vertical: bool
) -> None:
    length_deg = spec.size_deg if spec.size_deg is not None else cfg.bar_length_deg
    length_px = _deg_to_px(length_deg, cfg)
    width_px = float(max(cfg.contour_width_px, cfg.bar_width_px))
    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    if vertical:
        left = x_px - width_px / 2.0
        right = x_px + width_px / 2.0
        top = y_px - length_px / 2.0
        bottom = y_px + length_px / 2.0
    else:
        left = x_px - length_px / 2.0
        right = x_px + length_px / 2.0
        top = y_px - width_px / 2.0
        bottom = y_px + width_px / 2.0
    draw.rectangle((left, top, right, bottom), fill=_color_rgb(spec.color))


def render_stimulus(spec: StimulusSpec, cfg: RenderConfig | None = None) -> np.ndarray:
    """
    Render one stimulus as RGB uint8 array with shape (H, W, 3).
    """
    cfg = cfg or RenderConfig()
    img = Image.new(
        "RGB",
        (cfg.canvas_size, cfg.canvas_size),
        color=(cfg.background_gray, cfg.background_gray, cfg.background_gray),
    )
    draw = ImageDraw.Draw(img)

    if not spec.is_blank:
        if spec.shape_type == "point":
            _draw_point(draw, spec, cfg)
        elif spec.shape_type == "filled_circle":
            _draw_filled_circle(draw, spec, cfg)
        elif spec.shape_type == "circle_contour":
            _draw_circle_contour(draw, spec, cfg)
        elif spec.shape_type == "triangle_contour":
            _draw_triangle_contour(draw, spec, cfg)
        elif spec.shape_type == "bar_vertical":
            _draw_bar(draw, spec, cfg, vertical=True)
        elif spec.shape_type == "bar_horizontal":
            _draw_bar(draw, spec, cfg, vertical=False)
        else:
            raise ValueError(f"Unsupported shape_type: {spec.shape_type}")

    return np.asarray(img, dtype=np.uint8)
