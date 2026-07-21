"""Render stimulus images for CNN encoding models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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


def _color_rgb(spec: StimulusSpec) -> tuple[int, int, int]:
    if spec.rgb is not None:
        return tuple(int(x) for x in spec.rgb)  # type: ignore[return-value]
    if spec.color == "white":
        return (255, 255, 255)
    if spec.color == "black":
        return (0, 0, 0)
    raise ValueError(f"Unsupported color: {spec.color!r}")


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
    color = _color_rgb(spec)
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
    color = _color_rgb(spec)
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
    outline = _color_rgb(spec)
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
    outline = _color_rgb(spec)
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
    draw.rectangle((left, top, right, bottom), fill=_color_rgb(spec))


def _render_letter_on_quadrant(spec: StimulusSpec, cfg: RenderConfig) -> np.ndarray:
    """
    Place a letter on the lower-right quadrant canvas.

    Catalog ``size_deg`` is the diameter of the circle that contains the letter
    (1° → center is 0.5° from the edges). The glyph is scaled to fit that
    diameter and centered at the swapped Target Location
    (``pos_x_deg``, ``pos_y_deg``). Fixation is the top-left of this quadrant.
    """
    if spec.source_path is None or spec.size_deg is None:
        raise ValueError(f"Letter stimulus missing source/size: {spec.stimulus_text}")
    path = Path(spec.source_path)

    if path.suffix.lower() == ".bmp":
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        bg_rgb = tuple(int(x) for x in rgb[0, 0])
        mask = np.any(rgb != rgb[0, 0], axis=2)
        if not mask.any():
            raise ValueError(f"No letter pixels found in {path}")
        ys, xs = np.where(mask)
        glyph = rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        glyph_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        import scipy.io as sio

        mat = sio.loadmat(path)
        if "img" not in mat:
            raise ValueError(f"Letter mat missing 'img': {path}")
        letter = np.asarray(mat["img"], dtype=np.float64)
        letter_u8 = np.clip(np.rint(letter * 255.0), 0, 255).astype(np.uint8)
        glyph = np.stack([letter_u8, letter_u8, letter_u8], axis=-1)
        glyph_mask = letter_u8 > 0
        bg_rgb = (
            int(spec.background_gray),
            int(spec.background_gray),
            int(spec.background_gray),
        ) if spec.background_gray is not None else (
            cfg.background_gray,
            cfg.background_gray,
            cfg.background_gray,
        )

    canvas_bg = (
        int(spec.background_gray)
        if spec.background_gray is not None
        and not (
            isinstance(spec.background_gray, float) and math.isnan(spec.background_gray)
        )
        else int(bg_rgb[0])
    )
    img = Image.new(
        "RGB",
        (cfg.canvas_size, cfg.canvas_size),
        color=(canvas_bg, canvas_bg, canvas_bg),
    )

    # Diameter of the 1° letter circle → longest glyph side fits that diameter.
    diameter_px = max(1, int(round(_deg_to_px(spec.size_deg, cfg))))
    gh, gw = glyph.shape[:2]
    scale = diameter_px / float(max(gh, gw))
    new_w = max(1, int(round(gw * scale)))
    new_h = max(1, int(round(gh * scale)))
    glyph_img = Image.fromarray(glyph, mode="RGB").resize(
        (new_w, new_h), resample=Image.Resampling.BILINEAR
    )
    alpha = Image.fromarray(
        (glyph_mask.astype(np.uint8) * 255), mode="L"
    ).resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    x_px, y_px = _deg_point_to_px(spec.pos_x_deg, spec.pos_y_deg, cfg)
    left = int(round(x_px - new_w / 2.0))
    top = int(round(y_px - new_h / 2.0))
    img.paste(glyph_img, (left, top), mask=alpha)
    return np.asarray(img, dtype=np.uint8)


def render_stimulus(spec: StimulusSpec, cfg: RenderConfig | None = None) -> np.ndarray:
    """
    Render one stimulus as RGB uint8 array with shape (H, W, 3).
    """
    cfg = cfg or RenderConfig()
    bg = (
        int(spec.background_gray)
        if spec.background_gray is not None and not (
            isinstance(spec.background_gray, float) and math.isnan(spec.background_gray)
        )
        else cfg.background_gray
    )
    img = Image.new(
        "RGB",
        (cfg.canvas_size, cfg.canvas_size),
        color=(bg, bg, bg),
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
        elif spec.shape_type == "letter":
            return _render_letter_on_quadrant(spec, cfg)
        else:
            raise ValueError(f"Unsupported shape_type: {spec.shape_type}")

    return np.asarray(img, dtype=np.uint8)
