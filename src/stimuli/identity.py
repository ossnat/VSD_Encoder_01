"""Stable stimulus identity keys shared by ROI / LOO tooling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Catalog bars are always 1° long (legacy 0.3° fallback was a labeling bug).
BAR_SHAPE_TYPES = frozenset({"bar_vertical", "bar_horizontal"})
CANONICAL_BAR_LENGTH_DEG = 1.0


def format_size_deg(size_deg: Any) -> str:
    if size_deg is None or (isinstance(size_deg, float) and np.isnan(size_deg)):
        return "na"
    v = float(size_deg)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def canonical_size_deg(shape_type: Any, size_deg: Any) -> Any:
    """Force bar lengths to the canonical 1° value used for stimulus_id."""
    shape = str(shape_type or "")
    if shape in BAR_SHAPE_TYPES:
        return CANONICAL_BAR_LENGTH_DEG
    return size_deg


def stimulus_id_from_row(row: pd.Series | dict[str, Any]) -> str | None:
    """
    Stable stimulus identity key from catalog / manifest fields.

    - shapes: ``{color}_{shape_type}_{size}`` e.g. ``white_point_0.1``
    - letters: ``letter_{L}_{color}_{size}`` e.g. ``letter_A_white_1``
    - bars: always ``…_1`` (canonical 1° length; remaps legacy ``…_0.3``)
    """
    get = row.get if hasattr(row, "get") else lambda k, default=None: row[k] if k in row else default  # type: ignore[index]
    shape = str(get("shape_type", "") or "")
    if not shape or shape == "blank" or bool(get("is_blank", False)):
        return None
    color = str(get("color", "unknown") or "unknown")
    size = format_size_deg(canonical_size_deg(shape, get("size_deg")))
    if shape == "letter":
        letter = get("letter")
        if letter is None or (isinstance(letter, float) and np.isnan(letter)):
            text = str(get("stimulus_text", "") or "")
            parts = text.strip().split()
            letter = parts[-1] if parts else "?"
        return f"letter_{str(letter).upper()}_{color}_{size}"
    return f"{color}_{shape}_{size}"


def attach_stimulus_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``stimulus_id`` column derived from row fields.

    Always recomputes ``stimulus_id`` so legacy bar rows with ``size_deg=0.3``
    map to ``black_bar_*_1``.
    """
    out = df.copy()
    if "size_deg" in out.columns and "shape_type" in out.columns:
        bar_mask = out["shape_type"].astype(str).isin(BAR_SHAPE_TYPES)
        if bar_mask.any():
            out.loc[bar_mask, "size_deg"] = CANONICAL_BAR_LENGTH_DEG
    out["stimulus_id"] = out.apply(stimulus_id_from_row, axis=1)
    return out


def _num(v: Any) -> str:
    """Compact, stable number formatting for uids (0.6 -> '0.6', 1.0 -> '1', -0.75 -> '-0.75')."""
    return format_size_deg(v)


def _rgb_of(row: Any) -> tuple[int, int, int] | None:
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    rgb = get("rgb", None)
    if rgb is None or (isinstance(rgb, float) and np.isnan(rgb)):
        return None
    vals = [int(x) for x in list(rgb)]
    return (vals[0], vals[1], vals[2])


def stimulus_uid_from_row(row: Any, default_background: int = 128) -> str:
    """Exact image identity: one uid per distinct stimulus image in the stimulus library.

    Unlike ``stimulus_id`` (shape family, used to group leave-one-out folds), the uid includes everything that
    changes the pixels: shape, colour or RGB, size (diameter; bar length), position and background grey.
    Examples::

        triangle_contour__black__d0.8__x0.6_y-0.75__bg128
        filled_circle__rgb158__d0.3__x0.6_y-0.75__bg186
        bar_vertical__black__len1__x0.6_y-0.75__bg128
        letter_D__x1_y-0.9__bg128
        blank__bg128
    """
    get = row.get if hasattr(row, "get") else (lambda k, d=None: d)
    bg_raw = get("background_gray", None)
    bg = default_background if bg_raw is None or (isinstance(bg_raw, float) and np.isnan(bg_raw)) else int(bg_raw)
    shape = str(get("shape_type", "") or "")
    if not shape or shape == "blank" or bool(get("is_blank", False)):
        return f"blank__bg{bg}"
    pos = f"x{_num(get('pos_x_deg'))}_y{_num(get('pos_y_deg'))}"
    if shape == "letter":
        letter = str(get("letter", "") or "?").upper()
        return f"letter_{letter}__{pos}__bg{bg}"
    rgb = _rgb_of(row)
    colour = f"rgb{rgb[0]}" if rgb is not None and rgb[0] == rgb[1] == rgb[2] else (
        f"rgb{rgb[0]}-{rgb[1]}-{rgb[2]}" if rgb is not None else str(get("color", "unknown") or "unknown"))
    size = get("size_deg", None)
    size_tag = f"len{_num(size)}" if shape in BAR_SHAPE_TYPES else f"d{_num(size)}"
    return f"{shape}__{colour}__{size_tag}__{pos}__bg{bg}"
