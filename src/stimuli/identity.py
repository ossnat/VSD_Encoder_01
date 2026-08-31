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
