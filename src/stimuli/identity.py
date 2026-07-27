"""Stable stimulus identity keys shared by ROI / LOO tooling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def format_size_deg(size_deg: Any) -> str:
    if size_deg is None or (isinstance(size_deg, float) and np.isnan(size_deg)):
        return "na"
    v = float(size_deg)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def stimulus_id_from_row(row: pd.Series | dict[str, Any]) -> str | None:
    """
    Stable stimulus identity key from catalog / manifest fields.

    - shapes: ``{color}_{shape_type}_{size}`` e.g. ``white_point_0.1``
    - letters: ``letter_{L}_{color}_{size}`` e.g. ``letter_A_white_1``
    """
    get = row.get if hasattr(row, "get") else lambda k, default=None: row[k] if k in row else default  # type: ignore[index]
    shape = str(get("shape_type", "") or "")
    if not shape or shape == "blank" or bool(get("is_blank", False)):
        return None
    color = str(get("color", "unknown") or "unknown")
    size = format_size_deg(get("size_deg"))
    if shape == "letter":
        letter = get("letter")
        if letter is None or (isinstance(letter, float) and np.isnan(letter)):
            text = str(get("stimulus_text", "") or "")
            parts = text.strip().split()
            letter = parts[-1] if parts else "?"
        return f"letter_{str(letter).upper()}_{color}_{size}"
    return f"{color}_{shape}_{size}"


def attach_stimulus_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``stimulus_id`` column derived from row fields."""
    out = df.copy()
    if "stimulus_id" in out.columns and out["stimulus_id"].notna().any():
        missing = out["stimulus_id"].isna()
        if missing.any():
            out.loc[missing, "stimulus_id"] = out.loc[missing].apply(
                stimulus_id_from_row, axis=1
            )
        return out
    out["stimulus_id"] = out.apply(stimulus_id_from_row, axis=1)
    return out
