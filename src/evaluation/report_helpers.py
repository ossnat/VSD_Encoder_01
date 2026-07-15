"""Shared config / label helpers for evaluation report scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.mask import masked_pearson_r


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def merge_default_window_ridge(
    default_path: Path,
    window_path: Path,
    ridge_path: Path,
) -> dict:
    cfg = load_yaml(default_path)
    cfg.update(load_yaml(window_path))
    cfg["ridge"] = load_yaml(ridge_path)
    return cfg


def model_display_label(model_cfg: dict, slug: str) -> str:
    if "resnet18" in slug:
        return "ResNet18"
    gamma = model_cfg.get("gwp", {}).get("gamma")
    if gamma is not None:
        return f"GWP (best, γ={gamma})"
    if model_cfg.get("type") == "gabor_gwp" or "gabor" in slug:
        return "GWP (best)"
    return str(model_cfg.get("name", slug))


def format_metrics_table(
    df: pd.DataFrame,
    *,
    columns: dict[str, str],
) -> pd.DataFrame:
    """Rename selected numeric columns and format to 4 decimal places."""
    display = df.copy()
    use = {k: v for k, v in columns.items() if k in display.columns}
    out = display[list(use.keys())].rename(columns=use)
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return out


def condition_balanced_spatial_r(
    conditions: list[dict[str, Any]],
    mask: np.ndarray,
) -> float:
    """Equal-weight mean of masked whole-map r(mean orig, recon) per condition."""
    rs = [
        masked_pearson_r(
            np.asarray(c["original"]),
            np.asarray(c["reconstruction"]),
            mask,
        )
        for c in conditions
    ]
    finite = [r for r in rs if np.isfinite(r)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))
