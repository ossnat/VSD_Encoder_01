"""Circular (or square) evaluation masks for VSD maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def region_mask(
    spatial_size: tuple[int, int],
    *,
    mask_type: str = "circle",
    radius: int,
) -> np.ndarray:
    """
    Boolean mask that is True inside the evaluation region.

    Matches VSD foundation model ``mae_system.MaskedReconstructionLoss``:
    center at (H/2, W/2), distance in pixel index space.
    """
    height, width = spatial_size
    cy, cx = height / 2.0, width / 2.0
    ys = np.arange(height, dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    if mask_type == "circle":
        inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(radius) ** 2
    elif mask_type == "square":
        inside = (np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type!r}")

    return inside


def mask_from_eval_cfg(
    eval_cfg: dict[str, Any] | None,
    spatial_size: tuple[int, int],
) -> np.ndarray | None:
    """Return a region mask when ``use_mask`` is true in evaluation config."""
    if not eval_cfg or not eval_cfg.get("use_mask", False):
        return None
    return region_mask(
        spatial_size,
        mask_type=str(eval_cfg.get("mask_type", "circle")),
        radius=int(eval_cfg["mask_radius"]),
    )


def masked_pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Pearson r over flattened pixels where ``mask`` is True."""
    flat_mask = mask.ravel()
    a = y_true.ravel().astype(np.float64)[flat_mask]
    b = y_pred.ravel().astype(np.float64)[flat_mask]
    if a.size == 0 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def apply_mask_nan(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy array with NaN outside the mask (2D map or 3D stack of maps)."""
    out = arr.astype(np.float32, copy=True)
    if out.ndim == 2:
        out[~mask] = np.nan
        return out
    if out.ndim == 3:
        return np.where(mask[None, ...], out, np.nan).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def masked_map_summary(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mean and median over finite values inside ``mask``."""
    values = arr[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
    }
