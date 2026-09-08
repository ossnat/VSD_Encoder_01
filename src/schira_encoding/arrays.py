"""Shared array-shape checks for Schira encoding models."""

from __future__ import annotations

import numpy as np


def as_feature_batch(
    x: np.ndarray, spatial_size: tuple[int, int]
) -> np.ndarray:
    """Require ``X`` as ``(n, C, H, W)`` with the given spatial size."""
    arr = np.asarray(x)
    height, width = spatial_size
    if arr.ndim != 4 or arr.shape[-2:] != (height, width):
        raise ValueError(
            f"Expected X (n, C, H, W) with H,W={spatial_size}, got {arr.shape}"
        )
    return arr.astype(np.float64, copy=False)


def as_target_maps(
    y: np.ndarray, spatial_size: tuple[int, int]
) -> np.ndarray:
    """Require ``Y`` as ``(n, H, W)`` or flattened ``(n, H*W)``."""
    arr = np.asarray(y)
    height, width = spatial_size
    if arr.ndim == 3 and arr.shape[-2:] == (height, width):
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[1] == height * width:
        return arr.reshape(arr.shape[0], height, width).astype(
            np.float64, copy=False
        )
    raise ValueError(
        f"Expected Y (n, H, W) or (n, H*W) with H,W={spatial_size}, got {arr.shape}"
    )
