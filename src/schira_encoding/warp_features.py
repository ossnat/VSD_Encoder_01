"""Resample CNN feature maps onto the anchor VSD grid via a Schira LUT."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from src.schira_encoding.lut import SchiraLUT


def warp_feature_map(
    feature_map: np.ndarray,
    lut: SchiraLUT,
    *,
    fill: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """Warp ``(C, Hf, Wf)`` features onto ``(C, Hv, Wv)`` anchor VSD layout.

    Invalid LUT pixels (and samples that fall outside the feature map) are
    filled with ``fill``. Sampling uses bilinear interpolation (``order=1``)
    in feature-map coordinates derived from the LUT stimulus coords and
    ``lut.input_size``.
    """
    feat = np.asarray(feature_map)
    if feat.ndim != 3:
        raise ValueError(f"Expected feature map (C, H, W), got {feat.shape}")
    n_channels, hf, wf = feat.shape
    feat_row, feat_col = lut.feature_sample_coords((hf, wf))
    # Stay inside the interpolable region (same half-pixel rule as the LUT).
    in_feat = (
        lut.valid
        & np.isfinite(feat_row)
        & np.isfinite(feat_col)
        & (feat_row >= -0.5)
        & (feat_col >= -0.5)
        & (feat_row < hf - 0.5)
        & (feat_col < wf - 0.5)
    )
    coords = np.vstack([feat_row.ravel(), feat_col.ravel()])

    out = np.full(
        (n_channels, lut.height, lut.width),
        float(fill),
        dtype=np.float32,
    )
    if not np.any(in_feat):
        return out

    for c in range(n_channels):
        sampled = map_coordinates(
            feat[c].astype(np.float64, copy=False),
            coords,
            order=int(order),
            mode="nearest",
        ).reshape(lut.height, lut.width)
        plane = np.full((lut.height, lut.width), float(fill), dtype=np.float64)
        plane[in_feat] = sampled[in_feat]
        out[c] = plane.astype(np.float32)

    return out


def warp_stimulus_rgb(
    stimulus_rgb: np.ndarray,
    lut: SchiraLUT,
    *,
    background_gray: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """QC helper: warp an RGB stimulus with the same LUT as features.

    Uses stimulus-canvas coordinates directly (no CNN downsample).
    """
    rgb = np.asarray(stimulus_rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 stimulus, got {rgb.shape}")
    if rgb.shape[0] != lut.canvas_size or rgb.shape[1] != lut.canvas_size:
        raise ValueError(
            f"Stimulus shape {rgb.shape[:2]} != LUT canvas "
            f"{(lut.canvas_size, lut.canvas_size)}"
        )

    coords = np.vstack([lut.stim_row.ravel(), lut.stim_col.ravel()])
    bg = int(background_gray)
    warped = np.full((lut.height, lut.width, 3), bg, dtype=np.uint8)
    valid = lut.valid
    if not np.any(valid):
        return warped, valid

    for channel in range(3):
        sampled = map_coordinates(
            rgb[:, :, channel].astype(np.float64),
            coords,
            order=1,
            mode="constant",
            cval=float(bg),
        ).reshape(lut.height, lut.width)
        plane = np.full((lut.height, lut.width), float(bg))
        plane[valid] = sampled[valid]
        warped[:, :, channel] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
    return warped, valid
