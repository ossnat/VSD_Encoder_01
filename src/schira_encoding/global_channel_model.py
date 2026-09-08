"""Global channel ridge: one weight per warped CNN channel (shared over pixels).

After Schira+affine pull-warp, each trial is ``X (C, H, W)``. Fit::

    Y_ij ≈ Σ_c w_c · X_{c,ij} + b

with the same ``(w, b)`` at every valid pixel. Spatial structure comes only
from the geometry warp; the learned part is filter gains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from src.schira_encoding.arrays import as_feature_batch, as_target_maps


@dataclass
class GlobalChannelRidgeResult:
    """Shared channel weights on Schira-warped CNN maps."""

    weights: np.ndarray  # (C,)
    intercept: float
    alpha: float
    valid: np.ndarray  # (H, W) bool — pixels used in the design matrix
    spatial_size: tuple[int, int]
    n_channels: int
    standardize_features: bool
    feature_mean: np.ndarray | None  # (C,) when standardized
    feature_scale: np.ndarray | None  # (C,)
    n_samples: int


def fit_global_channel_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    valid: np.ndarray,
    alphas: np.ndarray | list[float],
    standardize_features: bool = True,
    min_samples: int = 64,
) -> GlobalChannelRidgeResult:
    """Fit one ridge over all (trial, valid-pixel) samples.

    Feature standardization (option B): per-channel mean/std over the stacked
    train samples (trials × masked pixels), fit on train only.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim != 4:
        raise ValueError(f"Expected X (n, C, H, W), got {x_arr.shape}")
    n_trials, n_channels, height, width = x_arr.shape
    spatial_size = (height, width)
    x_arr = as_feature_batch(x_arr, spatial_size)
    y_arr = as_target_maps(y, spatial_size)
    if y_arr.shape[0] != n_trials:
        raise ValueError(f"X/Y trial count mismatch: {n_trials} vs {y_arr.shape[0]}")

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError(f"valid shape {mask.shape} != {(height, width)}")
    if not mask.any():
        raise ValueError("valid mask has no True pixels")

    alpha_grid = np.asarray(alphas, dtype=np.float64).ravel()
    if alpha_grid.size == 0:
        raise ValueError("alphas must be non-empty")

    # Stack (trial, pixel) → design matrix (n_samples, C).
    x_pix = x_arr[:, :, mask].transpose(0, 2, 1).reshape(-1, n_channels)
    y_pix = y_arr[:, mask].reshape(-1)
    finite = np.isfinite(y_pix) & np.isfinite(x_pix).all(axis=1)
    x_pix = x_pix[finite]
    y_pix = y_pix[finite]
    n_samples = int(x_pix.shape[0])
    if n_samples < min_samples:
        raise ValueError(
            f"Need at least {min_samples} finite (trial,pixel) samples, got {n_samples}"
        )

    feat_mean: np.ndarray | None = None
    feat_scale: np.ndarray | None = None
    x_fit = x_pix
    if standardize_features:
        feat_mean = np.mean(x_pix, axis=0)
        feat_scale = np.std(x_pix, axis=0)
        feat_scale = np.where(feat_scale < 1e-8, 1.0, feat_scale)
        x_fit = (x_pix - feat_mean) / feat_scale

    if alpha_grid.size > 1:
        model = RidgeCV(alphas=alpha_grid, cv=None, fit_intercept=True)
        model.fit(x_fit, y_pix)
        alpha = float(np.asarray(model.alpha_).reshape(-1)[0])
    else:
        model = Ridge(alpha=float(alpha_grid[0]), fit_intercept=True)
        model.fit(x_fit, y_pix)
        alpha = float(alpha_grid[0])

    return GlobalChannelRidgeResult(
        weights=np.asarray(model.coef_, dtype=np.float32).ravel(),
        intercept=float(model.intercept_),
        alpha=alpha,
        valid=mask,
        spatial_size=spatial_size,
        n_channels=n_channels,
        standardize_features=standardize_features,
        feature_mean=None if feat_mean is None else feat_mean.astype(np.float32),
        feature_scale=None if feat_scale is None else feat_scale.astype(np.float32),
        n_samples=n_samples,
    )


def predict_global_channel_ridge(
    x: np.ndarray,
    result: GlobalChannelRidgeResult,
    *,
    fill: float = np.nan,
) -> np.ndarray:
    """Predict maps ``(n, H, W)`` from warped features ``(n, C, H, W)``."""
    x_arr = as_feature_batch(x, result.spatial_size)
    n_trials, n_channels, height, width = x_arr.shape
    if n_channels != result.n_channels:
        raise ValueError(
            f"Channel mismatch: X has {n_channels}, model has {result.n_channels}"
        )

    if result.standardize_features:
        if result.feature_mean is None or result.feature_scale is None:
            raise ValueError("Missing feature_mean/scale for standardized model")
        mean = result.feature_mean.astype(np.float64).reshape(1, n_channels, 1, 1)
        scale = result.feature_scale.astype(np.float64).reshape(1, n_channels, 1, 1)
        x_use = (x_arr - mean) / scale
    else:
        x_use = x_arr

    w = result.weights.astype(np.float64).reshape(1, n_channels, 1, 1)
    pred = (x_use * w).sum(axis=1) + float(result.intercept)
    out = np.full((n_trials, height, width), fill, dtype=np.float32)
    out[:, result.valid] = pred[:, result.valid].astype(np.float32)
    return out
