"""Local (per-pixel) ridge: warped CNN channels → VSD amplitude at that pixel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.schira_encoding.arrays import as_feature_batch, as_target_maps


@dataclass
class LocalRidgeResult:
    """Per-pixel weights on ``C`` warped feature channels."""

    weights: np.ndarray  # (C, H, W)
    intercepts: np.ndarray  # (H, W)
    alphas: np.ndarray  # (H, W); NaN outside fit mask
    valid: np.ndarray  # (H, W) bool — pixels that were fit
    spatial_size: tuple[int, int]
    n_channels: int
    standardize_features: bool
    # Optional: per-pixel mean/scale when standardize_features (C, H, W)
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None


def fit_local_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    valid: np.ndarray,
    alphas: np.ndarray | list[float],
    standardize_features: bool = True,
    alpha_per_pixel: bool = True,
    min_trials: int = 2,
) -> LocalRidgeResult:
    """Fit one Ridge (or RidgeCV) per valid VSD pixel on ``C`` channels.

    Parameters
    ----------
    x :
        Warped features ``(n, C, H, W)``.
    y :
        Anchor-aligned targets ``(n, H, W)`` or ``(n, H*W)``.
    valid :
        Boolean ``(H, W)`` fit mask (typically LUT.valid ∩ eval disk).
    alphas :
        RidgeCV candidate α values (or a single α if ``alpha_per_pixel`` is
        False and only one value is given).
    """
    x_arr = np.asarray(x)
    if x_arr.ndim != 4:
        raise ValueError(f"Expected X (n, C, H, W), got {x_arr.shape}")
    n_trials, n_channels, height, width = x_arr.shape
    spatial_size = (height, width)
    x_arr = as_feature_batch(x_arr, spatial_size)
    y_arr = as_target_maps(y, spatial_size)
    if y_arr.shape[0] != n_trials:
        raise ValueError(
            f"X/Y trial count mismatch: {n_trials} vs {y_arr.shape[0]}"
        )

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != (height, width):
        raise ValueError(f"valid shape {mask.shape} != {(height, width)}")

    alpha_grid = np.asarray(alphas, dtype=np.float64).ravel()
    if alpha_grid.size == 0:
        raise ValueError("alphas must be non-empty")

    weights = np.zeros((n_channels, height, width), dtype=np.float64)
    intercepts = np.zeros((height, width), dtype=np.float64)
    alpha_map = np.full((height, width), np.nan, dtype=np.float64)
    feat_mean = (
        np.zeros((n_channels, height, width), dtype=np.float64)
        if standardize_features
        else None
    )
    feat_scale = (
        np.ones((n_channels, height, width), dtype=np.float64)
        if standardize_features
        else None
    )

    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("valid mask has no True pixels")
    if n_trials < min_trials:
        raise ValueError(
            f"Need at least {min_trials} trials for local ridge, got {n_trials}"
        )

    for r, c in zip(rows.tolist(), cols.tolist()):
        x_p = x_arr[:, :, r, c]  # (n, C)
        y_p = y_arr[:, r, c]
        if not np.isfinite(y_p).all():
            # Skip pixels with any non-finite target (e.g. after session warp).
            mask[r, c] = False
            continue

        scaler: StandardScaler | None = None
        x_fit = x_p
        if standardize_features:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x_p)
            assert feat_mean is not None and feat_scale is not None
            feat_mean[:, r, c] = scaler.mean_
            # sklearn stores scale_; avoid zeros
            scale = np.asarray(scaler.scale_, dtype=np.float64)
            scale = np.where(scale < 1e-8, 1.0, scale)
            feat_scale[:, r, c] = scale

        if alpha_per_pixel and alpha_grid.size > 1:
            model = RidgeCV(alphas=alpha_grid, cv=None, fit_intercept=True)
        else:
            model = Ridge(alpha=float(alpha_grid[0]), fit_intercept=True)
        model.fit(x_fit, y_p)
        if isinstance(model, RidgeCV):
            alpha_map[r, c] = float(np.asarray(model.alpha_).reshape(-1)[0])
        else:
            alpha_map[r, c] = float(alpha_grid[0])
        weights[:, r, c] = np.asarray(model.coef_, dtype=np.float64).ravel()
        intercepts[r, c] = float(model.intercept_)

    return LocalRidgeResult(
        weights=weights.astype(np.float32),
        intercepts=intercepts.astype(np.float32),
        alphas=alpha_map.astype(np.float32),
        valid=mask,
        spatial_size=spatial_size,
        n_channels=n_channels,
        standardize_features=standardize_features,
        feature_mean=None
        if feat_mean is None
        else feat_mean.astype(np.float32),
        feature_scale=None
        if feat_scale is None
        else feat_scale.astype(np.float32),
    )


def predict_local_ridge(
    x: np.ndarray,
    result: LocalRidgeResult,
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
    if result.weights.shape != (n_channels, height, width):
        raise ValueError("Model weights spatial/channel shape mismatch")

    out = np.full((n_trials, height, width), fill, dtype=np.float32)
    mask = result.valid
    rows, cols = np.where(mask)
    w = result.weights.astype(np.float64)
    b = result.intercepts.astype(np.float64)

    for r, c in zip(rows.tolist(), cols.tolist()):
        x_p = x_arr[:, :, r, c]
        if result.standardize_features:
            if result.feature_mean is None or result.feature_scale is None:
                raise ValueError("Missing feature_mean/scale for standardized model")
            mean = result.feature_mean[:, r, c].astype(np.float64)
            scale = result.feature_scale[:, r, c].astype(np.float64)
            x_p = (x_p - mean) / scale
        pred = x_p @ w[:, r, c] + b[r, c]
        out[:, r, c] = pred.astype(np.float32)
    return out
