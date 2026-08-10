"""Channel×space separable Ridge encoding (CM model).

Features stay as ``(C, H, W)`` tensors. The model uses a shared channel vector
``a ∈ R^C`` and a per-target-pixel spatial map ``M_p ∈ R^{H×W}`` (CNN feature
space)::

    y_p = sum_{c,h,w} a_c * M_p[h,w] * X[c,h,w] + b_p

Fit by alternating least squares (ALS) with sklearn ``Ridge`` / ``RidgeCV``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import xarray as xr

from src.encoding.ridge import (
    flatten_target_mask,
    scatter_target_pixels,
    select_target_pixels,
)
from src.paths import resolve_data_path


def _load_target(nc_path, spatial_size: tuple[int, int]) -> np.ndarray:
    da = xr.open_dataarray(nc_path)
    image = da.values.astype(np.float32)
    da.close()
    height, width = spatial_size
    if image.shape != (height, width):
        raise ValueError(f"Expected target shape {(height, width)}, got {image.shape}")
    return image.reshape(-1)


@dataclass
class CMEncodeResult:
    """Fitted channel×space separable encoder."""

    a: np.ndarray  # (C,)
    M: np.ndarray  # (n_targets, H*W) — targets = full FOV or mask pixels
    bias: np.ndarray  # (n_targets,)
    scaler: StandardScaler | None
    feature_shape: tuple[int, int, int]  # (C, H, W)
    spatial_size: tuple[int, int]  # VSD (H, W)
    alpha_M: float | np.ndarray
    alpha_a: float
    n_als_iters: int
    train_mse: list[float] = field(default_factory=list)
    alpha_per_target: bool = False
    target_mask: np.ndarray | None = None
    target_pixel_indices: np.ndarray | None = field(default=None, repr=False)
    feature_layer: str = ""
    model_slug: str = ""


def build_xy_maps(
    pairs: pd.DataFrame,
    *,
    repo,
    spatial_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features as ``(n, C, H, W)`` and flattened VSD targets ``(n, H*W)``.

    Unlike ``build_xy``, feature maps are **not** flattened.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    feat_shape: tuple[int, ...] | None = None

    for row in pairs.itertuples(index=False):
        feat = np.load(resolve_data_path(row.feature_path, repo))
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3:
            raise ValueError(
                f"Expected feature map (C,H,W), got shape {feat.shape} "
                f"for {row.feature_path}"
            )
        if feat_shape is None:
            feat_shape = feat.shape
        elif feat.shape != feat_shape:
            raise ValueError(
                f"Inconsistent feature shape {feat.shape} vs {feat_shape}"
            )
        xs.append(feat)
        ys.append(_load_target(resolve_data_path(row.nc_path, repo), spatial_size))

    if not xs:
        raise ValueError("No rows to build X/Y maps")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def _standardize_features(
    x: np.ndarray,
    *,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler | None]:
    """Standardize over trials for each (c,h,w) element; keep 4D layout."""
    n, c, h, w = x.shape
    flat = x.reshape(n, -1)
    if fit:
        scaler = StandardScaler()
        flat = scaler.fit_transform(flat).astype(np.float32, copy=False)
    elif scaler is not None:
        flat = scaler.transform(flat).astype(np.float32, copy=False)
    return flat.reshape(n, c, h, w), scaler


def _project_channels(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``Z[i] = sum_c a_c X[i,c]`` → ``(n, H*W)``."""
    # (n,C,H,W) × (C,) → (n,H,W)
    z = np.einsum("nchw,c->nhw", x, a.astype(np.float32, copy=False))
    return z.reshape(z.shape[0], -1).astype(np.float32, copy=False)


def _fit_M_ridge(
    z: np.ndarray,
    y: np.ndarray,
    *,
    alphas: np.ndarray,
    cv_folds: int,
    alpha_per_target: bool,
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray]:
    """
    Fit per-pixel spatial maps given channel-projected features ``Z``.

    Returns ``(M, bias, alpha)`` with ``M`` shape ``(n_targets, H*W)``.
    """
    if alpha_per_target:
        model = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    else:
        n_splits = min(int(cv_folds), len(z))
        if n_splits < 2:
            raise ValueError("Need at least 2 training trials for RidgeCV")
        model = RidgeCV(alphas=alphas, cv=n_splits, alpha_per_target=False)
    model.fit(z, y)
    coef = np.asarray(model.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    bias = np.asarray(model.intercept_, dtype=np.float64).ravel()
    selected = np.asarray(model.alpha_, dtype=np.float64)
    if alpha_per_target:
        alpha_value: float | np.ndarray = selected.astype(np.float64, copy=False)
    else:
        alpha_value = float(selected.reshape(-1)[0])
    return coef.astype(np.float32, copy=False), bias.astype(np.float32, copy=False), alpha_value


def _channel_design_gram(
    x: np.ndarray,
    M: np.ndarray,
    y_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build normal-equation blocks for shared ``a``.

    For each trial ``i``: ``U_i[c,p] = <M_p, X_i[c]>``, then
    ``A = sum_i U_i U_i.T``, ``b = sum_i U_i y_i``.
    """
    n, c, h, w = x.shape
    hw = h * w
    m_flat = M.reshape(-1, hw).astype(np.float64, copy=False)
    A = np.zeros((c, c), dtype=np.float64)
    b = np.zeros(c, dtype=np.float64)
    for i in range(n):
        # U: (C, P) via (C, HW) @ (HW, P)
        xi = x[i].reshape(c, hw).astype(np.float64, copy=False)
        U = xi @ m_flat.T
        A += U @ U.T
        b += U @ y_centered[i].astype(np.float64, copy=False)
    return A, b


def _select_alpha_a(
    A_full: np.ndarray,
    b_full: np.ndarray,
    x: np.ndarray,
    M: np.ndarray,
    y_centered: np.ndarray,
    alphas: np.ndarray,
    *,
    cv_folds: int,
    max_cv_trials: int = 96,
) -> float:
    """K-fold over a trial subset to pick Ridge alpha for shared ``a``."""
    n = x.shape[0]
    if len(alphas) == 1:
        return float(alphas[0])

    # Subsample trials for CV to keep ALS wall-time reasonable.
    rng = np.random.default_rng(0)
    if n > max_cv_trials:
        cv_idx = np.sort(rng.choice(n, size=max_cv_trials, replace=False))
        x_cv = x[cv_idx]
        y_cv = y_centered[cv_idx]
    else:
        x_cv = x
        y_cv = y_centered

    n_cv = x_cv.shape[0]
    n_splits = min(int(cv_folds), n_cv)
    eye = np.eye(A_full.shape[0], dtype=np.float64)

    if n_splits < 2:
        best_alpha = float(alphas[0])
        best_sse = float("inf")
        for alpha in alphas:
            try:
                a_hat = np.linalg.solve(A_full + float(alpha) * eye, b_full)
            except np.linalg.LinAlgError:
                a_hat = np.linalg.lstsq(
                    A_full + float(alpha) * eye, b_full, rcond=None
                )[0]
            sse = _a_train_sse(x_cv, M, y_cv, a_hat)
            if sse < best_sse:
                best_sse = sse
                best_alpha = float(alpha)
        return best_alpha

    order = rng.permutation(n_cv)
    folds = np.array_split(order, n_splits)
    best_alpha = float(alphas[0])
    best_sse = float("inf")
    for alpha in alphas:
        alpha_f = float(alpha)
        sse = 0.0
        for hold in folds:
            train_idx = np.setdiff1d(order, hold, assume_unique=False)
            A_tr, b_tr = _channel_design_gram(x_cv[train_idx], M, y_cv[train_idx])
            try:
                a_hat = np.linalg.solve(A_tr + alpha_f * eye, b_tr)
            except np.linalg.LinAlgError:
                a_hat = np.linalg.lstsq(A_tr + alpha_f * eye, b_tr, rcond=None)[0]
            sse += _a_train_sse(x_cv[hold], M, y_cv[hold], a_hat)
        if sse < best_sse:
            best_sse = sse
            best_alpha = alpha_f
    return best_alpha


def _a_train_sse(
    x: np.ndarray,
    M: np.ndarray,
    y_centered: np.ndarray,
    a: np.ndarray,
) -> float:
    n, c, h, w = x.shape
    hw = h * w
    m_flat = M.reshape(-1, hw).astype(np.float64, copy=False)
    a64 = a.astype(np.float64, copy=False)
    sse = 0.0
    for i in range(n):
        xi = x[i].reshape(c, hw).astype(np.float64, copy=False)
        pred = (xi @ m_flat.T).T @ a64
        err = y_centered[i].astype(np.float64, copy=False) - pred
        sse += float(err @ err)
    return sse


def _fit_a_ridge(
    x: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    bias: np.ndarray,
    *,
    alphas: np.ndarray,
    cv_folds: int,
) -> tuple[np.ndarray, float]:
    """Fit shared channel vector ``a`` with Ridge (trial K-fold α)."""
    y_centered = y - bias.reshape(1, -1)
    A, b = _channel_design_gram(x, M, y_centered)
    alpha = _select_alpha_a(
        A, b, x, M, y_centered, alphas, cv_folds=cv_folds
    )
    eye = np.eye(A.shape[0], dtype=np.float64)
    try:
        a = np.linalg.solve(A + alpha * eye, b)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(A + alpha * eye, b, rcond=None)[0]
    return a.astype(np.float32, copy=False), float(alpha)


def _normalize_a(a: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix scale gauge: ``||a||_2 = 1``, absorb scale into ``M``."""
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return a, M
    return (a / norm).astype(np.float32, copy=False), (M * norm).astype(
        np.float32, copy=False
    )


def _predict_masked(
    x: np.ndarray,
    a: np.ndarray,
    M: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Predict masked targets ``(n, n_targets)``."""
    z = _project_channels(x, a)  # (n, H*W)
    # y = Z @ M.T + bias
    return (z @ M.T + bias.reshape(1, -1)).astype(np.float32, copy=False)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true.astype(np.float64) - y_pred.astype(np.float64)
    return float(np.mean(err * err))


def fit_cm_encoder(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alphas: np.ndarray,
    cv_folds: int = 5,
    standardize_features: bool = True,
    alpha_per_target: bool = True,
    target_mask: np.ndarray | None = None,
    spatial_size: tuple[int, int] | None = None,
    n_als_iters: int = 8,
    tol: float = 1e-4,
) -> CMEncodeResult:
    """
    Fit channel×space separable encoder by ALS.

    Parameters
    ----------
    x_train
        ``(n, C, H, W)`` feature maps.
    y_train
        ``(n, H_vsd * W_vsd)`` flattened VSD maps.
    """
    if x_train.ndim != 4:
        raise ValueError(f"Expected X (n,C,H,W), got {x_train.shape}")
    if len(x_train) < 2:
        raise ValueError("Need at least 2 training trials for CM RidgeCV")

    alphas = np.asarray(alphas, dtype=np.float64)
    n, c, h, w = x_train.shape
    feature_shape = (c, h, w)

    x_fit, scaler = _standardize_features(
        x_train, fit=bool(standardize_features)
    )
    if not standardize_features:
        scaler = None

    indices: np.ndarray | None = None
    mask_arr: np.ndarray | None = None
    y_fit = y_train
    if target_mask is not None:
        if spatial_size is None:
            raise ValueError("spatial_size is required when target_mask is set")
        mask_arr = np.asarray(target_mask, dtype=bool)
        indices = flatten_target_mask(mask_arr, spatial_size)
        if y_train.ndim != 2 or y_train.shape[1] != spatial_size[0] * spatial_size[1]:
            raise ValueError(
                f"Expected Y shape (n, {spatial_size[0] * spatial_size[1]}), "
                f"got {y_train.shape}"
            )
        y_fit = select_target_pixels(y_train, indices)

    # Init a uniform over channels.
    a = np.ones(c, dtype=np.float32) / np.sqrt(float(c))
    M = np.zeros((y_fit.shape[1], h * w), dtype=np.float32)
    bias = np.zeros(y_fit.shape[1], dtype=np.float32)
    alpha_M: float | np.ndarray = float(alphas[len(alphas) // 2])
    alpha_a = float(alphas[len(alphas) // 2])
    mse_hist: list[float] = []

    prev_mse = float("inf")
    for it in range(int(n_als_iters)):
        z = _project_channels(x_fit, a)
        M, bias, alpha_M = _fit_M_ridge(
            z,
            y_fit,
            alphas=alphas,
            cv_folds=cv_folds,
            alpha_per_target=alpha_per_target,
        )
        a, alpha_a = _fit_a_ridge(
            x_fit,
            y_fit,
            M,
            bias,
            alphas=alphas,
            cv_folds=cv_folds,
        )
        a, M = _normalize_a(a, M)
        pred = _predict_masked(x_fit, a, M, bias)
        cur_mse = _mse(y_fit, pred)
        mse_hist.append(cur_mse)
        print(
            f"    ALS iter {it + 1}/{n_als_iters} train_mse={cur_mse:.6g} "
            f"alpha_a={alpha_a:g}",
            flush=True,
        )
        if prev_mse < float("inf") and abs(prev_mse - cur_mse) / max(prev_mse, 1e-12) < tol:
            break
        prev_mse = cur_mse

    return CMEncodeResult(
        a=a,
        M=M,
        bias=bias,
        scaler=scaler,
        feature_shape=feature_shape,
        spatial_size=spatial_size or (0, 0),
        alpha_M=alpha_M,
        alpha_a=alpha_a,
        n_als_iters=len(mse_hist),
        train_mse=mse_hist,
        alpha_per_target=alpha_per_target,
        target_mask=mask_arr,
        target_pixel_indices=indices,
    )


def predict_maps_cm(
    result: CMEncodeResult,
    x: np.ndarray,
    spatial_size: tuple[int, int],
) -> np.ndarray:
    """Predict full-FOV maps ``(n, H, W)``; out-of-mask pixels are NaN when masked."""
    if x.ndim != 4:
        raise ValueError(f"Expected X (n,C,H,W), got {x.shape}")
    x_in, _ = _standardize_features(x, scaler=result.scaler, fit=False)
    y_masked = _predict_masked(x_in, result.a, result.M, result.bias)
    if result.target_pixel_indices is not None:
        return scatter_target_pixels(
            y_masked, result.target_pixel_indices, spatial_size, fill=np.nan
        )
    height, width = spatial_size
    return y_masked.reshape(-1, height, width)


def mean_abs_M_map(result: CMEncodeResult) -> np.ndarray:
    """Mean absolute spatial map over targets → ``(H_feat, W_feat)``."""
    c, h, w = result.feature_shape
    del c
    abs_m = np.abs(result.M).astype(np.float64)
    return abs_m.mean(axis=0).reshape(h, w).astype(np.float32)


def mean_signed_M_map(result: CMEncodeResult) -> np.ndarray:
    """Mean signed spatial map over targets → ``(H_feat, W_feat)``."""
    c, h, w = result.feature_shape
    del c
    return result.M.mean(axis=0).reshape(h, w).astype(np.float32)


def alpha_metrics_cm(result: CMEncodeResult) -> dict[str, object]:
    """JSON-serializable alpha / ALS summary."""
    out: dict[str, object] = {
        "alpha_a": float(result.alpha_a),
        "n_als_iters": int(result.n_als_iters),
        "train_mse": [float(x) for x in result.train_mse],
        "alpha_per_target_M": bool(result.alpha_per_target),
    }
    if result.alpha_per_target:
        arr = np.asarray(result.alpha_M, dtype=np.float64).ravel()
        out.update(
            {
                "alpha_M_mean": float(np.mean(arr)),
                "alpha_M_median": float(np.median(arr)),
                "alpha_M_min": float(np.min(arr)),
                "alpha_M_max": float(np.max(arr)),
            }
        )
    else:
        out["alpha_M"] = float(np.asarray(result.alpha_M).reshape(-1)[0])
    return out
