"""RidgeCV encoding model training and prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.DL_features.schema import stimulus_key, stimulus_map_path
from src.paths import resolve_data_path


@dataclass
class RidgeEncodeResult:
    model: RidgeCV
    scaler: StandardScaler | None
    alpha: float | np.ndarray
    spatial_size: tuple[int, int]
    feature_layer: str
    model_slug: str
    alpha_per_target: bool = False
    # When set, model outputs only in-mask pixels; predict_maps scatters to full FOV.
    target_mask: np.ndarray | None = None
    target_pixel_indices: np.ndarray | None = field(default=None, repr=False)


def flatten_target_mask(mask: np.ndarray, spatial_size: tuple[int, int]) -> np.ndarray:
    """Return sorted flat indices of True pixels in a (H, W) boolean mask."""
    height, width = spatial_size
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != (height, width):
        raise ValueError(
            f"target_mask shape {mask_arr.shape} != spatial_size {(height, width)}"
        )
    indices = np.flatnonzero(mask_arr.ravel())
    if indices.size == 0:
        raise ValueError("target_mask has no True pixels")
    return indices.astype(np.int64, copy=False)


def select_target_pixels(y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Select columns of flattened Y ``(n, H*W)`` at ``indices`` → ``(n, n_mask)``."""
    if y.ndim != 2:
        raise ValueError(f"Expected 2D Y, got shape {y.shape}")
    return y[:, indices]


def scatter_target_pixels(
    y_masked: np.ndarray,
    indices: np.ndarray,
    spatial_size: tuple[int, int],
    *,
    fill: float = np.nan,
) -> np.ndarray:
    """
    Scatter masked targets ``(n, n_mask)`` back to full maps ``(n, H, W)``.

    Out-of-mask pixels are filled with ``fill`` (default NaN). NaN is required
    for raw F/F0 targets (~1.0): fill=0 poisons shared 1–99% color scales and
    full-FOV R² when the true baseline is near 1.
    """
    height, width = spatial_size
    n_pix = height * width
    y_m = np.asarray(y_masked)
    if y_m.ndim == 1:
        y_m = y_m.reshape(1, -1)
    if y_m.shape[1] != indices.size:
        raise ValueError(
            f"Expected {indices.size} masked targets, got {y_m.shape[1]}"
        )
    out = np.full((y_m.shape[0], n_pix), fill, dtype=np.float32)
    out[:, indices] = y_m.astype(np.float32, copy=False)
    return out.reshape(-1, height, width)


def scatter_vector_to_map(
    values: np.ndarray,
    indices: np.ndarray,
    spatial_size: tuple[int, int],
    *,
    fill: float = 0.0,
) -> np.ndarray:
    """Scatter a 1D vector of length ``n_mask`` onto an ``(H, W)`` map."""
    height, width = spatial_size
    out = np.full(height * width, fill, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32).ravel()
    if vals.size != indices.size:
        raise ValueError(f"Expected {indices.size} values, got {vals.size}")
    out[indices] = vals
    return out.reshape(height, width)

def alpha_metrics(alpha: float | np.ndarray, *, alpha_per_target: bool) -> dict[str, object]:
    """JSON-serializable summary of selected RidgeCV alpha(s)."""
    if alpha_per_target:
        arr = np.asarray(alpha, dtype=np.float64).ravel()
        return {
            "alpha_per_target": True,
            "alpha": float(np.mean(arr)),  # backward-compatible scalar summary
            "alpha_mean": float(np.mean(arr)),
            "alpha_median": float(np.median(arr)),
            "alpha_min": float(np.min(arr)),
            "alpha_max": float(np.max(arr)),
            "n_alphas": int(arr.size),
        }
    return {
        "alpha_per_target": False,
        "alpha": float(alpha),
    }


def _flatten_features(feat_map: np.ndarray) -> np.ndarray:
    return feat_map.astype(np.float32).reshape(-1)


def _load_target(nc_path: Path, spatial_size: tuple[int, int]) -> np.ndarray:
    da = xr.open_dataarray(nc_path)
    image = da.values.astype(np.float32)
    da.close()
    height, width = spatial_size
    if image.shape != (height, width):
        raise ValueError(f"Expected target shape {(height, width)}, got {image.shape}")
    return image.reshape(-1)


def attach_feature_paths(
    pairs: pd.DataFrame,
    *,
    features_root: Path,
    monkey: str,
    model_slug: str,
    feature_layer: str,
    repo: Path,
) -> pd.DataFrame:
    ws = repo.resolve().parent
    feature_paths: list[str] = []
    stimulus_keys: list[str] = []
    for row in pairs.itertuples(index=False):
        feat_path = stimulus_map_path(
            features_root,
            monkey,
            model_slug,
            feature_layer,
            str(row.date),
            str(row.condition),
        )
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature map: {feat_path}")
        try:
            rel = str(feat_path.resolve().relative_to(ws))
        except ValueError:
            rel = str(feat_path.resolve())
        feature_paths.append(rel)
        stimulus_keys.append(stimulus_key(str(row.date), str(row.condition)))

    out = pairs.copy()
    out["feature_path"] = feature_paths
    out["stimulus_key"] = stimulus_keys
    return out


def build_xy(
    pairs: pd.DataFrame,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    height, width = spatial_size

    for row in pairs.itertuples(index=False):
        feat = np.load(resolve_data_path(row.feature_path, repo))
        xs.append(_flatten_features(feat))
        ys.append(_load_target(resolve_data_path(row.nc_path, repo), spatial_size))

    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def fit_ridge_encoder(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alphas: np.ndarray,
    cv_folds: int,
    standardize_features: bool,
    alpha_per_target: bool = False,
    target_mask: np.ndarray | None = None,
    spatial_size: tuple[int, int] | None = None,
) -> RidgeEncodeResult:
    """
    Fit RidgeCV mapping features → multi-pixel VSD targets.

    When ``alpha_per_target`` is True, sklearn requires leave-one-out GCV
    (``cv=None``); ``cv_folds`` is ignored in that mode.

    When ``target_mask`` is provided (boolean ``(H, W)``), only in-mask pixels
    are fit as multi-output targets. Pass matching ``spatial_size`` (or set it
    on the returned result). ``predict_maps`` scatters predictions back to the
    full FOV (out-of-mask filled with NaN).
    """
    scaler: StandardScaler | None = None
    x_fit = x_train
    if standardize_features:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)

    if len(x_fit) < 2:
        raise ValueError("Need at least 2 training trials for RidgeCV")

    indices: np.ndarray | None = None
    y_fit = y_train
    mask_arr: np.ndarray | None = None
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

    if alpha_per_target:
        # Per-target α is only supported with efficient LOO GCV (cv=None).
        model = RidgeCV(
            alphas=alphas,
            cv=None,
            alpha_per_target=True,
        )
    else:
        n_splits = min(cv_folds, len(x_fit))
        if n_splits < 2:
            raise ValueError("Need at least 2 training trials for RidgeCV")
        model = RidgeCV(alphas=alphas, cv=n_splits, alpha_per_target=False)

    model.fit(x_fit, y_fit)
    selected = np.asarray(model.alpha_, dtype=np.float64)
    alpha_value: float | np.ndarray
    if alpha_per_target:
        alpha_value = selected.astype(np.float64, copy=False)
    else:
        alpha_value = float(selected.reshape(-1)[0])

    return RidgeEncodeResult(
        model=model,
        scaler=scaler,
        alpha=alpha_value,
        spatial_size=spatial_size or (0, 0),
        feature_layer="",
        model_slug="",
        alpha_per_target=alpha_per_target,
        target_mask=mask_arr,
        target_pixel_indices=indices,
    )


def predict_maps(
    result: RidgeEncodeResult,
    x: np.ndarray,
    spatial_size: tuple[int, int],
) -> np.ndarray:
    x_in = x
    if result.scaler is not None:
        x_in = result.scaler.transform(x)
    y_pred = result.model.predict(x_in)
    height, width = spatial_size
    if result.target_pixel_indices is not None:
        return scatter_target_pixels(
            y_pred, result.target_pixel_indices, spatial_size, fill=np.nan
        )
    return y_pred.reshape(-1, height, width)


def bias_map(result: RidgeEncodeResult, spatial_size: tuple[int, int]) -> np.ndarray:
    height, width = spatial_size
    intercept = np.asarray(result.model.intercept_, dtype=np.float32).ravel()
    if result.target_pixel_indices is not None:
        return scatter_vector_to_map(
            intercept, result.target_pixel_indices, spatial_size, fill=0.0
        )
    return intercept.reshape(height, width)


def weight_norm_map(
    result: RidgeEncodeResult,
    spatial_size: tuple[int, int],
) -> np.ndarray:
    """
    Per-pixel L2 norm of RidgeCV coefficients across features.

    ``coef_`` is ``(n_targets, n_features)``; the map is ``||w_pixel||_2``.
    When a target mask was used, out-of-mask pixels are 0.
    """
    height, width = spatial_size
    coef = np.asarray(result.model.coef_, dtype=np.float64)
    if coef.ndim != 2:
        raise ValueError(f"Expected 2D coef_, got shape {coef.shape}")
    norms = np.linalg.norm(coef, axis=1).astype(np.float32)
    if result.target_pixel_indices is not None:
        if norms.size != result.target_pixel_indices.size:
            raise ValueError(
                f"Expected {result.target_pixel_indices.size} coef rows, "
                f"got {norms.size}"
            )
        return scatter_vector_to_map(
            norms, result.target_pixel_indices, spatial_size, fill=0.0
        )
    if coef.shape[0] != height * width:
        raise ValueError(
            f"Expected {height * width} target rows in coef_, got {coef.shape[0]}"
        )
    return norms.reshape(height, width)


def alpha_map(result: RidgeEncodeResult, spatial_size: tuple[int, int]) -> np.ndarray:
    """Reshape per-target RidgeCV alphas to a spatial map."""
    if not result.alpha_per_target:
        raise ValueError("alpha_map requires alpha_per_target=True")
    height, width = spatial_size
    arr = np.asarray(result.alpha, dtype=np.float64).ravel()
    if result.target_pixel_indices is not None:
        if arr.size != result.target_pixel_indices.size:
            raise ValueError(
                f"Expected {result.target_pixel_indices.size} alphas, got {arr.size}"
            )
        return scatter_vector_to_map(
            arr.astype(np.float32),
            result.target_pixel_indices,
            spatial_size,
            fill=0.0,
        )
    if arr.size != height * width:
        raise ValueError(
            f"Expected {height * width} alphas, got {arr.size}"
        )
    return arr.reshape(height, width)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.ravel().astype(np.float64)
    b = y_pred.ravel().astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    a = a[finite]
    b = b[finite]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
