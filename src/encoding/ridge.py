"""RidgeCV encoding model training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
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
    alpha: float
    spatial_size: tuple[int, int]
    feature_layer: str
    model_slug: str


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
) -> RidgeEncodeResult:
    scaler: StandardScaler | None = None
    x_fit = x_train
    if standardize_features:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)

    n_splits = min(cv_folds, len(x_fit))
    if n_splits < 2:
        raise ValueError("Need at least 2 training trials for RidgeCV")

    model = RidgeCV(alphas=alphas, cv=n_splits)
    model.fit(x_fit, y_train)

    return RidgeEncodeResult(
        model=model,
        scaler=scaler,
        alpha=float(model.alpha_),
        spatial_size=(0, 0),  # filled by caller
        feature_layer="",
        model_slug="",
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
    return y_pred.reshape(-1, height, width)


def bias_map(result: RidgeEncodeResult, spatial_size: tuple[int, int]) -> np.ndarray:
    height, width = spatial_size
    return np.asarray(result.model.intercept_, dtype=np.float32).reshape(height, width)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.ravel().astype(np.float64)
    b = y_pred.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
