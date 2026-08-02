from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.encoding.ridge import RidgeEncodeResult, alpha_map, bias_map, pearson_r, weight_norm_map
from src.encoding.ridge_plotting import plot_alpha_map, plot_weight_norm_map, select_one_trial_per_condition


def test_pearson_r_perfect():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert pearson_r(x, x) == pytest.approx(1.0)


def test_bias_map_shape():
    model = Ridge(fit_intercept=True)
    x = np.random.randn(20, 8)
    y = np.random.randn(20, 100)
    model.fit(x, y)

    result = RidgeEncodeResult(
        model=model,
        scaler=None,
        alpha=1.0,
        spatial_size=(10, 10),
        feature_layer="layer3",
        model_slug="test",
    )
    bias = bias_map(result, (10, 10))
    assert bias.shape == (10, 10)
    wnorm = weight_norm_map(result, (10, 10))
    assert wnorm.shape == (10, 10)
    assert np.all(wnorm >= 0)


def test_fit_ridge_encoder_alpha_per_target(tmp_path):
    from src.encoding.ridge import alpha_metrics, fit_ridge_encoder

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 12))
    y = rng.normal(size=(40, 8))
    result = fit_ridge_encoder(
        x,
        y,
        alphas=np.asarray([0.1, 1.0, 10.0]),
        cv_folds=5,
        standardize_features=True,
        alpha_per_target=True,
    )
    assert result.alpha_per_target is True
    assert np.asarray(result.alpha).shape == (8,)
    summary = alpha_metrics(result.alpha, alpha_per_target=True)
    assert summary["alpha_per_target"] is True
    assert "alpha_mean" in summary

    spatial = (2, 4)
    result.spatial_size = spatial
    amap = alpha_map(result, spatial)
    assert amap.shape == spatial
    underlay = rng.normal(size=spatial).astype(np.float32)
    out = plot_alpha_map(
        amap,
        tmp_path / "alpha.png",
        title="alpha test",
        underlay=underlay,
    )
    assert out.exists()


def test_fit_ridge_encoder_shared_alpha():
    from src.encoding.ridge import alpha_metrics, fit_ridge_encoder

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 12))
    y = rng.normal(size=(40, 8))
    result = fit_ridge_encoder(
        x,
        y,
        alphas=np.asarray([0.1, 1.0, 10.0]),
        cv_folds=5,
        standardize_features=True,
        alpha_per_target=False,
    )
    assert result.alpha_per_target is False
    assert isinstance(result.alpha, float)
    summary = alpha_metrics(result.alpha, alpha_per_target=False)
    assert summary["alpha"] == result.alpha


def test_fit_ridge_encoder_target_mask_subset():
    from src.encoding.ridge import (
        fit_ridge_encoder,
        predict_maps,
        weight_norm_map,
    )

    rng = np.random.default_rng(2)
    spatial = (4, 5)
    n_pix = spatial[0] * spatial[1]
    mask = np.zeros(spatial, dtype=bool)
    mask[1:3, 2:4] = True
    n_mask = int(mask.sum())
    assert n_mask == 4

    x = rng.normal(size=(30, 6))
    y = rng.normal(size=(30, n_pix))
    result = fit_ridge_encoder(
        x,
        y,
        alphas=np.asarray([0.1, 1.0, 10.0]),
        cv_folds=5,
        standardize_features=True,
        alpha_per_target=True,
        target_mask=mask,
        spatial_size=spatial,
    )
    assert result.target_pixel_indices is not None
    assert result.target_pixel_indices.size == n_mask
    assert np.asarray(result.model.coef_).shape[0] == n_mask
    assert np.asarray(result.alpha).shape == (n_mask,)

    preds = predict_maps(result, x[:3], spatial)
    assert preds.shape == (3, *spatial)
    # Out-of-mask predictions are NaN-filled (not 0 — that breaks F/F0 scales).
    assert np.all(np.isnan(preds[:, ~mask]))
    assert np.all(np.isfinite(preds[:, mask]))
    wnorm = weight_norm_map(result, spatial)
    assert wnorm.shape == spatial
    assert np.allclose(wnorm[~mask], 0.0)
    assert np.all(wnorm[mask] >= 0)


def test_select_one_trial_per_condition_prefers_test():
    import pandas as pd

    pairs = pd.DataFrame(
        [
            {"date": "270618b", "condition": "condAN1", "condition_num": 1, "split": "train", "trial_index_in_condition": 0, "trial_global_id": 1},
            {"date": "270618b", "condition": "condAN1", "condition_num": 1, "split": "test", "trial_index_in_condition": 1, "trial_global_id": 2},
            {"date": "270618b", "condition": "condAN2", "condition_num": 2, "split": "test", "trial_index_in_condition": 0, "trial_global_id": 3},
        ]
    )
    out = select_one_trial_per_condition(pairs, prefer_split="test")
    assert len(out) == 2
    cond1 = out[out["condition"] == "condAN1"].iloc[0]
    assert int(cond1["trial_global_id"]) == 2
