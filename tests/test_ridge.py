from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.encoding.ridge import RidgeEncodeResult, bias_map, pearson_r
from src.encoding.ridge_plotting import select_one_trial_per_condition


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
