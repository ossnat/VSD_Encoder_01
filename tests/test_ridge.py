from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.encoding.ridge import bias_map, pearson_r


def test_pearson_r_perfect():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    assert pearson_r(x, x) == pytest.approx(1.0)


def test_bias_map_shape():
    model = Ridge(fit_intercept=True)
    x = np.random.randn(20, 8)
    y = np.random.randn(20, 100)
    model.fit(x, y)
    from src.encoding.ridge import RidgeEncodeResult

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
