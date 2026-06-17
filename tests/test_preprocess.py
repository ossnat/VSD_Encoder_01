from __future__ import annotations

import numpy as np

from src.DL_features.preprocess import preprocess_image


def test_preprocess_none_scaling():
    image = np.random.randn(100, 100).astype(np.float32)
    out = preprocess_image(
        image,
        input_size=224,
        input_channels=3,
        input_scaling="none",
        imagenet_normalize=False,
    )
    assert out.shape == (3, 224, 224)


def test_preprocess_per_image_zscore():
    image = np.random.randn(100, 100).astype(np.float32)
    out = preprocess_image(
        image,
        input_size=224,
        input_channels=3,
        input_scaling="per_image_zscore",
        imagenet_normalize=True,
    )
    assert out.shape == (3, 224, 224)


def test_preprocess_minmax():
    image = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    out = preprocess_image(
        image,
        input_size=64,
        input_channels=1,
        input_scaling="minmax_01",
        imagenet_normalize=False,
    )
    assert out.shape == (1, 64, 64)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
