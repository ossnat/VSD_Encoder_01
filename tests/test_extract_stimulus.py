from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.DL_features.preprocess import preprocess_stimulus_rgb
from src.DL_features.schema import stimulus_key, stimulus_map_path


def test_preprocess_stimulus_rgb():
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    out = preprocess_stimulus_rgb(image, input_size=224, imagenet_normalize=False)
    assert out.shape == (3, 224, 224)
    assert float(out[0, 0, 0]) == pytest.approx(128 / 255.0)


def test_stimulus_map_path():
    root = Path("/data/features")
    p = stimulus_map_path(
        root, "gandalf", "resnet18_imagenet", "layer3", "270618b", "condAN1"
    )
    assert p.name == "270618b__condAN1.npy"
    assert stimulus_key("270618b", "condAN1") == "270618b__condAN1"
