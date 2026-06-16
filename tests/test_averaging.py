from __future__ import annotations

import numpy as np
import pytest

from src.data.averaging import average_frames


def test_average_frames_mean():
    trial = np.arange(300, dtype=np.float32).reshape(100, 3)
    out = average_frames(trial, 0, 2, spatial_size=(10, 10), method="mean")
    assert out.shape == (10, 10)
    expected = trial[:, :2].mean(axis=1).reshape(10, 10)
    np.testing.assert_allclose(out, expected)


def test_average_frames_invalid_window():
    trial = np.zeros((100, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        average_frames(trial, 3, 10, spatial_size=(10, 10))
