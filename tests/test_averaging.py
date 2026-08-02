from __future__ import annotations

import numpy as np
import pytest

from src.data.averaging import average_frames, baseline_zscore_trial


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


def test_baseline_zscore_then_average():
    # Constant baseline [0, 2) with mean=2, std=0 → eps floor; analysis frames vary.
    trial = np.zeros((4, 6), dtype=np.float32)
    trial[:, 0:2] = 2.0
    trial[:, 2:4] = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)[:, None]
    z = baseline_zscore_trial(trial, 0, 2, eps=1e-8)
    # baseline mean=2, std floored to eps → z = (x-2)/eps
    assert z.shape == trial.shape
    out = average_frames(
        trial,
        2,
        4,
        spatial_size=(2, 2),
        normalization="baseline_zscore",
        baseline_start_frame=0,
        baseline_end_frame=2,
        baseline_std_eps=1.0,
    )
    # With eps=1, z = x-2; mean of frames 2..3 = [2,4,6,8]-2 = [0,2,4,6]
    np.testing.assert_allclose(out.ravel(), np.array([0, 2, 4, 6], dtype=np.float32))


def test_baseline_zscore_nonzero_std():
    rng = np.random.default_rng(0)
    trial = rng.normal(size=(100, 40)).astype(np.float32)
    # Make baseline mean 0 std 1 approximately by construction on first 10 frames
    baseline = trial[:, 2:26]
    mean = baseline.mean(axis=1, keepdims=True)
    std = np.maximum(baseline.std(axis=1, keepdims=True), 1e-8)
    expected = ((trial - mean) / std)[:, 30:35].mean(axis=1).reshape(10, 10)
    out = average_frames(
        trial,
        30,
        35,
        spatial_size=(10, 10),
        normalization="zscore_baseline",
        baseline_start_frame=2,
        baseline_end_frame=26,
        baseline_std_eps=1e-8,
    )
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-5, atol=1e-5)
