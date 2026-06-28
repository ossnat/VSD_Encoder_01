from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)


def test_pixel_correlation_perfect_linear():
    t, h, w = 8, 4, 4
    base = np.random.randn(h, w).astype(np.float32)
    coeffs = np.linspace(0.5, 2.0, t, dtype=np.float32)[:, None, None]
    originals = coeffs * base[None]
    reconstructions = originals.copy()
    corr = pixel_correlation_across_trials(originals, reconstructions)
    assert np.nanmean(corr) == pytest.approx(1.0, abs=1e-5)


def test_pixel_correlation_constant_reconstruction_is_nan():
    t, h, w = 6, 3, 3
    originals = np.random.randn(t, h, w).astype(np.float32)
    reconstructions = np.ones((t, h, w), dtype=np.float32)
    corr = pixel_correlation_across_trials(originals, reconstructions)
    assert np.all(np.isnan(corr))


def test_pixel_r2_perfect_prediction():
    t, h, w = 5, 3, 3
    originals = np.random.randn(t, h, w).astype(np.float32)
    r2 = pixel_r2_across_trials(originals, originals.copy())
    assert np.nanmean(r2) == pytest.approx(1.0, abs=1e-5)


def test_pixel_r2_can_be_negative():
    t, h, w = 4, 2, 2
    originals = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    reconstructions = np.zeros((t, h, w), dtype=np.float32)
    r2 = pixel_r2_across_trials(originals, reconstructions)
    assert np.nanmean(r2) < 0.0


def test_pixel_correlation_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        pixel_correlation_across_trials(
            np.zeros((2, 3, 3), dtype=np.float32),
            np.zeros((3, 3, 3), dtype=np.float32),
        )
