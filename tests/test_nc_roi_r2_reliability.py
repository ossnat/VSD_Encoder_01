from __future__ import annotations

import numpy as np
import pytest

from experiments.noise_ceiling_roi.nc_roi_utils import (
    pixel_r2_reliability_map,
    pixel_reliability_map,
)


def test_pixel_r2_reliability_perfect_odd_even():
    t, h, w = 8, 3, 3
    base = np.random.randn(t // 2, h, w).astype(np.float32)
    trials = np.zeros((t, h, w), dtype=np.float32)
    trials[0::2] = base
    trials[1::2] = base
    r = pixel_reliability_map(trials)
    r2 = pixel_r2_reliability_map(trials)
    assert np.nanmean(r) == pytest.approx(1.0, abs=1e-5)
    assert np.nanmean(r2) == pytest.approx(1.0, abs=1e-5)


def test_pixel_r2_reliability_scaled_even_half():
    t, h, w = 6, 2, 2
    base = np.random.randn(t // 2, h, w).astype(np.float32)
    trials = np.zeros((t, h, w), dtype=np.float32)
    trials[0::2] = base
    trials[1::2] = 2.0 * base
    r = pixel_reliability_map(trials)
    r2 = pixel_r2_reliability_map(trials)
    assert np.nanmean(r) == pytest.approx(1.0, abs=1e-5)
    assert np.nanmean(r2) < 1.0
