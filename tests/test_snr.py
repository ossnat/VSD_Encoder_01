from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.mask import region_mask
from src.evaluation.snr import map_snr_across_trials, scalar_roi_snr_across_trials


def test_map_snr_constant_signal_zero_noise():
    # Identical trials → std floors at eps → large SNR.
    base = np.ones((8, 8), dtype=np.float32)
    base[2:6, 2:6] = 3.0
    stack = np.stack([base] * 5, axis=0)
    out = map_snr_across_trials(stack, eps=1e-8)
    assert out["n_trials"] == 5
    assert out["snr"] > 1e5
    assert np.allclose(out["mean_map"], base)


def test_map_snr_known_ratio():
    rng = np.random.default_rng(0)
    signal = np.full((10, 10), 2.0, dtype=np.float64)
    noise = rng.normal(0.0, 0.5, size=(40, 10, 10))
    stack = signal[None, ...] + noise
    out = map_snr_across_trials(stack, ddof=1, eps=1e-12)
    # E[|μ|/σ] ≈ 2 / 0.5 = 4 (finite-sample noise).
    assert out["snr"] == pytest.approx(4.0, rel=0.25)


def test_map_snr_mask_and_single_trial():
    stack = np.stack([np.ones((6, 6)), np.full((6, 6), 2.0)], axis=0)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    out = map_snr_across_trials(stack, mask)
    assert out["n_pixels"] == int(mask.sum())
    assert np.isnan(out["snr_map"][~mask]).all()

    single = map_snr_across_trials(stack[:1], mask)
    assert np.isnan(single["snr"])


def test_scalar_roi_snr():
    mask = region_mask((20, 20), mask_type="circle", radius=5)
    highs = np.full((20, 20), 1.0)
    lows = np.full((20, 20), 0.0)
    highs[mask] = 4.0
    lows[mask] = 2.0
    stack = np.stack([highs, lows, highs, lows], axis=0)
    out = scalar_roi_snr_across_trials(stack, mask)
    # ROI means alternate 4 and 2 → μ=3, σ=√(4/3)≈1.1547, SNR≈2.598
    assert out["n_trials"] == 4
    assert out["snr"] == pytest.approx(abs(3.0) / np.std([4.0, 2.0, 4.0, 2.0], ddof=1))
