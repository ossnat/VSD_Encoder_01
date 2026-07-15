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


def test_stack_condition_mean_maps_and_across_conditions_r():
    import pandas as pd

    from src.evaluation.pixel_correlation import (
        build_condition_entries,
        pixel_correlation_across_conditions,
        pixel_r2_across_conditions,
        stack_condition_mean_maps,
        stack_from_condition_entries,
    )

    # Two conditions, 3 trials each; recon constant within condition.
    h, w = 4, 4
    base = np.linspace(0.5, 1.5, h * w, dtype=np.float32).reshape(h, w)
    originals = np.zeros((6, h, w), dtype=np.float32)
    recons = np.zeros((6, h, w), dtype=np.float32)
    originals[:3] = base + 0.05 * np.random.randn(3, h, w).astype(np.float32)
    recons[:3] = base
    originals[3:] = 2.0 * base + 0.05 * np.random.randn(3, h, w).astype(np.float32)
    recons[3:] = 2.0 * base

    # Condition means ≈ base / 2*base vs recons base / 2*base → r ≈ 1, R² ≈ 1.
    df = pd.DataFrame(
        {
            "date": ["d1"] * 3 + ["d2"] * 3,
            "condition": ["A"] * 3 + ["B"] * 3,
        }
    )
    cond_o, cond_r, meta = stack_condition_mean_maps(df, originals, recons)
    assert cond_o.shape == (2, h, w)
    assert len(meta) == 2
    assert meta[0]["n_trials"] == 3

    entries = build_condition_entries(df, originals, recons)
    cond_o2, cond_r2, _ = stack_from_condition_entries(entries)
    assert np.allclose(cond_o, cond_o2)
    assert np.allclose(cond_r, cond_r2)

    corr = pixel_correlation_across_conditions(cond_o, cond_r)
    assert np.nanmean(corr) == pytest.approx(1.0, abs=1e-5)

    r2 = pixel_r2_across_conditions(cond_o, cond_r)
    assert np.nanmean(r2) == pytest.approx(1.0, abs=5e-2)
