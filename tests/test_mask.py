from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.mask import (
    apply_mask_nan,
    masked_map_summary,
    masked_pearson_r,
    mask_from_eval_cfg,
    region_mask,
)


def test_region_mask_circle_center_inside():
    mask = region_mask((100, 100), mask_type="circle", radius=35)
    assert mask.shape == (100, 100)
    assert mask[50, 50]
    assert not mask[0, 0]
    assert mask.sum() > 3000


def test_region_mask_matches_foundation_center():
    mask = region_mask((100, 100), mask_type="circle", radius=30)
    cy, cx = 50.0, 50.0
    assert mask[50, 50]
    assert mask[int(cy), int(cx - 30)]
    assert not mask[0, 0]


def test_mask_from_eval_cfg_disabled():
    assert mask_from_eval_cfg({"use_mask": False}, (100, 100)) is None
    assert mask_from_eval_cfg(None, (100, 100)) is None


def test_mask_from_eval_cfg_enabled():
    mask = mask_from_eval_cfg(
        {"use_mask": True, "mask_type": "circle", "mask_radius": 35},
        (100, 100),
    )
    assert mask is not None
    assert mask.dtype == bool


def test_masked_pearson_r_perfect():
    h, w = 10, 10
    mask = region_mask((h, w), mask_type="circle", radius=4)
    x = np.random.randn(h, w).astype(np.float32)
    assert masked_pearson_r(x, x, mask) == pytest.approx(1.0)


def test_masked_pearson_r_ignores_outside_mask():
    h, w = 8, 8
    mask = np.zeros((h, w), dtype=bool)
    mask[2:6, 2:6] = True
    inside = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    y_true = np.zeros((h, w), dtype=np.float32)
    y_pred = np.zeros((h, w), dtype=np.float32)
    y_true[2:6, 2:6] = inside
    y_pred[2:6, 2:6] = inside
    y_pred[0, 0] = 999.0
    assert masked_pearson_r(y_true, y_pred, mask) == pytest.approx(1.0)


def test_apply_mask_nan_2d():
    arr = np.ones((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    out = apply_mask_nan(arr, mask)
    assert np.isnan(out[0, 0])
    assert out[1, 1] == pytest.approx(1.0)


def test_masked_map_summary():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    summary = masked_map_summary(arr, mask)
    assert summary["mean"] == pytest.approx(np.mean(arr[mask]))
