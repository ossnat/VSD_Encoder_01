from __future__ import annotations

import numpy as np

from src.evaluation.cornet_compare import (
    center_periphery_masks,
    plot_center_periphery_definition,
    spatial_band_stats,
)


def test_center_periphery_masks_split():
    mask = np.zeros((100, 100), dtype=bool)
    yy, xx = np.ogrid[:100, :100]
    mask[(yy - 49.5) ** 2 + (xx - 49.5) ** 2 <= 50**2] = True
    center, periphery = center_periphery_masks(
        (100, 100), eval_mask=mask, center_frac=0.5
    )
    assert center.any()
    assert periphery.any()
    assert not np.any(center & periphery)
    # Both subsets must lie inside eval mask.
    assert np.all(~center | mask)
    assert np.all(~periphery | mask)


def test_spatial_band_stats_ratio():
    arr = np.zeros((10, 10), dtype=np.float32)
    center = np.zeros((10, 10), dtype=bool)
    periphery = np.zeros((10, 10), dtype=bool)
    center[3:7, 3:7] = True
    periphery[0:2, :] = True
    arr[center] = 4.0
    arr[periphery] = 2.0
    stats = spatial_band_stats(arr, center, periphery)
    assert stats["center_mean"] == 4.0
    assert stats["periphery_mean"] == 2.0
    assert stats["center_over_periphery"] == 2.0


def test_plot_center_periphery_definition(tmp_path):
    mask = np.zeros((100, 100), dtype=bool)
    yy, xx = np.ogrid[:100, :100]
    mask[(yy - 49.5) ** 2 + (xx - 49.5) ** 2 <= 50**2] = True
    out = plot_center_periphery_definition(
        (100, 100),
        tmp_path / "def.png",
        eval_mask=mask,
        center_frac=0.5,
    )
    assert out.exists()
    assert out.stat().st_size > 0
