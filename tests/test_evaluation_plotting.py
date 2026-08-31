from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import Normalize

from src.evaluation.plotting import (
    _pixel_mean_map_clims,
    _recon_and_orig_trial_clims,
    plot_recon_with_sample_trial_originals,
)
from src.plotting_colormaps import VSD_CMAP


def _falsecolor(img: np.ndarray, clim: tuple[float, float]) -> np.ndarray:
    lo, hi = clim
    cmap = matplotlib.colormaps[VSD_CMAP]
    return np.asarray(cmap(Normalize(vmin=lo, vmax=hi)(np.asarray(img))))


def test_pixel_mean_map_clims_recon_independent_of_orig():
    recon = np.linspace(-0.4, 0.6, 16, dtype=np.float32).reshape(4, 4)
    orig_hi = np.full((4, 4), 10.0, dtype=np.float32)
    orig_hi[0, 0] = 20.0
    orig_lo = np.full((4, 4), 0.05, dtype=np.float32)
    orig_lo[0, 0] = 0.2
    diff_hi = (recon - orig_hi).astype(np.float32)
    diff_lo = (recon - orig_lo).astype(np.float32)

    orig_a, recon_a, resid_a = _pixel_mean_map_clims(orig_hi, recon, diff_hi)
    orig_b, recon_b, resid_b = _pixel_mean_map_clims(orig_lo, recon, diff_lo)

    assert recon_a == recon_b
    assert orig_a != orig_b
    assert resid_a != resid_b
    np.testing.assert_allclose(_falsecolor(recon, recon_a), _falsecolor(recon, recon_b))


def test_pixel_mean_map_clims_fixed_override_leaves_residual():
    orig = np.ones((4, 4), dtype=np.float32)
    recon = np.full((4, 4), 2.0, dtype=np.float32)
    diff = (recon - orig).astype(np.float32)
    orig_c, recon_c, resid_c = _pixel_mean_map_clims(
        orig, recon, diff, vmin=-1.0, vmax=3.0
    )
    assert orig_c == (-1.0, 3.0)
    assert recon_c == (-1.0, 3.0)
    assert resid_c[0] == pytest.approx(-resid_c[1])
    assert resid_c != orig_c


def test_pixel_mean_map_clims_requires_both_or_neither():
    img = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="vmin and vmax"):
        _pixel_mean_map_clims(img, img, img, vmin=0.0)
    with pytest.raises(ValueError, match="vmin and vmax"):
        _pixel_mean_map_clims(img, img, img, vmax=1.0)
    with pytest.raises(ValueError, match="residual_vmin"):
        _pixel_mean_map_clims(img, img, img, residual_vmin=-0.1)
    with pytest.raises(ValueError, match="residual_vmin"):
        _pixel_mean_map_clims(img, img, img, residual_vmax=0.1)


def test_pixel_mean_map_clims_residual_override_independent_of_orig():
    orig = np.ones((4, 4), dtype=np.float32)
    recon = np.full((4, 4), 2.0, dtype=np.float32)
    diff = (recon - orig).astype(np.float32)
    orig_c, recon_c, resid_c = _pixel_mean_map_clims(
        orig,
        recon,
        diff,
        vmin=0.998,
        vmax=1.003,
        residual_vmin=-0.0015,
        residual_vmax=0.0015,
    )
    assert orig_c == (0.998, 1.003)
    assert recon_c == (0.998, 1.003)
    assert resid_c == (-0.0015, 0.0015)


def test_shared_orig_recon_residual_clims_separates_scales():
    from src.evaluation.plotting import shared_orig_recon_residual_clims

    origs = [np.full((4, 4), 1.0, dtype=np.float32)]
    recons = [np.full((4, 4), 1.002, dtype=np.float32)]
    orig_recon, resid = shared_orig_recon_residual_clims(origs, recons)
    assert orig_recon[0] > 0.99
    assert orig_recon[1] < 1.01
    assert resid[0] == pytest.approx(-resid[1])
    assert resid[1] < 0.1


def test_plot_pixel_mean_maps_writes(tmp_path: Path):
    orig = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    recon = orig * 0.5
    diff = (recon - orig).astype(np.float32)
    from src.evaluation.plotting import plot_pixel_mean_maps

    out = plot_pixel_mean_maps(orig, recon, diff, tmp_path / "m.png", title="t")
    assert out.is_file() and out.stat().st_size > 0


def test_recon_and_orig_trial_clims_independent():
    recon = np.linspace(-0.4, 0.6, 16, dtype=np.float32).reshape(4, 4)
    trials = np.stack(
        [np.full((4, 4), float(i), dtype=np.float32) for i in (0.0, 10.0, 20.0)]
    )
    recon_c, orig_c = _recon_and_orig_trial_clims(recon, trials, [0, 1])
    assert recon_c != orig_c


def test_plot_recon_with_sample_trial_originals_writes(tmp_path: Path):
    recon = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    trials = np.stack([recon + i * 0.1 for i in range(5)], axis=0).astype(np.float32)
    out = plot_recon_with_sample_trial_originals(
        recon,
        trials,
        tmp_path / "c.png",
        title="t",
        n_samples=3,
        seed=17,
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_sanity_mean_triplet_accepts_2d_fold_means(tmp_path: Path):
    from experiments.loo_encoding.run_loo_encoding import _plot_sanity_orig_recon

    orig = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    recon = orig * 0.4
    out = tmp_path / "mean.png"
    _plot_sanity_orig_recon(orig, recon, out_path=out, title="t")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_sanity_sample_trials_opt_in(tmp_path: Path):
    from experiments.loo_encoding.run_loo_encoding import (
        SANITY_LAYOUT_TRIALS,
        _plot_sanity_orig_recon,
    )

    recon = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    trials = np.stack([recon + i * 0.1 for i in range(4)], axis=0).astype(np.float32)
    out = tmp_path / "trials.png"
    _plot_sanity_orig_recon(
        trials,
        recon,
        out_path=out,
        title="t",
        sanity_layout=SANITY_LAYOUT_TRIALS,
        n_sample_trials=3,
        plot_seed=17,
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_sanity_rejects_unknown_layout(tmp_path: Path):
    from experiments.loo_encoding.run_loo_encoding import _plot_sanity_orig_recon

    img = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="sanity_layout"):
        _plot_sanity_orig_recon(
            img, img, out_path=tmp_path / "x.png", title="t", sanity_layout="nope"
        )
