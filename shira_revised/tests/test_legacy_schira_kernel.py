"""Numeric tests for the faithful ``trans_cortex`` monopole port."""

from __future__ import annotations

import numpy as np
import pytest

from shira_revised.src.legacy_schira import (
    ECC_WIDTH_LEGACY,
    ISO_POLAR_GRAD,
    LegacySchiraParams,
    compressed_polar_legacy,
    forward_legacy_schira,
)


def _matlab_monopole_w(x: float, y: float, a: float, alpha: float, k: float) -> complex:
    """Closed-form mirror of ``trans_cortex`` model=2 monopole (abs VF, divide α)."""
    ax, ay = abs(x), abs(y)
    r = float(np.hypot(ax, ay))
    theta = float(np.arctan2(ay, ax))
    t = theta / alpha
    c = (1.0 / np.cosh(np.log(r / a) * ECC_WIDTH_LEGACY)) * ISO_POLAR_GRAD
    corr = t * (1.0 / np.cosh(t)) ** c
    return complex(k * np.log(r * np.exp(1j * corr) + a))


@pytest.mark.parametrize(
    "xy",
    [
        (1.0, -0.9),
        (1.4, -0.7),
        (0.6, -0.75),
        (0.5, -0.4),
        (1.5, -0.4),
        (3.0, -2.0),
    ],
)
def test_forward_matches_closed_form_divide_abs(xy):
    x, y = xy
    params = LegacySchiraParams(a=0.72, alpha=1.5, k=1.0, alpha_mode="divide", vf_abs=True)
    got = forward_legacy_schira(x, y, params)
    exp = _matlab_monopole_w(x, y, 0.72, 1.5, 1.0)
    assert np.isfinite(got)
    np.testing.assert_allclose(complex(got), exp, rtol=0, atol=1e-12)


def test_alpha_divide_vs_multiply_diverge():
    params_d = LegacySchiraParams(a=0.72, alpha=1.5, k=1.0, alpha_mode="divide", vf_abs=True)
    params_m = LegacySchiraParams(a=0.72, alpha=1.5, k=1.0, alpha_mode="multiply", vf_abs=True)
    w_d = forward_legacy_schira(1.0, -0.9, params_d)
    w_m = forward_legacy_schira(1.0, -0.9, params_m)
    assert np.abs(w_d - w_m) > 0.05


def test_compressed_polar_modes():
    theta = np.array([-0.8, 0.8])
    p = LegacySchiraParams(a=0.5, alpha=2.0, alpha_mode="divide")
    np.testing.assert_allclose(compressed_polar_legacy(theta, p), theta / 2.0)
    p2 = LegacySchiraParams(a=0.5, alpha=2.0, alpha_mode="multiply")
    np.testing.assert_allclose(compressed_polar_legacy(theta, p2), 2.0 * theta)


def test_alpha_one_divide_equals_multiply_on_abs_vf():
    """When α=1, divide and multiply coincide (same abs VF)."""
    pd = LegacySchiraParams(a=0.72, alpha=1.0, k=1.0, alpha_mode="divide", vf_abs=True)
    pm = LegacySchiraParams(a=0.72, alpha=1.0, k=1.0, alpha_mode="multiply", vf_abs=True)
    x = np.array([0.5, 1.0, 2.0])
    y = np.array([-0.4, -0.9, -1.5])
    np.testing.assert_allclose(
        forward_legacy_schira(x, y, pd),
        forward_legacy_schira(x, y, pm),
        atol=1e-12,
    )
