from __future__ import annotations

import numpy as np

from src.retinotopy.schira import (
    FA_COMBINE_POWER,
    SHEAR_CONSTANT,
    SHEAR_DOUBLE_SECH,
    SchiraParams,
    cartesian_to_polar,
    compressed_polar,
    forward_schira,
    fovea_w,
    inverse_schira,
    polar_to_cartesian,
    shear_fa,
)
from src.retinotopy.visual_field import (
    cartesian_deg_to_image_px,
    image_px_to_cartesian_deg,
    image_px_to_polar,
    stimulus_ink_mask,
)
from src.stimuli.catalog import csv_date_to_h5_prefix, h5_session_id
from src.stimuli.render import RenderConfig, _deg_point_to_px


def test_session_date_201118():
    assert csv_date_to_h5_prefix("20/11/2018") == "201118"
    assert csv_date_to_h5_prefix("20/11/18") == "201118"
    assert h5_session_id("20/11/2018", "a") == "201118a"


def test_hm_maps_to_real_axis():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    ecc = np.array([0.5, 1.0, 2.0, 4.0])
    polar = np.zeros_like(ecc)
    w = forward_schira(ecc, polar, params)
    np.testing.assert_allclose(w.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(w.real, params.k * np.log(ecc + params.a))


def test_fovea_is_k_log_a():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    w = forward_schira(0.0, 0.0, params)
    assert np.isclose(complex(w), fovea_w(params))
    assert np.isclose(w.imag, 0.0)


def test_k_scales_w():
    p1 = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    p2 = SchiraParams(a=0.72, alpha=1.5, k=2.0)
    w1 = forward_schira(1.2, -0.4, p1)
    w2 = forward_schira(1.2, -0.4, p2)
    np.testing.assert_allclose(w2, 2.0 * w1)


def test_a_shifts_hm_real_part():
    small = SchiraParams(a=0.5, alpha=1.0, k=1.0)
    large = SchiraParams(a=1.5, alpha=1.0, k=1.0)
    w_small = forward_schira(1.0, 0.0, small)
    w_large = forward_schira(1.0, 0.0, large)
    assert w_small.real < w_large.real


def test_alpha_stretches_polar_angle():
    lo = SchiraParams(a=0.72, alpha=1.0, k=1.0, shear=SHEAR_CONSTANT)
    hi = SchiraParams(a=0.72, alpha=1.5, k=1.0, shear=SHEAR_CONSTANT)
    w_lo = forward_schira(1.0, -0.4, lo)
    w_hi = forward_schira(1.0, -0.4, hi)
    assert abs(w_hi.imag) > abs(w_lo.imag)


def test_round_trip_constant_shear():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0, shear=SHEAR_CONSTANT)
    rng = np.random.default_rng(0)
    # Stay away from the VM: alpha=1.5 can push |θ α| past π/2 (branch cut).
    x = rng.uniform(0.4, 5.0, size=40)
    y = rng.uniform(-2.0, -0.05, size=40)
    ecc, polar = cartesian_to_polar(x, y)
    w = forward_schira(ecc, polar, params)
    e2, p2 = inverse_schira(w, params)
    x2, y2 = polar_to_cartesian(e2, p2)
    finite = np.isfinite(e2)
    assert finite.any()
    np.testing.assert_allclose(x2[finite], x[finite], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(y2[finite], y[finite], atol=1e-8, rtol=1e-6)


def test_round_trip_double_sech():
    params = SchiraParams(
        a=0.72, alpha=1.5, k=1.0, shear=SHEAR_DOUBLE_SECH, fa_combine=FA_COMBINE_POWER
    )
    rng = np.random.default_rng(1)
    # |α θ| must stay below the P sech(P) peak (~1.20) or the map folds.
    theta_lim = 1.15 / params.alpha
    ecc = rng.uniform(0.4, 5.0, size=50)
    polar = rng.uniform(-theta_lim, -0.02, size=50)
    x, y = polar_to_cartesian(ecc, polar)
    w = forward_schira(ecc, polar, params)
    e2, p2 = inverse_schira(w, params)
    x2, y2 = polar_to_cartesian(e2, p2)
    finite = np.isfinite(e2)
    assert finite.sum() == ecc.size
    np.testing.assert_allclose(x2[finite], x[finite], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(y2[finite], y[finite], atol=1e-6, rtol=1e-5)


def test_double_sech_letter_round_trip():
    """Catalog letter at (1.0, -0.9) deg is inside the injective P branch."""
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine=FA_COMBINE_POWER)
    ecc, theta = cartesian_to_polar(1.0, -0.9)
    assert abs(params.alpha * theta) < 1.20
    w = forward_schira(ecc, theta, params)
    e2, th2 = inverse_schira(w, params)
    x2, y2 = polar_to_cartesian(e2, th2)
    np.testing.assert_allclose([x2, y2], [1.0, -0.9], atol=1e-7)


def test_p_is_alpha_theta():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    theta = np.array([-0.4, -0.9, 0.0])
    np.testing.assert_allclose(compressed_polar(theta, params), 1.5 * theta)


def test_double_sech_fa_power_is_one_on_hm():
    """At P=0: sech(0)**(...) = 1."""
    params = SchiraParams(
        a=0.72, alpha=1.5, k=1.0, fa_combine=FA_COMBINE_POWER, sech_amp=0.1821
    )
    fa = shear_fa(np.array([1.0]), np.array([0.0]), params)
    np.testing.assert_allclose(fa, 1.0)
    assert params.fa_combine == FA_COMBINE_POWER


def test_fa_combine_product_and_mult_are_rejected():
    for bad in ("product", "mult", "multiply", "mul"):
        try:
            SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine=bad)
        except ValueError as exc:
            assert "power" in str(exc).lower() or "not implemented" in str(exc).lower()
        else:
            raise AssertionError(f"expected ValueError for fa_combine={bad!r}")


def test_schira2007_power_puts_s2_inside_exponent():
    """Eq.5 typography: fa = sech(P) ** (sech_E * S2), not (sech**sech_E)*S2."""
    params = SchiraParams(
        a=0.72, alpha=1.5, k=1.0, fa_combine=FA_COMBINE_POWER, sech_amp=0.1821
    )
    p = np.array([-0.8])
    ecc = np.array([1.2])
    fa = shear_fa(ecc, p, params)
    s_p = 1.0 / np.cosh(p)
    s_e = 1.0 / np.cosh(np.log(ecc / 0.72) * 0.76)
    expected = np.power(s_p, s_e * 0.1821)
    wrong_old = np.power(s_p, s_e) * 0.1821
    np.testing.assert_allclose(fa, expected)
    assert not np.allclose(fa, wrong_old)


def test_forward_letter_cloud_area_with_published_sech_amp():
    """Parafoveal 1° disk at (1,-0.9) stays a filled 2D cloud under power fa."""
    from numpy.linalg import svd

    rng_x = np.linspace(0.55, 1.45, 25)
    rng_y = np.linspace(-1.35, -0.45, 25)
    xx, yy = np.meshgrid(rng_x, rng_y)
    inside = (xx - 1.0) ** 2 + (yy + 0.9) ** 2 <= 0.5**2
    x = xx[inside]
    y = yy[inside]
    ecc, theta = cartesian_to_polar(x, y)
    params = SchiraParams(
        a=0.72,
        alpha=1.5,
        k=1.0,
        sech_amp=0.1821,
        fa_combine=FA_COMBINE_POWER,
    )
    w = forward_schira(ecc, theta, params)
    ok = np.isfinite(w.real)
    pts = np.column_stack([w.real[ok], w.imag[ok]])
    _, s, _ = svd(pts - pts.mean(0), full_matrices=False)
    aspect = float(s[1] / s[0])
    v_span = float(w.imag[ok].max() - w.imag[ok].min())
    assert aspect >= 0.2, f"aspect={aspect}"
    assert v_span >= 0.15, f"v_span={v_span}"


def test_inverse_outside_injective_branch_is_nan():
    """Do not clip out-of-range angles onto the P-sech peak (false wedges)."""
    params = SchiraParams(
        a=0.72,
        alpha=1.5,
        k=1.0,
        sech_amp=0.1821,
        fa_combine=FA_COMBINE_POWER,
    )
    # Angle far beyond |P sech(P)^exp| at this E (power inverse must NaN).
    ecc = 1.3
    bad_angle = 8.0
    z = ecc * np.exp(1j * bad_angle)
    e, th = inverse_schira(params.k * np.log(z + params.a), params)
    assert np.isnan(e) and np.isnan(th)


def test_left_hemifield_is_nan():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    # z with Re(z)<0 is left visual field; inverse must return NaN.
    z_left = -0.2 + 0.1j
    shifted = z_left + params.a
    assert shifted.real > 0
    w_left = params.k * np.log(shifted)
    e, _p = inverse_schira(w_left, params)
    assert np.isnan(e)
    e_ok, _ = inverse_schira(forward_schira(1.0, -0.3, params), params)
    assert np.isfinite(e_ok)


def test_image_px_matches_renderer():
    cfg = RenderConfig()
    x_deg, y_deg = 1.0, -0.9
    x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)
    ref = _deg_point_to_px(x_deg, y_deg, cfg)
    np.testing.assert_allclose((x_px, y_px), ref)
    x2, y2 = image_px_to_cartesian_deg(x_px, y_px, cfg)
    np.testing.assert_allclose((x2, y2), (x_deg, y_deg))


def test_image_px_to_polar_letter_position():
    cfg = RenderConfig()
    x_deg, y_deg = 1.0, -0.9
    x_px, y_px = cartesian_deg_to_image_px(x_deg, y_deg, cfg)
    ecc, theta = image_px_to_polar(x_px, y_px, cfg)
    e_ref, th_ref = cartesian_to_polar(x_deg, y_deg)
    np.testing.assert_allclose(ecc, e_ref)
    np.testing.assert_allclose(theta, th_ref)
    np.testing.assert_allclose(theta, np.arctan2(y_deg, x_deg))
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    np.testing.assert_allclose(compressed_polar(theta, params), params.alpha * theta)


def test_ink_mask_ignores_gray():
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    rgb[2, 3] = 0
    mask = stimulus_ink_mask(rgb, background_gray=128)
    assert mask[2, 3]
    assert not mask[0, 0]


def test_july10_h5_prefix():
    assert csv_date_to_h5_prefix("10/7/2018") == "100718"
    assert csv_date_to_h5_prefix("10/07/18") == "100718"
    assert h5_session_id("10/7/2018", "a") == "100718a"
