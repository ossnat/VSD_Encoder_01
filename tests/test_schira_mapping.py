from __future__ import annotations

import numpy as np

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.register import (
    affine_to_mapping,
    fit_affine_from_w_pixels,
    landmark_mode_from_pairs,
    landmarks_pairs_to_w,
    schira_uv_to_w,
)
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
from src.retinotopy.warp import (
    fill_stimulus_ink_holes,
    forward_ink_cloud_w,
    forward_ink_mask_on_vsd,
    sample_stimulus_onto_cortical_w,
    sample_stimulus_onto_vsd_grid,
    warped_ink_mask,
)
from src.stimuli.catalog import csv_date_to_h5_prefix, h5_session_id
from src.stimuli.render import RenderConfig, _deg_point_to_px


def test_fit_affine_from_w_pixels_roundtrip():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine="power")
    true = CorticalAffine(
        origin_x=40.0,
        origin_y=30.0,
        pixels_per_unit=22.0,
        rotation_deg=35.0,
        flip_u=False,
        flip_v=True,
        needs_confirmation=True,
    )
    # Synthetic cortical points
    w = np.array(
        [
            complex(0.4, -0.5),
            complex(0.7, -0.3),
            complex(0.55, -0.8),
            complex(0.9, -0.6),
        ],
        dtype=np.complex128,
    )
    rows, cols = true.w_to_pixel(w, params)
    fitted, meta = fit_affine_from_w_pixels(w, rows, cols, params)
    assert meta["rmsd_px"] < 1e-6
    # Flip×rotation has discrete equivalents; require geometric fidelity.
    pred_r, pred_c = fitted.w_to_pixel(w, params)
    np.testing.assert_allclose(pred_r, rows, atol=1e-5)
    np.testing.assert_allclose(pred_c, cols, atol=1e-5)
    np.testing.assert_allclose(fitted.origin_x, true.origin_x, atol=1e-5)
    np.testing.assert_allclose(fitted.origin_y, true.origin_y, atol=1e-5)
    np.testing.assert_allclose(fitted.pixels_per_unit, true.pixels_per_unit, atol=1e-5)
    m = affine_to_mapping(fitted)
    assert "origin_xy" in m and m["pixels_per_unit"] > 0
    assert isinstance(m["flip_u"], bool) and isinstance(m["flip_v"], bool)


def test_landmarks_schira_uv_to_w_and_fit():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine="power")
    w = np.array([0.4 + 0.5j, 0.7 - 0.3j, 0.55 - 0.8j], dtype=np.complex128)
    pairs = [
        {"schira_u": float(w[i].real), "schira_v": float(w[i].imag), "vsd_col": 0.0, "vsd_row": 0.0}
        for i in range(w.size)
    ]
    assert landmark_mode_from_pairs(pairs) == "schira_uv"
    w_out = landmarks_pairs_to_w(pairs, params=params, render_cfg=RenderConfig())
    np.testing.assert_allclose(w_out, w)
    np.testing.assert_allclose(schira_uv_to_w(w.real, w.imag), w)


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


def test_fill_stimulus_ink_holes_fills_letter_interior():
    rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
    # Hollow square (outline only).
    rgb[8:24, 8] = 0
    rgb[8:24, 23] = 0
    rgb[8, 8:24] = 0
    rgb[23, 8:24] = 0
    filled = fill_stimulus_ink_holes(
        rgb, background_gray=128, fill_value=0, thicken_px=0
    )
    assert filled[16, 16, 0] == 0
    assert filled[0, 0, 0] == 128
    thick = fill_stimulus_ink_holes(
        rgb, background_gray=128, fill_value=0, thicken_px=2
    )
    assert int(stimulus_ink_mask(thick).sum()) > int(stimulus_ink_mask(filled).sum())


def test_forward_ink_mask_on_vsd_is_thin_not_blob():
    """Forward stamp uses ink pixels only (no dilate); stays sparse."""
    cfg = RenderConfig(canvas_size=64, pixels_per_deg=64.0 / 6.0, quadrant_extent_deg=6.0)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    # Thin stroke in lower-right VF quadrant (fixation at top-left).
    img[12, 8:36] = 0
    img[12:28, 8] = 0
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine="power", sech_amp=0.1821)
    _u, _v, w = forward_ink_cloud_w(img, params=params, render_cfg=cfg)
    assert w.size > 0
    # Place fovea so the mapped stroke lands inside a 100×100 camera.
    affine = CorticalAffine(
        origin_x=50.0,
        origin_y=50.0,
        pixels_per_unit=25.0,
        rotation_deg=0.0,
        flip_u=False,
        flip_v=False,
        needs_confirmation=False,
    )
    ink = forward_ink_mask_on_vsd(
        img, params=params, affine=affine, render_cfg=cfg, spatial_size=(100, 100)
    )
    n = int(ink.sum())
    assert 10 <= n <= 120  # sparse stroke, not a filled blob
    r, c = affine.w_to_pixel(w, params)
    rr = np.rint(r).astype(int)
    cc = np.rint(c).astype(int)
    ok = (rr >= 0) & (rr < 100) & (cc >= 0) & (cc < 100)
    assert ok.any()
    assert ink[rr[ok], cc[ok]].all()


def test_warped_ink_is_filled_mask_not_edge_only():
    """Inverse sampling + ink threshold keeps a filled disk filled (not edges)."""
    cfg = RenderConfig(canvas_size=64, pixels_per_deg=64.0 / 6.0, quadrant_extent_deg=6.0)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    x_px, y_px = cartesian_deg_to_image_px(1.0, -0.5, cfg)
    ys, xs = np.ogrid[:64, :64]
    disk = (xs - x_px) ** 2 + (ys - y_px) ** 2 <= (0.35 * cfg.pixels_per_deg) ** 2
    img[disk] = 0

    params = SchiraParams(a=0.72, alpha=1.0, k=1.0, sech_amp=1.821)
    affine = CorticalAffine(
        origin_x=20.0, origin_y=10.0, pixels_per_unit=25.0, needs_confirmation=False
    )
    warped, valid = sample_stimulus_onto_vsd_grid(
        img,
        params=params,
        affine=affine,
        render_cfg=cfg,
        spatial_size=(64, 64),
    )
    ink = warped_ink_mask(warped, valid, background_gray=128)
    assert int(ink.sum()) >= 20
    # Edge-only would be a thin ring; filled mass should exceed a 1-px perimeter budget.
    from scipy import ndimage

    eroded = ndimage.binary_erosion(ink)
    assert int(eroded.sum()) >= 5, "ink should have interior pixels (filled, not edge)"


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


def test_affine_round_trip():
    params = SchiraParams(a=0.72, alpha=1.5, k=1.0)
    affine = CorticalAffine(origin_x=50.0, origin_y=20.0, pixels_per_unit=45.0)
    w = affine.pixel_to_w(30.0, 60.0, params)
    row, col = affine.w_to_pixel(w, params)
    np.testing.assert_allclose([row, col], [30.0, 60.0], atol=1e-10)


def test_sample_synthetic_blob():
    cfg = RenderConfig(canvas_size=32, pixels_per_deg=32.0 / 6.0, quadrant_extent_deg=6.0)
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    # White square at (1°, -1°) ≈ 5.3 px from top-left.
    x_px, y_px = cartesian_deg_to_image_px(1.0, -1.0, cfg)
    r0, c0 = int(round(y_px)), int(round(x_px))
    img[max(r0 - 2, 0) : r0 + 3, max(c0 - 2, 0) : c0 + 3] = 255

    params = SchiraParams(a=0.72, alpha=1.0, k=1.0)
    affine = CorticalAffine(
        origin_x=16.0, origin_y=8.0, pixels_per_unit=20.0, needs_confirmation=False
    )
    warped, valid = sample_stimulus_onto_vsd_grid(
        img,
        params=params,
        affine=affine,
        render_cfg=cfg,
        spatial_size=(32, 32),
    )
    ink = warped_ink_mask(warped, valid, background_gray=128)
    assert valid.any()
    assert ink.any()
    # Mass of the warped blob should be finite and not fill the FOV.
    assert 2 <= int(ink.sum()) < 32 * 32 * 0.5


def test_sample_stimulus_onto_cortical_w_no_affine():
    """Schira-only grid: letter support appears in (u,v) without camera affine."""
    cfg = RenderConfig(canvas_size=64, pixels_per_deg=64.0 / 6.0, quadrant_extent_deg=6.0)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    x_px, y_px = cartesian_deg_to_image_px(1.0, -0.9, cfg)
    r0, c0 = int(round(y_px)), int(round(x_px))
    img[max(r0 - 3, 0) : r0 + 4, max(c0 - 3, 0) : c0 + 4] = 0

    params = SchiraParams(
        a=0.72, alpha=1.5, k=1.0, shear=SHEAR_DOUBLE_SECH, sech_amp=1.821
    )
    fu, fv, w = forward_ink_cloud_w(img, params=params, render_cfg=cfg)
    assert w.size > 10
    assert np.isfinite(fu).all() and np.isfinite(fv).all()

    warped, valid, u_range, v_range = sample_stimulus_onto_cortical_w(
        img,
        params=params,
        render_cfg=cfg,
        grid_size=64,
        max_ecc_deg=6.0 * np.sqrt(2.0),
    )
    ink = warped_ink_mask(warped, valid, background_gray=128)
    assert valid.any()
    assert ink.any()
    assert u_range[0] < u_range[1]
    assert v_range[0] < v_range[1]
    # Forward cloud should sit inside the plotted bbox.
    assert fu.min() >= u_range[0] - 1e-6
    assert fu.max() <= u_range[1] + 1e-6
    assert fv.min() >= v_range[0] - 1e-6
    assert fv.max() <= v_range[1] + 1e-6


def test_ink_mask_ignores_gray():
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    rgb[2, 3] = 0
    mask = stimulus_ink_mask(rgb, background_gray=128)
    assert mask[2, 3]
    assert not mask[0, 0]


def test_load_named_set_from_registry():
    from src.paths import project_root
    from src.retinotopy.params import list_schira_sets, load_schira_set

    path = project_root() / "configs/schira/sets.yaml"
    assert "201118" in list_schira_sets(path)
    assert "100718" in list_schira_sets(path)
    name, params, affine, block = load_schira_set(path, set_name="201118")
    assert name == "201118"
    assert params.a == 0.72
    assert params.alpha == 1.5
    assert params.k == 1.0
    assert params.shear == SHEAR_DOUBLE_SECH
    assert params.sech_ecc_k == 0.76
    assert params.sech_amp == 0.1821
    assert params.fa_combine == FA_COMBINE_POWER
    assert block["date_prefix"] == "201118"
    np.testing.assert_allclose(affine.pixels_per_unit, 53.676406289148666)
    np.testing.assert_allclose(affine.origin_x, -5.90836181632753)
    np.testing.assert_allclose(affine.origin_y, 22.878432895128505)
    np.testing.assert_allclose(affine.rotation_deg, 130.0061123284688)
    assert affine.flip_u is True
    assert affine.flip_v is False
    assert affine.needs_confirmation is False
    name2, params2, _, block2 = load_schira_set(path, set_name="100718")
    assert name2 == "100718"
    assert params2.a == 0.74
    assert params2.alpha == 1.0
    assert params2.k == 1.0
    assert params2.shear == SHEAR_DOUBLE_SECH
    assert params2.sech_amp == 0.1821
    assert params2.fa_combine == FA_COMBINE_POWER
    assert block2["date_prefix"] == "100718"


def test_july10_h5_prefix():
    assert csv_date_to_h5_prefix("10/7/2018") == "100718"
    assert csv_date_to_h5_prefix("10/07/18") == "100718"
    assert h5_session_id("10/7/2018", "a") == "100718a"


def test_load_standalone_session_yaml():
    from src.paths import project_root
    from src.retinotopy.params import load_schira_set

    path = project_root() / "configs/schira/session_201118.yaml"
    name, params, _affine, block = load_schira_set(path)
    assert name == "201118"
    assert params.a == 0.72
    assert params.shear == SHEAR_DOUBLE_SECH
    assert params.sech_amp == 0.1821
    assert params.fa_combine == FA_COMBINE_POWER
    assert block["date_prefix"] == "201118"

    path_july = project_root() / "configs/schira/session_100718.yaml"
    name_j, params_j, _, block_j = load_schira_set(path_july)
    assert name_j == "100718"
    assert params_j.a == 0.74
    assert params_j.alpha == 1.0
    assert params_j.shear == SHEAR_DOUBLE_SECH
    assert params_j.sech_amp == 0.1821
    assert params_j.fa_combine == FA_COMBINE_POWER
    assert block_j["date_prefix"] == "100718"


def test_missing_schira_keys_raise():
    from src.retinotopy.params import params_from_mapping

    try:
        params_from_mapping({"a": 0.72, "k": 1.0})
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing alpha")
