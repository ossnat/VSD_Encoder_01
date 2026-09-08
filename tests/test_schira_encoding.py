"""Tests for Schira LUT + local ridge encoding."""

from __future__ import annotations

import numpy as np

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import SchiraParams
from src.schira_encoding.global_channel_model import (
    fit_global_channel_ridge,
    predict_global_channel_ridge,
)
from src.schira_encoding.lut import SchiraLUT, build_schira_lut
from src.schira_encoding.model import fit_local_ridge, predict_local_ridge
from src.schira_encoding.targets import is_anchor_session, warp_target_to_anchor
from src.schira_encoding.warp_features import warp_feature_map, warp_stimulus_rgb
from src.stimuli.render import RenderConfig


def _toy_params() -> SchiraParams:
    return SchiraParams(
        a=0.74,
        alpha=1.0,
        k=1.0,
        shear="double_sech",
        fa_combine="power",
        sech_ecc_k=0.76,
        sech_amp=0.1821,
    )


def _toy_affine() -> CorticalAffine:
    return CorticalAffine(
        origin_x=20.0,
        origin_y=20.0,
        pixels_per_unit=10.0,
        rotation_deg=0.0,
        flip_u=False,
        flip_v=False,
        needs_confirmation=True,
    )


def test_build_lut_roundtrip(tmp_path):
    cfg = RenderConfig(canvas_size=210, pixels_per_deg=35.0, quadrant_extent_deg=6.0)
    lut = build_schira_lut(
        params=_toy_params(),
        affine=_toy_affine(),
        schira_set="100718",
        anchor_session="100718a",
        spatial_size=(32, 32),
        render_cfg=cfg,
        input_size=224,
    )
    assert lut.valid.any()
    assert lut.spatial_size == (32, 32)
    assert np.isfinite(lut.x_deg[lut.valid]).all()

    path = tmp_path / "lut.npz"
    lut.save(path)
    loaded = SchiraLUT.load(path)
    assert loaded.schira_set == "100718"
    assert loaded.anchor_session == "100718a"
    np.testing.assert_array_equal(loaded.valid, lut.valid)
    np.testing.assert_allclose(loaded.stim_row[lut.valid], lut.stim_row[lut.valid])


def test_feature_coords_scale_with_map_size():
    cfg = RenderConfig(canvas_size=210, pixels_per_deg=35.0)
    lut = build_schira_lut(
        params=_toy_params(),
        affine=_toy_affine(),
        schira_set="100718",
        anchor_session="100718a",
        spatial_size=(16, 16),
        render_cfg=cfg,
        input_size=224,
    )
    r56, c56 = lut.feature_sample_coords((56, 56))
    r28, c28 = lut.feature_sample_coords((28, 28))
    m = lut.valid
    np.testing.assert_allclose(r28[m], r56[m] * 0.5, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c28[m], c56[m] * 0.5, rtol=1e-6, atol=1e-6)


def test_warp_feature_map_constant_channels():
    cfg = RenderConfig(canvas_size=210, pixels_per_deg=35.0, quadrant_extent_deg=6.0)
    lut = build_schira_lut(
        params=_toy_params(),
        affine=_toy_affine(),
        schira_set="toy",
        anchor_session="toy_a",
        spatial_size=(40, 40),
        render_cfg=cfg,
        input_size=224,
    )
    assert lut.valid.any(), "toy affine produced empty LUT.valid"
    feat = np.zeros((4, 28, 28), dtype=np.float32)
    for c in range(4):
        feat[c] = float(c + 1)
    warped = warp_feature_map(feat, lut, fill=0.0)
    assert warped.shape == (4, 40, 40)
    feat_row, feat_col = lut.feature_sample_coords((28, 28))
    in_feat = (
        lut.valid
        & (feat_row >= -0.5)
        & (feat_col >= -0.5)
        & (feat_row < 27.5)
        & (feat_col < 27.5)
    )
    assert in_feat.any()
    for c in range(4):
        vals = warped[c][in_feat]
        np.testing.assert_allclose(vals, float(c + 1), atol=1e-5)


def test_warp_stimulus_rgb_matches_canvas():
    cfg = RenderConfig(canvas_size=210, pixels_per_deg=35.0, quadrant_extent_deg=6.0)
    lut = build_schira_lut(
        params=_toy_params(),
        affine=_toy_affine(),
        schira_set="toy",
        anchor_session="toy_a",
        spatial_size=(40, 40),
        render_cfg=cfg,
        input_size=224,
    )
    assert lut.valid.any()
    rgb = np.full((210, 210, 3), 128, dtype=np.uint8)
    # Paint ink at the stimulus location of the first valid VSD pixel.
    r_i, c_i = np.argwhere(lut.valid)[0]
    sr = int(np.clip(np.round(lut.stim_row[r_i, c_i]), 0, 209))
    sc = int(np.clip(np.round(lut.stim_col[r_i, c_i]), 0, 209))
    rgb[sr, sc] = (255, 0, 0)
    warped, valid = warp_stimulus_rgb(rgb, lut, background_gray=128)
    assert valid[r_i, c_i]
    assert warped[r_i, c_i, 0] > 200


def test_local_ridge_recovers_channel_weights():
    rng = np.random.default_rng(0)
    n, c, h, w = 40, 8, 6, 6
    x = rng.normal(size=(n, c, h, w)).astype(np.float64)
    true_w = rng.normal(size=(c, h, w))
    true_b = rng.normal(size=(h, w)) * 0.1
    y = np.einsum("nchw,chw->nhw", x, true_w) + true_b[None, ...]
    y += rng.normal(scale=0.01, size=y.shape)

    valid = np.ones((h, w), dtype=bool)
    result = fit_local_ridge(
        x,
        y,
        valid=valid,
        alphas=[1e-6, 1e-3, 0.1],
        standardize_features=False,
        alpha_per_pixel=True,
    )
    pred = predict_local_ridge(x, result)
    for i in range(h):
        for j in range(w):
            r = np.corrcoef(y[:, i, j], pred[:, i, j])[0, 1]
            assert r > 0.99


def test_global_channel_ridge_recovers_shared_weights():
    rng = np.random.default_rng(1)
    n, c, h, w = 40, 8, 12, 12
    x = rng.normal(size=(n, c, h, w))
    true_w = rng.normal(size=(c,))
    true_b = 0.25
    y = np.einsum("nchw,c->nhw", x, true_w) + true_b
    y += rng.normal(scale=0.01, size=y.shape)
    valid = np.ones((h, w), dtype=bool)
    result = fit_global_channel_ridge(
        x,
        y,
        valid=valid,
        alphas=[1e-6, 1e-3, 0.1],
        standardize_features=False,
    )
    pred = predict_global_channel_ridge(x, result)
    np.testing.assert_allclose(result.weights, true_w, rtol=0.05, atol=0.05)
    assert abs(result.intercept - true_b) < 0.05
    for i in range(0, h, 3):
        for j in range(0, w, 3):
            r = np.corrcoef(y[:, i, j], pred[:, i, j])[0, 1]
            assert r > 0.99


def test_is_anchor_and_identity_target_warp():
    assert is_anchor_session("100718a", "100718a")
    assert not is_anchor_session("100718b", "100718a")
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = warp_target_to_anchor(y, None)
    np.testing.assert_array_equal(out, y)


def test_loo_dir_helpers_share_protocol_leaf(tmp_path):
    from src.schira_encoding.schema import (
        global_channel_ridge_loo_dir,
        local_ridge_loo_dir,
    )

    kwargs = dict(
        window_id="win_0035_0046",
        model_slug="vgg16_imagenet",
        feature_layer="block1_prepool",
        schira_set="100718",
        anchor_session="100718a",
        protocol="B",
        loss_roi="noise_ceiling_hull",
    )
    local = local_ridge_loo_dir(tmp_path, "gandalf", **kwargs)
    glob = global_channel_ridge_loo_dir(tmp_path, "gandalf", **kwargs)
    assert local.name == glob.name == "protocol_B_noise_ceiling_hull"
    assert glob.parent.name == "global_channel_ridge"
    assert local.parent.name == "set-100718__anchor-100718a"
