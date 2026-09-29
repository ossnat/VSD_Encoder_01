"""Torch Schira camera fit: parity with the numpy LUT, and a shift recovery."""

from __future__ import annotations

import numpy as np
import torch

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import SchiraParams
from src.schira_encoding.geometry_fit import GeometryFitConfig, StimulusMean, fit_schira_geometry
from src.schira_encoding.lut import build_schira_lut
from src.schira_encoding.torch_warp import stimulus_sample_coords
from src.stimuli.render import RenderConfig


def _params() -> SchiraParams:
    return SchiraParams(
        a=0.74,
        alpha=1.0,
        k=1.0,
        shear="double_sech",
        fa_combine="power",
        sech_ecc_k=0.76,
        sech_amp=0.1821,
    )


def _affine() -> CorticalAffine:
    return CorticalAffine(
        origin_x=20.0,
        origin_y=18.0,
        pixels_per_unit=12.0,
        rotation_deg=25.0,
        flip_u=True,
        flip_v=False,
        needs_confirmation=True,
    )


def test_torch_coords_match_numpy_lut():
    params = _params()
    affine = _affine()
    render = RenderConfig(canvas_size=80, pixels_per_deg=10.0, quadrant_extent_deg=8.0)
    spatial = (40, 40)
    lut = build_schira_lut(
        params=params,
        affine=affine,
        schira_set="toy",
        anchor_session="toy",
        spatial_size=spatial,
        render_cfg=render,
        input_size=80,
    )
    rows_np, cols_np = np.meshgrid(
        np.arange(spatial[0], dtype=np.float64),
        np.arange(spatial[1], dtype=np.float64),
        indexing="ij",
    )
    row_px, col_px, _gate = stimulus_sample_coords(
        origin_x=torch.tensor(affine.origin_x, dtype=torch.float64),
        origin_y=torch.tensor(affine.origin_y, dtype=torch.float64),
        rotation_deg=torch.tensor(affine.rotation_deg, dtype=torch.float64),
        pixels_per_unit=torch.tensor(affine.pixels_per_unit, dtype=torch.float64),
        flip_u=affine.flip_u,
        flip_v=affine.flip_v,
        rows=torch.from_numpy(rows_np),
        cols=torch.from_numpy(cols_np),
        params=params,
        pixels_per_deg=render.pixels_per_deg,
    )
    valid = lut.valid
    assert int(valid.sum()) > 20
    np.testing.assert_allclose(
        row_px.detach().numpy()[valid], lut.stim_row[valid], atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        col_px.detach().numpy()[valid], lut.stim_col[valid], atol=1e-4, rtol=1e-4
    )


def test_box_stays_inside_limits():
    start = _affine()
    params = _params()
    cfg = GeometryFitConfig(
        optimize="all",
        max_origin_px=20.0,
        max_rotation_deg=20.0,
        max_scale_frac=0.2,
        max_a_frac=0.3,
        max_alpha_frac=0.3,
        max_k_frac=0.3,
    )
    from src.schira_encoding.geometry_fit import _free_from_raw

    raw = torch.tensor(
        [20.0, -20.0, 20.0, -20.0, 20.0, -20.0, 20.0], dtype=torch.float64
    )
    ox, oy, rot, ppu, a_t, alpha_t, k_t = _free_from_raw(raw, start, params, cfg)
    assert abs(float(ox) - start.origin_x) <= 20.0 + 1e-6
    assert abs(float(oy) - start.origin_y) <= 20.0 + 1e-6
    assert abs(float(rot) - start.rotation_deg) <= 20.0 + 1e-6
    ratio = float(ppu) / start.pixels_per_unit
    assert 1.0 / 1.2 - 1e-6 <= ratio <= 1.2 + 1e-6
    assert 1.0 / 1.3 - 1e-6 <= float(a_t) / params.a <= 1.3 + 1e-6
    assert 1.0 / 1.3 - 1e-6 <= float(alpha_t) / params.alpha <= 1.3 + 1e-6
    assert 1.0 / 1.3 - 1e-6 <= float(k_t) / params.k <= 1.3 + 1e-6


def test_keep_geometry_fit_drops_points_and_letters():
    from src.schira_encoding.geometry_fit import keep_geometry_fit_stimulus

    cfg = GeometryFitConfig()
    assert keep_geometry_fit_stimulus("black_triangle_contour_0.4", cfg)
    assert keep_geometry_fit_stimulus("black_circle_contour_0.3", cfg)
    assert keep_geometry_fit_stimulus("black_bar_horizontal_1", cfg)
    assert not keep_geometry_fit_stimulus("black_point_0.1", cfg)
    assert not keep_geometry_fit_stimulus("letter_A_white_1", cfg)
    assert not keep_geometry_fit_stimulus(
        "black_bar_horizontal_1", GeometryFitConfig(exclude_bars=True)
    )


def test_camera_mode_keeps_schira_fixed():
    start = _affine()
    params = _params()
    cfg = GeometryFitConfig(optimize="camera")
    from src.schira_encoding.geometry_fit import _free_from_raw

    raw = torch.tensor([0.4, -0.3, 0.2, 0.1], dtype=torch.float64)
    _ox, _oy, _rot, _ppu, a_t, alpha_t, k_t = _free_from_raw(raw, start, params, cfg)
    assert float(a_t) == params.a
    assert float(alpha_t) == params.alpha
    assert float(k_t) == params.k


def test_schira_mode_keeps_camera_fixed():
    start = _affine()
    params = _params()
    cfg = GeometryFitConfig(optimize="schira")
    from src.schira_encoding.geometry_fit import _free_from_raw

    raw = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64)
    ox, oy, rot, ppu, _a, _alpha, _k = _free_from_raw(raw, start, params, cfg)
    assert float(ox) == start.origin_x
    assert float(oy) == start.origin_y
    assert float(rot) == start.rotation_deg
    assert float(ppu) == start.pixels_per_unit


def test_fit_recovers_a_shifted_origin():
    params = _params()
    true = _affine()
    render = RenderConfig(canvas_size=64, pixels_per_deg=8.0, quadrant_extent_deg=8.0)
    spatial = (36, 36)
    yy, xx = np.mgrid[0:64, 0:64]
    contrast = np.exp(-((xx - 28.0) ** 2 + (yy - 22.0) ** 2) / (2 * 4.0**2)).astype(
        np.float32
    )
    rows_np, cols_np = np.meshgrid(
        np.arange(spatial[0], dtype=np.float64),
        np.arange(spatial[1], dtype=np.float64),
        indexing="ij",
    )
    with torch.no_grad():
        from src.schira_encoding.torch_warp import sample_contrast_maps

        target = sample_contrast_maps(
            torch.from_numpy(contrast).to(torch.float64).view(1, 1, 64, 64),
            origin_x=torch.tensor(true.origin_x, dtype=torch.float64),
            origin_y=torch.tensor(true.origin_y, dtype=torch.float64),
            rotation_deg=torch.tensor(true.rotation_deg, dtype=torch.float64),
            pixels_per_unit=torch.tensor(true.pixels_per_unit, dtype=torch.float64),
            flip_u=true.flip_u,
            flip_v=true.flip_v,
            rows=torch.from_numpy(rows_np),
            cols=torch.from_numpy(cols_np),
            params=params,
            pixels_per_deg=render.pixels_per_deg,
            blur_sigma_px=1.0,
        )
    vsd = target[0].numpy().astype(np.float32)
    start = CorticalAffine(
        origin_x=true.origin_x + 6.0,
        origin_y=true.origin_y - 5.0,
        pixels_per_unit=true.pixels_per_unit,
        rotation_deg=true.rotation_deg,
        flip_u=true.flip_u,
        flip_v=true.flip_v,
    )
    mask = np.isfinite(vsd)
    result = fit_schira_geometry(
        {"blob": StimulusMean("blob", contrast * np.float32(100.0), vsd, n_trials=1)},
        init_params=params,
        init_affine=start,
        mask=mask,
        pixels_per_deg=render.pixels_per_deg,
        cfg=GeometryFitConfig(
            optimize="camera",
            loss="pearson",
            max_origin_px=20.0,
            max_rotation_deg=10.0,
            max_scale_frac=0.1,
            blur_sigma_px=1.0,
            n_steps=60,
            lr=0.2,
            penalty_weight=0.0,
            patience=60,
            min_delta=1e-6,
            exclude_points=False,
            exclude_letters=False,
            anchor_session_only=False,
        ),
    )
    start_err = np.hypot(start.origin_x - true.origin_x, start.origin_y - true.origin_y)
    fit_err = np.hypot(
        result.affine.origin_x - true.origin_x,
        result.affine.origin_y - true.origin_y,
    )
    assert result.correlation_after >= result.correlation_before - 1e-6
    assert fit_err < 0.5 * start_err
