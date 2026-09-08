"""Gandalf 10/07/18 anchors: VF cartesian px → Schira → table VSD pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.paths import project_root
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.params import load_schira_set
from src.retinotopy.register import (
    affine_from_two_pixel_anchors,
    affine_to_mapping,
    fit_affine_from_w_pixels,
    fit_rigid_from_w_pixels,
    vf_cartesian_px_to_w,
)
from src.retinotopy.schira import SchiraParams
from src.retinotopy.visual_field import image_px_to_cartesian_deg
from src.stimuli.render import RenderConfig

LANDMARKS = (
    project_root()
    / "experiments/schira2007/phase_0_register/landmarks_gandalf_100718.yaml"
)


def _load_landmarks() -> dict:
    with LANDMARKS.open() as f:
        return yaml.safe_load(f)


def test_gandalf_yaml_vf_cartesian_is_35_px_per_deg():
    data = _load_landmarks()
    cfg = RenderConfig(pixels_per_deg=35.0, canvas_size=210, quadrant_extent_deg=6.0)
    for pt in data["points"]:
        x_px, y_px = pt["vf_cartesian_px"]
        x_deg, y_deg = image_px_to_cartesian_deg(x_px, y_px, cfg)
        np.testing.assert_allclose(x_deg, pt["vf_deg"][0], atol=0.03)
        np.testing.assert_allclose(y_deg, pt["vf_deg"][1], atol=0.03)


def test_hbar_anchors_land_exactly_on_vsd_pixels():
    data = _load_landmarks()
    _, params, _, _ = load_schira_set(
        project_root() / "configs/schira/sets.yaml", set_name="100718"
    )
    by_id = {int(p["id"]): p for p in data["points"]}
    left, right, bottom = by_id[2], by_id[4], by_id[6]
    ppd = float(data["pixels_per_deg"])
    w_l = complex(vf_cartesian_px_to_w(*left["vf_cartesian_px"], params, pixels_per_deg=ppd))
    w_r = complex(vf_cartesian_px_to_w(*right["vf_cartesian_px"], params, pixels_per_deg=ppd))
    w_b = complex(vf_cartesian_px_to_w(*bottom["vf_cartesian_px"], params, pixels_per_deg=ppd))
    col_l, row_l = left["vsd_xy_px"]
    col_r, row_r = right["vsd_xy_px"]
    col_b, row_b = bottom["vsd_xy_px"]
    affine, meta = affine_from_two_pixel_anchors(
        w_l,
        w_r,
        float(row_l),
        float(col_l),
        float(row_r),
        float(col_r),
        params,
        check_w=w_b,
        check_row=float(row_b),
        check_col=float(col_b),
    )
    assert meta["anchor_rmsd_px"] < 1e-6
    pr, pc = affine.w_to_pixel(np.array([w_l, w_r], dtype=np.complex128), params)
    np.testing.assert_allclose(pc, [col_l, col_r], atol=1e-5)
    np.testing.assert_allclose(pr, [row_l, row_r], atol=1e-5)
    assert meta["check_err_px"] is not None
    assert Path(LANDMARKS).is_file()


def test_rigid_roundtrip_frozen_scale():
    params = SchiraParams(a=0.74, alpha=1.0, k=1.0, fa_combine="power")
    true = CorticalAffine(
        origin_x=38.0,
        origin_y=41.0,
        pixels_per_unit=31.0,
        rotation_deg=-18.0,
        flip_u=True,
        flip_v=False,
        needs_confirmation=True,
    )
    w = np.array(
        [0.3 - 0.4j, 0.8 - 0.2j, 0.5 - 0.9j, 1.0 - 0.55j],
        dtype=np.complex128,
    )
    rows, cols = true.w_to_pixel(w, params)
    fitted, meta = fit_rigid_from_w_pixels(
        w, rows, cols, params, pixels_per_unit=31.0
    )
    assert meta["mode"] == "rigid"
    assert fitted.pixels_per_unit == 31.0
    pred_r, pred_c = fitted.w_to_pixel(w, params)
    np.testing.assert_allclose(pred_r, rows, atol=1e-5)
    np.testing.assert_allclose(pred_c, cols, atol=1e-5)


def test_session_rigid_on_gandalf_landmarks():
    data = _load_landmarks()
    _, params, _, _ = load_schira_set(
        project_root() / "configs/schira/sets.yaml", set_name="100718"
    )
    ppd = float(data["pixels_per_deg"])
    ws = []
    rows = []
    cols = []
    ids = []
    for pt in data["points"]:
        w = vf_cartesian_px_to_w(
            *pt["vf_cartesian_px"], params, pixels_per_deg=ppd
        )
        ws.append(complex(w))
        c, r = pt["vsd_xy_px"]
        cols.append(float(c))
        rows.append(float(r))
        ids.append(int(pt["id"]))
    w = np.array(ws, dtype=np.complex128)
    affine, meta = fit_rigid_from_w_pixels(
        w, np.array(rows), np.array(cols), params
    )
    assert meta["n_landmarks"] == 6
    assert affine.pixels_per_unit > 0
    pred_r, pred_c = affine.w_to_pixel(w, params)
    err = np.hypot(pred_r - rows, pred_c - cols)
    by_id = dict(zip(ids, err.tolist()))
    # Rigid cannot hit H-bar ends exactly; session RMSD should still be usable.
    assert meta["rmsd_px"] < 20.0
    assert by_id[2] < 25.0
    assert by_id[4] < 25.0
    assert by_id[6] < 25.0
    # Frozen scale is not a free similarity: extra DOF would only lower RMSD.
    sim, sim_meta = fit_affine_from_w_pixels(w, np.array(rows), np.array(cols), params)
    assert sim_meta["rmsd_px"] <= meta["rmsd_px"] + 1e-6
    del sim
