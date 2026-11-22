"""The encoding geometry (src/retinotopy/revised.py, configs/schira/revised.yaml) is the five-step mapping of
registration_revised/fit_stretch.predict, and the encoding LUT is its exact inverse."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.paths import project_root
from src.retinotopy.revised import load_revised_geometry
from src.schira_encoding.lut import build_schira_lut
from src.stimuli.render import RenderConfig

REPO = project_root()
CFG = REPO / "configs/schira/revised.yaml"
sys.path.insert(0, str(REPO / "registration_revised"))
import fit_stretch as F  # noqa: E402

G = F.G


@pytest.fixture(autouse=True)
def _default_geometry_state():
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")
    yield
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")


def _anchors(day):
    an = yaml.safe_load(CFG.read_text())["days"][day]["anchors"]
    return [tuple(x["stim_rc"]) for x in an], [tuple(x["cam_rc"]) for x in an]


def test_config_is_the_joint_fit():
    m = yaml.safe_load(CFG.read_text())["model"]
    assert (m["a"], m["alpha_thesis"], m["r"], m["k"]) == (1.43, 1.50, 0.539, 1.0)
    assert G.JOINT_FIT == dict(a=1.43, alpha_thesis=1.50, r=0.539)       # the dashboard's "Our joint fit"
    assert (m["sech_ecc_k"], m["sech_amp"]) == (0.76, 0.1821)


@pytest.mark.parametrize("set_name, day", [("revised_100718", "100718"), ("revised_201118", "201118")])
def test_forward_equals_fit_stretch_predict(set_name, day):
    geo = load_revised_geometry(CFG, set_name)
    st_an, cam_an = _anchors(day)
    stim = st_an + [(32, 35), (15, 52), (49, 52), (25, 66), (8, 40), (20, 10)]
    cam = cam_an + [(1, 1)] * 6                                       # only the anchors' camera points matter
    u, _, _ = F.predict(stim, cam, st_an, geo.a, geo.alpha_thesis, geo.r)
    rows, cols = geo.deg_to_cam(np.array([c for _, c in stim]) / 35.0, -np.array([r for r, _ in stim]) / 35.0)
    np.testing.assert_allclose(cols + 1j * rows, u, atol=1e-9)
    np.testing.assert_allclose(cols[:2] + 1j * rows[:2], [complex(c - 1, r - 1) for r, c in cam_an], atol=1e-9)


@pytest.mark.parametrize("set_name", ["revised_100718", "revised_201118"])
def test_inverse_round_trip(set_name):
    geo = load_revised_geometry(CFG, set_name)
    x = np.linspace(0.2, 2.5, 12); y = -np.linspace(0.1, 2.0, 12)
    X, Y = np.meshgrid(x, y)
    rows, cols = geo.deg_to_cam(X, Y)
    xb, yb = geo.cam_to_deg(rows, cols)
    np.testing.assert_allclose(xb, X, atol=1e-6); np.testing.assert_allclose(yb, Y, atol=1e-6)


def test_lut_points_back_to_the_stimulus():
    """A camera pixel predicted for stimulus point s has LUT stimulus coords == s (35 px/deg, canvas corner origin)."""
    geo = load_revised_geometry(CFG, "revised_201118")
    cfg = RenderConfig(canvas_size=210, pixels_per_deg=35.0, quadrant_extent_deg=6.0)
    lut = build_schira_lut(geometry=geo, anchor_session="201118a", spatial_size=(100, 100), render_cfg=cfg)
    assert lut.schira_set == "revised_201118" and lut.valid.sum() > 500
    rr, cc = np.nonzero(lut.valid)
    rows, cols = geo.deg_to_cam(lut.x_deg[rr, cc], lut.y_deg[rr, cc])
    np.testing.assert_allclose(rows, rr, atol=1e-6); np.testing.assert_allclose(cols, cc, atol=1e-6)
    np.testing.assert_allclose(lut.stim_col[rr, cc], 35.0 * lut.x_deg[rr, cc])
    np.testing.assert_allclose(lut.stim_row[rr, cc], -35.0 * lut.y_deg[rr, cc])
