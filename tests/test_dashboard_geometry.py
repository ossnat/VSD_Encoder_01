"""Checks for schira_dashboard/schira_geometry.py.  Run: scripts/py -m pytest tests/test_dashboard_geometry.py"""
import sys
from pathlib import Path

import numpy as np, pandas as pd
import pytest

DASH = Path(__file__).resolve().parents[1] / "schira_dashboard"
sys.path.insert(0, str(DASH))
import schira_geometry as G  # noqa: E402

L = pd.read_csv(DASH / "cache/gandalf_landmarks.csv").fillna({"pair_id": ""})
J = L[(L.session == "201118a") & (L.source == "manual")]
x, y, cam = J.x_deg.values, J.y_deg.values, (J.cam_col + 1j * J.cam_row).values
@pytest.fixture(autouse=True)
def _default_geometry_state():
    """schira_geometry keeps the dashboard's r and pixel grid in module globals; start every test from the defaults."""
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")
    yield
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")


def check(name, cond, info=""):
    assert cond, f"{name} {info}"


def test_similarity_k_invariant():
    # k has no effect with a similarity registration
    r = [G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, k, "thesis", "similarity").rmse_px for k in (0.5, 1, 7.7, 100)]
    check("similarity: RMSE independent of k", np.ptp(r) < 1e-9, f"{np.round(r, 6)}")

def test_rigid_depends_on_k():
    # ... but it matters with a rigid (fixed 0.17 mm/px) registration
    r = [G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, k, "thesis", "rigid").rmse_px for k in (1, 7.7, 20)]
    check("rigid: RMSE depends on k", np.ptp(r) > 0.1, f"{np.round(r, 2)}")

def test_alpha_conventions_agree():
    # conventions agree: thesis alpha 1.5 == repo alpha 1/1.5
    a = G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, 1, "thesis").rmse_px
    b = G.fit_session(x, y, cam, J.pair_id, 0.72, 1 / 1.5, 1, "repo").rmse_px
    check("thesis/repo alpha conventions agree", abs(a - b) < 1e-9, f"{a:.4f} vs {b:.4f}")

def test_nested_fit_201118():
    # outer loop reproduces the earlier 20/11/18 Schira-only fit (train RMSE 7.9 px, landmark-fits-201118)
    o = G.optimize_schira(x, y, cam, J.pair_id, "thesis", "similarity")
    check("nested fit on Joanna's 20/11/18 points ~ 7.9 px", abs(o["rmse_px"] - 7.9) < 0.3, f"{o}")

def test_bar_end_order():
    # bar-end pairing: swapping the two ends of a bar must not change the best fit
    B = L[(L.session == "100718a")]
    xb, yb, cb, pb = B.x_deg.values, B.y_deg.values, (B.cam_col + 1j * B.cam_row).values, B.pair_id.values
    r1 = G.fit_session(xb, yb, cb, pb, 0.74, 1.0, 1, "thesis").rmse_px
    i = np.where(pb != "")[0][:2]; xs, ys = xb.copy(), yb.copy(); xs[i] = xs[i[::-1]]; ys[i] = ys[i[::-1]]
    r2 = G.fit_session(xs, ys, cb, pb, 0.74, 1.0, 1, "thesis").rmse_px
    check("bar-end order does not matter", abs(r1 - r2) < 1e-9, f"{r1:.4f} vs {r2:.4f}")
