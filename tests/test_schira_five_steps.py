"""The Schira mapping used everywhere, written out step by step (registration_revised/README.md):

1. pixels -> degrees   x = col/35, y = -row/35            (fixation at the canvas corner)
2. horizontal stretch  (x', y') = (r*x, y)
3. polar               E = |x' + i y'|, theta = atan2(y', x')
4. Schira 2007         P = theta/alpha, f_a = sech(P)^(S2*sech(S1*log(E/a))), S1 0.76, S2 0.1821,
                       z = E*exp(i*P*f_a), w = k*log(z + a), k = 1
5. 2-anchor camera     c = c_A + (c_B - c_A)/(w_B - w_A) * (w - w_A)

Checks that registration_revised/fit_stretch.predict (the code behind the dashboard's "Our joint fit") computes exactly
this, and that step 4 equals Joanna's MATLAB port (src/legacy_joanna/legacy_schira.py, retinotopic_model_joanna.m:
divide-alpha, S1 0.76; it reads |x|, |y|, which mirrors the lower field: w_legacy = conj(w)).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "registration_revised"))
import fit_stretch as F  # noqa: E402
G = F.G
from src.legacy_joanna.legacy_schira import LegacySchiraParams, forward_legacy_schira  # noqa: E402

A, ALPHA, R = 1.43, 1.50, 0.539          # joint fit
S1, S2 = 0.76, 0.1821

# synthetic stimulus points (row, col), 1-based like Joanna's tables, lower-right field; camera points arbitrary
STIM = [(49, 18), (15, 18), (32, 35), (15, 52), (49, 52), (25, 66), (8, 40)]
CAM = [(31, 78), (36, 25), (40, 50), (51, 42), (44, 81), (54, 63), (60, 30)]
ANCHORS = [(49, 18), (15, 18)]


@pytest.fixture(autouse=True)
def _default_geometry_state():
    """schira_geometry keeps the dashboard's r and pixel grid in module globals; start every test from the defaults."""
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")
    yield
    G.set_stretch(1.0); G.set_pixel_grid("35ppd")


def sech(x):
    return 1.0 / np.cosh(x)


def five_steps(stim, cam, anchors, a, alpha, r, k=1.0):
    rows = np.array([p[0] for p in stim], float); cols = np.array([p[1] for p in stim], float)
    x, y = cols / 35.0, -rows / 35.0                                   # 1
    xs, ys = r * x, y                                                  # 2
    E, th = np.hypot(xs, ys), np.arctan2(ys, xs)                       # 3
    P = th / alpha                                                     # 4
    fa = sech(P) ** (S2 * sech(S1 * np.log(E / a)))
    w = k * np.log(E * np.exp(1j * P * fa) + a)
    c = np.array([(cc - 1) + 1j * (rr - 1) for rr, cc in cam])         # 5 (0-based camera, col + i row)
    iA, iB = stim.index(anchors[0]), stim.index(anchors[1])
    return c[iA] + (c[iB] - c[iA]) / (w[iB] - w[iA]) * (w - w[iA]), w, (xs, ys)


@pytest.mark.parametrize("a, alpha, r", [(A, ALPHA, R), (0.72, 1.5, 1.0), (0.9, 1.1, 0.4)])
def test_predict_is_the_five_steps(a, alpha, r):
    assert F.MIRROR is False
    u, _, _ = F.predict(STIM, CAM, ANCHORS, a, alpha, r)
    ref, _, _ = five_steps(STIM, CAM, ANCHORS, a, alpha, r)
    np.testing.assert_allclose(u, ref, rtol=0, atol=1e-9)


def test_anchors_land_exactly():
    u, c, ia = F.predict(STIM, CAM, ANCHORS, A, ALPHA, R)
    np.testing.assert_allclose(u[ia], c[ia], atol=1e-9)


def test_step4_matches_joanna_matlab_port():
    _, w, (xs, ys) = five_steps(STIM, CAM, ANCHORS, A, ALPHA, R)
    leg = forward_legacy_schira(xs, ys, LegacySchiraParams(a=A, alpha=ALPHA, alpha_mode="divide", ecc_width=S1,
                                                           iso_polar_grad=S2, vf_abs=True))
    np.testing.assert_allclose(leg, np.conj(w), rtol=0, atol=1e-12)


def test_joanna_201118_points_joint_fit():
    """On Joanna's 8 letter points the joint-fit model gives ~1.9 px mean non-anchor distance (README)."""
    if not F.PTS_201118.joinpath("opt_points_8_5_22.mat").is_file():
        pytest.skip("legacy_code VSDI points not on this machine")
    s, c, an = F.set_201118()
    u, _, _ = F.predict(s, c, an, A, ALPHA, R)
    ref, _, _ = five_steps(s, c, an, A, ALPHA, R)
    np.testing.assert_allclose(u, ref, atol=1e-9)
    d, keep = F.score(s, c, an, A, ALPHA, R)
    assert 1.5 < d[keep].mean() < 2.3
