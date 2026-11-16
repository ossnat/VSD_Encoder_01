"""Shared pieces for geometry_v2: landmarks on the 100718a anchor camera, forward model, repo geometry.

Conventions
-----------
* Camera points are 0-based (row, col) on the 100x100 repo grid (legacy 1-based values minus 1).
* Visual field in degrees: x = col/35, y = -row/35 (fixation at the top-left canvas corner).
* Schira here uses the legacy *divide* convention P = theta/alpha_div. The repo uses P = alpha_repo*theta,
  so alpha_repo = 1/alpha_div (k = 1, double-sech, power combine, S1 0.7609 vs 0.76 is the only difference).
"""
from pathlib import Path
import numpy as np, scipy.io as sio, yaml, sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
LEG = REPO.parent / "legacy_code/Retinotopic model Joanna/Gandalf"
REG = REPO.parent / "Data/VSD_Encoder_01/session_register/AS_100718a"
PPD = 35.0
ANCHOR = "100718a"

sech = lambda x: 1 / np.cosh(x)
def schira_div(z, a, al):
    E, th = np.abs(z), np.angle(z); T = th / al
    return np.log(E * np.exp(1j * T * sech(T) ** (sech(np.log(E / a) * 0.7609) * 0.1821)) + a)

def stim_to_deg(r, c):
    return np.asarray(c, float) / PPD - 1j * np.asarray(r, float) / PPD

def load_transform(session):
    from src.session_register.transform import SessionTransform
    if session == ANCHOR:
        return None
    return SessionTransform.load_yaml(REG / f"gandalf__{ANCHOR}__{session}" / "transform.yaml")[0]

def landmarks(day201118_session="201118a"):
    """Return dict day -> (stim_row, stim_col, cam_row, cam_col) with camera already on the 100718a anchor."""
    out = {}
    # 20/11/18 letters, 8 points (Joanna, opt_points_8_5_22.mat): [stim r, stim c, 0, 0, cam r, cam c], 1-based
    P = sio.loadmat(LEG / "VSDI points/201118/opt_points_8_5_22.mat")["opt_points"].astype(float)
    cr, cc = P[:, 4] - 1, P[:, 5] - 1
    T = load_transform(day201118_session)
    ar, ac = T.apply_to_points(cr, cc)          # moving (201118) -> fixed (100718a)
    out["201118"] = dict(sr=P[:, 0], sc=P[:, 1], cr=ar, cc=ac, cr_day=cr, cc_day=cc)
    # 10/07/18 bars, 10 points (opt-points-10-07-18.xlsx): ((stim col,row),(cam col,row)), 1-based
    pts = [((21,26),(44,39)), ((14,14),(21,36)), ((14,38),(60,30)), ((35,26),(47,45)), ((21,14),(25,41)),
           ((21,37),(58,36)), ((21,9),(24,46)), ((21,43),(70,36)), ((4,26),(37,24)), ((38,26),(48,51))]
    sr = np.array([s[1] for s, _ in pts], float); sc = np.array([s[0] for s, _ in pts], float)
    cr = np.array([c[1] for _, c in pts], float) - 1; cc = np.array([c[0] for _, c in pts], float) - 1
    out["100718"] = dict(sr=sr, sc=sc, cr=cr, cc=cc, cr_day=cr, cc_day=cc)
    return out

def repo_set(name):
    from src.retinotopy.schira import SchiraParams
    from src.retinotopy.affine import CorticalAffine
    s = yaml.safe_load(open(REPO / "configs/schira/sets.yaml"))["sets"][name]
    sp = s["schira"]; af = s["affine"]
    params = SchiraParams(a=sp["a"], alpha=sp["alpha"], k=sp["k"], shear=sp["shear"], fa_combine=sp["fa_combine"],
                          sech_ecc_k=sp["sech_ecc_k"], sech_amp=sp["sech_amp"])
    aff = CorticalAffine(origin_x=af["origin_xy"][0], origin_y=af["origin_xy"][1], pixels_per_unit=af["pixels_per_unit"],
                         rotation_deg=af["rotation_deg"], flip_u=af["flip_u"], flip_v=af.get("flip_v", False))
    return params, aff

def repo_predict(name, sr, sc):
    """Repo forward: stimulus px -> deg -> forward_schira -> CorticalAffine.w_to_pixel. Returns (row, col)."""
    from src.retinotopy.schira import forward_schira
    params, aff = repo_set(name)
    z = stim_to_deg(sr, sc)
    w = forward_schira(np.abs(z), np.angle(z), params)
    return aff.w_to_pixel(w, params)
