"""Step 2: SchiraLUT with the pre-Schira stretch, on the 100718a anchor camera.

pixel (row, col) -> u = col + i*row -> w = (u - c0)/c1 -> repo inverse_schira (alpha_repo = 1/alpha_div)
                 -> (x', y') deg -> x = x'/rho, y = y' -> canvas px (35 px/deg, fixation top-left)
Saved under the repo's standard lut_path with schira_set = '100718_v2rho' (or '100718_v2none'), so scripts 11 / 20 / 27
pick it up through a config whose only change is schira_set.
"""
import json, sys, numpy as np
from common import REPO, PPD, ANCHOR, schira_div, stim_to_deg, landmarks
from src.retinotopy.schira import SchiraParams, inverse_schira, polar_to_cartesian
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import lut_path

def build(key, set_name):
    f = json.load(open("fit_anchor.json"))[key]
    c1, c0 = (complex(*f["camera_100718a"][0]), complex(*f["camera_100718a"][1]))
    params = SchiraParams(a=f["a"], alpha=f["alpha_repo"], k=1.0, shear="double_sech", fa_combine="power",
                          sech_ecc_k=0.7609, sech_amp=0.1821)
    rr, cc = np.meshgrid(np.arange(100.0), np.arange(100.0), indexing="ij")
    w = ((cc + 1j * rr) - c0) / c1
    ecc, th = inverse_schira(w, params, min_real_z=0.0)
    xp, yp = polar_to_cartesian(ecc, th)
    x, y = xp / f["rho"], yp
    scol, srow = x * PPD, -y * PPD
    canvas = 210
    valid = np.isfinite(scol) & np.isfinite(srow) & (scol >= -0.5) & (srow >= -0.5) & (scol < canvas - .5) & (srow < canvas - .5)
    nan = lambda a: np.where(valid, a, np.nan)
    lut = SchiraLUT(spatial_size=(100, 100), canvas_size=canvas, input_size=224, pixels_per_deg=PPD, quadrant_extent_deg=6.0,
                    schira_set=set_name, anchor_session=ANCHOR, valid=valid, u=nan(w.real), v=nan(w.imag),
                    x_deg=nan(x), y_deg=nan(y), stim_row=nan(srow), stim_col=nan(scol), max_ecc_deg=None)
    out = lut_path(REPO / "experiments/schira_encoding", "gandalf", schira_set=set_name, anchor_session=ANCHOR,
                   spatial_h=100, spatial_w=100, canvas_size=canvas, input_size=224)
    lut.save(out)
    # check: forward model at the 10/07/18 landmarks vs LUT inverse at the predicted pixel
    L = landmarks()["100718"]; z = stim_to_deg(L["sr"], L["sc"])
    u = c1 * schira_div(f["rho"] * z.real + 1j * z.imag, f["a"], f["alpha_div"]) + c0
    from scipy.ndimage import map_coordinates
    back = map_coordinates(lut.stim_col, [u.imag, u.real], order=1) + 1j * map_coordinates(lut.stim_row, [u.imag, u.real], order=1)
    print(f"{set_name}: valid px {valid.sum()}, round-trip stim px error max {np.nanmax(np.abs(back - (L['sc'] + 1j*L['sr']))):.2f} "
          f"-> {out.relative_to(REPO)}")

build("rho_perday", "100718_v2rho")
build("none_perday", "100718_v2none")
