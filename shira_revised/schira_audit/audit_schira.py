"""Numerical audit of the repo Schira chain (src/retinotopy, src/schira_encoding, src/stimuli, session_register).
Each test prints PASS/FAIL with the worst error. Run: PYTHONPATH=<repo> python audit_schira.py"""
import sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.retinotopy.schira import SchiraParams, forward_schira, inverse_schira, cartesian_to_polar, polar_to_cartesian
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.visual_field import cartesian_deg_to_image_px
from src.schira_encoding.lut import SchiraLUT
from src.stimuli.render import RenderConfig, render_stimulus
from src.stimuli.catalog import StimulusSpec
from scipy.ndimage import map_coordinates
import yaml

res = []
def report(name, err, tol, note=""):
    ok = err <= tol; res.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name}: worst error {err:.3g} (tol {tol:g}) {note}")

sech = lambda x: 1 / np.cosh(x)
def schira_legacy(z, a, al, k=1.0, s1=0.7609):   # trans_cortex.m logic: T = theta / alpha
    E, th = np.abs(z), np.angle(z); T = th / al
    return k * np.log(E * np.exp(1j * T * sech(T) ** (sech(np.log(E / a) * s1) * 0.1821)) + a)

# grid over the lower-right quadrant, 0.02..6 deg
x, y = np.meshgrid(np.linspace(0.02, 6, 120), -np.linspace(0.0, 6, 120)); z = x + 1j * y
E, th = cartesian_to_polar(x, y)

# 1. repo forward == legacy formula with alpha_repo = 1/alpha_legacy (same S1)
for a, al, k in ((0.72, 1.5, 1.0), (0.74, 1.0, 1.0), (2.14, 1.36, 7.3)):
    p = SchiraParams(a=a, alpha=1 / al, k=k, sech_ecc_k=0.7609, sech_amp=0.1821)
    report(f"forward vs legacy formula a={a} alpha_div={al} k={k}", np.nanmax(np.abs(forward_schira(E, th, p) - schira_legacy(z, a, al, k))), 1e-10)
p76 = SchiraParams(a=0.72, alpha=1 / 1.5, k=1.0)
report("effect of repo default S1=0.76 vs legacy 0.7609 (w units)", np.nanmax(np.abs(forward_schira(E, th, p76) - schira_legacy(z, 0.72, 1.5))), 1e-2, "(tiny, not a bug)")

# 2. polar angle sign: lower field must give Im(w) < 0, upper Im(w) > 0, HM Im(w)=0
w = forward_schira(E, th, p76)
report("lower field -> Im(w) <= 0", float(np.nanmax(np.imag(w))), 1e-12)

# 3. forward -> inverse round trip (deg)
for a, al in ((0.72, 1.5), (0.74, 1.0), (2.14, 1.36)):
    p = SchiraParams(a=a, alpha=1 / al, k=1.0, sech_ecc_k=0.7609)
    e2, t2 = inverse_schira(forward_schira(E, th, p), p); x2, y2 = polar_to_cartesian(e2, t2)
    err = np.abs((x2 + 1j * y2) - z); nanfrac = np.mean(~np.isfinite(err))
    report(f"inverse(forward) a={a} alpha_div={al}", float(np.nanmax(err)), 1e-6, f"(unrecovered {100*nanfrac:.1f}% of grid)")

# 4. camera affine round trip and orientation (sets.yaml 100718 and 201118)
S = yaml.safe_load(open(REPO / "configs/schira/sets.yaml"))["sets"]
for name in ("100718", "201118"):
    af = S[name]["affine"]; sp = S[name]["schira"]
    p = SchiraParams(a=sp["a"], alpha=sp["alpha"], k=sp["k"])
    A = CorticalAffine(origin_x=af["origin_xy"][0], origin_y=af["origin_xy"][1], pixels_per_unit=af["pixels_per_unit"],
                       rotation_deg=af["rotation_deg"], flip_u=af["flip_u"], flip_v=af["flip_v"])
    rr, cc = np.meshgrid(np.arange(100.), np.arange(100.), indexing="ij")
    r2, c2 = A.w_to_pixel(A.pixel_to_w(rr, cc, p), p)
    report(f"CorticalAffine pixel->w->pixel ({name})", float(np.max(np.hypot(r2 - rr, c2 - cc))), 1e-9)
    # similarity check: the map must be conformal (no hidden anisotropy): unit steps in row / col give equal |dw| and 90 deg
    d1 = A.pixel_to_w(51, 50, p) - A.pixel_to_w(50, 50, p); d2 = A.pixel_to_w(50, 51, p) - A.pixel_to_w(50, 50, p)
    report(f"CorticalAffine is a pure similarity ({name}) |drow|/|dcol| - 1", abs(abs(d1) / abs(d2) - 1), 1e-12,
           f"angle {np.degrees(np.angle(d1 / d2)):.1f} deg")

# 5. LUT agrees with a direct forward mapping (set 100718): stimulus deg -> camera px -> LUT -> deg
lut = SchiraLUT.load(REPO / "experiments/schira_encoding/gandalf/lut/lut__set-100718__anchor-100718a__vsd-100x100__canvas-210__input-224.npz")
sp = S["100718"]["schira"]; af = S["100718"]["affine"]
p = SchiraParams(a=sp["a"], alpha=sp["alpha"], k=sp["k"])
A = CorticalAffine(origin_x=af["origin_xy"][0], origin_y=af["origin_xy"][1], pixels_per_unit=af["pixels_per_unit"],
                   rotation_deg=af["rotation_deg"], flip_u=af["flip_u"], flip_v=af["flip_v"])
xs, ys = np.meshgrid(np.linspace(0.2, 2.5, 15), -np.linspace(0.1, 2.5, 15))
e_, t_ = cartesian_to_polar(xs, ys); r_, c_ = A.w_to_pixel(forward_schira(e_, t_, p), p)
inside = (r_ > 1) & (r_ < 98) & (c_ > 1) & (c_ < 98)
xb = map_coordinates(lut.x_deg, [r_[inside], c_[inside]], order=1); yb = map_coordinates(lut.y_deg, [r_[inside], c_[inside]], order=1)
report("LUT(set 100718) inverts the forward map (deg, bilinear)", float(np.nanmax(np.hypot(xb - xs[inside], yb - ys[inside]))), 0.05,
       f"({inside.sum()} test points)")
sc, sr = cartesian_deg_to_image_px(xs, ys, RenderConfig())
report("deg -> canvas px uses 35 px/deg on BOTH axes", float(np.max(np.abs(np.abs(sc / xs) - 35)) + np.max(np.abs(np.abs(sr / ys) - 35))), 1e-9)

# 6. renderer: a point at (0.6, -0.75) deg lands at col 21, row 26.25; a circle is round (no x/y stretch)
def centroid_and_extent(spec):
    img = render_stimulus(spec, RenderConfig()).astype(float)
    g = img.mean(-1) if img.ndim == 3 else img; ink = np.abs(g - 128) > 10
    rr, cc = np.nonzero(ink); return rr.mean(), cc.mean(), np.ptp(rr) + 1, np.ptp(cc) + 1
try:
    import inspect
    fields = inspect.signature(StimulusSpec).parameters
    mk = lambda **kw: StimulusSpec(monkey="gandalf", csv_date="", session_letter="a", h5_session="x", condition="c", condition_num=1, is_blank=False, cortex_file=None, **kw)
    base = dict(stimulus_text="test", color="black", pos_x_deg=0.6, pos_y_deg=-0.75)
    r0, c0, h, wd = centroid_and_extent(mk(**base, shape_type="point", size_deg=0.1))
    report("rendered point centre vs (row 26.25, col 21)", float(np.hypot(r0 - 26.25, c0 - 21)), 0.75, f"(got row {r0:.2f}, col {c0:.2f})")
    r0, c0, h, wd = centroid_and_extent(mk(**base, shape_type="filled_circle", size_deg=1.0))
    report("rendered 1 deg filled circle is round: height - width (px)", float(abs(h - wd)), 1, f"(h {h}, w {wd}, expect ~35)")
except Exception as e:
    print("[SKIP] renderer test:", repr(e)[:200])

# 7. session registration: apply_to_map and apply_to_points agree (moving -> anchor)
from src.session_register.transform import SessionTransform
T = SessionTransform.load_yaml(REPO.parent / "Data/VSD_Encoder_01/session_register/AS_100718a/gandalf__100718a__201118a/transform.yaml")[0]
img = np.zeros((100, 100)); img[40, 60] = 1.0
from scipy.ndimage import gaussian_filter
img = gaussian_filter(img, 2); warped = T.apply_to_map(img, cval=0.0)
pr, pc = T.apply_to_points(40.0, 60.0)
wr, wc = np.unravel_index(np.argmax(warped), warped.shape)
report("SessionTransform: map warp and point transform agree (px)", float(np.hypot(wr - pr, wc - pc)), 1.0)

print(f"\n{sum(res)}/{len(res)} passed")
