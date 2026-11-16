"""Fit Schira (a, alpha, k) + rigid registration so the mapped triangle overlaps the measured VSD response.

Sessions with the 0.4 deg triangle contour at (0.6, -0.75) deg: 100718a/b, 240718a/b/c, 270618b.
Landmark-free score: Pearson r (over the chamber) between the response map (frames 35-46 minus 5-25) and the
mapped triangle ink spread with a Gaussian of width s (the blur is a free nuisance parameter, 1-8 px).
Model: ink deg -> Schira(a, alpha, k) [repo convention P = alpha*theta] -> rigid (rotation + shift, 0.17 mm/px).
Start of the registration: the session's landmark fit (dashboard landmarks, similarity) with a 0.74, alpha 1.0.
Size check: the catalog says '0.4 radius' on 10/07 but '0.4 diameter' on 24/07 and 27/06 (same .tim file). The images
are rendered as diameter; each session is also fitted with the triangle scaled x2 about its centre ('radius' reading).
Run: python schira_dashboard/analyses/fit_triangle.py   (writes fit_triangle.csv / .png)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import schira_geometry as G
from src.paths import resolve_data_path

maps = dict(np.load(HERE.parent / "cache/gandalf_maps.npz"))
cat = pd.read_csv(HERE.parent / "cache/gandalf_catalog.csv")
LM = pd.read_csv(HERE.parent / "cache/gandalf_landmarks.csv").fillna({"pair_id": ""})
TRI = cat[cat.shape_type == "triangle_contour"].reset_index(drop=True)
gr, gc = np.mgrid[:100, :100]

from scipy.ndimage import gaussian_filter
def model_map(pts, s, ok):
    """Bilinear splat of the mapped ink points onto the camera grid, then Gaussian blur of width s (fast, smooth in p)."""
    p = pts[np.isfinite(pts)]
    if p.size == 0: return None
    x, y = p.real, p.imag; x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); fx, fy = x - x0, y - y0
    m = np.zeros((102, 102))
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx + 1, y0 + dy + 1; k = (xi >= 0) & (xi < 102) & (yi >= 0) & (yi < 102)
        np.add.at(m, (yi[k], xi[k]), wgt[k])
    return gaussian_filter(m, s, mode="constant")[1:101, 1:101][ok]

def run_session(row, scale=1.0):
    s = row.session; resp = maps[f"{s}|{row.condition}"]
    ok = np.isfinite(resp) & (np.abs(gr - 50) < 46) & (np.abs(gc - 50) < 46); y = resp[ok]
    img = np.asarray(Image.open(resolve_data_path(row.image_path, HERE.parents[1])).convert("RGB"))
    ix, iy = G.ink_to_deg(img); cx, cy = row.pos_x_deg, row.pos_y_deg
    ix, iy = cx + scale * (ix - cx), cy + scale * (iy - cy)
    L = LM[LM.session == s]
    init = G.fit_session(L.x_deg.values, L.y_deg.values, (L.cam_col + 1j * L.cam_row).values, L.pair_id.values,
                         0.74, 1.0, 1.0, "repo", "similarity")
    mirror = init.reg.mirror
    def pts(p):
        a, al, k = np.exp(p[:3]); w = G.schira_w(ix, iy, a, al, k, "repo"); w = np.conj(w) if mirror else w
        return np.exp(1j * p[3]) * w / G.MM_PER_PX + (p[4] + 1j * p[5])
    def cost(p, fix_schira=None):
        if fix_schira is not None: p = np.r_[fix_schira, p]
        if np.any(np.abs(p[:3]) > 6) or not (np.log(1) < p[6] < np.log(8)): return 1.0
        m = model_map(pts(p), np.exp(p[6]), ok)
        if m is None or m.std() == 0: return 1.0
        return -np.corrcoef(y, m)[0, 1]
    k0 = init.reg.px_per_w * G.MM_PER_PX; phi0 = np.angle(init.reg.c1); c0 = init.reg.c0
    base_s = np.log([0.74, 1.0, k0])
    p0 = [phi0, c0.real, c0.imag, np.log(3.0)]
    # (1) default Schira (a 0.74, alpha 1.0 repo), only registration + blur fitted
    rb = min((minimize(cost, np.r_[p0[:3], np.log(s0)], args=(base_s + [0, 0, np.log(f)],), method="Powell",
                       options=dict(maxiter=3000, xtol=1e-3, ftol=1e-5)) for s0 in (2, 4) for f in (0.7, 1.0, 1.4)), key=lambda r: r.fun)
    # (2) everything free, multi-start
    best = None
    for a0 in (0.2, 0.74, 2.0):
        for al0 in (0.5, 1.0):
            r = minimize(cost, np.r_[np.log([a0, al0, k0]), p0], method="Powell", options=dict(maxiter=3000, xtol=1e-3, ftol=1e-6))
            if best is None or r.fun < best.fun: best = r
    a, al, k = np.exp(best.x[:3])
    return dict(session=s, scale=scale, r_default=-rb.fun, r_best=-best.fun, a=a, alpha_repo=al, alpha_thesis=1 / al, k_mm=k,
                rot_deg=np.degrees(best.x[3]) % 360, blur_px=np.exp(best.x[6]), mirror=mirror), (resp, ok, pts(best.x), pts(np.r_[base_s, rb.x[:3]]) if False else None)

rows, figs = [], {}
for _, row in TRI.iterrows():
    for sc in (1.0, 2.0):
        res, extra = run_session(row, sc); rows.append(res); figs[(row.session, sc)] = extra
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
T = pd.DataFrame(rows); T.to_csv(HERE / "fit_triangle.csv", index=False)

fig, axs = plt.subplots(2, len(TRI), figsize=(3.3 * len(TRI), 7))
for j, (_, row) in enumerate(TRI.iterrows()):
    for i, sc in enumerate((1.0, 2.0)):
        resp, ok, p, _ = figs[(row.session, sc)]; r = T[(T.session == row.session) & (T.scale == sc)].iloc[0]
        ax = axs[i, j]; v = np.nanpercentile(np.abs(resp), 99); ax.imshow(resp, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.plot(p.real, p.imag, ".", ms=2, color="k"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{row.session} · {'0.4 diam.' if sc == 1 else '0.4 radius (x2)'}\n"
                     f"a {r.a:.2f} α {r.alpha_repo:.2f} k {r.k_mm:.1f} · r {r.r_best:.2f} (default {r.r_default:.2f})", fontsize=8.5)
fig.suptitle("Triangle: Schira a, α, k + rigid registration fitted to the measured response (Pearson r over the chamber)", x=0.01, ha="left")
plt.tight_layout(); plt.savefig(HERE / "fit_triangle.png", dpi=120); print("saved")
