"""Thesis Fig. 12B(ii): how close can Schira ALONE (a, alpha, k) get at the correct 35 px/deg?

Target  : the white grid lines in the thesis panel itself (page 30 of the thesis, 150 dpi render), moved to camera
          px with the two marker dots (scale 2.78, rotation -1.4 deg; see fig12_overlay.py).
Stimulus: 3 x 3 deg grid, lines every 0.5 deg (create_cart_stim.m positions 1,17,35,52,70,87,105 px), fixation at (1,1).
Models  : w = k*log(E*exp(i*P*fa) + a), P = theta/alpha (legacy convention), fa = double sech (S1 0.7609, S2 0.1821)
          camera = (A) the thesis 2-anchor similarity (winner201118.mat anchors), or (B) best similarity (free),
          (C) as B but S1, S2 also free. References: legacy grid (thesis pipeline), stretch rho, affine camera.
Score   : chamfer distance in camera px. data->model = mean distance from each thesis line pixel to the nearest
          model line; model->data = mean distance from model line points (inside the chamber) to the nearest thesis
          line pixel, capped at 6 px (thesis lines are dashed at this resolution).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

PAGE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "thesis_page30.png")
PPD = 35; X0, X1, Y0, Y1 = 0.0071428571, 6.0642857143, 0.0142857143, 6.6142857143
sech = lambda x: 1 / np.cosh(np.clip(x, -50, 50))
def schira(z, a, al, k=1.0, s1=0.7609, s2=0.1821):
    E, th = np.abs(z), np.angle(z); T = th / al
    return k * np.log(E * np.exp(1j * T * sech(T) ** (sech(np.log(E / a) * s1) * s2)) + a)
deg_correct = lambda r, c: c / PPD - 1j * r / PPD
deg_legacy = lambda r, c: (X0 + (c - 1) * (X1 - X0) / 511) - 1j * (Y0 + (r - 1) * (Y1 - Y0) / 300)

# ---- target: thesis line pixels -> camera px (1-based, complex col + i*row)
T = np.array(Image.open(PAGE).convert("RGB").crop((520, 910, 800, 1180))).astype(float)
g, sat = T.mean(-1), T.max(-1) - T.min(-1)
yy, xx = np.mgrid[:T.shape[0], :T.shape[1]]
inside = np.hypot(xx - 137.5, yy - 136.0) < 106
line = (g >= 225) & (sat < 25) & inside
dots_bv = np.array([34 + 43j, 59 + 53j]) + (1 + 1j); dots_th = np.array([104.4 + 142.4j, 174.5 + 168.5j])
k1 = (dots_th[1] - dots_th[0]) / (dots_bv[1] - dots_bv[0]); k0 = dots_th[0] - k1 * dots_bv[0]
SC = abs(k1)                                       # thesis px per camera px
ty, tx = np.nonzero(line); data_th = tx + 1j * ty
data_cam = (data_th - k0) / k1
tree_data = cKDTree(np.column_stack([data_th.real, data_th.imag]))
dist_to_line = distance_transform_edt(~line)       # thesis px
print(f"thesis line pixels: {line.sum()}, thesis px per camera px {SC:.2f}")

# ---- stimulus: dense samples along the 14 grid lines (stimulus px, 1-based)
P = np.array([1, 17, 35, 52, 70, 87, 105]); s = np.arange(1, 105.01, 0.25)
R = np.concatenate([np.full_like(s, p) for p in P] + [s for p in P]).astype(float)
C = np.concatenate([s for p in P] + [np.full_like(s, p) for p in P]).astype(float)
anch_deg = np.array([0.5 - 0.4j, 1.5 - 0.4j]); anch_cam = np.array([25 + 36j, 48 + 55j])
ar, ac = np.round(np.abs(anch_deg.imag) * PPD), np.round(anch_deg.real * PPD)

def score(model_cam):
    m = k1 * model_cam + k0
    ok = np.isfinite(m)
    m = m[ok]
    d_dm = tree_data.query(np.column_stack([m.real, m.imag]))[0] if False else None
    tm = cKDTree(np.column_stack([m.real, m.imag]))
    d_data = tm.query(np.column_stack([data_th.real, data_th.imag]))[0]           # data -> model
    mi = np.round(m.imag).astype(int); mr = np.round(m.real).astype(int)
    inb = (mi >= 0) & (mi < line.shape[0]) & (mr >= 0) & (mr < line.shape[1])
    inb[inb] &= inside[mi[inb], mr[inb]]
    d_model = np.minimum(dist_to_line[mi[inb], mr[inb]], 6 * SC) if inb.any() else np.array([6 * SC])
    return d_data.mean() / SC, d_model.mean() / SC

def total(sc): return sc[0] + sc[1]
def anchored(deg_fn, a, al, k=1.0, s1=0.7609, s2=0.1821):
    wa = schira(deg_fn(ar, ac), a, al, k, s1, s2); c1 = (anch_cam[1] - anch_cam[0]) / (wa[1] - wa[0])
    return c1 * schira(deg_fn(R, C), a, al, k, s1, s2) + anch_cam[0] - c1 * wa[0]

results = {}
ref = anchored(deg_legacy, 0.72, 1.5); results["legacy grid, thesis a=0.72 alpha=1.5 (= how Fig. 12 was made)"] = (score(ref), ref, "")
m = anchored(deg_correct, 0.72, 1.5); results["35 ppd, thesis params a=0.72 alpha=1.5"] = (score(m), m, "")

# ---- k: with the thesis 2-anchor similarity, k cancels exactly
ks = [0.1, 1, 10, 100, 1000]
dk = [np.nanmax(np.abs(anchored(deg_correct, 0.72, 1.5, k) - anchored(deg_correct, 0.72, 1.5, 1.0))) for k in ks]
print("max change of the mapped grid (camera px) when k = " + ", ".join(f"{k:g}: {d:.1e}" for k, d in zip(ks, dk)))

# ---- (A) Schira only, thesis 2-anchor camera, free a, alpha, k
def fitA():
    best = None
    for la in np.linspace(np.log(0.01), np.log(100), 7):
        for lal in np.linspace(np.log(0.1), np.log(20), 7):
            for lk in (np.log(0.1), 0.0, np.log(30)):
                f = lambda p: np.nan_to_num(total(score(anchored(deg_correct, *np.exp(p)))), nan=1e3)
                r = minimize(f, [la, lal, lk], method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-3, fatol=1e-3))
                if best is None or r.fun < best.fun: best = r
    return best
rA = fitA(); a, al, k = np.exp(rA.x); m = anchored(deg_correct, a, al, k)
results[f"(A) 35 ppd, Schira a,alpha,k free, thesis 2-anchor camera"] = (score(m), m, f"a={a:.3g} alpha={al:.3g} k={k:.3g}")

# ---- (B) free similarity camera (start from A), and (C) + S1, S2 free; optional mirror
def sim_model(p, deg_fn=deg_correct, extra=None, mirror=False):
    a, al, k = np.exp(p[:3]); c1 = p[3] + 1j * p[4]; c0 = p[5] + 1j * p[6]
    s1, s2 = (np.exp(p[7]), np.exp(p[8])) if extra else (0.7609, 0.1821)
    w = schira(deg_fn(R, C), a, al, k, s1, s2); w = np.conj(w) if mirror else w
    return c1 * w + c0
def init_from(a, al, k, mirror=False):
    wa = schira(deg_correct(ar, ac), a, al, k); wa = np.conj(wa) if mirror else wa
    c1 = (anch_cam[1] - anch_cam[0]) / (wa[1] - wa[0]); c0 = anch_cam[0] - c1 * wa[0]
    return [np.log(a), np.log(al), np.log(k), c1.real, c1.imag, c0.real, c0.imag]
def fitB(extra=False, mirror=False):
    best = None
    for a0 in (0.05, 0.3, 1, 3, 10):
        for al0 in (0.5, 1, 1.5, 3):
            p0 = init_from(a0, al0, 1.0, mirror) + ([np.log(0.7609), np.log(0.1821)] if extra else [])
            f = lambda p: np.nan_to_num(total(score(sim_model(p, extra=extra, mirror=mirror))), nan=1e3)
            r = minimize(f, p0, method="Powell", options=dict(maxiter=6000, xtol=1e-3, ftol=1e-4))
            if best is None or r.fun < best.fun: best = r
    return best
for label, kw in (("(B) 35 ppd, Schira a,alpha,k free, best similarity camera", {}),
                  ("(B') same, mirror allowed", dict(mirror=True)),
                  ("(C) 35 ppd, Schira a,alpha,k AND S1,S2 free, best similarity", dict(extra=True))):
    r = fitB(**kw); m = sim_model(r.x, **kw); a, al, k = np.exp(r.x[:3])
    note = f"a={a:.3g} alpha={al:.3g} k={k:.3g} (k*scale={abs(r.x[3]+1j*r.x[4])*k:.3g})" + (f" S1={np.exp(r.x[7]):.3g} S2={np.exp(r.x[8]):.3g}" if kw.get("extra") else "")
    results[label] = (score(m), m, note)

# ---- references that break Schira-only: stretch before Schira, affine camera
def stretch_model(p):
    a, al, rho = np.exp(p[:3]); c1 = p[3] + 1j * p[4]; c0 = p[5] + 1j * p[6]
    z = deg_correct(R, C); return c1 * schira(rho * z.real + 1j * z.imag, a, al) + c0
best = None
for a0 in (0.3, 1, 3):
    for rho0 in (0.5, 0.8):
        p0 = init_from(a0, 1.5, 1.0); p0[2] = np.log(rho0)
        r = minimize(lambda p: np.nan_to_num(total(score(stretch_model(p))), nan=1e3), p0, method="Powell", options=dict(maxiter=6000))
        if best is None or r.fun < best.fun: best = r
m = stretch_model(best.x); a, al, rho = np.exp(best.x[:3])
results["(ref) 35 ppd + x-stretch rho before Schira, best similarity"] = (score(m), m, f"a={a:.3g} alpha={al:.3g} rho={rho:.3g}")

print(f"\n{'model':66s} {'data->model':>11s} {'model->data':>11s}  params")
for kname, (sc, _, note) in results.items(): print(f"{kname:66s} {sc[0]:11.2f} {sc[1]:11.2f}  {note}")
np.save("results.npy", {k: (v[0], v[1], v[2]) for k, v in results.items()}, allow_pickle=True)

# ---- figure: each model's grid over the thesis panel
keys = list(results)
fig, axs = plt.subplots(2, 4, figsize=(17, 9.2)); fig.patch.set_facecolor("white")
cols = ["#eb6834", "#9a9993", "#2a78d6", "#2a78d6", "#4a3aa7", "#1baf7a", "#1baf7a"]
for ax, kname, col in zip(axs.flat, keys, cols):
    sc, mcam, note = results[kname]; mt = k1 * mcam + k0
    ax.imshow(T.astype(np.uint8)); ax.plot(mt.real, mt.imag, ".", ms=1.1, color=col, alpha=.8)
    ax.set_xlim(25, 250); ax.set_ylim(245, 25); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{kname}\n{note}\nthesis→model {sc[0]:.2f} px · model→thesis {sc[1]:.2f} px", fontsize=8.5, loc="left")
ax = axs.flat[len(keys)]; ax.imshow(T.astype(np.uint8)); ax.contour(line, [.5], colors="#e34948", linewidths=.4)
ax.set_xlim(25, 250); ax.set_ylim(245, 25); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("thesis Fig. 12B(ii) with the extracted\nline pixels used as target (red)", fontsize=8.5, loc="left")
for a_ in axs.flat[len(keys) + 1:]: a_.axis("off")
fig.suptitle("Fig. 12B(ii) grid: thesis (white) vs model (dots). Distances in camera px (1 px = 170 µm)", fontsize=12, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig("fig12_schira_only_fits.png", dpi=125)
