"""Reproduce Ayzenshtat et al. 2012 Fig. 2E with the repo Schira code (src/retinotopy/schira.py, copied as repo_schira.py).

Paper procedure (Methods, '2D analytical mapping'): w = k*log(E*exp(i*P*fa) + a), P = alpha*theta, double-sech fa
(S1 0.76, S2 0.1821); register with translation+rotation+scaling using 2 reference points (black stars) as anchors;
RMSD over the other 6 (black dots); grid search 1<k<20, 0.2<a<4, 0.3<alpha<1. Reported: monkey C a 0.57 k 7.7
alpha 0.52, RMSD 0.35 mm; monkey L a 0.46 k 4.87 alpha 0.51, RMSD 0.42 mm.

Reference points, read from Fig. 2E (stars/dots at crossings of the mapped bars):
  vertical bars at 0, 0.5, 1.5, 2.5 deg from the LVM (left hemifield; mirrored to +x here), horizontal bars at
  2.5 and 3.5 deg below the HM. Stars = (0.5, -2.5) and (0.5, -3.5).
Figure pixels -> camera px: the vessel image's pixel blocks are 11.08 figure px (= 1 camera px = 0.17 mm);
the 2 mm scale bar in panel F gives 63.5 figure px/mm (3% off; camera-block value used).
"""
import numpy as np, sys, itertools, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))   # repo root
try:
    from src.retinotopy.schira import SchiraParams, forward_schira, cartesian_to_polar
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "repo_src")); from repo_schira import SchiraParams, forward_schira, cartesian_to_polar

FIGPX_PER_CAM = 11.08; MM_PER_CAM = 0.17; MM_PER_FIG = MM_PER_CAM / FIGPX_PER_CAM
# Fig. 2E crop of page 6 rendered at 600 dpi (pdftoppm -f 6 -l 6 -r 600), pixels [1980:3180, 1980:3360]
E_img = np.array(Image.open(os.path.join(HERE, "ayz2012_fig2E_600dpi.png")).convert("RGB"))
# (figure x, figure y) of the 8 reference points, from blob detection; visual field (x from LVM, y)
REF = [((415.4, 530.4), (0.0, -2.5), "dot"), ((399.3, 610.1), (0.5, -2.5), "star"), ((353.9, 725.2), (1.5, -2.5), "dot"),
       ((288.2, 801.8), (2.5, -2.5), "dot"), ((750.5, 613.9), (0.0, -3.5), "dot"), ((715.5, 709.1), (0.5, -3.5), "star"),
       ((652.4, 851.3), (1.5, -3.5), "dot"), ((565.8, 949.9), (2.5, -3.5), "dot")]
FIG = np.array([complex(*r[0]) for r in REF]); VF = np.array([complex(*r[1]) for r in REF])
star = np.array([r[2] == "star" for r in REF])
# Label reading: the near-vertical red line on the LEFT ends next to the "3.5°" label and the right one next to "2.5°",
# so the left one is HM -3.5 and the right one HM -2.5. Left hemifield -> mirrored image (conj(w)).
VF = np.where(np.isclose(VF.imag, -2.5), VF.real - 3.5j, VF.real - 2.5j)
MIRROR = True

def w_of(z, a, al, k):
    E, th = cartesian_to_polar(z.real, z.imag)
    return forward_schira(E, th, SchiraParams(a=a, alpha=al, k=k, sech_ecc_k=0.76, sech_amp=0.1821))

def register(w, anchors_idx, mirror):
    w = np.conj(w) if mirror else w
    i, j = anchors_idx
    c1 = (FIG[j] - FIG[i]) / (w[j] - w[i]); c0 = FIG[i] - c1 * w[i]
    return c1, c0, w

def rmsd_mm(a, al, k, mirror, anchors_idx=None):
    anchors_idx = anchors_idx or tuple(np.where(star)[0])
    c1, c0, w = register(w_of(VF, a, al, k), anchors_idx, mirror)
    test = np.ones(len(VF), bool); test[list(anchors_idx)] = False
    return np.sqrt(np.mean(np.abs(c1 * w[test] + c0 - FIG[test]) ** 2)) * MM_PER_FIG


from scipy.optimize import minimize
def predict(p, stretch, m, anchors=False):
    a, al = np.exp(p[:2]); rho = np.exp(p[2]) if stretch else 1.0
    w = w_of(rho * VF.real + 1j * VF.imag, a, al, 1.0)
    if not np.all(np.isfinite(w)): return None, None
    w = np.conj(w)
    if anchors:
        i, j = np.where(star)[0]; c1 = (FIG[j] - FIG[i]) / (w[j] - w[i]); c0 = FIG[i] - c1 * w[i]
    else:
        M = np.column_stack([w, np.ones_like(w)]); c1, c0 = np.linalg.lstsq(M[m], FIG[m], rcond=None)[0]
    return c1 * w + c0, (c1, c0)
def fit(stretch, m):
    def err(p):
        if np.any(np.abs(p) > 7): return 99
        pr, _ = predict(p, stretch, m); return 99 if pr is None else np.sqrt(np.mean(np.abs(pr[m] - FIG[m]) ** 2)) * MM_PER_FIG
    st = [[np.log(a), np.log(al)] + ([np.log(r)] if stretch else []) for a in (0.1, 0.5, 2, 5) for al in (0.3, 0.8, 2, 5)
          for r in ((0.3, 0.6, 1.0) if stretch else (1,))]
    return min((minimize(err, s0, method="Nelder-Mead", options=dict(maxiter=4000)) for s0 in st), key=lambda r: r.fun)
rms = lambda e: float(np.sqrt(np.mean(np.abs(e) ** 2)) * MM_PER_FIG)
full = np.ones(8, bool); rows = []
# paper parameters, the paper's procedure (star anchors, RMSD on the 6 dots), and with a free similarity on all 8
for name, (a, k, al) in {"monkey C": (0.57, 7.7, 0.52), "monkey L": (0.46, 4.87, 0.51)}.items():
    p = [np.log(a), np.log(al)]
    pa, _ = predict(p, False, full, anchors=True); pf, _ = predict(p, False, full)
    rows.append((f"paper params {name} (a {a}, α {al}, k {k})", rms(pa[~star] - FIG[~star]), rms(pf - FIG), None))
def star_rmsd_k(k):
    w = np.conj(w_of(VF, 0.57, 0.52, k)); i, j = np.where(star)[0]; c1 = (FIG[j] - FIG[i]) / (w[j] - w[i]); c0 = FIG[i] - c1 * w[i]
    return rms(c1 * w[~star] + c0 - FIG[~star])
print("k sweep (paper params C, star anchors): " + ", ".join(f"k={k}: {star_rmsd_k(k):.6f} mm" for k in (1, 4.87, 7.7, 20, 1000)))
fits = {}
for stretch in (False, True):
    b = fit(stretch, full); loo = []
    for kk in range(8):
        m = full.copy(); m[kk] = False; r = fit(stretch, m); loo.append(predict(r.x, stretch, m)[0][kk] - FIG[kk])
    pa, _ = predict(b.x, stretch, full, anchors=True)
    lab = "Schira + x-stretch ρ, all free" if stretch else "Schira only, a and α free (k cancels)"
    rows.append((lab + "  [" + ", ".join(f"{n} {v:.3g}" for n, v in zip(("a", "α", "ρ"), np.exp(b.x))) + "]", rms(pa[~star] - FIG[~star]), b.fun, rms(np.array(loo))))
    fits[stretch] = b.x
print(f"\n{'model':62s} {'stars→6 dots':>12s} {'free sim, 8':>12s} {'LOO':>8s}   (mm; paper reports 0.35 ± 0.06 for C)")
for r in rows: print(f"{r[0]:62s} {r[1]:12.3f} {r[2]:12.3f} {'' if r[3] is None else f'{r[3]:8.3f}'}")

def draw(ax, p, stretch, anchors, col, title):
    _, (c1, c0) = predict(p, stretch, full, anchors=anchors); a, al = np.exp(p[:2]); rho = np.exp(p[2]) if stretch else 1.0
    m_ = lambda z: c1 * np.conj(w_of(rho * z.real + 1j * z.imag, a, al, 1.0)) + c0
    ax.imshow(E_img)
    for x in (0.0, 0.5, 1.5, 2.5, 3.5):
        l = m_(x - 1j * np.linspace(1.5, 4.6, 300)); ax.plot(l.real, l.imag, "-", color=col, lw=1.5)
    for yv in (-2.5, -3.5):
        l = m_(np.linspace(0.0, 4.0, 300) + 1j * yv); ax.plot(l.real, l.imag, "-", color=col, lw=1.5)
    pts = m_(VF); ax.plot(pts.real, pts.imag, "o", mfc="none", mec=col, ms=12, mew=2)
    ax.set_title(title, fontsize=9.5, loc="left"); ax.set_xlim(0, 1000); ax.set_ylim(1150, 250); ax.set_xticks([]); ax.set_yticks([])
fig, axs = plt.subplots(1, 3, figsize=(17, 6.6)); fig.patch.set_facecolor("white")
draw(axs[0], [np.log(0.57), np.log(0.52)], False, True, "#22d3ee",
     f"Paper params, monkey C (a 0.57, α 0.52, k 7.7)\npaper's procedure: stars = anchors, RMSD on 6 dots = {rows[0][1]:.2f} mm\n(paper reports 0.35 ± 0.06 mm)")
draw(axs[1], fits[False], False, False, "#facc15",
     f"Schira only, a {np.exp(fits[False][0]):.2f}, α {np.exp(fits[False][1]):.2f}, best similarity\nRMS on 8 pts {rows[2][2]:.2f} mm, leave-one-out {rows[2][3]:.2f} mm")
draw(axs[2], fits[True], True, False, "#4ade80",
     f"Schira + x-stretch ρ {np.exp(fits[True][2]):.2f} (a {np.exp(fits[True][0]):.2f}, α {np.exp(fits[True][1]):.2f}), best similarity\nRMS on 8 pts {rows[3][2]:.2f} mm, leave-one-out {rows[3][3]:.2f} mm")
fig.suptitle("Ayzenshtat 2012 Fig. 2E with the repo Schira code. Coloured: our mapped bars and the 8 reference points; red/black: the paper's", fontsize=11, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(os.path.join(HERE, "ayz_fig2e_repro.png"), dpi=120)
