"""How close can Schira alone (correct 35 px/deg, rigid registration at 0.17 mm/px) get to Joanna's thesis D?

Target  = the thesis D: the legacy pipeline (legacy px->deg grid, Schira a 0.72, alpha 1.5 in the theta/alpha convention,
          2-anchor similarity with the winner201118.mat anchors). This reproduces the thesis Fig. 13 outline
          (see project doc fig13-D-shape-finding). Same letter bitmap as the legacy code (EncoderData/letters_stimuli/D.bmp),
          session-A position (rows 15-49, cols 18-52 of the canvas, centre ~ (1.0, -0.9) deg).
Model   = correct 35 px/deg -> Schira(a, alpha, k) -> rigid registration (rotation + shift, fixed 0.17 mm/px), mirror allowed.
Score   = RMSE between the same ink pixels in target and model (camera px, 1 px = 0.17 mm).
Run: python schira_dashboard/analyses/fit_thesis_D.py
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import schira_geometry as G

DATA = HERE.parents[2] / "Data"
bmp = np.array(Image.open(DATA / "EncoderData/letters_stimuli/D.bmp"))
on = np.argwhere(bmp[281:316, 381:416] == 130)
R, C = on[:, 0] + 15.0, on[:, 1] + 18.0                     # 1-based canvas pixels (legacy session A)
X0, X1, Y0, Y1 = 0.0071428571, 6.0642857143, 0.0142857143, 6.6142857143
def deg_legacy(r, c): return (X0 + (c - 1) * (X1 - X0) / 511), -(Y0 + (r - 1) * (Y1 - Y0) / 300)
def deg_correct(r, c): return c / G.PPD, -r / G.PPD

# ---- target: thesis D (legacy pipeline), camera complex col + i*row, 0-based
anch_r, anch_c = np.array([14.0, 14.0]), np.array([18.0, 52.0])   # round(|(0.5,-0.4),(1.5,-0.4)| * 35)
anch_cam = np.array([25 + 36j, 48 + 55j]) - (1 + 1j)
wa = G.schira_w(*deg_legacy(anch_r, anch_c), 0.72, 1.5, 1.0, "thesis")
c1 = (anch_cam[1] - anch_cam[0]) / (wa[1] - wa[0]); c0 = anch_cam[0] - c1 * wa[0]
target = c1 * G.schira_w(*deg_legacy(R, C), 0.72, 1.5, 1.0, "thesis") + c0
xs, ys = deg_correct(R, C)

def fit(p, mode="rigid"):
    a, al, k = np.exp(p)
    w = G.schira_w(xs, ys, a, al, k, "repo")
    if not np.all(np.isfinite(w)): return None, 1e6
    best = None
    for mr in (False, True):
        reg = G.fit_registration(w, target, mode, mr); e = np.sqrt(np.mean(np.abs(reg.apply(w) - target) ** 2))
        if best is None or e < best[1]: best = (reg, e)
    return best

def outer(mode):
    starts = [np.log([a0, al0, k0]) for a0 in (0.05, 0.2, 0.7, 2, 6) for al0 in (0.3, 0.7, 1.0, 1.5, 3) for k0 in (1, 5, 20)]
    cost = lambda p: fit(p, mode)[1] if np.all(np.abs(p) < 6) else 1e6
    return min((minimize(cost, s, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-6)) for s in starts),
               key=lambda r: r.fun)

from scipy.optimize import minimize_scalar
kb = np.exp(minimize_scalar(lambda lk: fit(np.log([0.72, 1 / 1.5, 1.0]) + [0, 0, lk])[1], bounds=(-3, 5), method="bounded").x)
base = fit(np.log([0.72, 1 / 1.5, kb]))      # thesis params at the correct 35 px/deg, rigid, best k
ref_scale = abs(c1)
print(f"target (thesis D) size: {np.ptp(target.real):.1f} x {np.ptp(target.imag):.1f} camera px; legacy registration scale {ref_scale:.1f} px/w")
print(f"thesis params at 35 px/deg (a 0.72, alpha_div 1.5), rigid, best k {kb:.2f} mm:  RMSE {base[1]:.2f} px")
r = outer("rigid"); a, al, k = np.exp(r.x); reg, e = fit(r.x)
print(f"BEST rigid: a = {a:.3f} deg, alpha = {al:.3f} (repo, P = alpha*theta) = {1/al:.3f} (thesis, P = theta/alpha), "
      f"k = {k:.2f} mm, RMSE {e:.2f} px ({e*G.MM_PER_PX:.2f} mm), rotation {reg.rotation_deg:.0f} deg, mirror {reg.mirror}")
rs = outer("similarity"); print(f"(check) best similarity: a {np.exp(rs.x[0]):.3f}, alpha {np.exp(rs.x[1]):.3f}, RMSE {rs.fun:.2f} px  -> same shape freedom as rigid")
# reference: what the legacy grid needs = an x-stretch
def fit_rho(p):
    a_, al_, k_, rho = np.exp(p)
    w = G.schira_w(rho * xs, ys, a_, al_, k_, "repo")
    if not np.all(np.isfinite(w)): return 1e6
    return min(np.sqrt(np.mean(np.abs(G.fit_registration(w, target, "rigid", m).apply(w) - target) ** 2)) for m in (False, True))
rr = min((minimize(fit_rho, np.log([a0, al0, 8, rho0]), method="Nelder-Mead", options=dict(maxiter=4000)) for a0 in (0.2, 1) for al0 in (0.7, 1.0) for rho0 in (0.5, 0.8)), key=lambda q: q.fun)
print(f"(reference) + x-stretch rho: a {np.exp(rr.x[0]):.3f}, alpha {np.exp(rr.x[1]):.3f}, k {np.exp(rr.x[2]):.2f}, rho {np.exp(rr.x[3]):.3f}, RMSE {rr.fun:.3f} px")

# ---- figure
model = reg.apply(G.schira_w(xs, ys, a, al, k, "repo")); base_pts = base[0].apply(G.schira_w(xs, ys, 0.72, 1 / 1.5, kb, "repo"))
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, pts, col, ttl in ((axs[0], base_pts, "#9a9993", f"thesis params (a 0.72, α 1.5 thesis) at 35 px/deg, rigid, k {kb:.1f} mm\nRMSE {base[1]:.2f} px"),
                          (axs[1], model, "#2a78d6", f"best Schira only, rigid: a {a:.2f}, α {al:.2f} (thesis {1/al:.2f}), k {k:.1f} mm\nRMSE {e:.2f} px ({e*G.MM_PER_PX:.2f} mm)")):
    ax.plot(target.real, target.imag, "s", ms=3, color="#eb6834", alpha=.6, label="thesis D (legacy pipeline)")
    ax.plot(pts.real, pts.imag, ".", ms=3, color=col, label="model")
    ax.set_title(ttl, fontsize=10, loc="left"); ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(fontsize=8)
ax = axs[2]
grid = np.array([[fit(np.log([a_, al, k]))[1] for a_ in np.geomspace(0.02, 20, 40)] for al in np.geomspace(0.1, 10, 40)])
cs = ax.contourf(np.geomspace(0.02, 20, 40), np.geomspace(0.1, 10, 40), np.minimum(grid, 15), 30, cmap="viridis_r")
ax.set_xscale("log"); ax.set_yscale("log"); plt.colorbar(cs, ax=ax, label="RMSE (camera px), k at its best")
ax.plot(a, al, "o", mfc="white", mec="k"); ax.set_xlabel("a (deg)"); ax.set_ylabel("α (repo, P = α·θ)")
ax.set_title("RMSE landscape over a, α (k fixed at its optimum)", fontsize=10, loc="left")
fig.suptitle("Fitting Schira alone (35 px/deg, rigid registration) to Joanna's thesis D", fontsize=12, x=0.01, ha="left")
plt.tight_layout(); out = HERE / "fit_thesis_D.png"; plt.savefig(out, dpi=120); print("saved", out.name)
