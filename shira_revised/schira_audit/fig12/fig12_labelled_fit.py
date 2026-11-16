"""Best 'same grid' fit: every thesis line pixel must be explained by the SAME stimulus line (identity taken from the
legacy-pipeline fit, which explains the thesis lines to 0.31 cam px). Free: Schira a, alpha, k, camera similarity
(and optionally S1, S2 or a mirror). References: x-stretch before Schira, affine camera, legacy grid."""
import numpy as np, matplotlib
exec(open("fig12_schira_only.py").read().split("results = {}")[0])
import matplotlib.pyplot as plt
from scipy.optimize import minimize
F = np.load("fits_bug_vs_correct.npy", allow_pickle=True).item()
nl = len(s); lab = np.repeat(np.arange(14), nl)
mt = k1 * F["legacy grid (bug)"] + k0
d, i = cKDTree(np.column_stack([mt.real, mt.imag])).query(np.column_stack([data_th.real, data_th.imag]))
keep = d / SC < 1.5; DL = lab[i][keep]; DP = np.column_stack([data_th.real, data_th.imag])[keep]
lines_present = np.unique(DL)
print(f"labelled thesis pixels: {keep.sum()} on {len(lines_present)} grid lines")
def lscore(mcam):
    m = k1 * mcam + k0; tot_d, tot_m = [], []
    for j in lines_present:
        mj = m[lab == j]; mj = mj[np.isfinite(mj)]
        if len(mj) < 2: return 1e3
        dj = cKDTree(np.column_stack([mj.real, mj.imag])).query(DP[DL == j])[0]; tot_d.append(dj)
        mi, mr = np.round(mj.imag).astype(int), np.round(mj.real).astype(int)
        inb = (mi >= 0) & (mi < line.shape[0]) & (mr >= 0) & (mr < line.shape[1]); inb[inb] &= inside[mi[inb], mr[inb]]
        if inb.any():
            dd = cKDTree(DP[DL == j]).query(np.column_stack([mj.real, mj.imag])[inb])[0]; tot_m.append(np.minimum(dd, 6 * SC))
    return np.concatenate(tot_d).mean() / SC, (np.concatenate(tot_m).mean() / SC if tot_m else 6.0)
L = lambda m: (lambda sc: sc if np.isscalar(sc) else sc[0] + sc[1])(lscore(m))
def run(make, p0s, label, note):
    best = None
    for p0 in p0s:
        r = minimize(lambda p: np.nan_to_num(L(make(p)), nan=1e3), p0, method="Powell", options=dict(maxiter=4000, xtol=1e-3, ftol=1e-5))
        if best is None or r.fun < best.fun: best = r
    sc = lscore(make(best.x)); print(f"{label:58s} thesis->model {sc[0]:5.2f}  model->thesis {sc[1]:5.2f}  {note(best.x)}")
    return label, sc, make(best.x), note(best.x)
def cam(p, w): return (p[0] + 1j * p[1]) * w + (p[2] + 1j * p[3])
def init_cam(w_anch):
    c1 = (anch_cam[1] - anch_cam[0]) / (w_anch[1] - w_anch[0]); c0 = anch_cam[0] - c1 * w_anch[0]; return [c1.real, c1.imag, c0.real, c0.imag]
z = deg_correct(R, C); out = []
starts = lambda extra=[]: [[np.log(a0), np.log(al0), 0.0] + extra + init_cam(schira(deg_correct(ar, ac), a0, al0)) for a0 in (0.2, 1, 3, 10) for al0 in (0.7, 1.5)]
wide = [[np.log(a0), np.log(al0), 0.0] + init_cam(schira(deg_correct(ar, ac), a0, al0)) for a0 in (0.03, 0.1, 0.3, 1, 3, 10, 100) for al0 in (0.4, 0.8, 1.2, 2, 4)]
out.append(run(lambda p: cam(p[3:], schira(z, *np.exp(p[:3]))), wide, "Schira only: a, alpha, k + similarity camera",
               lambda p: "a=%.3g alpha_div=%.3g k=%.3g" % tuple(np.exp(p[:3]))))
out.append(run(lambda p: cam(p[3:], np.conj(schira(z, *np.exp(p[:3])))), starts(), "Schira only, mirrored camera",
               lambda p: "a=%.3g alpha_div=%.3g k=%.3g" % tuple(np.exp(p[:3]))))
out.append(run(lambda p: cam(p[5:], schira(z, *np.exp(p[:3]), np.exp(p[3]), np.exp(p[4]))), starts([np.log(0.7609), np.log(0.1821)]),
               "Schira only + S1, S2 free", lambda p: "a=%.3g alpha_div=%.3g k=%.3g S1=%.3g S2=%.3g" % tuple(np.exp(p[:5]))))
def stretch(p):
    a, al, rho = np.exp(p[:3]); return cam(p[3:], schira(rho * z.real + 1j * z.imag, a, al))
out.append(run(stretch, [[np.log(a0), np.log(al0), np.log(r0)] + init_cam(schira(r0 * deg_correct(ar, ac).real + 1j * deg_correct(ar, ac).imag, a0, al0))
                         for a0 in (0.3, 1, 3) for al0 in (1, 1.5) for r0 in (0.5, 0.8)],
               "ref: x-stretch rho before Schira + similarity", lambda p: "a=%.3g alpha_div=%.3g rho=%.3g" % tuple(np.exp(p[:3]))))
def affine(p):
    w = schira(z, *np.exp(p[:2])); return (p[2] * w.real + p[3] * w.imag + p[4]) + 1j * (p[5] * w.real + p[6] * w.imag + p[7])
def init_aff(a0, al0):
    c = init_cam(schira(deg_correct(ar, ac), a0, al0)); c1 = c[0] + 1j * c[1]
    return [np.log(a0), np.log(al0), c1.real, -c1.imag, c[2], c1.imag, c1.real, c[3]]
out.append(run(affine, [init_aff(a0, al0) for a0 in (0.3, 1, 3) for al0 in (1, 1.5)], "ref: Schira + affine camera (tilt)",
               lambda p: "a=%.3g alpha_div=%.3g" % tuple(np.exp(p[:2]))))
out.append(("legacy grid (bug) + similarity, a=0.727 alpha=1.46", lscore(F["legacy grid (bug)"]), F["legacy grid (bug)"], "a=0.727 alpha_div=1.46"))
print(f"{out[-1][0]:58s} thesis->model {out[-1][1][0]:5.2f}  model->thesis {out[-1][1][1]:5.2f}")
import pickle; pickle.dump(out, open("labelled_results.pkl", "wb"))

cols = ["#2a78d6", "#4a3aa7", "#1f9bd1", "#1baf7a", "#8a6d00", "#eb6834"]
fig, axs = plt.subplots(2, 4, figsize=(17.5, 9.6)); fig.patch.set_facecolor("white")
order = [5, 0, 1, 2, 3, 4]
ax = axs[0, 0]; ax.imshow(T.astype(np.uint8)); ax.set_title("Thesis Fig. 12B(ii)\n(white = thesis grid)", fontsize=9, loc="left")
for ax, idx in zip(list(axs.flat[1:7]), order):
    name, sc, m, note = out[idx]; mt_ = k1 * m + k0
    ax.imshow(T.astype(np.uint8))
    for j in range(14):
        seg = mt_[lab == j]; ax.plot(seg.real, seg.imag, "-", lw=1.1, color=cols[idx], alpha=.85)
    ax.set_title(f"{name}\n{note}\nthesis→model {sc[0]:.2f} · model→thesis {sc[1]:.2f} cam px", fontsize=8.3, loc="left")
for ax in axs.flat[:7]: ax.set_xlim(25, 250); ax.set_ylim(245, 25); ax.set_xticks([]); ax.set_yticks([])
ax = axs.flat[7]; ax.axis("off")
ax.text(0, 0.95, "Score = mean distance (camera px, 1 px = 170 µm)\nbetween each thesis line and the SAME stimulus\nline in the model (line identity fixed).\n\n"
        "Stimulus: 3×3° grid, lines every 0.5°, 35 px/deg.\nk is free in every Schira fit; it is absorbed\nby the camera scale (any k gives the same grid).",
        fontsize=9.5, va="top")
fig.suptitle("Can Schira alone (a, α, k) redraw thesis Fig. 12B(ii) at the correct 35 px/deg?", fontsize=12.5, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.965)); plt.savefig("fig12_same_grid_fits.png", dpi=125)
