"""Same target and score as fig12_schira_only.py. Fair comparison: BOTH pixel->degree conversions get free a, alpha
and a free similarity camera. Then: line-identity check (does each thesis line pixel go to the same grid line in
both fits?) and where each fit puts the thesis anchor stimuli relative to the thesis anchor camera points."""
import numpy as np
exec(open("fig12_schira_only.py").read().split("results = {}")[0])
from scipy.optimize import minimize
def model(p, deg_fn):
    a, al = np.exp(p[:2]); c1 = p[2] + 1j * p[3]; c0 = p[4] + 1j * p[5]
    return c1 * schira(deg_fn(R, C), a, al) + c0, c1, c0, a, al
def init(a, al, deg_fn):
    wa = schira(deg_fn(ar, ac), a, al); c1 = (anch_cam[1] - anch_cam[0]) / (wa[1] - wa[0]); c0 = anch_cam[0] - c1 * wa[0]
    return [np.log(a), np.log(al), c1.real, c1.imag, c0.real, c0.imag]
fits = {}
for name, fn in (("legacy grid (bug)", deg_legacy), ("35 ppd (correct)", deg_correct)):
    best = None
    for a0 in (0.1, 0.3, 0.72, 1.5, 3, 6):
        for al0 in (0.7, 1, 1.5, 2.5):
            r = minimize(lambda p: np.nan_to_num(total(score(model(p, fn)[0])), nan=1e3), init(a0, al0, fn), method="Powell",
                         options=dict(maxiter=8000, xtol=1e-4, ftol=1e-5))
            if best is None or r.fun < best.fun: best = r
    m, c1, c0, a, al = model(best.x, fn); sc = score(m)
    anch_pred = c1 * schira(fn(ar, ac), a, al) + c0
    fits[name] = (m, fn)
    print(f"{name:20s} a={a:.3g} alpha_div={al:.3g}  thesis->model {sc[0]:.2f}  model->thesis {sc[1]:.2f}  "
          f"scale {abs(c1):.1f} cam px/w-unit  anchors off by {np.abs(anch_pred - anch_cam).round(1)} cam px")
# line identity: label each thesis pixel by the nearest grid line (14 lines) in each fit
nl = len(s); lab = np.repeat(np.arange(14), nl)
def labels(mcam):
    mt = k1 * mcam + k0; ok = np.isfinite(mt)
    t = cKDTree(np.column_stack([mt[ok].real, mt[ok].imag])); d, i = t.query(np.column_stack([data_th.real, data_th.imag]))
    return lab[ok][i], d / SC
lb, db = labels(fits["legacy grid (bug)"][0]); lc, dc = labels(fits["35 ppd (correct)"][0])
near = (db < 1.5) & (dc < 1.5)
print(f"thesis pixels within 1.5 cam px of both fits: {near.sum()}/{len(near)}; same grid-line identity: {np.mean(lb[near] == lc[near])*100:.1f}%")
names = [f"row {p}" for p in P] + [f"col {p}" for p in P]
for j in range(14):
    nb, nc = np.sum(lb[db < 1.5] == j), np.sum(lc[dc < 1.5] == j)
    print(f"  {names[j]:8s} thesis pixels on it: bug fit {nb:4d}  correct fit {nc:4d}")
np.save("fits_bug_vs_correct.npy", {k: v[0] for k, v in fits.items()}, allow_pickle=True)
