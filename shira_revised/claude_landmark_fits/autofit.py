import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import autolm as A, landmarks as L
from scipy.optimize import minimize
names = ["point", "vbar_top", "vbar_bot", "hbar_near", "hbar_far"]
VFD = {"point": (0.6, -0.75), "vbar_top": (0.6, -0.25), "vbar_bot": (0.6, -1.25), "hbar_near": (0.1, -0.75), "hbar_far": (1.1, -0.75)}
# collect detections; assign bar ends by matching to consensus
det = {n: [] for n in names}
for s in A.SESS:
    z, res = A.extract(s)
    det["point"].append(res["point"][0])
    for k, ends in (("vbar", ("vbar_top", "vbar_bot")), ("hbar", ("hbar_near", "hbar_far"))):
        for p in res[k]: det["_" + k] = det.get("_" + k, []) + [p]
# consensus by clustering bar peaks into 2 clusters (k-means, 2 centres) + robust median
def two_clusters(P):
    P = np.array(P); c = P[[0, np.argmax(np.linalg.norm(P - P[0], axis=1))]]
    for _ in range(20):
        lab = np.argmin(np.linalg.norm(P[:, None] - c[None], axis=2), 1); c = np.array([np.median(P[lab == i], 0) for i in (0, 1)])
    return c, lab, P
cons = {}
def robust(P, tol=6):
    P = np.array(P); med = np.median(P, 0); keep = np.linalg.norm(P - med, axis=1) <= tol
    return np.median(P[keep], 0), keep.sum(), len(P)
cons["point"] = robust(det["point"])
cv, lab, P = two_clusters(det["_vbar"]); # vbar: top end (y=-0.25, nearer HM) vs bottom: decide by matching joanna-free rule: the end nearer the point response along the bar? use column: both ends known to differ in col
for i in (0, 1): cons[f"vb{i}"] = robust(P[lab == i])
ch, lab, P = two_clusters(det["_hbar"])
for i in (0, 1): cons[f"hb{i}"] = robust(P[lab == i])
for k, v in cons.items(): print(f"{k:6s} consensus (row,col) = {np.round(v[0],1)}  used {v[1]}/{v[2]}")

AUTO = {"point": cons["point"][0], "vbar_top": cons["vb0"][0], "vbar_bot": cons["vb1"][0], "hbar_near": cons["hb0"][0], "hbar_far": cons["hb1"][0]}
MANUAL = {"point": (39, 44), "vbar_top": (46, 24), "vbar_bot": (36, 70), "hbar_near": (24, 37), "hbar_far": (51, 48)}   # Joanna xlsx (row, col)
print("\nauto vs Joanna manual (row,col):")
for n in names: print(f"  {n:10s} auto {np.round(AUTO[n],1)}  manual {MANUAL[n]}  diff {np.hypot(*(np.array(AUTO[n])-MANUAL[n])):.1f}px")
def fitset(LM, use, label):
    vf = np.array([VFD[n] for n in use]); cam = np.array([LM[n][1] + 1j*LM[n][0] for n in use])
    def pred(p, m):
        rho = p[2] if len(p) > 2 else 1.0
        z = rho*vf[:, 0] + 1j*vf[:, 1]; w = L.schira(z, p[0], p[1])
        M = np.column_stack([w, np.ones_like(w)]); c = np.linalg.lstsq(M[m], cam[m], rcond=None)[0]; return M @ c
    out = {}
    for nm, k in (("no stretch", 2), ("stretch rho", 3)):
        st = [[a, al] + ([r] if k == 3 else []) for a in (0.3, 0.8, 1.5, 3) for al in (0.7, 1.2, 2) for r in (0.5, 0.8)]
        bnd = [(0.05, 5), (0.3, 6)] + ([(0.2, 1.5)] if k == 3 else [])
        def fit(m):
            e = lambda p: np.sqrt(np.mean(np.abs(pred(p, m)[m] - cam[m])**2))
            return min((minimize(e, s, method="L-BFGS-B", bounds=bnd) for s in st), key=lambda r: r.fun)
        b = fit(np.ones(len(use), bool))
        out[nm] = b; print(f"  {label:28s} {nm:12s} params={np.round(b.x,2)}  RMSD={b.fun:.2f}px")
    # profile: best RMSD as function of rho with a, alpha refit
    prof = []
    for r in np.round(np.arange(0.3, 1.31, 0.1), 2):
        st = [[a, al] for a in (0.3, 0.8, 1.5, 3) for al in (0.7, 1.2, 2)]
        e = lambda p: np.sqrt(np.mean(np.abs(pred([p[0], p[1], r], np.ones(len(use), bool)) - cam)**2))
        prof.append((r, min(minimize(e, s, method="L-BFGS-B", bounds=[(0.05,5),(0.3,6)]).fun for s in st)))
    print("   profile rho->RMSD:", " ".join(f"{r}:{v:.2f}" for r, v in prof))
for LM, lab in ((AUTO, "AUTO all 5"), (MANUAL, "MANUAL all 5")):
    fitset(LM, names, lab)
for LM, lab in ((AUTO, "AUTO without hbar_near"), (MANUAL, "MANUAL without hbar_near")):
    fitset(LM, [n for n in names if n != "hbar_near"], lab)
