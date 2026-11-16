"""Step 0 + 1: score the repo geometry on the landmarks, then fit Schira + stretch + ONE camera on the 100718a anchor.

Landmarks: 8 (20/11/18, moved onto 100718a with the repo vessel registration) + 10 (10/07/18, already on 100718a).
Models (all with a single similarity camera u = c1*w + c0 on the anchor grid):
  none     : u = cam(Schira(z))
  rho      : u = cam(Schira(rho*x + i*y))          stretch before Schira (hypothesis 1)
  affine   : u = B*Schira(z) + t, B general 2x2     stretch after Schira (hypothesis 2)
Writes fit_anchor.json (best params, in both alpha conventions).
"""
import json, numpy as np
from scipy.optimize import minimize
from common import landmarks, schira_div, stim_to_deg, repo_predict

LM = landmarks()
days = list(LM)
SR = np.concatenate([LM[d]["sr"] for d in days]); SC = np.concatenate([LM[d]["sc"] for d in days])
CAM = np.concatenate([LM[d]["cc"] + 1j * LM[d]["cr"] for d in days])
DAY = np.concatenate([[d] * len(LM[d]["sr"]) for d in days])
Z = stim_to_deg(SR, SC)
rms = lambda e: float(np.sqrt(np.mean(np.abs(e) ** 2)))

print("== Step 0: current repo geometry (sets.yaml + CorticalAffine), error on the 100718a anchor (px) ==")
for name in ("100718", "201118"):
    r, c = repo_predict(name, SR, SC); e = (c + 1j * r) - CAM
    print(f"  set {name}: all {rms(e):5.1f}  |  10/07/18 pts {rms(e[DAY=='100718']):5.1f}  20/11/18 pts {rms(e[DAY=='201118']):5.1f}")

def camfit(w, cam, m, kind):
    if kind == "affine":
        M = np.column_stack([w.real, w.imag, np.ones(len(w))])
        cx = np.linalg.lstsq(M[m], cam[m].real, rcond=None)[0]; cy = np.linalg.lstsq(M[m], cam[m].imag, rcond=None)[0]
        return M @ cx + 1j * (M @ cy), (cx, cy)
    M = np.column_stack([w, np.ones_like(w)]); c = np.linalg.lstsq(M[m], cam[m], rcond=None)[0]
    return M @ c, c

def predict(p, m, model):
    rho = p[2] if model == "rho" else 1.0
    w = schira_div(rho * Z.real + 1j * Z.imag, p[0], p[1])
    return camfit(w, CAM, m, "affine" if model == "affine" else "sim")

def fit(m, model):
    n = 3 if model == "rho" else 2
    err = lambda p: rms((predict(p, m, model)[0] - CAM)[m])
    starts = [[a, al, r][:n] for a in (0.3, 0.8, 1.5, 3) for al in (0.7, 1.2, 2) for r in (0.5, 0.8)]
    starts = [list(s) for s in {tuple(s) for s in starts}]
    bounds = [(0.05, 6), (0.3, 6), (0.2, 1.5)][:n]
    return min((minimize(err, s, method="L-BFGS-B", bounds=bounds) for s in starts), key=lambda r: r.fun)

print("\n== Step 1: joint fit on the 100718a anchor, one camera, 18 landmarks ==")
full = np.ones(len(CAM), bool); res = {}
for model in ("none", "rho", "affine"):
    b = fit(full, model); loo = []
    for k in range(len(CAM)):
        m = full.copy(); m[k] = False
        loo.append(predict(fit(m, model).x, m, model)[0][k] - CAM[k])
    loo = np.array(loo)
    pr, cam = predict(b.x, full, model)
    res[model] = dict(a=float(b.x[0]), alpha_div=float(b.x[1]), alpha_repo=float(1 / b.x[1]),
                      rho=float(b.x[2]) if model == "rho" else 1.0, train_rmsd=b.fun, loo_rmsd=rms(loo),
                      loo_100718=rms(loo[DAY == "100718"]), loo_201118=rms(loo[DAY == "201118"]),
                      camera=[[float(np.real(x)), float(np.imag(x))] for x in np.ravel(cam)] if model != "affine"
                      else [list(map(float, cam[0])), list(map(float, cam[1]))])
    r = res[model]
    print(f"  {model:7s} a={r['a']:.2f} alpha_div={r['alpha_div']:.2f} (repo {r['alpha_repo']:.2f}) rho={r['rho']:.2f} "
          f"train {r['train_rmsd']:.2f}  LOO {r['loo_rmsd']:.2f}  (10/07 {r['loo_100718']:.2f}, 20/11 {r['loo_201118']:.2f})")
json.dump(res, open("fit_anchor.json", "w"), indent=1)

# ---- Registration-free variant: shared (a, alpha, rho), one camera PER DAY (each day on its own camera grid).
# The 20/11/18 -> 100718a vessel registrations disagree (201118a/c/d: rotation 25 / 16 / 3 deg), so the landmark
# transfer above adds several px of error. The camera used for the LUT is the 10/07/18 one (= 100718a grid).
print("\n== Step 1b: shared a, alpha (+rho), per-day camera (no cross-day registration) ==")
CAMD = np.concatenate([LM[d]["cc_day"] + 1j * LM[d]["cr_day"] for d in days])
def predict_pd(p, m, use_rho):
    rho = p[2] if use_rho else 1.0
    w = schira_div(rho * Z.real + 1j * Z.imag, p[0], p[1]); out = np.zeros_like(CAMD); cams = {}
    for d in days:
        s = DAY == d; f, c = camfit(w[s], CAMD[s], m[s], "sim"); out[s] = f; cams[d] = c
    return out, cams
def fit_pd(m, use_rho):
    n = 3 if use_rho else 2
    err = lambda p: rms((predict_pd(p, m, use_rho)[0] - CAMD)[m])
    starts = {tuple([a, al, r][:n]) for a in (0.3, 0.8, 1.5, 3) for al in (0.7, 1.2, 2) for r in (0.5, 0.8)}
    return min((minimize(err, list(s), method="L-BFGS-B", bounds=[(0.05, 6), (0.3, 6), (0.2, 1.5)][:n]) for s in starts),
               key=lambda r: r.fun)
for use_rho in (False, True):
    b = fit_pd(full, use_rho); loo = []
    for k in range(len(CAMD)):
        m = full.copy(); m[k] = False
        loo.append(predict_pd(fit_pd(m, use_rho).x, m, use_rho)[0][k] - CAMD[k])
    loo = np.array(loo); _, cams = predict_pd(b.x, full, use_rho)
    key = "rho_perday" if use_rho else "none_perday"
    res[key] = dict(a=float(b.x[0]), alpha_div=float(b.x[1]), alpha_repo=float(1 / b.x[1]),
                    rho=float(b.x[2]) if use_rho else 1.0, train_rmsd=b.fun, loo_rmsd=rms(loo),
                    loo_100718=rms(loo[DAY == "100718"]), loo_201118=rms(loo[DAY == "201118"]),
                    camera_100718a=[[float(np.real(x)), float(np.imag(x))] for x in cams["100718"]])
    r = res[key]
    print(f"  {key:12s} a={r['a']:.2f} alpha_div={r['alpha_div']:.2f} (repo {r['alpha_repo']:.2f}) rho={r['rho']:.2f} "
          f"train {r['train_rmsd']:.2f}  LOO {r['loo_rmsd']:.2f}  (10/07 {r['loo_100718']:.2f}, 20/11 {r['loo_201118']:.2f})")
json.dump(res, open("fit_anchor.json", "w"), indent=1)
