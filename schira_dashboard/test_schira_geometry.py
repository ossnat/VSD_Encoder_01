"""Checks for schira_geometry.py.  Run: python schira_dashboard/test_schira_geometry.py"""
import numpy as np, pandas as pd
from pathlib import Path
import schira_geometry as G

L = pd.read_csv(Path(__file__).parent / "cache/gandalf_landmarks.csv").fillna({"pair_id": ""})
J = L[(L.session == "201118a") & (L.source == "manual")]
x, y, cam = J.x_deg.values, J.y_deg.values, (J.cam_col + 1j * J.cam_row).values
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

# 1. k has no effect with a similarity registration
r = [G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, k, "thesis", "similarity").rmse_px for k in (0.5, 1, 7.7, 100)]
check("similarity: RMSE independent of k", np.ptp(r) < 1e-9, f"{np.round(r, 6)}")
# 2. ... but it matters with a rigid (fixed 0.17 mm/px) registration
r = [G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, k, "thesis", "rigid").rmse_px for k in (1, 7.7, 20)]
check("rigid: RMSE depends on k", np.ptp(r) > 0.1, f"{np.round(r, 2)}")
# 3. conventions agree: thesis alpha 1.5 == repo alpha 1/1.5
a = G.fit_session(x, y, cam, J.pair_id, 0.72, 1.5, 1, "thesis").rmse_px
b = G.fit_session(x, y, cam, J.pair_id, 0.72, 1 / 1.5, 1, "repo").rmse_px
check("thesis/repo alpha conventions agree", abs(a - b) < 1e-9, f"{a:.4f} vs {b:.4f}")
# 4. outer loop reproduces the earlier 20/11/18 Schira-only fit (train RMSE 7.9 px, landmark-fits-201118)
o = G.optimize_schira(x, y, cam, J.pair_id, "thesis", "similarity")
check("nested fit on Joanna's 20/11/18 points ~ 7.9 px", abs(o["rmse_px"] - 7.9) < 0.3, f"{o}")
# 5. bar-end pairing: swapping the two ends of a bar must not change the best fit
B = L[(L.session == "100718a")]
xb, yb, cb, pb = B.x_deg.values, B.y_deg.values, (B.cam_col + 1j * B.cam_row).values, B.pair_id.values
r1 = G.fit_session(xb, yb, cb, pb, 0.74, 1.0, 1, "thesis").rmse_px
i = np.where(pb != "")[0][:2]; xs, ys = xb.copy(), yb.copy(); xs[i] = xs[i[::-1]]; ys[i] = ys[i[::-1]]
r2 = G.fit_session(xs, ys, cb, pb, 0.74, 1.0, 1, "thesis").rmse_px
check("bar-end order does not matter", abs(r1 - r2) < 1e-9, f"{r1:.4f} vs {r2:.4f}")
print("ALL PASS" if ok else "SOME FAILED")
