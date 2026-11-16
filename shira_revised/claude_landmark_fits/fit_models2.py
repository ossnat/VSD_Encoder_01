from landmarks import *
from fit_models import reg
from scipy.optimize import minimize
n = len(CAM)
def run(label, model, x0, bounds):
    def fitmask(m):
        def err(p):
            pred = model(p, m); return np.sqrt(np.mean(np.abs(pred[m]-CAM[m])**2))
        best = None
        for s in x0:
            r = minimize(err, s, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun: best = r
        return best
    b = fitmask(np.ones(n, bool)); loo = []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False; bb = fitmask(m); loo.append(abs(model(bb.x, m)[k]-CAM[k]))
    loo = np.array(loo)
    print(f"{label:52s} params={np.round(b.x,2)}  train={b.fun:.2f}px  LOO={np.sqrt(np.mean(loo**2)):.2f}px")
    return b
def m_aff(p, m):     # p = a, alpha ; affine registration
    w = schira(deg_correct(VF_R, VF_C), p[0], p[1]); f, Aff = reg(w[m], CAM[m], "aff"); return f(w)
def m_sim(p, m):
    w = schira(deg_correct(VF_R, VF_C), p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
def m_xsq(p, m):     # p = a, alpha, rho ; pre-Schira squeeze of screen x by rho (what the bug did, 0.54)
    z = deg_correct(VF_R, VF_C); z = p[2]*z.real + 1j*z.imag
    w = schira(z, p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
starts2 = [[a, al] for a in (0.3, 0.7, 1.2) for al in (0.7, 1.2, 1.8)]
run("CORRECT + similarity, plausible a 0.3-1.5, alpha 0.67-2", m_sim, starts2, [(0.3,1.5),(0.67,2)])
b = run("CORRECT + affine, plausible a 0.3-1.5, alpha 0.67-2", m_aff, starts2, [(0.3,1.5),(0.67,2)])
w = schira(deg_correct(VF_R, VF_C), *b.x); _, Aff = reg(w, CAM, "aff"); s = np.linalg.svd(Aff, compute_uv=False)
print(f"   -> affine stretch ratio {s[1]/s[0]:.2f} (tilt {np.degrees(np.arccos(s[1]/s[0])):.0f} deg)")
run("CORRECT + free screen-x squeeze rho + similarity", m_xsq, [s+[r] for s in starts2 for r in (0.5, 1.0)], [(0.05,5),(0.3,6),(0.2,1.5)])
