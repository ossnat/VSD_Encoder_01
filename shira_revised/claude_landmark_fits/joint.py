import io, contextlib, numpy as np, scipy.io as sio
with contextlib.redirect_stdout(io.StringIO()):
    import landmarks as L
    from fit_models import reg
from scipy.optimize import minimize
# day 1: 20/11/18 letters (8 pts); day 2: 10/07/18 bars (10 xlsx pts)
P1 = sio.loadmat(L.G + "VSDI points/201118/opt_points_8_5_22.mat")["opt_points"].astype(float)
pts = [((21,26),(44,39)), ((14,14),(21,36)), ((14,38),(60,30)), ((35,26),(47,45)), ((21,14),(25,41)),
       ((21,37),(58,36)), ((21,9),(24,46)), ((21,43),(70,36)), ((4,26),(37,24)), ((38,26),(48,51))]
P2 = np.array([[s[1], s[0], 0, 0, c[1], c[0]] for s, c in pts], float)
days = [(P[:,0], P[:,1], P[:,5] + 1j*P[:,4]) for P in (P1, P2)]
def pred(p, masks, shared_rho=True, use_rho=True):
    a, al = p[0], p[1]; out = []
    for d, (r, c, cam) in enumerate(days):
        rho = (p[2] if shared_rho else p[2+d]) if use_rho else 1.0
        z = L.deg_correct(r, c); z = rho*z.real + 1j*z.imag
        w = L.schira(z, a, al); f, _ = reg(w[masks[d]], cam[masks[d]], "sim"); out.append(f(w))
    return out
def fit(masks, **kw):
    nr = 0 if not kw.get("use_rho", True) else (1 if kw.get("shared_rho", True) else 2)
    def err(p):
        pr = pred(p, masks, **kw); e = np.concatenate([np.abs(pr[d]-days[d][2])[masks[d]] for d in range(2)])
        return np.sqrt(np.mean(e**2))
    starts = [[a, al] + [r]*nr for a in (0.3, 0.8, 1.5, 3) for al in (0.7, 1.2, 2) for r in (0.5, 0.8)]
    bounds = [(0.05, 5), (0.3, 6)] + [(0.2, 1.5)]*nr
    return min((minimize(err, s, method="L-BFGS-B", bounds=bounds) for s in starts), key=lambda r: r.fun)
for label, kw in (("no squeeze (shared a, alpha)", dict(use_rho=False)),
                  ("ONE shared squeeze for both days", dict()),
                  ("separate squeeze per day", dict(shared_rho=False))):
    full = [np.ones(len(d[2]), bool) for d in days]; b = fit(full, **kw); loo = []
    for d in range(2):
        for k in range(len(days[d][2])):
            m = [x.copy() for x in full]; m[d][k] = False
            loo.append(abs(pred(fit(m, **kw).x, m, **kw)[d][k] - days[d][2][k]))
    print(f"{label:34s} params={np.round(b.x,2)}  train={b.fun:.2f}px  LOO={np.sqrt(np.mean(np.array(loo)**2)):.2f}px")
