from landmarks import *
from scipy.optimize import minimize
A_GRID = np.geomspace(0.05, 5, 40); AL_GRID = np.geomspace(0.3, 6, 40)
def reg(w, cam, kind):
    if kind == "sim":
        M = np.column_stack([w, np.ones_like(w)]); p = np.linalg.lstsq(M, cam, rcond=None)[0]
        return lambda q: p[0]*q + p[1], None
    M = np.column_stack([w.real, w.imag, np.ones(len(w))])
    px = np.linalg.lstsq(M, cam.real, rcond=None)[0]; py = np.linalg.lstsq(M, cam.imag, rcond=None)[0]
    Aff = np.array([[px[0], px[1]], [py[0], py[1]]])
    return (lambda q: px[0]*q.real + px[1]*q.imag + px[2] + 1j*(py[0]*q.real + py[1]*q.imag + py[2])), Aff
def fit(fn, kind, mask):
    z = fn(VF_R, VF_C)
    def err(a, al):
        w = schira(z, a, al); f, _ = reg(w[mask], CAM[mask], kind)
        return np.sqrt(np.mean(np.abs(f(w[mask]) - CAM[mask])**2))
    E = np.array([[err(a, al) for al in AL_GRID] for a in A_GRID]); i, j = np.unravel_index(E.argmin(), E.shape)
    r = minimize(lambda p: err(*np.exp(p)), [np.log(A_GRID[i]), np.log(AL_GRID[j])], method="L-BFGS-B",
                 bounds=[(np.log(0.05), np.log(5)), (np.log(0.3), np.log(6))])
    a, al = np.exp(r.x); w = schira(z, a, al); f, Aff = reg(w[mask], CAM[mask], kind)
    return a, al, f(w), Aff, r.fun
n = len(CAM); rows = []
for label, fn, kind in (("legacy grid + similarity", deg_legacy, "sim"),
                        ("CORRECT + similarity", deg_correct, "sim"),
                        ("CORRECT + affine (= tilted camera)", deg_correct, "aff")):
    a, al, pred, Aff, tr = fit(fn, kind, np.ones(n, bool))
    loo = []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        loo.append(abs(fit(fn, kind, m)[2][k] - CAM[k]))
    loo = np.array(loo); extra = ""
    if Aff is not None:
        s = np.linalg.svd(Aff, compute_uv=False); extra = f" | stretch ratio {s[1]/s[0]:.2f} -> tilt {np.degrees(np.arccos(s[1]/s[0])):.0f} deg"
    print(f"{label:36s} a={a:.2f} alpha={al:.2f}  train RMSD={tr:.2f}px  LOO RMSD={np.sqrt(np.mean(loo**2)):.2f}px "
          f"(median {np.median(loo):.2f}, max {loo.max():.1f}){extra}")
