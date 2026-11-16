import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import landmarks as L
from scipy.optimize import minimize
# Boromir 24/11/21, 4 dots (deg, from first_points_241121.xlsx / opt_points_241121.mat); camera (row, col) from opt_points cols 5-6
VF = np.array([[-1.5, -1.0], [-1.0, -1.0], [-1.5, -0.5], [-1.0, -0.5]])     # columns 3-4 of opt_points = (y?, x?)
P = np.array([[75,50,-1.5,-1,54,33],[50,50,-1,-1,64,49],[75,25,-1.5,-0.5,44,33],[50,25,-1,-0.5,48,47]], float)
# xlsx: cond4 (-1,-1.5) -> pixels (50,75): x=-1 (50 px at 50 ppd), y=-1.5 (75 px). So x = -P[:,1]/50, y = -P[:,0]/50
x = -P[:,1]/50; y = -P[:,0]/50; cam = P[:,5] + 1j*P[:,4]
def predict(rho, a, al, mask):
    z = rho*np.abs(x) + 1j*y          # left visual field mirrored into the right (Schira models one hemifield)
    w = L.schira(z, a, al); best = None
    for ww in (w, np.conj(w)):       # allow mirror (right hemisphere / camera orientation)
        M = np.column_stack([ww, np.ones_like(ww)]); p = np.linalg.lstsq(M[mask], cam[mask], rcond=None)[0]
        pr = M @ p; e = np.sqrt(np.mean(np.abs(pr[mask]-cam[mask])**2))
        if best is None or e < best[0]: best = (e, pr)
    return best
m = np.ones(4, bool)
print("Boromir legacy-grid ratio: x %.1f px/deg, y %.1f px/deg -> x/y = %.2f" % (959/6.34, 539/5.30, (539/5.30)/(959/6.34)))
for a, al in ((1.24, 1.46), (2.14, 1.36), (0.72, 1.5)):
    rr = np.linspace(0.2, 1.5, 131); es = [predict(r, a, al, m)[0] for r in rr]
    i = int(np.argmin(es)); print(f"a={a}, alpha={al} fixed: best rho = {rr[i]:.2f}  RMSD {es[i]:.2f}px   (rho=1: {predict(1,a,al,m)[0]:.2f}px)")
