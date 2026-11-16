import numpy as np, scipy.io as sio, glob, sys
import landmarks as L
from fit_models import reg
from scipy.optimize import minimize
G = "/mnt/user-data/uploads/VSD_FM/legacy_code/Retinotopic model Joanna/Gandalf/VSDI points/100718/"
def use(P):
    L.VF_R, L.VF_C = P[:,0].astype(float), P[:,1].astype(float)
    L.CAM = P[:,5].astype(float) + 1j*P[:,4].astype(float)
def models():
    VF_R, VF_C, CAM, schira, dc, dl = L.VF_R, L.VF_C, L.CAM, L.schira, L.deg_correct, L.deg_legacy
    n = len(CAM)
    def m_leg(p, m):  w = schira(dl(VF_R, VF_C), p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
    def m_sim(p, m):  w = schira(dc(VF_R, VF_C), p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
    def m_aff(p, m):  w = schira(dc(VF_R, VF_C), p[0], p[1]); f, _ = reg(w[m], CAM[m], "aff"); return f(w)
    def m_xsq(p, m):
        z = dc(VF_R, VF_C); z = p[2]*z.real + 1j*z.imag; w = schira(z, p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
    st2 = [[a, al] for a in (0.3, 0.7, 1.2, 2.5) for al in (0.7, 1.2, 1.8, 3)]
    specs = [("legacy grid + similarity", m_leg, st2, [(0.05,5),(0.3,6)]),
             ("CORRECT + similarity", m_sim, st2, [(0.05,5),(0.3,6)]),
             ("CORRECT + affine camera", m_aff, st2, [(0.05,5),(0.3,6)]),
             ("CORRECT + free screen-x squeeze", m_xsq, [s+[r] for s in st2 for r in (0.4, 0.7, 1.0)], [(0.05,5),(0.3,6),(0.2,1.5)])]
    for label, model, x0, bounds in specs:
        def fitmask(m):
            err = lambda p: np.sqrt(np.mean(np.abs(model(p, m)[m]-CAM[m])**2))
            return min((minimize(err, s, method="L-BFGS-B", bounds=bounds) for s in x0), key=lambda r: r.fun)
        b = fitmask(np.ones(n, bool)); loo = []
        for k in range(n):
            m = np.ones(n, bool); m[k] = False; loo.append(abs(model(fitmask(m).x, m)[k]-CAM[k]))
        print(f"   {label:34s} params={np.round(b.x,2)}  train={b.fun:.2f}px  LOO={np.sqrt(np.mean(np.array(loo)**2)):.2f}px")
# 1) the file behind the thesis winner (a=0.74, alpha=1, anchors (9,21),(37,21))
P = sio.loadmat(G + "opt_points_25_09_23_23_01.mat")["opt_points"]; use(P)
print("Thesis file opt_points_25_09_23_23_01 (n=%d)" % len(P))
w = L.schira(L.deg_legacy(L.VF_R, L.VF_C), 0.74, 1.0)
i1 = np.where((P[:,0]==9)&(P[:,1]==21))[0][0]; i2 = np.where((P[:,0]==37)&(P[:,1]==21))[0][0]
c1 = (L.CAM[i2]-L.CAM[i1])/(w[i2]-w[i1]); d = np.abs(c1*w + L.CAM[i1]-c1*w[i1] - L.CAM); mk = np.ones(len(d), bool); mk[[i1,i2]] = False
print("   reproduce thesis: joanna metric %.2f (thesis 2.56), true RMSD %.2f px" % (np.sqrt(d[mk].mean()), np.sqrt((d[mk]**2).mean())))
models()
# 2) union of all distinct landmarks across the 100718 files (latest value per stimulus point)
U = {}
for f in sorted(glob.glob(G + "opt_points*.mat"), key=lambda f: __import__('os').path.getmtime(f)):
    for r in sio.loadmat(f)["opt_points"]: U[(r[0], r[1])] = r
P = np.array(list(U.values())); use(P)
print("Union of all 100718 landmark files (n=%d distinct stimulus points)" % len(P)); models()
