from fit_models2 import *
def m_off(p, m):      # a, alpha, dx, dy : fixation offset in deg (eye calibration), similarity registration
    z = deg_correct(VF_R, VF_C) + (p[2] + 1j*p[3]); w = schira(z, p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
def m_scl(p, m):      # a, alpha, sx, sy : separate px/deg per axis (free), similarity
    z = deg_correct(VF_R, VF_C); z = p[2]*z.real + 1j*p[3]*z.imag; w = schira(z, p[0], p[1]); f, _ = reg(w[m], CAM[m], "sim"); return f(w)
st = [s + [dx, dy] for s in starts2 for dx in (-0.3, 0, 0.3) for dy in (-0.3, 0.3)]
run("CORRECT + fixation offset (dx,dy) + similarity", m_off, st, [(0.05,5),(0.3,6),(-1,1),(-1,1)])
