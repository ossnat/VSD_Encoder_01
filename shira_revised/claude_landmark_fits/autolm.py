import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
D = "/mnt/user-data/uploads/VSD_Encoder_01/shira_revised/claude_landmark_fits/cond_avgs/"
# shapes sessions centred at (0.6, -0.75); 1-deg bars; point 0.1
SESS = {"100718a": dict(point="condAN1", vbar="condAN5", hbar="condAN7"),
        "100718b": dict(point="condAN1", vbar="condAN5", hbar="condAN7"),
        "270618b": dict(point="condAN1", vbar="condAN5", hbar="condAN7"),
        "240718b": dict(point="condAN1", vbar="condAN5", hbar="condAN7"),
        "240718c": dict(point="condAN1", vbar="condAN5", hbar="condAN7")}
VF = {"point": [(0.6, -0.75)], "vbar": [(0.6, -0.25), (0.6, -1.25)], "hbar": [(0.1, -0.75), (1.1, -0.75)]}
def resp(x, w0=35, w1=46, sig=2.0):
    x = x.reshape(100, 100, -1); r = np.nanmean(x[:, :, w0:w1], 2) - np.nanmean(x[:, :, 5:25], 2)
    m = np.isnan(r); r = gaussian_filter(np.nan_to_num(r), sig); r[m] = np.nan; return r
def peaks(r, n, mind=15, border=8):
    rr = np.nan_to_num(r, nan=-np.inf).copy(); rr[:border] = rr[-border:] = -np.inf; rr[:, :border] = rr[:, -border:] = -np.inf
    loc = (rr == maximum_filter(rr, size=7)) & np.isfinite(rr); cand = np.argwhere(loc)
    cand = cand[np.argsort(-rr[loc])]; out = []
    for p in cand:
        if all(np.hypot(*(p - q)) >= mind for q in out): out.append(p)
        if len(out) == n: break
    return np.array(out, float)
def extract(s):
    z = np.load(D + f"session_{s}.npz"); cf = SESS[s]; res = {}
    pt = peaks(resp(z[cf["point"]]), 1)[0]; res["point"] = [pt]
    for k in ("vbar", "hbar"):
        pk = peaks(resp(z[cf[k]]), 2)
        # order endpoints: the end closer (on camera) to the point response belongs to... decide by VF eccentricity via distance ranking below
        res[k] = list(pk)
    return z, res
if __name__ == "__main__":
    fig, axs = plt.subplots(len(SESS), 3, figsize=(9, 3*len(SESS)))
    for i, s in enumerate(SESS):
        z, res = extract(s)
        for j, k in enumerate(("point", "vbar", "hbar")):
            r = resp(z[SESS[s][k]]); v = np.nanpercentile(np.abs(r), 99); ax = axs[i, j]
            ax.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v); ax.axis("off"); ax.set_title(f"{s} {k}", fontsize=9)
            for p in res[k]: ax.plot(p[1], p[0], "o", mfc="none", mec="k", ms=10, mew=2)
            print(s, k, [tuple(p.astype(int)) for p in res[k]])
    plt.tight_layout(); plt.savefig("/tmp/claude-0/-home-claude/7857d7c0-375a-5b4c-859e-71f22b2bf6ee/scratchpad/auto_peaks.png", dpi=80)
