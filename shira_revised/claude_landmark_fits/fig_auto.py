import io, contextlib, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import autofit as F, autolm as A
z = np.load(A.D + "session_100718b.npz"); cf = A.SESS["100718b"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6)); fig.patch.set_facecolor("white")
for ax, (k, ttl, keys) in zip(axs, (("point", "0.1° point at (0.6, −0.75)", ["point"]),
                                    ("vbar", "1° vertical bar", ["vbar_top", "vbar_bot"]),
                                    ("hbar", "1° horizontal bar", ["hbar_near", "hbar_far"]))):
    r = A.resp(z[cf[k]]); v = np.nanpercentile(np.abs(r), 99)
    ax.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v)
    for n in keys:
        a, m = F.AUTO[n], F.MANUAL[n]
        ax.plot(a[1], a[0], "o", mfc="none", mec="#0b0b0b", ms=13, mew=2.2)
        ax.plot(m[1], m[0], "x", color="#1baf7a", ms=11, mew=2.5)
        ax.annotate(n.replace("_", " "), (a[1], a[0]), xytext=(9, -9), textcoords="offset points", fontsize=8.5)
    ax.set_title(f"{ttl}\n(100718b, frames 35–45 minus baseline)", fontsize=10, loc="left"); ax.set_xticks([]); ax.set_yticks([])
axs[0].plot([], [], "o", mfc="none", mec="#0b0b0b", ms=10, mew=2, label="automatic (consensus of 5 sessions)")
axs[0].plot([], [], "x", color="#1baf7a", ms=10, mew=2.5, label="Joanna's manual landmark")
axs[0].legend(loc="lower left", fontsize=8, frameon=True)
fig.suptitle("Automatic vs manual landmarks, Gandalf shapes sessions: agree within 2–5 px, except the bar end nearest the vertical meridian (14 px)",
             fontsize=11, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig("/mnt/user-data/outputs/auto_vs_manual_landmarks.png", dpi=130)
