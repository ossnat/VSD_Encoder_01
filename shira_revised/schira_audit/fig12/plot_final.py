"""Final Fig. 12 figure from saved fits (labelled_results.pkl, fits_bug_vs_correct.npy)."""
import numpy as np, pickle
exec(open("fig12_schira_only.py").read().split("results = {}")[0])
import matplotlib.pyplot as plt
out = pickle.load(open("labelled_results.pkl", "rb")); F = np.load("fits_bug_vs_correct.npy", allow_pickle=True).item()
nl = len(s); lab = np.repeat(np.arange(14), nl)
by = {o[0]: o for o in out}
panels = [("legacy grid (bug) + similarity, a=0.727 alpha=1.46", "#eb6834", "How the thesis figure was made\n(legacy px→deg grid, a=0.727, α=1.46)"),
          ("Schira only: a, alpha, k + similarity camera", "#2a78d6", "35 px/deg, Schira only: a, α, k free\n+ best camera (similarity)"),
          ("Schira only + S1, S2 free", "#1f9bd1", "35 px/deg, Schira only, also the\ntwo sech constants S1, S2 free"),
          ("SLIP", "#7f7f7f", "35 px/deg, Schira only, line identity NOT\nenforced: looks close, wrong lines"),
          ("ref: x-stretch rho before Schira + similarity", "#1baf7a", "35 px/deg + x-stretch ρ before Schira"),
          ("ref: Schira + affine camera (tilt)", "#8a6d00", "35 px/deg, Schira + affine camera (tilt)")]
fig, axs = plt.subplots(2, 4, figsize=(17.5, 10.4)); fig.patch.set_facecolor("white")
axs[0, 0].imshow(T.astype(np.uint8)); axs[0, 0].set_title("Thesis Fig. 12B(ii)\n(white = thesis grid)", fontsize=9.5, loc="left")
for ax, (key, col, title) in zip(list(axs.flat[1:7]), panels):
    if key == "SLIP":
        m = F["35 ppd (correct)"]; txt = "a=1.47 α_div=1.40 · anchors 10 cam px off\nthesis line 'row 17' lands on stimulus rows 70/105"
    else:
        _, sc, m, note = by[key]; txt = f"{note}\nthesis→model {sc[0]:.2f} · model→thesis {sc[1]:.2f} cam px"
    mt = k1 * m + k0; ax.imshow(T.astype(np.uint8))
    for j in range(14):
        seg = mt[lab == j]; ax.plot(seg.real, seg.imag, "-", lw=1.1, color=col, alpha=.9)
    ax.set_title(f"{title}\n{txt}", fontsize=8.6, loc="left")
for ax in axs.flat[:7]: ax.set_xlim(25, 250); ax.set_ylim(245, 25); ax.set_xticks([]); ax.set_yticks([])
ax = axs.flat[7]; ax.axis("off")
ax.text(0, 0.97, "Score: mean distance in camera px (1 px = 170 µm)\nbetween each thesis line and the SAME stimulus\nline in the model.\n\n"
        "Stimulus: 3×3° grid, lines every 0.5°, 35 px/deg.\n\nk is free in every Schira fit. It only multiplies w,\nso the camera scale absorbs it: k = 0.1 … 1000\nchanges the mapped grid by < 1e-13 px.", fontsize=9.5, va="top")
fig.suptitle("Can Schira alone (a, α, k) redraw thesis Fig. 12B(ii) at the correct 35 px/deg?", fontsize=12.5, x=0.01, ha="left")
plt.tight_layout(rect=(0, 0, 1, 0.965)); plt.savefig("fig12_same_grid_fits.png", dpi=125); print("ok")
