"""Original vs reconstruction inside the noise-ceiling hull, three geometries side by side (global-channel ridge,
VGG16 block1_prepool, LOO protocol B, test-trial means). r = Pearson inside the pixels all three share."""
import numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from common import REPO
from src.plotting_colormaps import VSD_CMAP, register_mapgeog
from src.paths import resolve_data_path
register_mapgeog(); cmap = matplotlib.colormaps[VSD_CMAP].copy(); cmap.set_bad("white")
B = REPO / "experiments/schira_encoding/gandalf/loo/win_0035_0046/vgg16_imagenet/block1_prepool/set-{}__anchor-100718a/global_channel_ridge/protocol_B_noise_ceiling_hull"
SETS = [("100718", "current repo\n(sets.yaml 100718)"), ("100718_v2none", "v2: landmark fit,\nno stretch"), ("100718_v2rho", "v2: landmark fit,\nstretch ρ=0.58")]
hull = np.load(REPO / "experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy").astype(bool)
folds = sorted(f for f in os.listdir(str(B).format("100718")) if f.startswith("B__"))
M = {(s, f): (np.load(f"{str(B).format(s)}/{f}/fold_mean_orig.npy"), np.load(f"{str(B).format(s)}/{f}/fold_mean_recon.npy")) for s, _ in SETS for f in folds}
common = hull.copy()
for o, p in M.values(): common &= np.isfinite(o) & np.isfinite(p)
r = lambda o, p: np.corrcoef(o[common], p[common])[0, 1]
z = lambda a: np.where(common, (a - np.nanmean(a[common])) / np.nanstd(a[common]), np.nan)
fig, axs = plt.subplots(len(folds), 5, figsize=(10.5, 2.15 * len(folds)))
for i, f in enumerate(folds):
    pairs = pd.read_parquet(f"{str(B).format('100718')}/{f}/fold_pairs.parquet"); t = pairs[pairs.loo_split == "test"]
    img = np.asarray(Image.open(resolve_data_path(str(t.image_path.iloc[0]), REPO)).convert("RGB"))
    axs[i, 0].imshow(img[:120, :120]); axs[i, 0].set_ylabel(f[3:].replace("_", " "), fontsize=8)
    o = M["100718", f][0]; axs[i, 1].imshow(z(o), cmap=cmap, vmin=-2.5, vmax=2.5)
    for j, (s, lab) in enumerate(SETS):
        axs[i, 2 + j].imshow(z(M[s, f][1]), cmap=cmap, vmin=-2.5, vmax=2.5)
        axs[i, 2 + j].text(3, 96, f"r = {r(*M[s, f]):.2f}", fontsize=8, color="k", bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    for a in axs[i]: a.set_xticks([]); a.set_yticks([]); a.contour(hull, [.5], colors="#999", linewidths=.5) if a is not axs[i, 0] else None
for a, t in zip(axs[0], ["stimulus (top-left 120 px)", "original (test-trial mean)", *[l for _, l in SETS]]): a.set_title(t, fontsize=8.5)
means = [np.mean([r(*M[s, f]) for f in folds]) for s, _ in SETS]
fig.suptitle("Original vs reconstruction, noise-ceiling hull ROI (z-scored inside the shared hull pixels, %d px)\n"
             "mean r: current %.3f · v2 no stretch %.3f · v2 stretch %.3f" % (common.sum(), *means), fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.975)); out = REPO / "shira_revised/geometry_v2/plots/orig_vs_recon_3geometries.png"; plt.savefig(out, dpi=120); print(out)
