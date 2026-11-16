"""Step 2 visual check: stimulus ink pushed through each LUT, drawn on the mean 100718a response (window 35-46).
Red = current repo LUT (set 100718), cyan = v2 (Schira + stretch, set 100718_v2rho). Gray = noise-ceiling hull."""
import numpy as np, pandas as pd, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
from common import REPO
from src.schira_encoding.lut import SchiraLUT
from src.paths import resolve_data_path
from src.encoding.schema import encoding_pairs_manifest_path

LUTDIR = REPO / "experiments/schira_encoding/gandalf/lut"
luts = {k: SchiraLUT.load(LUTDIR / f"lut__set-{k}__anchor-100718a__vsd-100x100__canvas-210__input-224.npz") for k in ("100718", "100718_v2rho")}
hull = np.load(REPO / "experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy").astype(bool)
m = pd.read_parquet(encoding_pairs_manifest_path(resolve_data_path("Data/VSD_Encoder_01/encoding_pairs", REPO), "gandalf", "win_0035_0046"))
m = m[m.date.astype(str) == "100718a"]
conds = [("condAN1", "point 0.1°"), ("condAN4", "circle 0.3°"), ("condAN3", "triangle 0.4°"), ("condAN5", "vertical bar 1°"), ("condAN7", "horizontal bar 1°")]
fig, axs = plt.subplots(2, len(conds), figsize=(3.1 * len(conds), 6.6))
for j, (cond, title) in enumerate(conds):
    rows = m[m.condition == cond]
    maps = np.stack([xr.open_dataarray(resolve_data_path(p, REPO)).values for p in rows.nc_path])
    resp = -(np.nanmean(maps, 0) - 1.0)           # dark-ink response: F/F0 drop -> positive
    img = np.asarray(Image.open(resolve_data_path(rows.image_path.iloc[0], REPO)).convert("L"), float)
    ink = np.abs(img - np.median(img)); ink /= ink.max()
    stim = axs[0, j]; stim.imshow(img[:120, :120], cmap="gray"); stim.set_title(f"{title}\n(canvas, top-left 120 px)", fontsize=9)
    ax = axs[1, j]; v = np.nanpercentile(resp[hull], [2, 99.5])
    ax.imshow(resp, cmap="magma", vmin=v[0], vmax=v[1]); ax.contour(hull, [0.5], colors="#bbbbbb", linewidths=0.8)
    for k, col in (("100718", "#ff3b30"), ("100718_v2rho", "#22d3ee")):
        L = luts[k]; s = np.where(L.valid, map_coordinates(ink, [np.nan_to_num(L.stim_row), np.nan_to_num(L.stim_col)], order=1), 0)
        if s.max() > 0: ax.contour(s, [0.25 * s.max()], colors=col, linewidths=1.3)
    ax.set_xticks([]); ax.set_yticks([]); stim.set_xticks([]); stim.set_yticks([])
axs[1, 0].set_ylabel("mean 100718a response (−ΔF/F0)\nred: current LUT  ·  cyan: v2 stretch", fontsize=9)
fig.suptitle("Stimulus ink mapped onto the 100718a camera: current repo geometry (red) vs Schira + stretch v2 (cyan)", fontsize=11)
plt.tight_layout(); out = REPO / "shira_revised/geometry_v2/plots/mapping_ink_overlay_100718a.png"; out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=140); print(out)
