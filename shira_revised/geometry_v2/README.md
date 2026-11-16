# geometry_v2: Schira + stretch fitted on landmarks, used for encoding (anchor 100718a)

Main `src/` is untouched. Everything here plugs into the repo through a new `schira_set` name.

| file | what |
|---|---|
| `common.py` | landmarks on the 100718a grid (0-based), divide-convention Schira, repo-geometry forward |
| `fit_anchor.py` → `fit_anchor.json` | step 0 (score sets.yaml geometry) + step 1 (joint fits, LOO) |
| `build_lut_v2.py` | step 2: SchiraLUT with stretch → `experiments/schira_encoding/gandalf/lut/lut__set-100718_v2rho__…npz` (+ `100718_v2none`) |
| `plot_mapping.py` | ink of the stimulus through old vs new LUT on the mean 100718a response |
| `configs/schira_encoding_100718_v2*.yaml` | copy of `configs/schira_encoding/default.yaml`, only `schira_set` changed |
| `loo_lean.py` | memory-lean twin of script 20 (same folds, targets, model, files; weights identical to 1e-12 on set 100718) |
| `run_no_torch.py` | run scripts 11 / 27 where torch is not installed (the Linux VM) |
| `plot_compare.py` | orig vs recon, hull ROI, three geometries side by side |

## Results (Oct 13 2026)

Landmarks (camera px, LOO): sets.yaml 100718 = 5.9 on 10/07/18 points, 27 on 20/11/18 points.
Shared a, alpha, rho + per-day camera: no stretch 9.6, **stretch 4.0** (a 2.14, alpha_div 1.36 = alpha_repo 0.74, rho 0.58).
One camera through the vessel registrations is worse (8.8): the three 201118→100718a registrations disagree (rotation 25 / 16 / 3 deg).

Encoding (global-channel ridge, VGG16 block1_prepool, protocol B, hull r of test-mean maps, 3661 shared px):
current 0.440 · v2 no stretch 0.422 · v2 stretch 0.373. The stretch places the bars better but makes every mapped shape
smaller than the measured response, and its LUT edge (horizontal meridian = canvas top row) adds a CNN border artifact.
The shared-channel readout cannot blur, so hull Pearson rewards a geometry that magnifies, not one that positions.

Plots: `plots/mapping_ink_overlay_100718a.png`, `plots/orig_vs_recon_3geometries.png`,
script-27 panels in `plots/orig_vs_recon/{current_repo,v2_stretch}/schira_shared_channels/`.
