# Legacy experiments + visual tests we can run

## Experiments present in the legacy tree

### A. Letter retinotopy — Gandalf `201118` (sessions a / d)

- **Stimuli:** G, A, N, D, L, F via `create_letter_stim.m` / `create_stimulus.m`
  - Centers: A `(1, -0.9)`, D `(1.4, -0.7)` (deg; top-left origin convention)
- **Opt:** `optimization_joanna.m` → `Opt_results/201118/<timestamp>/`
- **Viz / correlation:** `choose_best_trans_left.m`, `meanLetters.m`,
  `quantification_CS.m`, `CS_quantification_*_hists.m`
- **Empirical images:** `images/201118/Session {a,d}/{A,D,F,G,L,N}.png`,
  `BV_resized.png`
- **Winner to reproduce:** `25-07-2022-17` → `a=0.72, α=1.5, k=1`, mean RMSD ≈ 1.52

### B. Bar retinotopy — Gandalf `100718`

- **Stimuli:** `vertical_line.m` / HorizBar; center ≈ `(0.6, -0.75)`, length 1°
- **Opt:** many runs under `Opt_results/100718/` (Sep 2023)
- **Empirical:** `images/100718/Session b/VerticalBar.png`, `horizBar.png`, …
- **Late winners:** `a ≈ 0.69–0.81`, `α ≈ 1.1`, `k=1`

### C. Grid QC (map fingerprint)

- `create_cart_stim.m`, `create_polar_stim.m`
- Saved PNGs: `Results Data/28_03_2022 results/cartGrid*.png`, `polarGrid_*.png`

### D. Vertical meridian / intrinsic — `200617`

- `settingVM.m`, `calc_VM_rmsd.m`, `find_vm.m`
- Images under `images/200617/` (orientation map, VM overlays)

### E. Chamber↔chamber registration — `161018` / pixels4Retino

- `register_transform.m`, `two_chambers_trans.m`, `linear_registration_chambers.m`
- Mats: `gandalf_*_pixels4Retino.mat` at Gandalf root

### F. Amit reference path

- `Amit old code/retinoForNoam.m`, `winner201118.mat` (`a=0.39, s=1.5, k=1`)
- Anchors: brain `(25,36)↔VF(0.5,-0.4)`, `(48,55)↔(1.5,-0.4)`

---

## Visual / quantitative tests (this side-track)

Priority order for `shira_revised/scripts/`:

| # | Test | Pass criterion | Script / status |
|---|------|----------------|-----------------|
| 1 | **α convention divergence** | Same `(a,α,k)`, multiply vs divide produce clearly different `w` clouds; plot saved | `01_compare_alpha_conventions.py` |
| 2 | **Numeric spot-check vs formula** | Single VF points: Python legacy port matches closed-form `trans_cortex` monopole | `tests/test_legacy_schira_kernel.py` |
| 3 | **Letter overlay vs Session PNG** | Forward ink (legacy path + Joanna params) blended on `images/201118/.../A.png` looks like lab overlays | `02_overlay_letter_vs_legacy_png.py` |
| 4 | **Point RMSD regression** | With winner anchors + `opt_points_8_5_22`, mean RMSD ≈ 1.5 px | `03_point_rmsd_regression.py` |
| 5 | **Cart / polar grid fingerprint** | Side-by-side with `28_03_2022` PNGs | (next) |
| 6 | **Bar spatial-cut** | Pearson on ROI cuts vs `100718` VerticalBar | (next) |
| 7 | **Current-repo overlay control** | Same letter with `src.retinotopy.forward_schira` + `sets.yaml` affine — expect mismatch until fixed | extend script 02 `--mode current` |
| 8 | **Abs-VF vs signed-VF** | Isolate whether α-only fix is enough or abs() is required | unit test in `test_alpha_convention_divergence.py` |

### How to judge overlays without MATLAB

1. Load legacy empirical PNG (true VSD response).
2. Render Schira ink under candidate convention → chamber 100×100.
3. Save blend + difference; optionally compute ink∩response overlap / centroid error.
4. Compare to lab’s own transformed overlays when present
   (e.g. Progress PPTX slides, `Results Data/*` letter RMSD PNGs).

### MATLAB optional golden check

If Octave/MATLAB available later: run `trans_cortex` on a 1-pixel stimulus and
dump `w` / `x1,y1` for a few `(E,θ)`; assert Python port matches to 1e-6 before
`round(*1000)`.
