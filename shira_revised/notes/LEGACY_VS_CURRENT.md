# Legacy MATLAB vs current Python — mismatch notes

Legacy root (local):

```text
/Users/ossnat/GondaResearch/VSD_FM/legacy_code/Retinotopic model Joanna/Gandalf/
```

## 1. What actually ran in the lab

Orchestration: `optimization_joanna.m`, `choose_best_trans_left.m`,
`retinotopy_transform.m`.

Kernel: **`Code/Amit old code/trans_cortex.m`** (model=2, b=201 → monopole).

Chamber warp: **`trans_and_registration.m`** /
`leftHem_trans_opt.m` — `fitgeotrans(..., 'nonreflectivesimilarity')` onto
100×100 with low-hemisphere row flips.

**Not** ground truth for Gandalf outputs:

- `retinotopic_model_joanna.m` — incomplete draft; uses `P = alpha*theta`
  (paper-like) and even overwrites `a = s`. Never called by the opt/viz path.

## 2. Formula comparison

### Legacy monopole Double-Sech (`trans_cortex.m`, model=2, b≥200)

```matlab
[T,R] = cart2pol(mat(:,1), mat(:,2));  % after abs() linspace of retino_mat
T = T / s;                              % s ≡ alpha  ← DIVIDE
c = sech(log(R/a)*0.7609) * 0.1821;
corrPolar = T .* (sech(T).^c);
w = K * log(R .* exp(1i*corrPolar) + a);
% then: abs(angle(w)), abs(w) → pol2cart → round(*1000) integer grid
```

### Current Python (`src/retinotopy/schira.py`)

```text
P = α * θ                               ← MULTIPLY
f_a = sech(P) ** (sech(log(E/a)*0.76) * 0.1821)
w = k * log(E * exp(i * P * f_a) + a)
```

Algebraically these are **not** the same map for α ≠ 1. With α=1.5
(Joanna 201118 winner) the compressed angle is `θ/1.5` in MATLAB vs
`1.5·θ` in Python — opposite direction relative to identity.

Amit himself commented in `trans_cortex.m`:

> why did they divide T by alpha? According to M. Schira T = theta*alpha

So the lab knowingly ran the **divide** convention. Matching prior student
figures means matching **divide**, not “fixing” to the paper form first.

## 3. Visual-field sampling footguns

1. **Stimulus placement** (`create_letter_stim.m`, `vertical_line.m`):
   origin = **top-left** of 301×512, `ppd=35`. Letter A center `(1, -0.9)`,
   D `(1.4, -0.7)`, bar `(0.6, -0.75)`.

2. **`create_retino_map`** builds a full-frame world map with an absolute
   center (`cal`), then Joanna **crops** to `x>0`, `y<0`
   (saved crop: `retino_mat_20_03_22.mat`, 232×213×2).

3. **`trans_cortex` does not use that map pixelwise.** It takes
   `abs(corners)` and `linspace`s onto the full stimulus size — both axes
   become **positive** (first-quadrant Schira input). That is why
   top-left-as-fovea placement is consistent with the warp even though the
   cropped `retino_mat` stores signed degrees.

4. Our encoding canvas is **210×210** LRQ with **signed** `y_deg < 0` and
   `θ = atan2(y,x) ∈ [-π/2, 0]`. Even with the same `(a,α,k)`, the complex
   `w` cloud will not match unless we adopt the abs / Q1 (or an equivalent
   signed) convention consistently.

## 4. Registration / alignment

| Stage | Legacy | Current |
|-------|--------|---------|
| Schira → cortex pixels | Integer `xcoord,ycoord` from `abs(w)` | Continuous complex `w` |
| Fit to chamber | Exactly **2** anchors, nonreflective similarity | N landmarks, free or rigid similarity |
| Hemisphere | `lowH`: flip brain `y` before fit, flip image after; leftH: no x flip | `flip_u` / `flip_v` on continuous affine |
| Inter-session | `gandalf_*_pixels4Retino.mat` chamber landmarks | `session_register` vessel transforms |
| Golden RMSD | Joanna `25-07-2022-17`: mean ≈ **1.52 px** | Phase-0 letter F: ≈ **8.7–9.7 px** |

Likely causes of the RMSD gap (ordered):

1. Wrong α direction / VF abs convention → Schira ink geometry wrong before any affine.
2. Different registration objective (N-landmark free fit vs 2-pt constrained).
3. Different stimulus rasterization (210 vs 301×512, ink style).
4. Camera affine `needs_confirmation` / unconfirmed 100718 defaults.

## 5. Suggested corrections (priority)

### P0 — Match legacy forward map for QC

1. Implement `alpha_mode="divide"` (+ `vf_abs=True`, `ecc_width=0.7609`) as in
   `shira_revised/src/legacy_schira.py`.
2. Re-run letter/bar forward overlays with Joanna winner
   `(a=0.72, α=1.5, k=1)` and Amit `(a=0.39, α=1.5, k=1)`.
3. Reproduce 2-pt chamber registration with the same anchors as
   `Opt_results/201118/25-07-2022-17` row 0
   (VF pix anchors `(49,18)` & `(15,18)` — see winner matrix cols 1–4 /
   opt_points mapping).

### P1 — Landmark RMSD regression

1. Load `opt_points_8_5_22.mat` (8 landmarks).
2. Fit/apply legacy path; assert mean RMSD ≈ 1.52 for the published winner.
3. Compare same landmarks through current `forward_schira` + phase-0 affine;
   quantify gap.

**Empirical note (2026-09-29):** `scripts/03_point_rmsd_regression.py` with
continuous `w` + naive `deg = pix/ppd` gives ~**48 px** RMSD (legacy α) vs
Joanna’s **1.52**. So α alone is not enough: MATLAB looks up
`xcoord(round(y_pix), round(x_pix))` after `trans_cortex`’s abs-linspace VF
grid, then runs `fitgeotrans` + low-hem flips. Port that registration path
next (do not treat continuous-`w` 2-pt as parity).

### P2 — Decide encoding convention

Once overlays match, choose one:

- **A.** Keep paper `P=αθ` in encoding but **re-fit** `(a,α)` and camera under
  that convention until RMSD matches thesis figures; or
- **B.** Switch encoding to lab `θ/α` (+ abs VF) so parameters transfer
  directly from Joanna winners.

Document the choice in this folder before touching main `src/retinotopy`.

### P3 — Canvas / stim parity (only if P0–P1 still fail)

Port 301×512 letter/bar placement and/or resample our 210 canvas through the
legacy linspace abs map for fair visual diffs.

## 6. Golden artifacts (already on disk)

| Artifact | Role |
|----------|------|
| `Code/Amit old code/winner201118.mat` | Amit `a=0.39,s=1.5,k=1` + 2 anchors |
| `Opt_results/201118/25-07-2022-17/min_50_rmsd_winners.mat` | Joanna best ≈ `(0.72,1.5,1)`, RMSD 1.52 |
| `Opt_results/201118/08-05-2022-18/` | Early same anchors, RMSD 1.14 |
| `VSDI points/201118/opt_points_8_5_22.mat` | 8 VF↔brain landmarks |
| `Results Data/20_03_2022 results/retino_mat_20_03_22.mat` | Cropped VF deg map |
| `images/201118/Session {a,d}/*.png` | Empirical letter VSD |
| `images/100718/Session b/*Bar*.png` | Empirical bars |
| `gandalf_{100718,201118,200617}_pixels4Retino.mat` | Chamber↔chamber pts |
| `Opt_results/100718/*/min_50_rmsd_winners.mat` | Bar opt winners (~a 0.69–0.81, α≈1.1) |
