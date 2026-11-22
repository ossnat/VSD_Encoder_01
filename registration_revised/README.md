# registration_revised

A check of Joanna's 10/07/18 registration points (`legacy_code/.../Gandalf/VSDI points/100718/opt-points-10-07-18.xlsx`)
against the stimuli, the VSD maps and thesis Fig. 11B.

- `points.py` reads the two tables in the xlsx: the left one has stimulus points per condition, the right one has the 6 registration points.
- `make_figs.py` writes the figures to `figs/`. Run it with `scripts/py registration_revised/make_figs.py` (about 1 min).

## Figures

| File | Content |
|---|---|
| `figs/stimulus_points_100718a.png`, `_100718b.png` | One row per condition (1–5, 7). Columns: (1) the stimulus with the VF point(s) in red; (2) the VSD response map with the VSD point(s) in red, read with (1,1) at the **top-left** corner; (3) the same map read with (1,1) at the **top-right** corner. Each red mark is the point and its 8 neighbours. |
| `figs/registration_points_vessels.png` | The 6 registration points on Joanna's chamber vessel image (`BV_resized.png`, high-passed), read both ways, next to thesis Fig. 11B. The thesis plate is registered onto the vessel image (NCC 0.97). Its red dots and blue crosses are drawn as yellow and cyan rings. Bottom row: the mean of frames 1–150 of the condsAN trials, high-passed. |
| `figs/origin_scores.csv` | Scores for each point under four readings: top-left, top-right, and each of those with x and y swapped. |

## Results

**(1,1) is the top-left corner.** The VSD point is (x, y), 1-based, so camera row = y − 1 and column = x − 1. That is how `schira_dashboard/prepare_data.py` already reads these points.

| Reading | Response percentile at stimulus points (a / b) | Distance to thesis Fig. 11B marks |
|---|---|---|
| top-left | **94 / 98** | **4.3 px** |
| top-right | 73 / 82 | 7.5 px |
| top-left, x/y swapped | 67 / 67 | 11.0 px |
| top-right, x/y swapped | 49 / 50 | 12.2 px |

**Thesis Fig. 11B shows 6 marks, but not the same 6 points as the right-hand table.** Under the top-left reading:

- **Blue crosses (anchors):**
  - circle bottom, (58,36), which is table point 5;
  - vertical-bar upper, (24,46), which is only in the left table.
  - These two are the anchor pair of Joanna's 10/07/18 winner in `gandalf_legacy.yaml`, where brain anchors are written (row, col) as (46,24) and (36,58).
- **Red dots:** table points 2, 3, 4 and 6 (horizontal-bar left, 0.05 point, horizontal-bar right, vertical-bar bottom).
- **Not in the figure:** table point 1 (triangle top, (21,36)).

The thesis marks sit about 1–2 px right of and below the xlsx points, and point 4 is off by about 3 px. That is within what the figure's own marker placement and our plate registration can explain.

**The xlsx and our catalog disagree on which point size is condition 1 and which is condition 2.** The xlsx has condition 1 = 0.05° point and condition 2 = 0.1° point; our catalog has the reverse. Condition 1 gives the larger response, which fits the catalog. Both points are at the same place with the same VSD point, so the check isn't affected.

**The blank trials of 10/07/18 are not on this machine.** The condsAN trials are already divided by their baseline (values about 1.0), so their high-passed mean of frames 1–150 shows almost no vessels: only the big vessel near row 65–70, plus artefacts. The vessel panels therefore use Joanna's `BV_resized.png`, which is the image of thesis Fig. 11. Once the raw blank `.mat` files are available, swap the blank trials into `mean_early_frames()`.

The session isn't recorded in the xlsx. The points fit 100718b slightly better than 100718a (98 against 94 percentile). Joanna's image folder also has maps for session b only.

## Summary figures (`summary_figs.py`)

Run with `scripts/py registration_revised/summary_figs.py`.

| File | Content |
|---|---|
| `figs/summary_points_on_thesis.png` | **Left:** the xlsx points (top-left reading) on thesis Fig. 11B. Each thesis mark is paired with its nearest xlsx point, with the distance shown. **Right:** Joanna's thesis winner re-run on the same plate: measured points, model predictions and per-point error. |
| `figs/summary_points_on_stimuli.png` | One row per condition. **Left:** the stimulus with its VF points. **Right:** the 100718b response map with the measured VSD point (green), Joanna's model prediction (cyan ○; yellow for anchors) and our pipeline's prediction (white ×). |
| `figs/summary_thesis_pairs.csv`, `figs/summary_stimulus_points.csv` | The numbers behind the two figures. |

### How Joanna's "RMSD" is calculated

The code is in `leftHem_trans_opt.m`, `find_optimization_pts.m`, `calc_rmsd.m` and `mean_rmsd.m`. We re-run it with the Python port of `trans_cortex`.

1. **Map the canvas.** Run `trans_cortex` on the 301 × 512 canvas, using her `retino_mat` grid.
2. **Fit the camera to the two anchors.** Each anchor's stimulus pixel is mapped to cortex coordinates. Its camera point is (col, 101 − row) (the lower-field flip). An exact two-point `nonreflectivesimilarity` (shift, rotation, scale) is fitted between the two.
3. **Predict the other points.** Each is mapped through the same transform, rounded, and flipped back.
4. **Score.** d = distance between predicted and measured, in camera px.
   - The number she reports is `mean_rmsd` = sum(d) / (n − 2). That is the **mean distance** over the non-anchor points, not a root-mean-square.
   - `calc_rmsd` itself is sqrt(sum(d) / n).

For her thesis winner (a 0.74, α 1.0 with P = θ/α, k 1; anchors vertical-bar upper and circle bottom; points file `opt_points_25_09_23_23_01.mat`), the re-run gives d = 1.41, 5.00, 2.83, 1.00 px. That is:

- mean distance **2.561 px**, the thesis value (2.56);
- `calc_rmsd` 1.307, which matches her saved winner table;
- true RMSE 3.00 px.

So her calculation is reproduced exactly.

### Other findings

- **The xlsx points against the thesis Fig. 11B marks:** a mean distance of 2.57 px, RMSE 2.67 px. That includes our plate registration and the size of the figure's markers. Table point 1 (triangle top) has no thesis mark.
- **One xlsx typo:** for the horizontal-bar right point, the xlsx has x and y swapped compared with her `.mat` file and the thesis figure. The xlsx gives (48, 51); the `.mat` file gives (x, y) = (51, 48). With the xlsx values the error is 5.1 px; with hers it's 2.8 px. The magenta square marks her value in `summary_points_on_stimuli.png`.
- **Her model on all 8 non-anchor stimulus points:** mean distance 4.27 px, RMSE 4.74 px. The worst are the horizontal-bar left end (7.1 px), the triangle top (6.3 px) and the circle top (5.1 px). These are points she did not include in the thesis RMSD.

### Our pipeline compared with Joanna's (white × in `summary_points_on_stimuli.png`)

The white × is our pipeline, `schira_dashboard/schira_geometry.py`, set up to match her model as closely as possible:

- the true 35 px/° instead of her `retino_mat` grid;
- the repo's Schira code instead of her `trans_cortex`;
- the same a 0.74, α 1 (P = θ/α) and k 1;
- an exact similarity fit on the same two anchors;
- no rounding.

A two-point fit can't tell whether the map is mirrored, so we take the mirror that lies closest to Joanna's own predictions, never the one that fits the measured points better.

| | Joanna (thesis RMSD points) | Ours (same points) | Joanna (all 8 non-anchor) | Ours (all 8) |
|---|---|---|---|---|
| Mean distance | 2.56 px | 5.53 px | 4.27 px | 5.00 px |
| RMSE | 3.00 px | — | 4.74 px | 5.38 px |

Our prediction is not identical to hers. The only real difference is the pixel→degree step, plus small differences in the Schira code. Her a and α were tuned on her distorted grid, so with the true geometry they no longer fit as well. A fair comparison would refit a and α with our pipeline.

### Which point set is which

Joanna saved many point sets (`VSDI points/100718/opt_points_*.mat`):

- **The right-hand xlsx table** is `opt_points_26_09_23_11_33.mat`. All 6 points match exactly, apart from the swapped horizontal-bar-right value.
- **Thesis Fig. 11B** shows the set in `opt_points_25_09_23_new3.mat`: point, vertical-bar upper and bottom, horizontal-bar left and right, circle bottom. `opt_points_26_09_23_11_46.mat` contains all 7 points, meaning the xlsx six plus the vertical-bar upper.
- **The thesis RMSD of 2.56 px** comes from `opt_points_25_09_23_23_01.mat`, which uses the triangle-right point in place of horizontal-bar left.

So the triangle top in the xlsx is not a mix-up with the vertical-bar upper. They are two different measured points, from different stimuli, at different VF and VSD positions, and both appear in her files. The xlsx table is simply a later point set than the one shown in the thesis figure.

All points are ends of stimuli (bar ends, a triangle vertex, circle top and bottom), except the 0.05/0.1° dot, which is its centre.

## Horizontal stretch r (`fit_stretch.py`)

Run with `scripts/py registration_revised/fit_stretch.py` (about 10 s). It writes `figs/stretch_fits.csv` and `figs/stretch_profile.png`.

**The model** is Joanna's method plus one parameter, applied in this order:

1. **Stretch:** VF (x, y), at the true 35 px/°, becomes (r·x, y). That is the matrix [[r, 0], [0, 1]].
2. **Schira:** a, α (P = θ/α), k = 1.
3. **Camera:** exact similarity through the 2 anchors, with the flip fixed.
4. **Score:** her mean distance over the non-anchor points.

So the free parameters are a, α and r.

**Why one parameter is enough:** a uniform scale of the VF is exactly equivalent to changing a, so only the ratio r can be identified. Her `retino_mat` grid corresponds to r = 0.415 / 0.77 = 0.54 with a_true = a_joanna / 0.77.

**Results (mean distance over non-anchor points, px; LOO = leave-one-out):**

| Point set (anchors) | Scored points | r = 1 (true geometry) | Fitted r | a, α at fitted r | Mean distance | LOO, r = 1 | LOO, r fitted |
|---|---|---|---|---|---|---|---|
| Thesis set, `opt_points_25_09_23_23_01` (vertical-bar upper, circle bottom) | 4 | 5.40 | **0.62** | 4.9, 1.42 | 1.93 | 6.87 | 2.88 |
| A: xlsx right table (triangle top, vertical-bar bottom: her winner anchors for this set) | 4 | 5.01 | **0.61** | 3.3, 1.61 | 2.03 | 5.53 | 4.61 |
| B: both xlsx tables, 10 points (same anchors) | 8 | 5.06 | **0.52** | 1.3, 0.35 | 2.49 | 5.32 | 4.05 |

**What this shows:**
- **r is close to Joanna's 0.54 in every set** (0.52–0.62). The profile has a broad minimum at about 0.45–0.7 and rises steeply towards r = 1.
- **The stretch also improves the leave-one-out error in every set,** so it isn't only absorbing noise.
- **a and α are poorly determined.** They drift (a up to 4.9, α 0.35–1.6), because all the points lie within 1.25° of fixation. Only r is stable.
- **Sanity check:** her own values written as a stretch (r 0.54, a = a_joanna / 0.77) give 2.82 px on the thesis set, against her 2.56. The difference comes from her code details: rounding, the grid's small offsets, and her `trans_cortex` variant.

### Adding 20/11/18

20/11/18 uses Joanna's `opt_points_8_5_22.mat`: 8 points at the corners and ends of letters. Her winner anchors were (49,18) and (15,18), again on one vertical line. The Python re-run of her pipeline gives exactly her 1.520 px.

The camera sits differently on each day, so each day keeps its own anchors. Only a, α and r are shared between days.

| Fit | Scored points | r | a | α (θ/α) | Mean distance | Per day (10/07 / 20/11) |
|---|---|---|---|---|---|---|
| 20/11/18 alone, r = 1 | 6 | 1 | 0.23 | 1.22 | **18.0 px** | — |
| 20/11/18 alone, r fitted | 6 | **0.536** | 1.12 | 1.54 | 1.72 px (leave-one-out 2.39) | — |
| Joint: B + 20/11/18 | 14 | **0.545** | 1.54 | 1.46 | 2.63 px | 3.17 / 1.92 |
| Joint: thesis set + 20/11/18 | 10 | **0.539** | 1.43 | 1.50 | 2.27 px | 2.95 / 1.82 |

**What this shows:**
- **20/11/18 pins r down sharply, at 0.536.** The letter points reach 1.9° horizontally, so a wrong r costs a lot: with r = 1 the mean error is 18 px. The other mirror is worse still, so the mirror choice isn't the cause.
- **The two days agree.** Fitted jointly, r = 0.54–0.55, essentially Joanna's grid ratio of 0.54.
- **a and α become stable once 20/11/18 is included:** a ≈ 1.4–1.5 and α ≈ 1.5. Her α for 20/11/18 was also 1.5.

### 20/11/18 with all xlsx points (`fit_201118_all.py`)

`opt_points_8_5_22.mat` is a subset of the lower table of `VSDI points/201118/points Gandalf.xlsx` (rows 25–32; the
"(AutoRecovered)" copy is identical). All 8 .mat points match it exactly, and they mix two sessions: 4 from session A
(including both anchors) and 4 from session D (letters shifted by −7 rows, +14 cols). The table lists 15 pairs: 7 from
A, 8 from D; position (25, 66) of session D has two camera values, "F·" (54, 63) and "D·" (60, 60). One camera for both
sessions is assumed (as Joanna did), with her anchors (49,18) and (15,18) unless stated.

| Point set | Scored | r | a | α (θ/α) | Mean distance | LOO | r = 1 |
|---|---|---|---|---|---|---|---|
| Joanna's 8 (.mat) | 6 | **0.536** | 1.12 | 1.54 | 1.72 px | 2.39 | 18.0 |
| All 15 | 13 | **0.533** | 1.08 | 1.51 | 3.16 px | 3.62 | 18.6 |
| 14, without D· | 12 | **0.537** | 1.10 | 1.54 | 3.00 px | 3.48 | 18.0 |
| 14, without F· | 12 | 0.531 | 1.01 | 1.43 | 3.23 px | 3.77 | 17.9 |
| Session A only | 5 | 0.474 | 1.47 | 1.48 | 1.69 px | 2.74 | 16.8 |
| Session D only (D anchors (42,32), (8,32)) | 6 | 0.482 | 0.66 | 1.02 | 3.05 px | 4.46 | 11.3 |
| A + D, each its own camera | 11 | 0.456 | 0.95 | 1.15 | 2.95 px (A 2.42 / D 3.39) | – | – |
| Joint: 10/07 thesis set + all 15 | 4 + 13 | **0.532** | 1.27 | 1.54 | 3.16 px (3.12 / 3.17) | – | – |
| Joint: 10/07 thesis set + 14 without D· | 4 + 12 | **0.533** | 1.25 | 1.56 | 3.05 px (3.15 / 3.01) | – | – |

**What this shows:**
- **r, a and α hardly change with all 15 points** (0.533, 1.08, 1.51 vs 0.536, 1.12, 1.54), and r = 1 stays ~18 px.
- **The error roughly doubles (1.72 → 3.16 px)** because the added points include the worst ones: session D
  (8,32) 7.8 px and (42,32) 6.4 px (the left letter corners, i.e. the session-D twins of the anchors), "D·" 4.9 px and
  session A (15,52) 4.8 px. Joanna's 8 leave all of these out, so her 1.52 px is on a selected, well-fitting subset.
- Of the two camera values at (25, 66) in session D, F· (54, 63) fits; D· (60, 60) is the outlier.
- Giving session D its own camera does not clearly help (2 more points become anchors, r drops to 0.46); the two
  cameras differ by ~8% in scale and ~4° in rotation, which could be a small camera move or point-picking noise.
- Jointly with 10/07/18: r 0.53, α 1.55, a ≈ 1.25, in line with the earlier joint fit (0.54, 1.50, 1.43).
