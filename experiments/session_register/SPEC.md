# Session camera registration (vasculature) — implementation spec

**Status:** spec only. Do **not** mix this with Schira landmark fitting (`src/retinotopy/register.py`, `CorticalAffine`).

**Goal:** Within one monkey, register a **new session’s VSD camera** onto an **anchor session** (first: Gandalf `100718a`) using **blood vessels**. Then every later stimulus on the new day can reuse the **same Schira + 100718 camera YAML**; only this extra 2D warp changes.

Hand-off for a second agent: implement this module, then a thin wrapper that applies it after the existing Schira overlay.

---

## 0. What is already done (do not redo)

Forward / backward for **one** session (anchor):

1. Stimulus VF (deg / 35 px/deg canvas) → Schira \(w\) (`SchiraParams` in `configs/schira/sets.yaml`).
2. \(w\) ↔ VSD pixels via `CorticalAffine` (`src/retinotopy/affine.py`: `w_to_pixel`, `pixel_to_w`).
3. 100718 session map: `experiments/schira2007/phase_0_register/affine_fit__100718__gandalf_rigid.yaml` (also copied under `sets.yaml` → `100718.affine`).

That affine lives in **anchor camera coordinates** (100×100, row 0 = top, `reshape(H,W)`).

A new session (`201118a`, `240718a`, …) is a **different camera pose** of the same cortex. Vessels should match; evoked blobs need not.

---

## 1. What this module computes

A **2D image transform** \(T\):

- **Moving** = new session (source)
- **Fixed** = anchor session (target), e.g. `100718a`

\(T\) maps a pixel in the **moving** image into the **fixed** image.

- Overlay Schira ink (defined on the anchor) onto a new session: apply \(T^{-1}\) to the ink.
- Put a new-session VSD map into Schira \(w\): `pixel_moving → T(pixel) → CorticalAffine.pixel_to_w`.

**Allowed transform (v1):** rigid — rotation + translation only (same philosophy as the session Schira camera: no extra stretch). Optional later: similarity (one isotropic scale) if the chamber zoom changed; **no** elastic / B-spline in v1.

**Scope v1:** same `monkey` only (Gandalf first). Refuse mixed-monkey pairs.

---

## 2. Suggested layout

```
src/session_register/
  __init__.py
  config.py          # dataclass from YAML / CLI
  load.py            # mean maps from H5 (reuse trial_frames)
  vessels.py         # “bleach” / vessel emphasis
  fit.py             # call registration backend
  transform.py       # apply / invert; YAML save-load
  plot.py            # QC: fixed | moving | overlay | checkerboard
experiments/session_register/
  run_register.py    # CLI
  configs/gandalf_100718a_anchor.yaml
```

Do **not** put this under `src/retinotopy/`. Name clash with landmark `register.py`.

Reuse:

- `src.data.trial_frames.load_h5_mean_frame`
- `src.data.splits.load_trial_table`
- `src.paths.resolve_data_path`, `project_root`
- `spatial_size: [100, 100]` from `configs/default.yaml`
- Half-open frame windows like `configs/windows/*.yaml`

---

## 3. Building the two images

For **each** of fixed and moving:

1. Resolve H5 via split/index CSV (`date` = `h5_session`, e.g. `100718a`).
2. Average frames `[start_frame, end_frame)` (Python slice). **Default for this task: 35–40** → `start_frame: 35`, `end_frame: 40`.
3. Mean across **several trials** (all local trials for that session, or a cap). Condition should not matter for vessels; pooling all conditions is OK. Prefer `normalization: none` (raw intensity). Baseline-zscore **removes** static vessel shadows — bad for this.
4. Reshape to `(100, 100)`. Save `mean_vessel__{session}.npy` so a failed later step does not re-average H5.

**Config knobs (required):**

| Key | Default | Why |
|---|---|---|
| `monkey` | `gandalf` | Within-animal only |
| `fixed_session` | `100718a` | Anchor camera |
| `moving_session` | (CLI) | Session to bring onto the anchor |
| `start_frame` | `35` | User start |
| `end_frame` | `40` | Half-open; frames 35–39 |
| `spatial_size` | `[100, 100]` | Match encoding maps |
| `normalization` | `none` | Keep anatomy |
| `n_trials` | `all` (or cap e.g. 40) | Stable mean |
| `seed` | `17` | If subsampling trials |

**Config knobs (recommended):**

| Key | Default | Why |
|---|---|---|
| `conditions` | `all` | Or a list `condAN1,…`; vessels should be similar |
| `invert_intensity` | `true` | Vessels often dark on VSD; some metrics want bright lines |
| `vessel_method` | `clip_tophat` | See §4 |
| `clip_percentiles` | `[1, 99]` then optional hard saturate | “Bleach” until only vessels |
| `transform_type` | `rigid` | `translation` / `rigid` / `similarity` |
| `metric` | `mattes_mi` or `ncc` | Mutual information vs correlation |
| `mask_border_px` | `4` | Ignore chamber edge |
| `max_rotation_deg` | `25` | Bound the search |
| `max_translation_px` | `30` | Bound the search |
| `output_dir` | `experiments/session_register/runs/` | YAML + QC PNGs |

**Note for the implementer:** evoked 35–40 is what the user asked for. Vessels are often **clearer on pre-stimulus / baseline frames** (e.g. `[5, 26)`). Keep the window configurable; add a one-line comment in the YAML. Do not change the default unless the user says so.

---

## 4. “Bleach until only blood vessels”

Not dye photobleaching. **Saturate / filter** so the map is dominated by static vasculature:

Suggested v1 pipeline (all configurable):

1. Optional invert so vessels are bright.
2. Contrast-limited adaptive histogram equalization (CLAHE) **or** morphological white/black tophat.
3. Percentile clip (e.g. 1–99), then optionally clip again harder (e.g. 20–80) so gray cortex saturates and only vessel ridges remain.
4. Optional Frangi / Sato vesselness (`skimage.filters`).
5. Gaussian smooth \(\sigma \approx 0.5\)–1 px to reduce speckle.

Save **before** and **after** PNGs. Registration must run on the **vessel** image, not the raw evoked rainbow map.

---

## 5. Registration backend

Pick **one** primary library (document it); keep the fit function swappable.

Good fits for 100×100 rigid:

- **SimpleITK** — `Euler2DTransform` + Mattes MI (robust, standard).
- **scikit-image** — `phase_cross_correlation` (translation) + small rotation grid; lighter dependency.
- **pystackreg** — ImageJ StackReg-style rigid; common in microscopy.

v1: **rigid** (rotation + translation). Initialize with phase-correlation translation, then refine rotation.

Output parameters (pixel convention **must** match `imshow` / `CorticalAffine`: `row` down, `col` right):

```yaml
fixed_session: 100718a
moving_session: 240718a
monkey: gandalf
transform_type: rigid
rotation_deg: ...
translation_xy: [dx_col, dy_row]   # moving → fixed
# if similarity:
scale: 1.0
inverse: { rotation_deg: ..., translation_xy: [...] }
rmsd_px: ...          # vessel-image residual after warp, if available
metric_value: ...
frame_window: [35, 40]
n_trials_fixed: ...
n_trials_moving: ...
```

Must support **apply** (warp moving → fixed grid) and **inverse apply** (fixed → moving grid), bilinear, `cval` for outside.

---

## 6. QC (required before calling it done)

One figure per pair:

1. Fixed vessel image  
2. Moving vessel image  
3. Moving warped onto fixed, transparent overlay  
4. Checkerboard fixed vs warped moving  

Print / YAML: translation (px), rotation (deg), metric. Fail the job if translation or rotation hits the configured max (likely a bad pair).

---

## 7. How this plugs into Schira later (do not implement Schira here)

Keep Schira YAML **unchanged** on the anchor.

For a stimulus on `moving_session`:

```
VF → Schira w → CorticalAffine.w_to_pixel   # pixels in 100718a
                 → T^{-1}                     # pixels in moving camera
                 → overlay on that day’s VSD
```

The session-register module should expose:

```python
apply_to_points(rows, cols, *, inverse: bool) -> (rows, cols)
apply_to_map(image, *, inverse: bool) -> image
```

Composition with `CorticalAffine` can be a 10-line helper in a **later** PR.

---

## 8. Tests (no full H5 required for unit tests)

- YAML roundtrip of the transform.
- Synthetic: take a vessel-like 100×100 image, rotate 8° and shift (5, −3), recover within ~0.5 px / 0.5°.
- Inverse: \(T^{-1}(T(p)) \approx p\).
- `monkey` mismatch raises.
- Frame window: `end_frame` is exclusive (`35, 40` → 5 frames).

Optional integration test (skip if H5 missing): `100718a` vs `100718b` should be a **small** warp (same day, two letters).

---

## 9. Efficiency

- Cache mean maps (`npy` + yaml of trial ids). Do not re-read all H5 trials to replot QC.
- Registration itself is cheap (100×100). The expensive part is averaging trials — skip-existing if trial ids match.

---

## 10. Out of scope (v1)

- Cross-monkey.
- Elastic registration.
- Re-fitting Schira \(a,\alpha,k\).
- Changing `CorticalAffine` for 100718.
- Git commit / push.

---

## 11. First CLI to implement

```bash
scripts/py experiments/session_register/run_register.py \
  --config experiments/session_register/configs/gandalf_100718a_anchor.yaml \
  --moving-session 240718a
```

Default frames `[35, 40)`, monkey `gandalf`, fixed `100718a`, rigid, vessel clip/tophat.
