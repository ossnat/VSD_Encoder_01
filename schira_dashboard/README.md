# Schira dashboard

Set Schira's **a, α, k**; the registration onto the camera is fitted automatically (inner loop); see the mapped
stimulus on top of the measured VSD response, with the landmark RMSE. An **Optimise** button runs the outer loop
(nested optimisation of a, α, and k in rigid mode).

## Run (from the repo root)

```bash
scripts/py -m pip install streamlit                         # once
scripts/py schira_dashboard/prepare_data.py                 # once: builds schira_dashboard/cache/ (~1 min)
scripts/py -m streamlit run schira_dashboard/schira_dash_app.py
```
A browser tab opens. Tests: `cd schira_dashboard && ../scripts/py test_schira_geometry.py`.

## Files

| file | what |
|---|---|
| `prepare_data.py` | response maps (frames 35–46 minus 5–25), stimulus catalog, landmark table → `cache/` |
| `schira_geometry.py` | Schira (repo `src/retinotopy/schira.py`), registration fits, nested optimiser, stimulus ink mapping |
| `schira_dash_app.py` | the Streamlit dashboard |
| `test_schira_geometry.py` | checks (k invariance, conventions, reproduces the earlier 20/11/18 fit of 7.9 px) |
| `figures/` | figures saved from the dashboard |

## Conventions

- **α**: switch between `P = α·θ` (repo, Ayzenshtat 2012) and `P = θ/α` (thesis / legacy MATLAB; thesis α 1.5 = repo 0.667).
- **Registration** `u = c1·w + c0` (optionally mirrored):
  - *similarity*: shift + rotation + scale, as in the thesis and Ayzenshtat. **k has no effect** (the scale absorbs it).
  - *rigid*: shift + rotation at a fixed 0.17 mm per camera px; w is read in mm, so **k is the cortical scale** and is fitted.
- Camera coordinates are 0-based (row, col) on the 100 × 100 map; RMSE in px and mm (1 px = 0.17 mm).
- The registration is fitted per session on all of that session's selected landmarks; the overlay shows the selected stimulus.

## Landmarks

- **auto**: response peak for points and circles (→ stimulus centre), two strongest peaks for 1° bars (→ the bar ends;
  which peak is which end is decided by the fit). Quality = peak / noise SD; filter with the slider.
- **manual**: Joanna's points, 20/11/18 (8) and 10/07/18 (10). They are day-level, so they appear on every session of that day.
- Untick individual landmarks in the "Landmarks" table to exclude them (e.g. peaks that sit in V2 at the top of the chamber).

## Known limits

- Most non-letter stimuli are centred at the same position (0.6°, −0.75°), so a session has few independent
  landmarks. With ≤ 4 points the similarity registration fits exactly; the dashboard warns when the RMSE is not informative.
- Not detected yet: triangle corners, the 0.95° circle around fixation, letter corners.
- "Overlap r" (Pearson between the response and the blurred mapped stimulus) is a landmark-free secondary score.

## Stimulus images

- Default: the repo's rendered images, `Data/VSD_Encoder_01/stimuli/gandalf/images/<session>/<condition>.png`
  (made by `scripts/01b_build_stimulus_images.py` with `assume_size_is_diameter: true`); the path comes from the
  encoding manifest (`Data/VSD_Encoder_01/encoding_pairs/gandalf/win_0035_0046/manifest.parquet`).
- Radius readings (`add_size_variants.py`, run automatically by `prepare_data.py`): 100718a/b triangle 0.4 and circle 0.3,
  labelled "radius" in the catalog, are also drawn with diameter 0.8° and 0.6° into `schira_dashboard/stimuli/`.
  They appear as extra entries in the Stimulus list, marked "[drawn Ø …°, radius reading]"; they share the response map
  of the same condition.
