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
  - *2 anchors (Joanna)*: shift + rotation + scale solved **exactly** through two chosen landmarks
    (`c1 = (u_B − u_A)/(w_B − w_A)`, `c0 = u_A − c1·w_A`); the other landmarks only score the fit with her "RMSD" =
    mean distance over the non-anchor landmarks. Default anchors = her winning pair per day (10/07/18: vertical-bar
    upper + circle bottom; 20/11/18: stimulus px (49,18) + (15,18)), found among the manual landmarks.
- **Stretched Schira, r**: the visual field is read as `(r·x, y)` before the Schira equation (matrix `[[r,0],[0,1]]`).
  r = 1 is the true geometry. Fitted on 10/07/18 + 20/11/18 with her anchors, r ≈ 0.54, the same factor her
  retino_mat grid has built in (0.415/0.77); don't combine r < 1 with the retino_mat grid (squeeze applied twice).
  A uniform scale of the visual field is the same as changing a, so only the ratio r is identifiable.
  See `registration_revised/README.md` and `registration_revised/fit_stretch.py`.
- **Outer loop** ("Optimise"): multi-start Nelder–Mead over a, α (k in rigid mode) and optionally r; score = RMSE, or the
  mean non-anchor distance in anchor mode.
- **Joint fit over days** (sidebar expander): our suggested fit, one a, α, r shared by the chosen sessions, each
  registered exactly through its day's two anchors on the manual landmarks; score = mean distance over all non-anchor
  landmarks. Bounds a 0.05–5, α 0.3–4 (P = θ/α), r 0.15–3. a is loosely determined (≈ 0.9–1.6 fit about equally
  well); α and r are stable.
- **Model** (top of "Schira parameters"): *Set by hand*, or *Our joint fit (stretched Schira)* = the fit_stretch.py
  result (a 1.43, α 1.50 with P = θ/α, r 0.539, 2-anchor registration, 35 px/°). While the joint fit is selected its
  settings are locked; any fit result (Optimise, joint fit, thesis-line fits) switches back to *Set by hand*.
- Camera coordinates are 0-based (row, col) on the 100 × 100 map; RMSE in px and mm (1 px = 0.17 mm).
- The registration is fitted per session on all of that session's selected landmarks; the overlay shows the selected stimulus.

## Landmarks

- **auto**: response peak for points and circles (→ stimulus centre), two strongest peaks for 1° bars (→ the bar ends;
  which peak is which end is decided by the fit). Quality = peak / noise SD; filter with the slider.
- **manual**: Joanna's points, 20/11/18 (8) and 10/07/18 (10). They are day-level, so they appear on every session of that day.
  The 10/07/18 horizontal-bar right point uses camera (x, y) = (51, 48) as in her .mat files and the thesis (the xlsx has
  x and y swapped).
- Untick individual landmarks in the "Landmarks" table to exclude them (e.g. peaks that sit in V2 at the top of the chamber).

## Known limits

- Most non-letter stimuli are centred at the same position (0.6°, −0.75°), so a session has few independent
  landmarks. With ≤ 4 points the similarity registration fits exactly; the dashboard warns when the RMSE is not informative.
- Not detected yet: triangle corners, the 0.95° circle around fixation, letter corners.
- "Overlap r" (Pearson between the response and the blurred mapped stimulus) is a landmark-free secondary score.


## Choosing what to show: shape → VSD

- **Shape type / Shape**: every distinct stimulus image: the 55 images of the stimulus library
  (`Data/VSD_Encoder_01/stimuli/gandalf/library/`, catalogue: `stimulus_catalog_gandalf.pdf` in the same folder), plus
  images only the dashboard has (`_retired/stimuli/<session>/<condition>__<tag>.png`, and any cached catalog image outside
  the library), de-duplicated by pixels. Today that adds one: the 0.8° triangle of 10/07/18 as drawn by the pre-fix renderer.
  The label shows how many VSD recordings exist for that image.
- **VSD recording of this shape**: the session · condition recordings that showed exactly this image. The chosen shape is
  mapped through Schira with that session's registration and drawn on that session's response.
- **Compare on all VSDs**: the same mapped shape on every recording of it, each with its own session's registration
  (sidebar landmark filters apply; rows unticked in the Landmarks table only affect the selected session).
- Shapes without a recording (the two blanks: blank trials are not in the processed h5; the six 23/05/18 d circles: no
  processed h5 for 230518d) show the image and a note.

## Joanna's grids and thesis Fig. 12

- `make_thesis_fig12.py` (run once, ~1 min) writes `stimuli/thesis_fig12/`: the Cartesian grid (white lines every 0.5°)
  and the polar grid (rings every 0.5°, rays every 10°), 0–3°, white on grey 130 as in `create_cart_stim.m` /
  `create_polar_stim.m`, the thesis and Joanna's saved stimulus PNGs (pixel-identical to the MATLAB code). They were never
  shown to the monkey, so they stay out of the stimulus library and appear only as dashboard shapes ("grid cartesian",
  "grid polar"), which you can lay on any VSD recording.
- With a recording from 20/11/18 or 10/07/18, the dashboard adds a **Thesis Fig. 12** section: the thesis plate
  (registered onto `BV_resized.png` by its blood vessels), Joanna's prediction (her pipeline re-run: full MATLAB grid,
  `retino_mat` degree grid, her session a/α and 2-point registration), the dashboard's own mapping (sidebar a, α, k + this
  session's registration), all three overlaid, and their distances to the thesis lines. "Use Joanna's values" copies her
  a, α (converted to the current convention) and k into the sidebar.
- Her pipeline reads stimulus pixels at about 84 px/° horizontally and 45 px/° vertically (`retino_mat`), the dashboard at
  35 px/°, so the same a and α do not give the same map (`shira_revised/schira_audit/README.md`).
- **Background to draw the grid on**: the thesis Fig. 12 plate itself (A plate for the polar grid, B plate for the
  Cartesian grid; one entry per session of that day, whose landmarks register the grid), blank-trial recordings, or any
  stimulus recording. Blank trials come from `session_<s>_blank.h5` (made by `scripts/extract_blank_trials_h5.py` from the
  raw condsAN .mat `blankAN`); today only 24/07/18 c and 27/06/18 b have them, not 20/11/18 or 10/07/18 (their raw .mat
  files are not on this machine). `prepare_data.py` adds every blank file it finds as `<session>|blank`.
- **Stimulus px → degrees** (sidebar): 35 px/° (true) or Joanna's retino_mat grid (thesis), for the stimulus, the
  landmarks and the meridians.
- **Camera for the grid** (thesis days): the session's landmark registration, or a similarity camera fitted live to
  the thesis lines for the current a, α, k (only the Schira shape is compared).
- **Fit a, α to the thesis lines** (two buttons: landmark camera / free camera), from 4 starts, Nelder-Mead on the
  symmetric capped chamfer distance. Offline result, free camera (camera px): retino_mat grid B(ii) 0.57 (a 0.734,
  α 1.461 thesis), A(ii) 0.56 (0.730, 1.466), B(iii) 0.70 (0.939, 0.995), A(iii) 0.98 (0.959, 0.994); 35 px/° 1.2–2.0 px
  with unstable a, α (no value reproduces the thesis); landmark camera 0.9–2.4 px. k has no effect with a similarity camera.
- QC: `stimuli/thesis_fig12/check_thesis_fig12.png` (thesis lines vs the re-run; 0.8–1.8 camera px).
