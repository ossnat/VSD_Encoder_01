# Cleanup inventory: Schira + registration (branch `schira_and_registration_after_revision`)

Draft for approval, 2026-11-22. Nothing is deleted yet. Everything below is recoverable from tag `pre-cleanup-2026-11`.

Baseline before cleanup: `pytest tests` → 234 passed, 1 failed (`test_extract_stimulus.py::test_stimulus_map_path`,
pre-existing: needs `/data/stimuli/gandalf/lookup.parquet`), 1 skipped; `shira_revised/tests` → 14 passed;
`schira_dashboard/test_schira_geometry.py` and `test_dashboard_app.py` pass.

## What "currently used" means here

The current method is: Schira with horizontal stretch r → exact 2-anchor similarity registration (Joanna's anchors) →
score = mean non-anchor distance; fitted jointly over 10/07/18 + 20/11/18. It lives in two places:

```
schira_dashboard/schira_dash_app.py ─┬─ schira_dashboard/schira_geometry.py ── src/retinotopy/schira.py (Schira kernel)
                                     └─ schira_dashboard/thesis_fig12.py
registration_revised/fit_stretch.py ──── schira_dashboard/schira_geometry.py, registration_revised/points.py
registration_revised/summary_figs.py ─┐
schira_dashboard/make_thesis_fig12.py ┴─ shira_revised/src/legacy_{schira,trans_cortex,warp,registration,stimuli}.py
                                         shira_revised/configs/gandalf_legacy.yaml
                                         shira_revised/artifacts/thesis_fig12/thesis_page_029.png, _030.png  (gitignored!)
```

## A. KEEP as is

| path | why |
|---|---|
| `schira_dashboard/` app, `schira_geometry.py`, `thesis_fig12.py`, `make_thesis_fig12.py`, `prepare_data.py`, `cache/*.csv`, `stimuli/thesis_fig12/*.{csv,json}`, `README.md` | the dashboard |
| `schira_dashboard/_retired/stimuli/` (local, untracked) | the dashboard still shows these shapes |
| `registration_revised/` (all 6 scripts, README, `figs/*.csv`) | the stretched-Schira fits and talk figures |
| `src/retinotopy/` | Schira kernel for the dashboard **and** the encoding pipeline (all 8 modules are imported by scripts 10/22/26 or tests) |

## B. KEEP but MOVE (to get rid of the `shira_revised/` side-track folder)

| from | to | why |
|---|---|---|
| `shira_revised/src/legacy_schira.py`, `legacy_trans_cortex.py`, `legacy_warp.py`, `legacy_registration.py`, `legacy_stimuli.py` (+ `legacy_vf.py`, `compare.py`, used by their tests) | `src/legacy_joanna/` | Python ports of Joanna's MATLAB, needed by the thesis Fig. 12 re-run and `summary_figs.py` |
| `shira_revised/tests/test_legacy_schira_kernel.py`, `test_legacy_artifacts.py`, `test_alpha_convention_divergence.py` | `tests/` | they test those ports (14 pass); today pytest never runs them |
| `shira_revised/configs/gandalf_legacy.yaml`, `paths.yaml` | `src/legacy_joanna/configs/` | Joanna's session values used by `make_thesis_fig12.py` |
| `shira_revised/artifacts/thesis_fig12/thesis_page_029.png`, `thesis_page_030.png` (0.8 MB, **not in git**) | `schira_dashboard/stimuli/thesis_pages/`, tracked | inputs of `make_figs.py` and `make_thesis_fig12.py`; today a fresh clone cannot run them |
| `shira_revised/stimulus_audit/` (5 scripts + README) | `tools/stimulus_audit/` | stimulus-library tooling (not Schira); used by `scripts/02c_*.sh` |
| `shira_revised/notes/*.md` (4 notes), `shira_revised/schira_audit/README.md`, `shira_revised/README.md` | `docs/schira_history/` | the reasoning behind the α convention, legacy vs current, the audit; the dashboard links to the audit README |
| `schira_dashboard/test_schira_geometry.py`, `test_dashboard_app.py` | `tests/` (as pytest tests) | so one `pytest` run covers the dashboard |

Imports in the dashboard, `registration_revised/`, `scripts/02c_*.sh` and the READMEs are updated to the new paths.

## C. REMOVE (superseded experiments)

| path | files | last change | superseded by |
|---|---|---|---|
| `shira_revised/scripts/01–14_*.py` | 14 | 2026-10-12 | legacy-vs-current comparisons; conclusions are in the notes (moved, B) |
| `shira_revised/claude_landmark_fits/` | 13 | 2026-11-16 | `registration_revised/fit_stretch.py` |
| `shira_revised/geometry_v2/` (incl. 2 encoding configs, `loo_done.flag`) | 13 | 2026-11-16 | `registration_revised/` |
| `shira_revised/schira_audit/` scripts + `fig12/` (incl. a 1734-line `.pkl`) | 6 | 2026-11-16 | dashboard Fig. 12 section (README kept, B) |
| `shira_revised/ayzenshtat_2012/ayz_fig2e.py` | 1 | 2026-11-16 | one-off figure check |
| `shira_revised/src/plot_vf.py`, `vf_top_right.py` | 2 | 2026-10-12 | only used by the removed scripts |
| `shira_revised/__init__.py`, `src/__init__.py`, `artifacts/.gitignore` | 3 | | folder goes away (local `artifacts/` images stay on disk, untracked) |
| `experiments/schira2007/` (Phase 0) | 12 | 2026-09-08 | dashboard + `registration_revised/` |
| `experiments/retinotopic_map/README.md` | 1 | | plan only, never implemented here |
| `schira_dashboard/analyses/` (`fit_thesis_D.py`, `fit_triangle.py`, `run_tri.sh`) | 3 | 2026-11-16 | one-off fits, results in the READMEs |
| `schira_dashboard/_retired/add_size_variants.py` | 1 | 2026-11-22 | already retired |
| `configs/schira/session_100718.yaml`, `session_201118.yaml` | 2 | 2026-08-31 | moved to section D (used by `tests/test_schira_mapping.py`) |

Tests tied to removed code: `tests/test_schira_vf_scale.py` and `tests/test_gandalf_100718_landmarks.py` read files
from `experiments/schira2007/`. Either remove them with it, or keep the parts that test `src/retinotopy` only (recommended:
trim them).

About 75 files, ~10k lines of code.

## D. Schira encoding pipeline: switch its geometry to the five-step model (decided 2026-11-22)

Only the **geometry stage** of the encoding is old; the rest is model-independent and stays.

| part | what happens |
|---|---|
| LUT builder `src/schira_encoding/lut.py` (camera px → w → inverse Schira → degrees → stimulus px) | **rewritten** to invert the five steps: camera px → inverse 2-anchor similarity → w → inverse Schira (P = θ/α, a 1.43, α 1.50, k 1) → (x′, y) → x = x′/r (r 0.539) → stimulus px at 35 px/° |
| `configs/schira/sets.yaml` (Phase 0 affine, old a/α), `configs/schira/session_*.yaml` | **replaced** by one config with the joint-fit values and the per-day anchors (100718a; 201118 A and D) |
| `src/schira_encoding/geometry_fit.py`, `scripts/22_run_global_channel_schira_opt.py`, `23_compare_schira_opt_modes.py`, `25_schira_opt_summary_pdf.py`, `tests/test_schira_geometry_fit.py`, `configs/schira_encoding/geometry_fit.yaml` | **removed**: fitted the geometry through the encoding loss; the geometry now comes from the landmark fit |
| `src/retinotopy/` register / plotting / params pieces used only by Phase 0 and the old affine | **removed** after the switch (Schira kernel, inverse, visual-field and warp helpers stay) |
| `scripts/26_overlay_schira_ink_on_vsd.py` | ported to the new mapping (or replaced by the dashboard) |
| ridge / LOO / global-channel models, feature warping, `scripts/10–21, 24`, `src/session_register/` (vessel session-to-session alignment to the anchor session), `experiments/loo_encoding/`, `experiments/session_register/` | **kept**, they only consume the LUT |

No `OLD_`/`ARCV_` copies: the old code is in tag `pre-cleanup-2026-11`. Old encoding *results* under `Data/` and
`experiments/schira_encoding/` are not touched, so old vs new can still be compared.

After the switch the LUT, warped features and LOO results must be re-run (scripts 10 → 11 → 17/20 → 18/19/21).

Check before porting: the encoding renderer's degree → stimulus-pixel convention (`cartesian_deg_to_image_px`,
`RenderConfig`) must equal step 1 (fixation at the canvas corner, 35 px/°); a test will compare the new LUT with
`fit_stretch.predict` on Joanna's points (forward then inverse = identity).

## Also

- Rename nothing else, change no results; only moves, deletions and import fixes.
- After the cleanup: full `pytest` (expect the same single pre-existing failure, plus the moved tests passing), dashboard
  smoke test, `registration_revised/fit_stretch.py` reproduces r 0.536 / a 1.12 / α 1.54 on 20/11/18.
- Open a PR to `main`; delete this file before merging (or keep it as the record).
