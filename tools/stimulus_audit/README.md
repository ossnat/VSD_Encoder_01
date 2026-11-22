# Stimulus image audit (Gandalf, Nov 15 2026)

Read-only check of `Data/VSD_Encoder_01/stimuli/gandalf/images/` (the single image store; 90 manifest rows) against the
lab's requirements. Script: `audit_stimuli.py` → `audit_gandalf.csv` (one row per stimulus) and `audit_gandalf_sheet.png`
(zoomed tiles: red + = catalog position, red dashed = catalog circle as labelled, blue = legacy letter placement).
Nothing was changed.

## Passes
- Straight lines are straight and unbroken: the triangle's vertical side (one column), all bars (identical rows/columns),
  letter strokes; every contour is one connected piece.
- Triangles are equilateral within pixel quantisation (sides 12.5/12.0/12.5 px), inscribed in a circle centred on the
  catalog position, tip to the right.
- Letters are the pre-rendered bitmaps (`Data/EncoderData/letters_stimuli/*.bmp`), pasted pixel-for-pixel (35 x 35, no resampling).

## Problems
1. **Letters are BLACK; legacy and catalog say WHITE.** The bmp palette maps index 130 to black (0,0,0); the repo converts
   to RGB and keeps black. Legacy `create_letter_stim.m` uses the index (130 -> 1 = white) on a 130/255 grey.
2. **Letters 1 px to the right in 201118a and 201118d** vs legacy placement (rounding: legacy `round(35x)-17`, MATLAB rounds
   .5 up; Python rounds to even). 201118c matches exactly.
3. **Background grey is 128 everywhere; the experiment's blank was 186** (ContrastCurve catalog: "Blank (186)"; letter bmp
   background 188). Consequence: the 'black' contrast circles of 130618d / 230518d (rgb 158–182) are drawn LIGHTER than
   the background, i.e. with the wrong sign; all contrasts are wrong in size.
4. **130618c/d cond2 polarity swapped:** catalog says c cond2 = 100 (0) (black) and d cond2 = 100 (249) (white);
   the manifest/images have the opposite.
5. **Radius vs diameter:** the renderer always reads size as diameter.
   - 100718a/b triangle "0.4 radius" and circle "0.3 radius" are drawn half size (diameter 0.4 / 0.3 instead of 0.8 / 0.6).
   - 240718a/b/c circle "0.95 radius" (around fixation) drawn with radius 0.475 instead of 0.95.
   - 290518a sizes are "diameter/radius?" in the catalog – unresolved.
   - Contrast circles: "r=0.4" (23/5) and "r=0.15" (13/6) are correctly drawn as diameter 0.8 / 0.3.
6. **240718a/b/c condAN6: two manifest rows (Blank and black bar horizontal) share one file**; the file holds the bar.
7. Minor: points and contour circles are ~1 px larger than nominal and 1 px non-round (PIL inclusive bounding box);
   bars are 36 x 2 px instead of 35 x 1 px.

## Fixes applied (Nov 15 2026, decisions by the user + PI)

Backup of the previous images/manifest: `Data/VSD_Encoder_01/stimuli/gandalf/_backup_before_fix_2026-11-15/`.
Re-rendered with `scripts/01b_build_stimulus_images.py --overwrite` after these source changes:

| item | decision | change |
|---|---|---|
| 1 letter colour | leave as is | none |
| 2 background | use the Excel value where it is given | contrast-curve sessions (23/05, 13/06) on 186 (`Blank (186)`); others unchanged (128) |
| 3 13/06 cond2 | follow the Excel | c cond2 = black (0), d cond2 = white (249); removed the polarity override in `contrast_letters_catalog.py` |
| 4 radius vs diameter | re-render radius items | `catalog.py`: '… radius' → diameter = 2×; 'd/r?' read as diameter (majority) with a note; manifest column `size_note` |
| 5 24/07 cond6 | Excel lists Blank and the horizontal bar as cond6; the recordings have condAN7 and no condAN6 | condAN6 = blank (empty), horizontal bar = condAN7 |
| 6 letter 1-px offset | letters come from the legacy bitmaps | none |
| 7 bars | exact size | `render.py`: bars 35 × 1 px; triangle's two left vertices share one x (vertical side was split over two columns at 0.8°) |

Result: every check passes except the letter colour (left on purpose). Catalogue of all distinct stimuli:
`Data/VSD_Encoder_01/stimuli/gandalf/stimulus_catalog_gandalf.pdf` (`make_stimulus_catalog_pdf.py`).
Downstream still to redo: CNN features / encoding pairs for the changed images (and 24/07 condAN7 is new).

## Stimulus library (Nov 16 2026)

`Data/VSD_Encoder_01/stimuli/gandalf/`: `library/<stimulus_uid>.png` (55 images), `library.parquet`, `lookup.parquet`
(97 session-conditions → uid), `manifest.parquet` (as before + `stimulus_uid`, image_path → library),
`catalog_check_gandalf.csv` (Excel vs image), `stimulus_catalog_gandalf.pdf`.
Build: `scripts/01b_build_stimulus_images.py --overwrite`; check: `scripts/01d_check_stimuli_against_catalog.py`.
Blank conditions of the contrast/letters sessions are now in the library too (Excel lists them).
Migration checks: `check_library_migration.py` (pixel identity, uniqueness, coverage, stimulus_id changes);
determinism (rebuild twice → byte-identical); repo tests (`tests/test_stimulus_library.py` new,
`tests/test_contrast_letters.py` updated); dashboard smoke test (`tests/test_dashboard_app.py`).
Encoding-pair manifests (7 windows) updated: image_path → library, corrected text/colour/size, + 24/07 condAN7 rows;
backup in `Data/VSD_Encoder_01/encoding_pairs/_backup_before_library_2026-11-16/`.
Old per-session `images/` folder still in place (950 experiment parquet files point to it).
