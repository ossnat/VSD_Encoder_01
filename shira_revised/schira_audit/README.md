# Schira audit + thesis Fig. 12 with Schira alone (Oct 17 2026)

## 1. Code audit: `audit_schira.py` (run: `PYTHONPATH=../.. python audit_schira.py`), 17/17 tests pass

| check | result |
|---|---|
| repo `forward_schira` vs legacy formula (trans_cortex.m logic) | identical to 1e-15 when alpha_repo = 1/alpha_legacy |
| repo default S1 0.76 vs legacy 0.7609 | 5e-5 in w, negligible |
| lower field → Im(w) ≤ 0 | correct sign |
| inverse(forward) on the 0–6° quadrant | 5e-15 deg, nothing lost |
| CorticalAffine pixel→w→pixel | 4e-14 px; a pure similarity (no hidden anisotropy) |
| LUT (set 100718) inverts the forward map | 7e-5 deg |
| deg → canvas px | 35 px/deg on both axes |
| renderer | point at (0.6, −0.75)° lands at row 26.0, col 20.5 (expected 26.25, 21: half-pixel, 0.015°); 1° circle is 36 × 36 px (round) |
| SessionTransform map vs points | agree to 0.3 px |
| h5 camera orientation | auto landmarks from the h5 data match Joanna's (row, col) within 2–5 px; a transpose would put them up to 31 px off |

Things that ARE wrong or uncertain (none of them changes shapes):
1. **`configs/schira/sets.yaml` set 201118 uses alpha 1.5 from the thesis, but the thesis value is in the divide
   convention (P = θ/α) and the repo multiplies (P = α·θ).** The repo equivalent is 1/1.5 = 0.67. Set 100718 (alpha 1.0) is unaffected.
2. **Circle and triangle size annotation is ambiguous.** The catalog calls the same `27June2018_Matt.tim` stimuli "0.3 diameter" on
   27/6 and "0.3 radius" on 10/7; the renderer assumes diameter. A factor of 2 in size, not shape.
3. **Coarse CNN layers have a half-cell offset.** `SchiraLUT.feature_sample_coords` uses row·(Hf/224), but F.interpolate(align_corners=False)
   and the conv grid put feature cell i at input (i+0.5)·224/Hf. This is negligible for block1_prepool (224×224, 0.03 px) but about 0.2° for 14×14 layers.

## 2. Fig. 12B(ii) with Schira alone at 35 px/deg: `fig12/`

Target: the white grid lines of the thesis panel (thesis page 30). They are moved to camera px with the two marker dots.
Stimulus: 3×3° grid, lines every 0.5° (create_cart_stim.m positions). Score: the mean distance (camera px, 1 px = 170 µm) from each
thesis line to the SAME stimulus line in the model. Line identity comes from the legacy fit, which explains the thesis to 0.31 px.

| model (all at 35 px/deg unless noted) | thesis→model | model→thesis |
|---|---|---|
| legacy px→deg grid, a 0.727, α 1.46 (how the figure was made) | 0.31 | 1.24 |
| **Schira only: a, α, k free + best similarity camera** | **4.44** | **3.81** |
| Schira only + S1, S2 also free | 2.90 | 3.58 |
| Schira only, mirrored camera | 12.3 | 5.3 |
| + x-stretch ρ before Schira (ρ = 0.544) | 0.30 | 1.24 |
| Schira + affine camera (tilt) | 1.15 | 1.88 |

- k only multiplies w, so any camera with a free scale absorbs it. Changing k from 0.1 to 1000 changes the mapped grid by < 1e-13 px.
  Ayzenshtat 2012 fits k in mm and then registers with translation, rotation and *scaling*, so there too k only sets the mm scale.
- If line identity is not enforced, a 35 ppd Schira fit can look close (0.80 / 2.15 px), but only by putting thesis lines on the
  wrong stimulus lines: the thesis "row 17" line lands on stimulus rows 70 and 105, and the anchors are 10 px off.
