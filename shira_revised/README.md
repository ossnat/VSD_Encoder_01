# shira_revised

Side-track: re-examine Gandalf retinotopic mapping / Schira alignment against
the lab's legacy MATLAB pipeline (Joanna / Amit), then re-merge into the main
encoding stack only once overlays and landmark RMSD match prior student work.

**Do not** treat this as the production encoding path. Main-project
`src/retinotopy/` and `src/schira_encoding/` stay untouched until we confirm
parity.

## Why this exists

Feedback: our Schira overlays are not aligned with previous students' results
on the same Gandalf datasets (`201118` letters, `100718` bars). The live legacy
kernel is **not** `retinotopic_model_joanna.m` (incomplete draft). It is:

```text
legacy_code/.../Gandalf/Code/Amit old code/trans_cortex.m
  → trans_and_registration.m  (2-point nonreflective similarity → 100×100)
```

## Headline mismatches vs current Python

| Topic | Legacy (live) | Current repo (`src/retinotopy`) |
|-------|---------------|----------------------------------|
| α on polar angle | **`T = θ / α`** | **`P = α · θ`** |
| Shear angle into `exp(i·)` | `T · sech(T)^c` | `P · sech(P)^(…)` |
| VF axes into Schira | **`abs(x), abs(y)`** (Q1) | signed LRQ (`y < 0`) |
| `eccWidth` | `0.7609` | `0.76` |
| Stim canvas | `301×512` @ 35 ppd | `210×210` @ 35 ppd |
| Cortex → chamber | 2-pt similarity on integer `xcoord/ycoord` + low-hem flips | free/rigid similarity on continuous `w` landmarks |
| Reported RMSD | ~**1.5 px** (Joanna winners) | ~**9–10 px** (phase-0 landmark fits) |

Details: [`notes/LEGACY_VS_CURRENT.md`](notes/LEGACY_VS_CURRENT.md).  
Visual test catalog: [`notes/EXPERIMENTS_AND_VISUAL_TESTS.md`](notes/EXPERIMENTS_AND_VISUAL_TESTS.md).

## Layout

```text
shira_revised/
  src/           # faithful legacy ports + comparison helpers
  tests/         # numeric / convention tests (no MATLAB required)
  scripts/       # visual diagnostics writing to artifacts/
  configs/       # paths + golden params from Opt_results
  notes/         # analysis write-ups
  artifacts/     # generated figures (gitignored content OK)
```

## Quick start

```bash
# From repo root
scripts/py -m pytest shira_revised/tests -q

# Schira-only: same legacy stimuli, VF deg ticks, legacy OM vs repo (no registration)
scripts/py shira_revised/scripts/04_schira_only_stim_compare.py
# → shira_revised/artifacts/schira_only_stim_compare.png

# α-convention divergence plot (multiply vs divide)
scripts/py shira_revised/scripts/01_compare_alpha_conventions.py

# Letter ink overlay vs legacy Session PNG (needs local legacy tree)
scripts/py shira_revised/scripts/02_overlay_letter_vs_legacy_png.py --letter A --session a

# Fig. 12 style: polar/cart grids → chamber via legacy 2-pt registration
scripts/py shira_revised/scripts/10_legacy_fig12_chamber.py
# → artifacts/legacy_fig12_chamber_overlays.png
```

## Success criteria (before re-merge)

1. Pointwise `w` / chamber RMSD matches Joanna winner row (~1.5 px mean) under the
   **legacy** α / abs-VF / 2-pt registration path.
2. Visual letter/bar overlays match `Gandalf/images/{201118,100718}/Session */`.
3. Cart/polar grid fingerprints match `Results Data/28_03_2022 results/`.
4. Document which convention (paper `αθ` vs lab `θ/α`) we keep for encoding,
   and port only the confirmed registration path into main.
