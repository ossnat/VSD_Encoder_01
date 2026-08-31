# Schira 2007 Phase 0 (no training)

Three **separated** steps — do not mix registration into Schira-only, and do
not treat VSD overlays as part of the affine fit.

1. **Schira only** — warp stimulus ink into model cortical ``w = u + i v``.
2. **Affine registration only** — landmark pick + similarity fit (no overlays).
3. **Combine** Schira + affine for VSD overlays — uses `run_phase_0.py` and the
   affine block in `configs/schira/sets.yaml` (not the registration script).

## Step 1 — Schira only (working)

```bash
scripts/py experiments/schira2007/run_schira_only.py --set 201118
scripts/py experiments/schira2007/run_schira_only.py --set 201118 --overwrite
```

**Figures:** `experiments/schira2007/phase_0_schira/`  
One PNG per stimulus: ink pixels only, forward-mapped through Schira onto
Cartesian cortical ``(u, v)``. Montage: `all_stimuli_schira_only_montage.png`.

```bash
scripts/py experiments/schira2007/run_schira_only.py --set 201118 \
  --stimuli letter_A_white_1 letter_D_white_1 --overwrite
```

Uses YAML ``(a, α, k, fa_combine, sech_*)`` only — affine is ignored.

Optional amp override for comparison (does not edit YAML):

```bash
scripts/py experiments/schira2007/run_schira_only.py --set 201118 --sech-amp 0.1821 --overwrite
```

## Step 2 — Affine registration only (pick + fit)

Schira stays fixed. This step writes landmarks and a fitted camera affine; it
does **not** auto-edit `configs/schira/sets.yaml` and does **not** plot VSD
overlays.

Letter **F** on session set **201118**:

```bash
# Schira ink ↔ VSD (one letter)
scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 \
    --stimulus letter_F_white_1 --pick-schira

# All Fig. 13 letters — GUI for each, then pooled fit
scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 \
    --pick-schira --stimuli letter_D_white_1 letter_F_white_1 letter_L_white_1 letter_N_white_1

# Pooled fit only (YAMLs must already exist; does not open GUI)
scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 \
    --fit-merge experiments/schira2007/phase_0_register/landmarks_schira__201118__letter_*.yaml

# Plot thin black Schira ink on mean VSD using pooled affine
scripts/py experiments/schira2007/plot_pooled_registration_overlays.py

# Short status PDF (goal, results, figures)
scripts/py experiments/schira2007/make_phase_0_status_pdf.py
```

**Outputs** under `experiments/schira2007/phase_0_register/`:

- `landmarks_schira__201118__letter_F_white_1.yaml` — Schira (u,v) ↔ VSD pairs
- `landmarks__201118__letter_F_white_1.yaml` — legacy stimulus ↔ VSD pairs
- `affine_fit__201118__letter_F_white_1.yaml` — per-stimulus fit + RMSD
- `affine_fit__201118__pooled.yaml` — pooled multi-letter fit + per-stimulus RMSD
- optional cached `mean_vsd__*.npy` from `--pick` / `--pick-schira`

When the fit looks good, **manually copy** the `affine:` block from
`affine_fit__*.yaml` into `configs/schira/sets.yaml` for that set.

Fig. 10 guide for F: (1) upper stem∩top bar, (2) stem∩middle bar,
(3) bottom of stem, (4) end of top bar, (5) end of middle bar.

Thesis Fig. 11 reports **RMSD ≈ 1.5 px** for 20/11/18 after chamber
registration. Our landmark-only similarity fit currently lands ~**9–10 px**
RMSD — expect worse alignment than thesis Fig. 13 until landmarks / chamber
affine improve.

## Step 3 — Combine Schira + affine for VSD overlays

After copying a fitted `affine:` into `configs/schira/sets.yaml` (Step 2),
render **thin-line forward Schira ink** through the camera affine onto mean
raw VSD (thesis **Fig. 13** style — not thick dilate+fill blobs):

```bash
scripts/py experiments/schira2007/run_phase_0.py --set 201118 --overwrite
```

**Figures:** `experiments/schira2007/phase_0/`  
One PNG per session×condition (letter), plus `all_stimuli_overlay_montage.png`
and `index.yaml`. Do **not** use `run_phase_0_register.py` for this step —
registration stays pick+fit only.

Default ``--ink-style stroke`` stamps forward-mapped catalog ink pixels onto
the VSD grid (slim letter). Optional thick support view:

```bash
scripts/py experiments/schira2007/run_phase_0.py --set 201118 \
  --ink-style filled --overwrite
```

Alignment diagnostic (Schira-only | forward+affine | filled blob | vs thesis
Fig. 13 when the throwaway page is present):

```bash
scripts/py experiments/schira2007/make_overlay_alignment_diagnostics.py \
  --set 201118 --stimulus letter_F_white_1
```

**Thesis Fig. 13 grid** (panels c–h: D/F/L/N/bars vs our Schira + overlay):

```bash
scripts/py experiments/schira2007/make_vs_thesis_fig13_comparison.py --set 201118
```

Outputs: `experiments/schira2007/phase_0_register/vs_thesis_fig13_grid__201118.png`
(and `vs_thesis_fig13_reference_page__201118.png` for the full thesis page).
Thesis source: `_throwaway_thesis_fig10_pdf_pages/page_031.png`.

**Fig. 13 caption positions** (thesis VSD+model vs our Schira, no alignment):

```bash
scripts/py experiments/schira2007/make_fig13_caption_schira_compare.py
```

Output: `phase_0_register/fig13_caption_schira_vs_thesis_vsd.png`

**Single stimulus** (Schira only vs thesis VSD overlay, custom position):

```bash
scripts/py experiments/schira2007/make_schira_vs_thesis_vsd.py \\
  --stimulus letter_L_white_1 --pos-x 1.4 --pos-y -0.7 --thesis-panel e
```

Output: `phase_0_register/schira_vs_thesis_vsd__201118__letter_L_white_1__thesis_pos_1.4_-0.7__panel_e.png`

Same command also writes **`schira_vs_thesis_overlay_aligned__*.png`**: thesis black
overlay pixels extracted from the VSD crop, registered onto our Schira ``(u,v)``
with a similarity (translate, scale, rotation, flips). Blue = our ink; red =
thesis overlay.

## Parameter sets (YAML)

`configs/schira/sets.yaml`

- **201118** (20/11/2018): ``a=0.72``, ``alpha=1.5``, ``k=1``,
  ``shear: double_sech``, ``fa_combine: power``, ``sech_amp: 0.1821``
- **100718** (10/07/2018): ``a=0.74``, ``alpha=1.0``, ``k=1`` (not run)

## Formula (thesis monopole Double-Sech)

Schira 2007 eq. 5 (power / superscript form)::

    θ = atan2(y_deg, x_deg)                         # radians
    P = α * θ
    fa(E, P) = sech(P) ** ( sech(log(E/a) * S1) * S2 )
    w(E, P) = k * log(E * exp(i * P * fa) + a)

YAML: ``fa_combine: power``, ``sech_ecc_k`` = S1 (= 0.76),
``sech_amp`` = S2 (= **0.1821**). Flat-text “product” readings and
``sech_amp: 1.821`` are ablations only.

## Phase 1 reuse

Keep `src/retinotopy/` and the YAML sets. Registration is a separate affine
on top of Schira ``w`` (`src/retinotopy/register.py`).
