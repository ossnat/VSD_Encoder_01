# Box → gentle ellipse / corner-trim ROI refinement

Refines **accepted rectangular ROIs** from the first-pass LOO review with a
**small trim** only: cut low-reliability (green/blue) corners while keeping
**most of the box** (target ≈70–85% area retained).

## Motivation

Manual box ROIs under `experiments/loo_encoding/roi_review/` already capture the
response region with margin. An earlier version of this experiment built
**aggressive high-r contours** inside the box (seeds at `r ≥ 0.5`, morph-close,
outer contour). Those polygons were **too small** (often &lt;10% of the box) and
**too jagged** (hundreds of vertices).

This rewrite does the opposite:

1. Keep the trusted **box** as the starting shape.
2. Use split-half **r only inside the box** to decide what little to remove.
3. Prefer **simple** shapes: inscribed/soft ellipse, or a 4–8 vertex polygon.

## Methods

### Option 1 — Corner-trim polygon

1. Load box from `experiments/loo_encoding/roi_review/rois/{stimulus_id}.yaml`.
2. Compute per-stimulus odd-even split-half `r_map` (`win_0035_0042`).
3. For each of the **four corners**, measure mean / fraction of low-r
   (`r < 0.20` or `0.35`) in a corner square (~35% of `min(w,h)`).
4. If a corner is green-dominated (or clearly worse than the box core), cut a
   **triangular chamfer** (~15% of `min(w,h)`). This avoids wiping weak-
   reliability boxes where absolute low-r covers most pixels.
5. Outer contour → **Douglas–Peucker** simplify to **4–8 vertices**.
6. Retention stays in the ~70–90% band by construction (modest chamfers).

### Option 2 — Ellipse (usually primary)

1. Start from the **axis-aligned inscribed ellipse** (area ≈ `π/4` ≈ **78.5%**
   of the box — already in the target band).
2. Optional **soft shrink**: blend center/axes slightly toward the
   reliability-weighted mass (`r − 0.20`) so green corners fall outside while
   the yellow-red core stays in. If retention dips below 70%, fall back to the
   full inscribed ellipse.

## Sample stimuli

- `letter_A_white_1`
- `black_triangle_contour_0.4`
- `black_bar_vertical_0.3`
- `white_point_0.1`

## Usage

```bash
scripts/py experiments/box_polygon/refine_box_to_polygon.py
```

Subset:

```bash
scripts/py experiments/box_polygon/refine_box_to_polygon.py \
  --stimuli letter_A_white_1 white_point_0.1
```

## Outputs

| Path | Contents |
|------|----------|
| `figures/win_0035_0042/{id}__box_vs_polygon.png` | 4-panel review |
| `rois/win_0035_0042/{id}__ellipse.yaml` + `__ellipse_mask.npy` | Ellipse candidate |
| `rois/win_0035_0042/{id}__corner_trim.yaml` + `__corner_trim_mask.npy` | Corner-trim polygon |
| `rois/win_0035_0042/{id}__polygon.yaml` + `__polygon_mask.npy` | **Primary** alias (usually ellipse) |
| `box_polygon_summary__win_0035_0042.csv` | Retention %, vertices, primary pick |

### Figure layout

1. Stimulus mean + dashed box  
2. Split-half r (masked to box) + box  
3. Ellipse crop + cyan outline (★ if primary)  
4. Corner-trim polygon crop + lime outline (★ if primary)

## Params (defaults)

| Param | Value | Role |
|-------|-------|------|
| `R_TRIM_CANDIDATES` | 0.20, 0.35 | Low-r cutoffs for corner trim |
| `R_SOFT_FLOOR` | 0.20 | Soft-ellipse weight floor |
| `TARGET_FRAC_MIN/MAX` | 0.70 / 0.90 | Retention band |
| `BORDER_FRAC` | 0.35 | Corner square size vs `min(w,h)` |
| Chamfer leg | ~0.15 × `min(w,h)` | Triangle cut per bad corner |
| `TRIM_OPEN_RADIUS` | 1 | Optional fleck cleanup in chamfered corners |
| `MIN/MAX_VERTICES` | 4 / 8 | Douglas–Peucker budget |

## Related

- Box source: `experiments/loo_encoding/roi_review/`
- Reliability helpers: `experiments/noise_ceiling_roi/compute_noise_ceiling_rois.py`
