# Legacy OM vs Ayzenshtat 2012 Fig. 2C

Paper figure (your screenshot): visual quadrant → cortical “banana” with
**fovea on the right**, **HM on the bottom**, **LVM on the top**, eccentricity
labels along the LVM.

## What legacy `trans_cortex` actually does

Same Double-Sech monopole family, then:

```matlab
logAngle = abs(angle(w));
logRad   = abs(w);
[x,y] = pol2cart(logAngle, logRad);
x1 = round(x*1000) - min(...) + 1;   % OM column
y1 = round(y*1000) - min(...) + 1;   % OM row
% polygon-fill OM
```

For lower-field points with `Im(w) ≥ 0`, folded `(x,y) ≈ (Re(w), Im(w))`.

After the **min-shift**:

| Landmark | Where it lands in raw OM |
|----------|--------------------------|
| Fovea (`Re(w) = k log a < 0`) | **LEFT** (smallest column) |
| Large eccentricity | **RIGHT** |
| HM (`Im ≈ 0`) | **TOP** (smallest row) |
| LVM (larger `|Im|`) | **BOTTOM** |

So raw `imshow(OM)` looks “backwards” relative to the paper figure.

## What the paper figure does differently (display, not formula)

Ayzenshtat Fig. 2C is the **same cortical plane** drawn with:

1. **x flipped** → fovea on the **right**
2. **y flipped** → HM on the **bottom**

That is exactly `rot90(OM, 2)` (180° rotation) of the MATLAB OM.

See: `scripts/07_legacy_vs_ayzenshtat_orientation.py` → panel B = raw OM,
panel C = paper orientation.

## Other real differences (secondary)

| Topic | Legacy Gandalf (`trans_cortex`) | Ayzenshtat 2012 Fig. 2 |
|-------|----------------------------------|-------------------------|
| α on θ | **`T = θ / α`** (Amit) | Paper text: **`P = α·θ`** (multiply) |
| `(a,k,α)` | Joanna `0.72, 1, 1.5` or Amit `0.39, 1, 1.5` | Monkey C `0.57, 7.7, 0.52`; L `0.46, 4.87, 0.51` |
| Cortical axes | Integer OM indices (`×1000`) | Continuous `Re(w), Im(w)` (and `k` in mm) |
| Registration | Later: 2-pt similarity → 100×100 chamber | Paper panel is Schira cortex only |

The **banana shape / polar→nearly-rectilinear grid** should already match once
orientation is fixed. Magnification spacing (1° vs 2° vs 5°) will still differ
if `(a,α,k)` differ — that is expected across animals/fits.

## Bottom line for “did we get legacy right?”

1. Raw OM looking “wrong” next to the paper is mostly **axis orientation**.
2. Flip to fovea-right / HM-bottom → should read like Fig. 2C.
3. Remaining shape differences after that flip → check `(a,α,k)` and α
   divide vs multiply — not registration (we have not applied it yet).
