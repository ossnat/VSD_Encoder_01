# GWP exploratory visualizations

Side-channel scripts for understanding Gabor wavelet pyramid features on simple geometric stimuli. **Not part of the encoding pipeline.**

## Run

```bash
scripts/py exploratory/gwp_visualization/scripts/plot_gwp_layers.py
```

Outputs PNGs under `exploratory/gwp_visualization/results/`.

## What it shows

For 2–3 stimuli (circle, triangle from the catalog):

1. Original luminance input
2. Per-scale energy maps (max over 8 orientations) — the pyramid “layers”
3. Per-scale orientation fan (8 orientations) for scale 3

Uses the same filter bank as `configs/models/gabor_serre.yaml` but **without** spatial pooling, so maps stay at full input resolution.

### Ridge feature importance

Which scales, orientations, and stimulus locations Ridge uses (|coefficient| mass):

```bash
scripts/py exploratory/gwp_visualization/scripts/plot_gwp_ridge_importance.py
```

Outputs: `exploratory/gwp_visualization/results/importance/{model_slug}/`
