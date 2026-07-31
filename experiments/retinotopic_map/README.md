# Retinotopic map probe — synthetic grid stimulus

Unseen **grid** stimulus for retinotopic / encoding probes: gray canvas with
five evenly spaced vertical + horizontal black lines (**no** edge border /
contour box).

**Stimulus PNG has no text** — only gray background + black grid lines. Axis
labels / titles appear only on the review figure outside the image.

Treated as a **left-out** probe: train a ResNet18 ImageNet **layer3** ridge
encoder on **all real encoding pairs** for `win_0035_0043`, then predict the
mean VSD map for this grid (never seen in training).

## Stimulus geometry

| Property | Value |
|----------|-------|
| Canvas | **224×224** (same as `RenderConfig` / `configs/stimuli/default.yaml`) |
| Background | RGB gray **128** |
| Border | **None** (`border_px: 0`) |
| Grid lines | **5** vertical + **5** horizontal, **2 px** wide |
| Text on stimulus | **None** (`has_text_on_stimulus: false`) |
| Spacing | `n+1` equal gaps across the full canvas `[0, 223]`; line *k* at `round(k·223/6)` → **37, 74, 112, 149, 186** |

Earlier drafts used a 3 px black edge border and 3 px lines (`grid_5x5_border`);
current default is **no border** and **2 px** grid lines under id `grid_5x5`.

## Usage

```bash
# 1) Render raw stimulus + clean review figure (black grid only)
scripts/py experiments/retinotopic_map/render_grid_stimulus.py

# 2) Predict mean VSD as unseen / left-out
#    Loads cached full all-pairs ResNet18/layer3 ridge for win_0035_0043
#    (trains only if the cache is missing)
scripts/py experiments/retinotopic_map/predict_grid_vsd.py

# Force retrain of the full all-pairs model
scripts/py experiments/retinotopic_map/predict_grid_vsd.py --retrain-full
```

## Model choice (prediction)

1. **Preferred:** full all-pairs ridge under
   `experiments/retinotopic_map/models/win_0035_0043/resnet18_imagenet/layer3/full_all_pairs/`
   — ResNet18 / layer3 trained on **all** complete encoding pairs for
   `win_0035_0043` (train+val+test combined = all real conditions). Cached on
   first run; use `--retrain-full` to rebuild.
2. Only if that training is impossible: fall back to LOO protocol B or the
   older main train-split ridge under `win_0035_0042`.

### Clarification on prior LOO run

A previous prediction used LOO protocol B fold `B__letter_D_white_1`. That is
**not** “all other conditions” in the sense of a full model: it holds out
`letter_D_white_1` and trains on the remaining trials in that fold
(`n_train=1247`). It is a **partial / held-out-stimulus** encoder, not a
pretrained main ridge on every condition. The default path above trains the
full all-pairs model instead.

The synthetic grid was never a training stimulus for either source.

## Outputs

| Path | Contents |
|------|----------|
| `figures/grid_5x5__stimulus.png` | Raw RGB stimulus (**no text**) |
| `figures/grid_5x5__stimulus_review.png` | Black-grid review (labels outside image) |
| `figures/grid_5x5__geometry.json` | Line coordinates + spacing rule |
| `figures/grid_5x5__stimulus_vs_predicted_vsd.png` | Stimulus \| predicted mean VSD |
| `figures/grid_5x5__predicted_vsd.npy` | Predicted 100×100 map |
| `figures/grid_5x5__prediction_meta.json` | Model provenance |
| `models/.../full_all_pairs/model.joblib` | Full all-pairs ridge weights |

There is **no real VSD “original”** for this probe — the prediction figure is
stimulus vs predicted mean map only.
