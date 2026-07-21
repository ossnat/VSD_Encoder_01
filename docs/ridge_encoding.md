# RidgeCV encoding (stage 03)

Train a linear encoder from stimulus CNN features to trial-averaged VSD maps.

## Model

For each trial:

- **X**: flattened ResNet activation map from the condition's stimulus image
- **Y**: frame-averaged VSD map (100×100) from stage 01

```text
Y = X @ W + b
```

`RidgeCV` selects L2 regularization `alpha` by cross-validation on the **train** split only.

By default (`alpha_per_target: true` in `configs/ridge/default.yaml`), a **separate α is chosen per VSD pixel**. Sklearn only supports that mode with leave-one-out GCV (`cv=None`), so `cv_folds` is ignored when `alpha_per_target` is true. Set `alpha_per_target: false` to use one shared α with K-fold CV (`cv_folds`).

The intercept `b` (per pixel) is saved and plotted as the **bias map** — expected to resemble the mean evoked response.

## Output layout

```
Data/VSD_Encoder_01/ridge_encode/
└── {monkey}/{window_id}/{model_slug}/{feature_layer}/
    ├── model.joblib
    ├── config.json
    └── metrics.json

plots/ridge_encode/
└── {monkey}/{window_id}/{model_slug}/{feature_layer}/
    ├── bias.png
    ├── alpha_per_pixel.png   # when alpha_per_target
    ├── reconstructions_grid.png
    └── reconstruction_{trial_id}.png
```

## QC plots

1. **bias.png** — `intercept_` reshaped to 100×100 (over train-mean VSD underlay)
2. **weight_norm_per_pixel.png** — per-pixel L2 norm of Ridge weights across features (same underlay style)
3. **alpha_per_pixel.png** — selected α per pixel (log scale), over train-mean VSD underlay (when `alpha_per_target`)
4. **by_condition/{date}__{condition}.png** — side-by-side per condition (one trial each):
   - **Original (H5 mean)** — mean of raw trial frames `[start_frame, end_frame)` from session H5
   - **Reconstructed (RidgeCV)** — model prediction (same for all trials in a condition)
5. **reconstructions_by_condition.png** — paginated orig|recon grid (all conditions)
6. **reconstructions_by_condition_recon_only.png** — grid of reconstructions only for shape comparison

## Run

Prerequisites: stages 01, 01b, 01c, 02b.

```bash
pip install scikit-learn joblib

scripts/py scripts/03_train_ridge_encoder.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml \
  --model configs/models/resnet18.yaml \
  --ridge-config configs/ridge/default.yaml
```

Cluster:

```bash
sbatch slurm/train_ridge_encoder.slurm
```

## Config

`configs/ridge/default.yaml` — `alphas`, `alpha_per_target`, `cv_folds`, `plot_prefer_split`, evaluation mask.
