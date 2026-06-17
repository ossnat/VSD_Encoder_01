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
    ├── reconstructions_grid.png
    └── reconstruction_{trial_id}.png
```

## QC plots

1. **bias.png** — `intercept_` reshaped to 100×100
2. **reconstruction_*.png** — side-by-side per trial:
   - **Original (H5 mean)** — mean of raw trial frames `[start_frame, end_frame)` from session H5
   - **Reconstructed (RidgeCV)** — model prediction
3. **reconstructions_grid.png** — 3–4 sample trials (default: test split)

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

`configs/ridge/default.yaml` — `alphas`, `cv_folds`, `n_plot_samples`, `plot_prefer_split`.
