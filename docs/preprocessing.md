# Preprocessing: frame-averaged trials

Pipeline stage **01** reads session HDF5 trials (see `DATA_LAYOUT.md`), averages a frame window per trial, and writes self-describing NetCDF files plus a manifest.

These averaged maps are the **targets (Y)** in the stimulus→VSD encoding pipeline. See `docs/encoding_pipeline.md`.

## Window convention

- Slice is **half-open**: `[start_frame, end_frame)` in Python indexing.
- Example: `start_frame: 5`, `end_frame: 25` → frames 5, 6, …, 24 (20 frames).
- Output directory name: `win_{start:04d}_{end:04d}` (e.g. `win_0030_0050`).

## Output layout

```
Data/VSD_Encoder_01/averaged/
└── {monkey}/
    └── {window_id}/
        ├── manifest.parquet
        └── trials/
            ├── 000154.nc
            └── ...
```

Sample plots (3–4 trials per run) are written to:

```
plots/averaged/{monkey}/{window_id}/
```

## xarray schema (per trial `.nc`)

| Item | Value |
|------|--------|
| DataArray name | `vsd` |
| dims | `y`, `x` |
| shape | `(100, 100)` |
| dtype | `float32` |

### Attributes

| Attribute | Description |
|-----------|-------------|
| `trial_global_id` | Global trial id (join key to split CSV) |
| `monkey` | Subject name |
| `date` | Session id (H5 `date` attribute) |
| `condition` | Condition string, e.g. `condAN1` |
| `condition_code` | Integer code from split CSV |
| `trial_index_in_condition` | 0-based index within condition block |
| `trial_dataset` | H5 dataset name, e.g. `trial_000154` |
| `target_file` | Portable path to source session H5 |
| `window_id` | e.g. `win_0030_0050` |
| `start_frame`, `end_frame` | Averaging window (half-open) |
| `n_frames_averaged` | `end_frame - start_frame` |
| `source_n_frames` | Original trial length (axis 1 of H5 array) |
| `avg_method` | `mean` (default) |
| `normalization` | `none` (default); future: `baseline_zscore`, etc. |
| `split` | `train` \| `val` \| `test` from v3 split |
| `created` | ISO timestamp when file was written |

## Why `manifest.parquet`?

Each trial is stored as its own `.nc` file. Training and QC need to filter by `split`, `condition`, or `date` without opening thousands of NetCDF files. The manifest is a single table (one row per trial) with paths and metadata for fast pandas queries. Individual `.nc` attrs remain so any file is self-describing if copied elsewhere.

## Run locally

```bash
cd VSD_Encoder_01
pip install -e .

PYTHONPATH=. python scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_30_50.yaml
```

Dry run on a few trials:

```bash
PYTHONPATH=. python scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_30_50.yaml \
  --max-trials 10
```

Trials whose `target_file` H5 is not present under the workspace `Data/` tree are skipped automatically (common when only a subset of sessions is synced locally).

## Cluster (SLURM)

One-time setup on the login node:

```bash
bash scripts/cluster_setup.sh
```

Submit:

```bash
sbatch slurm/build_averaged_trials.slurm
```

Assumes sibling layout:

```
<workspace>/
├── Data/
└── VSD_Encoder_01/
```

Dependencies are listed in `requirements.txt` at the repo root.

## Related

- `docs/encoding_pipeline.md` — full stimulus→VSD encoding flow
- `docs/stimulus_rendering.md` — stage 01b stimulus images
