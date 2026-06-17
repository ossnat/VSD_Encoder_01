# Encoding pair manifest (stage 01c)

Trial-level join table linking each encoding-session trial to:

- **X**: rendered stimulus image (`image_path`)
- **Y**: frame-averaged VSD map (`nc_path`)
- **split**: train / val / test from the v3 split CSV

## Join logic

```
split trials (monkey, date, condition)
    ⨝ stimulus manifest (h5_session, condition)   # date == h5_session
```

Only sessions present in the stimulus catalog are included (encoder experiments).

Duplicate catalog rows for the same `(h5_session, condition)` are deduplicated, preferring non-blank stimuli.

## Output layout

```
Data/VSD_Encoder_01/encoding_pairs/
└── {monkey}/
    └── {window_id}/
        └── manifest.parquet
```

Key columns:

| Column | Description |
|--------|-------------|
| `trial_global_id` | Global trial id |
| `date` | Session id (e.g. `270618b`) |
| `condition` | e.g. `condAN1` |
| `split` | `train` \| `val` \| `test` |
| `image_path` | Rendered stimulus PNG |
| `nc_path` | Averaged VSD target `.nc` |
| `nc_exists` | Whether `nc_path` is on disk |
| `stimulus_exists` | Whether `image_path` is on disk |
| `shape_type`, `color`, `is_blank` | Stimulus metadata |

CNN features (stage 02b) are computed once per `(date, condition)` and joined back to all trials in that condition.

## Run

Requires stage **01** (averaged VSD) and **01b** (stimulus images) for the same monkey.

```bash
PYTHONPATH=. python scripts/01c_build_encoding_pairs.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml
```

Drop pairs without averaged `.nc` files (e.g. before stage 01 finishes):

```bash
PYTHONPATH=. python scripts/01c_build_encoding_pairs.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml \
  --require-nc
```

Cluster:

```bash
sbatch slurm/build_encoding_pairs.slurm
```

## Notes

- Only trials from encoder sessions (present in the stimulus catalog) are included.
- Some `(date, condition)` groups in the split CSV may lack a catalog entry — the build script prints a warning listing them (e.g. `240718a` / `condAN2` when the catalog maps that condition to `240718b`).
- Duplicate catalog rows for the same `(h5_session, condition)` are deduplicated (non-blank preferred).
