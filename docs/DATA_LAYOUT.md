# VSD Foundation Data Layout

Shared reference for all code repos under the workspace. Describes **where data lives** and **how files are structured** — not how any particular project preprocesses or trains on them.

---

## Workspace layout

Code repos and `Data/` are **siblings**. Neither repo should commit raw or processed H5 files.

```
<workspace>/                          e.g. .../VSD_FM/
├── Data/                             # shared; NOT in git
│   └── FoundationData/
│       ├── RawData/                  # source .mat (cluster / Drive); optional locally
│       └── ProcessedData/            # session H5s + split tables
├── VSD_foundation_model/             # MAE / foundation-model repo (example)
└── <your_new_project>/               # sibling repo
```

**Portable path convention:** configs and CSV cells use paths starting with `Data/...`, resolved relative to `<workspace>/`, not inside a repo:

```
Data/FoundationData/ProcessedData/gandalf/session_270618b_condsAN.h5
```

**Resolve rule:** `absolute_path = <workspace> / "Data" / "FoundationData" / ...`

On Colab or other machines, absolute paths in old index rows may appear; the portable form above is canonical.

---

## Subjects (monkeys)

| `monkey`   | Notes                          |
|-----------|--------------------------------|
| boromir   |                                |
| frodo     |                                |
| gandalf   |                                |
| legolas   |                                |
| unknown   | Exclude from splits / training |

Monkey names are lowercase in CSVs and H5 file attributes.

---

## Raw data (optional reference)

```
Data/FoundationData/RawData/
└── DataFromYarden/
    └── AN_files_uploaded/
        └── {monkey}_{date}/          e.g. gandalf_270618b/
            └── condsAN.mat           # MATLAB source for one recording session
```

One `.mat` per session typically contains multiple visual conditions (`condAN1`, `condAN2`, …). Ingestion (not covered here) flattens trials into per-session HDF5 files under `ProcessedData/`.

---

## Processed session HDF5 files

### Directory pattern

```
Data/FoundationData/ProcessedData/
├── {monkey}/                         # boromir | frodo | gandalf | legolas
│   ├── session_{date}_condsAN.h5     # one file per recording session
│   └── frames/                       # optional exported PNGs (not required for training)
└── splits/                           # global index + train/val/test tables (below)
```

**Filename:** `session_{date}_condsAN.h5`  
- `{date}` — session id, e.g. `270618b`, `011221a`  
- `condsAN` — all condition blocks from that session’s `condsAN.mat` live in one H5

### Internal structure

- **Flat layout:** trial datasets at the **root** of the file (no nested groups).
- **Dataset names:** `trial_000000`, `trial_000001`, … (six-digit zero-padded).
- **Dataset shape:** `(n_pixels, n_frames)` = `(10000, n_frames)` almost always.
  - `n_pixels = 100 × 100` (square VSD map).
  - `n_frames` varies by session (often ~200–256); see `shape` in the index/split CSV.
- **dtype:** `float32` typical.
- **Per-trial dataset attributes:** usually empty; metadata is at file level.

### File-level HDF5 attributes

| Attribute              | Example / meaning |
|------------------------|-------------------|
| `monkey`               | `gandalf` |
| `date`                 | `270618b` |
| `n_trials`             | count of `trial_*` datasets in this file |
| `created`              | ISO timestamp when H5 was written |
| `trial_metadata_json`  | JSON list: one dict per trial with `trial_global_id`, `monkey`, `date`, `condition`, `trial_index_in_condition`, `shape`, etc. |

### Reading a trial in Python

```python
import h5py
import numpy as np

h5_path = "<workspace>/Data/FoundationData/ProcessedData/gandalf/session_270618b_condsAN.h5"
with h5py.File(h5_path, "r") as f:
  data = f["trial_000154"][...]   # shape (10000, n_frames)

img = data[:, frame_idx].reshape(100, 100)   # one frame → 100×100
```

**Frame index:** 0-based along axis 1. Early frames (e.g. 5–24) are often used as baseline; stimulus/evoked windows are project-specific.

---

## Identifiers and grouping

| Concept | Fields | Notes |
|---------|--------|--------|
| **Global trial** | `trial_global_id` | Unique int across entire dataset (0 … N−1) |
| **Session** | `monkey` + `date` | One H5 file per session |
| **Condition** | `condition` | e.g. `condAN1`, `condAN2`, … or `condsXAN1` (rare) |
| **Condition within session** | `(monkey, date, condition)` | **Different sessions with the same cond label are different conditions** |
| **Trial in condition** | `trial_index_in_condition` | 0-based within that condition block |
| **H5 pointer** | `target_file` + `trial_dataset` | which file and which `trial_XXXXXX` |

**Important:** `condAN1` on session `100718a` and `condAN1` on `270618b` are **not** the same experimental condition for splitting purposes.

---

## Index and split tables (`splits/`)

```
Data/FoundationData/ProcessedData/splits/
├── all_trials_index.csv              # master list of all trials (no split column)
├── all_trials_index_gandalf.csv      # gandalf-only index (encoder default)
├── split_v3_seed17_session_condition_group.csv   # global v3 split
├── split_v3_seed17_session_condition_group_gandalf.csv   # gandalf v3 split (encoder default)
├── baseline_stats_v3_seed17_session_condition_group.json
├── baseline_stats_v3_seed17_session_condition_group.h5
└── split_v2_seed17_session_split.csv # legacy (session-level holdout)
```

The encoder project (`VSD_Encoder_01`) uses the **gandalf** split and index files by default (`configs/default.yaml`).

### `all_trials_index.csv`

One row per trial. Key columns:

| Column | Description |
|--------|-------------|
| `trial_global_id` | Global id |
| `monkey`, `date`, `condition` | Identifiers (`condition` as string, e.g. `condAN2`) |
| `source_file` | Original `.mat` path (may be machine-specific) |
| `target_file` | Portable `Data/FoundationData/ProcessedData/.../*.h5` |
| `trial_index_in_condition` | Index within condition |
| `shape` | e.g. `(10000, 256)` |
| `trial_dataset` | e.g. `trial_000042` |

~5,700+ trials after excluding `unknown` monkey.

### Split CSV (e.g. v3)

Same trials as the index, plus:

| Column | Description |
|--------|-------------|
| `split` | `train` \| `val` \| `test` |
| `condition` | Integer code in split CSV (maps from `condAN*` / `condsXAN*`) |
| `shutter_off` | Legacy scalar metadata (from v2); may be NaN |

**v3 split unit:** whole **`(monkey, date, condition)`** groups — all trials in a group share the same split (75% / 15% / 10% train/val/test, seed 17).

Your new project may define its **own** splits; you can still use `all_trials_index.csv` and session H5s as the ground truth for trial locations.

### Baseline stats artifacts (optional)

Produced by the foundation-model pipeline for z-score normalization:

- **JSON** — metadata; `stats_h5_path` is a filename relative to `splits/`.
- **H5** — datasets `mean` and `std`, shape `(1, 1, 100, 100)` per-pixel maps.

Baseline frames used to compute stats: **5–24 inclusive** (slice `[5:25)` in Python). Only if your project uses the same normalization.

---

## Condition name ↔ integer codes

In **split CSVs**, `condition` is an integer. Mapping from index strings:

| Index string | Split CSV int |
|--------------|---------------|
| `condAN1` / `condsXAN1` | 1 |
| `condAN2` / `condsXAN2` | 2 |
| … | … |

Regex: `cond[s]?X?AN(\d+)` → group 1.

---

## Path resolution (recommended helper)

```python
from pathlib import Path

def workspace_root(project_root: Path) -> Path:
    """Parent of repo; contains Data/ and code repos."""
    return project_root.resolve().parent

def resolve_data_path(project_root: Path, path: str) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p.resolve()
    if str(path).startswith("Data/"):
        return (workspace_root(project_root) / path).resolve()
    return (project_root / path).resolve()
```

Reference implementation: `VSD_foundation_model/src/utils/data_paths.py`.

---

## What belongs in git

| Track in repo | Do not track |
|---------------|--------------|
| Code, configs, small CSV references | `Data/` (H5, large CSV copies on cluster) |
| This doc (`docs/DATA_LAYOUT.md` copy) | Checkpoints, `runs/`, logs |
| Project-specific split outputs | |

Add to `.gitignore`:

```
Data/
../Data/
checkpoints*/
runs/
```

---

## Quick inspection commands

```bash
# H5 hierarchy + trial count (from VSD_foundation_model repo)
PYTHONPATH=. python scripts/report_h5_structure.py \
  --h5-path ../Data/FoundationData/ProcessedData/gandalf/session_270618b_condsAN.h5

# Row counts per split
python -c "import pandas as pd; print(pd.read_csv('../Data/FoundationData/ProcessedData/splits/split_v3_seed17_session_condition_group.csv')['split'].value_counts())"
```

---

## Related code (foundation model only)

These live in `VSD_foundation_model/` and are **not** required for other projects:

- `data_pp_and_split/` — regenerate v3 split + baseline stats
- `src/data/datasets.py` — PyTorch dataset over split CSV
- `src/utils/data_paths.py` — path helpers

For a new sibling repo: copy this file into `docs/DATA_LAYOUT.md`, implement or reuse `resolve_data_path`, and point configs at `Data/FoundationData/...`.

---

## Version

- Document version: 2026-06 (v3 split, session×condition groups)
- Index trials: ~5,706 (excluding `unknown`)
- Spatial size: 100×100 pixels per frame
