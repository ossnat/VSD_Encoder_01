"""Checks after re-extracting stimulus features per stimulus_uid (scripts/02c_extract_stimulus_features_by_uid.sh).

For every model / layer under DL_features_stimuli/<monkey>/ that has a maps_by_uid/ folder:
  1. unchanged images  -> new features == old per-session features (max |diff| / max |old| < 1e-4)
     ('unchanged' = the library image equals the image in stimuli/<monkey>/_backup_before_fix_2026-11-15/images,
      i.e. the image the old features were computed from)
  2. changed images    -> features differ (listed with the size of the change)
  3. coverage          -> one file per library uid; every trial in the encoding-pair tables resolves to an existing file
Usage: python shira_revised/stimulus_audit/check_uid_features.py [--features-root PATH] [--layers model/layer ...]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
REPO = Path(__file__).resolve().parents[2]; WS = REPO.parent; sys.path.insert(0, str(REPO))
try:
    import torch  # noqa: F401  (only needed by src.DL_features.__init__, not by these checks)
except ImportError:
    from unittest import mock
    for _n in ("torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.models", "torchvision.transforms"):
        sys.modules.setdefault(_n, mock.MagicMock())
from src.DL_features.schema import UID_MAPS_DIRNAME, legacy_stimulus_map_path, stimulus_map_path, stimulus_uid_map_path
RED, BOLD, END = "\033[91m", "\033[1m", "\033[0m"

ap = argparse.ArgumentParser(); ap.add_argument("--monkey", default="gandalf")
ap.add_argument("--features-root", type=Path, default=WS / "Data/VSD_Encoder_01/DL_features_stimuli")
ap.add_argument("--layers", nargs="*", default=None); args = ap.parse_args()
FR, M = args.features_root, args.monkey
STIM = WS / f"Data/VSD_Encoder_01/stimuli/{M}"
lookup = pd.read_parquet(STIM / "lookup.parquet"); lib = pd.read_parquet(STIM / "library.parquet")
old_img_dir = STIM / "_backup_before_fix_2026-11-15/images"
img = lambda p: np.asarray(Image.open(p).convert("RGB"))
# which session-conditions still show exactly the image the old features were computed from
unchanged, changed = [], []
for r in lookup.itertuples(index=False):
    old = old_img_dir / r.h5_session / f"{r.condition}.png"
    if not old.is_file(): continue
    new = STIM / "library" / f"{r.stimulus_uid}.png"
    (unchanged if np.array_equal(img(old), img(new)) else changed).append((r.h5_session, r.condition))
print(f"session-conditions with an old image: {len(unchanged)} unchanged, {len(changed)} changed")

layers = args.layers or sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in (FR / M).glob(f"*/*/{UID_MAPS_DIRNAME}"))
missing_layers = sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in (FR / M).glob("*/*/maps") if not (p.parent / UID_MAPS_DIRNAME).is_dir())
pairs = pd.concat([pd.read_parquet(p) for p in (WS / f"Data/VSD_Encoder_01/encoding_pairs/{M}").glob("*/manifest.parquet")])
pair_keys = sorted(set(zip(pairs.date.astype(str), pairs.condition.astype(str))))
problems, rows = [], []
for ml in layers:
    model, layer = ml.split("/")
    have = {p.stem for p in (FR / M / model / layer / UID_MAPS_DIRNAME).glob("*.npy")}
    miss_uid = sorted(set(lib.stimulus_uid) - have)
    miss_pairs = [k for k in pair_keys if not stimulus_map_path(FR, M, model, layer, *k).is_file()]
    worst_same, n_same, n_noold = 0.0, 0, 0
    for s, c in unchanged:
        o = legacy_stimulus_map_path(FR, M, model, layer, s, c)
        if not o.is_file(): n_noold += 1; continue
        a, b = np.load(o), np.load(stimulus_map_path(FR, M, model, layer, s, c))
        rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)) if a.shape == b.shape else np.inf
        worst_same = max(worst_same, rel); n_same += 1
    ch = []
    for s, c in changed:
        o = legacy_stimulus_map_path(FR, M, model, layer, s, c)
        if not o.is_file(): continue
        a, b = np.load(o), np.load(stimulus_map_path(FR, M, model, layer, s, c))
        ch.append(float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-12)) if a.shape == b.shape else np.inf)
    ok = (not miss_uid) and (not miss_pairs) and worst_same < 1e-4 and all(v > 1e-4 for v in ch)
    rows.append(dict(layer=ml, files=len(have), missing_uids=len(miss_uid), unresolved_pairs=len(miss_pairs),
                     unchanged_compared=n_same, max_rel_diff_unchanged=worst_same, changed_compared=len(ch),
                     min_rel_diff_changed=min(ch) if ch else np.nan, ok=ok))
    if not ok: problems.append(ml)
T = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(T.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
if missing_layers: print(f"layers with old maps but no {UID_MAPS_DIRNAME}/ yet: {missing_layers}")
if problems:
    print(f"{RED}{BOLD}{'█' * 70}\n██  FEATURE CHECK FAILED for: {', '.join(problems)}\n{'█' * 70}{END}"); sys.exit(1)
print(f"{BOLD}ALL FEATURE CHECKS PASS ({len(layers)} layers){END}")
