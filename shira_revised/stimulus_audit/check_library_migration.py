"""One-off checks for the move from per-session images (images/<session>/<cond>.png) to the stimulus library.
Usage: python shira_revised/stimulus_audit/check_library_migration.py <old_manifest.parquet>
"""
import hashlib, json, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd
from PIL import Image
REPO = Path(__file__).resolve().parents[2]; WS = REPO.parent; sys.path.insert(0, str(REPO))
from src.stimuli.identity import attach_stimulus_ids
ROOT = WS / "Data/VSD_Encoder_01/stimuli/gandalf"
old = pd.read_parquet(sys.argv[1]); new = pd.read_parquet(ROOT / "manifest.parquet")
lib = pd.read_parquet(ROOT / "library.parquet"); look = pd.read_parquet(ROOT / "lookup.parquet")
ok = True
def report(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")
px = lambda p: np.asarray(Image.open(WS / p).convert("RGB"))

# 1. same pixels: every old session-condition image == the library image the new manifest points to
newk = new.set_index(["h5_session", "condition"])
diff = []
for _, r in old.drop_duplicates(["h5_session", "condition", "image_path"]).iterrows():
    cand = newk.loc[[(r.h5_session, r.condition)]] if (r.h5_session, r.condition) in newk.index else None
    if cand is None: diff.append((r.h5_session, r.condition, "missing in new manifest")); continue
    cand = cand[(cand.shape_type == r.shape_type)] if (cand.shape_type == r.shape_type).any() else cand
    if not np.array_equal(px(r.image_path), px(cand.image_path.iloc[0])): diff.append((r.h5_session, r.condition, "pixels differ"))
report("1. pixel-identical to the per-session images", not diff, f"({len(old)} old rows) {diff[:4]}")
# 2. library consistency
report("2a. each session-condition -> exactly one uid", not look.duplicated(["h5_session", "condition"]).any() and look.stimulus_uid.notna().all())
files = {p.stem for p in (ROOT / "library").glob("*.png")}
report("2b. library files == uids in use", files == set(look.stimulus_uid), f"files {len(files)}, uids {look.stimulus_uid.nunique()}")
h = {}
for u in lib.stimulus_uid: h.setdefault(hashlib.md5(px(f"Data/VSD_Encoder_01/stimuli/gandalf/library/{u}.png").tobytes()).hexdigest(), []).append(u)
dups = [v for v in h.values() if len(v) > 1]
report("2c. no two uids with identical pixels", not dups, str(dups[:3]))
report("2d. counts", True, f"{len(new)} session-conditions -> {new.stimulus_uid.nunique()} images; uses per image max {lib.n_uses.max()}")
# 3. coverage of the recordings
miss = []
for f in sorted((WS / "Data/FoundationData/ProcessedData/gandalf").glob("session_*_condsAN.h5")):
    s = f.name.split("_")[1]
    with h5py.File(f) as hf: conds = sorted({m["condition"] for m in json.loads(hf.attrs["trial_metadata_json"])})
    for c in conds:
        if not ((new.h5_session == s) & (new.condition == c)).any(): miss.append(f"{s}:{c}")
KNOWN = {"201118b:condAN1", "201118b:condAN2"}   # 'Control attention' paradigm, not parsed (circle contour 3 deg, in/out of chamber)
report("3. every recorded condition has a stimulus", not (set(miss) - KNOWN), f"missing: {sorted(set(miss) - KNOWN)}; known exclusions: {sorted(set(miss) & KNOWN)}")
report("3b. 24/07 horizontal bar is condAN7", all(((new.h5_session == s) & (new.condition == "condAN7") & (new.shape_type == "bar_horizontal")).any() for s in ("240718a", "240718b", "240718c")))
# 5. stimulus_id (shape family used for leave-one-out) unchanged except the known size fixes
base = pd.read_parquet(ROOT / "_backup_before_fix_2026-11-15/manifest.parquet").drop_duplicates(["h5_session", "condition", "shape_type"])
o = attach_stimulus_ids(base)[["h5_session", "condition", "shape_type", "stimulus_id"]]; n = attach_stimulus_ids(new)[["h5_session", "condition", "shape_type", "stimulus_id"]]
m = o.merge(n, on=["h5_session", "condition", "shape_type"], suffixes=("_old", "_new"))
ch = m[m.stimulus_id_old.fillna("-") != m.stimulus_id_new.fillna("-")]
print(f"[INFO] stimulus_id changes vs the images before 15/11 ({len(ch)}):"); print(ch.to_string(index=False) if len(ch) else "  none")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
