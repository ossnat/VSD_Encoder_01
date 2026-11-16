"""Average VSD trials per condition (all frames kept, float32) -> cond_avgs/<session>.npz
Usage: python3 make_cond_avgs.py <session_h5_name> ..."""
import sys, json, time, numpy as np, h5py, os
ROOT = os.path.expanduser("~/mnt/VSD_FM/Data/FoundationData/ProcessedData/gandalf")
for name in sys.argv[1:]:
    t0 = time.time(); f = h5py.File(os.path.join(ROOT, name), "r")
    md = json.loads(f.attrs["trial_metadata_json"]); acc = {}
    for i, m in enumerate(md):
        d = f[f"trial_{i:06d}"][...].astype(np.float64)
        ok = np.isfinite(d) & (np.abs(d) < 1e3); d = np.where(ok, d, 0.0)
        s, k, n = acc.get(m["condition"], (0, 0, 0)); acc[m["condition"]] = (s + d, k + ok, n + 1)
    out = {c: (s / np.maximum(k, 1)).astype(np.float32) for c, (s, k, n) in acc.items()}
    for c, (s, k, n) in acc.items(): out[c][k < 0.5 * n] = np.nan     # pixels missing in most trials
    np.savez_compressed(f"cond_avgs/{name.replace('_condsAN.h5','')}.npz", **out, **{f"n_{c}": n for c, (s, k, n) in acc.items()})
    print(name, {c: n for c, (s, k, n) in acc.items()}, f"{time.time()-t0:.0f}s", flush=True)
