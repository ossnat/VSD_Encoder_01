"""Memory-lean twin of scripts/20_run_schira_global_channel_loo.py (protocol B, global-channel ridge).

Same folds (src.loo.folds), same targets (build_schira_xy, session -> anchor registration), same model
(standardize over train trial x pixel, ridge with intercept, alpha by efficient LOO as RidgeCV(cv=None)),
same per-fold files (global_channel_ridge.joblib, fold_pairs.parquet, metrics.json, fold_mean_*.npy),
so scripts/27 reads the result unchanged. Difference: features are kept once per stimulus (75 maps), not per
trial, and the ridge is solved from sufficient statistics, so it runs in < 1 GB instead of > 8 GB.

Usage: python loo_lean.py --config <schira_encoding config> [--run-tag TAG]
"""
import argparse, json, sys, types
from unittest import mock
for n in ("torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.models", "torchvision.transforms"):
    sys.modules.setdefault(n, mock.MagicMock(name=n))
from pathlib import Path
import numpy as np, pandas as pd, yaml, joblib
from common import REPO
from src.paths import resolve_data_path
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import parse_loss_roi_arg, resolve_loss_roi
from src.evaluation.mask import masked_pearson_r
from src.loo.folds import build_protocol_b_folds, expand_stimulus_id_patterns, load_heldout_config, strip_train_only_from_protocol_b_test
from src.schira_encoding.global_channel_model import GlobalChannelRidgeResult
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import global_channel_ridge_loo_dir, lut_path
from src.stimuli.identity import attach_stimulus_ids

p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--run-tag", default=None)
args = p.parse_args()
cfg = yaml.safe_load(open(args.config)); win = yaml.safe_load(open(REPO / cfg["window_config"]))
monkey, sset, anchor = cfg["monkey"], str(cfg["schira_set"]), str(cfg["anchor_session"])
spatial = tuple(cfg["spatial_size"]); H, W = spatial
out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], REPO)
lut = SchiraLUT.load(lut_path(out_root, monkey, schira_set=sset, anchor_session=anchor, spatial_h=H, spatial_w=W, canvas_size=210, input_size=224))
register_root = resolve_output_root(cfg["paths"]["session_register_root"], REPO)
window_id = win.get("window_id") or f"win_{int(win['start_frame']):04d}_{int(win['end_frame']):04d}"
pairs = pd.read_parquet(encoding_pairs_manifest_path(resolve_data_path(cfg["paths"]["encoding_pairs_root"], REPO), monkey, window_id))
pairs = attach_stimulus_ids(pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy())
held = load_heldout_config(REPO / "experiments/loo_encoding/heldout_non_letters.yaml")
avail = sorted(pairs["stimulus_id"].astype(str).unique())
heldout = [s for s in expand_stimulus_id_patterns(list(held.stimulus_ids), available_ids=avail) if s in set(avail)]
folds = strip_train_only_from_protocol_b_test(build_protocol_b_folds(pairs, heldout), held.train_only_sessions)
roi, roi_meta = resolve_loss_roi(mode=parse_loss_roi_arg(cfg["loss_roi"])[0], mask_path=parse_loss_roi_arg(cfg["loss_roi"])[1],
                                 repo=REPO, spatial_size=spatial, disk_radius=50)
fit_mask = lut.valid & np.asarray(roi, bool)
alphas = np.asarray(cfg["global_channel_ridge"]["alphas"], float)
out_dir = global_channel_ridge_loo_dir(out_root, monkey, window_id, "vgg16_imagenet", "block1_prepool", schira_set=sset,
                                       anchor_session=anchor, protocol="B", loss_roi=cfg["loss_roi"], run_tag=args.run_tag)
print(f"set {sset}: folds {len(folds)}, fit pixels {fit_mask.sum()}, out {out_dir}", flush=True)

# ---- cache: features once per (date, condition), targets per trial (all 100x100 px)
kw = dict(repo=REPO, out_root=out_root, monkey=monkey, model_slug="vgg16_imagenet", feature_layer="block1_prepool",
          schira_set=sset, anchor_session=anchor, spatial_size=spatial, register_root=register_root)
allp = pd.concat([f for _, f in folds]).drop_duplicates("trial_global_id").reset_index(drop=True)
Y = {}; X = {}
for i in range(0, len(allp), 40):
    ch = allp.iloc[i:i + 40]
    x, y = build_schira_xy(ch, **kw)
    for j, row in enumerate(ch.itertuples(index=False)):
        Y[row.trial_global_id] = y[j].reshape(H, W).astype(np.float32)
        X.setdefault((str(row.date), str(row.condition)), x[j].reshape(-1, H, W).astype(np.float32))
print(f"cached {len(Y)} trials, {len(X)} feature maps", flush=True)
C = next(iter(X.values())).shape[0]

def design(df, mask):
    for row in df.itertuples(index=False):
        xs = X[(str(row.date), str(row.condition))][:, mask].T.astype(np.float64); ys = Y[row.trial_global_id][mask].astype(np.float64)
        ok = np.isfinite(ys) & np.isfinite(xs).all(1); yield xs[ok], ys[ok]

def fit(df):
    n = 0; S1 = np.zeros(C); S2 = np.zeros((C, C)); Sy = 0.0; Sxy = np.zeros(C)
    for xs, ys in design(df, fit_mask):
        n += len(ys); S1 += xs.sum(0); S2 += xs.T @ xs; Sy += ys.sum(); Sxy += xs.T @ ys
    mu = S1 / n; sd = np.sqrt(np.maximum(np.diag(S2) / n - mu ** 2, 0)); sd = np.where(sd < 1e-8, 1.0, sd); ybar = Sy / n
    Czz = (S2 - n * np.outer(mu, mu)) / np.outer(sd, sd); Czy = (Sxy - mu * Sy) / sd
    sols = {a: np.linalg.solve(Czz + a * np.eye(C), np.column_stack([Czy, np.eye(C)])) for a in alphas}
    sse = dict.fromkeys(alphas, 0.0)
    for key, grp in df.groupby(["date", "condition"], sort=False):   # efficient LOO residuals, as RidgeCV(cv=None)
        xs_all = X[(str(key[0]), str(key[1]))][:, fit_mask].T.astype(np.float64); z_all = (xs_all - mu) / sd
        fin_x = np.isfinite(xs_all).all(1)
        H_ = {a: 1.0 / n + np.sum((z_all @ S[:, 1:]) * z_all, 1) for a, S in sols.items()}
        P_ = {a: ybar + z_all @ S[:, 0] for a, S in sols.items()}
        for t in grp.trial_global_id:
            ys = Y[t][fit_mask].astype(np.float64); ok = np.isfinite(ys) & fin_x
            for a in alphas:
                sse[a] += np.sum(((ys[ok] - P_[a][ok]) / (1 - H_[a][ok])) ** 2)
    a = min(sse, key=sse.get)
    return GlobalChannelRidgeResult(weights=sols[a][:, 0].astype(np.float32), intercept=float(ybar), alpha=float(a), valid=fit_mask,
                                    spatial_size=spatial, n_channels=C, standardize_features=True, feature_mean=mu.astype(np.float32),
                                    feature_scale=sd.astype(np.float32), n_samples=int(n))

def predict(df, res):
    w = res.weights / res.feature_scale; b = res.intercept - float(res.feature_mean @ w)
    return np.stack([np.where(res.valid, np.tensordot(w, X[(str(r.date), str(r.condition))], 1) + b, np.nan) for r in df.itertuples(index=False)])

rows = []
for spec, fdf in folds:
    fd = out_dir / spec.fold_id; fd.mkdir(parents=True, exist_ok=True)
    sp = {s: fdf[fdf.loo_split == s] for s in ("train", "val", "test")}
    res = fit(sp["train"]); met = dict(fold_id=spec.fold_id, protocol="B", readout="global_channel_ridge (loo_lean)",
                                        heldout_stimulus_id=spec.heldout_stimulus_id, n_train=len(sp["train"]), n_val=len(sp["val"]),
                                        n_test=len(sp["test"]), n_fit_pixels=int(fit_mask.sum()), n_samples=res.n_samples, alpha=res.alpha,
                                        intercept=res.intercept, schira_set=sset, loss_roi=cfg["loss_roi"])
    for s, d in sp.items():
        if d.empty: continue
        yh = predict(d, res); yt = np.stack([Y[t] for t in d.trial_global_id])
        rs = [masked_pearson_r(yt[i], yh[i], res.valid) for i in range(len(d))]
        met[f"r_mean_{s}_masked"] = float(np.nanmean(rs)); met[f"r_median_{s}_masked"] = float(np.nanmedian(rs))
    with np.errstate(all="ignore"):
        mo, mr = np.nanmean(yt, 0).astype(np.float32), np.nanmean(yh, 0).astype(np.float32)
    met["r_hull_mean_maps"] = float(masked_pearson_r(mo, mr, fit_mask))
    np.save(fd / "fold_mean_orig.npy", mo); np.save(fd / "fold_mean_recon.npy", mr); np.save(fd / "valid.npy", res.valid)
    np.save(fd / "weights.npy", res.weights); joblib.dump(res, fd / "global_channel_ridge.joblib")
    (fd / "metrics.json").write_text(json.dumps(met, indent=2) + "\n"); fdf.to_parquet(fd / "fold_pairs.parquet", index=False)
    rows.append(met)
    print(f"  {spec.fold_id:34s} alpha={res.alpha:<9g} r_test(trial mean)={met['r_mean_test_masked']:.3f}  r_hull(mean maps)={met['r_hull_mean_maps']:.3f}", flush=True)
pd.DataFrame(rows).to_csv(out_dir / "loo_summary.csv", index=False)
print("wrote", out_dir / "loo_summary.csv")
