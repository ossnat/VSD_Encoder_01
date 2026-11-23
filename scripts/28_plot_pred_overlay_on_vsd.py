#!/usr/bin/env python3
"""Predicted map overlaid on the faded measured VSD map, one panel per held-out stimulus.

Works on a saved global-channel-ridge run (no refit):

* a Protocol-B LOO directory (``.../global_channel_ridge/protocol_B_<roi>``): one panel per fold, the fold's
  own model applied to its test trials (``fold_pairs.parquet``, split == test);
* a letters directory (``.../global_channel_ridge/letters_eval_<session>_no_letter_train``): one panel per
  letter of that session.

Background: trial-mean VSD of the stimulus (anchor-aligned), grey, faded. Foreground: trial-mean prediction on
the whole Schira LUT area (same weights; the fit used the noise-ceiling hull only to choose them), coloured, with
opacity rising with the predicted value. Dashed line: noise-ceiling hull. Writes ``overlay_pred_on_vsd.png``.

Example::

  scripts/py scripts/28_plot_pred_overlay_on_vsd.py --run-dir experiments/schira_encoding/gandalf/loo/\\
win_0035_0046/vgg16_imagenet/block1_prepool/set-revised_100718__anchor-100718a/global_channel_ridge/\\
letters_eval_201118d_no_letter_train
"""
from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import to_rgb

from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import resolve_loss_roi
from src.paths import project_root, resolve_data_path
from src.schira_encoding.global_channel_model import predict_global_channel_ridge
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import lut_path
from src.stimuli.identity import attach_stimulus_ids

PRED_RGB = np.array(to_rgb("#d62728"))


def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) or {}


def _norm(img, mask, lo, hi):
    v = img[mask & np.isfinite(img)]
    a, b = np.percentile(v, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-12), 0, 1)


def overlay_rgba(vsd, pred, lut_valid, fade=0.55):
    """Faded grey VSD (full chamber) + red prediction whose opacity follows the predicted value."""
    has = np.isfinite(vsd)
    g = _norm(vsd, has, 1, 99)
    base = np.where(has, 1.0 - fade + fade * g, 1.0)                 # fade toward white
    rgb = np.repeat(base[..., None], 3, axis=2)
    p = np.zeros_like(g)
    ok = lut_valid & np.isfinite(pred)
    p[ok] = _norm(pred, ok, 50, 99.5)[ok]                              # only the upper half of the prediction shows
    a = (p ** 1.2)[..., None] * 0.9
    return rgb * (1 - a) + PRED_RGB * a


def main() -> None:
    repo = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=repo / "configs/schira_encoding/default.yaml")
    ap.add_argument("--model-config", type=Path, default=repo / "configs/models/vgg16.yaml")
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else repo / args.run_dir

    cfg = _yaml(args.config)
    win = _yaml(repo / cfg.get("window_config", "configs/windows/evoked_35_46.yaml"))
    m = re.search(r"set-(.+?)__anchor-([^/]+)", str(run))
    schira_set, anchor = m.group(1), m.group(2)
    if schira_set != str(cfg["schira_set"]):
        raise SystemExit(f"run is set {schira_set!r} but --config has {cfg['schira_set']!r}")
    layer = run.parts[run.parts.index("global_channel_ridge") - 2]
    slug = run.parts[run.parts.index("global_channel_ridge") - 3]
    monkey, spatial = str(cfg["monkey"]), tuple(int(x) for x in cfg["spatial_size"])
    canvas = int(_yaml(repo / cfg.get("stimuli_config", "configs/stimuli/default.yaml")).get("canvas_size", 210))
    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    lut = SchiraLUT.load(lut_path(out_root, monkey, schira_set=schira_set, anchor_session=anchor,
                                  spatial_h=spatial[0], spatial_w=spatial[1], canvas_size=canvas,
                                  input_size=int(cfg.get("input_size", 224))))
    hull, _ = resolve_loss_roi(mode="noise_ceiling_hull", mask_path=None, repo=repo, spatial_size=spatial,
                               disk_radius=50)
    xy_kw = dict(repo=repo, out_root=out_root, monkey=monkey, model_slug=slug, feature_layer=layer,
                 schira_set=schira_set, anchor_session=anchor, spatial_size=spatial,
                 register_root=resolve_output_root(cfg["paths"]["session_register_root"], repo))

    items = []          # (title, model, test pairs)
    if (run / "global_channel_ridge.joblib").is_file():            # letters run: one model, one panel per letter
        session = re.search(r"letters_eval_(\w+?)_no_letter_train", run.name).group(1)
        pairs = pd.read_parquet(encoding_pairs_manifest_path(
            resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo), monkey, str(win["window_id"])))
        pairs = attach_stimulus_ids(pairs[pairs.nc_exists & pairs.stimulus_exists])
        pairs = pairs[pairs.stimulus_id.str.startswith("letter_") & (pairs.date.astype(str) == session)]
        model = joblib.load(run / "global_channel_ridge.joblib")
        for sid, g in pairs.groupby("stimulus_id"):
            items.append((f"{sid}  ({session}, n={len(g)})", model, g))
        title = f"Letters {session}, model trained on non-letters only"
    else:                                                           # LOO: one model per fold
        for fd in sorted(p for p in run.iterdir() if (p / "global_channel_ridge.joblib").is_file()):
            fp = pd.read_parquet(fd / "fold_pairs.parquet")
            te = fp[fp.loo_split == "test"]                       # "split" is the dataset split, not the fold
            items.append((f"{te.stimulus_id.iloc[0]}\n{te.date.nunique()} sessions, n={len(te)}",
                          joblib.load(fd / "global_channel_ridge.joblib"), te))
        title = f"Leave-one-stimulus-out ({run.name}), held-out stimulus per panel"

    n = len(items)
    nc = min(args.ncols, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.2 * nr + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, model, te) in zip(axes.flat, items):
        x, y = build_schira_xy(te, **xy_kw)
        pred = predict_global_channel_ridge(x, replace(model, valid=lut.valid.copy())).mean(axis=0)
        vsd = np.nanmean(y, axis=0)
        ax.imshow(overlay_rgba(vsd, pred, lut.valid), interpolation="nearest")
        ax.contour(hull.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
        ax.set_title(name, fontsize=8)
    fig.suptitle(f"{title}\nred = prediction ({slug}/{layer}, shared channel weights) on faded VSD mean\n"
                 f"set {schira_set}, anchor {anchor} · dashed = noise-ceiling hull", fontsize=9)
    fig.tight_layout()
    out = run / "overlay_pred_on_vsd.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
