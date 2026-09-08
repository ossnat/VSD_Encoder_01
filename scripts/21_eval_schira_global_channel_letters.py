#!/usr/bin/env python3
"""Train global-channel ridge on **non-letters only**; predict letters on one session.

No letter stimulus_id enters training (including 201118a/c/d). Default test
session is ``201118d`` (more trials than ``c``).

Example::

  scripts/py scripts/21_eval_schira_global_channel_letters.py \\
    --model-config configs/models/vgg16.yaml \\
    --feature-layer block1_prepool \\
    --test-session 201118d
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import parse_loss_roi_arg, resolve_loss_roi
from src.evaluation.mask import apply_mask_nan, masked_map_summary, masked_pearson_r
from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
)
from src.evaluation.plotting import (
    plot_pixel_correlation_heatmap,
    plot_pixel_mean_maps,
    shared_orig_recon_residual_clims,
)
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import register_mapgeog
from src.schira_encoding.global_channel_model import (
    fit_global_channel_ridge,
    predict_global_channel_ridge,
)
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import lut_path
from src.stimuli.identity import attach_stimulus_ids


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _is_letter(sid: object) -> bool:
    return str(sid).startswith("letter_")


def _odd_even_means(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    even = maps[0::2]
    odd = maps[1::2]
    if even.size == 0 or odd.size == 0:
        raise RuntimeError(f"Need both odd/even halves (n={maps.shape[0]})")
    return (
        np.nanmean(even, axis=0).astype(np.float32),
        np.nanmean(odd, axis=0).astype(np.float32),
    )


def main() -> None:
    repo = project_root()
    register_mapgeog()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--window-config", type=Path, default=None)
    p.add_argument("--feature-layer", type=str, default=None)
    p.add_argument(
        "--test-session",
        type=str,
        default="201118d",
        choices=["201118c", "201118d"],
        help="Single letter session used only for prediction (default: 201118d).",
    )
    p.add_argument("--loss-roi", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    win_path = args.window_config
    if win_path is None:
        win_path = repo / str(
            cfg.get("window_config", "configs/windows/evoked_35_46.yaml")
        )
    win = _load_yaml(win_path)
    model_cfg = _load_yaml(args.model_config)
    merged = {**cfg, **{k: v for k, v in win.items() if k != "paths"}}
    merged["paths"] = {**(win.get("paths") or {}), **(cfg.get("paths") or {})}

    monkey = str(merged["monkey"])
    schira_set = str(merged["schira_set"])
    anchor = str(merged["anchor_session"])
    spatial = tuple(int(x) for x in merged["spatial_size"])
    input_size = int(merged.get("input_size", model_cfg.get("input_size", 224)))
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "block1_prepool"))
    slug = model_slug(model_cfg)
    test_session = str(args.test_session)

    stimuli_cfg = _load_yaml(
        repo / str(merged.get("stimuli_config", "configs/stimuli/default.yaml"))
    )
    canvas = int(stimuli_cfg.get("canvas_size", 210))

    out_root = resolve_output_root(merged["paths"]["schira_encoding_root"], repo)
    lut = SchiraLUT.load(
        lut_path(
            out_root,
            monkey,
            schira_set=schira_set,
            anchor_session=anchor,
            spatial_h=spatial[0],
            spatial_w=spatial[1],
            canvas_size=canvas,
            input_size=input_size,
        )
    )
    register_root = resolve_output_root(
        merged["paths"]["session_register_root"], repo
    )

    pairs_root = resolve_data_path(merged["paths"]["encoding_pairs_root"], repo)
    start = int(merged["start_frame"])
    end = int(merged["end_frame"])
    window_id = merged.get("window_id") or f"win_{start:04d}_{end:04d}"
    pairs = pd.read_parquet(
        encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    )
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)
    sid = pairs["stimulus_id"].astype(str)
    is_letter = sid.map(_is_letter)

    train_df = pairs.loc[~is_letter].copy()
    test_df = pairs.loc[
        is_letter & (pairs["date"].astype(str) == test_session)
    ].copy()
    if train_df.empty:
        raise RuntimeError("No non-letter training pairs")
    if test_df.empty:
        raise RuntimeError(f"No letter test pairs for session {test_session}")

    # Sanity: no letter ids in train
    train_letters = sorted(
        s for s in train_df["stimulus_id"].astype(str).unique() if _is_letter(s)
    )
    if train_letters:
        raise RuntimeError(f"Letter leakage in train: {train_letters}")

    loss_roi_raw = str(
        args.loss_roi
        if args.loss_roi is not None
        else merged.get("loss_roi", "noise_ceiling_hull")
    )
    loss_mode, loss_path = parse_loss_roi_arg(loss_roi_raw)
    train_roi, train_meta = resolve_loss_roi(
        mode=loss_mode,
        mask_path=loss_path,
        repo=repo,
        spatial_size=spatial,
        disk_radius=50,
    )
    fit_mask = lut.valid.copy()
    if train_roi is not None:
        fit_mask &= np.asarray(train_roi, dtype=bool)

    out_dir = (
        out_root
        / monkey
        / "loo"
        / window_id
        / slug
        / layer
        / f"set-{schira_set}__anchor-{anchor}"
        / "global_channel_ridge"
        / f"letters_eval_{test_session}_no_letter_train"
    )
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file() and not args.overwrite:
        print(f"Exists (skip): {out_dir}  (pass --overwrite)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    ridge_cfg = merged.get("ridge") or {}
    gcr_cfg = merged.get("global_channel_ridge") or {}
    alphas = np.asarray(
        gcr_cfg.get("alphas")
        or ridge_cfg.get(
            "alphas",
            [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
        ),
        dtype=np.float64,
    )
    standardize = bool(
        gcr_cfg.get(
            "standardize_features",
            ridge_cfg.get("standardize_features", True),
        )
    )

    xy_kwargs = dict(
        repo=repo,
        out_root=out_root,
        monkey=monkey,
        model_slug=slug,
        feature_layer=layer,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_size=spatial,
        register_root=register_root,
    )

    print(
        f"Train non-letters: n={len(train_df)}  "
        f"Test letters {test_session}: n={len(test_df)}  "
        f"stimuli={sorted(test_df['stimulus_id'].astype(str).unique())}",
        flush=True,
    )
    print(f"Fit pixels: {int(fit_mask.sum())}  out={out_dir}", flush=True)

    x_train, y_train = build_schira_xy(train_df, **xy_kwargs)
    result = fit_global_channel_ridge(
        x_train,
        y_train,
        valid=fit_mask,
        alphas=alphas,
        standardize_features=standardize,
    )
    print(
        f"Fit done: alpha={result.alpha:.4g} intercept={result.intercept:.6g} "
        f"n_samples={result.n_samples}",
        flush=True,
    )

    x_te, y_te = build_schira_xy(test_df, **xy_kwargs)
    y_hat = predict_global_channel_ridge(x_te, result)
    rs = [
        masked_pearson_r(y_te[i], y_hat[i], result.valid)
        for i in range(y_te.shape[0])
    ]
    r_mean = float(np.nanmean(rs))
    r_med = float(np.nanmedian(rs))
    print(f"Trial-wise masked r: mean={r_mean:.3f} median={r_med:.3f}", flush=True)

    # Per-stimulus fold means + trial r
    per_stim_rows = []
    samples = []
    stim_ids = sorted(test_df["stimulus_id"].astype(str).unique())
    for sid_s in stim_ids:
        mask_rows = test_df["stimulus_id"].astype(str).to_numpy() == sid_s
        o_mean = np.nanmean(y_te[mask_rows], axis=0).astype(np.float32)
        r_mean_map = np.nanmean(y_hat[mask_rows], axis=0).astype(np.float32)
        rs_s = [rs[i] for i, m in enumerate(mask_rows) if m]
        per_stim_rows.append(
            {
                "stimulus_id": sid_s,
                "n_trials": int(mask_rows.sum()),
                "r_mean_masked": float(np.nanmean(rs_s)),
                "r_median_masked": float(np.nanmedian(rs_s)),
            }
        )
        samples.append(
            (
                {
                    "date": test_session,
                    "condition": sid_s,
                    "stimulus_label": sid_s,
                    "shape_type": "letter",
                    "trial_global_id": 0,
                    "split": "letter_test_mean",
                    "trial_dataset": "",
                },
                apply_mask_nan(o_mean, result.valid),
                apply_mask_nan(r_mean_map, result.valid),
            )
        )

    plot_reconstruction_grid(
        samples,
        out_dir / "all_letters_orig_recon.png",
        title=(
            f"Global channel ridge · letters {test_session} "
            f"(train=non-letters only)\n{slug}/{layer} · {window_id}"
        ),
    )

    mean_o = np.nanmean(y_te, axis=0).astype(np.float32)
    mean_r = np.nanmean(y_hat, axis=0).astype(np.float32)
    mean_d = (mean_r - mean_o).astype(np.float32)
    (vmin, vmax), (rvmin, rvmax) = shared_orig_recon_residual_clims(
        [apply_mask_nan(mean_o, result.valid)],
        [apply_mask_nan(mean_r, result.valid)],
    )
    plot_pixel_mean_maps(
        apply_mask_nan(mean_o, result.valid),
        apply_mask_nan(mean_r, result.valid),
        apply_mask_nan(mean_d, result.valid),
        out_dir / "sanity_orig_recon_residual.png",
        title=f"Letters {test_session} · pooled mean · global_channel_ridge",
        vmin=vmin,
        vmax=vmax,
        residual_vmin=rvmin,
        residual_vmax=rvmax,
    )

    # Pixel-r (encoding) vs odd/even OE (per-stimulus half-means stacked)
    enc_r = pixel_correlation_across_trials(y_te, y_hat)
    odd_list = []
    even_list = []
    for sid_s in stim_ids:
        mask_rows = test_df["stimulus_id"].astype(str).to_numpy() == sid_s
        maps = y_te[mask_rows]
        if maps.shape[0] < 2:
            continue
        o_m, e_m = _odd_even_means(maps)
        odd_list.append(o_m)
        even_list.append(e_m)
    if len(odd_list) >= 2:
        oe_r_map = pixel_correlation_across_trials(
            np.stack(odd_list, axis=0),
            np.stack(even_list, axis=0),
        )
    else:
        oe_r_map = np.full(spatial, np.nan, dtype=np.float32)

    hull = result.valid
    enc_mean_r = float(masked_map_summary(enc_r, hull)["mean"])
    oe_mean_r = float(masked_map_summary(oe_r_map, hull)["mean"])
    print(
        f"Pixel-r encoding mean={enc_mean_r:.3f}  OE mean={oe_mean_r:.3f}",
        flush=True,
    )

    plot_pixel_correlation_heatmap(
        apply_mask_nan(enc_r, hull),
        out_dir / "encoding_pixel_r.png",
        title=f"Encoding pixel-r · letters {test_session}",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_pixel_correlation_heatmap(
        apply_mask_nan(oe_r_map, hull),
        out_dir / "oe_pixel_r.png",
        title=f"Odd/even pixel-r · letters {test_session}",
        vmin=-1.0,
        vmax=1.0,
    )

    pd.DataFrame(per_stim_rows).to_csv(out_dir / "per_stimulus_r.csv", index=False)
    np.save(out_dir / "weights.npy", result.weights)
    np.save(out_dir / "valid.npy", result.valid)
    np.save(out_dir / "fold_mean_orig.npy", mean_o)
    np.save(out_dir / "fold_mean_recon.npy", mean_r)
    joblib.dump(result, out_dir / "global_channel_ridge.joblib")

    metrics = {
        "readout": "global_channel_ridge",
        "train": "non_letters_only",
        "test_session": test_session,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_stimuli": len(stim_ids),
        "stimulus_ids": stim_ids,
        "r_mean_masked": r_mean,
        "r_median_masked": r_med,
        "pixel_mean_r_masked": enc_mean_r,
        "oe_mean_r_masked": oe_mean_r,
        "alpha": float(result.alpha),
        "intercept": float(result.intercept),
        "n_fit_pixels": int(result.valid.sum()),
        "n_samples": int(result.n_samples),
        "loss_roi": loss_roi_raw,
        "train_mask_meta": train_meta,
        "model_slug": slug,
        "feature_layer": layer,
        "window_id": window_id,
        "schira_set": schira_set,
        "anchor_session": anchor,
        "standardize_features": standardize,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "params.yaml").write_text(yaml.safe_dump(metrics, sort_keys=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
