#!/usr/bin/env python3
"""Train Schira local ridge on all sessions (anchor-aligned), flatten-style splits."""

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
from src.encoding.ridge import pearson_r
from src.encoding.ridge_plotting import (
    plot_reconstruction_grid,
    select_one_trial_per_condition,
)
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import parse_loss_roi_arg, resolve_loss_roi
from src.evaluation.mask import apply_mask_nan, masked_pearson_r
from src.paths import project_root, resolve_data_path
from src.schira_encoding.io import (
    build_schira_xy,
    resolve_output_root,
    stimulus_label,
)
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.model import fit_local_ridge, predict_local_ridge
from src.schira_encoding.schema import local_ridge_output_dir, lut_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument(
        "--window-config",
        type=Path,
        default=None,
        help="Default: window_config from --config (raw evoked_35_46).",
    )
    p.add_argument("--feature-layer", type=str, default=None)
    p.add_argument(
        "--loss-roi",
        type=str,
        default=None,
        help=(
            "Train/eval ROI ∩ LUT.valid. Default: loss_roi from --config "
            "(noise_ceiling_hull)."
        ),
    )
    p.add_argument(
        "--anchor-only",
        action="store_true",
        help="Train/eval only on anchor_session trials (debug).",
    )
    p.add_argument(
        "--date-prefix",
        type=str,
        default=None,
        help="Keep only sessions whose date starts with this (e.g. 100718).",
    )
    p.add_argument(
        "--n-plot-examples",
        type=int,
        default=4,
        help="Condition examples for recon grid (prefer test split).",
    )
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
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "layer3"))
    slug = model_slug(model_cfg)
    loss_roi_raw = str(
        args.loss_roi
        if args.loss_roi is not None
        else merged.get("loss_roi", "noise_ceiling_hull")
    )

    stimuli_cfg = _load_yaml(
        repo / str(merged.get("stimuli_config", "configs/stimuli/default.yaml"))
    )
    canvas = int(stimuli_cfg.get("canvas_size", 210))

    out_root = resolve_output_root(merged["paths"]["schira_encoding_root"], repo)
    lut_file = lut_path(
        out_root,
        monkey,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_h=spatial[0],
        spatial_w=spatial[1],
        canvas_size=canvas,
        input_size=input_size,
    )
    lut = SchiraLUT.load(lut_file)

    pairs_root = resolve_data_path(merged["paths"]["encoding_pairs_root"], repo)
    start = int(merged["start_frame"])
    end = int(merged["end_frame"])
    window_id = merged.get("window_id") or f"win_{start:04d}_{end:04d}"
    pairs = pd.read_parquet(
        encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    )
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    if args.date_prefix:
        pairs = pairs[
            pairs["date"].astype(str).str.startswith(str(args.date_prefix))
        ].copy()
    if args.anchor_only:
        pairs = pairs[pairs["date"].astype(str) == anchor].copy()
    if pairs.empty:
        raise RuntimeError("No trials left after filters")

    register_root = None
    if not args.anchor_only and "session_register_root" in merged.get("paths", {}):
        register_root = resolve_output_root(
            merged["paths"]["session_register_root"], repo
        )

    train_df = pairs[pairs["split"].astype(str) == "train"].copy()
    if train_df.empty:
        raise RuntimeError("No train-split trials")

    print(
        f"Loading XY: train={len(train_df)} all={len(pairs)} "
        f"anchor_only={args.anchor_only}",
        flush=True,
    )
    x_train, y_train = build_schira_xy(
        train_df,
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

    ridge_cfg = merged.get("ridge") or {}
    alphas = np.asarray(
        ridge_cfg.get("alphas", [1.0, 10.0, 100.0]), dtype=np.float64
    )
    print(
        f"Fitting local ridge: n={x_train.shape[0]} C={x_train.shape[1]} "
        f"fit_pixels={int(fit_mask.sum())} "
        f"(LUT.valid ∩ loss-roi={loss_roi_raw})",
        flush=True,
    )
    result = fit_local_ridge(
        x_train,
        y_train,
        valid=fit_mask,
        alphas=alphas,
        standardize_features=bool(ridge_cfg.get("standardize_features", True)),
        alpha_per_pixel=bool(ridge_cfg.get("alpha_per_pixel", True)),
    )

    metrics: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "monkey": monkey,
        "schira_set": schira_set,
        "anchor_session": anchor,
        "window_id": window_id,
        "model_slug": slug,
        "feature_layer": layer,
        "n_train": int(len(train_df)),
        "n_all": int(len(pairs)),
        "n_channels": int(result.n_channels),
        "n_fit_pixels": int(result.valid.sum()),
        "anchor_only": bool(args.anchor_only),
        "date_prefix": args.date_prefix,
        "lut_path": str(lut_file),
        "loss_roi": loss_roi_raw,
        "train_mask_meta": train_meta,
    }

    split_scores: dict[str, list[float]] = {}
    for split_name, split_df in pairs.groupby(pairs["split"].astype(str)):
        split_df = split_df.reset_index(drop=True)
        if split_df.empty:
            continue
        x_s, y_s = build_schira_xy(
            split_df,
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
        y_hat = predict_local_ridge(x_s, result)
        rs_unmasked = [
            pearson_r(y_s[i], y_hat[i]) for i in range(y_s.shape[0])
        ]
        rs = [
            masked_pearson_r(y_s[i], y_hat[i], result.valid)
            for i in range(y_s.shape[0])
        ]
        metrics[f"r_mean_{split_name}"] = float(np.nanmean(rs_unmasked))
        metrics[f"r_median_{split_name}"] = float(np.nanmedian(rs_unmasked))
        metrics[f"r_mean_{split_name}_masked"] = float(np.nanmean(rs))
        metrics[f"r_median_{split_name}_masked"] = float(np.nanmedian(rs))
        split_scores[str(split_name)] = rs
        print(
            f"  r ({split_name}): mean={np.nanmean(rs):.3f} "
            f"median={np.nanmedian(rs):.3f} (masked)  "
            f"n={len(split_df)}",
            flush=True,
        )

    metrics["split_r"] = {k: [float(x) for x in v] for k, v in split_scores.items()}

    out_dir = local_ridge_output_dir(
        out_root,
        monkey,
        window_id,
        slug,
        layer,
        schira_set=schira_set,
        anchor_session=anchor,
        run_tag=(f"dates-{args.date_prefix}" if args.date_prefix else None),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "weights.npy", result.weights)
    np.save(out_dir / "intercepts.npy", result.intercepts)
    np.save(out_dir / "alphas.npy", result.alphas)
    np.save(out_dir / "valid.npy", result.valid)
    if result.feature_mean is not None:
        np.save(out_dir / "feature_mean.npy", result.feature_mean)
        np.save(out_dir / "feature_scale.npy", result.feature_scale)
    joblib.dump(result, out_dir / "local_ridge.joblib")
    (out_dir / "meta.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Example reconstructions grouped by shape_type when available.
    prefer = str(ridge_cfg.get("plot_prefer_split", "test"))
    plot_src = pairs
    if (plot_src["split"].astype(str) == prefer).any():
        plot_src = plot_src[plot_src["split"].astype(str) == prefer]
    plot_rows = select_one_trial_per_condition(plot_src, prefer_split=prefer)

    def _append_samples(frame: pd.DataFrame) -> list:
        if frame.empty:
            return []
        x_p, y_p = build_schira_xy(
            frame,
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
        y_hat_p = predict_local_ridge(x_p, result)
        out = []
        for i, row in enumerate(frame.itertuples(index=False)):
            out.append(
                (
                    {
                        "date": str(row.date),
                        "condition": str(row.condition),
                        "shape_type": str(getattr(row, "shape_type", "")),
                        "stimulus_label": stimulus_label(row),
                        "trial_global_id": int(row.trial_global_id),
                        "split": str(row.split),
                        "trial_dataset": str(getattr(row, "trial_dataset", "")),
                    },
                    apply_mask_nan(y_p[i], result.valid),
                    apply_mask_nan(y_hat_p[i], result.valid),
                )
            )
        return out

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    all_samples = _append_samples(plot_rows.head(int(args.n_plot_examples)))
    if all_samples:
        plot_reconstruction_grid(
            all_samples,
            plot_dir / "reconstructions_by_condition.png",
            title=(
                f"Schira local ridge · {slug}/{layer}\n"
                f"set={schira_set} anchor={anchor} · {window_id}"
                + (f" · dates={args.date_prefix}" if args.date_prefix else "")
            ),
        )

    # One orig|recon grid per shape_type across all splits (prefer test trial).
    if "shape_type" in pairs.columns:
        shape_src = select_one_trial_per_condition(pairs, prefer_split=prefer)
        for shape, shape_df in shape_src.groupby(shape_src["shape_type"].astype(str)):
            samples = _append_samples(shape_df)
            if not samples:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in shape)
            plot_reconstruction_grid(
                samples,
                plot_dir / f"reconstructions_by_shape__{safe}.png",
                title=(
                    f"Schira local ridge · shape={shape} · {slug}/{layer}\n"
                    f"set={schira_set} · {window_id}"
                    + (f" · dates={args.date_prefix}" if args.date_prefix else "")
                ),
            )

    print(json.dumps({k: v for k, v in metrics.items() if k != "split_r"}, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
