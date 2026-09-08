#!/usr/bin/env python3
"""Schira local-ridge LOO using the same fold logic as flatten ridge.

Default: **Protocol B** (stimulus_id LOO) from ``src.loo.folds``, same
``heldout_list.yaml``. First use-case: ``--date-prefix 100718``.

Example::

    scripts/py scripts/17_run_schira_loo.py \\
      --model-config configs/models/vgg16.yaml \\
      --window-config configs/windows/evoked_35_46.yaml \\
      --feature-layer block1 \\
      --protocol B \\
      --date-prefix 100718 \\
      --loss-roi noise_ceiling_hull
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.encoding.ridge import pearson_r
from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import parse_loss_roi_arg, resolve_loss_roi
from src.evaluation.mask import apply_mask_nan, masked_pearson_r
from src.evaluation.plotting import plot_pixel_mean_maps
from src.loo.folds import (
    build_protocol_a_folds,
    build_protocol_b_folds,
    build_protocol_c_folds,
    expand_stimulus_id_patterns,
    filter_folds_excluding_train_only_dates,
    load_heldout_config,
    strip_train_only_from_protocol_b_test,
)
from src.paths import project_root, resolve_data_path
from src.schira_encoding.io import (
    build_schira_xy,
    resolve_output_root,
    stimulus_label,
)
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.model import fit_local_ridge, predict_local_ridge
from src.schira_encoding.schema import local_ridge_loo_dir, lut_path
from src.stimuli.identity import attach_stimulus_ids

HELDOUT_DEFAULT = Path("experiments/loo_encoding/heldout_list.yaml")


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
    p.add_argument("--protocol", choices=["A", "B", "C"], default="B")
    p.add_argument(
        "--heldout",
        type=Path,
        default=repo / HELDOUT_DEFAULT,
        help="Same held-out list as flatten LOO.",
    )
    p.add_argument(
        "--date-prefix",
        type=str,
        default=None,
        help="Restrict pairs to sessions starting with this (e.g. 100718).",
    )
    p.add_argument(
        "--stimuli",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset / globs of stimulus_ids (default: heldout list ∩ data).",
    )
    p.add_argument(
        "--loss-roi",
        type=str,
        default=None,
        help=(
            "Train/eval ROI ∩ LUT.valid. Default: loss_roi from --config "
            "(noise_ceiling_hull; same as flatten LOO)."
        ),
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional leaf suffix (e.g. letters) to avoid clobbering another LOO run.",
    )
    p.add_argument(
        "--all-sessions",
        action="store_true",
        help=(
            "Protocol C only: one fold per (date, condition) of each held-out "
            "stimulus (needed for letters on 201118c/d). Default C picks one "
            "test group per stimulus_id."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
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
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "block1"))
    slug = model_slug(model_cfg)
    protocol = str(args.protocol).upper()

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
    if args.date_prefix:
        pairs = pairs[
            pairs["date"].astype(str).str.startswith(str(args.date_prefix))
        ].copy()
    if pairs.empty:
        raise RuntimeError("No pairs left after filters")

    pairs = attach_stimulus_ids(pairs)
    held_cfg = load_heldout_config(args.heldout)
    patterns = list(args.stimuli) if args.stimuli else list(held_cfg.stimulus_ids)
    available = sorted(pairs["stimulus_id"].astype(str).unique())
    heldout_ids = expand_stimulus_id_patterns(patterns, available_ids=available)
    # Only ids present in this filtered pairs table.
    heldout_ids = [s for s in heldout_ids if s in set(available)]
    if not heldout_ids:
        raise RuntimeError(
            f"No held-out stimulus_ids present in filtered pairs. "
            f"available={available}"
        )

    if protocol == "A":
        folds = build_protocol_a_folds(pairs, heldout_ids)
        folds = filter_folds_excluding_train_only_dates(
            folds, held_cfg.train_only_sessions
        )
    elif protocol == "B":
        folds = build_protocol_b_folds(pairs, heldout_ids)
        folds = strip_train_only_from_protocol_b_test(
            folds, held_cfg.train_only_sessions
        )
    else:
        folds = build_protocol_c_folds(
            pairs, heldout_ids, all_sessions=bool(args.all_sessions)
        )
        folds = filter_folds_excluding_train_only_dates(
            folds, held_cfg.train_only_sessions
        )
    if not folds:
        raise RuntimeError("No folds left after train-only session filtering")

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

    out_dir = local_ridge_loo_dir(
        out_root,
        monkey,
        window_id,
        slug,
        layer,
        schira_set=schira_set,
        anchor_session=anchor,
        protocol=protocol,
        loss_roi=loss_roi_raw,
        date_prefix=args.date_prefix,
        run_tag=args.run_tag,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folds: {len(folds)}  out={out_dir}")
    print(f"Held-out ids: {heldout_ids}")
    if held_cfg.train_only_sessions:
        print(
            f"Train-only sessions (excluded from Protocol A/C test folds; "
            f"stripped from Protocol B test): {held_cfg.train_only_sessions}"
        )
    if protocol == "C":
        print(f"Protocol C all_sessions={bool(args.all_sessions)}")
    print(f"Fit pixels: {int(fit_mask.sum())} (LUT.valid ∩ loss-roi={loss_roi_raw})")

    if args.dry_run:
        for spec, fold_df in folds:
            test = fold_df[fold_df["loo_split"] == "test"]
            test_sess = sorted(
                test["h5_session"].astype(str).unique()
                if "h5_session" in test.columns
                else test["date"].astype(str).unique()
            )
            print(
                f"  {spec.fold_id}: train={spec.n_train} val={spec.n_val} "
                f"test={spec.n_test} test_sessions={test_sess} "
                f"leakage_ok={spec.leakage_ok}"
            )
        return

    ridge_cfg = merged.get("ridge") or {}
    alphas = np.asarray(
        ridge_cfg.get("alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]),
        dtype=np.float64,
    )

    summary_rows: list[dict] = []
    for spec, fold_df in folds:
        fold_dir = out_dir / spec.fold_id
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.is_file() and not args.overwrite:
            print(f"skip existing {spec.fold_id}")
            summary_rows.append(json.loads(metrics_path.read_text()))
            continue

        train_df = fold_df[fold_df["loo_split"] == "train"].copy()
        val_df = fold_df[fold_df["loo_split"] == "val"].copy()
        test_df = fold_df[fold_df["loo_split"] == "test"].copy()
        print(
            f"\n=== {spec.fold_id}  train={len(train_df)} "
            f"val={len(val_df)} test={len(test_df)} ===",
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
        result = fit_local_ridge(
            x_train,
            y_train,
            valid=fit_mask,
            alphas=alphas,
            standardize_features=bool(ridge_cfg.get("standardize_features", True)),
            alpha_per_pixel=bool(ridge_cfg.get("alpha_per_pixel", True)),
        )

        metrics: dict = {
            "fold_id": spec.fold_id,
            "protocol": protocol,
            "heldout_stimulus_id": spec.heldout_stimulus_id,
            "heldout_date": spec.heldout_date,
            "heldout_condition": spec.heldout_condition,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_fit_pixels": int(result.valid.sum()),
            "leakage_ok": bool(spec.leakage_ok),
            "notes": spec.notes,
            "date_prefix": args.date_prefix,
            "loss_roi": loss_roi_raw,
            "train_mask_meta": train_meta,
            "model_slug": slug,
            "feature_layer": layer,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        for split_name, split_df in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ):
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
            rs = [
                masked_pearson_r(y_s[i], y_hat[i], result.valid)
                for i in range(y_s.shape[0])
            ]
            metrics[f"r_mean_{split_name}_masked"] = float(np.nanmean(rs))
            metrics[f"r_median_{split_name}_masked"] = float(np.nanmedian(rs))
            print(
                f"  r ({split_name}): mean={np.nanmean(rs):.3f} "
                f"median={np.nanmedian(rs):.3f}",
                flush=True,
            )

        # Test mean maps + one-per-condition recon grid
        x_te, y_te = build_schira_xy(
            test_df,
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
        y_hat_te = predict_local_ridge(x_te, result)
        mean_o = np.nanmean(y_te, axis=0).astype(np.float32)
        mean_r = np.nanmean(y_hat_te, axis=0).astype(np.float32)
        mean_d = (mean_r - mean_o).astype(np.float32)

        fold_dir.mkdir(parents=True, exist_ok=True)
        plot_pixel_mean_maps(
            apply_mask_nan(mean_o, result.valid),
            apply_mask_nan(mean_r, result.valid),
            apply_mask_nan(mean_d, result.valid),
            fold_dir / "sanity_orig_recon_residual.png",
            title=(
                f"{spec.fold_id} · held-out {spec.heldout_stimulus_id}\n"
                f"Schira local ridge · {slug}/{layer} · test mean"
            ),
        )

        # One trial per (date, condition) for held-out stimulus
        plot_df = (
            test_df.sort_values(["date", "condition", "trial_global_id"])
            .drop_duplicates(["date", "condition"], keep="first")
            .reset_index(drop=True)
        )
        if len(plot_df):
            x_p, y_p = build_schira_xy(
                plot_df,
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
            samples = []
            for i, row in enumerate(plot_df.itertuples(index=False)):
                samples.append(
                    (
                        {
                            "date": str(row.date),
                            "condition": str(row.condition),
                            "shape_type": str(getattr(row, "shape_type", "")),
                            "stimulus_label": stimulus_label(row),
                            "trial_global_id": int(row.trial_global_id),
                            "split": "loo_test",
                            "trial_dataset": str(
                                getattr(row, "trial_dataset", "")
                            ),
                        },
                        apply_mask_nan(y_p[i], result.valid),
                        apply_mask_nan(y_hat_p[i], result.valid),
                    )
                )
            plot_reconstruction_grid(
                samples,
                fold_dir / "reconstructions_by_condition.png",
                title=(
                    f"{spec.fold_id} · held-out {spec.heldout_stimulus_id}\n"
                    f"Schira LOO protocol {protocol}"
                ),
            )

        np.save(fold_dir / "fold_mean_orig.npy", mean_o)
        np.save(fold_dir / "fold_mean_recon.npy", mean_r)
        np.save(fold_dir / "valid.npy", result.valid)
        joblib.dump(result, fold_dir / "local_ridge.joblib")
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        fold_df.to_parquet(fold_dir / "fold_pairs.parquet", index=False)
        summary_rows.append(metrics)

    # Summary table
    summary_csv = out_dir / "loo_summary.csv"
    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        print(f"\nWrote {summary_csv}")
        show = pd.DataFrame(summary_rows)
        cols = [
            c
            for c in (
                "fold_id",
                "heldout_stimulus_id",
                "n_train",
                "n_test",
                "r_mean_test_masked",
                "r_median_test_masked",
            )
            if c in show.columns
        ]
        print(show[cols].to_string(index=False))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "date_prefix": args.date_prefix,
        "run_tag": args.run_tag,
        "heldout_ids": heldout_ids,
        "train_only_sessions": list(held_cfg.train_only_sessions),
        "all_sessions": bool(args.all_sessions) if protocol == "C" else None,
        "n_folds": len(folds),
        "loss_roi": loss_roi_raw,
        "window_id": window_id,
        "model_slug": slug,
        "feature_layer": layer,
        "schira_set": schira_set,
        "anchor_session": anchor,
        "monkey": monkey,
        "heldout_config": str(args.heldout),
    }
    (out_dir / "params.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    folds_index = {
        "n_folds": len(summary_rows),
        "folds": [
            {
                "fold_id": row.get("fold_id"),
                "heldout_stimulus_id": row.get("heldout_stimulus_id"),
                "n_train": row.get("n_train"),
                "n_test": row.get("n_test"),
                "r_mean_test_masked": row.get("r_mean_test_masked"),
            }
            for row in summary_rows
        ],
    }
    (out_dir / "folds_index.yaml").write_text(
        yaml.safe_dump(folds_index, sort_keys=False)
    )


if __name__ == "__main__":
    main()
