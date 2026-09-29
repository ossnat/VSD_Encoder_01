"""Per-fold Schira camera fit, then global-channel ridge.

Same Protocol B split as ``scripts/20_run_schira_global_channel_loo.py``.
Each fold starts from the YAML camera, drops the held-out stimulus, and
moves origin, rotation, and scale to match the other stimuli's mean VSD
maps. The ridge then uses that fold's lookup table.

Example::

  scripts/py scripts/22_run_global_channel_schira_opt.py \\
    --model-config configs/models/vgg16.yaml \\
    --feature-layer block1_prepool
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
from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import parse_loss_roi_arg, resolve_loss_roi
from src.evaluation.mask import apply_mask_nan, masked_map_summary, masked_pearson_r
from src.evaluation.pixel_correlation import pixel_correlation_across_trials
from src.evaluation.plotting import (
    plot_loo_all_shapes_orig_recon,
    plot_pixel_correlation_heatmap,
    plot_pixel_mean_maps,
    shared_orig_recon_residual_clims,
)
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
from src.plotting_colormaps import register_mapgeog
from src.retinotopy.params import load_schira_set
from src.schira_encoding.geometry_fit import (
    OPTIMIZE_NONE,
    GeometryFitConfig,
    build_anchor_stimulus_means,
    fit_schira_geometry,
    frozen_geometry_result,
    keep_geometry_fit_stimulus,
    normalize_optimize,
)
from src.schira_encoding.global_channel_model import (
    fit_global_channel_ridge,
    predict_global_channel_ridge,
)
from src.schira_encoding.io import build_schira_xy, resolve_output_root, stimulus_label
from src.schira_encoding.lut import build_schira_lut
from src.schira_encoding.schema import global_channel_ridge_schira_opt_dir
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig

HELDOUT_DEFAULT = Path("experiments/loo_encoding/heldout_non_letters.yaml")


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict) -> RenderConfig:
    return RenderConfig(
        canvas_size=int(stimuli_cfg.get("canvas_size", 210)),
        pixels_per_deg=float(stimuli_cfg.get("pixels_per_deg", 35.0)),
        quadrant_extent_deg=float(stimuli_cfg.get("quadrant_extent_deg", 6.0)),
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
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
        "--geometry-config",
        type=Path,
        default=repo / "configs/schira_encoding/geometry_fit.yaml",
    )
    p.add_argument("--protocol", choices=["A", "B", "C"], default="B")
    p.add_argument("--heldout", type=Path, default=repo / HELDOUT_DEFAULT)
    p.add_argument("--date-prefix", type=str, default=None)
    p.add_argument("--stimuli", type=str, nargs="*", default=None)
    p.add_argument("--loss-roi", type=str, default=None)
    p.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["none", "schira", "camera", "all"],
        help="Override geometry_fit.yaml optimize (none|schira|camera|all).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="schira_opt",
        help="Directory prefix under global_channel_ridge/ (leaf is {run-name}_{optimize}).",
    )
    p.add_argument("--all-sessions", action="store_true")
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
    geom_raw = _load_yaml(args.geometry_config)
    if args.optimize is not None:
        geom_raw["optimize"] = args.optimize
    geom_cfg = GeometryFitConfig.from_mapping(geom_raw)
    optimize = normalize_optimize(geom_cfg.optimize)

    monkey = str(merged["monkey"])
    schira_set = str(merged["schira_set"])
    anchor = str(merged["anchor_session"])
    spatial = tuple(int(x) for x in merged["spatial_size"])
    input_size = int(merged.get("input_size", model_cfg.get("input_size", 224)))
    layer = str(args.feature_layer or model_cfg.get("feature_layer", "block1_prepool"))
    slug = model_slug(model_cfg)
    protocol = str(args.protocol).upper()

    stimuli_cfg = _load_yaml(
        repo / str(merged.get("stimuli_config", "configs/stimuli/default.yaml"))
    )
    render_cfg = _render_config(stimuli_cfg)
    canvas = int(render_cfg.canvas_size)
    max_ecc = merged.get("max_ecc_deg")
    max_ecc_deg = None if max_ecc is None else float(max_ecc)

    schira_config = repo / str(merged.get("schira_config", "configs/schira/sets.yaml"))
    _set_name, init_params, init_affine, _block = load_schira_set(
        schira_config, set_name=schira_set
    )

    out_root = resolve_output_root(merged["paths"]["schira_encoding_root"], repo)
    register_root = resolve_output_root(
        merged["paths"]["session_register_root"], repo
    )
    features_root = resolve_data_path(merged["paths"]["dl_features_stimuli_root"], repo)

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
    heldout_ids = [s for s in heldout_ids if s in set(available)]
    if not heldout_ids:
        raise RuntimeError(f"No held-out stimulus_ids in pairs. available={available}")

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
    init_lut = build_schira_lut(
        params=init_params,
        affine=init_affine,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_size=spatial,
        render_cfg=render_cfg,
        input_size=input_size,
        max_ecc_deg=max_ecc_deg,
    )
    geom_mask = init_lut.valid.copy()
    if train_roi is not None:
        geom_mask &= np.asarray(train_roi, dtype=bool)

    out_dir = global_channel_ridge_schira_opt_dir(
        out_root,
        monkey,
        window_id,
        slug,
        layer,
        schira_set=schira_set,
        anchor_session=anchor,
        run_name=str(args.run_name),
        optimize=optimize,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Folds: {len(folds)}  optimize={optimize}  out={out_dir}")
    print(f"Held-out ids: {heldout_ids}")
    print(
        f"Geometry mask pixels: {int(geom_mask.sum())} "
        f"(start LUT.valid ∩ loss-roi={loss_roi_raw})"
    )
    if args.dry_run:
        for spec, _fold_df in folds:
            print(
                f"  {spec.fold_id}: train={spec.n_train} val={spec.n_val} "
                f"test={spec.n_test} leakage_ok={spec.leakage_ok}"
            )
        return

    means: dict = {}
    if optimize != OPTIMIZE_NONE:
        print("Building per-stimulus mean VSD maps...", flush=True)
        mean_sessions = {anchor} if geom_cfg.anchor_session_only else None
        means = build_anchor_stimulus_means(
            pairs,
            repo=repo,
            spatial_size=spatial,
            anchor_session=anchor,
            monkey=monkey,
            register_root=register_root,
            background_gray=float(render_cfg.background_gray),
            canvas_size=canvas,
            sessions=mean_sessions,
            per_session=bool(geom_cfg.per_session_examples),
        )
        means = {
            key: stim
            for key, stim in means.items()
            if keep_geometry_fit_stimulus(stim.stimulus_id, geom_cfg)
        }
        print(
            f"Stimulus means: {len(means)} "
            f"(anchor_only={geom_cfg.anchor_session_only} "
            f"per_session={geom_cfg.per_session_examples} "
            f"ids={sorted({s.stimulus_id for s in means.values()})})",
            flush=True,
        )

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

    summary_rows: list[dict] = []
    pooled_orig: list[np.ndarray] = []
    pooled_recon: list[np.ndarray] = []
    pooled_valid: np.ndarray | None = None
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
        held = str(spec.heldout_stimulus_id)
        if optimize == OPTIMIZE_NONE:
            geom = frozen_geometry_result(init_params, init_affine)
            fold_lut = init_lut
            print("  geometry: YAML start (optimize=none)", flush=True)
        else:
            fit_ids = set(train_df["stimulus_id"].astype(str)) | set(
                val_df["stimulus_id"].astype(str)
            )
            fit_ids.discard(held)
            fit_stimuli = {
                key: stim
                for key, stim in means.items()
                if stim.stimulus_id != held and stim.stimulus_id in fit_ids
            }
            geom = fit_schira_geometry(
                fit_stimuli,
                init_params=init_params,
                init_affine=init_affine,
                mask=geom_mask,
                pixels_per_deg=float(render_cfg.pixels_per_deg),
                cfg=geom_cfg,
            )
            fold_lut = build_schira_lut(
                params=geom.params,
                affine=geom.affine,
                schira_set=schira_set,
                anchor_session=anchor,
                spatial_size=spatial,
                render_cfg=render_cfg,
                input_size=input_size,
                max_ecc_deg=max_ecc_deg,
            )
            delta = geom.delta_mapping(init_affine, init_params)
            print(
                f"  geometry r {geom.correlation_before:.3f} → {geom.correlation_after:.3f}  "
                f"d_origin=({delta['origin_x']:.2f}, {delta['origin_y']:.2f}) px  "
                f"d_rot={delta['rotation_deg']:.2f} deg  "
                f"scale={delta['scale_ratio']:.3f}  "
                f"a={delta['a_ratio']:.3f}  alpha={delta['alpha_ratio']:.3f}  "
                f"k={delta['k_ratio']:.3f}  steps={geom.n_steps}",
                flush=True,
            )
        delta = geom.delta_mapping(init_affine, init_params)
        fit_mask = fold_lut.valid.copy()
        if train_roi is not None:
            fit_mask &= np.asarray(train_roi, dtype=bool)

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
            lut=None if optimize == OPTIMIZE_NONE else fold_lut,
            features_root=features_root,
        )
        x_train, y_train = build_schira_xy(train_df, **xy_kwargs)
        result = fit_global_channel_ridge(
            x_train,
            y_train,
            valid=fit_mask,
            alphas=alphas,
            standardize_features=standardize,
        )

        metrics: dict = {
            "fold_id": spec.fold_id,
            "protocol": protocol,
            "readout": "global_channel_ridge",
            "geometry": f"schira_opt_{optimize}",
            "optimize": optimize,
            "heldout_stimulus_id": spec.heldout_stimulus_id,
            "heldout_date": spec.heldout_date,
            "heldout_condition": spec.heldout_condition,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_fit_pixels": int(result.valid.sum()),
            "n_samples": int(result.n_samples),
            "alpha": float(result.alpha),
            "intercept": float(result.intercept),
            "leakage_ok": bool(spec.leakage_ok),
            "notes": spec.notes,
            "date_prefix": args.date_prefix,
            "loss_roi": loss_roi_raw,
            "train_mask_meta": train_meta,
            "model_slug": slug,
            "feature_layer": layer,
            "standardize_features": standardize,
            "geom_r_before": geom.correlation_before,
            "geom_r_after": geom.correlation_after,
            "geom_contrast_sign": geom.contrast_sign,
            "geom_n_stimuli": geom.n_stimuli,
            "geom_n_steps": geom.n_steps,
            "geom_origin_x": geom.affine.origin_x,
            "geom_origin_y": geom.affine.origin_y,
            "geom_rotation_deg": geom.affine.rotation_deg,
            "geom_pixels_per_unit": geom.affine.pixels_per_unit,
            "geom_d_origin_x": delta["origin_x"],
            "geom_d_origin_y": delta["origin_y"],
            "geom_d_rotation_deg": delta["rotation_deg"],
            "geom_scale_ratio": delta["scale_ratio"],
            "geom_a": geom.params.a,
            "geom_alpha_schira": geom.params.alpha,
            "geom_a_ratio": delta["a_ratio"],
            "geom_alpha_ratio": delta["alpha_ratio"],
            "geom_k": geom.params.k,
            "geom_k_ratio": delta["k_ratio"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        for split_name, split_df in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ):
            if split_df.empty:
                continue
            x_s, y_s = build_schira_xy(split_df, **xy_kwargs)
            y_hat = predict_global_channel_ridge(x_s, result)
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

        x_te, y_te = build_schira_xy(test_df, **xy_kwargs)
        y_hat_te = predict_global_channel_ridge(x_te, result)
        pixel_r = apply_mask_nan(
            pixel_correlation_across_trials(y_te, y_hat_te), result.valid
        )
        pixel_r_mean = float(masked_map_summary(pixel_r, result.valid)["mean"])
        metrics["r_pixel_test"] = pixel_r_mean
        print(f"  pixel-r (test, across trials): mean={pixel_r_mean:.3f}", flush=True)
        pooled_orig.append(y_te)
        pooled_recon.append(y_hat_te)
        pooled_valid = result.valid if pooled_valid is None else (pooled_valid & result.valid)
        mean_o = np.nanmean(y_te, axis=0).astype(np.float32)
        mean_r = np.nanmean(y_hat_te, axis=0).astype(np.float32)
        mean_d = (mean_r - mean_o).astype(np.float32)
        (vmin, vmax), (rvmin, rvmax) = shared_orig_recon_residual_clims(
            [apply_mask_nan(mean_o, result.valid)],
            [apply_mask_nan(mean_r, result.valid)],
        )

        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "schira_fit.yaml").write_text(
            yaml.safe_dump(
                geom.to_yaml_block(
                    start_affine=init_affine,
                    start_params=init_params,
                    heldout_stimulus_id=held,
                ),
                sort_keys=False,
            )
        )
        fold_lut.save(fold_dir / "lut.npz")
        plot_pixel_mean_maps(
            apply_mask_nan(mean_o, result.valid),
            apply_mask_nan(mean_r, result.valid),
            apply_mask_nan(mean_d, result.valid),
            fold_dir / "sanity_orig_recon_residual.png",
            title=(
                f"{spec.fold_id} · held-out {spec.heldout_stimulus_id}\n"
                f"Schira opt global_channel_ridge · {slug}/{layer} · test mean"
            ),
            vmin=vmin,
            vmax=vmax,
            residual_vmin=rvmin,
            residual_vmax=rvmax,
        )

        plot_df = (
            test_df.sort_values(["date", "condition", "trial_global_id"])
            .drop_duplicates(["date", "condition"], keep="first")
            .reset_index(drop=True)
        )
        if len(plot_df):
            x_p, y_p = build_schira_xy(plot_df, **xy_kwargs)
            y_hat_p = predict_global_channel_ridge(x_p, result)
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
                            "trial_dataset": str(getattr(row, "trial_dataset", "")),
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
                    f"Schira opt global_channel_ridge LOO protocol {protocol}"
                ),
            )

        np.save(fold_dir / "pixel_r_test.npy", pixel_r.astype(np.float32))
        plot_pixel_correlation_heatmap(
            pixel_r,
            fold_dir / "pixel_r_test.png",
            title=(
                f"{spec.fold_id} · per-pixel r across test trials\n"
                f"mean={pixel_r_mean:.3f} · LUT.valid ∩ loss-roi"
            ),
        )
        np.save(fold_dir / "fold_mean_orig.npy", mean_o)
        np.save(fold_dir / "fold_mean_recon.npy", mean_r)
        np.save(fold_dir / "valid.npy", result.valid)
        np.save(fold_dir / "weights.npy", result.weights)
        joblib.dump(result, fold_dir / "global_channel_ridge.joblib")
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        fold_df.to_parquet(fold_dir / "fold_pairs.parquet", index=False)
        summary_rows.append(metrics)

    if pooled_orig and pooled_valid is not None:
        y_all = np.concatenate(pooled_orig, axis=0)
        yhat_all = np.concatenate(pooled_recon, axis=0)
        pooled_r = apply_mask_nan(
            pixel_correlation_across_trials(y_all, yhat_all), pooled_valid
        )
        pooled_mean = float(masked_map_summary(pooled_r, pooled_valid)["mean"])
        overview = out_dir / "overview"
        overview.mkdir(parents=True, exist_ok=True)
        np.save(overview / "pixel_r_test_pooled.npy", pooled_r.astype(np.float32))
        plot_pixel_correlation_heatmap(
            pooled_r,
            overview / "pixel_r_test_pooled.png",
            title=(
                f"Per-pixel r across held-out trials · {optimize}\n"
                f"mean={pooled_mean:.3f} · n={y_all.shape[0]} · LUT.valid ∩ loss-roi"
            ),
        )
        print(f"Pooled pixel-r mean={pooled_mean:.3f}  {overview / 'pixel_r_test_pooled.png'}")

    fold_dirs_done = sorted(
        p
        for p in out_dir.iterdir()
        if p.is_dir() and (p / "fold_mean_orig.npy").is_file()
    )
    if fold_dirs_done:
        overview = out_dir / "overview"
        overview.mkdir(parents=True, exist_ok=True)
        valid = (
            pooled_valid
            if pooled_valid is not None
            else np.load(fold_dirs_done[0] / "valid.npy").astype(bool)
        )
        grid_path = overview / "all_shapes_orig_recon.png"
        plot_loo_all_shapes_orig_recon(
            fold_dirs_done,
            valid=valid,
            out_path=grid_path,
            title=(
                f"Schira opt {optimize} · fold-mean orig|recon · {slug}/{layer}"
            ),
        )
        print(f"Wrote {grid_path}")

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
                "geom_r_before",
                "geom_r_after",
                "geom_d_origin_x",
                "geom_d_origin_y",
                "geom_d_rotation_deg",
                "geom_scale_ratio",
                "geom_a_ratio",
                "geom_alpha_ratio",
                "geom_k_ratio",
                "r_mean_test_masked",
                "r_pixel_test",
                "r_median_test_masked",
            )
            if c in show.columns
        ]
        print(show[cols].to_string(index=False))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "readout": "global_channel_ridge",
        "geometry": f"schira_opt_{optimize}",
        "optimize": optimize,
        "protocol": protocol,
        "date_prefix": args.date_prefix,
        "heldout_ids": heldout_ids,
        "train_only_sessions": list(held_cfg.train_only_sessions),
        "n_folds": len(folds),
        "loss_roi": loss_roi_raw,
        "window_id": window_id,
        "model_slug": slug,
        "feature_layer": layer,
        "schira_set": schira_set,
        "anchor_session": anchor,
        "monkey": monkey,
        "heldout_config": str(args.heldout),
        "geometry_config": str(args.geometry_config),
        "standardize_features": standardize,
        "init_affine": {
            "origin_xy": [init_affine.origin_x, init_affine.origin_y],
            "pixels_per_unit": init_affine.pixels_per_unit,
            "rotation_deg": init_affine.rotation_deg,
            "flip_u": init_affine.flip_u,
            "flip_v": init_affine.flip_v,
        },
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
                "geom_r_before": row.get("geom_r_before"),
                "geom_r_after": row.get("geom_r_after"),
                "r_mean_test_masked": row.get("r_mean_test_masked"),
            }
            for row in summary_rows
        ],
    }
    (out_dir / "folds_index.yaml").write_text(yaml.safe_dump(folds_index, sort_keys=False))
    print(f"Wrote {out_dir / 'params.yaml'}")
    print(f"Wrote {out_dir / 'folds_index.yaml'}")


if __name__ == "__main__":
    main()
