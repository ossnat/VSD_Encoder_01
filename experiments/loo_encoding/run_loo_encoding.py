#!/usr/bin/env python3
"""Run leave-one-out ridge encoding folds (ResNet18 / layer3 baseline).

Protocols (same held-out stimulus list):
  A — leave one (date, condition) matching a held-out stimulus for test;
      other sessions of that stimulus may remain in train/val.
  B — leave entire stimulus_id out of train/val; all its trials are test.

Inner train/val is taken from the remainder (prefer existing split labels).

Outputs (default flat layout)::

  experiments/loo_encoding/runs/
    YYYY-MM-DD_{start}-{end}_{model}_{layer}/   # e.g. 2026-08-06_35-46_resnet18_l3
      protocol_{A|B}_{zscore|raw}_{NChull|disk|full}_{clean|all}/
        params.yaml  folds_index.yaml  loo_summary.csv  <fold_id>/...

Legacy deep layout (``--layout deep``)::

  experiments/loo_encoding/runs/<window_id>/<model>/<layer>/protocol_*/

Usage examples:
  # Build fold manifests only
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_43.yaml --protocol B --dry-run

  # Smoke one protocol-B fold
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_43.yaml --protocol B \\
    --fold-id B__white_point_0.1 --smoke

  # Subset of stimulus IDs (protocol B folds B__<id>)
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_43.yaml --protocol B \\
    --stimuli black_point_0.1 black_circle_contour_0.3

  # Full protocol B (all held-outs present in pairs)
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_43.yaml --protocol B

  # Protocol B: train Ridge only on held-out stimulus box ROI pixels
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_46_zscore.yaml --protocol B \\
    --target-mask roi --stimuli black_triangle_contour_0.4 \\
    --force --no-save-model

  # Named loss ROIs (aliases also accepted via --loss-roi)
  #   none | disk | box_union | noise_ceiling_hull | roi | /path/to/mask.npy
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_46_zscore.yaml --protocol B \\
    --loss-roi disk --stimuli black_triangle_contour_0.4 \\
    --force --no-save-model

  # Custom mask path (polygon/ellipse .npy or box YAML) for all folds
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_46_zscore.yaml --protocol B \\
    --target-mask path/to/mask.npy

  # Protocol A with one random (date, condition) per stimulus_id (seed=17)
  scripts/py experiments/loo_encoding/run_loo_encoding.py \\
    --window configs/windows/evoked_35_43.yaml --protocol A \\
    --one-fold-per-stimulus --seed 17
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.DL_features.schema import model_slug
from src.data.averaging import resolve_normalization
from src.encoding.ridge import (
    alpha_metrics,
    attach_feature_paths,
    build_xy,
    fit_ridge_encoder,
    predict_maps,
)
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.condition_report import plot_condition_orig_recon_corr
from src.evaluation.dual_metrics import (
    dual_mask_metrics,
    dual_metrics_by_stimulus,
    mean_trial_spatial_r,
)
from src.evaluation.loss_roi import (
    parse_loss_roi_arg,
    protocol_dir_suffix,
    resolve_loss_roi,
)
from src.evaluation.mask import mask_from_eval_cfg
from src.evaluation.pixel_correlation import (
    build_condition_entries,
    load_trial_mean_maps,
)
from src.evaluation.plotting import plot_pixel_mean_maps
from src.evaluation.roi_mask import DEFAULT_ROIS_DIR, load_roi_mask
from src.loo.folds import (
    build_protocol_a_folds,
    build_protocol_b_folds,
    load_heldout_list,
    select_one_fold_per_stimulus,
    write_fold_manifest,
)
from src.loo.paths import (
    OUT_ROOT,
    build_run_params,
    cleanliness_leaf_tag,
    resolve_flat_out_dir,
    safe_dir_token,
)
from src.paths import project_root, resolve_data_path
from src.qc.trial_cleanliness import (
    default_cleanliness_run_tag,
    filter_pairs_by_cleanliness,
)
from src.stimuli.identity import attach_stimulus_ids

HELDOUT_DEFAULT = Path("experiments/loo_encoding/heldout_list.yaml")

_safe_dir_token = safe_dir_token


def _protocol_dir_name(
    protocol: str,
    target_mask_mode: str,
    *,
    run_tag: str | None = None,
    mask_path: Path | None = None,
) -> str:
    """
    Output directory name under ``.../<layer>/``.

    Examples:
      none                 → protocol_B
      disk                 → protocol_B_disk
      box_union            → protocol_B_box_union
      noise_ceiling_hull   → protocol_B_noise_ceiling_hull
      roi                  → protocol_B_box_roi
      path + run-tag       → protocol_B_box_union
      path (no tag)        → protocol_B_union_of_boxes  (from mask stem)
    """
    base = f"protocol_{protocol}"
    # Custom path + explicit run-tag: tag is the primary suffix.
    if target_mask_mode == "path" and run_tag:
        return f"{base}_{_safe_dir_token(run_tag)}"
    suffix = protocol_dir_suffix(
        target_mask_mode, run_tag=None, mask_path=mask_path
    )
    name = base if suffix is None else f"{base}_{suffix}"
    if run_tag:
        name = f"{name}__{_safe_dir_token(run_tag)}"
    return name

def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _fold_out_dir(
    repo: Path,
    *,
    window_id: str,
    model_name: str,
    feature_layer: str,
    protocol: str,
    target_mask_mode: str = "none",
    target_mask_path: Path | None = None,
    run_tag: str | None = None,
) -> Path:
    protocol_dir = _protocol_dir_name(
        protocol,
        target_mask_mode,
        run_tag=run_tag,
        mask_path=target_mask_path,
    )
    return (
        repo
        / OUT_ROOT
        / window_id
        / model_name
        / feature_layer
        / protocol_dir
    )


def _resolve_train_target_mask(
    *,
    mode: str,
    mask_path: Path | None,
    heldout_stimulus_id: str,
    repo: Path,
    spatial_size: tuple[int, int],
    roi_dir: Path | None,
    disk_radius: int = 50,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (mask or None for full-frame, meta dict)."""
    return resolve_loss_roi(
        mode,
        mask_path=mask_path,
        spatial_size=spatial_size,
        disk_radius=disk_radius,
        heldout_stimulus_id=heldout_stimulus_id,
        repo=repo,
        roi_dir=roi_dir,
    )


def _fold_artifacts_complete(fold_dir: Path) -> bool:
    """True when metrics + sanity (+ ROI overlay if present path exists) are done."""
    required = [
        fold_dir / "metrics.json",
        fold_dir / "sanity_orig_recon_residual.png",
        fold_dir / "dual_metrics_by_stimulus.csv",
    ]
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        return False
    by_cond = fold_dir / "by_condition"
    if not by_cond.is_dir():
        return False
    if not any(by_cond.glob("*.png")):
        return False
    return True


def _summary_row_from_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open() as f:
        payload = json.load(f)
    fold = payload.get("fold") or {}
    test_metrics = payload.get("test_metrics") or {}
    train_mask_meta = payload.get("train_target_mask") or {}
    return {
        "fold_id": fold.get("fold_id"),
        "protocol": fold.get("protocol"),
        "heldout_stimulus_id": fold.get("heldout_stimulus_id"),
        "heldout_date": fold.get("heldout_date"),
        "heldout_condition": fold.get("heldout_condition"),
        "target_mask_mode": train_mask_meta.get("target_mask_mode"),
        "n_pixels_train": train_mask_meta.get("n_pixels_train"),
        "trained_on_masked_subset": train_mask_meta.get(
            "trained_on_masked_subset"
        ),
        "n_train": payload.get("n_train", fold.get("n_train")),
        "n_val": payload.get("n_val", fold.get("n_val")),
        "n_test": payload.get("n_test", fold.get("n_test")),
        "leakage_ok": bool(fold.get("leakage_ok", True)),
        "mean_r_disk": test_metrics.get("mean_r_disk"),
        "mean_r_roi": test_metrics.get("mean_r_roi"),
        "mean_trial_spatial_r_disk": test_metrics.get("mean_trial_spatial_r_disk"),
        "mean_trial_spatial_r_roi": test_metrics.get("mean_trial_spatial_r_roi"),
        "mean_trial_spatial_r_train_mask": test_metrics.get(
            "mean_trial_spatial_r_train_mask"
        ),
        "mean_r2_disk": test_metrics.get("mean_r2_disk"),
        "mean_r2_roi": test_metrics.get("mean_r2_roi"),
        "n_test_conditions": payload.get("n_test_conditions"),
        "n_by_condition_figures": len(payload.get("by_condition_figures") or []),
        "status": "skipped_existing",
        "error": None,
    }


def _plot_sanity_orig_recon(
    originals: np.ndarray,
    recons: np.ndarray,
    *,
    out_path: Path,
    title: str,
    roi_mask: np.ndarray | None = None,
) -> None:
    """Write the main orig | recon | residual sanity PNG (VSD colormap).

    ``roi_mask`` is accepted for call-site compatibility but is unused: the
    obsolete gray-colormap ROI-outline overlay is no longer written.
    """
    del roi_mask  # kept for API compatibility with replot_sanity_from_models
    mean_o = np.nanmean(originals, axis=0).astype(np.float32)
    mean_r = np.nanmean(recons, axis=0).astype(np.float32)
    mean_diff = (mean_r - mean_o).astype(np.float32)
    plot_pixel_mean_maps(
        mean_o,
        mean_r,
        mean_diff,
        out_path,
        title=title,
    )


def _plot_per_condition_orig_recon(
    test_df: pd.DataFrame,
    originals: np.ndarray,
    recons: np.ndarray,
    *,
    fold_dir: Path,
    fold_id: str,
    disk_mask: np.ndarray | None,
) -> list[str]:
    """Write orig|recon|residual for each (date, condition) in the test fold."""
    out_dir = fold_dir / "by_condition"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = build_condition_entries(
        test_df.reset_index(drop=True),
        originals,
        recons,
        eval_mask=disk_mask,
    )
    written: list[str] = []
    for entry in entries:
        date = str(entry["date"])
        condition = str(entry["condition"])
        mean_o = entry["original"]
        mean_r = entry["reconstruction"]
        residual = (mean_o - mean_r).astype(np.float32)
        stim_text = ""
        if "stimulus_text" in test_df.columns:
            sub = test_df[
                (test_df["date"].astype(str) == date)
                & (test_df["condition"].astype(str) == condition)
            ]
            if not sub.empty:
                stim_text = str(sub.iloc[0].get("stimulus_text", "") or "")
        payload = {
            "date": date,
            "condition": condition,
            "split": "loo_test",
            "stimulus_text": stim_text,
            "n_trials": int(entry["n_trials"]),
            "mean_original": mean_o,
            "mean_recon": mean_r,
            "residual": residual,
            "mean_trial_spatial_r": float(entry["trial_r_masked"]),
        }
        out_path = out_dir / f"{date}__{condition}.png"
        plot_condition_orig_recon_corr(
            payload,
            out_path,
            title_prefix=fold_id,
            eval_mask=disk_mask,
        )
        written.append(str(out_path.name))
    return written


def run_fold(
    *,
    cfg: dict,
    repo: Path,
    model_cfg: dict,
    feature_layer: str,
    model_name: str,
    spec,
    fold_df: pd.DataFrame,
    out_dir: Path,
    write_plots: bool = True,
    save_model: bool = True,
    target_mask_mode: str = "none",
    target_mask_path: Path | None = None,
    roi_dir: Path | None = None,
) -> dict[str, Any]:
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    avg_method = cfg.get("avg_method", "mean")
    normalization = resolve_normalization(cfg.get("normalization", "none"))
    baseline_start_frame = int(cfg.get("baseline_start_frame", 2))
    baseline_end_frame = int(cfg.get("baseline_end_frame", 26))
    baseline_std_eps = float(cfg.get("baseline_std_eps", 1e-8))
    ridge_cfg = cfg["ridge"]
    eval_cfg = ridge_cfg.get("evaluation", {})
    disk_mask = mask_from_eval_cfg(eval_cfg, spatial_size)
    disk_radius = int(eval_cfg["mask_radius"]) if disk_mask is not None else None

    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    fold_df = attach_feature_paths(
        fold_df,
        features_root=features_root,
        monkey=cfg["monkey"],
        model_slug=model_name,
        feature_layer=feature_layer,
        repo=repo,
    )
    fold_df = attach_stimulus_ids(fold_df)

    train_df = fold_df[fold_df["loo_split"] == "train"].copy()
    val_df = fold_df[fold_df["loo_split"] == "val"].copy()
    test_df = fold_df[fold_df["loo_split"] == "test"].copy()
    if train_df.empty:
        raise RuntimeError(f"{spec.fold_id}: empty train split")
    if test_df.empty:
        raise RuntimeError(f"{spec.fold_id}: empty test split")

    sid = spec.heldout_stimulus_id
    train_disk_radius = int(eval_cfg.get("mask_radius", 50))
    train_mask, train_mask_meta = _resolve_train_target_mask(
        mode=target_mask_mode,
        mask_path=target_mask_path,
        heldout_stimulus_id=sid,
        repo=repo,
        spatial_size=spatial_size,
        roi_dir=roi_dir,
        disk_radius=train_disk_radius,
    )

    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    alphas = np.asarray(ridge_cfg["alphas"], dtype=np.float64)
    alpha_per_target = bool(ridge_cfg.get("alpha_per_target", True))
    result = fit_ridge_encoder(
        x_train,
        y_train,
        alphas=alphas,
        cv_folds=int(ridge_cfg.get("cv_folds", 5)),
        standardize_features=bool(ridge_cfg.get("standardize_features", True)),
        alpha_per_target=alpha_per_target,
        target_mask=train_mask,
        spatial_size=spatial_size,
    )
    result.spatial_size = spatial_size
    result.feature_layer = feature_layer
    result.model_slug = model_name

    fold_dir = out_dir / spec.fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_fold_manifest(fold_dir, spec, fold_df)
    if save_model:
        joblib.dump(
            {"result": result, "fold_spec": spec.to_dict()},
            fold_dir / "model.joblib",
        )
    if alpha_per_target:
        np.save(
            fold_dir / "alphas_per_target.npy",
            np.asarray(result.alpha, dtype=np.float64),
        )
    if train_mask is not None:
        np.save(fold_dir / "train_target_mask.npy", train_mask.astype(bool))

    originals = load_trial_mean_maps(
        test_df,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_method=avg_method,
        normalization=normalization,
        baseline_start_frame=baseline_start_frame,
        baseline_end_frame=baseline_end_frame,
        baseline_std_eps=baseline_std_eps,
    )
    x_test, _ = build_xy(test_df, repo=repo, spatial_size=spatial_size)
    recons = predict_maps(result, x_test, spatial_size)

    try:
        roi = load_roi_mask(
            sid, repo=repo, spatial_size=spatial_size, roi_dir=roi_dir
        )
    except FileNotFoundError:
        roi = None
    # Prefer train mask for overlay when it differs from stimulus ROI.
    overlay_mask = train_mask if train_mask is not None else roi

    test_metrics = dual_mask_metrics(
        originals,
        recons,
        disk_mask=disk_mask,
        roi_mask=roi,
        disk_radius=disk_radius,
    )
    test_metrics["stimulus_id"] = sid
    if train_mask is not None:
        # Explicit train-mask spatial-r (same as ROI when mode=roi).
        test_metrics["mean_trial_spatial_r_train_mask"] = mean_trial_spatial_r(
            originals, recons, train_mask
        )
        test_metrics["n_pixels_train_mask"] = int(train_mask.sum())

    # Val dual metrics when available (disk only if mixed stimuli).
    val_metrics: dict[str, Any] | None = None
    if not val_df.empty:
        orig_v = load_trial_mean_maps(
            val_df,
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
            normalization=normalization,
            baseline_start_frame=baseline_start_frame,
            baseline_end_frame=baseline_end_frame,
            baseline_std_eps=baseline_std_eps,
        )
        x_val, _ = build_xy(val_df, repo=repo, spatial_size=spatial_size)
        recon_v = predict_maps(result, x_val, spatial_size)
        val_metrics = dual_mask_metrics(
            orig_v,
            recon_v,
            disk_mask=disk_mask,
            roi_mask=None,
            disk_radius=disk_radius,
        )

    per_stim = dual_metrics_by_stimulus(
        test_df,
        originals,
        recons,
        disk_mask=disk_mask,
        repo=repo,
        spatial_size=spatial_size,
        disk_radius=disk_radius,
        roi_dir=roi_dir,
    )
    per_stim_path = fold_dir / "dual_metrics_by_stimulus.csv"
    per_stim.to_csv(per_stim_path, index=False)

    if write_plots:
        _plot_sanity_orig_recon(
            originals,
            recons,
            out_path=fold_dir / "sanity_orig_recon_residual.png",
            title=(
                f"{spec.fold_id} | test n={len(test_df)} | "
                f"disk r={test_metrics.get('mean_r_disk', float('nan')):.3f} | "
                f"ROI r={test_metrics.get('mean_r_roi', float('nan')):.3f} | "
                f"train={train_mask_meta.get('train_targets', 'full_frame')}"
            ),
            roi_mask=overlay_mask,
        )
        cond_figs = _plot_per_condition_orig_recon(
            test_df,
            originals,
            recons,
            fold_dir=fold_dir,
            fold_id=spec.fold_id,
            disk_mask=disk_mask,
        )
    else:
        cond_figs = []

    # Confirm model output width matches train pixel count.
    n_model_targets = int(np.asarray(result.model.coef_).shape[0])
    train_mask_meta["n_model_targets"] = n_model_targets
    train_mask_meta["trained_on_masked_subset"] = bool(
        train_mask is not None
        and n_model_targets == int(train_mask.sum())
    )

    payload: dict[str, Any] = {
        "fold": spec.to_dict(),
        "model_slug": model_name,
        "feature_layer": feature_layer,
        "window_id": cfg.get("window_id"),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "normalization": normalization,
        "baseline_start_frame": baseline_start_frame,
        "baseline_end_frame": baseline_end_frame,
        "baseline_std_eps": baseline_std_eps,
        "train_target_mask": train_mask_meta,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_test_conditions": int(
            test_df.groupby(["date", "condition"]).ngroups
        ),
        "alpha": alpha_metrics(result.alpha, alpha_per_target=alpha_per_target),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "dual_metrics_csv": str(per_stim_path.relative_to(repo)),
        "by_condition_figures": cond_figs,
        "pixel_r_note": (
            "mean_r_disk/roi = mean of per-pixel Pearson r across ALL test "
            "trials in this fold, then averaged inside the mask. Undefined "
            "(NaN) when reconstructions are constant across trials "
            "(identical stimulus features)."
        ),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with (fold_dir / "metrics.json").open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    # Flat row for aggregate CSV
    row = {
        "fold_id": spec.fold_id,
        "protocol": spec.protocol,
        "heldout_stimulus_id": sid,
        "heldout_date": spec.heldout_date,
        "heldout_condition": spec.heldout_condition,
        "target_mask_mode": target_mask_mode,
        "n_pixels_train": train_mask_meta.get("n_pixels_train"),
        "trained_on_masked_subset": train_mask_meta.get(
            "trained_on_masked_subset"
        ),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "leakage_ok": bool(spec.leakage_ok),
        "mean_r_disk": test_metrics.get("mean_r_disk"),
        "mean_r_roi": test_metrics.get("mean_r_roi"),
        "mean_trial_spatial_r_disk": test_metrics.get("mean_trial_spatial_r_disk"),
        "mean_trial_spatial_r_roi": test_metrics.get("mean_trial_spatial_r_roi"),
        "mean_trial_spatial_r_train_mask": test_metrics.get(
            "mean_trial_spatial_r_train_mask"
        ),
        "mean_r2_disk": test_metrics.get("mean_r2_disk"),
        "mean_r2_roi": test_metrics.get("mean_r2_roi"),
        "n_test_conditions": int(
            test_df.groupby(["date", "condition"]).ngroups
        ),
        "n_by_condition_figures": len(cond_figs),
        "status": "ok",
        "error": None,
    }
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--window", type=Path, required=True)
    p.add_argument(
        "--ridge-config",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model YAML (default: configs/models/resnet18.yaml)",
    )
    p.add_argument("--feature-layer", type=str, default=None)
    p.add_argument("--protocol", choices=["A", "B", "both"], default="B")
    p.add_argument(
        "--heldout",
        type=Path,
        default=None,
        help="Held-out list YAML",
    )
    p.add_argument("--fold-id", type=str, default=None, help="Run only this fold_id")
    p.add_argument(
        "--stimuli",
        nargs="+",
        default=None,
        help=(
            "Run only these held-out stimulus_id values "
            "(overrides --heldout list for fold building)"
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first matching fold (after --fold-id filter)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and write fold manifests only (no training)",
    )
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument(
        "--one-fold-per-stimulus",
        action="store_true",
        help=(
            "For protocol A: randomly keep one (date, condition) fold per "
            "heldout_stimulus_id so fold count matches protocol B. "
            "Uses --seed (default 17). Writes "
            "one_fold_per_stimulus_selection.yaml under the protocol dir."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=17,
        help=(
            "RNG seed for inner train/val splits and for "
            "--one-fold-per-stimulus selection (default: 17)"
        ),
    )
    p.add_argument("--monkey", type=str, default=None)
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first fold failure (default: continue and report)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run folds even when metrics+figures already exist",
    )
    p.add_argument(
        "--no-save-model",
        action="store_true",
        help=(
            "Skip writing model.joblib (models are saved by default; "
            "useful for large backbones where weights are ~GB/fold). "
            "Metrics and sanity figures are still written."
        ),
    )
    p.add_argument(
        "--layout",
        choices=["flat", "deep"],
        default="flat",
        help=(
            "Output directory layout (default: flat). "
            "flat → runs/YYYY-MM-DD_35-46_resnet18_l3/protocol_A_zscore_NChull_clean/; "
            "deep → legacy runs/<window_id>/<model>/<layer>/protocol_*/"
        ),
    )
    p.add_argument(
        "--run-date",
        type=str,
        default=None,
        help=(
            "ISO date for flat run-root name (default: today). "
            "Use to continue yesterday's run root."
        ),
    )
    p.add_argument(
        "--run-root",
        type=str,
        default=None,
        help=(
            "Explicit flat run-root name or path under "
            "experiments/loo_encoding/runs/ (overrides --run-date naming)"
        ),
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Flat layout only: if the leaf directory already exists, create a "
            "new sibling with _HHMM (or _v2) instead of resuming into it"
        ),
    )
    p.add_argument(
        "--target-mask",
        "--loss-roi",
        dest="target_mask",
        type=str,
        default="none",
        help=(
            "Training Y / MSE ROI (default: none = full FOV). "
            "Named: none|disk|circular|box_union|noise_ceiling_hull|roi; "
            "or a path to .npy/.yaml. Alias: --loss-roi."
        ),
    )
    p.add_argument(
        "--roi-dir",
        type=Path,
        default=None,
        help=(
            "Directory of stimulus ROI YAML/masks "
            f"(default: {DEFAULT_ROIS_DIR})"
        ),
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional tag. Deep layout: appended to protocol dir "
            "(protocol_*__<tag>). Flat layout: appended as "
            "protocol_*__<tag> extra segment (cleanliness is already a "
            "leaf token). When --trial-cleanliness-csv is set and this is "
            "omitted, deep layout defaults to clean_good."
        ),
    )
    p.add_argument(
        "--trial-cleanliness-csv",
        type=Path,
        default=None,
        help=(
            "QC CSV keyed by trial_global_id with trial_cleanliness labels "
            "(good / pattern_outlier / amp_edge_outlier). Joined at load "
            "time; does not mutate FoundationData indexes."
        ),
    )
    p.add_argument(
        "--trial-cleanliness-keep",
        nargs="+",
        default=["good"],
        help=(
            "Cleanliness labels to keep when --trial-cleanliness-csv is set "
            "(default: good)."
        ),
    )
    p.add_argument(
        "--array-worker",
        action="store_true",
        help=(
            "SLURM array worker mode: run fold(s) only; do not rewrite "
            "leaf-level folds_index.yaml / loo_summary.csv / params.yaml "
            "(those are owned by prepare + finalize stages). "
            "Requires --fold-id (or --smoke / --max-folds)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(argv_list)
    repo = project_root()
    config_path = args.config or (repo / "configs/default.yaml")
    ridge_path = args.ridge_config or (repo / "configs/ridge/default.yaml")
    model_path = args.model or (repo / "configs/models/resnet18.yaml")
    window_path = args.window if args.window.is_absolute() else repo / args.window
    if not ridge_path.is_absolute():
        ridge_path = repo / ridge_path
    if not model_path.is_absolute():
        model_path = repo / model_path
    if not config_path.is_absolute():
        config_path = repo / config_path

    cfg = _merge_config(config_path, window_path, ridge_path)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    model_cfg = _load_yaml(model_path)
    feature_layer = args.feature_layer or model_cfg.get("feature_layer", "layer3")
    model_name = model_slug(model_cfg)
    normalization = resolve_normalization(cfg.get("normalization", "none"))
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])

    target_mask_mode, target_mask_path = parse_loss_roi_arg(args.target_mask)
    roi_dir: Path | None = None
    if args.roi_dir is not None:
        roi_dir = (
            args.roi_dir if args.roi_dir.is_absolute() else repo / args.roi_dir
        )

    window_id = cfg.get("window_id") or (
        f"win_{start_frame:04d}_{end_frame:04d}"
    )
    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        cfg["monkey"],
        window_id,
    )
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Encoding pairs not found: {pairs_path}. "
            "Run scripts/01_build_averaged_trials.py and "
            "scripts/01c_build_encoding_pairs.py for this window first."
        )
    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)

    user_run_tag = args.run_tag  # capture before cleanliness auto-tag
    cleanliness_stats: dict[str, Any] | None = None
    if args.trial_cleanliness_csv is not None:
        pairs, cleanliness_stats = filter_pairs_by_cleanliness(
            pairs,
            csv_path=args.trial_cleanliness_csv,
            keep=args.trial_cleanliness_keep,
            repo=repo,
        )
        print(
            "trial_cleanliness filter: "
            f"keep={cleanliness_stats['keep']} "
            f"n_before={cleanliness_stats['n_before']} "
            f"n_after={cleanliness_stats['n_after']} "
            f"n_dropped={cleanliness_stats['n_dropped']} "
            f"label_counts={cleanliness_stats['label_counts_before_keep']}"
        )
        if pairs.empty:
            raise RuntimeError(
                "No encoding pairs left after trial-cleanliness filter"
            )
        if args.run_tag is None and args.layout == "deep":
            args.run_tag = default_cleanliness_run_tag(args.trial_cleanliness_keep)

    cleanliness_tag = cleanliness_leaf_tag(
        trial_cleanliness_csv=args.trial_cleanliness_csv,
        keep=args.trial_cleanliness_keep,
    )
    # Flat leaf already encodes cleanliness; only append explicit user tags.
    flat_extra_tag = user_run_tag

    heldout_path = args.heldout or (repo / HELDOUT_DEFAULT)
    heldout_ids = load_heldout_list(
        heldout_path if heldout_path.is_absolute() else repo / heldout_path
    )
    if args.stimuli:
        heldout_ids = list(args.stimuli)

    protocols = ["A", "B"] if args.protocol == "both" else [args.protocol]
    summary_rows: list[dict[str, Any]] = []
    save_model = not args.no_save_model
    run_date = args.run_date or date.today().isoformat()
    flat_run_root_dir: Path | None = None
    array_worker = bool(args.array_worker)
    if array_worker:
        if args.protocol == "both":
            raise SystemExit(
                "--array-worker requires a single --protocol (A or B), not both"
            )
        if not (args.fold_id or args.smoke or args.max_folds is not None):
            raise SystemExit(
                "--array-worker requires --fold-id (recommended) or "
                "--smoke / --max-folds to avoid rewriting the full leaf"
            )

    print(
        f"layout={args.layout}  target_mask_mode={target_mask_mode}"
        + (f" path={target_mask_path}" if target_mask_path else "")
        + (f" roi_dir={roi_dir}" if roi_dir else "")
        + f"  save_model={save_model}"
    )

    for protocol in protocols:
        if protocol == "A":
            folds = build_protocol_a_folds(
                pairs, heldout_ids, seed=args.seed
            )
        else:
            folds = build_protocol_b_folds(
                pairs, heldout_ids, seed=args.seed
            )

        if args.one_fold_per_stimulus:
            if protocol != "A":
                print(
                    "WARNING: --one-fold-per-stimulus is intended for "
                    f"protocol A; ignoring for protocol {protocol}"
                )
            else:
                before = len(folds)
                folds = select_one_fold_per_stimulus(folds, seed=args.seed)
                print(
                    f"[A] --one-fold-per-stimulus seed={args.seed}: "
                    f"{before} folds -> {len(folds)} "
                    "(one random date/condition per stimulus_id)"
                )

        if args.fold_id:
            folds = [(s, d) for s, d in folds if s.fold_id == args.fold_id]
            if not folds:
                raise SystemExit(
                    f"No fold matched --fold-id {args.fold_id!r} "
                    f"(protocol={protocol})"
                )
        if args.smoke:
            folds = folds[:1]
        if args.max_folds is not None:
            folds = folds[: args.max_folds]

        if args.layout == "flat":
            root_dir, out_dir = resolve_flat_out_dir(
                repo,
                run_date=run_date,
                start_frame=start_frame,
                end_frame=end_frame,
                model_slug=model_name,
                feature_layer=feature_layer,
                protocol=protocol,
                normalization=normalization,
                target_mask_mode=target_mask_mode,
                cleanliness=cleanliness_tag,
                run_root=args.run_root or (
                    flat_run_root_dir.name if flat_run_root_dir else None
                ),
                mask_path=target_mask_path,
                extra_tag=flat_extra_tag,
                fresh=bool(args.fresh),
            )
            flat_run_root_dir = root_dir
            root_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = _fold_out_dir(
                repo,
                window_id=window_id,
                model_name=model_name,
                feature_layer=feature_layer,
                protocol=protocol,
                target_mask_mode=target_mask_mode,
                target_mask_path=target_mask_path,
                run_tag=args.run_tag,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir.relative_to(repo)}")
        if array_worker:
            print("  mode=array-worker (skip leaf index/summary/params rewrite)")

        if not array_worker:
            params_payload = build_run_params(
                layout=args.layout,
                run_root=(
                    flat_run_root_dir
                    if flat_run_root_dir is not None
                    else out_dir.parent
                ),
                leaf_dir=out_dir,
                repo=repo,
                window_path=window_path,
                window_id=window_id,
                start_frame=start_frame,
                end_frame=end_frame,
                normalization=normalization,
                model_path=model_path,
                model_slug=model_name,
                feature_layer=feature_layer,
                protocol=protocol,
                heldout_stimuli=heldout_ids,
                target_mask_mode=target_mask_mode,
                target_mask_path=target_mask_path,
                roi_dir=roi_dir if roi_dir is not None else DEFAULT_ROIS_DIR,
                trial_cleanliness_csv=args.trial_cleanliness_csv,
                trial_cleanliness_keep=(
                    args.trial_cleanliness_keep
                    if args.trial_cleanliness_csv is not None
                    else None
                ),
                ridge_path=ridge_path,
                ridge_cfg=cfg.get("ridge"),
                cli_argv=[
                    "experiments/loo_encoding/run_loo_encoding.py",
                    *argv_list,
                ],
                save_model=save_model,
                run_tag=args.run_tag,
                seed=args.seed,
                extra={
                    "one_fold_per_stimulus": bool(args.one_fold_per_stimulus),
                    "dry_run": bool(args.dry_run),
                    "monkey": cfg.get("monkey"),
                },
            )
            if cleanliness_stats is not None:
                params_payload["trial_cleanliness_stats"] = cleanliness_stats
            params_path = out_dir / "params.yaml"
            with params_path.open("w") as f:
                yaml.safe_dump(params_payload, f, sort_keys=False)
            print(f"Wrote params: {params_path.relative_to(repo)}")

            if args.one_fold_per_stimulus and protocol == "A":
                selection = {
                    "seed": args.seed,
                    "n_folds": len(folds),
                    "note": (
                        "One randomly chosen (date, condition) fold per "
                        "heldout_stimulus_id; candidates sorted by fold_id "
                        "before sampling."
                    ),
                    "folds": [
                        {
                            "fold_id": spec.fold_id,
                            "heldout_stimulus_id": spec.heldout_stimulus_id,
                            "heldout_date": spec.heldout_date,
                            "heldout_condition": spec.heldout_condition,
                            "n_train": spec.n_train,
                            "n_val": spec.n_val,
                            "n_test": spec.n_test,
                        }
                        for spec, _ in folds
                    ],
                }
                sel_path = out_dir / "one_fold_per_stimulus_selection.yaml"
                with sel_path.open("w") as f:
                    yaml.safe_dump(selection, f, sort_keys=False)
                print(f"Wrote selection: {sel_path.relative_to(repo)}")

        index_rows: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        protocol_summary_rows: list[dict[str, Any]] = []
        for spec, fold_df in folds:
            index_rows.append(spec.to_dict())
            print(
                f"[{protocol}] {spec.fold_id}  "
                f"n_train={spec.n_train} n_val={spec.n_val} n_test={spec.n_test} "
                f"leakage_ok={spec.leakage_ok}"
            )
            if args.dry_run:
                write_fold_manifest(out_dir / spec.fold_id, spec, fold_df)
                continue
            fold_dir = out_dir / spec.fold_id
            if not args.force and _fold_artifacts_complete(fold_dir):
                row = _summary_row_from_metrics(fold_dir / "metrics.json")
                summary_rows.append(row)
                protocol_summary_rows.append(row)
                print(
                    f"  SKIP existing  "
                    f"disk mean_r={row.get('mean_r_disk')}  "
                    f"ROI mean_r={row.get('mean_r_roi')}  "
                    f"spatial disk={row.get('mean_trial_spatial_r_disk')}  "
                    f"spatial ROI={row.get('mean_trial_spatial_r_roi')}  "
                    f"cond_figs={row.get('n_by_condition_figures')}"
                )
                continue
            try:
                row = run_fold(
                    cfg=cfg,
                    repo=repo,
                    model_cfg=model_cfg,
                    feature_layer=feature_layer,
                    model_name=model_name,
                    spec=spec,
                    fold_df=fold_df,
                    out_dir=out_dir,
                    save_model=save_model,
                    target_mask_mode=target_mask_mode,
                    target_mask_path=target_mask_path,
                    roi_dir=roi_dir,
                )
            except Exception as exc:  # noqa: BLE001 — continue other folds
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {msg}")
                failed.append({"fold_id": spec.fold_id, "error": msg})
                fail_row = {
                    "fold_id": spec.fold_id,
                    "protocol": spec.protocol,
                    "heldout_stimulus_id": spec.heldout_stimulus_id,
                    "heldout_date": spec.heldout_date,
                    "heldout_condition": spec.heldout_condition,
                    "n_train": spec.n_train,
                    "n_val": spec.n_val,
                    "n_test": spec.n_test,
                    "leakage_ok": bool(spec.leakage_ok),
                    "status": "failed",
                    "error": msg,
                }
                summary_rows.append(fail_row)
                protocol_summary_rows.append(fail_row)
                if args.fail_fast:
                    raise
                continue
            summary_rows.append(row)
            protocol_summary_rows.append(row)
            print(
                f"  disk mean_r={row.get('mean_r_disk')}  "
                f"ROI mean_r={row.get('mean_r_roi')}  "
                f"spatial disk={row.get('mean_trial_spatial_r_disk')}  "
                f"spatial ROI={row.get('mean_trial_spatial_r_roi')}  "
                f"spatial train_mask={row.get('mean_trial_spatial_r_train_mask')}  "
                f"n_pix_train={row.get('n_pixels_train')}  "
                f"masked={row.get('trained_on_masked_subset')}  "
                f"cond_figs={row.get('n_by_condition_figures')}"
            )

        if not array_worker:
            idx_path = out_dir / "folds_index.yaml"
            # Subset / --stimuli runs merge into any existing index so other folds
            # are not wiped.
            prev_index: dict[str, Any] = {}
            if idx_path.exists():
                with idx_path.open() as f:
                    prev_index = yaml.safe_load(f) or {}
            prev_folds = {
                str(r.get("fold_id")): r
                for r in (prev_index.get("folds") or [])
                if isinstance(r, dict) and r.get("fold_id")
            }
            for row in index_rows:
                prev_folds[str(row["fold_id"])] = row
            merged_folds = list(prev_folds.values())
            prev_failed = {
                str(r.get("fold_id")): r
                for r in (prev_index.get("failed_folds") or [])
                if isinstance(r, dict) and r.get("fold_id")
            }
            for row in failed:
                prev_failed[str(row["fold_id"])] = row
            for row in index_rows:
                # Successful (or dry-run) folds clear prior failure entries.
                if row["fold_id"] not in {f["fold_id"] for f in failed}:
                    prev_failed.pop(str(row["fold_id"]), None)
            merged_failed = list(prev_failed.values())
            merged_heldout = list(
                dict.fromkeys(
                    list(prev_index.get("heldout_list") or [])
                    + list(heldout_ids)
                )
            )
            index_payload: dict[str, Any] = {
                "protocol": protocol,
                "window_id": window_id,
                "model_slug": model_name,
                "feature_layer": feature_layer,
                "layout": args.layout,
                "normalization": normalization,
                "target_mask_mode": target_mask_mode,
                "target_mask_path": (
                    str(target_mask_path) if target_mask_path else None
                ),
                "roi_dir": str(roi_dir) if roi_dir else str(DEFAULT_ROIS_DIR),
                "run_tag": args.run_tag,
                "heldout_list": merged_heldout,
                "n_folds": len(merged_folds),
                "n_failed": len(merged_failed),
                "failed_folds": merged_failed,
                "folds": merged_folds,
            }
            if cleanliness_stats is not None:
                index_payload["trial_cleanliness"] = cleanliness_stats
            with idx_path.open("w") as f:
                yaml.safe_dump(
                    index_payload,
                    f,
                    sort_keys=False,
                )
            print(f"Wrote fold index: {idx_path.relative_to(repo)}")
            if failed:
                print(
                    f"WARNING: {len(failed)} fold(s) failed: "
                    f"{[f['fold_id'] for f in failed]}"
                )

            # Flat layout: summary lives inside the leaf next to params.yaml.
            if args.layout == "flat" and protocol_summary_rows:
                summary = pd.DataFrame(protocol_summary_rows)
                summary_path = out_dir / "loo_summary.csv"
                if summary_path.exists():
                    prev = pd.read_csv(summary_path)
                    summary = (
                        pd.concat([prev, summary], ignore_index=True)
                        .drop_duplicates(subset=["fold_id"], keep="last")
                    )
                summary.to_csv(summary_path, index=False)
                print(f"Wrote {summary_path.relative_to(repo)}")
        elif failed:
            print(
                f"WARNING: {len(failed)} fold(s) failed: "
                f"{[f['fold_id'] for f in failed]}"
            )

    if args.layout == "deep" and summary_rows and not array_worker:
        summary = pd.DataFrame(summary_rows)
        # Prefer window-level summary next to protocol dirs.
        # Include run_tag / target mask so filtered runs do not clobber
        # the all-trials baseline summary.
        summary_name = "loo_summary.csv"
        if target_mask_mode != "none":
            summary_name = f"loo_summary__target_{target_mask_mode}.csv"
        if args.run_tag:
            stem = summary_name[:-4] if summary_name.endswith(".csv") else summary_name
            summary_name = f"{stem}__{_safe_dir_token(args.run_tag)}.csv"
        summary_path = (
            repo
            / OUT_ROOT
            / window_id
            / model_name
            / feature_layer
            / summary_name
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists():
            prev = pd.read_csv(summary_path)
            summary = (
                pd.concat([prev, summary], ignore_index=True)
                .drop_duplicates(subset=["fold_id"], keep="last")
            )
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path.relative_to(repo)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
