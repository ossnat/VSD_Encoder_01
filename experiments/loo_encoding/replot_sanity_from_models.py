#!/usr/bin/env python3
"""Replot LOO sanity / by-condition figures without RidgeCV.

Default: rewrite ``sanity_orig_recon_residual.png`` from saved
``fold_mean_orig.npy`` / ``fold_mean_recon.npy`` (no retrain, no r/r²).
If those arrays are missing, uses ``model.joblib`` when present; otherwise
refits Ridge with saved ``alphas_per_target.npy`` + fold manifest (no alpha
search). Recon sanity panels use recon-only color limits so identical
``y_hat`` matches across sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from experiments.loo_encoding.run_loo_encoding import (
    SANITY_LAYOUT_MEAN,
    SANITY_LAYOUTS,
    _plot_per_condition_orig_recon,
    _plot_sanity_orig_recon,
    fold_plot_seed,
)
from src.data.averaging import resolve_normalization
from src.encoding.ridge import build_xy, fit_ridge_fixed_alphas, predict_maps
from src.evaluation.mask import mask_from_eval_cfg
from src.evaluation.pixel_correlation import load_trial_mean_maps
from src.loo.paths import list_loo_fold_dirs
from src.paths import project_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(config_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def _filter_fold_dirs(fold_dirs: list[Path], fold_ids: list[str] | None) -> list[Path]:
    if not fold_ids:
        return fold_dirs
    selected: list[Path] = []
    for fold_dir in fold_dirs:
        if fold_dir.name in fold_ids or any(tok in fold_dir.name for tok in fold_ids):
            selected.append(fold_dir)
    return selected


def _load_or_refit_result(
    fold_dir: Path,
    fold_df: pd.DataFrame,
    *,
    cfg: dict,
    repo: Path,
    spatial_size: tuple[int, int],
):
    model_path = fold_dir / "model.joblib"
    if model_path.is_file():
        import joblib

        payload = joblib.load(model_path)
        return payload["result"] if isinstance(payload, dict) else payload

    alphas_path = fold_dir / "alphas_per_target.npy"
    if not alphas_path.is_file():
        return None
    train_df = fold_df[fold_df["loo_split"] == "train"].reset_index(drop=True)
    if train_df.empty:
        return None
    train_mask = None
    mask_path = fold_dir / "train_target_mask.npy"
    if mask_path.is_file():
        train_mask = np.load(mask_path).astype(bool)
    alphas = np.load(alphas_path).astype(np.float64)
    ridge_cfg = cfg.get("ridge") or {}
    x_train, y_train = build_xy(train_df, repo=repo, spatial_size=spatial_size)
    return fit_ridge_fixed_alphas(
        x_train,
        y_train,
        alphas,
        standardize_features=bool(ridge_cfg.get("standardize_features", True)),
        target_mask=train_mask,
        spatial_size=spatial_size,
    )


def _sanity_title(
    fold_dir: Path,
    *,
    n_label: str,
    train_label: str = "noise_ceiling_hull",
) -> str:
    disk_r = float("nan")
    roi_r = float("nan")
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.is_file():
        payload_m = json.loads(metrics_path.read_text())
        tm = payload_m.get("test_metrics") or {}
        disk_r = float(tm.get("mean_r_disk", float("nan")))
        roi_r = float(tm.get("mean_r_roi", float("nan")))
        train_label = (payload_m.get("train_target_mask") or {}).get(
            "train_targets", train_label
        )
    return (
        f"{fold_dir.name} | {n_label} | "
        f"disk r={disk_r:.3f} | ROI r={roi_r:.3f} | "
        f"train={train_label}"
    )


def _replot_from_fold_means(
    fold_dir: Path,
    *,
    orig_recon_vmin: float | None,
    orig_recon_vmax: float | None,
) -> Path | None:
    orig_p = fold_dir / "fold_mean_orig.npy"
    recon_p = fold_dir / "fold_mean_recon.npy"
    if not (orig_p.is_file() and recon_p.is_file()):
        return None
    orig = np.load(orig_p).astype(np.float32)
    recon = np.load(recon_p).astype(np.float32)
    n_orig = int(orig.shape[0]) if orig.ndim == 3 else 1
    eval_mode = "fold_mean"
    meta_path = fold_dir / "fold_mean_meta.yaml"
    if meta_path.is_file():
        meta = _load_yaml(meta_path) or {}
        eval_mode = str(meta.get("evaluation", eval_mode))
        n_orig = int(meta.get("n_trials", n_orig))
    sanity = fold_dir / "sanity_orig_recon_residual.png"
    _plot_sanity_orig_recon(
        orig,
        recon,
        out_path=sanity,
        title=_sanity_title(fold_dir, n_label=f"{eval_mode} n={n_orig}"),
        vmin=orig_recon_vmin,
        vmax=orig_recon_vmax,
        sanity_layout=SANITY_LAYOUT_MEAN,
    )
    return sanity


def replot_protocol_dir(
    protocol_dir: Path,
    *,
    cfg: dict,
    repo: Path,
    fold_ids: list[str] | None = None,
    orig_recon_vmin: float | None = None,
    orig_recon_vmax: float | None = None,
    sanity_layout: str = SANITY_LAYOUT_MEAN,
    rewrite_by_condition: bool = False,
) -> list[Path]:
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    avg_method = cfg.get("avg_method", "mean")
    normalization = resolve_normalization(cfg.get("normalization", "none"))
    baseline_start_frame = int(cfg.get("baseline_start_frame", 2))
    baseline_end_frame = int(cfg.get("baseline_end_frame", 26))
    baseline_std_eps = float(cfg.get("baseline_std_eps", 1e-8))
    ridge_cfg = cfg.get("ridge") or {}
    eval_cfg = ridge_cfg.get("evaluation") or {}
    disk_mask = mask_from_eval_cfg(eval_cfg, spatial_size)

    written: list[Path] = []
    fold_dirs = _filter_fold_dirs(list_loo_fold_dirs(protocol_dir), fold_ids)
    if fold_ids and not fold_dirs:
        print(f"No matching folds for {fold_ids!r} under {protocol_dir}")
        return written

    for fold_dir in fold_dirs:
        if sanity_layout == SANITY_LAYOUT_MEAN:
            sanity = _replot_from_fold_means(
                fold_dir,
                orig_recon_vmin=orig_recon_vmin,
                orig_recon_vmax=orig_recon_vmax,
            )
            if sanity is not None:
                written.append(sanity)
                print(
                    f"Replotted {fold_dir.name} "
                    "(fold_mean arrays; recon clim independent; no retrain)"
                )
                continue

        manif_paths = list(fold_dir.glob("*__manifest.parquet"))
        if not manif_paths:
            print(f"SKIP {fold_dir.name}: missing manifest")
            continue
        manif = pd.read_parquet(manif_paths[0])
        test_df = manif[manif["loo_split"] == "test"].reset_index(drop=True)
        if test_df.empty:
            print(f"SKIP {fold_dir.name}: empty loo test")
            continue

        result = _load_or_refit_result(
            fold_dir,
            manif,
            cfg=cfg,
            repo=repo,
            spatial_size=spatial_size,
        )
        if result is None:
            print(f"SKIP {fold_dir.name}: missing model.joblib and alphas")
            continue

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

        train_mask = None
        mask_path = fold_dir / "train_target_mask.npy"
        if mask_path.is_file():
            train_mask = np.load(mask_path).astype(bool)

        sanity = fold_dir / "sanity_orig_recon_residual.png"
        plot_seed = fold_plot_seed(17, fold_dir.name)
        _plot_sanity_orig_recon(
            originals,
            recons,
            out_path=sanity,
            title=_sanity_title(fold_dir, n_label=f"test n={len(test_df)}"),
            roi_mask=train_mask,
            vmin=orig_recon_vmin,
            vmax=orig_recon_vmax,
            sanity_layout=sanity_layout,
            plot_seed=plot_seed,
        )
        if rewrite_by_condition:
            _plot_per_condition_orig_recon(
                test_df,
                originals,
                recons,
                fold_dir=fold_dir,
                fold_id=fold_dir.name,
                disk_mask=disk_mask,
            )
        written.append(sanity)
        source = (
            "model.joblib"
            if (fold_dir / "model.joblib").is_file()
            else "fixed-alpha refit"
        )
        print(
            f"Replotted {fold_dir.name} "
            f"({source}; recon clim independent; n_test={len(test_df)})"
        )
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol-dir",
        type=Path,
        required=True,
        help="Path to protocol_* run directory with fold subdirs",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--ridge-config", type=Path, default=None)
    p.add_argument(
        "--fold-id",
        action="append",
        default=None,
        help=(
            "Only these fold ids (repeatable). Exact name or substring, e.g. "
            "100718a or C__black_bar_vertical_1__100718a_condAN5."
        ),
    )
    p.add_argument(
        "--orig-recon-vmin",
        type=float,
        default=None,
        help=(
            "Optional fixed vmin for orig+recon sanity panels (must pair with "
            "--orig-recon-vmax). Default: independent 1-99 percentiles."
        ),
    )
    p.add_argument(
        "--orig-recon-vmax",
        type=float,
        default=None,
        help="Optional fixed vmax for orig+recon sanity panels.",
    )
    p.add_argument(
        "--sanity-layout",
        choices=list(SANITY_LAYOUTS),
        default=SANITY_LAYOUT_MEAN,
        help=(
            "Sanity PNG layout. mean_triplet (default): fold-mean orig | recon "
            "| residual. sample_trials: recon plus random trial originals."
        ),
    )
    p.add_argument(
        "--rewrite-by-condition",
        action="store_true",
        help="Also rewrite by_condition/ figures (needs model or alphas).",
    )
    p.add_argument(
        "--overview-alias",
        type=str,
        default="all_folds_triplets.png",
        help="Alias filename for overview batch-01 (default: all_folds_triplets.png).",
    )
    p.add_argument(
        "--per-page",
        type=int,
        default=12,
        help="Fold rows per overview collage page (default: 12).",
    )
    p.add_argument(
        "--suptitle-note",
        type=str,
        default=(
            "VSD_CMAP (mapgeog); mean orig | recon | residual; "
            "recon clim independent"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.orig_recon_vmin is None) != (args.orig_recon_vmax is None):
        raise SystemExit(
            "--orig-recon-vmin and --orig-recon-vmax must be set together"
        )
    repo = project_root()
    config_path = args.config or (repo / "configs/default.yaml")
    ridge_path = args.ridge_config or (repo / "configs/ridge/default.yaml")
    window_path = args.window if args.window.is_absolute() else repo / args.window
    protocol_dir = (
        args.protocol_dir
        if args.protocol_dir.is_absolute()
        else repo / args.protocol_dir
    )
    cfg = _merge_config(config_path, window_path, ridge_path)
    replot_protocol_dir(
        protocol_dir,
        cfg=cfg,
        repo=repo,
        fold_ids=args.fold_id,
        orig_recon_vmin=args.orig_recon_vmin,
        orig_recon_vmax=args.orig_recon_vmax,
        sanity_layout=args.sanity_layout,
        rewrite_by_condition=bool(args.rewrite_by_condition),
    )

    from experiments.loo_encoding.make_loo_triplet_overview import (
        write_overview_batches,
    )

    written = write_overview_batches(
        protocol_dir,
        per_page=args.per_page,
        alias=args.overview_alias,
        suptitle_note=args.suptitle_note,
    )
    for path in written:
        print(path.relative_to(repo) if path.is_relative_to(repo) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
