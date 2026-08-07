#!/usr/bin/env python3
"""Replot LOO sanity / by-condition figures from saved fold models (no retrain).

Use after clim / plotting fixes so overview originals match across models for
the same window. Reads ``model.joblib`` + fold manifest ``loo_split==test``.
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
    _plot_per_condition_orig_recon,
    _plot_sanity_orig_recon,
)
from src.data.averaging import resolve_normalization
from src.encoding.ridge import build_xy, predict_maps
from src.evaluation.dual_metrics import dual_mask_metrics
from src.evaluation.mask import mask_from_eval_cfg
from src.evaluation.pixel_correlation import load_trial_mean_maps
from src.paths import project_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(config_path: Path, window_path: Path, ridge_path: Path) -> dict:
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(ridge_path)
    return cfg


def replot_protocol_dir(
    protocol_dir: Path,
    *,
    cfg: dict,
    repo: Path,
) -> list[Path]:
    import joblib

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
    disk_radius = int(eval_cfg.get("mask_radius", 50))

    written: list[Path] = []
    fold_dirs = sorted(
        p
        for p in protocol_dir.iterdir()
        if p.is_dir() and p.name.startswith(("A__", "B__"))
    )
    for fold_dir in fold_dirs:
        model_path = fold_dir / "model.joblib"
        manif_paths = list(fold_dir.glob("*__manifest.parquet"))
        if not model_path.is_file() or not manif_paths:
            print(f"SKIP {fold_dir.name}: missing model or manifest")
            continue
        manif = pd.read_parquet(manif_paths[0])
        test_df = manif[manif["loo_split"] == "test"].reset_index(drop=True)
        if test_df.empty:
            print(f"SKIP {fold_dir.name}: empty loo test")
            continue

        payload = joblib.load(model_path)
        result = payload["result"] if isinstance(payload, dict) else payload
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

        metrics_path = fold_dir / "metrics.json"
        disk_r = float("nan")
        roi_r = float("nan")
        train_label = "noise_ceiling_hull"
        if metrics_path.is_file():
            payload_m = json.loads(metrics_path.read_text())
            tm = payload_m.get("test_metrics") or {}
            disk_r = float(tm.get("mean_r_disk", float("nan")))
            roi_r = float(tm.get("mean_r_roi", float("nan")))
            train_label = (payload_m.get("train_target_mask") or {}).get(
                "train_targets", train_label
            )

        train_mask = None
        mask_path = fold_dir / "train_target_mask.npy"
        if mask_path.is_file():
            train_mask = np.load(mask_path).astype(bool)

        sanity = fold_dir / "sanity_orig_recon_residual.png"
        _plot_sanity_orig_recon(
            originals,
            recons,
            out_path=sanity,
            title=(
                f"{fold_dir.name} | test n={len(test_df)} | "
                f"disk r={disk_r:.3f} | ROI r={roi_r:.3f} | "
                f"train={train_label}"
            ),
            roi_mask=train_mask,
        )
        _plot_per_condition_orig_recon(
            test_df,
            originals,
            recons,
            fold_dir=fold_dir,
            fold_id=fold_dir.name,
            disk_mask=disk_mask,
        )
        written.append(sanity)
        print(
            f"Replotted {fold_dir.name} "
            f"(orig clim-anchored; n_test={len(test_df)})"
        )
        # Touch dual metrics only for logging consistency check.
        m = dual_mask_metrics(
            originals,
            recons,
            disk_mask=disk_mask,
            roi_mask=None,
            disk_radius=disk_radius,
        )
        print(
            f"  mean_r2_full={m.get('mean_r2_full')} "
            f"mean_r_disk={m.get('mean_r_disk')}"
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
        "--overview-alias",
        type=str,
        default=None,
        help="Optional alias filename for overview batch-01",
    )
    p.add_argument(
        "--suptitle-note",
        type=str,
        default="VSD_CMAP; hull outline omitted; clim=original",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    replot_protocol_dir(protocol_dir, cfg=cfg, repo=repo)

    from experiments.loo_encoding.make_loo_triplet_overview import (
        write_overview_batches,
    )

    written = write_overview_batches(
        protocol_dir,
        per_page=7,
        alias=args.overview_alias,
        suptitle_note=args.suptitle_note,
    )
    for path in written:
        print(path.relative_to(repo) if path.is_relative_to(repo) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
