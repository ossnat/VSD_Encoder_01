#!/usr/bin/env python3
"""Rebuild leaf-level ``loo_summary.csv`` after SLURM array workers finish.

Array workers (``run_loo_encoding.py --array-worker``) intentionally skip
rewriting shared leaf files. Call this once per leaf after the array job
completes (``afterok`` dependency).

Also refreshes ``folds_index.yaml`` failure bookkeeping from on-disk metrics,
and optionally writes triplet overview batches.

Usage::

  scripts/py experiments/loo_encoding/finalize_loo_leaf.py \\
    --protocol-dir experiments/loo_encoding/runs/.../protocol_A_zscore_NChull_clean \\
    --make-overview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiments.loo_encoding.make_loo_triplet_overview import write_overview_batches
from src.paths import project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _fold_dirs(protocol_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in protocol_dir.iterdir()
        if p.is_dir() and p.name.startswith(("A__", "B__"))
    )


def _fold_artifacts_complete(fold_dir: Path) -> bool:
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
    return any(by_cond.glob("*.png"))


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
        "mean_trial_spatial_r_disk": test_metrics.get(
            "mean_trial_spatial_r_disk"
        ),
        "mean_trial_spatial_r_roi": test_metrics.get(
            "mean_trial_spatial_r_roi"
        ),
        "mean_trial_spatial_r_train_mask": test_metrics.get(
            "mean_trial_spatial_r_train_mask"
        ),
        "mean_r2_disk": test_metrics.get("mean_r2_disk"),
        "mean_r2_roi": test_metrics.get("mean_r2_roi"),
        "n_test_conditions": payload.get("n_test_conditions"),
        "n_by_condition_figures": len(payload.get("by_condition_figures") or []),
        "status": "ok",
        "error": None,
    }


def finalize_leaf(
    protocol_dir: Path,
    *,
    make_overview: bool = False,
    overview_per_page: int = 8,
) -> dict[str, Any]:
    protocol_dir = protocol_dir.resolve()
    index_path = protocol_dir / "folds_index.yaml"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing folds_index.yaml under {protocol_dir} "
            "(run prepare / dry-run first)"
        )
    index = _load_yaml(index_path)
    expected = [str(f["fold_id"]) for f in (index.get("folds") or [])]

    summary_rows: list[dict[str, Any]] = []
    complete: list[str] = []
    incomplete: list[str] = []
    failed_rows: list[dict[str, str]] = []

    fold_id_set = set(expected)
    for fold_dir in _fold_dirs(protocol_dir):
        fold_id_set.add(fold_dir.name)

    for fold_id in sorted(fold_id_set):
        fold_dir = protocol_dir / fold_id
        metrics_path = fold_dir / "metrics.json"
        if _fold_artifacts_complete(fold_dir) and metrics_path.is_file():
            row = _summary_row_from_metrics(metrics_path)
            row["status"] = row.get("status") or "ok"
            summary_rows.append(row)
            complete.append(fold_id)
            continue
        incomplete.append(fold_id)
        err = "incomplete artifacts"
        if metrics_path.is_file():
            try:
                payload = json.loads(metrics_path.read_text())
                err = str(payload.get("error") or err)
            except (OSError, json.JSONDecodeError):
                pass
        failed_rows.append({"fold_id": fold_id, "error": err})

    summary = pd.DataFrame(summary_rows)
    summary_path = protocol_dir / "loo_summary.csv"
    if not summary.empty:
        summary = summary.drop_duplicates(subset=["fold_id"], keep="last")
        summary = summary.sort_values("fold_id").reset_index(drop=True)
        summary.to_csv(summary_path, index=False)
    elif summary_path.exists():
        summary_path.unlink()

    index["n_complete"] = len(complete)
    index["n_incomplete"] = len(incomplete)
    index["n_failed"] = len(failed_rows)
    index["failed_folds"] = failed_rows
    index["complete_folds"] = complete
    with index_path.open("w") as f:
        yaml.safe_dump(index, f, sort_keys=False)

    overview_paths: list[str] = []
    if make_overview and complete:
        written = write_overview_batches(
            protocol_dir,
            per_page=overview_per_page,
            alias="all_folds_triplets.png",
        )
        overview_paths = [str(p) for p in written]

    return {
        "protocol_dir": str(protocol_dir),
        "n_expected": len(expected),
        "n_complete": len(complete),
        "n_incomplete": len(incomplete),
        "incomplete_folds": incomplete,
        "summary_csv": str(summary_path) if summary_path.exists() else None,
        "overview_paths": overview_paths,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol-dir",
        type=Path,
        required=True,
        help="Flat leaf dir (e.g. protocol_A_zscore_NChull_clean)",
    )
    p.add_argument(
        "--make-overview",
        action="store_true",
        help="Also write overview/ triplet batch PNGs",
    )
    p.add_argument("--overview-per-page", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    protocol_dir = (
        args.protocol_dir
        if args.protocol_dir.is_absolute()
        else repo / args.protocol_dir
    )
    result = finalize_leaf(
        protocol_dir,
        make_overview=bool(args.make_overview),
        overview_per_page=args.overview_per_page,
    )
    print(yaml.safe_dump(result, sort_keys=False))
    if result["n_incomplete"]:
        print(
            f"WARNING: {result['n_incomplete']} incomplete fold(s)",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
