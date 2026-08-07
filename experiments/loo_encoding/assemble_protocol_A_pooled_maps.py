#!/usr/bin/env python3
"""Pooled fold-level encoding pixel-r maps for flat Protocol A run roots.

Computes per-leaf maps for the four encoding leaves
(``{zscore,raw} × {clean,all}``), writes them under each leaf ``overview/``
and a comparison grid + CSV/JSON summary at the run root.

Usage::

  scripts/py experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \\
    --run-root experiments/loo_encoding/runs/2026-08-07_35-46_resnet18_l3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.loo_encoding.plot_pooled_fold_pixel_r_maps import (
    _load_hull_mask,
    _mean_underlay,
    pooled_fold_pixel_r_map,
    save_single_map,
)
from src.paths import project_root
from src.plotting_colormaps import VSD_CMAP

LEAF_ORDER = (
    ("zscore", "clean", "protocol_A_zscore_NChull_clean"),
    ("zscore", "all", "protocol_A_zscore_NChull_all"),
    ("raw", "clean", "protocol_A_raw_NChull_clean"),
    ("raw", "all", "protocol_A_raw_NChull_all"),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_leaves(run_root: Path) -> list[tuple[str, str, Path]]:
    repo = project_root()
    manifest = run_root / "pipeline_manifest.yaml"
    if manifest.is_file():
        plan = _load_yaml(manifest)
        return [
            (
                str(leaf["window_kind"]),
                str(leaf["cleanliness"]),
                repo / leaf["leaf_dir"],
            )
            for leaf in plan.get("leaves") or []
        ]
    found: list[tuple[str, str, Path]] = []
    for window_kind, cleanliness, name in LEAF_ORDER:
        path = run_root / name
        if path.is_dir():
            found.append((window_kind, cleanliness, path))
    if not found:
        raise FileNotFoundError(
            f"No protocol_A_*_NChull_* leaves under {run_root}"
        )
    return found


def _save_zscore_raw_pair(
    *,
    zscore_map: np.ndarray,
    raw_map: np.ndarray,
    zscore_mean: float,
    raw_mean: float,
    zscore_n: int,
    raw_n: int,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
    panels = [
        (f"zscore · n={zscore_n}\nmean r (NC hull) = {zscore_mean:.3f}", zscore_map),
        (f"raw · n={raw_n}\nmean r (NC hull) = {raw_mean:.3f}", raw_map),
    ]
    im = None
    for ax, (panel_title, arr) in zip(axes, panels):
        im = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def assemble(
    run_root: Path,
    *,
    repo: Path,
    ridge_config: Path,
    spatial_size: tuple[int, int],
    avg_method: str = "mean",
) -> dict[str, Any]:
    ridge_cfg = _load_yaml(
        ridge_config if ridge_config.is_absolute() else repo / ridge_config
    )
    standardize_features = bool(ridge_cfg.get("standardize_features", True))
    hull_mask = _load_hull_mask(repo, spatial_size)

    leaves = _resolve_leaves(run_root)
    rows: list[dict[str, Any]] = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    means: dict[tuple[str, str], float] = {}
    ns: dict[tuple[str, str], int] = {}

    for window_kind, cleanliness, protocol_dir in leaves:
        if not (protocol_dir / "folds_index.yaml").is_file():
            print(f"SKIP missing folds_index: {protocol_dir}", flush=True)
            continue
        print(
            f"Computing pooled encoding r · {window_kind}/{cleanliness} …",
            flush=True,
        )
        corr_map, mean_r, n_folds, fold_ids = pooled_fold_pixel_r_map(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            hull_mask=hull_mask,
            avg_method=avg_method,
            standardize_features=standardize_features,
        )
        underlay = _mean_underlay(
            protocol_dir,
            repo=repo,
            spatial_size=spatial_size,
            avg_method=avg_method,
            standardize_features=standardize_features,
        )
        overview = protocol_dir / "overview"
        overview.mkdir(parents=True, exist_ok=True)
        title = (
            f"Protocol A · {window_kind} · {cleanliness} · n={n_folds}\n"
            f"fold-mean orig vs recon · NC hull · mean r={mean_r:.3f}"
        )
        png = overview / f"pooled_fold_pixel_r__{window_kind}.png"
        save_single_map(corr_map, out_path=png, title=title, underlay=underlay)
        np.save(
            overview / f"pooled_fold_pixel_r__{window_kind}.npy",
            corr_map.astype(np.float32),
        )
        key = (window_kind, cleanliness)
        maps[key] = corr_map
        means[key] = float(mean_r)
        ns[key] = int(n_folds)
        rows.append(
            {
                "window_kind": window_kind,
                "cleanliness": cleanliness,
                "leaf_dir": str(protocol_dir.relative_to(repo)),
                "n_folds": n_folds,
                "mean_r_hull": float(mean_r),
                "n_pixels_hull": int(hull_mask.sum()),
                "png": str(png.relative_to(repo)),
                "fold_ids": fold_ids,
            }
        )

    # Pair plots: clean zscore vs raw, all zscore vs raw
    for cleanliness in ("clean", "all"):
        zk = ("zscore", cleanliness)
        rk = ("raw", cleanliness)
        if zk in maps and rk in maps:
            out = (
                run_root
                / f"pooled_fold_pixel_r__protocol_A_{cleanliness}_zscore_vs_raw.png"
            )
            _save_zscore_raw_pair(
                zscore_map=maps[zk],
                raw_map=maps[rk],
                zscore_mean=means[zk],
                raw_mean=means[rk],
                zscore_n=ns[zk],
                raw_n=ns[rk],
                out_path=out,
                title=(
                    f"Protocol A encoding · {cleanliness} trials · "
                    "pooled fold pixel-r (NC hull)"
                ),
            )

    # 2×2 grid if all four present
    grid_keys = [
        ("zscore", "clean"),
        ("raw", "clean"),
        ("zscore", "all"),
        ("raw", "all"),
    ]
    if all(k in maps for k in grid_keys):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), layout="constrained")
        im = None
        for ax, key in zip(axes.ravel(), grid_keys):
            window_kind, cleanliness = key
            title = (
                f"{window_kind} · {cleanliness} · n={ns[key]}\n"
                f"mean r (NC hull) = {means[key]:.3f}"
            )
            im = ax.imshow(maps[key], cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.colorbar(
            im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Pearson r"
        )
        fig.suptitle(
            "Protocol A pooled fold-level encoding pixel-r (NC hull)",
            fontsize=11,
        )
        grid_path = run_root / "pooled_fold_pixel_r__2x2_clean_all_zscore_raw.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    table_rows = []
    for window_kind in ("zscore", "raw"):
        table_rows.append(
            {
                "window": window_kind,
                "clean_encoding_mean_r_hull": means.get((window_kind, "clean")),
                "all_data_encoding_mean_r_hull": means.get((window_kind, "all")),
            }
        )
    table = pd.DataFrame(table_rows)
    table_path = run_root / "corr_summary_encoding.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "run_root": str(run_root.relative_to(repo)),
        "leaves": rows,
        "encoding_mean_r_hull": {
            f"{w}_{c}": means[(w, c)] for (w, c) in means
        },
    }
    summary_path = run_root / "pooled_fold_pixel_r_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {summary_path.relative_to(repo)}")
    print(f"Wrote {table_path.relative_to(repo)}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Flat run root (…/YYYY-MM-DD_35-46_resnet18_l3)",
    )
    p.add_argument(
        "--ridge-config",
        type=Path,
        default=Path("configs/ridge/default.yaml"),
    )
    p.add_argument("--spatial-size", type=int, nargs=2, default=(100, 100))
    p.add_argument("--avg-method", type=str, default="mean")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    run_root = (
        args.run_root if args.run_root.is_absolute() else repo / args.run_root
    )
    assemble(
        run_root,
        repo=repo,
        ridge_config=args.ridge_config,
        spatial_size=tuple(int(x) for x in args.spatial_size),
        avg_method=args.avg_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
