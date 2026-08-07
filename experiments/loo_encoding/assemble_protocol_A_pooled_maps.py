#!/usr/bin/env python3
"""Pooled fold-level encoding pixel-r maps for flat Protocol A run roots.

Computes per-leaf maps for the four encoding leaves
(``{zscore,raw} × {clean,all}``), writes them under each leaf ``overview/``
and a comparison grid + CSV/JSON summary at the run root.

Usage::

  scripts/py experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \\
    --run-root experiments/loo_encoding/runs/2026-08-07_35-46_resnet18_l3

  # Only raw leaves (e.g. parallel job while zscore is still running):
  scripts/py experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \\
    --run-root … --only-raw

  # Resume after TIMEOUT (skip leaves that already have .npy):
  scripts/py experiments/loo_encoding/assemble_protocol_A_pooled_maps.py \\
    --run-root … --skip-existing
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
from src.evaluation.mask import masked_map_summary
from src.paths import project_root
from src.plotting_colormaps import VSD_CMAP

LEAF_ORDER = (
    ("zscore", "clean", "protocol_A_zscore_NChull_clean"),
    ("zscore", "all", "protocol_A_zscore_NChull_all"),
    ("raw", "clean", "protocol_A_raw_NChull_clean"),
    ("raw", "all", "protocol_A_raw_NChull_all"),
)

LEAF_KEY_ALIASES = {
    "zscore_clean": ("zscore", "clean"),
    "zscore_all": ("zscore", "all"),
    "raw_clean": ("raw", "clean"),
    "raw_all": ("raw", "all"),
}


def _leaf_key(window_kind: str, cleanliness: str) -> str:
    return f"{window_kind}_{cleanliness}"


def _npy_path(protocol_dir: Path, window_kind: str) -> Path:
    return protocol_dir / "overview" / f"pooled_fold_pixel_r__{window_kind}.npy"


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


def _parse_leaf_keys(raw: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for part in raw.split(","):
        token = part.strip().lower().replace("-", "_")
        if not token:
            continue
        if token not in LEAF_KEY_ALIASES:
            allowed = ", ".join(sorted(LEAF_KEY_ALIASES))
            raise argparse.ArgumentTypeError(
                f"Unknown leaf key {part!r}; expected one of: {allowed}"
            )
        keys.add(LEAF_KEY_ALIASES[token])
    if not keys:
        raise argparse.ArgumentTypeError("--leaves must list at least one key")
    return keys


def _filter_leaves(
    leaves: list[tuple[str, str, Path]],
    *,
    leaf_keys: set[tuple[str, str]] | None,
    window_kinds: set[str] | None,
    cleanlinesses: set[str] | None,
) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for window_kind, cleanliness, path in leaves:
        if leaf_keys is not None and (window_kind, cleanliness) not in leaf_keys:
            continue
        if window_kinds is not None and window_kind not in window_kinds:
            continue
        if cleanlinesses is not None and cleanliness not in cleanlinesses:
            continue
        out.append((window_kind, cleanliness, path))
    return out


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


def _load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _n_folds_from_index(protocol_dir: Path) -> int:
    idx = protocol_dir / "folds_index.yaml"
    if not idx.is_file():
        return 0
    plan = _load_yaml(idx)
    folds = plan.get("folds") or []
    return int(len(folds))


def _load_cached_map(
    protocol_dir: Path,
    window_kind: str,
    hull_mask: np.ndarray,
) -> tuple[np.ndarray, float, int] | None:
    npy = _npy_path(protocol_dir, window_kind)
    if not npy.is_file():
        return None
    corr_map = np.load(npy)
    mean_r = float(masked_map_summary(corr_map, hull_mask)["mean"])
    n_folds = _n_folds_from_index(protocol_dir)
    return corr_map, mean_r, n_folds


def assemble(
    run_root: Path,
    *,
    repo: Path,
    ridge_config: Path,
    spatial_size: tuple[int, int],
    avg_method: str = "mean",
    leaf_keys: set[tuple[str, str]] | None = None,
    window_kinds: set[str] | None = None,
    cleanlinesses: set[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    ridge_cfg = _load_yaml(
        ridge_config if ridge_config.is_absolute() else repo / ridge_config
    )
    standardize_features = bool(ridge_cfg.get("standardize_features", True))
    hull_mask = _load_hull_mask(repo, spatial_size)

    leaves = _filter_leaves(
        _resolve_leaves(run_root),
        leaf_keys=leaf_keys,
        window_kinds=window_kinds,
        cleanlinesses=cleanlinesses,
    )
    if not leaves:
        raise ValueError("No leaves selected after --leaves / --window-kind filters")

    rows: list[dict[str, Any]] = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    means: dict[tuple[str, str], float] = {}
    ns: dict[tuple[str, str], int] = {}

    for window_kind, cleanliness, protocol_dir in leaves:
        if not (protocol_dir / "folds_index.yaml").is_file():
            print(f"SKIP missing folds_index: {protocol_dir}", flush=True)
            continue

        npy = _npy_path(protocol_dir, window_kind)
        if skip_existing and npy.is_file():
            cached = _load_cached_map(protocol_dir, window_kind, hull_mask)
            if cached is None:
                print(f"SKIP existing missing/unreadable: {npy}", flush=True)
                continue
            corr_map, mean_r, n_folds = cached
            print(
                f"SKIP existing · {window_kind}/{cleanliness} "
                f"({npy.name}, mean r={mean_r:.3f})",
                flush=True,
            )
            png = protocol_dir / "overview" / f"pooled_fold_pixel_r__{window_kind}.png"
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
                    "png": str(png.relative_to(repo)) if png.is_file() else None,
                    "fold_ids": [],
                    "skipped_existing": True,
                }
            )
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
        np.save(npy, corr_map.astype(np.float32))
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

    summary_path = run_root / "pooled_fold_pixel_r_summary.json"
    existing = _load_existing_summary(summary_path)
    existing_means = dict(existing.get("encoding_mean_r_hull") or {})
    existing_rows = {
        _leaf_key(str(r["window_kind"]), str(r["cleanliness"])): r
        for r in (existing.get("leaves") or [])
        if isinstance(r, dict) and "window_kind" in r and "cleanliness" in r
    }
    for row in rows:
        existing_rows[_leaf_key(row["window_kind"], row["cleanliness"])] = row
    for (w, c), mean_r in means.items():
        existing_means[f"{w}_{c}"] = mean_r

    # Prefer LEAF_ORDER for stable row order; append any extras.
    ordered_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for w, c, _ in LEAF_ORDER:
        k = _leaf_key(w, c)
        if k in existing_rows:
            ordered_rows.append(existing_rows[k])
            seen.add(k)
    for k, row in existing_rows.items():
        if k not in seen:
            ordered_rows.append(row)

    table_rows = []
    for window_kind in ("zscore", "raw"):
        table_rows.append(
            {
                "window": window_kind,
                "clean_encoding_mean_r_hull": existing_means.get(
                    f"{window_kind}_clean"
                ),
                "all_data_encoding_mean_r_hull": existing_means.get(
                    f"{window_kind}_all"
                ),
            }
        )
    table = pd.DataFrame(table_rows)
    table_path = run_root / "corr_summary_encoding.csv"
    table.to_csv(table_path, index=False)

    summary = {
        "run_root": str(run_root.relative_to(repo)),
        "leaves": ordered_rows,
        "encoding_mean_r_hull": existing_means,
    }
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
    p.add_argument(
        "--leaves",
        type=str,
        default=None,
        help=(
            "Comma-separated leaf keys to compute: "
            "zscore_clean,zscore_all,raw_clean,raw_all (default: all four)"
        ),
    )
    p.add_argument(
        "--only-raw",
        action="store_true",
        help="Shorthand for --window-kind raw (raw/clean + raw/all)",
    )
    p.add_argument(
        "--only-zscore",
        action="store_true",
        help="Shorthand for --window-kind zscore",
    )
    p.add_argument(
        "--window-kind",
        choices=("raw", "zscore"),
        action="append",
        default=None,
        help="Restrict to window kind(s); repeatable",
    )
    p.add_argument(
        "--cleanliness",
        choices=("clean", "all"),
        action="append",
        default=None,
        help="Restrict to cleanliness(es); repeatable",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If overview/pooled_fold_pixel_r__{window}.npy exists, load it "
            "and skip recomputation (resume after TIMEOUT)"
        ),
    )
    return p.parse_args(argv)


def _selection_from_args(
    args: argparse.Namespace,
) -> tuple[set[tuple[str, str]] | None, set[str] | None, set[str] | None]:
    leaf_keys: set[tuple[str, str]] | None = None
    window_kinds: set[str] | None = None
    cleanlinesses: set[str] | None = None

    if args.leaves:
        leaf_keys = _parse_leaf_keys(args.leaves)

    kinds: set[str] = set()
    if args.only_raw:
        kinds.add("raw")
    if args.only_zscore:
        kinds.add("zscore")
    if args.window_kind:
        kinds.update(args.window_kind)
    if kinds:
        window_kinds = kinds

    if args.cleanliness:
        cleanlinesses = set(args.cleanliness)

    if args.only_raw and args.only_zscore:
        raise SystemExit("Use only one of --only-raw / --only-zscore")
    if leaf_keys is not None and (window_kinds is not None or cleanlinesses is not None):
        raise SystemExit(
            "Do not combine --leaves with --only-raw/--only-zscore/"
            "--window-kind/--cleanliness"
        )
    return leaf_keys, window_kinds, cleanlinesses


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    run_root = (
        args.run_root if args.run_root.is_absolute() else repo / args.run_root
    )
    leaf_keys, window_kinds, cleanlinesses = _selection_from_args(args)
    assemble(
        run_root,
        repo=repo,
        ridge_config=args.ridge_config,
        spatial_size=tuple(int(x) for x in args.spatial_size),
        avg_method=args.avg_method,
        leaf_keys=leaf_keys,
        window_kinds=window_kinds,
        cleanlinesses=cleanlinesses,
        skip_existing=bool(args.skip_existing),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
