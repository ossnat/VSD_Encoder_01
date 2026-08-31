#!/usr/bin/env python3
"""Pooled fold-level odd/even noise correlation for Protocol A/C folds.

Mirrors across-condition NC ROI split-half logic
(``experiments/noise_ceiling_roi/nc_roi_utils.py``), but units follow the
encoding fold list:

Protocol A:
  1. For each fold: sort trials by ``trial_global_id`` within the held-out
     ``(date, condition)``, odd/even split, mean map per half.

Protocol C (stimulus-level, aligned with encoding stimulus-mean pooled maps):
  1. For each fold: odd/even split of **all** trials with the held-out
     ``stimulus_id`` across sessions (sorted by ``trial_global_id``).

Then:
  2. Stack → ``(n_folds, H, W)`` odd and even.
  3. Per pixel: Pearson r across the n_folds vector (same n as encoding
     pooled fold-level pixel r).
  4. Summarize mean r inside the official NC hull mask.

Also reports a secondary within-fold map average (per-fold odd-even pixel
``r_map``, then mean across folds) for reference.

Usage (from repo root)::

  scripts/py experiments/loo_encoding/compute_fold_noise_corr_odd_even.py
  scripts/py experiments/loo_encoding/compute_fold_noise_corr_odd_even.py \\
    --protocol C --window-kind raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.noise_ceiling_roi.nc_roi_utils import (
    DEFAULT_CONFIG,
    half_mean_maps,
    load_pairs,
    load_stimulus_trial_stack,
    pixel_correlate_stacks,
    pixel_r2_reliability_map,
    pixel_reliability_map,
)
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan, masked_map_summary
from src.evaluation.plotting import plot_pixel_correlation_heatmap, plot_pixel_r2_heatmap
from src.evaluation.pixel_correlation import pixel_r2_across_trials
from src.evaluation.roi_mask import load_mask_from_path
from src.paths import project_root
from src.plotting_colormaps import VSD_CMAP

DEFAULT_RUN_ROOT_A = Path(
    "experiments/loo_encoding/runs/2026-08-06_35-46_resnet18_l3"
)
DEFAULT_RUN_ROOT_C = Path(
    "experiments/loo_encoding/runs/2026-08-14_35-46_resnet18_l3"
)
DEFAULT_FOLDS_INDEX_A = (
    DEFAULT_RUN_ROOT_A / "protocol_A_zscore_NChull_all" / "folds_index.yaml"
)
DEFAULT_FOLDS_INDEX_C = (
    DEFAULT_RUN_ROOT_C / "protocol_C_raw_NChull_all" / "folds_index.yaml"
)
DEFAULT_WINDOWS = {
    "zscore": Path("configs/windows/evoked_35_46_zscore.yaml"),
    "raw": Path("configs/windows/evoked_35_46.yaml"),
}


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_folds(folds_index: Path) -> list[dict]:
    payload = _load_yaml(folds_index)
    folds = payload.get("folds") or []
    if not folds:
        raise RuntimeError(f"No folds in {folds_index}")
    return folds


def compute_for_window(
    *,
    window_kind: str,
    window_yaml: Path,
    folds: list[dict],
    repo: Path,
    hull_mask: np.ndarray,
    spatial_size: tuple[int, int],
    protocol: str = "A",
) -> dict:
    cfg = _load_yaml(repo / "configs/default.yaml")
    cfg.update(_load_yaml(window_yaml if window_yaml.is_absolute() else repo / window_yaml))
    spatial_size = tuple(int(x) for x in cfg.get("spatial_size", spatial_size))
    pairs = load_pairs(cfg, repo)

    odd_means: list[np.ndarray] = []
    even_means: list[np.ndarray] = []
    within_r_maps: list[np.ndarray] = []
    within_r2_maps: list[np.ndarray] = []
    fold_rows: list[dict] = []

    print(f"\n=== {window_kind} ({cfg.get('window_id')}) protocol={protocol} ===")
    for fold in folds:
        fold_id = str(fold["fold_id"])
        date = str(fold["heldout_date"])
        condition = str(fold["heldout_condition"])
        stim = str(fold["heldout_stimulus_id"])
        # Protocol C: odd/even over ALL trials with stimulus_id (all sessions),
        # matching stimulus-level encoding pooled maps (n≈18 stimuli).
        # Protocol A: odd/even within the held-out (date, condition) only.
        if protocol == "C":
            unit = pairs[pairs["stimulus_id"].astype(str) == stim].copy()
            unit_label = f"stimulus_id={stim}"
        else:
            unit = pairs[
                (pairs["date"] == date) & (pairs["condition"] == condition)
            ].copy()
            unit_label = f"{date}/{condition}"
        if unit.empty:
            raise RuntimeError(f"{fold_id}: no trials for {unit_label}")
        trials = load_stimulus_trial_stack(
            unit, repo=repo, cfg=cfg, spatial_size=spatial_size
        )
        odd_mean, even_mean, n_odd, n_even = half_mean_maps(trials)
        if n_odd == 0 or n_even == 0:
            raise RuntimeError(
                f"{fold_id}: need both halves (n_odd={n_odd}, n_even={n_even})"
            )
        within_r = pixel_reliability_map(trials)
        within_r2 = pixel_r2_reliability_map(trials)
        odd_means.append(odd_mean)
        even_means.append(even_mean)
        within_r_maps.append(within_r)
        within_r2_maps.append(within_r2)
        row = {
            "fold_id": fold_id,
            "heldout_stimulus_id": stim,
            "heldout_date": date,
            "heldout_condition": condition,
            "n_trials": int(trials.shape[0]),
            "n_sessions": int(
                unit[["date", "condition"]].drop_duplicates().shape[0]
            ),
            "n_odd": int(n_odd),
            "n_even": int(n_even),
            "within_fold_mean_r_hull": float(
                masked_map_summary(within_r, hull_mask)["mean"]
            ),
            "within_fold_mean_r2_hull": float(
                masked_map_summary(within_r2, hull_mask)["mean"]
            ),
        }
        fold_rows.append(row)
        print(
            f"  {fold_id}: n={trials.shape[0]} sessions={row['n_sessions']} "
            f"odd={n_odd} even={n_even} "
            f"within_r_hull={row['within_fold_mean_r_hull']:.4f} "
            f"within_r2_hull={row['within_fold_mean_r2_hull']:.4f}"
        )

    odd_stack = np.stack(odd_means, axis=0)
    even_stack = np.stack(even_means, axis=0)
    r_map = pixel_correlate_stacks(odd_stack, even_stack)
    r2_map = pixel_r2_across_trials(odd_stack, even_stack)
    r_map_masked = apply_mask_nan(r_map, hull_mask)
    r2_map_masked = apply_mask_nan(r2_map, hull_mask)
    hull_summary = masked_map_summary(r_map, hull_mask)
    hull_r2_summary = masked_map_summary(r2_map, hull_mask)

    within_stack = np.stack(within_r_maps, axis=0)
    within_mean_map = np.nanmean(within_stack, axis=0).astype(np.float32)
    within_mean_masked = apply_mask_nan(within_mean_map, hull_mask)
    within_summary = masked_map_summary(within_mean_map, hull_mask)

    within_r2_stack = np.stack(within_r2_maps, axis=0)
    within_mean_r2_map = np.nanmean(within_r2_stack, axis=0).astype(np.float32)
    within_mean_r2_masked = apply_mask_nan(within_mean_r2_map, hull_mask)
    within_r2_summary = masked_map_summary(within_mean_r2_map, hull_mask)

    return {
        "window_kind": window_kind,
        "window_id": str(cfg.get("window_id")),
        "window_yaml": str(window_yaml),
        "normalization": str(cfg.get("normalization", "none")),
        "start_frame": int(cfg["start_frame"]),
        "end_frame": int(cfg["end_frame"]),
        "n_folds": int(odd_stack.shape[0]),
        "r_map": r_map.astype(np.float32),
        "r_map_masked": r_map_masked.astype(np.float32),
        "mean_r_hull": float(hull_summary["mean"]),
        "median_r_hull": float(hull_summary["median"]),
        "r2_map": r2_map.astype(np.float32),
        "r2_map_masked": r2_map_masked.astype(np.float32),
        "mean_r2_hull": float(hull_r2_summary["mean"]),
        "median_r2_hull": float(hull_r2_summary["median"]),
        "within_mean_r_map": within_mean_map,
        "within_mean_r_map_masked": within_mean_masked.astype(np.float32),
        "within_mean_r_hull": float(within_summary["mean"]),
        "within_mean_r2_map": within_mean_r2_map,
        "within_mean_r2_map_masked": within_mean_r2_masked.astype(np.float32),
        "within_mean_r2_hull": float(within_r2_summary["mean"]),
        "fold_rows": fold_rows,
        "odd_stack": odd_stack,
        "even_stack": even_stack,
    }


def plot_side_by_side(
    *,
    zscore_map: np.ndarray,
    raw_map: np.ndarray,
    zscore_mean: float,
    raw_mean: float,
    output_path: Path,
    n_folds: int | None = None,
    metric_label: str = "r",
    suptitle_metric: str = "noise corr",
    protocol_label: str = "A",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
    panels = [
        (f"zscore\nmean {metric_label} (NC hull) = {zscore_mean:.3f}", zscore_map),
        (f"raw\nmean {metric_label} (NC hull) = {raw_mean:.3f}", raw_map),
    ]
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    n_txt = f"n={n_folds} folds, " if n_folds is not None else ""
    fig.suptitle(
        f"Protocol {protocol_label} fold-level odd/even {suptitle_metric} "
        f"({n_txt}NC hull)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _protocol_method_note(protocol: str, n_folds: int) -> str:
    if protocol == "C":
        return (
            f"pooled_stimulus_level_odd_even: for each of {n_folds} Protocol C "
            "stimulus folds, odd/even split of ALL trials with that "
            "stimulus_id across sessions (sorted by trial_global_id; "
            "even=0::2, odd=1::2), mean maps per half; per-pixel Pearson r "
            f"across the fold-level odd vs even vectors (n={n_folds}, aligned "
            "with Protocol C encoding stimulus-level pooled pixel-r). "
            "Mean summarized inside official NC hull. Secondary: within-unit "
            "trial-vector odd/even r_map averaged across folds."
        )
    return (
        f"pooled_fold_level_odd_even: for each of {n_folds} Protocol A "
        "(date,condition) folds, odd/even split of ALL trials "
        "(sorted by trial_global_id; even=0::2, odd=1::2), mean maps "
        "per half; per-pixel Pearson r across the fold-level odd vs even "
        "vectors (same n as encoding pooled fold pixel-r). "
        "Mean summarized inside official NC hull. "
        "Secondary: within-fold trial-vector odd/even r_map averaged "
        "across folds."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pooled fold odd/even noise correlation maps (Protocol A or C)."
    )
    parser.add_argument(
        "--protocol",
        choices=("A", "C"),
        default="A",
        help="A: date×condition folds (~69); C: one fold per stimulus_id (~20)",
    )
    parser.add_argument(
        "--folds-index",
        type=Path,
        default=None,
        help="folds_index.yaml (default: protocol A or C leaf under run root)",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run root when --folds-index omitted (default depends on --protocol)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--window-zscore",
        type=Path,
        default=DEFAULT_WINDOWS["zscore"],
        help="Window YAML for zscore branch",
    )
    parser.add_argument(
        "--window-raw",
        type=Path,
        default=DEFAULT_WINDOWS["raw"],
        help="Window YAML for raw branch",
    )
    parser.add_argument(
        "--window-kind",
        choices=("raw", "zscore"),
        action="append",
        default=None,
        help="Restrict to window kind(s); repeatable. Default: both (A) or raw (C).",
    )
    args = parser.parse_args()
    protocol = str(args.protocol).upper()

    if args.run_root is not None:
        run_root = args.run_root
    elif protocol == "C":
        run_root = DEFAULT_RUN_ROOT_C
    else:
        run_root = DEFAULT_RUN_ROOT_A

    if args.folds_index is not None:
        folds_index = args.folds_index
    elif protocol == "C":
        folds_index = DEFAULT_RUN_ROOT_C / "protocol_C_raw_NChull_all" / "folds_index.yaml"
        if not folds_index.is_file():
            folds_index = run_root / "protocol_C_raw_NChull_all" / "folds_index.yaml"
    else:
        folds_index = DEFAULT_FOLDS_INDEX_A

    if args.output_dir is not None:
        out_dir = args.output_dir
    elif protocol == "C":
        out_dir = run_root / "noise_corr_odd_even__protocol_C"
    else:
        out_dir = run_root / "noise_corr_odd_even"

    if args.window_kind:
        kinds = list(dict.fromkeys(args.window_kind))
    elif protocol == "C":
        kinds = ["raw"]
    else:
        kinds = ["zscore", "raw"]
    windows = {
        kind: (args.window_raw if kind == "raw" else args.window_zscore)
        for kind in kinds
    }

    repo = project_root()
    folds_index = (
        folds_index if folds_index.is_absolute() else repo / folds_index
    )
    run_root = run_root if run_root.is_absolute() else repo / run_root
    out_dir = (
        out_dir if out_dir.is_absolute() else repo / out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg0 = _load_yaml(args.config if args.config.is_absolute() else repo / args.config)
    spatial_size = tuple(int(x) for x in cfg0["spatial_size"])
    hull_path = repo / NOISE_CEILING_HULL_MASK_RELPATH
    hull_mask = load_mask_from_path(hull_path, spatial_size=spatial_size)
    folds = _load_folds(folds_index)
    n_folds = len(folds)

    results: dict[str, dict] = {}
    for kind, win_yaml in windows.items():
        results[kind] = compute_for_window(
            window_kind=kind,
            window_yaml=win_yaml,
            folds=folds,
            repo=repo,
            hull_mask=hull_mask,
            spatial_size=spatial_size,
            protocol=protocol,
        )

    # Save maps + fold tables
    for kind, res in results.items():
        np.save(out_dir / f"r_map_pooled_folds__{kind}.npy", res["r_map"])
        np.save(
            out_dir / f"r_map_pooled_folds_masked__{kind}.npy",
            res["r_map_masked"],
        )
        np.save(
            out_dir / f"r2_map_pooled_folds__{kind}.npy",
            res["r2_map"],
        )
        np.save(
            out_dir / f"r2_map_pooled_folds_masked__{kind}.npy",
            res["r2_map_masked"],
        )
        np.save(
            out_dir / f"r_map_within_fold_mean__{kind}.npy",
            res["within_mean_r_map"],
        )
        np.save(
            out_dir / f"r2_map_within_fold_mean__{kind}.npy",
            res["within_mean_r2_map"],
        )
        plot_pixel_correlation_heatmap(
            res["r_map_masked"],
            out_dir / f"r_map_pooled_folds__{kind}.png",
            title=(
                f"Odd/even noise corr ({kind})  |  "
                f"pooled n={res['n_folds']} folds  |  "
                f"mean hull r={res['mean_r_hull']:.3f}"
            ),
        )
        plot_pixel_r2_heatmap(
            res["r2_map_masked"],
            out_dir / f"r2_map_pooled_folds__{kind}.png",
            title=(
                f"Odd/even noise R² ({kind})  |  "
                f"pooled n={res['n_folds']} folds  |  "
                f"mean hull R²={res['mean_r2_hull']:.3f}"
            ),
        )
        plot_pixel_correlation_heatmap(
            res["within_mean_r_map_masked"],
            out_dir / f"r_map_within_fold_mean__{kind}.png",
            title=(
                f"Within-fold odd/even mean ({kind})  |  "
                f"mean hull r={res['within_mean_r_hull']:.3f}"
            ),
        )
        plot_pixel_r2_heatmap(
            res["within_mean_r2_map_masked"],
            out_dir / f"r2_map_within_fold_mean__{kind}.png",
            title=(
                f"Within-fold odd/even R² mean ({kind})  |  "
                f"mean hull R²={res['within_mean_r2_hull']:.3f}"
            ),
        )
        pd.DataFrame(res["fold_rows"]).to_csv(
            out_dir / f"fold_trial_counts__{kind}.csv", index=False
        )

    if "zscore" in results and "raw" in results:
        plot_side_by_side(
            zscore_map=results["zscore"]["r_map_masked"],
            raw_map=results["raw"]["r_map_masked"],
            zscore_mean=results["zscore"]["mean_r_hull"],
            raw_mean=results["raw"]["mean_r_hull"],
            output_path=out_dir / "r_map_pooled_folds__zscore_vs_raw.png",
            n_folds=n_folds,
            protocol_label=protocol,
        )
        plot_side_by_side(
            zscore_map=results["zscore"]["r2_map_masked"],
            raw_map=results["raw"]["r2_map_masked"],
            zscore_mean=results["zscore"]["mean_r2_hull"],
            raw_mean=results["raw"]["mean_r2_hull"],
            output_path=out_dir / "r2_map_pooled_folds__zscore_vs_raw.png",
            n_folds=n_folds,
            metric_label="R²",
            suptitle_metric="noise R²",
            protocol_label=protocol,
        )

    params = {
        "protocol": protocol,
        "method": _protocol_method_note(protocol, n_folds),
        "comparable_to": (
            f"Protocol {protocol} pooled fold-level encoding pixel r (n={n_folds})"
        ),
        "mirrors": "experiments/noise_ceiling_roi across-condition split-half",
        "folds_index": str(folds_index.relative_to(repo)),
        "n_folds": n_folds,
        "fold_ids": [str(f["fold_id"]) for f in folds],
        "window_kinds": list(results.keys()),
        "trial_filter": (
            "all trials with held-out stimulus_id across sessions "
            "(stimulus-level odd/even; aligned with Protocol C encoding)"
            if protocol == "C"
            else "all trials in held-out (date, condition); no cleanliness filter"
        ),
        "mask": str(NOISE_CEILING_HULL_MASK_RELPATH),
        "n_pixels_hull": int(hull_mask.sum()),
        "windows": {
            kind: {
                "window_yaml": str(windows[kind]),
                "window_id": results[kind]["window_id"],
                "normalization": results[kind]["normalization"],
                "start_frame": results[kind]["start_frame"],
                "end_frame": results[kind]["end_frame"],
            }
            for kind in results
        },
        "primary_mean_r_hull": {
            kind: results[kind]["mean_r_hull"] for kind in results
        },
        "primary_mean_r2_hull": {
            kind: results[kind]["mean_r2_hull"] for kind in results
        },
        "secondary_within_fold_mean_r_hull": {
            kind: results[kind]["within_mean_r_hull"] for kind in results
        },
        "secondary_within_fold_mean_r2_hull": {
            kind: results[kind]["within_mean_r2_hull"] for kind in results
        },
    }
    with (out_dir / "params.yaml").open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False, default_flow_style=False)

    summary: dict = {
        "protocol": protocol,
        "window_kinds": list(results.keys()),
        "n_folds": n_folds,
        "n_pixels_hull": int(hull_mask.sum()),
        "output_dir": str(out_dir.relative_to(repo)),
    }
    for kind, res in results.items():
        summary[f"{kind}_mean_r_hull"] = res["mean_r_hull"]
        summary[f"{kind}_mean_r2_hull"] = res["mean_r2_hull"]
        summary[f"{kind}_within_fold_mean_r_hull"] = res["within_mean_r_hull"]
        summary[f"{kind}_within_fold_mean_r2_hull"] = res["within_mean_r2_hull"]
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Primary (pooled fold-level odd/even, n={n_folds}) ===")
    for kind in results:
        print(
            f"  {kind:6s} mean r / R² (NC hull): "
            f"{results[kind]['mean_r_hull']:.6f} / {results[kind]['mean_r2_hull']:.6f}"
        )
    print("=== Secondary (mean of within-fold trial-vector r maps) ===")
    for kind in results:
        print(
            f"  {kind:6s} within r / R²: "
            f"{results[kind]['within_mean_r_hull']:.6f} / "
            f"{results[kind]['within_mean_r2_hull']:.6f}"
        )
    print(f"Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
