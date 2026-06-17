#!/usr/bin/env python3
"""Build frame-averaged trial xarray NetCDF files from session H5 data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.averaging import average_frames
from src.data.h5_io import read_trial
from src.data.plotting import plot_averaged_samples, select_sample_rows
from src.data.splits import load_trial_table
from src.data.xarray_schema import (
    build_averaged_dataarray,
    save_averaged_trial,
    trial_output_path,
    window_id_from_frames,
)
from src.paths import project_root, resolve_data_path, workspace_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    window_cfg = _load_yaml(window_path)
    cfg.update(window_cfg)
    return cfg


def _portable_data_path(abs_path: Path) -> str:
    ws = workspace_root()
    resolved = abs_path.resolve()
    try:
        rel = resolved.relative_to(ws)
        return str(rel)
    except ValueError:
        return str(resolved)


def build_averaged_trials(
    cfg: dict,
    *,
    max_trials: int | None = None,
    overwrite: bool = False,
    n_plot_samples: int = 4,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    repo = repo_root or project_root()
    monkey = cfg["monkey"]
    spatial_size = tuple(cfg["spatial_size"])
    height, width = spatial_size

    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    avg_method = cfg.get("avg_method", "mean")
    normalization = cfg.get("normalization", "none")
    window_id = cfg.get("window_id") or window_id_from_frames(
        start_frame, end_frame
    )

    averaged_root = resolve_data_path(cfg["paths"]["averaged_root"], repo)
    plots_root = repo / cfg["paths"]["plots_root"]

    trials = load_trial_table(
        cfg["split_csv"],
        monkey,
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    available_mask = trials["target_file"].apply(
        lambda p: resolve_data_path(p, repo).exists()
    )
    n_missing_h5 = int((~available_mask).sum())
    if n_missing_h5:
        print(
            f"Skipping {n_missing_h5} trials with missing session H5 "
            f"(local/workspace copy incomplete)."
        )
        trials = trials[available_mask].reset_index(drop=True)
    if trials.empty:
        raise FileNotFoundError(
            f"No trials with existing H5 files for monkey={monkey!r}"
        )
    if max_trials is not None:
        trials = trials.head(max_trials)

    out_dir = averaged_root / monkey / window_id
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    plot_candidates: list[dict] = []
    plot_images: list[tuple[dict, object]] = []
    n_skipped = 0
    n_written = 0

    for row in trials.itertuples(index=False):
        trial_global_id = int(row.trial_global_id)
        nc_path = trial_output_path(averaged_root, monkey, window_id, trial_global_id)

        if nc_path.exists() and not overwrite:
            n_skipped += 1
            manifest_rows.append(
                {
                    "trial_global_id": trial_global_id,
                    "monkey": monkey,
                    "date": row.date,
                    "condition": row.condition,
                    "condition_code": int(row.condition_code),
                    "trial_index_in_condition": int(row.trial_index_in_condition),
                    "trial_dataset": row.trial_dataset,
                    "target_file": row.target_file,
                    "split": row.split,
                    "window_id": window_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "avg_method": avg_method,
                    "normalization": normalization,
                    "nc_path": _portable_data_path(nc_path),
                }
            )
            continue

        h5_path = resolve_data_path(row.target_file, repo)
        trial = read_trial(h5_path, row.trial_dataset)
        image = average_frames(
            trial,
            start_frame,
            end_frame,
            spatial_size=(height, width),
            method=avg_method,
        )

        da = build_averaged_dataarray(
            image,
            trial_global_id=trial_global_id,
            monkey=monkey,
            date=str(row.date),
            condition=str(row.condition),
            condition_code=int(row.condition_code),
            trial_index_in_condition=int(row.trial_index_in_condition),
            trial_dataset=str(row.trial_dataset),
            target_file=str(row.target_file),
            window_id=window_id,
            start_frame=start_frame,
            end_frame=end_frame,
            source_n_frames=trial.shape[1],
            avg_method=avg_method,
            normalization=normalization,
            split=str(row.split),
        )
        save_averaged_trial(da, nc_path)
        n_written += 1

        meta = {
            "trial_global_id": trial_global_id,
            "monkey": monkey,
            "date": row.date,
            "condition": row.condition,
            "condition_code": int(row.condition_code),
            "trial_index_in_condition": int(row.trial_index_in_condition),
            "trial_dataset": row.trial_dataset,
            "target_file": row.target_file,
            "split": row.split,
            "window_id": window_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "avg_method": avg_method,
            "normalization": normalization,
            "nc_path": _portable_data_path(nc_path),
        }
        manifest_rows.append(meta)
        plot_candidates.append(meta)
        plot_images.append((meta, image))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    # Plot from newly written trials; if all skipped, load a few existing .nc files
    samples_for_plot: list[tuple[dict, object]] = []
    if plot_images:
        sample_meta = select_sample_rows(
            [m for m, _ in plot_images], n_samples=n_plot_samples
        )
        sample_ids = {m["trial_global_id"] for m in sample_meta}
        samples_for_plot = [
            (m, img) for m, img in plot_images if m["trial_global_id"] in sample_ids
        ]
    elif not manifest.empty:
        import xarray as xr

        sample_meta = select_sample_rows(
            manifest.head(max(n_plot_samples * 3, n_plot_samples)).to_dict("records"),
            n_samples=n_plot_samples,
        )
        for meta in sample_meta:
            nc = resolve_data_path(meta["nc_path"], repo)
            da = xr.open_dataarray(nc)
            samples_for_plot.append((meta, da.values))
            da.close()

    plot_dir = plots_root / monkey / window_id
    plot_paths = plot_averaged_samples(samples_for_plot, plot_dir)

    print(f"Monkey: {monkey}")
    print(f"Window: {window_id} [{start_frame}, {end_frame})")
    print(
        f"Trials processed: {len(trials)} "
        f"(written={n_written}, skipped_existing={n_skipped}, "
        f"missing_h5={n_missing_h5})"
    )
    print(f"Manifest: {_portable_data_path(manifest_path)}")
    if plot_paths:
        print("Sample plots:")
        for p in plot_paths:
            print(f"  {p.relative_to(repo)}")
    else:
        print("No sample plots written.")

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
        help="Base config (monkey, paths, split CSV)",
    )
    parser.add_argument(
        "--window",
        type=Path,
        required=True,
        help="Window config (start_frame, end_frame, avg_method, normalization)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Process only the first N trials (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild .nc files even if they already exist",
    )
    parser.add_argument(
        "--n-plot-samples",
        type=int,
        default=4,
        help="Number of averaged-frame QC plots to save",
    )
    parser.add_argument(
        "--monkey",
        type=str,
        default=None,
        help="Override monkey from config (e.g. gandalf)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey
    build_averaged_trials(
        cfg,
        max_trials=args.max_trials,
        overwrite=args.overwrite,
        n_plot_samples=args.n_plot_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
