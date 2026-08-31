#!/usr/bin/env python3
"""Phase 0: warp session stimuli through Schira and overlay on raw VSD.

No training. Parameter sets live in YAML (``configs/schira/sets.yaml``).

Default overlay style is **thin-line forward Schira ink** (thesis Fig. 13):
catalog stroke pixels → Schira ``w`` → camera affine → stamp on mean VSD.
Optional ``--ink-style filled`` dilates+hole-fills then inverse-samples (blob).

Usage:
  scripts/py experiments/schira2007/run_phase_0.py
  scripts/py experiments/schira2007/run_phase_0.py --set 201118 --overwrite
  scripts/py experiments/schira2007/run_phase_0.py --set 201118 \\
    --ink-style filled --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.retinotopy.params import load_schira_set
from src.retinotopy.plotting import plot_overlay_montage, plot_vsd_with_warped_stimulus
from src.retinotopy.warp import (
    fill_stimulus_ink_holes,
    forward_ink_mask_on_vsd,
    sample_stimulus_onto_vsd_grid,
    warped_ink_mask,
)
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA_CONFIG = project_root() / "configs/schira/sets.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
PHASE0_DIR = HERE / "phase_0"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _render_config(stimuli_cfg: dict) -> RenderConfig:
    canvas_size = int(stimuli_cfg.get("canvas_size", 224))
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    pixels_per_deg = stimuli_cfg.get("pixels_per_deg")
    if pixels_per_deg is None:
        pixels_per_deg = canvas_size / quadrant_extent_deg
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=float(pixels_per_deg),
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=int(stimuli_cfg.get("bar_width_px", 1)),
        contour_width_px=int(stimuli_cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _sample_trial_rows(group: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    ordered = group.sort_values("trial_global_id").reset_index(drop=True)
    if len(ordered) <= n:
        return ordered
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ordered), size=n, replace=False))
    return ordered.iloc[idx].reset_index(drop=True)


def _mean_frame_cache_path(cache_dir: Path, date: str, condition: str) -> Path:
    return cache_dir / f"{date}__{condition}__mean_raw.npy"


def _load_or_average_trials(
    sample: pd.DataFrame,
    *,
    cache_path: Path,
    repo: Path,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
    overwrite: bool,
) -> tuple[np.ndarray, list[int]]:
    meta_path = cache_path.with_suffix(".yaml")
    trial_ids = [int(x) for x in sample["trial_global_id"].tolist()]
    if cache_path.exists() and meta_path.exists() and not overwrite:
        cached_meta = _load_yaml(meta_path)
        if cached_meta.get("trial_global_ids") == trial_ids:
            return np.load(cache_path), trial_ids
    maps = [
        load_h5_mean_frame(
            target_file=str(row.target_file),
            trial_global_id=int(row.trial_global_id),
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
            normalization="none",
        )
        for row in sample.itertuples(index=False)
    ]
    mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mean_map)
    meta_path.write_text(
        yaml.safe_dump(
            {
                "date": str(sample["date"].iloc[0]),
                "condition": str(sample["condition"].iloc[0]),
                "trial_global_ids": trial_ids,
                "n_trials": len(trial_ids),
                "normalization": "none",
            },
            sort_keys=False,
        )
    )
    return mean_map, trial_ids


def run_phase_0(
    *,
    repo: Path | None = None,
    schira_config: Path = DEFAULT_SCHIRA_CONFIG,
    set_name: str | None = None,
    config_path: Path = DEFAULT_CONFIG,
    window_path: Path = DEFAULT_WINDOW,
    stimuli_yaml: Path = DEFAULT_STIMULI,
    output_dir: Path | None = None,
    n_trials: int = 8,
    seed: int = 17,
    overwrite: bool = False,
    skip_existing: bool = True,
    ink_style: str = "stroke",
) -> dict:
    repo = repo or project_root()
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    stimuli_cfg = _load_yaml(stimuli_yaml)
    render_cfg = _render_config(stimuli_cfg)
    set_id, params, affine, session_raw = load_schira_set(
        schira_config, set_name=set_name
    )

    monkey = str(cfg["monkey"])
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    avg_method = str(cfg.get("avg_method", "mean"))
    date_prefix = session_raw.get("date_prefix")
    if not date_prefix:
        raise ValueError(
            f"Schira set {set_id!r} must define date_prefix (e.g. '201118')"
        )
    date_prefix = str(date_prefix)

    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=monkey,
        bar_length_deg=render_cfg.bar_length_deg,
    )
    catalog = catalog[catalog["h5_session"].astype(str).str.startswith(date_prefix)].copy()
    catalog = catalog[~catalog["is_blank"]].copy()
    catalog = attach_stimulus_ids(catalog)

    trials = load_trial_table(
        cfg["split_csv"],
        monkey,
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    trials = trials[trials["date"].astype(str).str.startswith(date_prefix)].copy()
    available = trials["target_file"].apply(lambda p: resolve_data_path(p, repo).exists())
    n_missing_h5 = int((~available).sum())
    if n_missing_h5:
        print(
            f"Skipping {n_missing_h5} trials with missing session H5 "
            f"(local/workspace copy incomplete)."
        )
    trials = trials.loc[available].reset_index(drop=True)

    merged = trials.merge(
        catalog,
        left_on=["date", "condition"],
        right_on=["h5_session", "condition"],
        how="inner",
        suffixes=("", "_catalog"),
    )
    if merged.empty:
        raise RuntimeError(
            f"No catalog×trial rows for date prefix {date_prefix!r} "
            f"(set {set_id!r}). Check EncoderData CSVs and local session H5s."
        )
    exclude = merged.apply(
        lambda r: is_excluded_encoding_trial(
            str(r["date"]), str(r["condition"]), shape_type=str(r.get("shape_type"))
        ),
        axis=1,
    )
    merged = merged.loc[~exclude].reset_index(drop=True)

    style = str(ink_style).strip().lower()
    if style not in ("filled", "stroke"):
        raise ValueError(
            f"ink_style must be 'filled' or 'stroke', got {ink_style!r}"
        )
    # "stroke" == thesis Fig. 13 thin forward ink; "filled" is optional blob view.

    out_dir = Path(output_dir) if output_dir is not None else PHASE0_DIR
    # PNGs live directly in phase_0/ so they are visible next to the driver.
    fig_dir = out_dir
    cache_dir = out_dir / "cache"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir} (ink_style={style})")

    conf = "UNCONFIRMED" if affine.needs_confirmation else "from landmark fit"
    affine_note = (
        f"set={set_id} affine {conf}: "
        f"origin_xy=({affine.origin_x:g},{affine.origin_y:g}) "
        f"px/unit={affine.pixels_per_unit:g} rot={affine.rotation_deg:g}° "
        f"flip_u={affine.flip_u} flip_v={affine.flip_v} "
        f"(k={params.k:g}, a={params.a:g}, alpha={params.alpha:g}, "
        f"shear={params.shear}, sech_amp={params.sech_amp:g})"
    )
    index_rows: list[dict] = []
    montage_panels: list[tuple[np.ndarray, np.ndarray, str]] = []
    groups = list(merged.groupby(["date", "condition"], sort=True))

    for (date, condition), group in groups:
        spec_row = group.iloc[0].to_dict()
        spec = stimulus_spec_from_mapping(spec_row)
        stim_id = spec_row.get("stimulus_id") or f"{date}_{condition}"
        fig_name = f"{date}__{condition}__{stim_id}.png"
        fig_path = fig_dir / fig_name
        if skip_existing and fig_path.exists() and not overwrite:
            index_rows.append(
                {
                    "date": date,
                    "condition": condition,
                    "stimulus_id": stim_id,
                    "figure": str(fig_path.relative_to(repo)),
                    "skipped": True,
                }
            )
            continue

        sample = _sample_trial_rows(group, n=n_trials, seed=seed)
        cache_path = _mean_frame_cache_path(cache_dir, str(date), str(condition))
        mean_vsd, trial_ids = _load_or_average_trials(
            sample,
            cache_path=cache_path,
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
            overwrite=overwrite,
        )
        stimulus_rgb = render_stimulus(spec, render_cfg)
        # Default "stroke": forward Schira ink cloud → affine → thin VSD
        # stamp (thesis Fig. 13). "filled": dilate+hole-fill then inverse
        # sample (thick 2D support; not the thesis quality bar).
        warped = None
        if style == "filled":
            stimulus_for_warp = fill_stimulus_ink_holes(
                stimulus_rgb,
                background_gray=render_cfg.background_gray,
                thicken_px=3,
            )
            warped, valid = sample_stimulus_onto_vsd_grid(
                stimulus_for_warp,
                params=params,
                affine=affine,
                render_cfg=render_cfg,
                spatial_size=spatial_size,
                max_ecc_deg=render_cfg.quadrant_extent_deg * np.sqrt(2.0),
            )
            ink = warped_ink_mask(
                warped, valid, background_gray=render_cfg.background_gray
            )
            outline_iters = 1
            overlay_title = "VSD + filled inverse warp"
        else:
            ink = forward_ink_mask_on_vsd(
                stimulus_rgb,
                params=params,
                affine=affine,
                render_cfg=render_cfg,
                spatial_size=spatial_size,
            )
            outline_iters = 0
            overlay_title = "VSD + thin forward Schira ink"
        title = (
            f"{date} {condition} · {stim_id} · n={len(trial_ids)} raw trials "
            f"[{start_frame}, {end_frame})"
        )
        plot_vsd_with_warped_stimulus(
            mean_vsd,
            stimulus_rgb,
            warped,
            ink,
            fig_path,
            title=title,
            affine_note=affine_note,
            outline_iters=outline_iters,
            overlay_title=overlay_title,
        )
        montage_panels.append(
            (mean_vsd, ink, f"{date}\n{condition} {stim_id}")
        )
        index_rows.append(
            {
                "date": str(date),
                "condition": str(condition),
                "stimulus_id": str(stim_id),
                "stimulus_text": spec.stimulus_text,
                "pos_x_deg": float(spec.pos_x_deg),
                "pos_y_deg": float(spec.pos_y_deg),
                "n_trials_available": int(len(group)),
                "n_trials_averaged": len(trial_ids),
                "trial_global_ids": trial_ids,
                "n_ink_pixels": int(np.count_nonzero(ink)),
                "n_valid_pixels": (
                    int(np.count_nonzero(valid)) if style == "filled" else int(np.count_nonzero(ink))
                ),
                "figure": str(fig_path.relative_to(repo)),
                "cache": str(cache_path.relative_to(repo)),
                "skipped": False,
            }
        )
        print(
            f"{date} {condition} {stim_id}: "
            f"{len(trial_ids)} trials, ink={int(np.count_nonzero(ink))} px"
        )

    montage_path = fig_dir / "all_stimuli_overlay_montage.png"
    if montage_panels:
        plot_overlay_montage(
            montage_panels,
            montage_path,
            title=(
                f"Schira Phase 0 · {date_prefix} · raw VSD + "
                f"{'thin forward ink' if style == 'stroke' else 'filled inverse'} "
                f"({style})"
            ),
            outline_iters=0 if style == "stroke" else 1,
        )

    index = {
        "set_name": set_id,
        "schira_config": str(schira_config),
        "date_prefix": date_prefix,
        "csv_date": session_raw.get("csv_date"),
        "window": f"[{start_frame}, {end_frame})",
        "normalization": "none",
        "ink_style": style,
        "n_trials_per_stimulus": n_trials,
        "seed": seed,
        "output_dir": str(out_dir.relative_to(repo)),
        "schira": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp": params.sech_amp,
        },
        "affine": {
            "needs_confirmation": affine.needs_confirmation,
            "origin_xy": [affine.origin_x, affine.origin_y],
            "pixels_per_unit": affine.pixels_per_unit,
            "rotation_deg": affine.rotation_deg,
            "flip_u": affine.flip_u,
            "flip_v": affine.flip_v,
        },
        "n_stimuli": len(index_rows),
        "montage": str(montage_path.relative_to(repo)) if montage_panels else None,
        "stimuli": index_rows,
    }
    index_path = out_dir / "index.yaml"
    index_path.write_text(yaml.safe_dump(index, sort_keys=False))
    print(f"Wrote {len(index_rows)} overlays → {out_dir.relative_to(repo)}")
    return index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schira-config",
        type=Path,
        default=DEFAULT_SCHIRA_CONFIG,
        help="YAML registry (sets:) or a standalone single-set file.",
    )
    parser.add_argument(
        "--set",
        dest="set_name",
        type=str,
        default=None,
        help="Named set inside --schira-config (default: YAML default_set).",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Deprecated alias for --schira-config (standalone YAML).",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    parser.add_argument("--stimuli-config", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: experiments/schira2007/phase_0).",
    )
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ink-style",
        choices=("filled", "stroke"),
        default="stroke",
        help=(
            "Overlay ink: 'stroke' (default) = thin forward Schira ink through "
            "affine (thesis Fig. 13). 'filled' = dilate+hole-fill then inverse "
            "sample (thick blob; optional)."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Replot even when the figure PNG already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schira_config = args.session if args.session is not None else args.schira_config
    run_phase_0(
        schira_config=schira_config,
        set_name=args.set_name,
        config_path=args.config,
        window_path=args.window,
        stimuli_yaml=args.stimuli_config,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_existing=not args.no_skip_existing and not args.overwrite,
        ink_style=args.ink_style,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
