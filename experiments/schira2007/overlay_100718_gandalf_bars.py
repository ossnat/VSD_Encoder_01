#!/usr/bin/env python3
"""Overlay Schira-warped 10/07/18 bars on mean VSD.

Two methods:

* ``session`` (default, experiments): one frozen w→pixel scale + one
  rotation+shift from **all** Gandalf landmarks. Same map for every stimulus.
* ``report``: each stimulus uses only **its** table points (H-bar: two-point
  exact similarity; V-bar: one-point shift). For figures/diagnostics only.

Usage:
  scripts/py experiments/schira2007/overlay_100718_gandalf_bars.py
  scripts/py experiments/schira2007/overlay_100718_gandalf_bars.py --placement report
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP
from src.retinotopy.params import load_schira_set
from src.retinotopy.plotting import ink_rgba_overlay
from src.retinotopy.register import (
    affine_from_two_pixel_anchors,
    affine_to_mapping,
    fit_rigid_from_w_pixels,
    fit_shift_from_w_pixel,
    isotropic_scale_w_to_vsd,
    vf_cartesian_px_to_w,
)
from src.retinotopy.warp import forward_ink_mask_on_vsd
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_LANDMARKS = HERE / "phase_0_register" / "landmarks_gandalf_100718.yaml"
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
OUT_DIR_PER_STIM = HERE / "phase_0" / "gandalf_100718_per_stim"
OUT_DIR_RIGID = HERE / "phase_0" / "gandalf_100718_rigid"
OUT_DIR_EXACT = HERE / "phase_0" / "gandalf_100718_bars"
AFFINE_OUT = HERE / "phase_0_register" / "affine_fit__100718__gandalf_rigid.yaml"

BAR_JOBS = (
    {
        "stimulus_id": "black_bar_horizontal_1",
        "label": "horizontal bar 1°",
        "anchor_ids": (2, 4),
    },
    {
        "stimulus_id": "black_bar_vertical_1",
        "label": "vertical bar 1°",
        "anchor_ids": (6,),
    },
)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int | None = None) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 210))
    canvas = int(canvas_size) if canvas_size is not None else base_canvas
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas / base_canvas)
    return RenderConfig(
        canvas_size=canvas,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=max(
            1, int(round(int(stimuli_cfg.get("bar_width_px", 1)) * canvas / base_canvas))
        ),
        contour_width_px=max(
            1,
            int(round(int(stimuli_cfg.get("contour_width_px", 1)) * canvas / base_canvas)),
        ),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _point_by_id(landmarks: dict, point_id: int) -> dict:
    for row in landmarks["points"]:
        if int(row["id"]) == int(point_id):
            return row
    raise KeyError(f"No landmark id {point_id}")


def _vsd_col_row(point: dict) -> tuple[float, float]:
    x, y = point["vsd_xy_px"]
    return float(x), float(y)


def _landmarks_w_pixels(landmarks: dict, params) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    ppd = float(landmarks.get("pixels_per_deg", 35.0))
    ws = []
    rows = []
    cols = []
    ids = []
    for pt in landmarks["points"]:
        w = vf_cartesian_px_to_w(
            *pt["vf_cartesian_px"], params, pixels_per_deg=ppd
        )
        ws.append(complex(np.asarray(w).reshape(())))
        c, r = _vsd_col_row(pt)
        cols.append(c)
        rows.append(r)
        ids.append(int(pt["id"]))
    return (
        np.array(ws, dtype=np.complex128),
        np.array(rows, dtype=np.float64),
        np.array(cols, dtype=np.float64),
        ids,
    )


def _point_residuals(
    affine,
    params,
    w: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ids: list[int],
) -> list[dict]:
    pred_r, pred_c = affine.w_to_pixel(w, params)
    err = np.hypot(pred_r - rows, pred_c - cols)
    out = []
    for i, pid in enumerate(ids):
        out.append(
            {
                "id": int(pid),
                "vsd_xy_px": [float(cols[i]), float(rows[i])],
                "pred_xy_px": [float(pred_c[i]), float(pred_r[i])],
                "err_px": float(err[i]),
            }
        )
    return out


def _w_row_col_for_ids(
    landmarks: dict,
    params,
    ids: tuple[int, ...],
    w_all: np.ndarray,
    rows_all: np.ndarray,
    cols_all: np.ndarray,
    ids_all: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {int(i): k for k, i in enumerate(ids_all)}
    sel = [index[int(i)] for i in ids]
    return w_all[sel], rows_all[sel], cols_all[sel]


def _affine_per_stimulus(
    job: dict,
    *,
    params,
    ppu: float,
    w_all: np.ndarray,
    rows_all: np.ndarray,
    cols_all: np.ndarray,
    ids_all: list[int],
    landmarks: dict,
) -> tuple[object, dict]:
    ids = tuple(int(i) for i in job["anchor_ids"])
    w, rows, cols = _w_row_col_for_ids(
        landmarks, params, ids, w_all, rows_all, cols_all, ids_all
    )
    if w.size >= 2:
        affine, meta = affine_from_two_pixel_anchors(
            complex(w[0]),
            complex(w[1]),
            float(rows[0]),
            float(cols[0]),
            float(rows[1]),
            float(cols[1]),
            params,
        )
        meta = {**meta, "mode": "report_two_point", "anchor_ids": list(ids)}
        return affine, meta
    affine, meta = fit_shift_from_w_pixel(
        complex(w[0]),
        float(rows[0]),
        float(cols[0]),
        params,
        pixels_per_unit=ppu,
        rotation_deg=0.0,
        flip_u=False,
        flip_v=False,
    )
    meta = {**meta, "anchor_ids": list(ids)}
    return affine, meta


def _mean_vsd(
    *,
    repo: Path,
    h5_session: str,
    condition: str,
    trials,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
    cache_dir: Path,
) -> tuple[np.ndarray, list[int]]:
    cache_path = cache_dir / f"mean_vsd__{h5_session}__{condition}.npy"
    meta_path = cache_path.with_suffix(".yaml")
    group = trials[
        (trials["date"].astype(str) == str(h5_session))
        & (trials["condition"].astype(str) == str(condition))
    ].copy()
    available = group["target_file"].apply(
        lambda p: resolve_data_path(p, repo).exists()
    )
    group = group.loc[available].sort_values("trial_global_id").reset_index(drop=True)
    if group.empty:
        raise RuntimeError(f"No local H5 trials for {h5_session} {condition}")
    trial_ids = [int(x) for x in group["trial_global_id"].tolist()]
    if cache_path.exists() and meta_path.exists():
        cached = _load_yaml(meta_path)
        if cached.get("trial_global_ids") == trial_ids:
            return np.load(cache_path), trial_ids
    maps = [
        load_h5_mean_frame(
            target_file=str(r.target_file),
            trial_global_id=int(r.trial_global_id),
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
            normalization="none",
        )
        for r in group.itertuples(index=False)
    ]
    mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mean_map)
    meta_path.write_text(
        yaml.safe_dump(
            {
                "h5_session": h5_session,
                "condition": condition,
                "trial_global_ids": trial_ids,
                "n_trials": len(trial_ids),
                "normalization": "none",
            },
            sort_keys=False,
        )
    )
    return mean_map, trial_ids


def _plot_bar_overlay(
    *,
    stimulus_rgb: np.ndarray,
    mean_vsd: np.ndarray,
    ink: np.ndarray,
    landmarks: dict,
    highlight_ids: tuple[int, ...],
    affine,
    params,
    title: str,
    note: str,
    output_path: Path,
) -> None:
    vsd = np.asarray(mean_vsd, dtype=np.float64)
    finite = vsd[np.isfinite(vsd)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    if lo == hi:
        hi = lo + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), layout="constrained")
    axes[0].imshow(np.asarray(stimulus_rgb))
    axes[0].set_title("Stimulus (VF)", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
    axes[1].set_title("Mean VSD", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
    axes[2].imshow(ink_rgba_overlay(ink, outline_iters=0, alpha=0.55))
    axes[2].set_title("VSD + Schira ink", fontsize=10)
    axes[2].axis("off")

    ppd = float(landmarks["pixels_per_deg"])
    for pt in landmarks["points"]:
        pid = int(pt["id"])
        col, row = _vsd_col_row(pt)
        x_px, y_px = pt["vf_cartesian_px"]
        w = vf_cartesian_px_to_w(x_px, y_px, params, pixels_per_deg=ppd)
        pr, pc = affine.w_to_pixel(w, params)
        hi_pt = pid in highlight_ids
        axes[1].plot(col, row, "o", ms=8 if hi_pt else 5, mfc="none",
                     mec="#f4d03f", mew=1.4 if hi_pt else 0.8, zorder=5)
        axes[2].plot(col, row, "o", ms=8 if hi_pt else 5, mfc="none",
                     mec="#f4d03f", mew=1.4 if hi_pt else 0.8, zorder=5,
                     label="table VSD" if pid == highlight_ids[0] else None)
        axes[2].plot(float(pc), float(pr), "+", ms=9 if hi_pt else 6,
                     color="#5dade2", mew=1.6 if hi_pt else 1.0, zorder=6,
                     label="Schira→VSD" if pid == highlight_ids[0] else None)
    axes[2].legend(loc="lower right", fontsize=7, framealpha=0.8)

    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="0.35")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    parser.add_argument("--set", dest="set_name", default="100718")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    parser.add_argument("--stimuli-config", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument(
        "--placement",
        choices=("session", "report"),
        default="session",
        help=(
            "session: one R+t for all experiments (default). "
            "report: per-stimulus two-point map, figures only."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--canvas-size", type=int, default=630)
    args = parser.parse_args(argv)

    repo = project_root()
    landmarks = _load_yaml(args.landmarks)
    cfg = _load_yaml(args.config)
    cfg.update(_load_yaml(args.window))
    stimuli_cfg = _load_yaml(args.stimuli_config)
    render_cfg = _render_config(stimuli_cfg, canvas_size=args.canvas_size)
    set_id, params, _placeholder, session_raw = load_schira_set(
        args.schira_config, set_name=args.set_name
    )
    date_prefix = str(session_raw["date_prefix"])
    center = landmarks.get("bar_center_deg", [0.6, -0.75])
    cx, cy = float(center[0]), float(center[1])
    w_all, rows_all, cols_all, ids_all = _landmarks_w_pixels(landmarks, params)
    ppu, scale_meta = isotropic_scale_w_to_vsd(w_all, rows_all, cols_all, params)

    shared_affine = None
    shared_meta: dict = {}
    if args.placement == "session":
        shared_affine, shared_meta = fit_rigid_from_w_pixels(
            w_all, rows_all, cols_all, params, pixels_per_unit=ppu
        )
        default_out = OUT_DIR_RIGID
    else:
        default_out = OUT_DIR_PER_STIM

    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=str(cfg["monkey"]),
        bar_length_deg=render_cfg.bar_length_deg,
    )
    catalog = catalog[catalog["h5_session"].astype(str).str.startswith(date_prefix)].copy()
    catalog = catalog[~catalog["is_blank"]].copy()
    catalog = attach_stimulus_ids(catalog)
    catalog = catalog[
        ~catalog.apply(
            lambda r: is_excluded_encoding_trial(
                str(r["h5_session"]),
                str(r["condition"]),
                shape_type=str(r.get("shape_type")),
            ),
            axis=1,
        )
    ]
    trials = load_trial_table(
        cfg["split_csv"],
        str(cfg["monkey"]),
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    out_arg = args.output_dir if args.output_dir is not None else default_out
    out_dir = out_arg if out_arg.is_absolute() else repo / out_arg
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    job_affines: list[dict] = []
    for job in BAR_JOBS:
        hit = catalog[catalog["stimulus_id"].astype(str) == job["stimulus_id"]]
        if hit.empty:
            raise RuntimeError(f"No catalog row for {job['stimulus_id']} under {date_prefix}")
        row = hit.sort_values("h5_session").iloc[0]
        spec = replace(
            stimulus_spec_from_mapping(row.to_dict()),
            pos_x_deg=cx,
            pos_y_deg=cy,
        )
        stimulus_rgb = render_stimulus(spec, render_cfg)
        mean_vsd, trial_ids = _mean_vsd(
            repo=repo,
            h5_session=str(row["h5_session"]),
            condition=str(row["condition"]),
            trials=trials,
            spatial_size=spatial_size,
            start_frame=int(cfg["start_frame"]),
            end_frame=int(cfg["end_frame"]),
            avg_method=str(cfg.get("avg_method", "mean")),
            cache_dir=out_dir,
        )
        if args.placement == "session":
            affine = shared_affine
            place_meta = shared_meta
            place_note = (
                f"session map (all landmarks, frozen s={ppu:.3g}, R+t) · "
                f"RMSD={place_meta.get('rmsd_px'):.2f} px"
            )
        else:
            affine, place_meta = _affine_per_stimulus(
                job,
                params=params,
                ppu=ppu,
                w_all=w_all,
                rows_all=rows_all,
                cols_all=cols_all,
                ids_all=ids_all,
                landmarks=landmarks,
            )
            place_note = (
                f"report two-point (ids {list(job['anchor_ids'])}, "
                f"mode={place_meta.get('mode')})"
            )
        residuals = _point_residuals(
            affine, params, w_all, rows_all, cols_all, ids_all
        )
        ink = forward_ink_mask_on_vsd(
            stimulus_rgb,
            params=params,
            affine=affine,
            render_cfg=render_cfg,
            spatial_size=spatial_size,
        )
        fig_name = f"overlay__{set_id}__{job['stimulus_id']}.png"
        note = (
            f"set={set_id} a={params.a:g} α={params.alpha:g} k={params.k:g} · "
            f"{place_note} · {row['h5_session']} {row['condition']} n={len(trial_ids)} · "
            f"flip_u={affine.flip_u} flip_v={affine.flip_v}"
        )
        _plot_bar_overlay(
            stimulus_rgb=stimulus_rgb,
            mean_vsd=mean_vsd,
            ink=ink,
            landmarks=landmarks,
            highlight_ids=job["anchor_ids"],
            affine=affine,
            params=params,
            title=f"100718 · {job['label']} · Schira on mean VSD ({args.placement})",
            note=note,
            output_path=out_dir / fig_name,
        )
        mapping = affine_to_mapping(affine)
        job_affines.append(
            {
                "stimulus_id": job["stimulus_id"],
                "anchor_ids": list(job["anchor_ids"]),
                "affine": mapping,
                "place_meta": {
                    k: place_meta.get(k)
                    for k in ("mode", "rmsd_px", "anchor_rmsd_px", "check_err_px")
                    if k in place_meta or place_meta.get(k) is not None
                },
                "point_residuals": residuals,
            }
        )
        index_rows.append(
            {
                "stimulus_id": job["stimulus_id"],
                "h5_session": str(row["h5_session"]),
                "condition": str(row["condition"]),
                "n_trials": len(trial_ids),
                "figure": str((out_dir / fig_name).relative_to(repo)),
                "ink_pixels": int(ink.sum()),
            }
        )
        print(f"Wrote {fig_name}  ink={int(ink.sum())} px  trials={len(trial_ids)}")

    summary = {
        "set_name": set_id,
        "placement": args.placement,
        "landmarks": str(args.landmarks),
        "pixels_per_unit_session": ppu,
        "scale": scale_meta,
        "session_affine": (
            affine_to_mapping(shared_affine) if shared_affine is not None else None
        ),
        "session_rmsd_px": shared_meta.get("rmsd_px"),
        "per_stimulus": job_affines,
        "schira": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "fa_combine": params.fa_combine,
            "sech_amp": params.sech_amp,
        },
        "stimuli": index_rows,
    }
    (out_dir / "index.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    if args.placement == "session" and shared_affine is not None:
        AFFINE_OUT.parent.mkdir(parents=True, exist_ok=True)
        AFFINE_OUT.write_text(
            yaml.safe_dump(
                {
                    "set_name": set_id,
                    "mode": "session",
                    "use_for": "all 100718 experiments (not the report two-point maps)",
                    "rmsd_px": shared_meta.get("rmsd_px"),
                    "scale": scale_meta,
                    "affine": affine_to_mapping(shared_affine),
                },
                sort_keys=False,
            )
        )
        print(f"Saved session affine → {AFFINE_OUT.relative_to(repo)}")
    print(f"placement={args.placement} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
