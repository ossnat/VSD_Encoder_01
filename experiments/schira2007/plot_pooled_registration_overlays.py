#!/usr/bin/env python3
"""Thin black Schira ink on mean raw VSD using a pooled registration fit.

Reads ``affine_fit__*__pooled.yaml`` (or any affine_fit YAML), loads each
letter's landmark YAML for trials / canvas, forward-maps ink through the fitted
affine, and saves overlay PNGs (thesis Fig. 13 thin-line style).

Usage:
  scripts/py experiments/schira2007/plot_pooled_registration_overlays.py

  scripts/py experiments/schira2007/plot_pooled_registration_overlays.py \\
      --affine-fit experiments/schira2007/phase_0_register/affine_fit__201118__pooled.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.plotting import plot_overlay_montage, plot_vsd_with_warped_stimulus
from src.retinotopy.schira import SchiraParams
from src.retinotopy.warp import forward_ink_mask_on_vsd
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
DEFAULT_AFFINE_FIT = HERE / "phase_0_register" / "affine_fit__201118__pooled.yaml"
OUT_DIR = HERE / "phase_0_register" / "overlays_pooled"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int | None = None) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 224))
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


def _params_from_fit_block(block: dict) -> SchiraParams:
    return SchiraParams(
        a=float(block["a"]),
        alpha=float(block["alpha"]),
        k=float(block["k"]),
        shear=str(block.get("shear", "double_sech")),
        fa_combine=str(block.get("fa_combine", "power")),
        sech_ecc_k=float(block.get("sech_ecc_k", 0.76)),
        sech_amp=float(block.get("sech_amp", 0.1821)),
    )


def _load_mean_vsd_for_landmarks(
    lm: dict,
    *,
    repo: Path,
    config_path: Path,
    window_path: Path,
) -> np.ndarray:
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    trial_ids = [int(x) for x in lm.get("trial_global_ids") or []]
    if not trial_ids:
        raise RuntimeError(f"No trial_global_ids in landmarks for {lm.get('stimulus_id')}")
    h5_session = str(lm["h5_session"])
    condition = str(lm["condition"])

    trials = load_trial_table(
        cfg["split_csv"],
        str(cfg["monkey"]),
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    trials = trials[trials["date"].astype(str) == h5_session].copy()
    trials = trials[trials["condition"].astype(str) == condition].copy()
    trials = trials[trials["trial_global_id"].isin(trial_ids)].reset_index(drop=True)
    if trials.empty:
        raise RuntimeError(
            f"No trials for {h5_session} {condition} ids={trial_ids[:3]}..."
        )
    maps = [
        load_h5_mean_frame(
            target_file=str(r.target_file),
            trial_global_id=int(r.trial_global_id),
            repo=repo,
            spatial_size=spatial_size,
            start_frame=int(cfg["start_frame"]),
            end_frame=int(cfg["end_frame"]),
            avg_method=str(cfg.get("avg_method", "mean")),
            normalization="none",
        )
        for r in trials.itertuples(index=False)
    ]
    return np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)


def _stimulus_rgb_for_landmarks(
    lm: dict,
    *,
    repo: Path,
    config_path: Path,
    stimuli_yaml: Path,
) -> tuple[np.ndarray, RenderConfig]:
    stimuli_cfg = _load_yaml(stimuli_yaml)
    canvas = lm.get("schira_canvas_size") or lm.get("canvas_size")
    render_cfg = _render_config(
        stimuli_cfg,
        canvas_size=int(canvas) if canvas is not None else None,
    )
    cfg = _load_yaml(config_path)
    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=str(cfg["monkey"]),
        bar_length_deg=render_cfg.bar_length_deg,
    )
    date_prefix = str(lm.get("date_prefix") or lm.get("set_name") or "")
    catalog = catalog[catalog["h5_session"].astype(str).str.startswith(date_prefix)].copy()
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
    stim_id = str(lm["stimulus_id"])
    hit = catalog[catalog["stimulus_id"].astype(str) == stim_id]
    if hit.empty:
        raise RuntimeError(f"No catalog row for {stim_id!r}")
    row = hit.sort_values("h5_session").iloc[0]
    spec = stimulus_spec_from_mapping(row.to_dict())
    return render_stimulus(spec, render_cfg), render_cfg


def plot_pooled_overlays(
    *,
    affine_fit_path: Path,
    repo: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    window_path: Path = DEFAULT_WINDOW,
    stimuli_yaml: Path = DEFAULT_STIMULI,
    output_dir: Path | None = None,
    three_panel: bool = False,
) -> Path:
    repo = repo or project_root()
    fit = _load_yaml(affine_fit_path)
    if not affine_fit_path.is_absolute():
        affine_fit_path = (repo / affine_fit_path).resolve()

    params = _params_from_fit_block(fit.get("schira_held_fixed") or {})
    affine = CorticalAffine.from_mapping(fit["affine"])
    spatial_size = tuple(int(x) for x in _load_yaml(config_path)["spatial_size"])

    per_stim = (fit.get("fit") or {}).get("per_stimulus") or {}
    if not per_stim:
        raise RuntimeError(f"No fit.per_stimulus in {affine_fit_path}")

    out_dir = output_dir or OUT_DIR
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rmsd = float((fit.get("fit") or {}).get("rmsd_px", 0.0))
    affine_note = (
        f"pooled RMSD={rmsd:.2f} px · origin=({affine.origin_x:.1f},{affine.origin_y:.1f}) "
        f"ppu={affine.pixels_per_unit:.1f} rot={affine.rotation_deg:.1f}°"
    )

    montage_panels: list[tuple[np.ndarray, np.ndarray, str]] = []
    index_rows: list[dict] = []

    for stim_id, stim_meta in per_stim.items():
        lm_path = Path(stim_meta["landmarks_file"])
        if not lm_path.is_absolute():
            lm_path = (repo / lm_path).resolve()
        lm = _load_yaml(lm_path)
        mean_vsd = _load_mean_vsd_for_landmarks(
            lm,
            repo=repo,
            config_path=config_path,
            window_path=window_path,
        )
        stimulus_rgb, render_cfg = _stimulus_rgb_for_landmarks(
            lm,
            repo=repo,
            config_path=config_path,
            stimuli_yaml=stimuli_yaml,
        )
        ink = forward_ink_mask_on_vsd(
            stimulus_rgb,
            params=params,
            affine=affine,
            render_cfg=render_cfg,
            spatial_size=spatial_size,
        )
        stim_rmsd = float(stim_meta.get("rmsd_px", 0.0))
        label = f"{stim_id}\nRMSD {stim_rmsd:.1f} px"
        overlay_path = out_dir / f"overlay__{fit.get('set_name')}__{stim_id}.png"
        if three_panel:
            plot_vsd_with_warped_stimulus(
                mean_vsd,
                stimulus_rgb,
                None,
                ink,
                overlay_path,
                title=f"{lm['h5_session']} {stim_id}",
                affine_note=affine_note,
                outline_iters=0,
                overlay_title="VSD + thin forward Schira ink",
            )
        else:
            plot_overlay_montage(
                [(mean_vsd, ink, label)],
                overlay_path,
                title=f"{lm['h5_session']} · {stim_id} · pooled affine",
                outline_iters=0,
                ncol=1,
            )
        montage_panels.append((mean_vsd, ink, label))
        index_rows.append(
            {
                "stimulus_id": stim_id,
                "landmarks_file": str(lm_path),
                "figure": str(overlay_path.relative_to(repo)),
                "n_ink_pixels": int(np.count_nonzero(ink)),
                "rmsd_px": stim_rmsd,
            }
        )
        print(
            f"{stim_id}: ink={int(np.count_nonzero(ink))} px → "
            f"{overlay_path.relative_to(repo)}",
            flush=True,
        )

    montage_path = out_dir / f"montage__{fit.get('set_name')}__pooled.png"
    plot_overlay_montage(
        montage_panels,
        montage_path,
        title=(
            f"Pooled registration overlays · {fit.get('set_name')} · "
            f"RMSD={rmsd:.2f} px · thin black Schira ink"
        ),
        outline_iters=0,
        ncol=2,
    )
    meta_path = out_dir / f"index__{fit.get('set_name')}__pooled.yaml"
    meta_path.write_text(
        yaml.safe_dump(
            {
                "affine_fit": str(affine_fit_path),
                "montage": str(montage_path.relative_to(repo)),
                "affine_note": affine_note,
                "panels": index_rows,
            },
            sort_keys=False,
        )
    )
    print(f"Montage → {montage_path.relative_to(repo)}", flush=True)
    return montage_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--affine-fit",
        type=Path,
        default=DEFAULT_AFFINE_FIT,
        help="Pooled (or single) affine_fit YAML from run_phase_0_register",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    p.add_argument("--stimuli-yaml", type=Path, default=DEFAULT_STIMULI)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--three-panel",
        action="store_true",
        help="Also save stimulus | VSD | overlay triptychs per letter",
    )
    args = p.parse_args()
    plot_pooled_overlays(
        affine_fit_path=args.affine_fit,
        config_path=args.config,
        window_path=args.window,
        stimuli_yaml=args.stimuli_yaml,
        output_dir=args.output_dir,
        three_panel=args.three_panel,
    )


if __name__ == "__main__":
    main()
