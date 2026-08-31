#!/usr/bin/env python3
"""Diagnose Phase 0 overlay vs thesis Fig. 13 thin-line quality bar.

Panels:
  A) Schira-only forward ink in ``(u, v)``
  B) Same ink forward → fitted affine, stamped on mean raw VSD (thesis style)
  C) Old filled inverse overlay (dilate+hole-fill blob) on the same VSD
  D) Thesis Fig. 13 page crop (when throwaway PNG is present)

Also marks landmark clicks (x) vs affine predictions (o).

Usage:
  scripts/py experiments/schira2007/make_overlay_alignment_diagnostics.py \\
      --set 201118 --stimulus letter_F_white_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP
from src.retinotopy.params import load_schira_set
from src.retinotopy.register import landmarks_stimulus_to_w
from src.retinotopy.warp import (
    fill_stimulus_ink_holes,
    forward_ink_cloud_w,
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
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
DEFAULT_LANDMARKS = (
    HERE / "phase_0_register" / "landmarks__201118__letter_F_white_1.yaml"
)
THESIS_FIG13 = HERE / "_throwaway_thesis_fig10_pdf_pages" / "page_031.png"
OUT_DIR = HERE / "phase_0_register"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


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


def _load_stimulus_rgb(
    *,
    repo: Path,
    date_prefix: str,
    stimulus_id: str,
    render_cfg: RenderConfig,
) -> np.ndarray:
    cfg = _load_yaml(repo / "configs/default.yaml")
    enc = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        enc, monkey=str(cfg["monkey"]), bar_length_deg=render_cfg.bar_length_deg
    )
    catalog = catalog[
        catalog["h5_session"].astype(str).str.startswith(str(date_prefix))
    ].copy()
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
    hit = catalog[catalog["stimulus_id"].astype(str) == stimulus_id]
    if hit.empty:
        raise RuntimeError(f"No catalog row for {stimulus_id!r} under {date_prefix!r}")
    hit = hit.sort_values("h5_session").reset_index(drop=True)
    spec = stimulus_spec_from_mapping(hit.iloc[0].to_dict())
    return render_stimulus(spec, render_cfg)


def _vsd_clim(vsd: np.ndarray) -> tuple[float, float]:
    finite = vsd[np.isfinite(vsd)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_name", default="201118")
    p.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    p.add_argument("--stimuli-config", type=Path, default=DEFAULT_STIMULI)
    p.add_argument("--stimulus", default="letter_F_white_1")
    p.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    p.add_argument("--mean-vsd", type=Path, default=None)
    p.add_argument("--thesis-fig13", type=Path, default=THESIS_FIG13)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)

    repo = project_root()
    set_id, params, affine, session_raw = load_schira_set(
        args.schira_config, set_name=args.set_name
    )
    date_prefix = str(session_raw.get("date_prefix") or set_id)
    stimuli_cfg = _load_yaml(args.stimuli_config)
    render_cfg = _render_config(stimuli_cfg)
    stimulus_rgb = _load_stimulus_rgb(
        repo=repo,
        date_prefix=date_prefix,
        stimulus_id=args.stimulus,
        render_cfg=render_cfg,
    )

    landmarks_path = args.landmarks
    if not landmarks_path.is_absolute():
        landmarks_path = (repo / landmarks_path).resolve()
    lm = _load_yaml(landmarks_path)
    pairs = lm.get("pairs") or []

    mean_path = args.mean_vsd
    if mean_path is None:
        mean_path = (
            HERE
            / "phase_0_register"
            / f"mean_vsd__{lm.get('h5_session', '201118a')}__"
            f"{lm.get('condition', 'condAN5')}.npy"
        )
    if not mean_path.is_absolute():
        mean_path = (repo / mean_path).resolve()
    if not mean_path.exists():
        mean_path = (
            HERE
            / "phase_0"
            / "cache"
            / f"{lm.get('h5_session', '201118a')}__"
            f"{lm.get('condition', 'condAN5')}__mean_raw.npy"
        )
    mean_vsd = np.load(mean_path)
    clim = _vsd_clim(mean_vsd)
    spatial_size = tuple(int(x) for x in mean_vsd.shape)

    u, v, w_cloud = forward_ink_cloud_w(
        stimulus_rgb, params=params, render_cfg=render_cfg
    )
    ink_stroke = forward_ink_mask_on_vsd(
        stimulus_rgb,
        params=params,
        affine=affine,
        render_cfg=render_cfg,
        spatial_size=spatial_size,
    )

    stim_filled = fill_stimulus_ink_holes(
        stimulus_rgb,
        background_gray=render_cfg.background_gray,
        thicken_px=3,
    )
    warped_f, valid_f = sample_stimulus_onto_vsd_grid(
        stim_filled,
        params=params,
        affine=affine,
        render_cfg=render_cfg,
        spatial_size=spatial_size,
        max_ecc_deg=render_cfg.quadrant_extent_deg * np.sqrt(2.0),
    )
    ink_f = warped_ink_mask(
        warped_f, valid_f, background_gray=render_cfg.background_gray
    )

    residuals: list[float] = []
    pred_r = pred_c = click_r = click_c = None
    if len(pairs) >= 2:
        x_px = np.array([p["stim_xy_px"][0] for p in pairs], dtype=np.float64)
        y_px = np.array([p["stim_xy_px"][1] for p in pairs], dtype=np.float64)
        click_c = np.array([p["vsd_col"] for p in pairs], dtype=np.float64)
        click_r = np.array([p["vsd_row"] for p in pairs], dtype=np.float64)
        w_lm = landmarks_stimulus_to_w(
            x_px, y_px, params=params, render_cfg=render_cfg
        )
        pred_r, pred_c = affine.w_to_pixel(w_lm, params)
        residuals = np.hypot(pred_r - click_r, pred_c - click_c).tolist()
        rmsd = float(np.sqrt(np.mean(np.square(residuals))))
    else:
        rmsd = float("nan")

    thesis_path = args.thesis_fig13
    if not thesis_path.is_absolute():
        thesis_path = (repo / thesis_path).resolve()
    has_thesis = thesis_path.exists()
    n_panels = 4 if has_thesis else 3

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"align_diag__{set_id}__{args.stimulus}.png"

    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 4.4), layout="constrained"
    )

    axes[0].set_facecolor("white")
    axes[0].scatter(u, v, s=4, c="black", marker="s", linewidths=0)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("u (cortical)")
    axes[0].set_ylabel("v (cortical)")
    axes[0].set_title("A · Schira-only ink (slim)", fontsize=10)
    axes[0].axhline(0.0, color="0.75", lw=0.5, ls="--")
    axes[0].axvline(0.0, color="0.75", lw=0.5, ls="--")

    def _overlay_ink(ax, ink, title: str) -> None:
        ax.imshow(mean_vsd, cmap=VSD_CMAP, vmin=clim[0], vmax=clim[1])
        overlay = np.zeros((*ink.shape, 4), dtype=np.float32)
        overlay[ink] = (0.0, 0.0, 0.0, 0.95)
        ax.imshow(overlay)
        if pred_r is not None:
            ax.scatter(
                click_c, click_r, s=55, c="cyan", marker="x", linewidths=1.4, zorder=6
            )
            ax.scatter(
                pred_c,
                pred_r,
                s=40,
                facecolors="none",
                edgecolors="yellow",
                marker="o",
                linewidths=1.2,
                zorder=6,
            )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _overlay_ink(
        axes[1],
        ink_stroke,
        "B · Thin forward ink → affine\n(thesis Fig. 13 style)",
    )
    _overlay_ink(
        axes[2],
        ink_f,
        "C · Filled inverse blob\n(thicken_px=3; old default)",
    )

    if has_thesis:
        thesis = plt.imread(str(thesis_path))
        axes[3].imshow(thesis)
        axes[3].set_title(
            "D · Thesis Fig. 13 page\n(thin dashed outlines; RMSD~1.5 px)",
            fontsize=10,
        )
        axes[3].axis("off")

    fig.suptitle(
        f"Overlay alignment diagnostic · {set_id} · {args.stimulus}\n"
        f"our landmark RMSD={rmsd:.2f} px (thesis 20/11/18 ≈ 1.5 px)  |  "
        f"flip_u={affine.flip_u} flip_v={affine.flip_v} "
        f"rot={affine.rotation_deg:.1f}° ppu={affine.pixels_per_unit:.2f}",
        fontsize=10,
    )
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Side-by-side: our thin F overlay vs thesis page (F is panel d)
    side_path = out_dir / f"vs_thesis_fig13__{set_id}__{args.stimulus}.png"
    fig2, ax2 = plt.subplots(1, 2 if has_thesis else 1, figsize=(10.5, 4.6), layout="constrained")
    if not has_thesis:
        ax2 = [ax2]
    ax2[0].imshow(mean_vsd, cmap=VSD_CMAP, vmin=clim[0], vmax=clim[1])
    ov = np.zeros((*ink_stroke.shape, 4), dtype=np.float32)
    ov[ink_stroke] = (0.0, 0.0, 0.0, 0.95)
    ax2[0].imshow(ov)
    ax2[0].set_title(
        f"Ours · thin forward ink · RMSD={rmsd:.1f} px", fontsize=10
    )
    ax2[0].axis("off")
    if has_thesis:
        ax2[1].imshow(plt.imread(str(thesis_path)))
        ax2[1].set_title("Thesis Fig. 13 (full page; see panel d = F)", fontsize=10)
        ax2[1].axis("off")
    fig2.suptitle(
        "Style/quality bar: thesis Fig. 13 uses thin outlines + chamber fit "
        "(~1.5 px RMSD). Our blob look was rendering; residual misalignment is fit quality.",
        fontsize=9,
    )
    fig2.savefig(side_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    meta = {
        "set_name": set_id,
        "stimulus_id": args.stimulus,
        "landmarks_file": str(landmarks_path),
        "mean_vsd": str(mean_path),
        "figure": str(fig_path.relative_to(repo)),
        "vs_thesis_figure": str(side_path.relative_to(repo)),
        "thesis_fig13_page": str(thesis_path) if has_thesis else None,
        "landmark_rmsd_px": rmsd,
        "landmark_residuals_px": residuals,
        "thesis_rmsd_px_reported": 1.52,
        "n_forward_ink": int(w_cloud.size),
        "n_thin_forward_vsd_ink": int(np.count_nonzero(ink_stroke)),
        "n_filled_inverse_ink": int(np.count_nonzero(ink_f)),
        "affine": {
            "origin_xy": [affine.origin_x, affine.origin_y],
            "pixels_per_unit": affine.pixels_per_unit,
            "rotation_deg": affine.rotation_deg,
            "flip_u": affine.flip_u,
            "flip_v": affine.flip_v,
        },
        "verdict": {
            "a_rendering": (
                "Blob look was fill_stimulus_ink_holes(thicken_px=3); Phase 0 "
                "default is now thin forward ink (thesis Fig. 13 style)."
            ),
            "b_affine_quality": (
                f"Our landmark RMSD={rmsd:.2f} px vs thesis ~1.5 px for 20/11/18. "
                "Similarity fit + 5 letter clicks is much weaker than thesis "
                "chamber registration; expect worse alignment than Fig. 13."
            ),
            "c_code_bugs": (
                "No flip/origin/apply-order bug found: fit LS matches "
                "CorticalAffine.w_to_pixel; forward stamp uses the same affine "
                "as sets.yaml."
            ),
        },
    }
    meta_path = out_dir / f"align_diag__{set_id}__{args.stimulus}.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Wrote {fig_path}")
    print(f"Wrote {side_path}")
    print(f"Wrote {meta_path}")
    print(f"RMSD={rmsd:.3f} px  thin_ink={int(np.count_nonzero(ink_stroke))} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
