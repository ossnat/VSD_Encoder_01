#!/usr/bin/env python3
"""Side-by-side comparison with thesis Fig. 13 (Molad thesis, page 31).

Rows: D, F, L, N, vertical bar, horizontal bar (panels c–h).
Columns: thesis stimulus crop | thesis VSD crop (20/11/18 left column) |
          our Schira-only ``(u,v)`` | our thin overlay on mean raw VSD.

Thesis crops are taken from ``_throwaway_thesis_fig10_pdf_pages/page_031.png``.
Bars are not in the 201118 Encoder catalog; overlays use 100718a VSD with
201118 Schira params (same geometry as ``run_schira_only.py``).

Usage:
  scripts/py experiments/schira2007/make_vs_thesis_fig13_comparison.py --set 201118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP
from src.retinotopy.params import load_schira_set
from src.retinotopy.warp import forward_ink_cloud_w, forward_ink_mask_on_vsd
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
THESIS_PAGE = HERE / "_throwaway_thesis_fig10_pdf_pages" / "page_031.png"
OUT_DIR = HERE / "phase_0_register"

# Panel c–h, 20/11/18 left-column VSD circles (fraction of page height).
THESIS_ROWS: dict[str, dict] = {
    "letter_D_white_1": {"panel": "c", "y_frac": 0.30, "label": "D"},
    "letter_F_white_1": {"panel": "d", "y_frac": 0.33, "label": "F"},
    "letter_L_white_1": {"panel": "e", "y_frac": 0.40, "label": "L"},
    "letter_N_white_1": {"panel": "f", "y_frac": 0.35, "label": "N"},
    "black_bar_vertical_1": {"panel": "g", "y_frac": 0.55, "label": "bar V"},
    "black_bar_horizontal_1": {"panel": "h", "y_frac": 0.50, "label": "bar H"},
}

def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int = 630) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 224))
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas_size / base_canvas)
    return RenderConfig(
        canvas_size=canvas_size,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=int(stimuli_cfg.get("bar_width_px", 1)),
        contour_width_px=int(stimuli_cfg.get("contour_width_px", 1)),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _crop_thesis(page: Image.Image, y_frac: float, *, stim: bool) -> np.ndarray:
    w, h = page.size
    y = int(y_frac * h)
    half = 80
    if stim:
        box = (int(0.55 * w), y - half, int(0.72 * w), y + half)
    else:
        box = (int(0.38 * w), y - half, int(0.54 * w), y + half)
    return np.asarray(page.crop(box))


def _vsd_clim(vsd: np.ndarray) -> tuple[float, float]:
    finite = vsd[np.isfinite(vsd)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _catalog_row(
    catalog: pd.DataFrame,
    stimulus_id: str,
    *,
    prefer_prefix: str,
) -> pd.Series:
    sub = catalog[catalog["stimulus_id"].astype(str) == stimulus_id].copy()
    if sub.empty:
        raise RuntimeError(f"No catalog row for {stimulus_id!r}")
    on_date = sub[sub["h5_session"].astype(str).str.startswith(prefer_prefix)]
    if not on_date.empty:
        return on_date.sort_values("h5_session").iloc[0]
    return sub.sort_values("h5_session").iloc[0]


def _mean_vsd_for_row(
    row: pd.Series,
    *,
    repo: Path,
    trials: pd.DataFrame,
    spatial_size: tuple[int, int],
    start_frame: int,
    end_frame: int,
    avg_method: str,
    cache_dir: Path,
) -> np.ndarray:
    date = str(row["h5_session"])
    condition = str(row["condition"])
    cache_path = cache_dir / f"{date}__{condition}__mean_raw.npy"
    if cache_path.exists():
        return np.load(cache_path)

    group = trials[
        (trials["date"].astype(str) == date)
        & (trials["condition"].astype(str) == condition)
    ].copy()
    if group.empty:
        raise RuntimeError(f"No trials for {date} / {condition}")
    group = group.sort_values("trial_global_id").head(8)
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
    return mean_map


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_name", default="201118")
    p.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    p.add_argument("--thesis-page", type=Path, default=THESIS_PAGE)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)

    repo = project_root()
    set_id, params, affine, session_raw = load_schira_set(
        args.schira_config, set_name=args.set_name
    )
    date_prefix = str(session_raw.get("date_prefix") or set_id)

    cfg = _load_yaml(DEFAULT_CONFIG)
    cfg.update(_load_yaml(DEFAULT_WINDOW))
    stimuli_cfg = _load_yaml(DEFAULT_STIMULI)
    render_cfg = _render_config(stimuli_cfg, canvas_size=630)

    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=str(cfg["monkey"]),
        bar_length_deg=render_cfg.bar_length_deg,
    )
    catalog = attach_stimulus_ids(catalog)
    catalog = catalog[~catalog["is_blank"]].copy()
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
    available = trials["target_file"].apply(lambda p: resolve_data_path(p, repo).exists())
    trials = trials.loc[available].reset_index(drop=True)

    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    avg_method = str(cfg.get("avg_method", "mean"))
    cache_dir = HERE / "phase_0" / "cache"

    thesis_path = args.thesis_page if args.thesis_page.is_absolute() else repo / args.thesis_page
    if not thesis_path.exists():
        raise FileNotFoundError(f"Thesis page not found: {thesis_path}")
    thesis_page = Image.open(thesis_path)

    schira_dir = HERE / "phase_0_schira"
    out_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(THESIS_ROWS)
    row_items = sorted(
        THESIS_ROWS.items(),
        key=lambda kv: kv[1]["y_frac"],
    )
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(11.5, 2.35 * n_rows), layout="constrained"
    )

    index_rows: list[dict] = []

    for i, (stim_id, meta) in enumerate(row_items):
        label = meta["label"]
        panel = meta["panel"]
        y_frac = meta["y_frac"]

        thesis_stim = _crop_thesis(thesis_page, y_frac, stim=True)
        thesis_vsd = _crop_thesis(thesis_page, y_frac, stim=False)

        axes[i, 0].imshow(thesis_stim)
        axes[i, 0].set_title(f"Thesis {panel} · stim", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(thesis_vsd)
        axes[i, 1].set_title(f"Thesis {panel} · VSD + model", fontsize=9)
        axes[i, 1].axis("off")

        schira_png = schira_dir / f"{date_prefix}__{stim_id}.png"
        if schira_png.exists():
            axes[i, 2].imshow(plt.imread(str(schira_png)))
            axes[i, 2].set_title("Ours · Schira (u,v)", fontsize=9)
        else:
            axes[i, 2].text(0.5, 0.5, "run_schira_only.py", ha="center", va="center")
            axes[i, 2].set_title("Schira missing", fontsize=9)
        axes[i, 2].axis("off")

        row = _catalog_row(catalog, stim_id, prefer_prefix=date_prefix)
        spec = stimulus_spec_from_mapping(row.to_dict())
        stimulus_rgb = render_stimulus(spec, render_cfg)
        vsd_session = str(row["h5_session"])
        overlay_note = vsd_session
        if not vsd_session.startswith(date_prefix):
            overlay_note = f"{vsd_session} (geom only; Schira {set_id})"

        mean_vsd = _mean_vsd_for_row(
            row,
            repo=repo,
            trials=trials,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
            cache_dir=cache_dir,
        )
        clim = _vsd_clim(mean_vsd)
        ink = forward_ink_mask_on_vsd(
            stimulus_rgb,
            params=params,
            affine=affine,
            render_cfg=render_cfg,
            spatial_size=mean_vsd.shape,
        )
        axes[i, 3].imshow(mean_vsd, cmap=VSD_CMAP, vmin=clim[0], vmax=clim[1])
        ov = np.zeros((*ink.shape, 4), dtype=np.float32)
        ov[ink] = (0.0, 0.0, 0.0, 0.95)
        axes[i, 3].imshow(ov)
        axes[i, 3].set_title(f"Ours · overlay ({overlay_note})", fontsize=9)
        axes[i, 3].axis("off")

        axes[i, 0].set_ylabel(label, fontsize=11, rotation=0, labelpad=28, va="center")

        fu, fv, _ = forward_ink_cloud_w(stimulus_rgb, params=params, render_cfg=render_cfg)
        index_rows.append(
            {
                "stimulus_id": stim_id,
                "thesis_panel": panel,
                "vsd_session": vsd_session,
                "n_schira_ink": int(fu.size),
                "n_overlay_ink": int(np.count_nonzero(ink)),
            }
        )

    fig.suptitle(
        f"Thesis Fig. 13 vs ours · Schira set {set_id} "
        f"(a={params.a}, α={params.alpha}, sech_amp={params.sech_amp})\n"
        "Thesis: chamber registration ~1.5 px RMSD; our overlay uses landmark affine",
        fontsize=11,
    )

    grid_path = out_dir / f"vs_thesis_fig13_grid__{set_id}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Full thesis page reference
    ref_path = out_dir / f"vs_thesis_fig13_reference_page__{set_id}.png"
    fig2, ax2 = plt.subplots(figsize=(8, 10), layout="constrained")
    ax2.imshow(np.asarray(thesis_page))
    ax2.set_title("Thesis Fig. 13 full page (reference)", fontsize=11)
    ax2.axis("off")
    fig2.savefig(ref_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    meta = {
        "set_name": set_id,
        "thesis_page": str(thesis_path.relative_to(repo)),
        "grid_figure": str(grid_path.relative_to(repo)),
        "reference_page": str(ref_path.relative_to(repo)),
        "rows": index_rows,
        "note": (
            "Thesis left column = 20/11/18. Bars use non-201118 VSD when absent "
            "from 201118 catalog; Schira params still from sets.yaml 201118."
        ),
    }
    meta_path = out_dir / f"vs_thesis_fig13_grid__{set_id}.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"Wrote {grid_path}")
    print(f"Wrote {ref_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
