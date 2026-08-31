#!/usr/bin/env python3
"""Forward-map letter clouds → cortical w (and camera) for Phase 0 diagnostics.

Distinguishes formula collapse (forward is already a curve) from inverse /
thresholding bugs (forward is a 2D patch but overlay is a curve).

Usage:
  scripts/py experiments/schira2007/make_forward_diagnostics.py --set 201118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.linalg import svd

from src.paths import project_root, resolve_data_path
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.params import load_schira_set
from src.retinotopy.schira import (
    SchiraParams,
    cartesian_to_polar,
    forward_schira,
)
from src.retinotopy.visual_field import (
    image_px_to_cartesian_deg,
    stimulus_ink_mask,
)
from src.retinotopy.warp import sample_stimulus_onto_vsd_grid, warped_ink_mask
from src.stimuli.catalog import load_full_encoder_catalog, stimulus_spec_from_mapping
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
OUT_DIR = HERE / "phase_0" / "diagnostics"


def _pca_aspect(xy: np.ndarray) -> tuple[float, np.ndarray]:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return float("nan"), np.array([np.nan, np.nan])
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, s, _ = svd(centered, full_matrices=False)
    aspect = float(s[1] / s[0]) if s[0] > 0 else float("nan")
    return aspect, s


def _render_cfg(stimuli_cfg: dict) -> RenderConfig:
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
    )


def _letter_spec(repo: Path, date_prefix: str, stimulus_id: str = "letter_A_white_1"):
    cfg = yaml.safe_load((repo / "configs/default.yaml").read_text())
    enc = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        enc, monkey=str(cfg["monkey"]), bar_length_deg=1.0
    )
    catalog = catalog[
        catalog["h5_session"].astype(str).str.startswith(str(date_prefix))
    ].copy()
    catalog = catalog[~catalog["is_blank"]].copy()
    catalog = attach_stimulus_ids(catalog)
    hit = catalog[catalog["stimulus_id"] == stimulus_id]
    if hit.empty:
        raise RuntimeError(
            f"No catalog row for {stimulus_id!r} under prefix {date_prefix!r}"
        )
    return stimulus_spec_from_mapping(hit.iloc[0].to_dict())


def plot_forward_comparison(
    *,
    stimulus_rgb: np.ndarray,
    render_cfg: RenderConfig,
    base_params: SchiraParams,
    affine: CorticalAffine,
    amps: list[float],
    output_path: Path,
    title: str,
    spatial_size: tuple[int, int] = (100, 100),
    vsd_underlay: np.ndarray | None = None,
) -> dict:
    """Save VF / forward-w / inverse-ink panels for each ``sech_amp``."""
    ink_vf = stimulus_ink_mask(
        stimulus_rgb, background_gray=render_cfg.background_gray
    )
    yy, xx = np.where(ink_vf)
    x_deg, y_deg = image_px_to_cartesian_deg(
        xx.astype(np.float64), yy.astype(np.float64), render_cfg
    )
    ecc, theta = cartesian_to_polar(x_deg, y_deg)

    n_amp = len(amps)
    fig, axes = plt.subplots(
        n_amp, 3, figsize=(10.5, 3.2 * n_amp), layout="constrained"
    )
    if n_amp == 1:
        axes = np.asarray([axes])

    summary: dict = {
        "n_vf_ink": int(ink_vf.sum()),
        "amps": {},
    }

    for row, amp in enumerate(amps):
        params = SchiraParams(
            a=base_params.a,
            alpha=base_params.alpha,
            k=base_params.k,
            shear=base_params.shear,
            sech_ecc_k=base_params.sech_ecc_k,
            sech_amp=float(amp),
        )
        w = forward_schira(ecc, theta, params)
        finite = np.isfinite(w.real) & np.isfinite(w.imag)
        uv = np.column_stack([w.real[finite], w.imag[finite]])
        aspect_w, s_w = _pca_aspect(uv)
        v_span = float(np.ptp(w.imag[finite])) if finite.any() else float("nan")
        u_span = float(np.ptp(w.real[finite])) if finite.any() else float("nan")

        warped, valid = sample_stimulus_onto_vsd_grid(
            stimulus_rgb,
            params=params,
            affine=affine,
            render_cfg=render_cfg,
            spatial_size=spatial_size,
        )
        ink = warped_ink_mask(
            warped, valid, background_gray=render_cfg.background_gray
        )
        ir, ic = np.where(ink)
        cam = np.column_stack([ic.astype(np.float64), ir.astype(np.float64)])
        aspect_cam, s_cam = _pca_aspect(cam)

        summary["amps"][str(amp)] = {
            "forward_pca_singular_values": [float(x) for x in s_w],
            "forward_aspect": aspect_w,
            "forward_u_span": u_span,
            "forward_v_span": v_span,
            "forward_v_span_px": v_span * float(affine.pixels_per_unit),
            "n_inverse_ink": int(ink.sum()),
            "inverse_pca_singular_values": [float(x) for x in s_cam],
            "inverse_aspect": aspect_cam,
        }

        axes[row, 0].imshow(stimulus_rgb)
        axes[row, 0].contour(ink_vf, levels=[0.5], colors="cyan", linewidths=0.6)
        axes[row, 0].set_title(f"VF ink (amp={amp:g})", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].scatter(
            uv[:, 0], uv[:, 1], s=2, c="k", alpha=0.35, linewidths=0
        )
        axes[row, 1].set_aspect("equal", adjustable="datalim")
        axes[row, 1].set_xlabel("Re(w)")
        axes[row, 1].set_ylabel("Im(w)")
        axes[row, 1].set_title(
            f"Forward letter cloud\naspect={aspect_w:.3f}  "
            f"v≈{v_span * affine.pixels_per_unit:.1f}px",
            fontsize=9,
        )
        axes[row, 1].grid(True, alpha=0.25)

        # Prefer mean VSD underlay when available; else light canvas.
        if vsd_underlay is not None:
            axes[row, 2].imshow(vsd_underlay, cmap="gray")
            if ink.any():
                axes[row, 2].scatter(
                    ic.astype(np.float64),
                    ir.astype(np.float64),
                    s=6,
                    c="cyan",
                    alpha=0.85,
                    linewidths=0,
                )
            axes[row, 2].set_title(
                f"Inverse ink on VSD\nn={int(ink.sum())} aspect={aspect_cam:.3f}",
                fontsize=9,
            )
        else:
            canvas = np.full((*spatial_size, 3), 230, dtype=np.uint8)
            canvas[ink] = 0
            axes[row, 2].imshow(canvas)
            axes[row, 2].set_title(
                f"Inverse ink on camera\nn={int(ink.sum())} aspect={aspect_cam:.3f}",
                fontsize=9,
            )
        axes[row, 2].axis("off")

    fig.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    parser.add_argument("--set", dest="set_name", type=str, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stimuli-config", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stimulus-id", type=str, default="letter_A_white_1")
    args = parser.parse_args(argv)

    repo = project_root()
    set_id, params, affine, block = load_schira_set(
        args.schira_config, set_name=args.set_name
    )
    date_prefix = str(block.get("date_prefix") or set_id)
    stimuli_cfg = yaml.safe_load(args.stimuli_config.read_text())
    render_cfg = _render_cfg(stimuli_cfg)
    spec = _letter_spec(repo, date_prefix, stimulus_id=args.stimulus_id)
    rgb = render_stimulus(spec, render_cfg)

    # Prefer 201118a letter-A mean as camera underlay when cache exists.
    vsd_underlay = None
    cache_a = (
        HERE
        / "phase_0"
        / "cache"
        / f"{date_prefix}a__condAN2__mean_raw.npy"
    )
    if cache_a.exists():
        vsd_underlay = np.load(cache_a)

    # Compare YAML amp against Schira's published 0.1821 and the 1.821 reading.
    amps = sorted({float(params.sech_amp), 0.1821, 1.821})
    out = Path(args.output_dir)
    fig_path = out / f"{set_id}__{args.stimulus_id}__forward_amp_compare.png"
    summary = plot_forward_comparison(
        stimulus_rgb=rgb,
        render_cfg=render_cfg,
        base_params=params,
        affine=affine,
        amps=amps,
        output_path=fig_path,
        title=(
            f"Forward diagnostic · set={set_id} · {args.stimulus_id} · "
            f"a={params.a:g} α={params.alpha:g} k={params.k:g} · "
            f"origin=({affine.origin_x:g},{affine.origin_y:g}) "
            f"ppu={affine.pixels_per_unit:g}"
        ),
        vsd_underlay=vsd_underlay,
    )
    summary_path = out / f"{set_id}__{args.stimulus_id}__forward_amp_compare.yaml"
    payload = {
        "set_name": set_id,
        "stimulus_id": args.stimulus_id,
        "schira": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp_yaml": params.sech_amp,
        },
        "figure": str(fig_path.relative_to(repo)),
        **summary,
        "note": (
            "If forward_aspect ≪ 0.15 and forward_v_span_px ≲ 2, the formula "
            "(usually sech_amp too small for this α) already collapses the "
            "glyph; inverse/plotting is not the primary cause."
        ),
    }
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {fig_path}")
    print(f"Wrote {summary_path}")
    for amp, row in payload["amps"].items():
        print(
            f"  amp={amp}: forward_aspect={row['forward_aspect']:.4f} "
            f"v_span_px={row['forward_v_span_px']:.2f} "
            f"inverse_ink={row['n_inverse_ink']} "
            f"inverse_aspect={row['inverse_aspect']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
