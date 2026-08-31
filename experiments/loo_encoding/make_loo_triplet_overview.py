#!/usr/bin/env python3
"""Batch fold sanity triplets (orig | recon | residual) into overview PNGs.

Prefers ``sanity_orig_recon_residual.png`` (VSD_CMAP panels) from each fold.
If that PNG is missing, renders the same orig|recon|residual row from
``fold_mean_orig.npy`` / ``fold_mean_recon.npy`` (no retrain; used with
``--skip-fold-plots``). Writes multipanel overview pages under
``<protocol_dir>/overview/``.

Pass ``--from-fold-means`` (or any custom cmap / clim) to ignore pre-rendered
PNGs and replot from fold-mean arrays. ``--shared-clim`` computes one orig/recon
1–99% range and one symmetric residual range across all folds. Use
``--name-suffix`` so a custom collage does not overwrite ``all_folds_triplets.png``.

Does **not** use ``sanity_orig_recon_residual__roi_overlay.png`` (grayscale +
lime outline) — that breaks colormap consistency with no-ROI pages. Hull
outline is omitted when collaging from pre-rendered PNGs or fold-mean arrays.

Usage:
  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir experiments/loo_encoding/runs/win_0035_0043/\\
resnet18_imagenet/layer3/protocol_B \\
    --per-page 7

  # Inclusive (every fold on one page → all_folds_triplets.png):
  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir .../protocol_B_zscore_NChull_all \\
    --inclusive --alias all_folds_triplets.png

  # Grayscale + shared clim (does not overwrite the default collage):
  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir .../protocol_B_raw_NChull_all \\
    --inclusive --from-fold-means --cmap mapgeog_gray --shared-clim \\
    --orig-recon-vmin 0.998 --orig-recon-vmax 1.003 \\
    --residual-vmin -0.0015 --residual-vmax 0.0015 \\
    --name-suffix __gray_shared --alias all_folds_triplets__gray_shared.png

  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir .../protocol_B_noise_ceiling_hull \\
    --alias raw__vbar_hbar_triangle_letterD__triplets_hull_outline.png \\
    --suptitle-note "VSD_CMAP; hull outline omitted"
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.evaluation.plotting import (
    plot_pixel_mean_maps,
    shared_orig_recon_residual_clims,
)
from src.loo.paths import list_loo_fold_dirs
from src.paths import project_root
from src.plotting_colormaps import VSD_CMAP

SANITY_NAME = "sanity_orig_recon_residual.png"
FOLD_MEAN_ORIG = "fold_mean_orig.npy"
FOLD_MEAN_RECON = "fold_mean_recon.npy"


def _fold_dirs(protocol_dir: Path) -> list[Path]:
    return list_loo_fold_dirs(protocol_dir)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _load_fold_mean_triplet(
    fold_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    orig_p = fold_dir / FOLD_MEAN_ORIG
    recon_p = fold_dir / FOLD_MEAN_RECON
    if not (orig_p.is_file() and recon_p.is_file()):
        return None
    orig = np.load(orig_p).astype(np.float32)
    recon = np.load(recon_p).astype(np.float32)
    residual = (recon - orig).astype(np.float32)
    return orig, recon, residual


def compute_protocol_shared_clims(
    fold_dirs: list[Path],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """1–99% orig+recon clim and symmetric residual clim across all folds."""
    origs: list[np.ndarray] = []
    recons: list[np.ndarray] = []
    for fold_dir in fold_dirs:
        loaded = _load_fold_mean_triplet(fold_dir)
        if loaded is None:
            continue
        origs.append(loaded[0])
        recons.append(loaded[1])
    if not origs:
        raise FileNotFoundError(
            f"No {FOLD_MEAN_ORIG}/{FOLD_MEAN_RECON} under the protocol dir"
        )
    return shared_orig_recon_residual_clims(origs, recons)


def _triplet_rgb_for_fold(
    fold_dir: Path,
    *,
    from_fold_means: bool = False,
    cmap: str | None = None,
    orig_recon_vmin: float | None = None,
    orig_recon_vmax: float | None = None,
    residual_vmin: float | None = None,
    residual_vmax: float | None = None,
) -> tuple[np.ndarray, str] | None:
    """RGB triplet row from a sanity PNG, else from fold-mean orig/recon arrays."""
    custom_plot = (
        from_fold_means
        or orig_recon_vmin is not None
        or residual_vmin is not None
        or (cmap is not None and cmap != VSD_CMAP)
    )
    sanity = fold_dir / SANITY_NAME
    if (
        not custom_plot
        and sanity.is_file()
        and sanity.stat().st_size > 0
    ):
        return _load_rgb(sanity), SANITY_NAME
    loaded = _load_fold_mean_triplet(fold_dir)
    if loaded is None:
        return None
    orig, recon, residual = loaded
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / SANITY_NAME
        plot_pixel_mean_maps(
            orig,
            recon,
            residual,
            tmp,
            title=fold_dir.name,
            vmin=orig_recon_vmin,
            vmax=orig_recon_vmax,
            residual_vmin=residual_vmin,
            residual_vmax=residual_vmax,
            cmap=cmap,
        )
        return _load_rgb(tmp), f"{FOLD_MEAN_ORIG}+{FOLD_MEAN_RECON}"


def write_overview_batches(
    protocol_dir: Path,
    *,
    out_dir: Path | None = None,
    per_page: int = 7,
    dpi: int = 140,
    alias: str | None = None,
    suptitle_note: str | None = None,
    inclusive: bool = False,
    from_fold_means: bool = False,
    cmap: str | None = None,
    orig_recon_vmin: float | None = None,
    orig_recon_vmax: float | None = None,
    residual_vmin: float | None = None,
    residual_vmax: float | None = None,
    shared_clim: bool = False,
    name_suffix: str = "",
) -> list[Path]:
    if (orig_recon_vmin is None) != (orig_recon_vmax is None):
        raise ValueError("orig_recon_vmin and orig_recon_vmax must be set together")
    if (residual_vmin is None) != (residual_vmax is None):
        raise ValueError("residual_vmin and residual_vmax must be set together")

    protocol_dir = protocol_dir.resolve()
    out_dir = (out_dir or (protocol_dir / "overview")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs = _fold_dirs(protocol_dir)
    cmap_name = cmap or VSD_CMAP
    custom_plot = (
        from_fold_means
        or shared_clim
        or orig_recon_vmin is not None
        or residual_vmin is not None
        or cmap_name != VSD_CMAP
    )

    auto_orig_recon = None
    auto_resid = None
    if shared_clim and (orig_recon_vmin is None or residual_vmin is None):
        auto_orig_recon, auto_resid = compute_protocol_shared_clims(fold_dirs)
    if orig_recon_vmin is None and auto_orig_recon is not None:
        orig_recon_vmin, orig_recon_vmax = auto_orig_recon
    if residual_vmin is None and auto_resid is not None:
        residual_vmin, residual_vmax = auto_resid

    entries: list[tuple[str, np.ndarray, str]] = []
    for fold_dir in fold_dirs:
        loaded = _triplet_rgb_for_fold(
            fold_dir,
            from_fold_means=custom_plot,
            cmap=cmap_name,
            orig_recon_vmin=orig_recon_vmin,
            orig_recon_vmax=orig_recon_vmax,
            residual_vmin=residual_vmin,
            residual_vmax=residual_vmax,
        )
        if loaded is None:
            continue
        rgb, source = loaded
        entries.append((fold_dir.name, rgb, source))

    if not entries:
        raise FileNotFoundError(
            f"No {SANITY_NAME} or fold_mean_orig/recon.npy under {protocol_dir}"
        )

    # Inclusive = one page with every fold (the "all_folds_triplets" figure).
    if inclusive:
        per_page = max(len(entries), 1)

    n_pages = max(1, math.ceil(len(entries) / per_page))
    written: list[Path] = []
    sources = {src for _, _, src in entries}
    if sources == {SANITY_NAME}:
        default_note = f"{cmap_name} sanity panels"
        hull_note = (
            "hull_outline=omitted (collage from pre-rendered sanity PNGs; "
            "no float maps / no retrain)"
        )
    elif SANITY_NAME not in sources:
        default_note = f"{cmap_name} from fold-mean orig/recon (no per-fold PNGs)"
        hull_note = (
            "hull_outline=omitted (collage from fold_mean_orig/recon.npy)"
        )
    else:
        default_note = f"{cmap_name} sanity PNGs and fold-mean arrays mixed"
        hull_note = "hull_outline=omitted (mixed PNG / fold-mean sources)"
    clim_bits = []
    if orig_recon_vmin is not None:
        clim_bits.append(
            f"orig/recon {orig_recon_vmin:g}–{orig_recon_vmax:g}"
        )
    if residual_vmin is not None:
        clim_bits.append(f"residual {residual_vmin:g}–{residual_vmax:g}")
    if clim_bits:
        default_note = f"{cmap_name}; shared " + "; ".join(clim_bits)
    note = suptitle_note or default_note
    suffix = name_suffix or ""
    for page_i in range(n_pages):
        batch = entries[page_i * per_page : (page_i + 1) * per_page]
        n = len(batch)
        fig_h = max(2.4 * n, 3.0)
        fig, axes = plt.subplots(n, 1, figsize=(11, fig_h))
        if n == 1:
            axes = [axes]
        for ax, (fold_id, rgb, _src) in zip(axes, batch):
            ax.imshow(rgb)
            ax.set_title(fold_id, fontsize=9, loc="left")
            ax.axis("off")
        protocol = protocol_dir.name
        fig.suptitle(
            f"{protocol} · orig | recon | residual  "
            f"({note}; batch {page_i + 1}/{n_pages}, n={n})",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = (
            out_dir
            / f"triplet_overview{suffix}__batch{page_i + 1:02d}_of_{n_pages:02d}.png"
        )
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    if alias and written:
        alias_path = out_dir / alias
        shutil.copy2(written[0], alias_path)
        written.append(alias_path)

    # Sidecar listing which folds went into which batch.
    index_name = f"overview_index{suffix}.txt" if suffix else "overview_index.txt"
    index_path = out_dir / index_name
    lines = [
        f"protocol_dir={protocol_dir}",
        f"n_folds={len(entries)}",
        f"sources={sorted(sources)}",
        f"cmap={cmap_name}",
        f"from_fold_means={custom_plot}",
        f"shared_clim={shared_clim}",
        f"orig_recon_vmin={orig_recon_vmin}",
        f"orig_recon_vmax={orig_recon_vmax}",
        f"residual_vmin={residual_vmin}",
        f"residual_vmax={residual_vmax}",
        hull_note,
        "",
    ]
    for page_i in range(n_pages):
        batch = entries[page_i * per_page : (page_i + 1) * per_page]
        lines.append(f"batch {page_i + 1}/{n_pages}:")
        for fold_id, _rgb, src in batch:
            lines.append(f"  {fold_id} <- {src}")
        lines.append("")
    if alias:
        lines.append(f"alias={alias}")
        lines.append("")
    index_path.write_text("\n".join(lines))
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol-dir",
        type=Path,
        required=True,
        help="Path to protocol_A / protocol_B / protocol_C run directory",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Overview output dir (default: <protocol-dir>/overview)",
    )
    p.add_argument("--per-page", type=int, default=7)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Also copy batch-01 overview to this filename under out-dir",
    )
    p.add_argument(
        "--inclusive",
        action="store_true",
        help=(
            "Put every fold on a single page (the inclusive all-triplets "
            "figure). Overrides --per-page."
        ),
    )
    p.add_argument(
        "--suptitle-note",
        type=str,
        default=None,
        help="Extra note in figure suptitle (default: VSD_CMAP sanity panels)",
    )
    p.add_argument(
        "--from-fold-means",
        action="store_true",
        help=(
            "Ignore pre-rendered sanity PNGs and replot each row from "
            "fold_mean_orig/recon.npy (needed for shared clim / cmap)."
        ),
    )
    p.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Colormap for fold-mean replots (default: mapgeog). Use mapgeog_gray.",
    )
    p.add_argument(
        "--shared-clim",
        action="store_true",
        help=(
            "Compute orig/recon 1–99%% and a symmetric residual clim across "
            "all fold-mean arrays. Explicit --orig-recon-* / --residual-* win."
        ),
    )
    p.add_argument(
        "--orig-recon-vmin",
        type=float,
        default=None,
        help="Shared orig+recon vmin (must pair with --orig-recon-vmax).",
    )
    p.add_argument(
        "--orig-recon-vmax",
        type=float,
        default=None,
        help="Shared orig+recon vmax.",
    )
    p.add_argument(
        "--residual-vmin",
        type=float,
        default=None,
        help="Shared residual vmin (must pair with --residual-vmax).",
    )
    p.add_argument(
        "--residual-vmax",
        type=float,
        default=None,
        help="Shared residual vmax.",
    )
    p.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help=(
            "Inserted into batch/index filenames so custom collages do not "
            "overwrite the default (e.g. __gray_shared)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.orig_recon_vmin is None) != (args.orig_recon_vmax is None):
        raise SystemExit(
            "--orig-recon-vmin and --orig-recon-vmax must be set together"
        )
    if (args.residual_vmin is None) != (args.residual_vmax is None):
        raise SystemExit(
            "--residual-vmin and --residual-vmax must be set together"
        )
    repo = project_root()
    protocol_dir = (
        args.protocol_dir
        if args.protocol_dir.is_absolute()
        else repo / args.protocol_dir
    )
    out_dir = None
    if args.out_dir is not None:
        out_dir = (
            args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
        )
    paths = write_overview_batches(
        protocol_dir,
        out_dir=out_dir,
        per_page=args.per_page,
        dpi=args.dpi,
        alias=args.alias,
        suptitle_note=args.suptitle_note,
        inclusive=bool(args.inclusive),
        from_fold_means=bool(args.from_fold_means),
        cmap=args.cmap,
        orig_recon_vmin=args.orig_recon_vmin,
        orig_recon_vmax=args.orig_recon_vmax,
        residual_vmin=args.residual_vmin,
        residual_vmax=args.residual_vmax,
        shared_clim=bool(args.shared_clim),
        name_suffix=args.name_suffix or "",
    )
    for path in paths:
        print(path.relative_to(repo) if path.is_relative_to(repo) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
