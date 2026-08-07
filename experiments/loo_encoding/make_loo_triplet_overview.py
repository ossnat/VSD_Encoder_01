#!/usr/bin/env python3
"""Batch fold sanity triplets (orig | recon | residual) into overview PNGs.

Reads ``sanity_orig_recon_residual.png`` (VSD_CMAP panels) from each fold under
a protocol directory and writes multipanel overview pages (~6–10 stimuli each)
under ``<protocol_dir>/overview/``.

Does **not** use ``sanity_orig_recon_residual__roi_overlay.png`` (grayscale +
lime outline) — that breaks colormap consistency with no-ROI pages. Hull
outline is omitted when collaging from the pre-rendered sanity PNGs alone
(no float maps available without retrain).

Usage:
  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir experiments/loo_encoding/runs/win_0035_0043/\\
resnet18_imagenet/layer3/protocol_B \\
    --per-page 7

  scripts/py experiments/loo_encoding/make_loo_triplet_overview.py \\
    --protocol-dir .../protocol_B_noise_ceiling_hull \\
    --alias raw__vbar_hbar_triangle_letterD__triplets_hull_outline.png \\
    --suptitle-note "VSD_CMAP; hull outline omitted"
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.paths import project_root

SANITY_NAME = "sanity_orig_recon_residual.png"


def _fold_dirs(protocol_dir: Path) -> list[Path]:
    dirs = [
        p
        for p in sorted(protocol_dir.iterdir())
        if p.is_dir() and p.name.startswith(("A__", "B__"))
    ]
    return dirs


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def write_overview_batches(
    protocol_dir: Path,
    *,
    out_dir: Path | None = None,
    per_page: int = 7,
    dpi: int = 140,
    alias: str | None = None,
    suptitle_note: str | None = None,
) -> list[Path]:
    protocol_dir = protocol_dir.resolve()
    out_dir = (out_dir or (protocol_dir / "overview")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, Path]] = []
    for fold_dir in _fold_dirs(protocol_dir):
        sanity = fold_dir / SANITY_NAME
        if sanity.is_file() and sanity.stat().st_size > 0:
            entries.append((fold_dir.name, sanity))

    if not entries:
        raise FileNotFoundError(f"No {SANITY_NAME} under {protocol_dir}")

    n_pages = max(1, math.ceil(len(entries) / per_page))
    written: list[Path] = []
    note = suptitle_note or "VSD_CMAP sanity panels"
    for page_i in range(n_pages):
        batch = entries[page_i * per_page : (page_i + 1) * per_page]
        n = len(batch)
        fig_h = max(2.4 * n, 3.0)
        fig, axes = plt.subplots(n, 1, figsize=(11, fig_h))
        if n == 1:
            axes = [axes]
        for ax, (fold_id, path) in zip(axes, batch):
            ax.imshow(_load_rgb(path))
            ax.set_title(fold_id, fontsize=9, loc="left")
            ax.axis("off")
        protocol = protocol_dir.name
        fig.suptitle(
            f"{protocol} · orig | recon | residual  "
            f"({note}; batch {page_i + 1}/{n_pages}, n={n})",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = out_dir / f"triplet_overview__batch{page_i + 1:02d}_of_{n_pages:02d}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    if alias and written:
        alias_path = out_dir / alias
        if len(written) == 1:
            shutil.copy2(written[0], alias_path)
            written.append(alias_path)
        else:
            # Multi-page: alias only the first batch; index notes all pages.
            shutil.copy2(written[0], alias_path)
            written.append(alias_path)

    # Sidecar listing which folds went into which batch.
    index_path = out_dir / "overview_index.txt"
    lines = [
        f"protocol_dir={protocol_dir}",
        f"n_folds={len(entries)}",
        f"source={SANITY_NAME} (VSD_CMAP)",
        "hull_outline=omitted (collage from pre-rendered sanity PNGs; "
        "no float maps / no retrain)",
        "",
    ]
    for page_i in range(n_pages):
        batch = entries[page_i * per_page : (page_i + 1) * per_page]
        lines.append(f"batch {page_i + 1}/{n_pages}:")
        for fold_id, path in batch:
            lines.append(f"  {fold_id} <- {path.name}")
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
        help="Path to protocol_A or protocol_B run directory",
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
        "--suptitle-note",
        type=str,
        default=None,
        help="Extra note in figure suptitle (default: VSD_CMAP sanity panels)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    )
    for path in paths:
        print(path.relative_to(repo) if path.is_relative_to(repo) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
