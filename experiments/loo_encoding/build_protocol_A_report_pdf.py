#!/usr/bin/env python3
"""Build Protocol A flat-run report PDF (table + maps + overviews + params).

Expected run-root layout::

  experiments/loo_encoding/runs/YYYY-MM-DD_35-46_<model>_<layer>/
    protocol_A_{zscore,raw}_NChull_{clean,all}/
    noise_corr_odd_even/
    pooled_fold_pixel_r__*.png
    corr_summary_encoding.csv
    pipeline_manifest.yaml
    report.pdf   ← this script

Usage::

  scripts/py experiments/loo_encoding/build_protocol_A_report_pdf.py \\
    --run-root experiments/loo_encoding/runs/2026-08-07_35-46_resnet18_l3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.paths import project_root


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{nd}f}"


def _add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        title,
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(
        0.05,
        0.88,
        body,
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
        va="top",
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_image_page(
    pdf: PdfPages,
    path: Path,
    title: str,
    *,
    max_h: float = 7.5,
) -> None:
    if not path.is_file():
        _add_text_page(pdf, title, f"(missing figure)\n{path}")
        return
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    h, w = arr.shape[:2]
    aspect = w / max(h, 1)
    fig_w = 11.0
    fig_h = min(max_h, fig_w / aspect + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(arr)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _corr_table(
    run_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Return 2×3 mean-r table (zscore/raw × clean/all/noise) and a note."""
    enc_csv = run_root / "corr_summary_encoding.csv"
    noise_summary = run_root / "noise_corr_odd_even" / "summary.json"
    rows: dict[str, dict[str, float | None]] = {
        "zscore": {
            "clean encoding": None,
            "all-data encoding": None,
            "odd-even noise": None,
        },
        "raw": {
            "clean encoding": None,
            "all-data encoding": None,
            "odd-even noise": None,
        },
    }
    note_parts: list[str] = []

    if enc_csv.is_file():
        enc = pd.read_csv(enc_csv)
        for _, r in enc.iterrows():
            w = str(r["window"])
            if w in rows:
                rows[w]["clean encoding"] = (
                    float(r["clean_encoding_mean_r_hull"])
                    if pd.notna(r.get("clean_encoding_mean_r_hull"))
                    else None
                )
                rows[w]["all-data encoding"] = (
                    float(r["all_data_encoding_mean_r_hull"])
                    if pd.notna(r.get("all_data_encoding_mean_r_hull"))
                    else None
                )
    else:
        note_parts.append("missing corr_summary_encoding.csv")

    if noise_summary.is_file():
        noise = json.loads(noise_summary.read_text())
        rows["zscore"]["odd-even noise"] = float(
            noise.get("zscore_mean_r_hull")
        )
        rows["raw"]["odd-even noise"] = float(noise.get("raw_mean_r_hull"))
    else:
        note_parts.append("missing noise_corr_odd_even/summary.json")

    df = pd.DataFrame(rows).T
    df.index.name = "window"
    return df, "; ".join(note_parts) if note_parts else ""


def _add_corr_table_page(
    pdf: PdfPages, table: pd.DataFrame, note: str
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(
        "Mean per-pixel r inside NC hull (pooled fold-level)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    cell = [
        [_fmt(table.loc[w, c]) for c in table.columns]
        for w in table.index
    ]
    tbl = ax.table(
        cellText=cell,
        rowLabels=list(table.index),
        colLabels=list(table.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)
    if note:
        ax.text(
            0.5,
            0.15,
            note,
            ha="center",
            fontsize=8,
            style="italic",
            transform=ax.transAxes,
        )
    ax.text(
        0.5,
        0.08,
        "Encoding: fold-mean orig vs recon across N Protocol A folds.\n"
        "Odd-even noise: odd/even trial means per fold, then same pooled pixel-r.",
        ha="center",
        fontsize=9,
        transform=ax.transAxes,
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def build_report(
    run_root: Path,
    *,
    repo: Path,
    out_pdf: Path | None = None,
    max_overview_pages_per_leaf: int = 3,
) -> Path:
    run_root = run_root.resolve()
    out_pdf = (out_pdf or (run_root / "report.pdf")).resolve()
    manifest = _load_yaml(run_root / "pipeline_manifest.yaml")
    table, note = _corr_table(run_root)

    title = (
        f"Protocol A LOO · {manifest.get('model_slug', '?')} / "
        f"{manifest.get('feature_layer', '?')} · "
        f"frames [{manifest.get('start_frame', '?')}, "
        f"{manifest.get('end_frame', '?')})"
    )
    cover_lines = [
        f"Run root: {run_root.relative_to(repo) if run_root.is_relative_to(repo) else run_root}",
        f"Model: {manifest.get('model')}  layer={manifest.get('feature_layer')}",
        f"Loss ROI: {manifest.get('loss_roi')}",
        f"Held-out list: {manifest.get('heldout_list')}",
        f"Run date: {manifest.get('run_date')}",
        "",
        "Leaves:",
    ]
    for leaf in manifest.get("leaves") or []:
        cover_lines.append(
            f"  - {leaf.get('leaf_name')}  n_folds={leaf.get('n_folds', '?')}"
        )
    cover_lines += [
        "",
        f"Noise corr: {manifest.get('noise_corr_dir', 'noise_corr_odd_even/')}",
        "Method: Protocol A = leave one (date, condition) out; Ridge on NC hull.",
        "Clean = trial_cleanliness keep=good; All = no cleanliness filter.",
    ]

    with PdfPages(out_pdf) as pdf:
        _add_text_page(pdf, title, "\n".join(cover_lines))
        _add_corr_table_page(pdf, table, note)

        # Encoding maps
        for name, label in [
            (
                "pooled_fold_pixel_r__2x2_clean_all_zscore_raw.png",
                "Encoding pooled pixel-r · 2×2 (clean/all × zscore/raw)",
            ),
            (
                "pooled_fold_pixel_r__protocol_A_clean_zscore_vs_raw.png",
                "Encoding · clean · zscore vs raw",
            ),
            (
                "pooled_fold_pixel_r__protocol_A_all_zscore_vs_raw.png",
                "Encoding · all-data · zscore vs raw",
            ),
        ]:
            _add_image_page(pdf, run_root / name, label)

        # Noise maps
        noise_dir = run_root / "noise_corr_odd_even"
        for name, label in [
            (
                "r_map_pooled_folds__zscore_vs_raw.png",
                "Odd-even noise corr · zscore vs raw",
            ),
            ("r_map_pooled_folds__zscore.png", "Odd-even noise · zscore"),
            ("r_map_pooled_folds__raw.png", "Odd-even noise · raw"),
        ]:
            _add_image_page(pdf, noise_dir / name, label)

        # Triplet overviews (batched; cap pages per leaf)
        for leaf in manifest.get("leaves") or []:
            leaf_dir = repo / leaf["leaf_dir"]
            overview = leaf_dir / "overview"
            batches = sorted(overview.glob("triplet_overview__batch*.png"))
            if not batches:
                alias = overview / "all_folds_triplets.png"
                if alias.is_file():
                    batches = [alias]
            for i, path in enumerate(batches[:max_overview_pages_per_leaf]):
                _add_image_page(
                    pdf,
                    path,
                    f"Triplets · {leaf.get('leaf_name')} · page {i + 1}",
                )
            if len(batches) > max_overview_pages_per_leaf:
                _add_text_page(
                    pdf,
                    f"Triplets · {leaf.get('leaf_name')} (truncated)",
                    f"Showing {max_overview_pages_per_leaf}/{len(batches)} "
                    f"overview batches.\nFull set: {overview}",
                )

        # Params dump
        params_body_parts: list[str] = []
        for leaf in manifest.get("leaves") or []:
            params_path = repo / leaf["leaf_dir"] / "params.yaml"
            if params_path.is_file():
                params_body_parts.append(
                    f"=== {leaf.get('leaf_name')} ===\n"
                    + params_path.read_text()[:3500]
                )
        noise_params = noise_dir / "params.yaml"
        if noise_params.is_file():
            params_body_parts.append(
                "=== noise_corr_odd_even ===\n" + noise_params.read_text()[:3500]
            )
        if params_body_parts:
            # Split across pages if huge
            chunk = "\n\n".join(params_body_parts)
            for i in range(0, len(chunk), 4500):
                _add_text_page(
                    pdf,
                    "Run parameters",
                    chunk[i : i + 4500],
                )

        # Persist table alongside PDF
        table_out = run_root / "corr_summary_table.csv"
        table.to_csv(table_out)

    print(f"Wrote {out_pdf}")
    return out_pdf


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-pdf", type=Path, default=None)
    p.add_argument("--max-overview-pages-per-leaf", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    run_root = (
        args.run_root if args.run_root.is_absolute() else repo / args.run_root
    )
    out_pdf = None
    if args.out_pdf is not None:
        out_pdf = (
            args.out_pdf if args.out_pdf.is_absolute() else repo / args.out_pdf
        )
    build_report(
        run_root,
        repo=repo,
        out_pdf=out_pdf,
        max_overview_pages_per_leaf=args.max_overview_pages_per_leaf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
