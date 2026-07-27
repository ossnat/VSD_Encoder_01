#!/usr/bin/env python3
"""Build master PDF for LOO protocol B (ResNet18 / layer3).

Includes method + held-out list, per-fold metrics (disk vs ROI),
orig|recon sanity and by-condition figures, and cross-fold summary.

Output:
  experiments/loo_encoding/loo_protocol_B_resnet18_layer3_report.pdf

Usage:
  scripts/py experiments/loo_encoding/make_loo_protocol_b_report_pdf.py
"""

from __future__ import annotations

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

REPO = project_root()
RUNS = (
    REPO
    / "experiments"
    / "loo_encoding"
    / "runs"
    / "win_0035_0043"
    / "resnet18_imagenet"
    / "layer3"
)
PROTO_DIR = RUNS / "protocol_B"
SUMMARY_CSV = RUNS / "loo_summary.csv"
HELDOUT_YAML = REPO / "experiments" / "loo_encoding" / "heldout_list.yaml"
OUT_PDF = (
    REPO / "experiments" / "loo_encoding" / "loo_protocol_B_resnet18_layer3_report.pdf"
)


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "NaN*"
    return f"{v:.{nd}f}"


def _load_heldout() -> list[str]:
    with HELDOUT_YAML.open() as f:
        data = yaml.safe_load(f)
    return list(data.get("heldout_stimulus_ids") or [])


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "protocol" in df.columns:
        df = df[df["protocol"].astype(str) == "B"].copy()
    elif "fold_id" in df.columns:
        df = df[df["fold_id"].astype(str).str.startswith("B__")].copy()
    return df.reset_index(drop=True)


def _add_cover(pdf: PdfPages, df: pd.DataFrame, heldout: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    n_ok = int((df.get("status", pd.Series(["ok"] * len(df))) != "failed").sum())
    n_fail = int(len(df) - n_ok)
    ax.text(
        0.05,
        0.92,
        "LOO Protocol B · ResNet18 / layer3 · win_0035_0043",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    body = (
        "Method\n"
        "  Protocol B: leave entire stimulus_id out of train/val; all its trials are test.\n"
        "  Inner train/val from remainder (prefer existing split labels; else 20% group holdout).\n"
        "  Encoder: RidgeCV, ResNet18 ImageNet features, layer3.\n"
        "  Window: frames [35, 43) → win_0035_0043.\n"
        "\n"
        "Dual metrics (per fold, all test trials)\n"
        "  pixel-r: per-pixel Pearson r across ALL test trials in the fold, then mean\n"
        "           inside circular eval disk vs held-out stimulus ROI.\n"
        "  spatial-r: mean over trials of whole-map spatial Pearson r inside disk / ROI.\n"
        "\n"
        "When pixel-r is undefined (NaN*)\n"
        "  Reconstructions are constant across test trials (identical stimulus CNN features\n"
        "  within stimulus_id / single condition). Prefer spatial-r in that case.\n"
        "\n"
        f"Held-out list ({len(heldout)}): {', '.join(heldout)}\n"
        f"Editable file: experiments/loo_encoding/heldout_list.yaml\n"
        f"Fold manifests: experiments/loo_encoding/runs/.../protocol_B/<fold_id>/\n"
        f"\nFolds in summary: {len(df)}  ok={n_ok}  failed={n_fail}"
    )
    ax.text(
        0.05,
        0.82,
        body,
        fontsize=9.5,
        va="top",
        family="monospace",
        transform=ax.transAxes,
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_metrics_table(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Per-fold metrics · disk vs ROI",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.05,
        0.91,
        "NaN* = pixel-r undefined (constant recon across trials). Prefer spatial-r.",
        fontsize=8,
        style="italic",
    )
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.axis("off")
    cols = [
        "fold_id",
        "n_test",
        "n_cond",
        "pixel-r disk",
        "pixel-r ROI",
        "spatial-r disk",
        "spatial-r ROI",
        "status",
    ]
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [
                str(row.get("fold_id", "")).replace("B__", ""),
                str(int(row["n_test"])) if pd.notna(row.get("n_test")) else "—",
                str(int(row["n_test_conditions"]))
                if pd.notna(row.get("n_test_conditions"))
                else "—",
                _fmt(row.get("mean_r_disk")),
                _fmt(row.get("mean_r_roi")),
                _fmt(row.get("mean_trial_spatial_r_disk")),
                _fmt(row.get("mean_trial_spatial_r_roi")),
                str(row.get("status", "ok")),
            ]
        )
    # Aggregate over successful folds with finite spatial-r
    ok = df[df.get("status", pd.Series(["ok"] * len(df))).fillna("ok") != "failed"]
    if not ok.empty:
        cell_text.append(
            [
                "MEAN (ok)",
                str(int(ok["n_test"].sum())) if "n_test" in ok else "—",
                "—",
                _fmt(ok["mean_r_disk"].mean(skipna=True)),
                _fmt(ok["mean_r_roi"].mean(skipna=True)),
                _fmt(ok["mean_trial_spatial_r_disk"].mean(skipna=True)),
                _fmt(ok["mean_trial_spatial_r_roi"].mean(skipna=True)),
                f"n={len(ok)}",
            ]
        )
    table = ax.table(
        cellText=cell_text, colLabels=cols, loc="upper center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif r == len(cell_text):
            cell.set_facecolor("#f3f3f3")
            cell.set_text_props(fontweight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_summary_bars(pdf: PdfPages, df: pd.DataFrame) -> None:
    ok = df[df.get("status", pd.Series(["ok"] * len(df))).fillna("ok") != "failed"]
    if ok.empty:
        return
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Cross-fold comparison · disk vs ROI",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.75])
    labels = [str(x).replace("B__", "").replace("_", "\n") for x in ok["fold_id"]]
    x = np.arange(len(ok))
    width = 0.18
    pixel_disk = ok.get("mean_r_disk", pd.Series([np.nan] * len(ok))).to_numpy()
    pixel_roi = ok.get("mean_r_roi", pd.Series([np.nan] * len(ok))).to_numpy()
    spat_disk = ok.get(
        "mean_trial_spatial_r_disk", pd.Series([np.nan] * len(ok))
    ).to_numpy()
    spat_roi = ok.get(
        "mean_trial_spatial_r_roi", pd.Series([np.nan] * len(ok))
    ).to_numpy()
    ax.bar(x - 1.5 * width, pixel_disk, width, label="pixel-r disk", color="#4c78a8")
    ax.bar(x - 0.5 * width, pixel_roi, width, label="pixel-r ROI", color="#f58518")
    ax.bar(x + 0.5 * width, spat_disk, width, label="spatial-r disk", color="#54a24b")
    ax.bar(x + 1.5 * width, spat_roi, width, label="spatial-r ROI", color="#e45756")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    vals = np.concatenate([pixel_disk, pixel_roi, spat_disk, spat_roi])
    finite = vals[np.isfinite(vals)]
    ymax = float(np.max(finite)) + 0.08 if finite.size else 0.55
    ax.set_ylim(-0.05, max(0.55, ymax))
    fig.text(
        0.05,
        0.02,
        "Missing pixel-r bars = NaN (identical features within held-out stim).",
        fontsize=8,
        style="italic",
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    img = Image.open(path)
    ax.imshow(img)
    pdf.savefig(fig, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _add_fold_pages(pdf: PdfPages, df: pd.DataFrame) -> int:
    n = 0
    for _, row in df.iterrows():
        fold_id = str(row["fold_id"])
        fold_dir = PROTO_DIR / fold_id
        if not fold_dir.is_dir():
            continue
        status = str(row.get("status", "ok"))
        if status == "failed":
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(
                0.05,
                0.9,
                f"{fold_id} — FAILED",
                fontsize=14,
                fontweight="bold",
                color="darkred",
                transform=ax.transAxes,
            )
            ax.text(
                0.05,
                0.8,
                str(row.get("error", "")),
                fontsize=10,
                family="monospace",
                transform=ax.transAxes,
                va="top",
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
            n += 1
            continue

        metrics_path = fold_dir / "metrics.json"
        meta = ""
        if metrics_path.is_file():
            m = json.loads(metrics_path.read_text())
            tm = m.get("test_metrics", {})
            meta = (
                f"n_test={m.get('n_test')}  n_cond={m.get('n_test_conditions')}  "
                f"pixel disk={_fmt(tm.get('mean_r_disk'))}  "
                f"pixel ROI={_fmt(tm.get('mean_r_roi'))}  "
                f"spatial disk={_fmt(tm.get('mean_trial_spatial_r_disk'))}  "
                f"spatial ROI={_fmt(tm.get('mean_trial_spatial_r_roi'))}"
            )

        sanity = fold_dir / "sanity_orig_recon_residual.png"
        if sanity.is_file():
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.set_title(
                f"{fold_id} · mean test maps (orig | recon | residual)",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            if meta:
                fig.text(0.05, 0.93, meta, fontsize=8, family="monospace")
            ax.imshow(Image.open(sanity))
            pdf.savefig(fig, bbox_inches="tight", dpi=140)
            plt.close(fig)
            n += 1

        overlay = fold_dir / "sanity_orig_recon_residual__roi_overlay.png"
        if overlay.is_file():
            _add_image_page(pdf, f"{fold_id} · ROI overlay", overlay)
            n += 1

        cond_dir = fold_dir / "by_condition"
        if cond_dir.is_dir():
            figs = sorted(cond_dir.glob("*.png"))
            # Pack up to 4 per page
            for i in range(0, len(figs), 4):
                batch = figs[i : i + 4]
                fig, axes = plt.subplots(
                    2, 2, figsize=(11, 8.5), constrained_layout=True
                )
                fig.suptitle(
                    f"{fold_id} · by condition ({i + 1}–{i + len(batch)} of {len(figs)})",
                    fontsize=12,
                    fontweight="bold",
                )
                for ax in axes.ravel():
                    ax.axis("off")
                for ax, path in zip(axes.ravel(), batch):
                    ax.imshow(Image.open(path))
                    ax.set_title(path.stem, fontsize=8)
                    ax.axis("off")
                pdf.savefig(fig, dpi=120)
                plt.close(fig)
                n += 1
    return n


def main() -> int:
    heldout = _load_heldout()
    df = _load_summary()
    if df.empty:
        raise RuntimeError(f"No protocol-B rows in {SUMMARY_CSV}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        _add_cover(pdf, df, heldout)
        _add_metrics_table(pdf, df)
        _add_summary_bars(pdf, df)
        n_fold = _add_fold_pages(pdf, df)

    print(f"Wrote {OUT_PDF.relative_to(REPO)}")
    print(f"Summary folds={len(df)}  figure pages≈{n_fold}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
