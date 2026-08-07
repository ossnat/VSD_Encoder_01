#!/usr/bin/env python3
"""Build a multi-page PDF overview of mean-frame sanity maps + ROI review figures.

Also adds:
  - first-pass disk-mask vs stimulus-ROI metrics (resnet18 / layer3)
  - stimulus taxonomy page (held-out flags)
  - optional LOO smoke-fold metrics when runs/ exist

Output:
  experiments/loo_encoding/sanity_and_roi_overview.pdf

Usage (from repo root):
  scripts/py experiments/loo_encoding/make_sanity_and_roi_overview_pdf.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Headless-friendly (cluster / sandbox / no display).
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.DL_features.schema import model_slug
from src.encoding.ridge import attach_feature_paths
from src.encoding.schema import encoding_pairs_manifest_path, ridge_output_dir
from src.evaluation.dual_metrics import (
    mean_trial_spatial_r,
    pixel_r_means_from_corr_map,
    roi_pixel_r_from_global_corr,
)
from src.evaluation.mask import mask_from_eval_cfg
from src.evaluation.pixel_correlation import (
    load_reconstructed_maps,
    load_trial_mean_maps,
    pixel_correlation_across_trials,
)
from src.paths import project_root, resolve_data_path
from src.stimuli.identity import attach_stimulus_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
SANITY_DIR = REPO_ROOT / "experiments" / "mean_frame_maps_sanity" / "plots"
ROI_DIR = REPO_ROOT / "experiments" / "loo_encoding" / "roi_review" / "figures"
ROI_MASK_DIR = REPO_ROOT / "experiments" / "loo_encoding" / "rois"
TAXONOMY_CSV = REPO_ROOT / "experiments" / "loo_encoding" / "stimulus_taxonomy.csv"
TAXONOMY_YAML = REPO_ROOT / "experiments" / "loo_encoding" / "stimulus_taxonomy.yaml"
LOO_SUMMARY = (
    REPO_ROOT
    / "experiments"
    / "loo_encoding"
    / "runs"
    / "win_0035_0043"
    / "resnet18_imagenet"
    / "layer3"
    / "loo_summary.csv"
)
OUT_PDF = REPO_ROOT / "experiments" / "loo_encoding" / "sanity_and_roi_overview.pdf"
OUT_METRICS_JSON = (
    REPO_ROOT / "experiments" / "loo_encoding" / "roi_vs_disk_metrics.json"
)

# Seed=17 order from mean_frame_maps_sanity/index.txt; skip the first entry.
SANITY_PAGES = [
    ("Sanity: 201118c · condAN4", SANITY_DIR / "201118c__condAN4.png"),
    ("Sanity: 240718a · condAN1", SANITY_DIR / "240718a__condAN1.png"),
    ("Sanity: 201118a · condAN1", SANITY_DIR / "201118a__condAN1.png"),
    ("Sanity: 270618b · condAN5", SANITY_DIR / "270618b__condAN5.png"),
    ("Sanity: 201118d · condAN7", SANITY_DIR / "201118d__condAN7.png"),
]

ROI_STIMULI = [
    "black_point_0.1",
    "black_bar_vertical_0.3",
    "black_circle_contour_0.3",
    "letter_A_white_1",
    "white_filled_circle_0.8",
    "white_point_0.1",
]

# Spatial-r still needs per-stimulus trials; prefer test when available.
SPLIT_PRIORITY = ("test", "val", "train")
# Pixel-r for the baseline dual page is always the full test split (stage-04).
PIXEL_R_SPLIT = "test"

MODEL_CFG = REPO_ROOT / "configs" / "models" / "resnet18.yaml"
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"
# Prefer win_0035_0043 when a ridge model exists; else fall back to 0042.
WINDOW_CFG_PREFERRED = REPO_ROOT / "configs" / "windows" / "evoked_35_43.yaml"
WINDOW_CFG_FALLBACK = REPO_ROOT / "configs" / "windows" / "evoked_35_42.yaml"
RIDGE_CFG = REPO_ROOT / "configs" / "ridge" / "default.yaml"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing figure: {path}")
    return path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _select_window_cfg(repo: Path) -> Path:
    """Pick window config that has an existing ridge model for metrics page."""
    for window_path in (WINDOW_CFG_PREFERRED, WINDOW_CFG_FALLBACK):
        cfg = _load_yaml(DEFAULT_CFG)
        cfg.update(_load_yaml(window_path))
        model_cfg = _load_yaml(MODEL_CFG)
        window_id = cfg.get("window_id") or (
            f"win_{int(cfg['start_frame']):04d}_{int(cfg['end_frame']):04d}"
        )
        model_path = (
            ridge_output_dir(
                resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
                cfg["monkey"],
                window_id,
                model_slug(model_cfg),
                str(model_cfg.get("feature_layer", "layer3")),
            )
            / "model.joblib"
        )
        if model_path.is_file():
            return window_path
    return WINDOW_CFG_FALLBACK


def _fmt_size(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "?"
    return f"{float(value):g}"


def stimulus_id_from_row(row: pd.Series) -> str | None:
    """Stable stimulus identity (same convention as scripts/16)."""
    shape = str(row.get("shape_type", "") or "")
    if not shape or shape == "blank" or bool(row.get("is_blank", False)):
        return None
    color = str(row.get("color", "unknown") or "unknown")
    size = _fmt_size(row.get("size_deg"))
    if shape == "letter":
        letter = row.get("letter") if "letter" in row.index else None
        if letter is None or (isinstance(letter, float) and np.isnan(letter)):
            text = str(row.get("stimulus_text", ""))
            parts = text.strip().split()
            letter = parts[-1] if parts else "?"
        return f"letter_{str(letter).upper()}_{color}_{size}"
    return f"{color}_{shape}_{size}"


def _choose_split(split_counts: dict[str, int]) -> str | None:
    for split in SPLIT_PRIORITY:
        if int(split_counts.get(split, 0)) > 0:
            return split
    return None


def compute_roi_vs_disk_metrics(
    *,
    repo: Path | None = None,
    window_cfg: Path | None = None,
) -> dict[str, Any]:
    """
    First-pass disk vs ROI metrics for the 6 overview stimuli.

    Pixel-r (stage-04 definition):
      - Compute per-pixel Pearson r across ALL trials in the full test split
        (all conditions), then average that map inside the disk and each ROI.
      - Do NOT compute pixel-r within a single stimulus/condition subset
        (identical reconstructions → NaN).

    Spatial-r (per stimulus):
      - Mean over that stimulus's trials of spatial Pearson r inside disk/ROI,
        using preferred split (test > val > train) for trial selection only.
    """
    repo = repo or project_root()
    window_path = window_cfg or _select_window_cfg(repo)
    cfg = _load_yaml(DEFAULT_CFG)
    cfg.update(_load_yaml(window_path))
    cfg["ridge"] = _load_yaml(RIDGE_CFG)
    model_cfg = _load_yaml(MODEL_CFG)

    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or f"win_{start_frame:04d}_{end_frame:04d}"
    avg_method = cfg.get("avg_method", "mean")
    monkey = cfg["monkey"]
    feature_layer = str(model_cfg.get("feature_layer", "layer3"))
    model_name = model_slug(model_cfg)
    eval_cfg = cfg["ridge"].get("evaluation", {})
    disk_mask = mask_from_eval_cfg(eval_cfg, spatial_size)
    if disk_mask is None:
        raise RuntimeError("Expected circular eval mask in ridge evaluation config")

    pairs_path = encoding_pairs_manifest_path(
        resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo),
        monkey,
        window_id,
    )
    model_path = (
        ridge_output_dir(
            resolve_data_path(cfg["paths"]["ridge_encode_root"], repo),
            monkey,
            window_id,
            model_name,
            feature_layer,
        )
        / "model.joblib"
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Encoding pairs manifest not found: {pairs_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Ridge model not found: {model_path}")

    pairs = pd.read_parquet(pairs_path)
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)

    features_root = resolve_data_path(cfg["paths"]["dl_features_stimuli_root"], repo)
    pairs = attach_feature_paths(
        pairs,
        features_root=features_root,
        monkey=monkey,
        model_slug=model_name,
        feature_layer=feature_layer,
        repo=repo,
    )

    payload = joblib.load(model_path)
    result = payload["result"]
    result.spatial_size = spatial_size

    # --- Full test split for pixel-r (same as stage-04) ---
    test_df = pairs[pairs["split"] == PIXEL_R_SPLIT].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError(f"No trials with split={PIXEL_R_SPLIT!r}")
    print(
        f"Loading full {PIXEL_R_SPLIT} split: {len(test_df)} trials for pixel-r "
        f"({model_name}/{feature_layer}, {window_id})..."
    )
    test_originals = load_trial_mean_maps(
        test_df,
        repo=repo,
        spatial_size=spatial_size,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_method=avg_method,
    )
    test_recons = load_reconstructed_maps(
        test_df,
        result=result,
        repo=repo,
        spatial_size=spatial_size,
    )
    global_corr = pixel_correlation_across_trials(test_originals, test_recons)
    global_pixel = pixel_r_means_from_corr_map(
        global_corr, disk_mask=disk_mask, roi_mask=None
    )

    roi_masks: dict[str, np.ndarray] = {}
    for sid in ROI_STIMULI:
        mask_path = ROI_MASK_DIR / f"{sid}__mask.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing ROI mask: {mask_path}")
        roi = np.load(mask_path).astype(bool)
        if roi.shape != spatial_size:
            raise ValueError(
                f"ROI mask shape {roi.shape} != spatial_size {spatial_size}"
            )
        roi_masks[sid] = roi

    roi_pixel_df = roi_pixel_r_from_global_corr(
        global_corr, disk_mask=disk_mask, roi_masks=roi_masks
    )
    roi_pixel_by_sid = {
        str(r["stimulus_id"]): r for r in roi_pixel_df.to_dict(orient="records")
    }

    # --- Per-stimulus spatial-r on preferred split trials ---
    spatial_frames: list[pd.DataFrame] = []
    plan: list[dict[str, Any]] = []
    for sid in ROI_STIMULI:
        g = pairs[pairs["stimulus_id"] == sid]
        counts = g["split"].value_counts().to_dict()
        split = _choose_split(counts)
        if split is None:
            plan.append(
                {
                    "stimulus_id": sid,
                    "spatial_split": None,
                    "n_trials_spatial": 0,
                    "note": "no trials found",
                }
            )
            continue
        sub = g[g["split"] == split].reset_index(drop=True)
        plan.append(
            {
                "stimulus_id": sid,
                "spatial_split": split,
                "n_trials_spatial": int(len(sub)),
                "n_conditions_spatial": int(
                    sub.groupby(["date", "condition"]).ngroups
                ),
                "split_counts": {k: int(v) for k, v in counts.items()},
            }
        )
        spatial_frames.append(sub.assign(_sid=sid))

    spatial_by_sid: dict[str, dict[str, float]] = {}
    if spatial_frames:
        spatial_df = pd.concat(spatial_frames, ignore_index=True)
        print(
            f"Loading {len(spatial_df)} stimulus trials for spatial-r "
            f"(split policy test>val>train)..."
        )
        spat_orig = load_trial_mean_maps(
            spatial_df,
            repo=repo,
            spatial_size=spatial_size,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_method=avg_method,
        )
        spat_recon = load_reconstructed_maps(
            spatial_df,
            result=result,
            repo=repo,
            spatial_size=spatial_size,
        )
        for item in plan:
            sid = item["stimulus_id"]
            if item.get("n_trials_spatial", 0) <= 0:
                continue
            idx = np.flatnonzero(spatial_df["_sid"].to_numpy() == sid)
            o = spat_orig[idx]
            r = spat_recon[idx]
            spatial_by_sid[sid] = {
                "mean_trial_spatial_r_disk": mean_trial_spatial_r(
                    o, r, disk_mask
                ),
                "mean_trial_spatial_r_roi": mean_trial_spatial_r(
                    o, r, roi_masks[sid]
                ),
            }

    rows: list[dict[str, Any]] = []
    for item in plan:
        sid = item["stimulus_id"]
        pix = roi_pixel_by_sid.get(sid, {})
        spat = spatial_by_sid.get(sid, {})
        rows.append(
            {
                "stimulus_id": sid,
                "split": PIXEL_R_SPLIT,  # pixel-r source
                "spatial_split": item.get("spatial_split"),
                "n_trials": int(len(test_df)),  # pixel-r n
                "n_trials_spatial": item.get("n_trials_spatial", 0),
                "n_conditions": int(
                    test_df.groupby(["date", "condition"]).ngroups
                ),
                "n_conditions_spatial": item.get("n_conditions_spatial", 0),
                "split_counts": item.get("split_counts", {}),
                "n_roi_pixels": int(roi_masks[sid].sum()),
                "mean_pixel_r_disk": pix.get(
                    "mean_pixel_r_disk", float("nan")
                ),
                "mean_pixel_r_roi": pix.get("mean_pixel_r_roi", float("nan")),
                "mean_trial_spatial_r_disk": spat.get(
                    "mean_trial_spatial_r_disk", float("nan")
                ),
                "mean_trial_spatial_r_roi": spat.get(
                    "mean_trial_spatial_r_roi", float("nan")
                ),
            }
        )

    finite_pixel = [
        r for r in rows if np.isfinite(r.get("mean_pixel_r_roi", np.nan))
    ]
    finite_spatial = [
        r
        for r in rows
        if np.isfinite(r.get("mean_trial_spatial_r_disk", np.nan))
    ]
    aggregate = {
        "n_stimuli_pixel_r": len(finite_pixel),
        "n_stimuli_spatial_r": len(finite_spatial),
        "n_stimuli": len(rows),
        "n_test_trials_pixel_r": int(len(test_df)),
        "n_test_conditions_pixel_r": int(
            test_df.groupby(["date", "condition"]).ngroups
        ),
        "mean_pixel_r_disk": float(global_pixel.get("mean_pixel_r_disk", float("nan"))),
        "mean_pixel_r_roi": float(
            np.nanmean([r["mean_pixel_r_roi"] for r in finite_pixel])
        )
        if finite_pixel
        else float("nan"),
        "mean_trial_spatial_r_disk": float(
            np.nanmean([r["mean_trial_spatial_r_disk"] for r in finite_spatial])
        )
        if finite_spatial
        else float("nan"),
        "mean_trial_spatial_r_roi": float(
            np.nanmean([r["mean_trial_spatial_r_roi"] for r in finite_spatial])
        )
        if finite_spatial
        else float("nan"),
    }

    return {
        "label": "First-pass comparison (not LOO)",
        "model_slug": model_name,
        "feature_layer": feature_layer,
        "window_id": window_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "monkey": monkey,
        "disk_mask": {
            "type": str(eval_cfg.get("mask_type", "circle")),
            "radius": int(eval_cfg.get("mask_radius", 50)),
            "n_pixels": int(disk_mask.sum()),
        },
        "split_policy": (
            f"pixel-r: full {PIXEL_R_SPLIT} split (all conditions); "
            "spatial-r per stimulus: test > val > train"
        ),
        "model_path": str(model_path),
        "encoding_pairs_manifest": str(pairs_path),
        "stimuli": rows,
        "aggregate": aggregate,
        "notes": [
            "mean_pixel_r = mean of per-pixel Pearson r across ALL test-split "
            "trials (stage-04), then averaged inside disk / each ROI mask.",
            "Do not compute pixel-r within a single condition or identical-"
            "feature stimulus subset (recon constant → NaN).",
            "mean_trial_spatial_r = mean over that stimulus's trials of spatial "
            "Pearson r inside the mask (preferred dual metric when pixel-r "
            "is undefined in LOO identical-feature folds).",
            "Disk pixel-r is identical across stimulus rows (one global map). "
            "ROI pixel-r is that same map averaged inside each ROI.",
            "Spatial-r rows that fall back to train are optimistic relative "
            "to held-out eval.",
        ],
    }


def _add_cover(pdf: PdfPages, *, has_loo: bool) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.88,
        "Sanity & ROI Overview",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.78,
        "Contents",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )
    body = (
        "1. Mean-frame sanity maps (seed=17; skipped 290518a__condAN3)\n"
        "     • 201118c__condAN4\n"
        "     • 240718a__condAN1\n"
        "     • 201118a__condAN1\n"
        "     • 270618b__condAN5\n"
        "     • 201118d__condAN7\n"
        "\n"
        "2. ROI review mean-map figures (one page, 2×3 grid)\n"
        "     • black_point_0.1 · black_bar_vertical_0.3 · black_circle_contour_0.3\n"
        "     • letter_A_white_1 · white_filled_circle_0.8 · white_point_0.1\n"
        "\n"
        "3. Disk-mask vs ROI-only metrics (first-pass, non-LOO ridge)\n"
        "\n"
        "4. Stimulus taxonomy (held-out flags + ROI status)\n"
    )
    if has_loo:
        body += (
            "\n"
            "5. LOO smoke folds (protocol B · win_0035_0043 · ResNet18/layer3)\n"
        )
    ax.text(
        0.12,
        0.70,
        body,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        transform=ax.transAxes,
        linespacing=1.45,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, title: str, path: Path) -> None:
    img = Image.open(_require(path))
    w, h = img.size
    page_w = 11.0
    page_h = max(6.0, min(11.0, page_w * h / w + 0.7))
    fig, ax = plt.subplots(figsize=(page_w, page_h))
    ax.imshow(img)
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)
    img.close()


def _add_roi_grid_page(pdf: PdfPages) -> None:
    """Six ROI mean-map figures in a single 2-column × 3-row figure."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 12.5))
    fig.suptitle(
        "ROI review · mean maps with accepted boxes (2×3)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    for ax, sid in zip(axes.ravel(), ROI_STIMULI):
        path = ROI_DIR / f"{sid}__mean_map_roi.png"
        img = Image.open(_require(path))
        ax.imshow(img)
        ax.set_title(sid, fontsize=10, pad=4)
        ax.axis("off")
        img.close()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.3f}"


def _add_metrics_page(pdf: PdfPages, metrics: dict[str, Any]) -> None:
    stimuli = metrics["stimuli"]
    agg = metrics["aggregate"]

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Disk mask vs stimulus ROI — first-pass metrics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    meta = (
        f"Model: {metrics['model_slug']} / {metrics['feature_layer']}    "
        f"Window: {metrics['window_id']} (frames "
        f"[{metrics['start_frame']}, {metrics['end_frame']}))    "
        f"Monkey: {metrics['monkey']}\n"
        f"Split policy: {metrics['split_policy']}    "
        f"Disk: {metrics['disk_mask']['type']} r="
        f"{metrics['disk_mask']['radius']} "
        f"({metrics['disk_mask']['n_pixels']} px)\n"
        "Honest label: not LOO — standard RidgeCV encoder on existing split; "
        "ROI-only summary newly added for this overview."
    )
    fig.text(0.05, 0.90, meta, fontsize=8.5, va="top", family="monospace")

    ax_table = fig.add_axes([0.05, 0.42, 0.90, 0.42])
    ax_table.axis("off")
    col_labels = [
        "stimulus_id",
        "spat.split",
        "n_spat",
        "pixel-r disk",
        "pixel-r ROI",
        "spatial-r disk",
        "spatial-r ROI",
    ]
    cell_text = []
    for row in stimuli:
        cell_text.append(
            [
                row["stimulus_id"],
                str(row.get("spatial_split") or row.get("split") or "—"),
                str(row.get("n_trials_spatial", row.get("n_trials", 0))),
                _fmt(row.get("mean_pixel_r_disk", float("nan"))),
                _fmt(row.get("mean_pixel_r_roi", float("nan"))),
                _fmt(row.get("mean_trial_spatial_r_disk", float("nan"))),
                _fmt(row.get("mean_trial_spatial_r_roi", float("nan"))),
            ]
        )
    cell_text.append(
        [
            "AGGREGATE",
            f"test n={agg.get('n_test_trials_pixel_r', '?')}",
            str(agg["n_stimuli"]),
            _fmt(agg["mean_pixel_r_disk"]),
            _fmt(agg["mean_pixel_r_roi"]),
            _fmt(agg["mean_trial_spatial_r_disk"]),
            _fmt(agg["mean_trial_spatial_r_roi"]),
        ]
    )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
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

    ax_bar = fig.add_axes([0.10, 0.08, 0.80, 0.28])
    labels = [s["stimulus_id"].replace("_", "\n") for s in stimuli]
    x = np.arange(len(stimuli))
    width = 0.18
    pixel_disk = [s.get("mean_pixel_r_disk", np.nan) for s in stimuli]
    pixel_roi = [s.get("mean_pixel_r_roi", np.nan) for s in stimuli]
    spat_disk = [s.get("mean_trial_spatial_r_disk", np.nan) for s in stimuli]
    spat_roi = [s.get("mean_trial_spatial_r_roi", np.nan) for s in stimuli]
    ax_bar.bar(x - 1.5 * width, pixel_disk, width, label="pixel-r disk", color="#4c78a8")
    ax_bar.bar(x - 0.5 * width, pixel_roi, width, label="pixel-r ROI", color="#f58518")
    ax_bar.bar(x + 0.5 * width, spat_disk, width, label="spatial-r disk", color="#54a24b")
    ax_bar.bar(x + 1.5 * width, spat_roi, width, label="spatial-r ROI", color="#e45756")
    ax_bar.axhline(0.0, color="0.5", lw=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylabel("Pearson r", fontsize=9)
    ax_bar.set_title("Per-stimulus: disk vs ROI", fontsize=10)
    ax_bar.legend(fontsize=7, ncol=4, loc="upper right")
    ax_bar.set_ylim(
        -0.05,
        max(0.55, np.nanmax([*pixel_disk, *pixel_roi, *spat_disk, *spat_roi]) + 0.08),
    )

    note = (
        "pixel-r = mean of full-test-split per-pixel Pearson map inside mask "
        "(disk value identical across rows; ROI = same map under each ROI). "
        "spatial-r = mean trial spatial Pearson r for that stimulus "
        "(split test>val>train). Prefer spatial-r in LOO when features are identical."
    )
    fig.text(0.05, 0.015, note, fontsize=7.5, va="bottom", style="italic")

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_taxonomy_page(pdf: PdfPages) -> None:
    """Stimulus taxonomy table with held-out / ROI flags."""
    if TAXONOMY_CSV.is_file():
        tax = pd.read_csv(TAXONOMY_CSV)
    elif TAXONOMY_YAML.is_file():
        with TAXONOMY_YAML.open() as f:
            payload = yaml.safe_load(f)
        tax = pd.DataFrame(payload.get("stimuli", []))
    else:
        raise FileNotFoundError(
            f"Missing taxonomy at {TAXONOMY_CSV} (run build_stimulus_taxonomy.py)"
        )

    heldout_list: list[str] = []
    if TAXONOMY_YAML.is_file():
        with TAXONOMY_YAML.open() as f:
            heldout_list = list(yaml.safe_load(f).get("heldout_list") or [])

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Stimulus taxonomy · held-out flags",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    meta = (
        f"n_stimuli={len(tax)}   "
        f"heldout={int(tax['heldout_candidate'].astype(bool).sum()) if 'heldout_candidate' in tax.columns else '?'}   "
        f"with_roi={int(tax['has_roi'].astype(bool).sum()) if 'has_roi' in tax.columns else '?'}\n"
        f"Held-out list: {', '.join(heldout_list) if heldout_list else '(see heldout_list.yaml)'}"
    )
    fig.text(0.04, 0.92, meta, fontsize=8, va="top", family="monospace")

    ax = fig.add_axes([0.03, 0.05, 0.94, 0.82])
    ax.axis("off")
    cols = [
        c
        for c in (
            "stimulus_id",
            "shape_type",
            "color",
            "size_deg",
            "n_sessions",
            "n_trials",
            "heldout_candidate",
            "has_roi",
        )
        if c in tax.columns
    ]
    cell_text = []
    for _, row in tax.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c in ("heldout_candidate", "has_roi"):
                vals.append("Y" if bool(v) else "")
            elif c == "size_deg":
                vals.append(_fmt_size(v))
            else:
                vals.append(str(v) if pd.notna(v) else "")
        cell_text.append(vals)

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.15)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
            continue
        # Highlight held-out rows
        if "heldout_candidate" in cols:
            held_idx = cols.index("heldout_candidate")
            if cell_text[r - 1][held_idx] == "Y":
                cell.set_facecolor("#fff3cd")
        if c == 0:
            cell.set_text_props(ha="left")

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _add_loo_smoke_page(pdf: PdfPages) -> bool:
    """Optional page summarizing LOO smoke-fold dual metrics."""
    if not LOO_SUMMARY.is_file():
        return False
    df = pd.read_csv(LOO_SUMMARY)
    if df.empty:
        return False

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "LOO smoke folds · protocol B · ResNet18 / layer3 · win_0035_0043",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    note = (
        "Train/val from remainder; held-out stimulus_id is test-only. "
        "Pixel-r across trials is NaN when recon is constant (single condition), "
        "e.g. white_point_0.1 — use spatial-r in that case."
    )
    fig.text(0.05, 0.92, note, fontsize=8, va="top", style="italic")

    ax_table = fig.add_axes([0.05, 0.48, 0.90, 0.38])
    ax_table.axis("off")
    col_labels = [
        "fold_id",
        "n_test",
        "pixel-r disk",
        "pixel-r ROI",
        "spatial-r disk",
        "spatial-r ROI",
    ]
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [
                str(row.get("fold_id", "")),
                str(int(row["n_test"])) if pd.notna(row.get("n_test")) else "—",
                _fmt(row.get("mean_r_disk", float("nan"))),
                _fmt(row.get("mean_r_roi", float("nan"))),
                _fmt(row.get("mean_trial_spatial_r_disk", float("nan"))),
                _fmt(row.get("mean_trial_spatial_r_roi", float("nan"))),
            ]
        )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        if c == 0:
            cell.set_text_props(ha="left")

    # Bar chart of spatial-r (always defined) and pixel-r when finite
    ax_bar = fig.add_axes([0.12, 0.08, 0.76, 0.32])
    labels = [str(x).replace("B__", "").replace("_", "\n") for x in df["fold_id"]]
    x = np.arange(len(df))
    width = 0.18
    pixel_disk = df.get("mean_r_disk", pd.Series([np.nan] * len(df))).to_numpy()
    pixel_roi = df.get("mean_r_roi", pd.Series([np.nan] * len(df))).to_numpy()
    spat_disk = df.get(
        "mean_trial_spatial_r_disk", pd.Series([np.nan] * len(df))
    ).to_numpy()
    spat_roi = df.get(
        "mean_trial_spatial_r_roi", pd.Series([np.nan] * len(df))
    ).to_numpy()
    ax_bar.bar(x - 1.5 * width, pixel_disk, width, label="pixel-r disk", color="#4c78a8")
    ax_bar.bar(x - 0.5 * width, pixel_roi, width, label="pixel-r ROI", color="#f58518")
    ax_bar.bar(x + 0.5 * width, spat_disk, width, label="spatial-r disk", color="#54a24b")
    ax_bar.bar(x + 1.5 * width, spat_roi, width, label="spatial-r ROI", color="#e45756")
    ax_bar.axhline(0.0, color="0.5", lw=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=7)
    ax_bar.set_ylabel("Pearson r", fontsize=9)
    ax_bar.set_title("LOO test metrics: disk vs ROI", fontsize=10)
    ax_bar.legend(fontsize=7, ncol=4, loc="upper right")
    vals = np.concatenate([pixel_disk, pixel_roi, spat_disk, spat_roi])
    finite = vals[np.isfinite(vals)]
    ymax = float(np.max(finite)) + 0.08 if finite.size else 0.55
    ax_bar.set_ylim(-0.05, max(0.55, ymax))

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    for _, path in SANITY_PAGES:
        _require(path)
    for sid in ROI_STIMULI:
        _require(ROI_DIR / f"{sid}__mean_map_roi.png")
        _require(ROI_MASK_DIR / f"{sid}__mask.npy")

    has_loo = LOO_SUMMARY.is_file() and not pd.read_csv(LOO_SUMMARY).empty

    print("Computing disk vs ROI metrics...")
    metrics = compute_roi_vs_disk_metrics(repo=REPO_ROOT)
    OUT_METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"  Wrote {OUT_METRICS_JSON.relative_to(REPO_ROOT)}")
    print(f"  Window used: {metrics['window_id']}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        _add_cover(pdf, has_loo=has_loo)
        for title, path in SANITY_PAGES:
            print(f"  + {title}  ← {path.relative_to(REPO_ROOT)}")
            _add_image_page(pdf, title, path)
        print("  + ROI grid (2×3)")
        _add_roi_grid_page(pdf)
        print("  + Disk vs ROI metrics")
        _add_metrics_page(pdf, metrics)
        print("  + Stimulus taxonomy")
        _add_taxonomy_page(pdf)
        n_extra = 0
        if has_loo:
            print("  + LOO smoke metrics")
            if _add_loo_smoke_page(pdf):
                n_extra = 1

    n_pages = 1 + len(SANITY_PAGES) + 1 + 1 + 1 + n_extra
    print(f"\nWrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print(
        f"Pages: 1 cover + {len(SANITY_PAGES)} sanity + 1 ROI grid "
        f"+ 1 metrics + 1 taxonomy"
        + (f" + {n_extra} LOO" if n_extra else "")
        + f" = {n_pages} total"
    )
    agg = metrics["aggregate"]
    print(
        "\nAggregate mean_pixel_r  "
        f"disk={_fmt(agg['mean_pixel_r_disk'])}  "
        f"ROI={_fmt(agg['mean_pixel_r_roi'])}  "
        f"(n_finite={agg.get('n_stimuli_pixel_r', 0)})"
    )
    print(
        "Aggregate mean_trial_spatial_r  "
        f"disk={_fmt(agg['mean_trial_spatial_r_disk'])}  "
        f"ROI={_fmt(agg['mean_trial_spatial_r_roi'])}  "
        f"(n_finite={agg.get('n_stimuli_spatial_r', 0)})"
    )


if __name__ == "__main__":
    main()
