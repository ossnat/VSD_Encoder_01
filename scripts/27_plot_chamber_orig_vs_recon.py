#!/usr/bin/env python3
"""Paper figures: full-chamber original vs predicted VSD, one stimulus each.

Two models, both from the current encoding code (not the legacy Schira path):

* Ridge baseline — leave-one-stimulus-out ResNet-18 / layer3
  (``experiments/loo_encoding/runs/2026-08-25_35-46_resnet18_l3_raw``).
  Weights are per pixel and were fit on the noise-ceiling hull, so the
  reconstruction is defined on that support. The panel is still the full
  100×100 chamber; pixels without a weight are left blank.
* Schira shared-channel readout — VGG-16 ``block1_prepool`` (64 channels),
  one weight per channel applied at every pixel
  (``global_channel_ridge``). The saved fit used the hull only to choose
  those weights; the figure applies the same weights on the full Schira
  lookup-table chamber.

Both panels are masked to the noise-ceiling hull, so original and
reconstruction share that polygon. Pearson r is the spatial correlation
inside the hull. There is no difference panel.

Example::

  scripts/py scripts/27_plot_chamber_orig_vs_recon.py
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter

from src.evaluation.mask import apply_mask_nan, masked_pearson_r
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP, register_mapgeog
from src.schira_encoding.global_channel_model import (
    predict_global_channel_ridge,
)
from src.schira_encoding.io import build_schira_xy, resolve_output_root
from src.schira_encoding.lut import SchiraLUT
from src.schira_encoding.schema import (
    global_channel_ridge_loo_dir,
    lut_path,
)

# Grant figures: the four 0.3° circles (outline and filled, black and white).
DEFAULT_STIMULI: tuple[str, ...] = (
    "black_circle_contour_0.3",
    "black_filled_circle_0.3",
    "white_circle_contour_0.3",
    "white_filled_circle_0.3",
)

RIDGE_PROTOCOL = (
    "experiments/loo_encoding/runs/2026-08-25_35-46_resnet18_l3_raw"
    "/protocol_B_raw_NChull_all"
)
PAPER_DPI = 300
HULL_MASK = Path(
    "experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy"
)


def _load_yaml(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


_TITLES = {
    "black_bar_vertical_1": "Vertical bar",
    "black_bar_horizontal_1": "Horizontal bar",
    "black_circle_contour_0.3": "Black circle outline, 0.3°",
    "black_circle_contour_0.95": "Black circle outline, 0.95°",
    "black_filled_circle_0.3": "Black filled circle, 0.3°",
    "white_circle_contour_0.3": "White circle outline, 0.3°",
    "white_filled_circle_0.3": "White filled circle, 0.3°",
    "white_filled_circle_0.8": "White filled circle, 0.8°",
    "black_triangle_contour_0.4": "Black triangle outline, 0.4°",
    "letter_D_white_1": "Letter D",
    "letter_N_white_1": "Letter N",
    "letter_G_white_1": "Letter G",
}


def _display_name(stimulus_id: str, stimulus_text: str) -> str:
    if stimulus_id in _TITLES:
        return _TITLES[stimulus_id]
    text = re.sub(r"^cond\d+:\s*", "", stimulus_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if text:
        return text[0].upper() + text[1:]
    return stimulus_id.replace("_", " ")


def _shared_limits(original: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """One color scale for both panels, anchored on the original chamber."""
    vals = np.asarray(original, dtype=np.float64).ravel()
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        pred = np.asarray(predicted, dtype=np.float64).ravel()
        both = np.concatenate([finite, pred[np.isfinite(pred)]])
        lo = float(np.min(both))
        hi = float(np.max(both))
    if hi <= lo:
        pad = 1e-4 if lo == 0 else abs(lo) * 1e-4
        return lo - pad, hi + pad
    return lo, hi


def _paper_cmap():
    register_mapgeog()
    cmap = matplotlib.colormaps[VSD_CMAP].copy()
    cmap.set_bad("#ffffff")
    return cmap


def _same_hull_shape(
    original: np.ndarray,
    predicted: np.ndarray,
    hull: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep a pixel only when both maps are finite inside the hull."""
    support = np.asarray(hull, dtype=bool) & np.isfinite(original) & np.isfinite(predicted)
    return (
        apply_mask_nan(original, support),
        apply_mask_nan(predicted, support),
        support,
    )


_PANEL_CM = 1.8


def _load_stimulus_rgb(
    protocol_dir: Path,
    stimulus_id: str,
    *,
    repo: Path,
    dates: list[str] | None = None,
) -> np.ndarray:
    """RGB stimulus image for the trials shown in the VSD panels."""
    from PIL import Image

    fold = protocol_dir / f"B__{stimulus_id}"
    manifest_path = fold / f"B__{stimulus_id}__manifest.parquet"
    if not manifest_path.is_file():
        manifest_path = fold / "fold_pairs.parquet"
    manifest = pd.read_parquet(manifest_path)
    rows = manifest
    if "loo_split" in manifest.columns:
        rows = manifest[manifest["loo_split"] == "test"]
    if dates:
        wanted = {str(d) for d in dates}
        rows = rows[rows["date"].astype(str).isin(wanted)]
    elif (rows["date"].astype(str) != "201118a").any():
        rows = rows[rows["date"].astype(str) != "201118a"]
    if rows.empty:
        raise RuntimeError(f"{stimulus_id}: no stimulus image for the plotted trials")
    path = resolve_data_path(str(rows.iloc[0]["image_path"]), repo)
    return np.asarray(Image.open(path).convert("RGB"))


def plot_orig_vs_predicted(
    original: np.ndarray,
    predicted: np.ndarray,
    stimulus: np.ndarray,
    out_stem: Path,
    *,
    title: str,
    model_label: str,
    pearson_r: float,
) -> None:
    """Stimulus | reconstructed | original. Each panel is 1.8 cm square."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = _shared_limits(original, predicted)
    cmap = _paper_cmap()
    norm = Normalize(vmin=lo, vmax=hi)

    panel = _PANEL_CM / 2.54
    gap = 0.08 / 2.54
    cbar_w = 0.22 / 2.54
    left = 0.04 / 2.54
    bottom = 0.04 / 2.54
    top = 1.25 / 2.54
    right = 0.55 / 2.54
    fig_w = left + 3 * panel + 2 * gap + gap + cbar_w + right
    fig_h = bottom + panel + top
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _axes(index: int):
        x = left + index * (panel + gap)
        return fig.add_axes([x / fig_w, bottom / fig_h, panel / fig_w, panel / fig_h])

    ax_stim, ax_recon, ax_orig = _axes(0), _axes(1), _axes(2)
    ax_stim.imshow(stimulus, interpolation="nearest", origin="upper")
    recon_im = ax_recon.imshow(
        predicted, cmap=cmap, norm=norm, interpolation="nearest", origin="upper"
    )
    ax_orig.imshow(
        original, cmap=cmap, norm=norm, interpolation="nearest", origin="upper"
    )
    for ax, label in (
        (ax_stim, "Stimulus"),
        (ax_recon, "Reconstructed"),
        (ax_orig, "Original"),
    ):
        ax.set_title(label, fontsize=5, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c8c8c8")
            spine.set_linewidth(0.4)

    cbar_x = left + 3 * panel + 2 * gap + gap
    cax = fig.add_axes(
        [cbar_x / fig_w, bottom / fig_h, cbar_w / fig_w, panel / fig_h]
    )
    cbar = fig.colorbar(recon_im, cax=cax)
    cbar.formatter = FormatStrFormatter("%.4f")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=4, length=2, pad=1)
    cbar.set_label("F/F0", fontsize=5)
    r_txt = "n/a" if not np.isfinite(pearson_r) else f"{pearson_r:.3f}"
    fig.suptitle(
        f"{title}\n{model_label}   ·   r = {r_txt}",
        fontsize=6,
        y=0.98,
    )
    # Stimulus ids already contain dots (for example ``0.3``), so do not use
    # Path.with_suffix. No tight crop: panel axes stay 1.8 cm.
    png_path = Path(f"{out_stem}.png")
    svg_path = Path(f"{out_stem}.svg")
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(svg_path)
    plt.close(fig)


def _stimulus_text(pairs: pd.DataFrame, stimulus_id: str) -> str:
    if "stimulus_text" not in pairs.columns or pairs.empty:
        return ""
    text = str(pairs.iloc[0].get("stimulus_text", "") or "")
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _ridge_maps(
    protocol_dir: Path,
    stimulus_id: str,
    *,
    repo: Path,
    spatial_size: tuple[int, int],
    dates: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    fold = protocol_dir / f"B__{stimulus_id}"
    manifest = fold / f"B__{stimulus_id}__manifest.parquet"
    text = ""
    if dates:
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing ridge manifest in {fold}")
        import joblib

        from src.encoding.ridge import build_xy, predict_maps

        manifest_df = pd.read_parquet(manifest)
        test = manifest_df
        if "loo_split" in manifest_df.columns:
            test = manifest_df[manifest_df["loo_split"] == "test"]
        wanted = {str(d) for d in dates}
        test = test[test["date"].astype(str).isin(wanted)].reset_index(drop=True)
        if test.empty:
            raise RuntimeError(f"{stimulus_id}: no test trials in {sorted(wanted)}")
        payload = joblib.load(fold / "model.joblib")
        result = payload["result"] if isinstance(payload, dict) else payload
        features, targets = build_xy(test, repo=repo, spatial_size=spatial_size)
        predicted_trials = predict_maps(result, features, spatial_size)
        height, width = spatial_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            original = np.nanmean(targets, axis=0).astype(np.float32).reshape(height, width)
            predicted = np.nanmean(predicted_trials, axis=0).astype(np.float32)
        if predicted.shape != (height, width):
            predicted = predicted.reshape(height, width)
        return original, predicted, _stimulus_text(test, stimulus_id)

    orig_path = fold / "fold_mean_orig.npy"
    recon_path = fold / "fold_mean_recon.npy"
    if not orig_path.is_file() or not recon_path.is_file():
        raise FileNotFoundError(f"Missing ridge fold means in {fold}")
    original = np.load(orig_path).astype(np.float32)
    predicted = np.load(recon_path).astype(np.float32)
    if manifest.is_file():
        manifest_df = pd.read_parquet(manifest)
        test = manifest_df
        if "loo_split" in manifest_df.columns:
            test = manifest_df[manifest_df["loo_split"] == "test"]
        held = test[test["stimulus_id"].astype(str) == stimulus_id] if "stimulus_id" in test.columns else test
        text = _stimulus_text(held if not held.empty else test, stimulus_id)
    return original, predicted, text


def _schira_maps(
    *,
    repo: Path,
    protocol_dir: Path,
    stimulus_id: str,
    lut: SchiraLUT,
    out_root: Path,
    register_root: Path,
    monkey: str,
    model_slug: str,
    feature_layer: str,
    schira_set: str,
    anchor_session: str,
    spatial_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, str, int]:
    import joblib

    fold = protocol_dir / f"B__{stimulus_id}"
    model_path = fold / "global_channel_ridge.joblib"
    pairs_path = fold / "fold_pairs.parquet"
    if not model_path.is_file() or not pairs_path.is_file():
        raise FileNotFoundError(f"Missing Schira fold artifacts in {fold}")
    result = joblib.load(model_path)
    if int(result.n_channels) != 64:
        raise ValueError(
            f"{stimulus_id}: expected 64 shared channels, got {result.n_channels}"
        )
    pairs = pd.read_parquet(pairs_path)
    test = pairs[pairs["loo_split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise RuntimeError(f"{stimulus_id}: empty Schira test split")
    features, targets = build_schira_xy(
        test,
        repo=repo,
        out_root=out_root,
        monkey=monkey,
        model_slug=model_slug,
        feature_layer=feature_layer,
        schira_set=schira_set,
        anchor_session=anchor_session,
        spatial_size=spatial_size,
        register_root=register_root,
        lut=None,
    )
    # Shared weights on the hull, where the lookup actually has a sample.
    chamber_model = replace(
        result, valid=np.asarray(lut.valid, dtype=bool)
    )
    predicted_trials = predict_global_channel_ridge(features, chamber_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        original = np.nanmean(targets, axis=0).astype(np.float32)
        predicted = np.nanmean(predicted_trials, axis=0).astype(np.float32)
    return original, predicted, _stimulus_text(test, stimulus_id), int(len(test))


def main() -> None:
    repo = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stimuli",
        nargs="*",
        default=list(DEFAULT_STIMULI),
        help="Stimulus ids (default: bars and circles).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "plots" / "Oct_11",
    )
    parser.add_argument(
        "--ridge-protocol",
        type=Path,
        default=repo / RIDGE_PROTOCOL,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="Ridge only: average these sessions instead of every test trial.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ridge", "schira"),
        default=["ridge", "schira"],
        help="Which readouts to plot.",
    )
    args = parser.parse_args()
    run_ridge = "ridge" in args.models
    run_schira = "schira" in args.models

    cfg = _load_yaml(args.config)
    stimuli_cfg = _load_yaml(repo / str(cfg.get("stimuli_config", "configs/stimuli/default.yaml")))
    monkey = str(cfg["monkey"])
    schira_set = str(cfg["schira_set"])
    anchor = str(cfg["anchor_session"])
    spatial = tuple(int(v) for v in cfg["spatial_size"])
    window_id = "win_0035_0046"
    model_name = "vgg16_imagenet"
    layer = "block1_prepool"
    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    register_root = resolve_data_path(cfg["paths"]["session_register_root"], repo)
    canvas = int(stimuli_cfg.get("canvas_size", 210))
    input_size = int(cfg.get("input_size", 224))
    lut = SchiraLUT.load(
        lut_path(
            out_root,
            monkey,
            schira_set=schira_set,
            anchor_session=anchor,
            spatial_h=spatial[0],
            spatial_w=spatial[1],
            canvas_size=canvas,
            input_size=input_size,
        )
    )
    schira_protocol = global_channel_ridge_loo_dir(
        out_root,
        monkey,
        window_id,
        model_name,
        layer,
        schira_set=schira_set,
        anchor_session=anchor,
        protocol="B",
        loss_roi=str(cfg.get("loss_roi", "noise_ceiling_hull")),
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    ridge_dir = (
        args.ridge_protocol
        if args.ridge_protocol.is_absolute()
        else repo / args.ridge_protocol
    )
    hull = np.load(repo / HULL_MASK).astype(bool)
    if hull.shape != spatial:
        raise ValueError(f"Hull shape {hull.shape} != {spatial}")
    rows: list[dict[str, object]] = []

    for stimulus_id in args.stimuli:
        ridge_r = float("nan")
        schira_r = float("nan")
        if run_ridge:
            print(f"ridge  {stimulus_id}", flush=True)
            ridge_orig, ridge_pred, ridge_text = _ridge_maps(
                ridge_dir,
                stimulus_id,
                repo=repo,
                spatial_size=spatial,
                dates=args.dates,
            )
            ridge_orig, ridge_pred, ridge_support = _same_hull_shape(
                ridge_orig, ridge_pred, hull
            )
            ridge_r = masked_pearson_r(ridge_orig, ridge_pred, ridge_support)
            stimulus_rgb = _load_stimulus_rgb(
                ridge_dir,
                stimulus_id,
                repo=repo,
                dates=args.dates,
            )
            plot_orig_vs_predicted(
                ridge_orig,
                ridge_pred,
                stimulus_rgb,
                out_dir / "ridge" / stimulus_id,
                title=_display_name(stimulus_id, ridge_text)
                + (f"  ·  {', '.join(args.dates)}" if args.dates else ""),
                model_label="Ridge baseline  ·  ResNet-18 layer 3",
                pearson_r=ridge_r,
            )
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "model": "ridge_resnet18_layer3",
                    "hull_pearson_r": ridge_r,
                    "n_hull_pixels": int(ridge_support.sum()),
                }
            )

        if not run_schira:
            print(f"  hull r  ridge={ridge_r:.3f}", flush=True)
            continue

        print(f"schira {stimulus_id}", flush=True)
        schira_orig, schira_pred, schira_text, n_test = _schira_maps(
            repo=repo,
            protocol_dir=schira_protocol,
            stimulus_id=stimulus_id,
            lut=lut,
            out_root=out_root,
            register_root=register_root,
            monkey=monkey,
            model_slug=model_name,
            feature_layer=layer,
            schira_set=schira_set,
            anchor_session=anchor,
            spatial_size=spatial,
        )
        schira_orig, schira_pred, schira_support = _same_hull_shape(
            schira_orig, schira_pred, hull
        )
        schira_r = masked_pearson_r(schira_orig, schira_pred, schira_support)
        plot_orig_vs_predicted(
            schira_orig,
            schira_pred,
            _load_stimulus_rgb(schira_protocol, stimulus_id, repo=repo),
            out_dir / "schira_shared_channels" / stimulus_id,
            title=_display_name(stimulus_id, schira_text),
            model_label="Schira  ·  64 channels, one weight shared across pixels",
            pearson_r=schira_r,
        )
        rows.append(
            {
                "stimulus_id": stimulus_id,
                "model": "schira_vgg16_block1_prepool_shared",
                "hull_pearson_r": schira_r,
                "n_hull_pixels": int(schira_support.sum()),
                "n_test_trials": n_test,
            }
        )
        print(
            f"  hull r  ridge={ridge_r:.3f}  schira={schira_r:.3f}",
            flush=True,
        )

    metrics_path = out_dir / "metrics.csv"
    fieldnames = [
        "stimulus_id",
        "model",
        "hull_pearson_r",
        "n_hull_pixels",
        "n_test_trials",
    ]
    kept: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open(newline="") as handle:
            written = {(row["stimulus_id"], row["model"]) for row in rows}
            for old in csv.DictReader(handle):
                if (old.get("stimulus_id"), old.get("model")) not in written:
                    kept.append(old)
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in kept + rows:
            writer.writerow(row)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
