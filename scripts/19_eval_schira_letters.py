#!/usr/bin/env python3
"""Eval a saved Schira local-ridge model on letter sessions (no retrain).

Default: average weights across Protocol-B non-letter LOO folds
(``block1_prepool`` / raw / NC hull), then predict letters on
``201118c`` + ``201118d`` (``201118b`` is excluded from encoding;
``201118a`` skipped by request).

Example::

  scripts/py scripts/19_eval_schira_letters.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.encoding.ridge_plotting import plot_reconstruction_grid
from src.encoding.schema import encoding_pairs_manifest_path
from src.evaluation.loss_roi import NOISE_CEILING_HULL_MASK_RELPATH
from src.evaluation.mask import apply_mask_nan, masked_map_summary, masked_pearson_r
from src.evaluation.pixel_correlation import (
    pixel_correlation_across_trials,
    pixel_r2_across_trials,
)
from src.evaluation.plotting import plot_pixel_correlation_heatmap
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP, register_mapgeog
from src.schira_encoding.io import (
    build_schira_xy,
    resolve_output_root,
    stimulus_label,
)
from src.schira_encoding.model import LocalRidgeResult, predict_local_ridge
from src.stimuli.identity import attach_stimulus_ids


DEFAULT_LOO = Path(
    "experiments/schira_encoding/gandalf/loo/win_0035_0046/"
    "vgg16_imagenet/block1_prepool/set-100718__anchor-100718a/"
    "protocol_B_noise_ceiling_hull"
)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _average_models(paths: list[Path]) -> LocalRidgeResult:
    models = [joblib.load(p) for p in paths]
    if not models:
        raise RuntimeError("No models to average")
    ref = models[0]
    valid = np.ones_like(ref.valid, dtype=bool)
    for m in models:
        if m.weights.shape != ref.weights.shape:
            raise ValueError("Model shape mismatch while averaging")
        valid &= m.valid
    w = np.mean([m.weights for m in models], axis=0).astype(np.float32)
    b = np.mean([m.intercepts for m in models], axis=0).astype(np.float32)
    a = np.mean([m.alphas for m in models], axis=0).astype(np.float32)
    feat_mean = None
    feat_scale = None
    if ref.standardize_features:
        feat_mean = np.mean(
            [m.feature_mean for m in models], axis=0
        ).astype(np.float32)
        feat_scale = np.mean(
            [m.feature_scale for m in models], axis=0
        ).astype(np.float32)
    return LocalRidgeResult(
        weights=w,
        intercepts=b,
        alphas=a,
        valid=valid,
        spatial_size=ref.spatial_size,
        n_channels=ref.n_channels,
        standardize_features=ref.standardize_features,
        feature_mean=feat_mean,
        feature_scale=feat_scale,
    )


def _odd_even_half_means(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    even = maps[0::2]
    odd = maps[1::2]
    if even.size == 0 or odd.size == 0:
        raise RuntimeError(f"Need both halves, n={maps.shape[0]}")
    return (
        np.nanmean(even, axis=0).astype(np.float32),
        np.nanmean(odd, axis=0).astype(np.float32),
    )


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "configs/schira_encoding/default.yaml")
    p.add_argument("--loo-dir", type=Path, default=repo / DEFAULT_LOO)
    p.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        default=["201118c", "201118d"],
        help="Letter sessions to eval (default: 201118c 201118d; not a/b).",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional single local_ridge.joblib (default: average all LOO folds).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <loo-dir>/letters_eval_201118cd",
    )
    args = p.parse_args()
    register_mapgeog()

    cfg = _load_yaml(args.config)
    loo_dir = args.loo_dir if args.loo_dir.is_absolute() else repo / args.loo_dir
    params = _load_yaml(loo_dir / "params.yaml") if (loo_dir / "params.yaml").is_file() else {}
    monkey = str(params.get("monkey") or cfg["monkey"])
    schira_set = str(params.get("schira_set") or cfg["schira_set"])
    anchor = str(params.get("anchor_session") or cfg["anchor_session"])
    spatial = tuple(int(x) for x in cfg["spatial_size"])
    model_slug = str(params.get("model_slug") or "vgg16_imagenet")
    feature_layer = str(params.get("feature_layer") or "block1_prepool")
    window_id = str(params.get("window_id") or "win_0035_0046")

    sessions = [str(s) for s in args.sessions]
    banned = {"201118a", "201118b"}
    if set(sessions) & banned:
        print(
            f"Note: skipping {sorted(set(sessions) & banned)} "
            "(a=off by request; b=excluded from encoding / no letter pairs)."
        )
        sessions = [s for s in sessions if s not in banned]
    if not sessions:
        raise SystemExit("No sessions left to evaluate")

    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else loo_dir / ("letters_eval_" + "".join(s.replace("201118", "") for s in sessions))
    )
    out_dir = out_dir if out_dir.is_absolute() else repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model is not None:
        model_path = args.model if args.model.is_absolute() else repo / args.model
        result = joblib.load(model_path)
        model_note = str(model_path)
    else:
        paths = sorted(loo_dir.glob("B__*/local_ridge.joblib"))
        if not paths:
            raise FileNotFoundError(f"No LOO models under {loo_dir}")
        result = _average_models(paths)
        model_note = f"mean of {len(paths)} LOO folds"
        joblib.dump(result, out_dir / "local_ridge_avg_folds.joblib")
    print(f"Model: {model_note}")

    hull = np.load(repo / NOISE_CEILING_HULL_MASK_RELPATH).astype(bool)
    mask = result.valid & hull

    pairs_root = resolve_data_path(cfg["paths"]["encoding_pairs_root"], repo)
    pairs = pd.read_parquet(encoding_pairs_manifest_path(pairs_root, monkey, window_id))
    pairs = pairs[pairs["nc_exists"] & pairs["stimulus_exists"]].copy()
    pairs = attach_stimulus_ids(pairs)
    letters = pairs[
        pairs["date"].astype(str).isin(sessions)
        & pairs["stimulus_id"].astype(str).str.startswith("letter_")
    ].copy()
    if letters.empty:
        raise RuntimeError(f"No letter trials for sessions {sessions}")
    print(
        f"Letter trials: {len(letters)}  "
        f"stimuli={sorted(letters['stimulus_id'].astype(str).unique())}  "
        f"sessions={sorted(letters['date'].astype(str).unique())}"
    )

    out_root = resolve_output_root(cfg["paths"]["schira_encoding_root"], repo)
    register_root = resolve_output_root(cfg["paths"]["session_register_root"], repo)
    x, y = build_schira_xy(
        letters,
        repo=repo,
        out_root=out_root,
        monkey=monkey,
        model_slug=model_slug,
        feature_layer=feature_layer,
        schira_set=schira_set,
        anchor_session=anchor,
        spatial_size=spatial,
        register_root=register_root,
    )
    y_hat = predict_local_ridge(x, result)

    rs = [masked_pearson_r(y[i], y_hat[i], mask) for i in range(y.shape[0])]
    metrics = {
        "model": model_note,
        "sessions": sessions,
        "n_trials": int(len(letters)),
        "n_stimuli": int(letters["stimulus_id"].nunique()),
        "r_mean_masked": float(np.nanmean(rs)),
        "r_median_masked": float(np.nanmedian(rs)),
        "feature_layer": feature_layer,
        "window_id": window_id,
        "n_mask_pixels": int(mask.sum()),
    }
    print(
        f"Trial-wise masked r: mean={metrics['r_mean_masked']:.3f} "
        f"median={metrics['r_median_masked']:.3f}"
    )

    # Per-stimulus mean orig|recon grid
    samples = []
    stim_rows = []
    for sid, g in letters.groupby(letters["stimulus_id"].astype(str), sort=True):
        mask_rows = letters["stimulus_id"].astype(str).to_numpy() == sid
        o_mean = np.nanmean(y[mask_rows], axis=0).astype(np.float32)
        r_mean = np.nanmean(y_hat[mask_rows], axis=0).astype(np.float32)
        samples.append(
            (
                {
                    "date": "+".join(sessions),
                    "condition": sid,
                    "stimulus_label": sid,
                    "shape_type": "letter",
                    "trial_global_id": 0,
                    "split": "letters_eval",
                    "trial_dataset": "",
                },
                apply_mask_nan(o_mean, mask),
                apply_mask_nan(r_mean, mask),
            )
        )
        rs_stim = [
            masked_pearson_r(y[i], y_hat[i], mask)
            for i in np.where(mask_rows)[0]
        ]
        stim_rows.append(
            {
                "stimulus_id": sid,
                "n_trials": int(mask_rows.sum()),
                "r_mean_masked": float(np.nanmean(rs_stim)),
                "r_median_masked": float(np.nanmedian(rs_stim)),
            }
        )
    plot_reconstruction_grid(
        samples,
        out_dir / "all_letters_orig_recon.png",
        title=(
            f"Schira letters eval · {feature_layer} · sessions {sessions}\n"
            f"model={model_note} · trial-mean r={metrics['r_mean_masked']:.3f}"
        ),
    )

    # Pixel-r across letter trials (encoding)
    corr = pixel_correlation_across_trials(y, y_hat)
    r2 = pixel_r2_across_trials(y, y_hat)
    corr_m = apply_mask_nan(corr, mask)
    r2_m = apply_mask_nan(r2, mask)
    enc_mean_r = float(masked_map_summary(corr, mask)["mean"])
    enc_mean_r2 = float(masked_map_summary(r2, mask)["mean"])
    metrics["pixel_mean_r_masked"] = enc_mean_r
    metrics["pixel_mean_r2_masked"] = enc_mean_r2
    plot_pixel_correlation_heatmap(
        corr_m,
        out_dir / "pixel_r_letters.png",
        title=f"Letters pixel-r · mean={enc_mean_r:.3f} · n={len(letters)}",
    )

    # Odd/even on letter trials (anchor-aligned), pooled per stimulus then across stimuli
    odd_means, even_means = [], []
    for sid, g in letters.groupby(letters["stimulus_id"].astype(str), sort=True):
        mask_rows = letters["stimulus_id"].astype(str).to_numpy() == sid
        y_s = y[mask_rows]
        order = np.argsort(
            letters.loc[mask_rows, "trial_global_id"].to_numpy()
        )
        y_s = y_s[order]
        if y_s.shape[0] < 2:
            continue
        odd_m, even_m = _odd_even_half_means(y_s)
        odd_means.append(odd_m)
        even_means.append(even_m)
    if odd_means:
        oe_r = pixel_correlation_across_trials(
            np.stack(odd_means), np.stack(even_means)
        )
        oe_r2 = pixel_r2_across_trials(
            np.stack(odd_means), np.stack(even_means)
        )
        oe_mean_r = float(masked_map_summary(oe_r, mask)["mean"])
        oe_mean_r2 = float(masked_map_summary(oe_r2, mask)["mean"])
        metrics["oe_mean_r_masked"] = oe_mean_r
        metrics["oe_mean_r2_masked"] = oe_mean_r2
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
        for ax, title, arr in zip(
            axes,
            (
                f"Encoding pixel-r\nmean={enc_mean_r:.3f}",
                f"Odd/even r\nmean={oe_mean_r:.3f}",
            ),
            (corr_m, apply_mask_nan(oe_r, mask)),
        ):
            im = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label="Pearson r")
        fig.suptitle(
            f"Schira letters · {feature_layer} · {sessions} · n_stim={len(odd_means)}",
            fontsize=11,
        )
        fig.savefig(out_dir / "encoding_vs_noise_corr_r.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"OE pooled mean r={oe_mean_r:.3f}  encoding pixel-r={enc_mean_r:.3f}")

    pd.DataFrame(stim_rows).to_csv(out_dir / "per_stimulus_r.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
