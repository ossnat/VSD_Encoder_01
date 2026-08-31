#!/usr/bin/env python3
"""Compare flatten Ridge encoding vs odd-even noise corr (Protocol A, NC hull).

Side-by-side maps for pooled fold-pixel r / R² under a LOO run root::

  protocol_A_{window}_NChull_all/overview/pooled_fold_pixel_r[2]__{window}.npy
  noise_corr_odd_even/r[2]_map_pooled_folds_masked__{window}.npy

Usage::

  scripts/py experiments/loo_encoding/compare_encoding_vs_noise_corr.py \\
    --run-root experiments/loo_encoding/runs/070826_cluster \\
    --window raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paths import project_root
from src.plotting_colormaps import VSD_CMAP, register_mapgeog

NOTE_BY_PROTOCOL = {
    "A": (
        "Both encoding and OE use Protocol A fold list (69 date×condition). "
        "Encoding: LOO flatten Ridge fold-mean orig vs recon. "
        "OE: odd vs even trial-mean stacks per fold, then pooled pixel metric. "
        "Both summarized in NC hull. Trial filter: all trials (no cleanliness filter) "
        "for OE; encoding leaf is protocol_A_raw_NChull_all."
    ),
    "C": (
        "Both encoding and OE use Protocol C fold list (~18 stimulus_id folds). "
        "Stimulus-level mean evaluation: encoding orig = mean VSD over ALL trials "
        "with that stimulus_id (all sessions); recon = single ŷ from the fold's "
        "held-out (date, condition) feature X on Protocol C weights "
        "(stimulus out of train; not averaged across sessions). "
        "OE: odd/even split of the same all-session trial pool per stimulus, then "
        "pooled pixel r/R² across folds (n≈18). "
        "Both summarized in NC hull; leaf protocol_C_raw_NChull_all."
    ),
}


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _pick_map(preferred: Path, fallback: Path) -> Path:
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing map: tried {preferred} and {fallback}")


def _encoding_means(pooled_summary: dict, window: str, cleanliness: str) -> tuple[float, float, int]:
    key = f"{window}_{cleanliness}"
    mean_r = pooled_summary.get("encoding_mean_r_hull", {}).get(key)
    mean_r2 = pooled_summary.get("encoding_mean_r2_hull", {}).get(key)
    n_folds = None
    for leaf in pooled_summary.get("leaves") or []:
        if (
            leaf.get("window_kind") == window
            and leaf.get("cleanliness") == cleanliness
        ):
            if mean_r is None:
                mean_r = leaf.get("mean_r_hull")
            if mean_r2 is None:
                mean_r2 = leaf.get("mean_r2_hull")
            n_folds = leaf.get("n_folds")
            break
    if mean_r is None or mean_r2 is None:
        raise KeyError(
            f"Encoding means missing for {key!r} in pooled_fold_pixel_r_summary.json"
        )
    return float(mean_r), float(mean_r2), int(n_folds or 0)


def _plot_1x2(
    *,
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
    im = None
    for ax, title, arr in zip(
        axes, (left_title, right_title), (left, right), strict=True
    ):
        im = ax.imshow(arr, cmap=VSD_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, label=colorbar_label)
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_2x2(
    *,
    enc_r: np.ndarray,
    oe_r: np.ndarray,
    enc_r2: np.ndarray,
    oe_r2: np.ndarray,
    titles: tuple[str, str, str, str],
    out_path: Path,
    n_folds: int,
    window: str,
    protocol: str = "A",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0), layout="constrained")
    r_panels = [
        (axes[0, 0], titles[0], enc_r),
        (axes[0, 1], titles[1], oe_r),
    ]
    r2_panels = [
        (axes[1, 0], titles[2], enc_r2),
        (axes[1, 1], titles[3], oe_r2),
    ]
    im_r = None
    for ax, title, arr in r_panels:
        im_r = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    im_r2 = None
    for ax, title, arr in r2_panels:
        im_r2 = ax.imshow(arr, cmap=VSD_CMAP, vmin=-1.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(
        im_r, ax=axes[0, :].ravel().tolist(), fraction=0.035, pad=0.02, label="Pearson r"
    )
    fig.colorbar(
        im_r2, ax=axes[1, :].ravel().tolist(), fraction=0.035, pad=0.02, label="R²"
    )
    fig.suptitle(
        f"Protocol {protocol} · flatten Ridge vs odd/even noise corr · "
        f"{window} · all · n={n_folds}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        choices=("A", "C"),
        default="A",
        help="Protocol A (date×condition) or C (stimulus_id, ~20 folds)",
    )
    p.add_argument(
        "--run-root",
        type=Path,
        default=None,
    )
    p.add_argument("--window", choices=["raw", "zscore"], default="raw")
    p.add_argument(
        "--cleanliness",
        default="all",
        help="Encoding leaf cleanliness tag (default: all)",
    )
    p.add_argument(
        "--encoding-dir",
        type=Path,
        default=None,
        help="Default: <run-root>/protocol_{A|C}_{window}_NChull_{cleanliness}",
    )
    p.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="Default: <run-root>/noise_corr_odd_even or …__protocol_C",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--pooled-summary",
        type=Path,
        default=None,
        help="Override pooled summary JSON (default: protocol-specific under run root)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    register_mapgeog()
    repo = project_root()
    protocol = str(args.protocol).upper()
    if args.run_root is not None:
        run_root = _resolve(repo, args.run_root)
    elif protocol == "C":
        run_root = _resolve(
            repo, Path("experiments/loo_encoding/runs/2026-08-14_35-46_resnet18_l3")
        )
    else:
        run_root = _resolve(repo, Path("experiments/loo_encoding/runs/070826_cluster"))

    window = args.window
    cleanliness = args.cleanliness
    encoding_dir = _resolve(
        repo,
        args.encoding_dir
        or (
            run_root
            / f"protocol_{protocol}_{window}_NChull_{cleanliness}"
        ),
    )
    noise_dir = _resolve(
        repo,
        args.noise_dir
        or (
            run_root / "noise_corr_odd_even__protocol_C"
            if protocol == "C"
            else run_root / "noise_corr_odd_even"
        ),
    )
    out_dir = _resolve(repo, args.out_dir or run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pooled_summary is not None:
        pooled_summary_path = _resolve(repo, args.pooled_summary)
    elif protocol == "C":
        pooled_summary_path = (
            run_root / "pooled_fold_pixel_r_summary__protocol_C.json"
        )
    else:
        pooled_summary_path = run_root / "pooled_fold_pixel_r_summary.json"

    pooled_summary = _load_json(pooled_summary_path)
    enc_mean_r, enc_mean_r2, enc_n = _encoding_means(
        pooled_summary, window, cleanliness
    )
    # Prefer r2 summary if present for mean_r2
    r2_summary_path = (
        run_root / f"pooled_fold_pixel_r2_summary__protocol_{protocol}.json"
        if protocol == "C"
        else run_root / "pooled_fold_pixel_r2_summary.json"
    )
    if r2_summary_path.is_file():
        r2_sum = _load_json(r2_summary_path)
        key = f"{window}_{cleanliness}"
        if key in (r2_sum.get("encoding_mean_r2_hull") or {}):
            enc_mean_r2 = float(r2_sum["encoding_mean_r2_hull"][key])

    noise = _load_json(noise_dir / "summary.json")
    oe_mean_r = float(noise[f"{window}_mean_r_hull"])
    oe_mean_r2 = float(noise[f"{window}_mean_r2_hull"])
    oe_n = int(noise.get("n_folds", 0))
    n_folds_enc = enc_n
    n_folds_oe = oe_n
    n_folds = enc_n or oe_n
    n_folds_label = (
        f"{n_folds_enc}"
        if n_folds_enc == n_folds_oe
        else f"enc={n_folds_enc}, oe={n_folds_oe}"
    )

    overview = encoding_dir / "overview"
    enc_r_path = overview / f"pooled_fold_pixel_r__{window}.npy"
    enc_r2_path = overview / f"pooled_fold_pixel_r2__{window}.npy"
    oe_r_path = _pick_map(
        noise_dir / f"r_map_pooled_folds_masked__{window}.npy",
        noise_dir / f"r_map_pooled_folds__{window}.npy",
    )
    oe_r2_path = _pick_map(
        noise_dir / f"r2_map_pooled_folds_masked__{window}.npy",
        noise_dir / f"r2_map_pooled_folds__{window}.npy",
    )

    enc_r = np.load(enc_r_path)
    enc_r2 = np.load(enc_r2_path)
    oe_r = np.load(oe_r_path)
    oe_r2 = np.load(oe_r2_path)

    tag = f"{window} · all · n={n_folds_label}"
    enc_r_title = f"Flatten Ridge encoding\n{tag}\nmean r (NC hull) = {enc_mean_r:.3f}"
    oe_r_title = f"Odd/even noise corr\n{tag}\nmean r (NC hull) = {oe_mean_r:.3f}"
    enc_r2_title = (
        f"Flatten Ridge encoding\n{tag}\nmean R² (NC hull) = {enc_mean_r2:.3f}"
    )
    oe_r2_title = (
        f"Odd/even noise corr\n{tag}\nmean R² (NC hull) = {oe_mean_r2:.3f}"
    )

    if protocol == "A":
        r_png = out_dir / f"encoding_vs_noise_corr__{window}.png"
        r2_png = out_dir / f"encoding_vs_noise_corr_r2__{window}.png"
        grid_png = out_dir / f"encoding_vs_noise_corr_r_and_r2__{window}.png"
        json_name = f"encoding_vs_noise_corr_comparison__{window}.json"
    else:
        r_png = out_dir / f"encoding_vs_noise_corr__{window}.png"
        r2_png = out_dir / f"encoding_vs_noise_corr_r2__{window}.png"
        grid_png = out_dir / f"encoding_vs_noise_corr_r_and_r2__{window}.png"
        json_name = f"encoding_vs_noise_corr_comparison__{window}.json"

    _plot_1x2(
        left=enc_r,
        right=oe_r,
        left_title=enc_r_title,
        right_title=oe_r_title,
        out_path=r_png,
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="Pearson r",
        suptitle=f"Protocol {protocol} · encoding vs odd/even noise corr (r, {window})",
    )
    _plot_1x2(
        left=enc_r2,
        right=oe_r2,
        left_title=enc_r2_title,
        right_title=oe_r2_title,
        out_path=r2_png,
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="R²",
        suptitle=f"Protocol {protocol} · encoding vs odd/even noise corr (R², {window})",
    )
    _plot_2x2(
        enc_r=enc_r,
        oe_r=oe_r,
        enc_r2=enc_r2,
        oe_r2=oe_r2,
        titles=(enc_r_title, oe_r_title, enc_r2_title, oe_r2_title),
        out_path=grid_png,
        n_folds=n_folds,
        window=window,
        protocol=protocol,
    )

    note = NOTE_BY_PROTOCOL[protocol]
    leaf_name = f"protocol_{protocol}_{window}_NChull_{cleanliness}"
    if window != "raw" or cleanliness != "all":
        note = note.replace(
            f"protocol_{protocol}_raw_NChull_all",
            leaf_name,
        )

    comparison = {
        "protocol": protocol,
        "window": window,
        "cleanliness": cleanliness,
        "n_folds_encoding": n_folds_enc,
        "n_folds_odd_even": n_folds_oe,
        "n_folds": n_folds,
        "n_folds_mismatch": n_folds_enc != n_folds_oe,
        "encoding_mean_r_hull": enc_mean_r,
        "encoding_mean_r2_hull": enc_mean_r2,
        "odd_even_mean_r_hull": oe_mean_r,
        "odd_even_mean_r2_hull": oe_mean_r2,
        "encoding_r_npy": str(enc_r_path.relative_to(repo)),
        "encoding_r2_npy": str(enc_r2_path.relative_to(repo)),
        "odd_even_r_npy": str(oe_r_path.relative_to(repo)),
        "odd_even_r2_npy": str(oe_r2_path.relative_to(repo)),
        "r_png": str(r_png.relative_to(repo)),
        "r2_png": str(r2_png.relative_to(repo)),
        "r_and_r2_png": str(grid_png.relative_to(repo)),
        "trial_filter": "all",
        "note": note,
    }
    json_path = out_dir / json_name
    with json_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Encoding vs odd-even · Protocol {protocol} ({window}, all trials, NC hull)")
    print(
        f"  encoding mean r / R²: {enc_mean_r:.4f} / {enc_mean_r2:.4f}  "
        f"(n={n_folds_enc})"
    )
    print(
        f"  odd-even mean r / R²: {oe_mean_r:.4f} / {oe_mean_r2:.4f}  "
        f"(n={n_folds_oe})"
    )
    if n_folds_enc != n_folds_oe:
        print("  WARNING: n_folds mismatch between encoding and odd-even")
    print(f"  r figure  → {r_png.relative_to(repo)}")
    print(f"  R² figure → {r2_png.relative_to(repo)}")
    print(f"  2×2 grid  → {grid_png.relative_to(repo)}")
    print(f"  summary   → {json_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
