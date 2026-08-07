"""One-off: diagnostic maps for 100718a noisy-trial outliers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.plotting_colormaps import VSD_CMAP, register_mapgeog

SESSION = "100718a"
WINDOW_TAG = "win_0035_0046_zscore"
REPO = project_root()

PATTERN_OUTLIERS = [
    3015, 2999,  # AN1
    3021, 3040,  # AN2
    3060, 3053,  # AN3
    3077,  # AN4
    3109, 3119, 3113,  # AN5
    3142,  # AN7
]

AMP_CLIM_ONLY = [3038, 3046, 3069, 3125, 3122, 3148, 2996]

METRICS_CSV = REPO / "scripts/_tmp_messy_outliers_100718a.csv"
FIG_DIR = REPO / "experiments/stimulus_catalog/figures"
FIG_STEM = f"noisy_trials_{SESSION}__{WINDOW_TAG}"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _row_clim(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float, float]:
    m = np.isfinite(arr)
    if m.sum() == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(arr[m], [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _resid_clim(resid: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(resid)
    if m.sum() == 0:
        return -1.0, 1.0
    rmax = float(np.percentile(np.abs(resid[m]), 99))
    rmax = max(rmax, 1e-12)
    return -rmax, rmax


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    av = a[m].ravel()
    bv = b[m].ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def load_session_maps(cfg: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, list[int]]]:
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    index_path = resolve_data_path(cfg["trials_index_csv"], REPO)
    df = pd.read_csv(index_path)
    df = df[df["date"].astype(str) == SESSION].copy()
    df = df.sort_values(["condition", "trial_global_id"]).reset_index(drop=True)

    load_kw = dict(
        repo=REPO,
        spatial_size=spatial_size,
        start_frame=int(cfg["start_frame"]),
        end_frame=int(cfg["end_frame"]),
        avg_method=str(cfg["avg_method"]),
        normalization=str(cfg["normalization"]),
        baseline_start_frame=int(cfg["baseline_start_frame"]),
        baseline_end_frame=int(cfg["baseline_end_frame"]),
        baseline_std_eps=float(cfg["baseline_std_eps"]),
    )

    stacks: dict[str, np.ndarray] = {}
    gids_by_cond: dict[str, list[int]] = {}
    for cond, g in df.groupby("condition", sort=True):
        g = g.reset_index(drop=True)
        maps = []
        gids = []
        for r in g.itertuples(index=False):
            maps.append(
                load_h5_mean_frame(
                    target_file=str(r.target_file),
                    trial_global_id=int(r.trial_global_id),
                    **load_kw,
                )
            )
            gids.append(int(r.trial_global_id))
        stacks[cond] = np.stack(maps, axis=0).astype(np.float64)
        gids_by_cond[cond] = gids
    return df, stacks, gids_by_cond


def loo_for_index(stack: np.ndarray, idx: int) -> np.ndarray:
    trial = stack[idx]
    total = np.nansum(stack, axis=0)
    finite_counts = np.sum(np.isfinite(stack), axis=0)
    count_loo = finite_counts - np.isfinite(trial).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        loo = (total - np.where(np.isfinite(trial), trial, 0.0)) / np.maximum(count_loo, 1)
    return np.where(count_loo > 0, loo, np.nan)


def best_sibling_index(
    metrics: pd.DataFrame,
    *,
    condition: str,
    exclude_gid: int,
    gids_by_cond: dict[str, list[int]],
) -> int:
    """Sibling with highest corr_loo in the same condition (cleanest vs LOO mean)."""
    cond_gids = gids_by_cond[condition]
    sub = metrics[
        (metrics["condition"] == condition)
        & (metrics["trial_global_id"] != exclude_gid)
    ]
    if sub.empty:
        # Fallback: first other index in condition
        for i, g in enumerate(cond_gids):
            if g != exclude_gid:
                return i
        return 0
    best_gid = int(sub.loc[sub["corr_loo"].idxmax(), "trial_global_id"])
    return cond_gids.index(best_gid)


def _paired_clim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Shared 1–99% clim for side-by-side trial vs best-sibling comparison."""
    finite = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi - lo < 1e-12:
        hi = lo + 1e-6
    return float(lo), float(hi)


def gid_to_cond_idx(
    gid: int,
    df: pd.DataFrame,
    gids_by_cond: dict[str, list[int]],
) -> tuple[str, int]:
    row = df[df["trial_global_id"] == gid]
    if row.empty:
        raise KeyError(f"gid {gid} not in session index")
    cond = str(row.iloc[0]["condition"])
    gids = gids_by_cond[cond]
    idx = gids.index(gid)
    return cond, idx


def plot_panel_grid(
    gids: list[int],
    *,
    section_title: str,
    df: pd.DataFrame,
    stacks: dict[str, np.ndarray],
    gids_by_cond: dict[str, list[int]],
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    n_rows = len(gids)
    n_cols = 4
    col_titles = ["Trial", "Best sibling", "LOO mean", "Residual"]
    fig_w = 3.2 * n_cols
    fig_h = 2.6 * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, pad=8)

    for row_i, gid in enumerate(gids):
        cond, idx = gid_to_cond_idx(gid, df, gids_by_cond)
        stack = stacks[cond]
        trial = stack[idx]
        loo = loo_for_index(stack, idx)
        resid = trial - loo
        best_idx = best_sibling_index(
            metrics,
            condition=cond,
            exclude_gid=gid,
            gids_by_cond=gids_by_cond,
        )
        best = stack[best_idx]
        best_gid = gids_by_cond[cond][best_idx]
        best_corr_loo = float(
            metrics.loc[metrics["trial_global_id"] == best_gid, "corr_loo"].iloc[0]
        )

        mrow = metrics[metrics["trial_global_id"] == gid]
        if mrow.empty:
            corr_loo = spatial_corr(trial, loo)
            rms = float(np.sqrt(np.nanmean((trial - loo) ** 2)))
            p99 = float(np.nanpercentile(np.abs(trial[np.isfinite(trial)]), 99))
            flag = ""
            stim = ""
        else:
            r = mrow.iloc[0]
            corr_loo = float(r["corr_loo"])
            rms = float(r["rms_resid"])
            p99 = float(r["p99_abs_z"])
            flag = str(r.get("flag", "") or "")
            stim = str(r.get("stimulus_id", "") or "")

        row_label = (
            f"gid={gid}  {cond}  {stim}\n"
            f"corr_loo={corr_loo:.3f}  rms={rms:.3f}  p99|z|={p99:.2f}"
            + (f"  [{flag}]" if flag else "")
        )
        axes[row_i, 0].text(
            -0.08,
            0.5,
            row_label,
            transform=axes[row_i, 0].transAxes,
            fontsize=8,
            va="center",
            ha="right",
            wrap=True,
        )

        trial_vmin, trial_vmax = _paired_clim(trial, best)
        loo_vmin, loo_vmax = _row_clim(loo)
        rvmin, rvmax = _resid_clim(resid)

        im0 = axes[row_i, 0].imshow(trial, cmap=VSD_CMAP, vmin=trial_vmin, vmax=trial_vmax)
        axes[row_i, 0].axis("off")
        fig.colorbar(im0, ax=axes[row_i, 0], fraction=0.046, pad=0.02)

        im1 = axes[row_i, 1].imshow(best, cmap=VSD_CMAP, vmin=trial_vmin, vmax=trial_vmax)
        axes[row_i, 1].axis("off")
        axes[row_i, 1].set_title(
            f"gid={best_gid}  corr_loo={best_corr_loo:.3f}", fontsize=8
        )
        fig.colorbar(im1, ax=axes[row_i, 1], fraction=0.046, pad=0.02)

        im2 = axes[row_i, 2].imshow(loo, cmap=VSD_CMAP, vmin=loo_vmin, vmax=loo_vmax)
        axes[row_i, 2].axis("off")
        fig.colorbar(im2, ax=axes[row_i, 2], fraction=0.046, pad=0.02)

        im3 = axes[row_i, 3].imshow(resid, cmap="coolwarm", vmin=rvmin, vmax=rvmax)
        axes[row_i, 3].axis("off")
        fig.colorbar(im3, ax=axes[row_i, 3], fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{SESSION} noisy trials — {section_title} | evoked frames 35–46 z-score",
        fontsize=12,
        y=1.002,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.02, wspace=0.55, hspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_summary_rows(metrics: pd.DataFrame, gids: list[int], category: str) -> list[dict]:
    rows = []
    for gid in gids:
        m = metrics[metrics["trial_global_id"] == gid]
        if m.empty:
            continue
        r = m.iloc[0]
        rows.append(
            {
                "trial_global_id": int(gid),
                "condition": str(r["condition"]),
                "stimulus_id": str(r.get("stimulus_id", "")),
                "stimulus_text": str(r.get("stimulus_text", "")),
                "corr_loo": float(r["corr_loo"]),
                "rms_resid": float(r["rms_resid"]),
                "p99_abs_z": float(r["p99_abs_z"]),
                "frac_out_sib_clim": float(r["frac_out_sib_clim"]),
                "flag": str(r.get("flag", "") or ""),
                "why": str(r.get("why", "") or ""),
                "category": category,
            }
        )
    return rows


def main() -> None:
    register_mapgeog()
    cfg = _load_yaml(REPO / "configs/default.yaml")
    cfg.update(_load_yaml(REPO / "configs/windows/evoked_35_46_zscore.yaml"))

    metrics = pd.read_csv(METRICS_CSV)
    print(f"Loaded metrics: {METRICS_CSV} ({len(metrics)} rows)")

    df, stacks, gids_by_cond = load_session_maps(cfg)
    print(f"Loaded {len(df)} trial maps for {SESSION}")

    pattern_path = FIG_DIR / f"{FIG_STEM}.png"
    amp_path = FIG_DIR / f"{FIG_STEM}_amp_clim.png"

    plot_panel_grid(
        PATTERN_OUTLIERS,
        section_title="pattern outliers (low corr vs LOO siblings)",
        df=df,
        stacks=stacks,
        gids_by_cond=gids_by_cond,
        metrics=metrics,
        output_path=pattern_path,
    )
    print(f"Wrote {pattern_path}")

    plot_panel_grid(
        AMP_CLIM_ONLY,
        section_title="amplitude / shared-clim only (corr OK, high |z|)",
        df=df,
        stacks=stacks,
        gids_by_cond=gids_by_cond,
        metrics=metrics,
        output_path=amp_path,
    )
    print(f"Wrote {amp_path}")

    summary_rows = build_summary_rows(metrics, PATTERN_OUTLIERS, "pattern_outlier")
    summary_rows.extend(build_summary_rows(metrics, AMP_CLIM_ONLY, "amp_clim_only"))

    summary = {
        "session": SESSION,
        "window": WINDOW_TAG,
        "figures": {
            "pattern_outliers": str(pattern_path.relative_to(REPO)),
            "amp_clim_only": str(amp_path.relative_to(REPO)),
        },
        "trials": summary_rows,
    }

    json_path = FIG_DIR / f"{FIG_STEM}.json"
    csv_path = FIG_DIR / f"{FIG_STEM}.csv"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
