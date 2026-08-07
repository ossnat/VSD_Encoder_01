"""One-off: LOO full-FOV messy-outlier scores for session 100718a."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.trial_frames import load_h5_mean_frame
from src.encoding.pairs import dedupe_stimulus_manifest
from src.paths import project_root, resolve_data_path
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.schema import manifest_path as stimulus_manifest_path

SESSION = "100718a"
REPO = project_root()


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _robust_z(x: np.ndarray, *, higher_is_outlier: bool) -> np.ndarray:
    """Signed robust z using MAD; direction chosen so positive = more outlier-like."""
    med = float(np.median(x))
    mad = _mad(x)
    scale = 1.4826 * mad if mad > 1e-12 else (float(np.std(x)) or 1.0)
    z = (x - med) / scale
    return z if higher_is_outlier else -z


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


def rms_resid(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d)))


def amp_stats(z: np.ndarray) -> dict:
    m = np.isfinite(z)
    v = z[m]
    if v.size == 0:
        return {
            "max_abs_z": np.nan,
            "p99_abs_z": np.nan,
            "frac_abs_gt3": np.nan,
            "frac_abs_gt5": np.nan,
            "spatial_std": np.nan,
        }
    abs_v = np.abs(v)
    return {
        "max_abs_z": float(abs_v.max()),
        "p99_abs_z": float(np.percentile(abs_v, 99)),
        "frac_abs_gt3": float(np.mean(abs_v > 3)),
        "frac_abs_gt5": float(np.mean(abs_v > 5)),
        "spatial_std": float(np.std(v)),
    }


def frac_outside_sibling_clim(trial: np.ndarray, siblings: np.ndarray) -> float:
    """Fraction of finite pixels outside 1–99% clim of sibling stack (shared clim)."""
    m_sib = np.isfinite(siblings)
    if m_sib.sum() < 10:
        return float("nan")
    lo, hi = np.percentile(siblings[m_sib], [1, 99])
    m = np.isfinite(trial)
    if m.sum() == 0:
        return float("nan")
    v = trial[m]
    return float(np.mean((v < lo) | (v > hi)))


def main() -> None:
    cfg = _load_yaml(REPO / "configs/default.yaml")
    cfg.update(_load_yaml(REPO / "configs/windows/evoked_35_46_zscore.yaml"))

    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    index_path = resolve_data_path(cfg["trials_index_csv"], REPO)
    df = pd.read_csv(index_path)
    df = df[df["date"].astype(str) == SESSION].copy()
    df = df.sort_values(["condition", "trial_global_id"]).reset_index(drop=True)

    stimuli_root = resolve_data_path(cfg["paths"]["stimuli_root"], REPO)
    stim_path = stimulus_manifest_path(stimuli_root, cfg["monkey"])
    stim = dedupe_stimulus_manifest(pd.read_parquet(stim_path))
    stim = attach_stimulus_ids(stim)
    stim_cols = [
        c
        for c in [
            "h5_session",
            "condition",
            "stimulus_id",
            "stimulus_text",
            "shape_type",
            "color",
            "size_deg",
            "is_blank",
        ]
        if c in stim.columns
    ]
    df = df.merge(
        stim[stim_cols],
        left_on=["date", "condition"],
        right_on=["h5_session", "condition"],
        how="left",
    )

    print(f"Session {SESSION}: {len(df)} trials")
    print(df["condition"].value_counts().sort_index().to_string())
    print()

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

    rows: list[dict] = []
    for cond, g in df.groupby("condition", sort=True):
        g = g.reset_index(drop=True)
        n = len(g)
        print(f"Loading {cond} (n={n})...")
        maps = []
        for r in g.itertuples(index=False):
            maps.append(
                load_h5_mean_frame(
                    target_file=str(r.target_file),
                    trial_global_id=int(r.trial_global_id),
                    **load_kw,
                )
            )
        stack = np.stack(maps, axis=0).astype(np.float64)  # (n,H,W)
        total = np.nansum(stack, axis=0)
        finite_counts = np.sum(np.isfinite(stack), axis=0)

        for i, r in enumerate(g.itertuples(index=False)):
            trial = stack[i]
            # LOO mean over finite siblings
            count_loo = finite_counts - np.isfinite(trial).astype(np.int64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                loo = (total - np.where(np.isfinite(trial), trial, 0.0)) / np.maximum(
                    count_loo, 1
                )
            loo = np.where(count_loo > 0, loo, np.nan)

            sib_stack = np.concatenate([stack[:i], stack[i + 1 :]], axis=0)
            stats = amp_stats(trial)
            rows.append(
                {
                    "trial_global_id": int(r.trial_global_id),
                    "condition": cond,
                    "stimulus_id": getattr(r, "stimulus_id", None),
                    "stimulus_text": getattr(r, "stimulus_text", None),
                    "n_siblings": n - 1,
                    "n_condition": n,
                    "corr_loo": spatial_corr(trial, loo),
                    "rms_resid": rms_resid(trial, loo),
                    **stats,
                    "frac_out_sib_clim": frac_outside_sibling_clim(trial, sib_stack),
                }
            )

    out = pd.DataFrame(rows)

    # Within-condition robust z (positive = more outlier-like)
    metric_dirs = {
        "corr_loo": False,  # low corr = outlier
        "rms_resid": True,
        "max_abs_z": True,
        "p99_abs_z": True,
        "frac_abs_gt3": True,
        "frac_abs_gt5": True,
        "spatial_std": True,
        "frac_out_sib_clim": True,
    }
    for col, higher in metric_dirs.items():
        zcol = f"rz_{col}"
        out[zcol] = np.nan
        for cond, idx in out.groupby("condition").groups.items():
            vals = out.loc[idx, col].to_numpy(dtype=float)
            out.loc[idx, zcol] = _robust_z(vals, higher_is_outlier=higher)

    # Rank within condition (1 = most outlier-like)
    out["rank_corr_low"] = out.groupby("condition")["corr_loo"].rank(
        method="min", ascending=True
    )
    out["rank_rms_high"] = out.groupby("condition")["rms_resid"].rank(
        method="min", ascending=False
    )
    out["pct_corr_low"] = out.groupby("condition")["corr_loo"].rank(
        method="average", pct=True, ascending=True
    )
    out["pct_rms_high"] = out.groupby("condition")["rms_resid"].rank(
        method="average", pct=True, ascending=False
    )

    # Flagging rule (transparent):
    # A) corr among bottom 15% within condition OR rz_corr_loo >= 2.5
    # B) AND (rms rz>=2.5 OR amp rz on p99/max/frac_gt5 >= 2.5 OR top-2 rms)
    # Also soft-flag amplitude-only extremes for clim discussion.
    def flag_row(r: pd.Series) -> str:
        low_corr = (r["pct_corr_low"] <= 0.15) or (r["rz_corr_loo"] >= 2.5)
        high_resid = (
            (r["rz_rms_resid"] >= 2.5)
            or (r["rank_rms_high"] <= 2)
            or (r["rz_p99_abs_z"] >= 2.5)
            or (r["rz_max_abs_z"] >= 2.5)
            or (r["rz_frac_abs_gt5"] >= 2.5)
        )
        amp_only = (
            (r["rz_p99_abs_z"] >= 2.5)
            or (r["rz_max_abs_z"] >= 2.5)
            or (r["rz_frac_out_sib_clim"] >= 2.5)
        ) and not low_corr
        if low_corr and high_resid:
            return "outlier"
        if low_corr and (r["rz_rms_resid"] >= 1.5 or r["rank_rms_high"] <= 3):
            return "outlier_soft"
        if amp_only and r["frac_out_sib_clim"] >= 0.05:
            return "amp_clim_only"
        return ""

    out["flag"] = out.apply(flag_row, axis=1)

    # Why string for flagged
    def why(r: pd.Series) -> str:
        bits = []
        if r["pct_corr_low"] <= 0.15:
            bits.append(f"corr bottom {100*r['pct_corr_low']:.0f}% (rank {int(r['rank_corr_low'])}/{int(r['n_condition'])})")
        if r["rz_corr_loo"] >= 2.5:
            bits.append(f"corr rz={r['rz_corr_loo']:.1f}")
        if r["rz_rms_resid"] >= 1.5:
            bits.append(f"rms rz={r['rz_rms_resid']:.1f}")
        if r["rank_rms_high"] <= 3:
            bits.append(f"rms rank {int(r['rank_rms_high'])}")
        if r["rz_p99_abs_z"] >= 2.5:
            bits.append(f"p99|z| rz={r['rz_p99_abs_z']:.1f}")
        if r["rz_max_abs_z"] >= 2.5:
            bits.append(f"max|z| rz={r['rz_max_abs_z']:.1f}")
        if r["rz_frac_abs_gt5"] >= 2.5:
            bits.append(f"frac>|5| rz={r['rz_frac_abs_gt5']:.1f}")
        if r["rz_frac_out_sib_clim"] >= 2.0:
            bits.append(f"out_clim={r['frac_out_sib_clim']:.3f} rz={r['rz_frac_out_sib_clim']:.1f}")
        return "; ".join(bits)

    out["why"] = out.apply(why, axis=1)

    # Save full table
    out_path = REPO / "scripts/_tmp_messy_outliers_100718a.csv"
    out.sort_values(["condition", "corr_loo"]).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Focus: gid 2999
    print("\n=== GID 2999 (condAN1) ===")
    r2999 = out[out["trial_global_id"] == 2999]
    if r2999.empty:
        print("NOT FOUND")
    else:
        r = r2999.iloc[0]
        print(
            f"condition={r['condition']} stim={r['stimulus_id']} n={r['n_condition']}\n"
            f"corr_loo={r['corr_loo']:.4f}  rank_low={int(r['rank_corr_low'])}/{int(r['n_condition'])}  "
            f"pct_low={r['pct_corr_low']:.3f}  rz={r['rz_corr_loo']:.2f}\n"
            f"rms_resid={r['rms_resid']:.4f}  rank_high={int(r['rank_rms_high'])}/{int(r['n_condition'])}  "
            f"rz={r['rz_rms_resid']:.2f}\n"
            f"max|z|={r['max_abs_z']:.2f} rz={r['rz_max_abs_z']:.2f}  "
            f"p99|z|={r['p99_abs_z']:.2f} rz={r['rz_p99_abs_z']:.2f}\n"
            f"frac>|3|={r['frac_abs_gt3']:.4f}  frac>|5|={r['frac_abs_gt5']:.4f}  "
            f"spatial_std={r['spatial_std']:.3f}\n"
            f"frac_out_sib_clim={r['frac_out_sib_clim']:.4f} rz={r['rz_frac_out_sib_clim']:.2f}\n"
            f"flag={r['flag']!r}  why={r['why']}"
        )

    # CondAN1 ranked table (top messy by low corr)
    print("\n=== condAN1 ranked by lowest corr_loo (all trials) ===")
    a1 = out[out["condition"] == "condAN1"].sort_values("corr_loo")
    cols = [
        "trial_global_id",
        "stimulus_id",
        "corr_loo",
        "rank_corr_low",
        "rms_resid",
        "rank_rms_high",
        "p99_abs_z",
        "frac_abs_gt5",
        "frac_out_sib_clim",
        "rz_corr_loo",
        "rz_rms_resid",
        "flag",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(a1[cols].to_string(index=False))

    # Proposed outliers
    print("\n=== Proposed outliers (flag=outlier or outlier_soft) ===")
    flagged = out[out["flag"].isin(["outlier", "outlier_soft"])].sort_values(
        ["flag", "rz_corr_loo", "rz_rms_resid"], ascending=[True, False, False]
    )
    show = [
        "trial_global_id",
        "condition",
        "stimulus_id",
        "n_siblings",
        "corr_loo",
        "rms_resid",
        "p99_abs_z",
        "frac_out_sib_clim",
        "flag",
        "why",
    ]
    if flagged.empty:
        print("(none)")
    else:
        print(flagged[show].to_string(index=False))

    print("\n=== Amplitude/clim-only flags (messy look without pattern mismatch) ===")
    amp = out[out["flag"] == "amp_clim_only"].sort_values(
        "frac_out_sib_clim", ascending=False
    )
    if amp.empty:
        print("(none)")
    else:
        print(amp[show].to_string(index=False))

    # Per-condition summary of corr distribution
    print("\n=== Per-condition corr_loo summary ===")
    for cond, g in out.groupby("condition"):
        print(
            f"{cond}: n={len(g)}  corr median={g['corr_loo'].median():.3f}  "
            f"min={g['corr_loo'].min():.3f}  p10={g['corr_loo'].quantile(0.1):.3f}  "
            f"rms median={g['rms_resid'].median():.3f}  max={g['rms_resid'].max():.3f}"
        )

    # Top 3 lowest corr per condition for transparency
    print("\n=== Lowest-3 corr per condition ===")
    for cond, g in out.groupby("condition"):
        top = g.nsmallest(3, "corr_loo")
        for _, r in top.iterrows():
            print(
                f"  {cond} gid={int(r['trial_global_id'])} corr={r['corr_loo']:.3f} "
                f"rms={r['rms_resid']:.3f} p99={r['p99_abs_z']:.2f} "
                f"out_clim={r['frac_out_sib_clim']:.3f} flag={r['flag']!r}"
            )


if __name__ == "__main__":
    main()
