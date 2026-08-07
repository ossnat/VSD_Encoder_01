"""LOO full-FOV trial cleanliness metrics and classification.

Classification is computed within each ``(date, condition)`` group using
leave-one-out sibling statistics on window-mean z-scored maps.

Recommended storage
-------------------
Keep a standalone QC table at ``Data/VSD_Encoder_01/qc/trial_cleanliness_*.csv``
(keyed by ``trial_global_id``) rather than mutating
``all_trials_index_gandalf.csv`` (upstream FoundationData split artifact) or
window-specific encoding-pairs parquet. Join at training time::

    pairs = pd.read_parquet(...)
    qc = pd.read_csv(resolve_data_path("Data/VSD_Encoder_01/qc/..."))
    pairs = pairs.merge(qc[["trial_global_id", "trial_cleanliness"]], on="trial_global_id")
    pairs = pairs[pairs["trial_cleanliness"] == "good"]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.trial_frames import load_h5_mean_frame
from src.encoding.pairs import dedupe_stimulus_manifest
from src.paths import project_root, resolve_data_path
from src.stimuli.exclusions import EXCLUDED_H5_SESSIONS
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.schema import manifest_path as stimulus_manifest_path

CLEANLINESS_GOOD = "good"
CLEANLINESS_PATTERN = "pattern_outlier"
CLEANLINESS_AMP_EDGE = "amp_edge_outlier"
CLEANLINESS_VALUES = (
    CLEANLINESS_GOOD,
    CLEANLINESS_PATTERN,
    CLEANLINESS_AMP_EDGE,
)


def default_cleanliness_csv_path(
    *,
    monkey: str,
    window_id: str,
    repo: Path | None = None,
) -> Path:
    root = repo or project_root()
    return resolve_data_path(
        f"Data/VSD_Encoder_01/qc/trial_cleanliness_{monkey}__{window_id}.csv",
        root,
    )


def load_cleanliness_table(
    csv_path: Path | str,
    *,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Load a trial-cleanliness CSV (join key: ``trial_global_id``)."""
    root = repo or project_root()
    path = Path(csv_path)
    if not path.is_absolute():
        path = resolve_data_path(str(path), root)
    if not path.exists():
        # Fall back to repo-relative (e.g. checked-in fixtures).
        alt = root / csv_path
        path = alt if alt.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"Trial cleanliness CSV not found: {csv_path}")
    qc = pd.read_csv(path)
    if "trial_global_id" not in qc.columns:
        raise ValueError(f"{path}: missing required column 'trial_global_id'")
    if "trial_cleanliness" not in qc.columns:
        raise ValueError(f"{path}: missing required column 'trial_cleanliness'")
    out = qc[["trial_global_id", "trial_cleanliness"]].copy()
    out["trial_global_id"] = out["trial_global_id"].astype(int)
    out["trial_cleanliness"] = out["trial_cleanliness"].astype(str)
    # One label per trial; keep first if duplicates appear.
    return out.drop_duplicates(subset=["trial_global_id"], keep="first")


def filter_pairs_by_cleanliness(
    pairs: pd.DataFrame,
    *,
    csv_path: Path | str,
    keep: str | Iterable[str] = CLEANLINESS_GOOD,
    repo: Path | None = None,
    id_col: str = "trial_global_id",
    label_col: str = "trial_cleanliness",
    require_match: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter encoding-pair rows by cleanliness labels (join on trial id).

    Does **not** mutate FoundationData indexes or on-disk pair manifests.
    Unmatched ``trial_global_id`` values are dropped when ``require_match``
    is True (default); otherwise they are kept with a null label.
    """
    if id_col not in pairs.columns:
        raise ValueError(f"pairs missing join column {id_col!r}")
    keep_set = {str(x) for x in ([keep] if isinstance(keep, str) else list(keep))}
    if not keep_set:
        raise ValueError("keep must contain at least one cleanliness label")
    unknown = keep_set - set(CLEANLINESS_VALUES)
    if unknown:
        raise ValueError(
            f"Unknown cleanliness label(s) {sorted(unknown)}; "
            f"expected subset of {list(CLEANLINESS_VALUES)}"
        )

    qc = load_cleanliness_table(csv_path, repo=repo)
    before = len(pairs)
    merged = pairs.merge(qc, on=id_col, how="left", suffixes=("", "_qc"))
    # If pairs already had a cleanliness column, prefer the QC CSV join.
    if f"{label_col}_qc" in merged.columns:
        merged[label_col] = merged[f"{label_col}_qc"]
        merged = merged.drop(columns=[f"{label_col}_qc"])

    n_unmatched = int(merged[label_col].isna().sum())
    if require_match and n_unmatched:
        merged = merged[merged[label_col].notna()].copy()

    label_counts = (
        merged[label_col].value_counts(dropna=False).astype(int).to_dict()
        if not merged.empty
        else {}
    )
    kept = merged[merged[label_col].isin(keep_set)].copy()
    dropped = before - len(kept)
    stats: dict[str, Any] = {
        "csv_path": str(csv_path),
        "keep": sorted(keep_set),
        "n_before": int(before),
        "n_after": int(len(kept)),
        "n_dropped": int(dropped),
        "n_unmatched": int(n_unmatched),
        "label_counts_before_keep": {
            (str(k) if pd.notna(k) else "nan"): int(v) for k, v in label_counts.items()
        },
        "frac_kept": float(len(kept) / before) if before else float("nan"),
    }
    return kept.reset_index(drop=True), stats


def default_cleanliness_run_tag(keep: str | Iterable[str]) -> str:
    """Directory-safe run tag for cleanliness-filtered LOO outputs."""
    labels = [str(x) for x in ([keep] if isinstance(keep, str) else list(keep))]
    if labels == [CLEANLINESS_GOOD]:
        return "clean_good"
    token = "_".join(sorted(labels))
    return f"clean_{token}"


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def robust_z(x: np.ndarray, *, higher_is_outlier: bool) -> np.ndarray:
    """Signed robust z using MAD; positive values are more outlier-like."""
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


def amp_stats(z: np.ndarray) -> dict[str, float]:
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
    """Fraction of finite pixels outside 1–99% clim of sibling stack."""
    m_sib = np.isfinite(siblings)
    if m_sib.sum() < 10:
        return float("nan")
    lo, hi = np.percentile(siblings[m_sib], [1, 99])
    m = np.isfinite(trial)
    if m.sum() == 0:
        return float("nan")
    v = trial[m]
    return float(np.mean((v < lo) | (v > hi)))


def _metric_directions() -> dict[str, bool]:
    return {
        "corr_loo": False,
        "rms_resid": True,
        "max_abs_z": True,
        "p99_abs_z": True,
        "frac_abs_gt3": True,
        "frac_abs_gt5": True,
        "spatial_std": True,
        "frac_out_sib_clim": True,
    }


def _load_trial_maps(
    group: pd.DataFrame,
    *,
    load_kw: dict[str, Any],
) -> np.ndarray:
    maps = []
    for row in group.itertuples(index=False):
        maps.append(
            load_h5_mean_frame(
                target_file=str(row.target_file),
                trial_global_id=int(row.trial_global_id),
                **load_kw,
            )
        )
    return np.stack(maps, axis=0).astype(np.float64)


def compute_loo_metrics_for_group(
    group: pd.DataFrame,
    *,
    load_kw: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute LOO metrics for one ``(date, condition)`` trial group."""
    group = group.sort_values("trial_global_id").reset_index(drop=True)
    n = len(group)
    stack = _load_trial_maps(group, load_kw=load_kw)
    total = np.nansum(stack, axis=0)
    finite_counts = np.sum(np.isfinite(stack), axis=0)

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(group.itertuples(index=False)):
        trial = stack[i]
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
                "trial_global_id": int(row.trial_global_id),
                "date": str(row.date),
                "condition": str(row.condition),
                "n_siblings": n - 1,
                "n_condition": n,
                "corr_loo": spatial_corr(trial, loo),
                "rms_resid": rms_resid(trial, loo),
                **stats,
                "frac_out_sib_clim": frac_outside_sibling_clim(trial, sib_stack),
            }
        )
    return rows


def compute_loo_metrics_for_groups(
    trials: pd.DataFrame,
    *,
    load_kw: dict[str, Any],
    group_cols: Iterable[str] = ("date", "condition"),
    progress: bool = True,
) -> pd.DataFrame:
    """Score all trials with LOO metrics within ``group_cols``."""
    rows: list[dict[str, Any]] = []
    grouped = trials.groupby(list(group_cols), sort=True)
    sessions = sorted(trials["date"].astype(str).unique())
    done_sessions: set[str] = set()

    for key, group in grouped:
        date = str(group.iloc[0]["date"])
        if progress and date not in done_sessions:
            n_sess = int((trials["date"].astype(str) == date).sum())
            print(f"Session {date}: scoring {n_sess} trials…")
            done_sessions.add(date)
        rows.extend(compute_loo_metrics_for_group(group, load_kw=load_kw))

    return pd.DataFrame(rows)


def add_within_group_scores(
    metrics: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("date", "condition"),
) -> pd.DataFrame:
    out = metrics.copy()
    metric_dirs = _metric_directions()
    for col, higher in metric_dirs.items():
        zcol = f"rz_{col}"
        out[zcol] = np.nan
        for _, idx in out.groupby(list(group_cols)).groups.items():
            vals = out.loc[idx, col].to_numpy(dtype=float)
            out.loc[idx, zcol] = robust_z(vals, higher_is_outlier=higher)

    out["rank_corr_low"] = out.groupby(list(group_cols))["corr_loo"].rank(
        method="min", ascending=True
    )
    out["rank_rms_high"] = out.groupby(list(group_cols))["rms_resid"].rank(
        method="min", ascending=False
    )
    out["pct_corr_low"] = out.groupby(list(group_cols))["corr_loo"].rank(
        method="average", pct=True, ascending=True
    )
    out["pct_rms_high"] = out.groupby(list(group_cols))["rms_resid"].rank(
        method="average", pct=True, ascending=False
    )
    return out


def _pattern_reason(r: pd.Series) -> str:
    bits: list[str] = []
    if r["pct_corr_low"] <= 0.15:
        bits.append(
            f"corr bottom {100 * r['pct_corr_low']:.0f}% "
            f"(rank {int(r['rank_corr_low'])}/{int(r['n_condition'])})"
        )
    if r["rz_corr_loo"] >= 2.0:
        bits.append(f"corr rz={r['rz_corr_loo']:.1f}")
    return "; ".join(bits)


def _amp_reason(r: pd.Series) -> str:
    bits: list[str] = []
    if r["rz_rms_resid"] >= 2.0:
        bits.append(f"rms rz={r['rz_rms_resid']:.1f}")
    if r["rank_rms_high"] <= 3:
        bits.append(f"rms rank {int(r['rank_rms_high'])}")
    if r["rz_p99_abs_z"] >= 2.0:
        bits.append(f"p99|z| rz={r['rz_p99_abs_z']:.1f}")
    if r["rz_max_abs_z"] >= 2.0:
        bits.append(f"max|z| rz={r['rz_max_abs_z']:.1f}")
    if r["rz_frac_abs_gt5"] >= 2.0:
        bits.append(f"frac>|5| rz={r['rz_frac_abs_gt5']:.1f}")
    if r["rz_frac_out_sib_clim"] >= 2.0:
        bits.append(
            f"out_clim={r['frac_out_sib_clim']:.3f} "
            f"rz={r['rz_frac_out_sib_clim']:.1f}"
        )
    return "; ".join(bits)


def classify_trial_cleanliness(
    metrics: pd.DataFrame,
    *,
    pattern_pct_threshold: float = 0.15,
    pattern_rz_threshold: float = 2.0,
    pattern_corr_max: float = 0.40,
    amp_rz_threshold: float = 2.0,
    amp_frac_out_clim_min: float = 0.05,
) -> pd.DataFrame:
    """Assign ``trial_cleanliness`` and ``flag_reason`` from scored metrics.

    Pattern outliers require a low LOO spatial correlation that is either
    MAD-extreme (``rz_corr_loo``) or in the bottom ``pattern_pct_threshold``
    within date×condition **and** below ``pattern_corr_max`` (avoids labeling
    merely relatively-weak but still well-aligned trials).

    Amplitude/edge outliers are trials that are *not* pattern outliers but
    sit in the high amplitude / shared-clim tail (robust-z on rms, p99|z|,
    frac>|5|, or frac outside sibling clim), with ``frac_out_sib_clim`` at
    least ``amp_frac_out_clim_min`` (or a strong p99|z| rz).
    """
    out = metrics.copy()
    cleanliness: list[str] = []
    reasons: list[str] = []

    for _, r in out.iterrows():
        low_corr = (r["rz_corr_loo"] >= pattern_rz_threshold) or (
            (r["pct_corr_low"] <= pattern_pct_threshold)
            and (r["corr_loo"] < pattern_corr_max)
        )
        high_amp = (
            (r["rz_rms_resid"] >= amp_rz_threshold)
            or (r["rz_p99_abs_z"] >= amp_rz_threshold)
            or (r["rz_max_abs_z"] >= amp_rz_threshold)
            or (r["rz_frac_abs_gt5"] >= amp_rz_threshold)
            or (r["rz_frac_out_sib_clim"] >= amp_rz_threshold)
        )

        if low_corr:
            cleanliness.append(CLEANLINESS_PATTERN)
            reasons.append(_pattern_reason(r) or "low corr_loo vs LOO siblings")
        elif high_amp and (
            r["frac_out_sib_clim"] >= amp_frac_out_clim_min
            or r["rz_p99_abs_z"] >= (amp_rz_threshold + 0.5)
        ):
            cleanliness.append(CLEANLINESS_AMP_EDGE)
            reasons.append(_amp_reason(r) or "high amplitude vs LOO siblings")
        else:
            cleanliness.append(CLEANLINESS_GOOD)
            reasons.append("")

    out["trial_cleanliness"] = cleanliness
    out["flag_reason"] = reasons
    return out


def attach_stimulus_metadata(
    metrics: pd.DataFrame,
    *,
    repo: Path,
    cfg: dict[str, Any],
    monkey: str,
) -> pd.DataFrame:
    stimuli_root = resolve_data_path(cfg["paths"]["stimuli_root"], repo)
    stim_path = stimulus_manifest_path(stimuli_root, monkey)
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
    out = metrics.merge(
        stim[stim_cols],
        left_on=["date", "condition"],
        right_on=["h5_session", "condition"],
        how="left",
    )
    if "h5_session" in out.columns:
        out = out.drop(columns=["h5_session"])
    return out


def load_trials_for_cleanliness(
    *,
    repo: Path,
    cfg: dict[str, Any],
    exclude_sessions: frozenset[str] = EXCLUDED_H5_SESSIONS,
) -> pd.DataFrame:
    index_path = resolve_data_path(cfg["trials_index_csv"], repo)
    trials = pd.read_csv(index_path)
    trials["date"] = trials["date"].astype(str)
    trials["condition"] = trials["condition"].astype(str)
    if exclude_sessions:
        trials = trials[~trials["date"].isin(exclude_sessions)].copy()
    return trials.sort_values(["date", "condition", "trial_global_id"]).reset_index(
        drop=True
    )


def build_load_kwargs(cfg: dict[str, Any], *, repo: Path) -> dict[str, Any]:
    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    return dict(
        repo=repo,
        spatial_size=spatial_size,
        start_frame=int(cfg["start_frame"]),
        end_frame=int(cfg["end_frame"]),
        avg_method=str(cfg.get("avg_method", "mean")),
        normalization=str(cfg["normalization"]),
        baseline_start_frame=int(cfg.get("baseline_start_frame", 5)),
        baseline_end_frame=int(cfg.get("baseline_end_frame", 26)),
        baseline_std_eps=float(cfg.get("baseline_std_eps", 1e-8)),
    )
