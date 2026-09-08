"""Leave-one-out fold construction for encoding experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import yaml

from src.stimuli.identity import attach_stimulus_ids

Protocol = Literal["A", "B", "C"]


DEFAULT_HELDOUT = [
    "white_point_0.1",
    "black_triangle_contour_0.4",
    "black_bar_vertical_1",
    "black_bar_horizontal_1",
    "letter_A_white_1",
    "letter_D_white_1",
    "letter_F_white_1",
    "letter_G_white_1",
    "letter_L_white_1",
    "letter_N_white_1",
]


@dataclass(frozen=True)
class FoldSpec:
    protocol: Protocol
    fold_id: str
    heldout_stimulus_id: str
    heldout_date: str | None
    heldout_condition: str | None
    n_train: int
    n_val: int
    n_test: int
    leakage_ok: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeldoutConfig:
    """Held-out stimulus ids plus optional train-only session dates."""

    stimulus_ids: list[str]
    train_only_sessions: list[str]


def load_heldout_config(path: Path | None) -> HeldoutConfig:
    """
    Load a held-out YAML (or the built-in default list).

    Dict keys:
      ``heldout_stimulus_ids`` / ``heldouts`` — ids or glob patterns
        (e.g. ``letter_*``), expanded later against encoding-pair ids.
      ``train_only_sessions`` / ``train_only_dates`` — session dates that
        may appear in train/val but must not become Protocol A/C test folds.
    A bare YAML list is treated as stimulus ids only.
    """
    if path is None:
        return HeldoutConfig(list(DEFAULT_HELDOUT), [])
    with path.open() as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return HeldoutConfig([str(x) for x in data], [])
    if isinstance(data, dict):
        items = data.get("heldout_stimulus_ids") or data.get("heldouts") or []
        train_only = (
            data.get("train_only_sessions") or data.get("train_only_dates") or []
        )
        return HeldoutConfig(
            [str(x) for x in items],
            [str(x) for x in train_only],
        )
    raise ValueError(f"Unrecognized held-out list format: {path}")


def load_heldout_list(path: Path | None) -> list[str]:
    return load_heldout_config(path).stimulus_ids


_GLOB_CHARS = frozenset("*?[")


def expand_stimulus_id_patterns(
    patterns: Sequence[str],
    available_ids: Sequence[str],
) -> list[str]:
    """
    Expand glob patterns (e.g. ``letter_*``) against available stimulus ids.

    Exact ids (no glob characters) pass through even if currently missing;
    fold builders skip empty stimuli. A glob that matches nothing raises.
    """
    available = [str(x) for x in available_ids if x]
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        token = str(pat)
        if any(ch in token for ch in _GLOB_CHARS):
            hits = sorted(sid for sid in available if fnmatch(sid, token))
            if not hits:
                raise ValueError(
                    f"No stimulus_id matched pattern {token!r} "
                    f"(available n={len(available)})"
                )
        else:
            hits = [token]
        for sid in hits:
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def filter_folds_excluding_train_only_dates(
    folds: list[tuple[FoldSpec, pd.DataFrame]],
    train_only_dates: Sequence[str],
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """Drop folds whose ``heldout_date`` is a train-only session."""
    skip = {str(d) for d in train_only_dates if str(d).strip()}
    if not skip:
        return folds
    return [
        (spec, fold_df)
        for spec, fold_df in folds
        if spec.heldout_date is None or str(spec.heldout_date) not in skip
    ]


def _session_key_series(df: pd.DataFrame) -> pd.Series:
    """Prefer ``h5_session``, else ``date``, as the session-date key."""
    if "h5_session" in df.columns:
        return df["h5_session"].astype(str)
    if "date" in df.columns:
        return df["date"].astype(str)
    raise KeyError("fold frame needs h5_session or date for train-only filtering")


def strip_train_only_from_protocol_b_test(
    folds: list[tuple[FoldSpec, pd.DataFrame]],
    train_only_sessions: Sequence[str],
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """
    Protocol B: drop train-only session rows from **test** only.

    Held-out ``stimulus_id`` trials on those sessions are removed from the fold
    entirely (they are never in train/val under Protocol B). Other stimuli on
    train-only sessions remain in train/val.
    """
    skip = {str(d) for d in train_only_sessions if str(d).strip()}
    if not skip:
        return folds
    out: list[tuple[FoldSpec, pd.DataFrame]] = []
    for spec, fold_df in folds:
        sess = _session_key_series(fold_df)
        drop = (fold_df["loo_split"] == "test") & sess.isin(skip)
        if not bool(drop.any()):
            out.append((spec, fold_df))
            continue
        kept = fold_df.loc[~drop].copy()
        n_test = int((kept["loo_split"] == "test").sum())
        if n_test == 0:
            continue
        note = spec.notes or ""
        extra = f"stripped train-only test sessions {sorted(skip)}"
        notes = f"{note}; {extra}" if note else extra
        out.append(
            (
                FoldSpec(
                    protocol=spec.protocol,
                    fold_id=spec.fold_id,
                    heldout_stimulus_id=spec.heldout_stimulus_id,
                    heldout_date=spec.heldout_date,
                    heldout_condition=spec.heldout_condition,
                    n_train=int((kept["loo_split"] == "train").sum()),
                    n_val=int((kept["loo_split"] == "val").sum()),
                    n_test=n_test,
                    leakage_ok=spec.leakage_ok,
                    notes=notes,
                ),
                kept,
            )
        )
    return out


def _inner_train_val_split(
    remainder: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split remainder into train/val at (date, condition) group level.

    Prefer existing split labels when both train and val are present in the
    remainder; otherwise carve a reproducible group holdout.
    """
    rem = remainder.copy()
    if rem.empty:
        return rem, rem.iloc[0:0].copy()

    has_train = (rem["split"] == "train").any()
    has_val = (rem["split"] == "val").any()
    if has_train and has_val:
        train = rem[rem["split"] == "train"].copy()
        val = rem[rem["split"] == "val"].copy()
        # Drop any residual test labels from remainder into train.
        extra = rem[~rem["split"].isin(["train", "val"])].copy()
        if not extra.empty:
            train = pd.concat([train, extra], ignore_index=True)
        return train, val

    groups = (
        rem[["date", "condition"]]
        .drop_duplicates()
        .sort_values(["date", "condition"])
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    n_val_groups = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 else 0
    if n_val_groups == 0:
        out = rem.copy()
        out["loo_split"] = "train"
        return out, rem.iloc[0:0].copy()

    val_idx = set(
        rng.choice(len(groups), size=n_val_groups, replace=False).tolist()
    )
    val_keys = {
        (str(groups.loc[i, "date"]), str(groups.loc[i, "condition"]))
        for i in val_idx
    }
    key = list(zip(rem["date"].astype(str), rem["condition"].astype(str)))
    is_val = [k in val_keys for k in key]
    val = rem.loc[is_val].copy()
    train = rem.loc[[not v for v in is_val]].copy()
    return train, val


def _assign_loo_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name, df in (("train", train), ("val", val), ("test", test)):
        if df.empty:
            continue
        chunk = df.copy()
        chunk["loo_split"] = name
        parts.append(chunk)
    if not parts:
        raise ValueError("Empty fold: no train/val/test rows")
    return pd.concat(parts, ignore_index=True)


def audit_protocol_a_leakage(
    fold_df: pd.DataFrame, heldout_stimulus_id: str
) -> tuple[bool, str]:
    """
    Protocol A leakage check: train/val must not contain the held-out
    (date, condition) group, but may contain other sessions of the same
    stimulus_id.
    """
    test = fold_df[fold_df["loo_split"] == "test"]
    if test.empty:
        return False, "empty test"
    test_keys = set(
        zip(test["date"].astype(str), test["condition"].astype(str))
    )
    rem = fold_df[fold_df["loo_split"].isin(["train", "val"])]
    leak_keys = set(
        zip(rem["date"].astype(str), rem["condition"].astype(str))
    ) & test_keys
    if leak_keys:
        return False, f"train/val contains held-out (date,condition): {sorted(leak_keys)}"
    n_same_stim_in_rem = int((rem["stimulus_id"] == heldout_stimulus_id).sum())
    note = (
        f"same stimulus_id in train/val trials={n_same_stim_in_rem} "
        "(expected for protocol A)"
    )
    return True, note


def audit_protocol_b_leakage(
    fold_df: pd.DataFrame, heldout_stimulus_id: str
) -> tuple[bool, str]:
    """Protocol B: no train/val trial may share the held-out stimulus_id."""
    rem = fold_df[fold_df["loo_split"].isin(["train", "val"])]
    n_leak = int((rem["stimulus_id"] == heldout_stimulus_id).sum())
    if n_leak:
        return False, f"stimulus_id leakage into train/val: n={n_leak}"
    return True, "no stimulus_id in train/val"


def audit_protocol_c_leakage(
    fold_df: pd.DataFrame,
    heldout_stimulus_id: str,
    *,
    heldout_date: str,
    heldout_condition: str,
) -> tuple[bool, str]:
    """
    Protocol C leakage check: test is the held-out (date, condition); train/val
    must not contain that group and must not contain any trial with the
    held-out stimulus_id (other sessions of the same stimulus are excluded).
    """
    ok_keys, note_keys = audit_protocol_a_leakage(fold_df, heldout_stimulus_id)
    if not ok_keys:
        return False, note_keys
    ok_stim, note_stim = audit_protocol_b_leakage(fold_df, heldout_stimulus_id)
    if not ok_stim:
        return False, note_stim
    test = fold_df[fold_df["loo_split"] == "test"]
    test_keys = set(
        zip(test["date"].astype(str), test["condition"].astype(str))
    )
    expected = {(str(heldout_date), str(heldout_condition))}
    if test_keys != expected:
        return False, f"test keys {sorted(test_keys)} != {sorted(expected)}"
    return True, "no stimulus_id in train/val; test is held-out (date,condition)"


def build_protocol_a_folds(
    pairs: pd.DataFrame,
    heldout_ids: list[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """
    Protocol A — condition LOO.

    For each held-out stimulus_id, create one fold per (date, condition)
    matching that stimulus: that group is test; remainder is train/val.
    """
    df = attach_stimulus_ids(pairs)
    folds: list[tuple[FoldSpec, pd.DataFrame]] = []
    for sid in heldout_ids:
        stim = df[df["stimulus_id"] == sid]
        if stim.empty:
            continue
        groups = (
            stim[["date", "condition"]]
            .drop_duplicates()
            .sort_values(["date", "condition"])
        )
        for i, row in enumerate(groups.itertuples(index=False)):
            date, condition = str(row.date), str(row.condition)
            test = df[(df["date"] == date) & (df["condition"] == condition)].copy()
            rem = df[~((df["date"] == date) & (df["condition"] == condition))].copy()
            train, val = _inner_train_val_split(
                rem, val_fraction=val_fraction, seed=seed + i
            )
            fold_df = _assign_loo_split(train, val, test)
            fold_id = f"A__{sid}__{date}_{condition}"
            ok, note = audit_protocol_a_leakage(fold_df, sid)
            spec = FoldSpec(
                protocol="A",
                fold_id=fold_id,
                heldout_stimulus_id=sid,
                heldout_date=date,
                heldout_condition=condition,
                n_train=int((fold_df["loo_split"] == "train").sum()),
                n_val=int((fold_df["loo_split"] == "val").sum()),
                n_test=int((fold_df["loo_split"] == "test").sum()),
                leakage_ok=ok,
                notes=note,
            )
            folds.append((spec, fold_df))
    return folds


def select_one_fold_per_stimulus(
    folds: list[tuple[FoldSpec, pd.DataFrame]],
    *,
    seed: int = 17,
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """
    Keep exactly one fold per ``heldout_stimulus_id``.

    Within each stimulus, candidates are sorted by ``fold_id`` then one index
    is drawn with ``numpy.random.default_rng(seed)`` so the choice is
    reproducible (document the seed alongside results).
    """
    by_sid: dict[str, list[tuple[FoldSpec, pd.DataFrame]]] = {}
    order: list[str] = []
    for item in folds:
        sid = item[0].heldout_stimulus_id
        if sid not in by_sid:
            by_sid[sid] = []
            order.append(sid)
        by_sid[sid].append(item)

    rng = np.random.default_rng(seed)
    selected: list[tuple[FoldSpec, pd.DataFrame]] = []
    for sid in order:
        candidates = sorted(by_sid[sid], key=lambda x: x[0].fold_id)
        idx = int(rng.integers(0, len(candidates)))
        selected.append(candidates[idx])
    return selected


def build_protocol_b_folds(
    pairs: pd.DataFrame,
    heldout_ids: list[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 17,
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """
    Protocol B — stimulus LOO.

    Entire stimulus_id out of train/val; all of its trials are test.
    """
    df = attach_stimulus_ids(pairs)
    folds: list[tuple[FoldSpec, pd.DataFrame]] = []
    for i, sid in enumerate(heldout_ids):
        test = df[df["stimulus_id"] == sid].copy()
        if test.empty:
            continue
        rem = df[df["stimulus_id"] != sid].copy()
        train, val = _inner_train_val_split(
            rem, val_fraction=val_fraction, seed=seed + i
        )
        fold_df = _assign_loo_split(train, val, test)
        fold_id = f"B__{sid}"
        ok, note = audit_protocol_b_leakage(fold_df, sid)
        spec = FoldSpec(
            protocol="B",
            fold_id=fold_id,
            heldout_stimulus_id=sid,
            heldout_date=None,
            heldout_condition=None,
            n_train=int((fold_df["loo_split"] == "train").sum()),
            n_val=int((fold_df["loo_split"] == "val").sum()),
            n_test=int((fold_df["loo_split"] == "test").sum()),
            leakage_ok=ok,
            notes=note,
        )
        folds.append((spec, fold_df))
    return folds


PROTOCOL_C_TEST_SELECTION_RULE = (
    "One fold per held-out stimulus_id. Test is a single (date, condition) "
    "chosen as the first row after sorting groups by (date, condition). "
    "Identical stimulus instances share the same y_hat; other sessions of the "
    "held-out stimulus are omitted from the fold (not test)."
)


def _pick_protocol_c_test_group(stim: pd.DataFrame) -> tuple[str, str]:
    """Return the single (date, condition) used as Protocol C test."""
    groups = (
        stim[["date", "condition"]]
        .drop_duplicates()
        .sort_values(["date", "condition"])
    )
    if groups.empty:
        raise ValueError("stimulus has no (date, condition) groups")
    row = groups.iloc[0]
    return str(row["date"]), str(row["condition"])


def build_protocol_c_folds(
    pairs: pd.DataFrame,
    heldout_ids: list[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 17,
    all_sessions: bool = False,
) -> list[tuple[FoldSpec, pd.DataFrame]]:
    """
    Protocol C — condition LOO without same-stimulus train contamination.

    Default: **one fold per held-out stimulus_id** (~20 folds). Test is one
    ``(date, condition)`` per stimulus (see ``PROTOCOL_C_TEST_SELECTION_RULE``).
    Train/val exclude *all* trials with the held-out ``stimulus_id``.

    Set ``all_sessions=True`` to restore legacy behavior (one fold per held-out
    ``(date, condition)`` of each stimulus, matching Protocol A fold count).
    """
    df = attach_stimulus_ids(pairs)
    folds: list[tuple[FoldSpec, pd.DataFrame]] = []
    for i, sid in enumerate(heldout_ids):
        stim = df[df["stimulus_id"] == sid]
        if stim.empty:
            continue
        groups = (
            stim[["date", "condition"]]
            .drop_duplicates()
            .sort_values(["date", "condition"])
        )
        if all_sessions:
            iter_groups = [
                (str(r.date), str(r.condition))
                for r in groups.itertuples(index=False)
            ]
        else:
            iter_groups = [_pick_protocol_c_test_group(stim)]
        for j, (date, condition) in enumerate(iter_groups):
            test = df[(df["date"] == date) & (df["condition"] == condition)].copy()
            rem = df[df["stimulus_id"] != sid].copy()
            train, val = _inner_train_val_split(
                rem, val_fraction=val_fraction, seed=seed + i + j
            )
            fold_df = _assign_loo_split(train, val, test)
            fold_id = f"C__{sid}__{date}_{condition}"
            ok, note = audit_protocol_c_leakage(
                fold_df,
                sid,
                heldout_date=date,
                heldout_condition=condition,
            )
            if not all_sessions:
                note = f"{note}; {PROTOCOL_C_TEST_SELECTION_RULE}"
            spec = FoldSpec(
                protocol="C",
                fold_id=fold_id,
                heldout_stimulus_id=sid,
                heldout_date=date,
                heldout_condition=condition,
                n_train=int((fold_df["loo_split"] == "train").sum()),
                n_val=int((fold_df["loo_split"] == "val").sum()),
                n_test=int((fold_df["loo_split"] == "test").sum()),
                leakage_ok=ok,
                notes=note,
            )
            folds.append((spec, fold_df))
    return folds


def write_fold_manifest(
    out_dir: Path,
    spec: FoldSpec,
    fold_df: pd.DataFrame,
) -> Path:
    """Write fold manifest parquet + JSON sidecar under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"{spec.fold_id}__manifest.parquet"
    meta_path = out_dir / f"{spec.fold_id}__meta.yaml"
    fold_df.to_parquet(fold_path, index=False)
    with meta_path.open("w") as f:
        yaml.safe_dump(spec.to_dict(), f, sort_keys=False)
    return fold_path
