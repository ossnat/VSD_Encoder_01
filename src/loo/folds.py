"""Leave-one-out fold construction for encoding experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from src.stimuli.identity import attach_stimulus_ids

Protocol = Literal["A", "B"]


DEFAULT_HELDOUT = [
    "white_point_0.1",
    "black_triangle_contour_0.4",
    "black_bar_vertical_0.3",
    "black_bar_vertical_1",
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


def load_heldout_list(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_HELDOUT)
    with path.open() as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        items = data.get("heldout_stimulus_ids") or data.get("heldouts") or []
        return [str(x) for x in items]
    raise ValueError(f"Unrecognized held-out list format: {path}")


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
