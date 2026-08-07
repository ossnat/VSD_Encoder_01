"""Load and filter trial tables from shared split CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import resolve_data_path


def load_trial_table(
    split_csv: str | Path,
    monkey: str,
    *,
    trials_index_csv: str | Path | None = None,
    project_root_path: Path | None = None,
) -> pd.DataFrame:
    """
    Return one row per trial for the given monkey.

    Merges condition string from all_trials_index when available.
    """
    split_path = resolve_data_path(split_csv, project_root_path)
    df = pd.read_csv(split_path)
    df = df[df["monkey"] == monkey].copy()
    df = df.rename(columns={"condition": "condition_code"})

    if trials_index_csv is not None:
        index_path = resolve_data_path(trials_index_csv, project_root_path)
        index_df = pd.read_csv(index_path)[["trial_global_id", "condition"]]
        df = df.merge(index_df, on="trial_global_id", how="left")
        missing = df["condition"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} trials missing condition string in trials index"
            )
    else:
        df["condition"] = df["condition_code"].astype(str)

    df = df.sort_values("trial_global_id").reset_index(drop=True)
    return df
