"""Build trial-level encoding pair manifests (stimulus image + averaged VSD)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.splits import load_trial_table
from src.data.xarray_schema import trial_output_path, window_id_from_frames
from src.encoding.schema import encoding_pairs_manifest_path
from src.paths import resolve_data_path
from src.stimuli.schema import manifest_path as stimulus_manifest_path


def dedupe_stimulus_manifest(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per (h5_session, condition).

    When the catalog has duplicates (e.g. blank vs bar for condAN6), prefer the
    non-blank stimulus row.
    """
    ordered = stimulus_df.sort_values(
        ["h5_session", "condition", "is_blank", "condition_num"],
        ascending=[True, True, True, True],
    )
    return ordered.drop_duplicates(
        ["h5_session", "condition"], keep="first"
    ).reset_index(drop=True)


def build_encoding_pairs(
    *,
    monkey: str,
    window_id: str,
    start_frame: int,
    end_frame: int,
    split_csv: str | Path,
    trials_index_csv: str | Path | None,
    averaged_root: Path,
    stimuli_root: Path,
    encoding_pairs_root: Path,
    repo: Path,
    avg_method: str = "mean",
    normalization: str = "none",
    require_nc: bool = False,
    require_stimulus: bool = True,
    portable_path,
) -> pd.DataFrame:
    """
    Join split trials with stimulus images and averaged VSD .nc paths.

    Join key: (monkey, date, condition) where date == h5_session.
    """
    trials = load_trial_table(
        split_csv,
        monkey,
        trials_index_csv=trials_index_csv,
        project_root_path=repo,
    )

    stim_manifest = stimulus_manifest_path(stimuli_root, monkey)
    if not stim_manifest.exists():
        raise FileNotFoundError(
            f"Stimulus manifest not found: {stim_manifest}. Run stage 01b first."
        )
    stimulus_df = dedupe_stimulus_manifest(pd.read_parquet(stim_manifest))
    encoder_sessions = set(stimulus_df["h5_session"].unique())
    trials = trials[trials["date"].isin(encoder_sessions)].copy()

    stimulus_cols = [
        "h5_session",
        "condition",
        "condition_num",
        "image_path",
        "stimulus_text",
        "color",
        "shape_type",
        "size_deg",
        "pos_x_deg",
        "pos_y_deg",
        "is_blank",
    ]
    merged = trials.merge(
        stimulus_df[stimulus_cols],
        left_on=["date", "condition"],
        right_on=["h5_session", "condition"],
        how="left" if not require_stimulus else "inner",
        validate="m:1",
    )

    if not require_stimulus:
        pairs = merged
    else:
        trial_groups = (
            trials.groupby(["date", "condition"], as_index=False)
            .size()
            .rename(columns={"size": "n_trials"})
        )
        uncovered = trial_groups.merge(
            stimulus_df[["h5_session", "condition"]],
            left_on=["date", "condition"],
            right_on=["h5_session", "condition"],
            how="left",
            indicator=True,
        )
        uncovered = uncovered[uncovered["_merge"] == "left_only"]
        if not uncovered.empty:
            n_uncovered_trials = int(uncovered["n_trials"].sum())
            print(
                f"WARNING: {len(uncovered)} (date, condition) groups "
                f"({n_uncovered_trials} trials) have no stimulus mapping:"
            )
            for row in uncovered.itertuples(index=False):
                print(f"  {row.date} {row.condition} ({row.n_trials} trials)")
        pairs = merged

    nc_paths: list[str] = []
    nc_exists: list[bool] = []
    stimulus_exists: list[bool] = []
    for row in pairs.itertuples(index=False):
        nc_path = trial_output_path(
            averaged_root, monkey, window_id, int(row.trial_global_id)
        )
        nc_paths.append(portable_path(nc_path))
        nc_exists.append(nc_path.exists())
        if pd.isna(row.image_path):
            stimulus_exists.append(False)
        else:
            stimulus_exists.append(
                resolve_data_path(row.image_path, repo).exists()
            )

    pairs = pairs.assign(
        window_id=window_id,
        start_frame=start_frame,
        end_frame=end_frame,
        avg_method=avg_method,
        normalization=normalization,
        nc_path=nc_paths,
        nc_exists=nc_exists,
        stimulus_exists=stimulus_exists,
    )

    n_missing_stimulus = int((~pairs["stimulus_exists"]).sum())
    n_missing_nc = int((~pairs["nc_exists"]).sum())
    if require_nc:
        pairs = pairs[pairs["nc_exists"]].reset_index(drop=True)

    pairs = pairs.sort_values(
        ["date", "condition", "trial_index_in_condition", "trial_global_id"]
    ).reset_index(drop=True)

    out_path = encoding_pairs_manifest_path(encoding_pairs_root, monkey, window_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)

    print(f"Monkey: {monkey}")
    print(f"Window: {window_id} [{start_frame}, {end_frame})")
    print(f"Encoder sessions: {len(encoder_sessions)}")
    print(f"Stimulus conditions: {len(stimulus_df)}")
    print(f"Encoding pairs: {len(pairs)}")
    if n_missing_stimulus:
        print(f"WARNING: {n_missing_stimulus} pairs missing stimulus image on disk")
    if n_missing_nc:
        print(
            f"WARNING: {n_missing_nc} pairs missing averaged .nc "
            f"(run stage 01 for window {window_id})"
        )
    print(f"Manifest: {portable_path(out_path)}")
    return pairs
