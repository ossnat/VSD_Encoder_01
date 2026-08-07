from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.qc.trial_cleanliness import (
    CLEANLINESS_AMP_EDGE,
    CLEANLINESS_GOOD,
    CLEANLINESS_PATTERN,
    default_cleanliness_run_tag,
    filter_pairs_by_cleanliness,
)


def test_filter_pairs_by_cleanliness_keep_good(tmp_path: Path):
    pairs = pd.DataFrame(
        {
            "trial_global_id": [1, 2, 3, 4],
            "split": ["train", "train", "val", "test"],
        }
    )
    qc = pd.DataFrame(
        {
            "trial_global_id": [1, 2, 3, 4],
            "trial_cleanliness": [
                CLEANLINESS_GOOD,
                CLEANLINESS_PATTERN,
                CLEANLINESS_GOOD,
                CLEANLINESS_AMP_EDGE,
            ],
        }
    )
    csv_path = tmp_path / "qc.csv"
    qc.to_csv(csv_path, index=False)

    filtered, stats = filter_pairs_by_cleanliness(
        pairs, csv_path=csv_path, keep=CLEANLINESS_GOOD
    )
    assert list(filtered["trial_global_id"]) == [1, 3]
    assert stats["n_before"] == 4
    assert stats["n_after"] == 2
    assert stats["n_dropped"] == 2
    assert stats["keep"] == [CLEANLINESS_GOOD]


def test_filter_pairs_require_match_drops_unlabeled(tmp_path: Path):
    pairs = pd.DataFrame({"trial_global_id": [1, 99]})
    qc = pd.DataFrame(
        {"trial_global_id": [1], "trial_cleanliness": [CLEANLINESS_GOOD]}
    )
    csv_path = tmp_path / "qc.csv"
    qc.to_csv(csv_path, index=False)

    filtered, stats = filter_pairs_by_cleanliness(
        pairs, csv_path=csv_path, keep="good", require_match=True
    )
    assert list(filtered["trial_global_id"]) == [1]
    assert stats["n_unmatched"] == 1


def test_default_cleanliness_run_tag():
    assert default_cleanliness_run_tag("good") == "clean_good"
    assert default_cleanliness_run_tag(["good"]) == "clean_good"


def test_filter_rejects_unknown_label(tmp_path: Path):
    pairs = pd.DataFrame({"trial_global_id": [1]})
    qc = pd.DataFrame(
        {"trial_global_id": [1], "trial_cleanliness": [CLEANLINESS_GOOD]}
    )
    csv_path = tmp_path / "qc.csv"
    qc.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unknown cleanliness"):
        filter_pairs_by_cleanliness(pairs, csv_path=csv_path, keep="excellent")
