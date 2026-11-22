"""Stimulus feature maps are stored once per stimulus_uid and resolved through the stimulus lookup table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.DL_features.extract_stimulus import plan_stimulus_extraction
from src.DL_features.schema import (
    legacy_stimulus_map_path,
    stimulus_map_path,
    stimulus_uid_for,
    stimulus_uid_map_path,
)


@pytest.fixture()
def roots(tmp_path: Path):
    features_root = tmp_path / "VSD_Encoder_01" / "DL_features_stimuli"
    stim = tmp_path / "VSD_Encoder_01" / "stimuli" / "gandalf"
    stim.mkdir(parents=True)
    pd.DataFrame(
        {
            "h5_session": ["100718a", "270618b", "100718a"],
            "condition": ["condAN1", "condAN1", "condAN3"],
            "condition_num": [1, 1, 3],
            "stimulus_uid": ["point__black__d0.1__x0.6_y-0.75__bg128"] * 2 + ["triangle_contour__black__d0.8__x0.6_y-0.75__bg128"],
        }
    ).to_parquet(stim / "lookup.parquet")
    return features_root


def test_shared_stimulus_shares_one_feature_file(roots):
    a = stimulus_map_path(roots, "gandalf", "vgg16_imagenet", "block1_prepool", "100718a", "condAN1")
    b = stimulus_map_path(roots, "gandalf", "vgg16_imagenet", "block1_prepool", "270618b", "condAN1")
    c = stimulus_map_path(roots, "gandalf", "vgg16_imagenet", "block1_prepool", "100718a", "condAN3")
    assert a == b != c
    assert a.parent.name == "maps_by_uid" and a.name == "point__black__d0.1__x0.6_y-0.75__bg128.npy"
    assert a == stimulus_uid_map_path(roots, "gandalf", "vgg16_imagenet", "block1_prepool", "point__black__d0.1__x0.6_y-0.75__bg128")


def test_unknown_condition_raises(roots):
    with pytest.raises(KeyError, match="No stimulus"):
        stimulus_uid_for(roots, "gandalf", "100718a", "condAN9")


def test_legacy_path_unchanged(roots):
    p = legacy_stimulus_map_path(roots, "gandalf", "vgg16_imagenet", "block1_prepool", "100718a", "condAN1")
    assert p.parent.name == "maps" and p.name == "100718a__condAN1.npy"


def test_plan_extracts_each_file_once(roots):
    man = pd.DataFrame({"h5_session": ["100718a", "270618b", "100718a"], "condition": ["condAN1", "condAN1", "condAN3"],
                        "condition_num": [1, 1, 3]})
    fn = lambda s, c: stimulus_map_path(roots, "gandalf", "m", "l", s, c)
    primary, aliases = plan_stimulus_extraction(man, fn)
    assert len(primary) == 2
    assert sum(len(v) for v in aliases.values()) == 1
