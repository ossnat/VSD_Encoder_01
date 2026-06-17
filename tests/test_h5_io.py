from __future__ import annotations

import numpy as np

from src.data.h5_io import resolve_trial_dataset
from src.paths import project_root, resolve_data_path


def test_resolve_trial_dataset_session_270618b():
    repo = project_root()
    h5 = resolve_data_path(
        "Data/FoundationData/ProcessedData/gandalf/session_270618b_condsAN.h5",
        repo,
    )
    if not h5.exists():
        return
    assert resolve_trial_dataset(h5, 4258) == "trial_000000"
    assert resolve_trial_dataset(h5, 4295) == "trial_000037"
    assert resolve_trial_dataset(h5, 4400) == "trial_000142"


def test_different_conditions_load_different_trials():
    repo = project_root()
    h5 = resolve_data_path(
        "Data/FoundationData/ProcessedData/gandalf/session_270618b_condsAN.h5",
        repo,
    )
    if not h5.exists():
        return
    from src.data.h5_io import read_trial_by_global_id
    from src.data.averaging import average_frames

    a = average_frames(read_trial_by_global_id(h5, 4258), 32, 42, (100, 100))
    b = average_frames(read_trial_by_global_id(h5, 4400), 32, 42, (100, 100))
    assert not np.allclose(a, b)
