from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_catalog_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "report_5"
        / "build_stimuli_catalog.py"
    )
    spec = importlib.util.spec_from_file_location("build_stimuli_catalog", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evenly_spaced_indices_unique_and_bounded():
    mod = _load_catalog_module()
    for n in range(1, 40):
        for k in range(1, n + 5):
            idxs = mod._evenly_spaced_indices(n, k)
            assert len(idxs) == min(k, n)
            assert len(set(idxs)) == len(idxs)
            assert all(0 <= i < n for i in idxs)
            assert idxs == sorted(idxs)


def test_evenly_spaced_spans_when_many_trials():
    mod = _load_catalog_module()
    idxs = mod._evenly_spaced_indices(20, 8)
    assert idxs[0] == 0
    assert idxs == [0, 2, 5, 7, 10, 12, 15, 17]
    assert len(idxs) == 8
