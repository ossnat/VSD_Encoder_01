"""Load golden mats from the legacy tree (skipped if missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shira_revised.src.legacy_vf import load_gandalf_legacy_yaml, load_paths

scipy = pytest.importorskip("scipy")
sio = scipy.io


@pytest.fixture(scope="module")
def paths():
    return load_paths()


def test_winner_amit_mat(paths):
    p = paths["winner_amit"]
    if not p.is_file():
        pytest.skip(f"missing {p}")
    m = sio.loadmat(p, squeeze_me=True)
    assert float(m["a"]) == pytest.approx(0.39)
    assert float(m["s"]) == pytest.approx(1.5)
    assert int(m["k"]) == 1


def test_joanna_winner_row(paths):
    p = paths["winners_201118"]
    if not p.is_file():
        pytest.skip(f"missing {p}")
    m = sio.loadmat(p, squeeze_me=True)
    key = "min_rmsd_winners" if "min_rmsd_winners" in m else "min_50_rmsd_winners"
    # file may use either name
    for cand in ("min_rmsd_winners", "min_50_rmsd_winners"):
        if cand in m:
            key = cand
            break
    else:
        pytest.fail(f"no winners key in {list(m)}")
    row = np.asarray(m[key])[0]
    # cols: a1r, a1c, a2r, a2c, a, alpha, k, mean_rmsd, ...
    assert row[4] == pytest.approx(0.72)
    assert row[5] == pytest.approx(1.5)
    assert row[7] == pytest.approx(1.52040292, rel=1e-5)
    cfg = load_gandalf_legacy_yaml()
    assert cfg["joanna_201118"]["a"] == pytest.approx(row[4])


def test_opt_points_shape(paths):
    p = paths["opt_points_201118"]
    if not p.is_file():
        pytest.skip(f"missing {p}")
    m = sio.loadmat(p, squeeze_me=True)
    pts = np.asarray(m["opt_points"])
    assert pts.shape == (8, 6)
