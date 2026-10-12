"""Show that current-repo convention diverges from live MATLAB."""

from __future__ import annotations

import numpy as np

from shira_revised.src.compare import w_divergence_stats
from shira_revised.src.legacy_schira import (
    LegacySchiraParams,
    forward_legacy_schira,
    thesis_like_params,
)


def test_full_stack_divergence_is_large():
    stats = w_divergence_stats(a=0.72, alpha=1.5, k=1.0, n=300)
    assert stats["n_valid_full"] > 50
    # With α=1.5, maps should differ substantially in complex plane.
    assert stats["median_abs_dw_full"] > 0.1
    assert stats["median_abs_dw_alpha_only"] > 0.05


def test_current_repo_forward_schira_vs_legacy_at_letter_a():
    """Control: main ``forward_schira`` ≠ legacy divide/abs at letter A center."""
    from src.retinotopy.schira import SchiraParams, cartesian_to_polar, forward_schira

    x, y = 1.0, -0.9
    legacy = LegacySchiraParams(a=0.72, alpha=1.5, k=1.0, alpha_mode="divide", vf_abs=True)
    w_leg = complex(forward_legacy_schira(x, y, legacy))

    repo = SchiraParams(a=0.72, alpha=1.5, k=1.0, fa_combine="power")
    ecc, theta = cartesian_to_polar(x, y)
    w_repo = complex(forward_schira(ecc, theta, repo))

    assert np.isfinite(w_leg) and np.isfinite(w_repo)
    assert abs(w_leg - w_repo) > 0.05

    # Thesis-like port should be closer to repo (signed VF + multiply).
    w_th = complex(forward_legacy_schira(x, y, thesis_like_params(0.72, 1.5, 1.0)))
    assert abs(w_th - w_repo) < abs(w_leg - w_repo)
