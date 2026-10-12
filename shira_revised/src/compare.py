"""Compare legacy (divide/abs) vs current-repo (multiply/signed) Schira maps."""

from __future__ import annotations

import numpy as np

from shira_revised.src.legacy_schira import (
    LegacySchiraParams,
    forward_legacy_schira,
    thesis_like_params,
)


def sample_lrq_points(n: int = 40, extent_deg: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform samples in lower-right quadrant (signed catalog degrees)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, extent_deg, size=n)
    y = rng.uniform(-extent_deg, -0.05, size=n)
    return x, y


def w_divergence_stats(
    a: float = 0.72,
    alpha: float = 1.5,
    k: float = 1.0,
    n: int = 200,
) -> dict[str, float]:
    """Median |Δw| between legacy divide/abs and thesis multiply/signed."""
    x, y = sample_lrq_points(n=n)
    legacy = LegacySchiraParams(a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True)
    thesis = thesis_like_params(a=a, alpha=alpha, k=k)
    w_leg = forward_legacy_schira(x, y, legacy)
    w_th = forward_legacy_schira(x, y, thesis)
    ok = np.isfinite(w_leg) & np.isfinite(w_th)
    d = np.abs(w_leg[ok] - w_th[ok])
    # Also compare divide vs multiply with *same* abs VF (isolate α).
    multiply_abs = LegacySchiraParams(
        a=a, alpha=alpha, k=k, alpha_mode="multiply", vf_abs=True
    )
    w_mul_abs = forward_legacy_schira(x, y, multiply_abs)
    ok2 = np.isfinite(w_leg) & np.isfinite(w_mul_abs)
    d_alpha = np.abs(w_leg[ok2] - w_mul_abs[ok2])
    return {
        "n_valid_full": float(ok.sum()),
        "median_abs_dw_full": float(np.median(d)) if d.size else float("nan"),
        "mean_abs_dw_full": float(np.mean(d)) if d.size else float("nan"),
        "n_valid_alpha_only": float(ok2.sum()),
        "median_abs_dw_alpha_only": float(np.median(d_alpha)) if d_alpha.size else float("nan"),
        "mean_abs_dw_alpha_only": float(np.mean(d_alpha)) if d_alpha.size else float("nan"),
    }
