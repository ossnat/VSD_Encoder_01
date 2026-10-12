#!/usr/bin/env python3
"""Plot w-clouds: legacy divide/abs vs thesis multiply/signed vs α-only swap.

Writes ``shira_revised/artifacts/alpha_convention_compare.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shira_revised.src.compare import sample_lrq_points, w_divergence_stats
from shira_revised.src.legacy_schira import LegacySchiraParams, forward_legacy_schira, thesis_like_params


def main() -> None:
    out_dir = REPO / "shira_revised" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = sample_lrq_points(n=400, extent_deg=5.0)
    a, alpha, k = 0.72, 1.5, 1.0

    modes = {
        "legacy divide+abs\n(T=θ/α, |x|,|y|)": LegacySchiraParams(
            a=a, alpha=alpha, k=k, alpha_mode="divide", vf_abs=True
        ),
        "multiply+abs\n(α only swapped)": LegacySchiraParams(
            a=a, alpha=alpha, k=k, alpha_mode="multiply", vf_abs=True
        ),
        "thesis-like\n(P=αθ, signed VF)": thesis_like_params(a, alpha, k),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (title, params) in zip(axes, modes.items()):
        w = forward_legacy_schira(x, y, params)
        ok = np.isfinite(w)
        ax.scatter(w.real[ok], w.imag[ok], s=6, c=np.hypot(x[ok], y[ok]), cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Re(w)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Im(w)")
    stats = w_divergence_stats(a=a, alpha=alpha, k=k, n=400)
    fig.suptitle(
        f"Gandalf Joanna params a={a}, α={alpha}, k={k}\n"
        f"median |Δw| full={stats['median_abs_dw_full']:.3f}, "
        f"α-only={stats['median_abs_dw_alpha_only']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "alpha_convention_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    for k_, v in stats.items():
        print(f"  {k_}: {v}")


if __name__ == "__main__":
    main()
