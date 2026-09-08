"""Correlation-based vessel detection diagnostic.

Loads per-trial frame stacks for one or more sessions, computes pixel-wise
inter-trial correlation maps, thresholds them to isolate blood vessels, finds
junction landmarks, and saves rich QC plots.

Usage
-----
  python experiments/session_register/run_correlation_vessels.py

Or with explicit sessions / config overrides:
  python experiments/session_register/run_correlation_vessels.py \
    --sessions 100718a 240718a 270618b \
    --corr-method inter_trial_corr \
    --vessel-percentile 85 \
    --out-dir experiments/session_register/runs/corr_vessels

The script intentionally does NOT re-run registration — it is a *diagnostic*
for the vessel detection step only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from repo root
_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.session_register.config import load_config
from src.session_register.load import load_session_mean_map, load_session_trial_stack
from src.session_register.vessels import correlation_vessel_map
from src.session_register.plot import plot_correlation_vessel_gallery


_DEFAULT_CONFIG = Path("experiments/session_register/configs/gandalf_100718a_anchor.yaml")
_DEFAULT_SESSIONS = ["100718a", "240718a", "270618b"]
_DEFAULT_OUT = Path("experiments/session_register/runs/corr_vessels")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="SessionRegisterConfig YAML (default: %(default)s)",
    )
    p.add_argument(
        "--sessions",
        nargs="+",
        default=_DEFAULT_SESSIONS,
        help="Sessions to process (default: %(default)s)",
    )
    p.add_argument(
        "--corr-method",
        default="dark_and_stable",
        choices=["baseline_stability", "dark_and_stable", "inter_trial_corr"],
        help="Stability metric (default: %(default)s)",
    )
    p.add_argument(
        "--vessel-percentile",
        type=float,
        default=80.0,
        help="Percentile threshold within chamber for vessel pixels (default: %(default)s)",
    )
    p.add_argument(
        "--n-vessel-objects",
        type=int,
        default=2,
        help="Keep N largest vessel trunks (default: %(default)s)",
    )
    p.add_argument(
        "--junction-percentile",
        type=float,
        default=80.0,
        help="Percentile for skeleton junction detection (default: %(default)s)",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma on stability map (default: %(default)s)",
    )
    p.add_argument(
        "--baseline-start",
        type=int,
        default=5,
        help="First baseline frame (inclusive, 0-indexed). Default: 5",
    )
    p.add_argument(
        "--baseline-end",
        type=int,
        default=25,
        help="Last baseline frame (exclusive). Default: 25  (pre-stimulus window)",
    )
    p.add_argument(
        "--roi-row-min",
        type=int,
        default=38,
        help="Top row of vessel search band (default: 38)",
    )
    p.add_argument(
        "--roi-row-max",
        type=int,
        default=82,
        help="Bottom row of vessel search band (default: 82)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also run old top-hat vessel extraction and save side-by-side comparison",
    )
    return p.parse_args()


def _compare_panel(
    session: str,
    mean_raw: np.ndarray,
    stability_map: np.ndarray,
    debug: dict,
    corr_vmaps,
    old_vmaps,
    out_path: Path,
) -> None:
    """Save a side-by-side comparison: old top-hat vs new correlation method."""
    from src.session_register.plot import _as_01

    raw01 = _as_01(np.asarray(mean_raw), stretch=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))

    def _overlay(ax, raw, mask, lms, title):
        rgb = np.stack([raw, raw, raw], axis=-1).copy()
        rgb[mask] = [0.92, 0.08, 0.08]
        ax.imshow(rgb, origin="upper", interpolation="nearest")
        if lms is not None and len(lms):
            pts = np.asarray(lms).reshape(-1, 2)
            ax.plot(pts[:, 1], pts[:, 0], "+", color="cyan", ms=10, mew=1.6)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()

    # Row 0: old method
    old_mask = old_vmaps.vessel_mask
    old_lms = old_vmaps.landmarks
    axes[0, 0].imshow(raw01, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[0, 0].set_title(f"{session}  mean raw", fontsize=9); axes[0, 0].set_axis_off()
    _overlay(axes[0, 1], raw01, old_mask, None, "old: top-hat vessels")
    _overlay(axes[0, 2], raw01, old_mask, old_lms, f"old: landmarks (N={len(old_lms)})")

    # Stability map in (0,3)
    chamber = debug["chamber"]
    smoothed = debug["smoothed"]
    lo = float(np.percentile(smoothed[chamber], 2))
    hi = float(np.percentile(smoothed[chamber], 98))
    norm = np.clip((smoothed - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    rgba = plt.cm.hot(norm)
    rgba[~chamber, :3] = 0.35
    axes[0, 3].imshow(rgba, origin="upper", interpolation="nearest")
    axes[0, 3].set_title(f"stability map ({debug.get('method','?')})", fontsize=9)
    axes[0, 3].set_axis_off()

    # Row 1: new method
    new_mask = corr_vmaps.vessel_mask
    new_lms = corr_vmaps.landmarks
    axes[1, 0].imshow(raw01, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[1, 0].set_title("mean raw (same)", fontsize=9); axes[1, 0].set_axis_off()
    _overlay(axes[1, 1], raw01, new_mask, None, "new: correlation vessels")
    _overlay(axes[1, 2], raw01, new_mask, new_lms, f"new: landmarks (N={len(new_lms)})")

    # Overlay of both masks: old=red, new=green, overlap=yellow
    rgb_cmp = np.stack([raw01, raw01, raw01], axis=-1).copy() * 0.5
    rgb_cmp[old_mask & ~new_mask] += [0.5, 0.0, 0.0]
    rgb_cmp[new_mask & ~old_mask] += [0.0, 0.5, 0.0]
    rgb_cmp[old_mask & new_mask] += [0.5, 0.5, 0.0]
    rgb_cmp = np.clip(rgb_cmp, 0, 1)
    axes[1, 3].imshow(rgb_cmp, origin="upper", interpolation="nearest")
    axes[1, 3].set_title("overlap: old=R, new=G, both=Y", fontsize=9)
    axes[1, 3].set_axis_off()

    fig.suptitle(
        f"Vessel detection comparison — {session}  "
        f"old=top-hat  new=correlation ({debug.get('method','?')})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  compare → {out_path}")


def main() -> None:
    args = parse_args()
    repo = _repo

    cfg = load_config(args.config, repo=repo)
    # Override key params from CLI
    cfg = cfg.with_updates(
        vessel_percentile=args.vessel_percentile,
        n_vessel_objects=args.n_vessel_objects,
        gaussian_sigma=args.smooth_sigma,
    )

    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Sessions: {args.sessions}")
    print(f"Method: {args.corr_method}  vessel_pct={args.vessel_percentile}")

    for session in args.sessions:
        print(f"\n=== {session} ===")

        # Load per-trial stack — use BASELINE frames (pre-stimulus)
        print(f"  Loading baseline trial stack for {session} "
              f"(frames {args.baseline_start}–{args.baseline_end})...")
        try:
            stack, trial_ids = load_session_trial_stack(
                cfg, session, repo=repo,
                start_frame=args.baseline_start,
                end_frame=args.baseline_end,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP: {e}")
            continue
        print(f"  Loaded {stack.shape[0]} trials, shape={stack.shape}")

        # Use the baseline mean (from the stack) as the raw overlay image —
        # this is what reveals the dark vessel rivers without the stimulus blob
        mean_map = stack.mean(axis=0)

        # Run correlation vessel detection
        print(f"  Computing {args.corr_method} stability map...")
        stability_map, vessel_maps, debug = correlation_vessel_map(
            stack,
            cfg=cfg,
            method=args.corr_method,
            smooth_sigma=args.smooth_sigma,
            vessel_percentile=args.vessel_percentile,
            n_vessel_objects=args.n_vessel_objects,
            roi_row_min=args.roi_row_min,
            roi_row_max=args.roi_row_max,
            junction_percentile=args.junction_percentile,
        )
        n_vessel_px = int(vessel_maps.vessel_mask.sum())
        n_lm = len(vessel_maps.landmarks)
        print(f"  Vessel pixels: {n_vessel_px}  Landmarks: {n_lm}")
        if n_lm:
            print(f"  Landmark positions (row, col):")
            for r, c in vessel_maps.landmarks:
                print(f"    ({r}, {c})")

        # Save main QC gallery
        gallery_path = out_dir / f"corr_vessels__{session}.png"
        plot_correlation_vessel_gallery(
            session=session,
            mean_raw=mean_map,
            stability_map=stability_map,
            debug=debug,
            vessel_maps=vessel_maps,
            path=gallery_path,
            corr_method=args.corr_method,
        )
        print(f"  gallery → {gallery_path}")

        # Save stability map as numpy for later inspection
        stab_npy = out_dir / f"stability__{session}.npy"
        np.save(stab_npy, stability_map)
        print(f"  stability → {stab_npy}")

        # Optional comparison with old method
        if args.compare:
            from src.session_register.vessels import extract_vessels
            print(f"  Running old top-hat extraction for comparison...")
            old_vmaps = extract_vessels(mean_map, cfg=cfg)
            cmp_path = out_dir / f"compare__{session}.png"
            _compare_panel(
                session, mean_map, stability_map, debug, vessel_maps, old_vmaps, cmp_path
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
