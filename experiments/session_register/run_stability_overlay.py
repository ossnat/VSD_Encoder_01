"""Stability ridges on VSD + junction landmarks (no tracing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.session_register.config import load_config
from src.session_register.load import load_session_trial_stack
from src.session_register.plot import (
    landmark_id_label,
    plot_stability_overlay_on_raw,
    plot_vsd_yellow_blue_landmarks,
)
from src.session_register.vessels import resolve_anchor_landmarks, stability_ridge_landmarks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/session_register/configs/gandalf_100718a_anchor.yaml"),
    )
    p.add_argument("--sessions", nargs="+", default=["100718a", "240718a", "270618b"])
    p.add_argument("--corr-method", default="dark_and_stable")
    p.add_argument("--baseline-start", type=int, default=5)
    p.add_argument("--baseline-end", type=int, default=25)
    p.add_argument("--roi-row-min", type=int, default=38)
    p.add_argument("--roi-row-max", type=int, default=82)
    p.add_argument("--ridge-percentile", type=float, default=80.0)
    p.add_argument("--n-landmarks", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/session_register/runs/corr_vessels"),
    )
    args = p.parse_args()

    repo = _repo
    cfg = load_config(args.config, repo=repo)
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for session in args.sessions:
        print(f"{session}: baseline frames {args.baseline_start}–{args.baseline_end}...")
        stack, _ = load_session_trial_stack(
            cfg,
            session,
            repo=repo,
            start_frame=args.baseline_start,
            end_frame=args.baseline_end,
        )
        result = stability_ridge_landmarks(
            stack,
            cfg=cfg,
            method=args.corr_method,
            ridge_percentile=args.ridge_percentile,
            roi_row_min=args.roi_row_min,
            roi_row_max=args.roi_row_max,
            n_landmarks=args.n_landmarks,
        )
        if session == cfg.fixed_session:
            result, freeze_msg = resolve_anchor_landmarks(
                result, cfg.anchor_landmarks_path(repo), session
            )
            print(f"  {freeze_msg}")
        else:
            result, freeze_msg = resolve_anchor_landmarks(
                result,
                cfg.session_landmarks_path(session, repo),
                session,
                write_if_missing=False,
            )
            print(f"  {freeze_msg}")
        print(f"  ridge pixels: {int(result.ridge_mask.sum())}")
        print(f"  landmarks: {len(result.landmarks)}")
        ids = np.asarray(result.landmark_ids).reshape(-1, 2)
        for i, (row, col) in enumerate(result.landmarks):
            label = (
                landmark_id_label(int(ids[i, 0]), int(ids[i, 1]))
                if i < len(ids)
                else "?"
            )
            print(f"    {label}  ({row}, {col})")

        overlay_path = out_dir / f"stability_overlay__{session}.png"
        plot_stability_overlay_on_raw(
            session=session,
            mean_raw=result.mean_map,
            score_display=result.score_display,
            chamber=result.chamber_mask,
            path=overlay_path,
            corr_method=args.corr_method,
        )

        lm_path = out_dir / f"landmarks__{session}.png"
        plot_vsd_yellow_blue_landmarks(
            session=session,
            mean_raw=result.mean_map,
            score_display=result.score_display,
            chamber=result.chamber_mask,
            landmarks=result.landmarks,
            path=lm_path,
            corr_method=args.corr_method,
            ridge_mask=result.ridge_mask,
            landmark_ids=result.landmark_ids,
        )
        print(f"  overlay → {overlay_path}")
        print(f"  landmarks → {lm_path}")


if __name__ == "__main__":
    main()
