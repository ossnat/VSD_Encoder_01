#!/usr/bin/env python3
"""Register a new-session VSD camera onto an anchor using blood vessels.

Example (Gandalf 240718a → 100718a):

  scripts/py experiments/session_register/run_register.py \\
    --config experiments/session_register/configs/gandalf_100718a_anchor.yaml \\
    --moving-session 240718a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.paths import project_root
from src.session_register.config import RegistrationBoundsError, load_config
from src.session_register.pipeline import run_registration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rigid vasculature registration of a moving session onto an anchor camera."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/session_register/configs/gandalf_100718a_anchor.yaml"),
        help="Anchor YAML (monkey, fixed_session, vessel / search knobs).",
    )
    p.add_argument(
        "--moving-session",
        required=True,
        help="H5 session id to warp onto the anchor (e.g. 240718a).",
    )
    p.add_argument("--fixed-session", default=None, help="Override YAML fixed_session.")
    p.add_argument("--monkey", default=None, help="Override YAML monkey (within-animal only).")
    p.add_argument("--output-dir", default=None, help="Override YAML output_dir.")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None, help="Exclusive end (Python slice).")
    p.add_argument(
        "--n-trials",
        default=None,
        help="Cap on trials per session, or 'all'.",
    )
    p.add_argument("--metric", choices=("ncc", "mattes_mi"), default=None)
    p.add_argument(
        "--transform-type",
        choices=("translation", "rigid", "similarity"),
        default=None,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Refit even if transform.yaml exists; also re-average H5 caches.",
    )
    p.add_argument(
        "--qc",
        dest="save_qc",
        action="store_true",
        default=True,
        help="Write QC PNGs under the pair directory (default).",
    )
    p.add_argument(
        "--no-qc",
        dest="save_qc",
        action="store_false",
        help="Skip QC figures (pipeline-friendly).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="No stdout (still writes transform.yaml).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Repeat for more logs (default: one summary line).",
    )
    return p.parse_args(argv)


def _n_trials_override(value: str | None):
    if value is None:
        return None
    if str(value).strip().lower() == "all":
        return "all"
    return int(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    overrides = {
        "fixed_session": args.fixed_session,
        "monkey": args.monkey,
        "output_dir": args.output_dir,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "n_trials": _n_trials_override(args.n_trials),
        "metric": args.metric,
        "transform_type": args.transform_type,
        "overwrite": True if args.overwrite else None,
        "save_qc": bool(args.save_qc),
        "verbose": 0 if args.quiet else max(int(args.verbose), 0),
    }
    cfg = load_config(
        args.config,
        moving_session=args.moving_session,
        repo=repo,
        overrides=overrides,
    )
    try:
        run_registration(cfg, repo=repo)
    except RegistrationBoundsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
