#!/usr/bin/env python3
"""Extract blankAN trials from Gandalf condsAN .mat files into session blank H5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.mat_blank_h5 import (
    DEFAULT_MAT_SESSIONS,
    DEFAULT_MONKEY,
    blank_h5_filename,
    default_output_h5,
    default_raw_mat_dir,
    write_blank_session_h5,
)
from src.paths import project_root, resolve_data_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract blankAN trials from Gandalf condsAN .mat files into "
            "ProcessedData/<monkey>/session_<date>_blank.h5"
        ),
    )
    p.add_argument(
        "--monkey",
        default=DEFAULT_MONKEY,
        help=f"Monkey name (default: {DEFAULT_MONKEY})",
    )
    p.add_argument(
        "--raw-dir",
        default=None,
        help="Directory containing condsAN .mat files (default: Data/FoundationData/RawData/gandalf)",
    )
    p.add_argument(
        "--mat-path",
        action="append",
        default=[],
        help="Explicit .mat path; repeat for multiple files (requires --date with one file)",
    )
    p.add_argument(
        "--date",
        action="append",
        default=[],
        help="Session date tag, e.g. 270618b (paired with --mat-path or overrides default map)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: Data/FoundationData/ProcessedData/<monkey>)",
    )
    p.add_argument(
        "--all-defaults",
        action="store_true",
        help="Process known condsAN-001/002 mats (default when no --mat-path given)",
    )
    return p.parse_args()


def _resolve_jobs(args: argparse.Namespace, repo: Path) -> list[tuple[Path, str]]:
    if args.mat_path:
        dates = args.date or []
        if len(dates) == 1 and len(args.mat_path) > 1:
            dates = dates * len(args.mat_path)
        if len(dates) != len(args.mat_path):
            raise SystemExit(
                "Provide one --date per --mat-path (or a single --date for all mats)."
            )
        return [
            (resolve_data_path(mat, repo), date)
            for mat, date in zip(args.mat_path, dates)
        ]

    raw_dir = (
        resolve_data_path(args.raw_dir, repo)
        if args.raw_dir
        else default_raw_mat_dir(repo=repo)
    )
    jobs: list[tuple[Path, str]] = []
    for mat_name, date in DEFAULT_MAT_SESSIONS:
        jobs.append((raw_dir / mat_name, date))
    return jobs


def main() -> int:
    args = _parse_args()
    repo = project_root()

    try:
        jobs = _resolve_jobs(args, repo)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.output_dir:
        out_dir = resolve_data_path(args.output_dir, repo)
    else:
        out_dir = default_output_h5(args.monkey, "placeholder", repo=repo).parent

    print(f"Output directory: {out_dir}")

    for mat_path, date in jobs:
        if not mat_path.exists():
            print(f"SKIP (missing): {mat_path}", file=sys.stderr)
            continue

        out_h5 = out_dir / blank_h5_filename(date)
        n_trials = write_blank_session_h5(
            mat_path,
            out_h5,
            monkey=args.monkey,
            date=date,
        )
        print(f"Wrote {out_h5}  ({n_trials} blank trials from {mat_path.name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
