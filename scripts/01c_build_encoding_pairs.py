#!/usr/bin/env python3
"""Build trial-level encoding pair manifest (stimulus image + averaged VSD target)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.xarray_schema import window_id_from_frames
from src.encoding.pairs import build_encoding_pairs
from src.paths import project_root, resolve_data_path, workspace_root


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_config(default_path: Path, window_path: Path) -> dict:
    cfg = _load_yaml(default_path)
    cfg.update(_load_yaml(window_path))
    return cfg


def _portable_path(abs_path: Path, repo: Path) -> str:
    ws = workspace_root(repo)
    resolved = abs_path.resolve()
    try:
        return str(resolved.relative_to(ws))
    except ValueError:
        return str(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--window",
        type=Path,
        required=True,
        help="Window config (start_frame, end_frame, avg_method, normalization)",
    )
    parser.add_argument("--monkey", type=str, default=None)
    parser.add_argument(
        "--require-nc",
        action="store_true",
        help="Drop pairs whose averaged .nc file is missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg = _merge_config(args.config, args.window)
    if args.monkey is not None:
        cfg["monkey"] = args.monkey

    monkey = cfg["monkey"]
    start_frame = int(cfg["start_frame"])
    end_frame = int(cfg["end_frame"])
    window_id = cfg.get("window_id") or window_id_from_frames(
        start_frame, end_frame
    )

    averaged_root = resolve_data_path(cfg["paths"]["averaged_root"], repo)
    stimuli_root = resolve_data_path(cfg["paths"]["stimuli_root"], repo)
    encoding_pairs_root = resolve_data_path(
        cfg["paths"]["encoding_pairs_root"], repo
    )

    build_encoding_pairs(
        monkey=monkey,
        window_id=window_id,
        start_frame=start_frame,
        end_frame=end_frame,
        split_csv=cfg["split_csv"],
        trials_index_csv=cfg.get("trials_index_csv"),
        averaged_root=averaged_root,
        stimuli_root=stimuli_root,
        encoding_pairs_root=encoding_pairs_root,
        repo=repo,
        avg_method=cfg.get("avg_method", "mean"),
        normalization=cfg.get("normalization", "none"),
        require_nc=args.require_nc,
        portable_path=lambda p: _portable_path(p, repo),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
