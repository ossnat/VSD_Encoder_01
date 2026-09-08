#!/usr/bin/env python3
"""Register all encoding-pair sessions onto an anchor camera.

Writes under::

    Data/VSD_Encoder_01/session_register/AS_{anchor}/
        {monkey}__{anchor}/                 # frozen landmarks
        {monkey}__{anchor}__{moving}/       # transform.yaml (+ optional QC)

Example::

    scripts/py scripts/14_register_all_to_anchor.py \\
      --config experiments/session_register/configs/gandalf_100718a_anchor.yaml \\
      --window-config configs/windows/evoked_35_46.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.encoding.schema import encoding_pairs_manifest_path
from src.paths import project_root, resolve_data_path
from src.session_register.config import RegistrationBoundsError, load_config
from src.session_register.pipeline import run_registration


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _default_as_root(anchor: str) -> str:
    return f"Data/VSD_Encoder_01/session_register/AS_{anchor}"


def _seed_anchor_landmarks(
    *,
    repo: Path,
    as_root: Path,
    monkey: str,
    anchor: str,
    legacy_runs: Path,
) -> None:
    """Copy frozen anchor landmarks from the old experiments/ runs tree if needed."""
    dest_dir = as_root / f"{monkey}__{anchor}"
    dest = dest_dir / "anchor_landmarks.yaml"
    if dest.is_file():
        return
    src = legacy_runs / f"{monkey}__{anchor}" / "anchor_landmarks.yaml"
    if not src.is_file():
        print(f"No legacy anchor landmarks at {src} (will extract on first fit)")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Seeded anchor landmarks: {dest}")


def main() -> None:
    repo = project_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=repo
        / "experiments/session_register/configs/gandalf_100718a_anchor.yaml",
    )
    p.add_argument(
        "--window-config",
        type=Path,
        default=repo / "configs/windows/evoked_35_46.yaml",
        help="Used only to discover sessions from the encoding-pairs manifest.",
    )
    p.add_argument(
        "--schira-encoding-config",
        type=Path,
        default=repo / "configs/schira_encoding/default.yaml",
        help="Optional: read encoding_pairs_root / monkey if present.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output root (default Data/.../session_register/AS_{anchor}).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-qc", action="store_true", help="Skip QC PNGs.")
    p.add_argument(
        "--sessions",
        type=str,
        default=None,
        help="Comma-separated moving sessions (default: all in pairs except anchor).",
    )
    args = p.parse_args()

    reg_cfg_raw = _load_yaml(args.config)
    win = _load_yaml(args.window_config)
    schira_cfg = (
        _load_yaml(args.schira_encoding_config)
        if args.schira_encoding_config.is_file()
        else {}
    )

    monkey = str(reg_cfg_raw.get("monkey") or schira_cfg.get("monkey") or "gandalf")
    anchor = str(reg_cfg_raw.get("fixed_session") or schira_cfg.get("anchor_session"))
    if not anchor:
        raise ValueError("fixed_session / anchor_session required")

    out_dir_str = args.output_dir or _default_as_root(anchor)
    as_root = resolve_data_path(out_dir_str, repo)
    as_root.mkdir(parents=True, exist_ok=True)

    legacy = repo / "experiments/session_register/runs"
    _seed_anchor_landmarks(
        repo=repo,
        as_root=as_root,
        monkey=monkey,
        anchor=anchor,
        legacy_runs=legacy,
    )

    pairs_root = resolve_data_path(
        (schira_cfg.get("paths") or {}).get(
            "encoding_pairs_root", "Data/VSD_Encoder_01/encoding_pairs"
        ),
        repo,
    )
    start = int(win.get("start_frame", 35))
    end = int(win.get("end_frame", 46))
    window_id = win.get("window_id") or f"win_{start:04d}_{end:04d}"
    manifest = encoding_pairs_manifest_path(pairs_root, monkey, window_id)
    pairs = pd.read_parquet(manifest)
    all_sessions = sorted(pairs["date"].astype(str).unique())

    if args.sessions:
        moving = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        moving = [s for s in all_sessions if s != anchor]

    print(f"Anchor: {anchor}")
    print(f"Output: {as_root}")
    print(f"Moving sessions ({len(moving)}): {moving}")

    results: list[dict] = []
    for session in moving:
        print(f"\n=== Register {session} → {anchor} ===", flush=True)
        cfg = load_config(
            args.config,
            moving_session=session,
            overrides={
                "output_dir": out_dir_str,
                "monkey": monkey,
                "fixed_session": anchor,
                "overwrite": bool(args.overwrite) or None,
                "save_qc": not args.no_qc,
                "verbose": 1,
            },
        )
        try:
            out = run_registration(
                cfg,
                repo=repo,
                save_qc=not args.no_qc,
                verbose=1,
                skip_existing=not args.overwrite,
            )
            results.append(
                {
                    "moving": session,
                    "ok": True,
                    "skipped": bool(out.get("skipped")),
                    "yaml": out.get("yaml_path"),
                    "rmsd_px": out.get("rmsd_px"),
                }
            )
            print(
                f"OK {session}: skipped={out.get('skipped')} "
                f"rmsd={out.get('rmsd_px')} → {out.get('yaml_path')}"
            )
        except RegistrationBoundsError as exc:
            results.append({"moving": session, "ok": False, "error": str(exc)})
            print(f"FAIL {session}: {exc}")
        except Exception as exc:  # noqa: BLE001 — batch continues
            results.append({"moving": session, "ok": False, "error": repr(exc)})
            print(f"FAIL {session}: {exc!r}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDone: {n_ok}/{len(results)} succeeded")
    for r in results:
        if not r.get("ok"):
            print(f"  failed {r['moving']}: {r.get('error')}")


if __name__ == "__main__":
    main()
