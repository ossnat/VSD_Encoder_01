#!/usr/bin/env python3
"""Prepare flat Protocol A run root + four encoding leaves for SLURM arrays.

For each of ``{zscore,raw} × {clean,all}``:
  1. Resolve the flat leaf path under the shared run root.
  2. Dry-run ``run_loo_encoding.py`` to write ``params.yaml``,
     ``folds_index.yaml``, and per-fold manifests (no training).
  3. Emit ``folds.txt`` (one fold_id per line) for ``#SBATCH --array``.

Also writes ``pipeline_manifest.yaml`` at the run root for downstream
noise-corr / report stages.

Does **not** launch training. Use ``--paths-only`` to print planned paths
without touching the filesystem or requiring encoding pairs.

Usage::

  scripts/py experiments/loo_encoding/prepare_protocol_A_pipeline.py \\
    --config experiments/loo_encoding/slurm/protocol_A_full.yaml

  scripts/py experiments/loo_encoding/prepare_protocol_A_pipeline.py \\
    --config experiments/loo_encoding/slurm/protocol_A_full.yaml --paths-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.DL_features.schema import model_slug
from src.data.averaging import resolve_normalization
from src.loo.paths import (
    cleanliness_leaf_tag,
    flat_leaf_name,
    flat_run_root_name,
    resolve_flat_out_dir,
)
from src.paths import project_root

DEFAULT_CONFIG = Path("experiments/loo_encoding/slurm/protocol_A_full.yaml")

LEAF_SPECS = (
    ("zscore", "clean"),
    ("zscore", "all"),
    ("raw", "clean"),
    ("raw", "all"),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _window_path(cfg: dict[str, Any], kind: str) -> Path:
    windows = cfg.get("windows") or {}
    if kind not in windows:
        raise KeyError(f"config.windows.{kind} missing")
    return Path(str(windows[kind]))


def _resolve_cleanliness_csv(cfg: dict[str, Any], repo: Path) -> Path | None:
    qc = cfg.get("trial_cleanliness") or {}
    raw = qc.get("csv")
    if raw in (None, "", "auto"):
        monkey = str((cfg.get("monkey") or "gandalf"))
        # Classification defaults to the zscore window id.
        win_yaml = repo / _window_path(cfg, "zscore")
        win_cfg = _load_yaml(win_yaml)
        window_id = str(
            win_cfg.get("window_id")
            or f"win_{int(win_cfg['start_frame']):04d}_{int(win_cfg['end_frame']):04d}"
        )
        return Path(
            f"Data/VSD_Encoder_01/qc/trial_cleanliness_{monkey}__{window_id}.csv"
        )
    return Path(str(raw))


def _rel_or_str(path: Path, repo: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(repo.resolve()))
        except ValueError:
            return str(path)
    return str(path)

def plan_pipeline(
    cfg: dict[str, Any],
    *,
    repo: Path,
    run_date: str | None = None,
) -> dict[str, Any]:
    model_path = Path(str(cfg.get("model") or "configs/models/resnet18.yaml"))
    if not model_path.is_absolute():
        model_path = repo / model_path
    model_cfg = _load_yaml(model_path)
    feature_layer = str(
        cfg.get("feature_layer") or model_cfg.get("feature_layer") or "layer3"
    )
    model_name = model_slug(model_cfg)

    # Frame range from either window (same start/end by design).
    win0 = _load_yaml(repo / _window_path(cfg, "zscore"))
    start_frame = int(win0["start_frame"])
    end_frame = int(win0["end_frame"])
    run_date_s = run_date or str(cfg.get("run_date") or date.today().isoformat())
    run_root_name = cfg.get("run_root") or flat_run_root_name(
        run_date=run_date_s,
        start_frame=start_frame,
        end_frame=end_frame,
        model_slug=model_name,
        feature_layer=feature_layer,
    )
    run_root = repo / "experiments/loo_encoding/runs" / str(run_root_name)

    loss_roi = str(cfg.get("loss_roi") or "noise_ceiling_hull")
    heldout = Path(
        str(cfg.get("heldout_list") or "experiments/loo_encoding/heldout_list.yaml")
    )
    cleanliness_csv = _resolve_cleanliness_csv(cfg, repo)
    keep = (cfg.get("trial_cleanliness") or {}).get("keep") or ["good"]
    if isinstance(keep, str):
        keep = [keep]

    leaves: list[dict[str, Any]] = []
    for window_kind, cleanliness in LEAF_SPECS:
        window_yaml = _window_path(cfg, window_kind)
        win_cfg = _load_yaml(repo / window_yaml)
        normalization = resolve_normalization(win_cfg.get("normalization", "none"))
        clean_tag = cleanliness_leaf_tag(
            trial_cleanliness_csv=(
                cleanliness_csv if cleanliness == "clean" else None
            ),
            keep=keep,
        )
        leaf_cleanliness = "all" if cleanliness == "all" else clean_tag
        leaf_name = flat_leaf_name(
            protocol="A",
            normalization=normalization,
            target_mask_mode=loss_roi,
            cleanliness=leaf_cleanliness,
        )
        root_dir, leaf_dir = resolve_flat_out_dir(
            repo,
            run_date=run_date_s,
            start_frame=start_frame,
            end_frame=end_frame,
            model_slug=model_name,
            feature_layer=feature_layer,
            protocol="A",
            normalization=normalization,
            target_mask_mode=loss_roi,
            cleanliness=leaf_cleanliness,
            run_root=run_root_name,
        )
        assert root_dir == run_root
        assert leaf_dir.name == leaf_name
        leaves.append(
            {
                "key": f"{window_kind}_{cleanliness}",
                "window_kind": window_kind,
                "cleanliness": cleanliness,
                "window": str(window_yaml),
                "leaf_name": leaf_name,
                "leaf_dir": str(leaf_dir.relative_to(repo)),
                "folds_txt": str((leaf_dir / "folds.txt").relative_to(repo)),
                "trial_cleanliness_csv": (
                    _rel_or_str(cleanliness_csv, repo)
                    if cleanliness == "clean" and cleanliness_csv is not None
                    else None
                ),
                "trial_cleanliness_keep": list(keep) if cleanliness == "clean" else None,
            }
        )

    return {
        "run_date": run_date_s,
        "run_root": str(run_root.relative_to(repo)),
        "run_root_name": str(run_root_name),
        "model": str(model_path.relative_to(repo)),
        "model_slug": model_name,
        "feature_layer": feature_layer,
        "loss_roi": loss_roi,
        "heldout_list": str(
            heldout if heldout.is_absolute() else heldout
        ),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "leaves": leaves,
        "noise_corr_dir": str(
            (run_root / "noise_corr_odd_even").relative_to(repo)
        ),
        "report_pdf": str((run_root / "report.pdf").relative_to(repo)),
    }


def _write_folds_txt(leaf_dir: Path) -> int:
    index_path = leaf_dir / "folds_index.yaml"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    payload = _load_yaml(index_path)
    fold_ids = [str(f["fold_id"]) for f in (payload.get("folds") or [])]
    folds_txt = leaf_dir / "folds.txt"
    folds_txt.write_text("\n".join(fold_ids) + ("\n" if fold_ids else ""))
    return len(fold_ids)


def _dry_run_leaf(
    *,
    repo: Path,
    cfg: dict[str, Any],
    plan: dict[str, Any],
    leaf: dict[str, Any],
) -> None:
    py = repo / ".venv" / "bin" / "python"
    python = str(py if py.is_file() else sys.executable)
    cmd = [
        python,
        "experiments/loo_encoding/run_loo_encoding.py",
        "--window",
        leaf["window"],
        "--model",
        plan["model"],
        "--feature-layer",
        plan["feature_layer"],
        "--protocol",
        "A",
        "--layout",
        "flat",
        "--run-root",
        plan["run_root_name"],
        "--run-date",
        plan["run_date"],
        "--loss-roi",
        plan["loss_roi"],
        "--heldout",
        plan["heldout_list"],
        "--dry-run",
    ]
    ridge = cfg.get("ridge_config")
    if ridge:
        cmd.extend(["--ridge-config", str(ridge)])
    monkey = cfg.get("monkey")
    if monkey:
        cmd.extend(["--monkey", str(monkey)])
    if leaf["trial_cleanliness_csv"]:
        cmd.extend(
            [
                "--trial-cleanliness-csv",
                leaf["trial_cleanliness_csv"],
                "--trial-cleanliness-keep",
                *(leaf["trial_cleanliness_keep"] or ["good"]),
            ]
        )
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)


def prepare(
    cfg: dict[str, Any],
    *,
    repo: Path,
    run_date: str | None = None,
    paths_only: bool = False,
) -> dict[str, Any]:
    plan = plan_pipeline(cfg, repo=repo, run_date=run_date)
    if paths_only:
        return plan

    run_root = repo / plan["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)

    for leaf in plan["leaves"]:
        _dry_run_leaf(repo=repo, cfg=cfg, plan=plan, leaf=leaf)
        leaf_dir = repo / leaf["leaf_dir"]
        n = _write_folds_txt(leaf_dir)
        leaf["n_folds"] = n
        print(f"  {leaf['leaf_name']}: {n} folds -> {leaf['folds_txt']}", flush=True)

    # Noise-corr uses the all-data fold list (no cleanliness filter).
    all_leaf = next(L for L in plan["leaves"] if L["key"] == "zscore_all")
    plan["noise_corr_folds_index"] = str(
        (repo / all_leaf["leaf_dir"] / "folds_index.yaml").relative_to(repo)
    )
    plan["noise_corr_folds_txt"] = all_leaf["folds_txt"]

    manifest_path = run_root / "pipeline_manifest.yaml"
    with manifest_path.open("w") as f:
        yaml.safe_dump(plan, f, sort_keys=False)
    plan["pipeline_manifest"] = str(manifest_path.relative_to(repo))
    print(f"Wrote {manifest_path.relative_to(repo)}", flush=True)
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Override config run_date / today for the flat run-root name",
    )
    p.add_argument(
        "--paths-only",
        action="store_true",
        help="Print planned paths only (no dry-run, no filesystem writes)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = project_root()
    cfg_path = args.config if args.config.is_absolute() else repo / args.config
    cfg = _load_yaml(cfg_path)
    plan = prepare(
        cfg,
        repo=repo,
        run_date=args.run_date,
        paths_only=bool(args.paths_only),
    )
    print(yaml.safe_dump(plan, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
