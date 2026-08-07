"""Flat LOO run path naming (new runs).

Layout::

    experiments/loo_encoding/runs/
      YYYY-MM-DD_{start}-{end}_{model}_{layer}/   # run root
        protocol_{A|B}_{zscore|raw}_{ROI}_{clean|all}/  # leaf
          params.yaml
          folds_index.yaml
          loo_summary.csv
          <fold_id>/...

Historical deep trees (``runs/<window_id>/<model>/<layer>/protocol_*/``)
remain readable; new runs default to this flat layout.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.data.averaging import NORMALIZATION_BASELINE_ZSCORE, resolve_normalization

OUT_ROOT = Path("experiments/loo_encoding/runs")

# Short ROI tokens for leaf directory names (full detail in params.yaml).
_ROI_LEAF_TAGS: dict[str, str] = {
    "none": "full",
    "disk": "disk",
    "noise_ceiling_hull": "NChull",
    "box_union": "boxunion",
    "roi": "boxroi",
}


def safe_dir_token(raw: str) -> str:
    token = "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "mask"


def short_model_slug(model_slug: str) -> str:
    """``resnet18_imagenet`` → ``resnet18``; other slugs pass through."""
    name = str(model_slug)
    for suffix in ("_imagenet", "_random"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def short_layer_slug(feature_layer: str) -> str:
    """``layer3`` → ``l3``; otherwise a filesystem-safe token."""
    m = re.fullmatch(r"layer(\d+)", str(feature_layer), flags=re.IGNORECASE)
    if m:
        return f"l{m.group(1)}"
    return safe_dir_token(str(feature_layer))


def normalization_leaf_tag(normalization: str | None) -> str:
    """Map window normalization to ``zscore`` or ``raw``."""
    mode = resolve_normalization(normalization)
    if mode == NORMALIZATION_BASELINE_ZSCORE:
        return "zscore"
    return "raw"


def roi_leaf_tag(
    target_mask_mode: str,
    *,
    run_tag: str | None = None,
    mask_path: Path | None = None,
) -> str:
    """Short ROI token for the leaf name (``NChull``, ``disk``, ``full``, …)."""
    if target_mask_mode in _ROI_LEAF_TAGS:
        return _ROI_LEAF_TAGS[target_mask_mode]
    if run_tag:
        return safe_dir_token(run_tag)
    if mask_path is not None:
        stem = mask_path.stem
        if stem.endswith("__mask"):
            stem = stem[: -len("__mask")]
        elif stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        return safe_dir_token(stem)
    return "mask"


def cleanliness_leaf_tag(
    *,
    trial_cleanliness_csv: Path | str | None,
    keep: str | Iterable[str] | None = None,
) -> str:
    """``clean`` when filtering to good trials; ``all`` when unfiltered."""
    if trial_cleanliness_csv is None:
        return "all"
    labels = (
        [str(keep)]
        if isinstance(keep, str)
        else [str(x) for x in (list(keep) if keep is not None else ["good"])]
    )
    if labels == ["good"]:
        return "clean"
    return "clean_" + "_".join(safe_dir_token(x) for x in sorted(labels))


def flat_run_root_name(
    *,
    run_date: date | str,
    start_frame: int,
    end_frame: int,
    model_slug: str,
    feature_layer: str,
) -> str:
    """
    Run-root directory name.

    Example: ``2026-08-06_35-46_resnet18_l3``
    """
    if isinstance(run_date, date):
        date_s = run_date.isoformat()
    else:
        date_s = str(run_date)
    return (
        f"{date_s}_{int(start_frame)}-{int(end_frame)}_"
        f"{short_model_slug(model_slug)}_{short_layer_slug(feature_layer)}"
    )


def flat_leaf_name(
    *,
    protocol: str,
    normalization: str | None,
    target_mask_mode: str,
    cleanliness: str,
    run_tag: str | None = None,
    mask_path: Path | None = None,
    extra_tag: str | None = None,
) -> str:
    """
    Leaf directory under the run root.

    Example: ``protocol_A_zscore_NChull_clean``
    """
    parts = [
        f"protocol_{protocol}",
        normalization_leaf_tag(normalization),
        roi_leaf_tag(
            target_mask_mode, run_tag=run_tag, mask_path=mask_path
        ),
        safe_dir_token(cleanliness),
    ]
    name = "_".join(parts)
    if extra_tag:
        name = f"{name}__{safe_dir_token(extra_tag)}"
    return name


def uniquify_leaf_dir(
    parent: Path,
    leaf_name: str,
    *,
    when: datetime | None = None,
) -> Path:
    """
    Avoid colliding with an existing leaf: try ``_HHMM``, then ``_v2``, ``_v3``, …

    Call only when intentionally creating a fresh leaf (``--fresh``).
    """
    candidate = parent / leaf_name
    if not candidate.exists():
        return candidate
    stamp = (when or datetime.now()).strftime("%H%M")
    timed = parent / f"{leaf_name}_{stamp}"
    if not timed.exists():
        return timed
    version = 2
    while True:
        alt = parent / f"{leaf_name}_v{version}"
        if not alt.exists():
            return alt
        version += 1


def resolve_flat_out_dir(
    repo: Path,
    *,
    run_date: date | str,
    start_frame: int,
    end_frame: int,
    model_slug: str,
    feature_layer: str,
    protocol: str,
    normalization: str | None,
    target_mask_mode: str,
    cleanliness: str,
    run_root: str | Path | None = None,
    mask_path: Path | None = None,
    extra_tag: str | None = None,
    fresh: bool = False,
    when: datetime | None = None,
) -> tuple[Path, Path]:
    """
    Return ``(run_root_dir, leaf_dir)`` under ``experiments/loo_encoding/runs/``.

    ``run_root`` may be an absolute path, a name under ``OUT_ROOT``, or None
    (build from date / window / model / layer).
    """
    if run_root is None:
        root_dir = repo / OUT_ROOT / flat_run_root_name(
            run_date=run_date,
            start_frame=start_frame,
            end_frame=end_frame,
            model_slug=model_slug,
            feature_layer=feature_layer,
        )
    else:
        root_path = Path(run_root)
        root_dir = (
            root_path
            if root_path.is_absolute()
            else repo / OUT_ROOT / root_path
        )

    leaf = flat_leaf_name(
        protocol=protocol,
        normalization=normalization,
        target_mask_mode=target_mask_mode,
        cleanliness=cleanliness,
        mask_path=mask_path,
        extra_tag=extra_tag,
    )
    if fresh:
        leaf_dir = uniquify_leaf_dir(root_dir, leaf, when=when)
    else:
        leaf_dir = root_dir / leaf
    return root_dir, leaf_dir


def try_git_commit(repo: Path) -> str | None:
    """Return HEAD SHA if ``git`` is available; else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_params(
    *,
    layout: str,
    run_root: Path,
    leaf_dir: Path,
    repo: Path,
    window_path: Path,
    window_id: str,
    start_frame: int,
    end_frame: int,
    normalization: str,
    model_path: Path,
    model_slug: str,
    feature_layer: str,
    protocol: str,
    heldout_stimuli: Sequence[str],
    target_mask_mode: str,
    target_mask_path: Path | None,
    roi_dir: Path | str | None,
    trial_cleanliness_csv: Path | str | None,
    trial_cleanliness_keep: Sequence[str] | None,
    ridge_path: Path,
    ridge_cfg: dict[str, Any] | None,
    cli_argv: Sequence[str],
    save_model: bool,
    run_tag: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble ``params.yaml`` payload for a leaf directory."""

    def _rel(path: Path | str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        try:
            return str(p.resolve().relative_to(repo.resolve()))
        except ValueError:
            return str(p)

    payload: dict[str, Any] = {
        "layout": layout,
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": try_git_commit(repo),
        "cli_argv": list(cli_argv),
        "window_config": _rel(window_path),
        "window_id": window_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "normalization": normalization,
        "model_config": _rel(model_path),
        "model_slug": model_slug,
        "model_slug_short": short_model_slug(model_slug),
        "feature_layer": feature_layer,
        "feature_layer_short": short_layer_slug(feature_layer),
        "protocol": protocol,
        "heldout_stimuli": list(heldout_stimuli),
        "loss_roi": target_mask_mode,
        "loss_roi_path": _rel(target_mask_path) if target_mask_path else None,
        "roi_dir": _rel(roi_dir) if roi_dir is not None else None,
        "trial_cleanliness_csv": _rel(trial_cleanliness_csv),
        "trial_cleanliness_keep": (
            list(trial_cleanliness_keep)
            if trial_cleanliness_keep is not None
            else None
        ),
        "run_tag": run_tag,
        "ridge_config": _rel(ridge_path),
        "ridge": ridge_cfg,
        "seed": seed,
        "save_model": bool(save_model),
        "run_root": _rel(run_root),
        "output_dir": _rel(leaf_dir),
    }
    if extra:
        payload.update(extra)
    return payload
