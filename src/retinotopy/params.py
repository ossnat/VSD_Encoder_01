"""Load named Schira + affine YAML (reusable by later phases).

Two file shapes are accepted:

1. Registry (``configs/schira/sets.yaml``)::

       default_set: 201118
       sets:
         201118:
           date_prefix: "201118"
           schira: {a, alpha, k, shear}
           affine: {origin_xy, pixels_per_unit, ...}

2. Standalone single-set file (top-level ``schira:`` / ``affine:``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.retinotopy.affine import CorticalAffine
from src.retinotopy.schira import (
    DEFAULT_FA_COMBINE,
    DEFAULT_SECH_AMP,
    DEFAULT_SECH_ECC_K,
    SHEAR_DOUBLE_SECH,
    SchiraParams,
)

_REQUIRED_SCHIRA = ("a", "alpha", "k")
_REQUIRED_AFFINE = ("origin_xy", "pixels_per_unit")


def _require_keys(data: dict, keys: tuple[str, ...], *, where: str) -> None:
    missing = [k for k in keys if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"{where} missing required key(s): {missing}")


def params_from_mapping(data: dict) -> SchiraParams:
    """Build :class:`SchiraParams` from a YAML mapping. No numeric fallbacks."""
    _require_keys(data, _REQUIRED_SCHIRA, where="schira")
    return SchiraParams(
        a=float(data["a"]),
        alpha=float(data["alpha"]),
        k=float(data["k"]),
        shear=str(data.get("shear", SHEAR_DOUBLE_SECH)),
        sech_ecc_k=float(data.get("sech_ecc_k", DEFAULT_SECH_ECC_K)),
        sech_amp=float(data.get("sech_amp", DEFAULT_SECH_AMP)),
        fa_combine=str(data.get("fa_combine", DEFAULT_FA_COMBINE)),
    )


def affine_from_mapping(data: dict) -> CorticalAffine:
    """Build :class:`CorticalAffine` from a YAML mapping. No origin/scale fallbacks."""
    _require_keys(data, _REQUIRED_AFFINE, where="affine")
    return CorticalAffine.from_mapping(data)


def _is_registry(raw: dict) -> bool:
    return isinstance(raw.get("sets"), dict)


def list_schira_sets(path: Path) -> list[str]:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if _is_registry(raw):
        return sorted(str(k) for k in raw["sets"])
    name = raw.get("set_name") or path.stem
    return [str(name)]


def load_schira_set(
    path: Path,
    set_name: str | None = None,
) -> tuple[str, SchiraParams, CorticalAffine, dict]:
    """Load one named parameter set.

    Returns ``(set_name, params, affine, set_yaml)``.
    """
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Schira config must be a mapping: {path}")

    if _is_registry(raw):
        sets = {str(k): v for k, v in raw["sets"].items()}
        name = set_name or raw.get("default_set")
        if not name:
            raise ValueError(
                f"{path} has named sets {sorted(sets)} but no --set / default_set"
            )
        name = str(name)
        if name not in sets:
            raise KeyError(
                f"Unknown Schira set {name!r} in {path}. "
                f"Available: {sorted(sets)}"
            )
        block = sets[name]
        if not isinstance(block, dict):
            raise ValueError(f"Set {name!r} in {path} must be a mapping")
    else:
        if set_name is not None and str(set_name) not in {
            str(raw.get("set_name", "")),
            path.stem,
            str(raw.get("date_prefix", "")),
        }:
            raise KeyError(
                f"{path} is a single-set file; cannot select set {set_name!r}"
            )
        name = str(set_name or raw.get("set_name") or raw.get("date_prefix") or path.stem)
        block = raw

    schira_cfg = block.get("schira")
    affine_cfg = block.get("affine")
    if not isinstance(schira_cfg, dict):
        raise ValueError(f"Set {name!r} missing 'schira' mapping")
    if not isinstance(affine_cfg, dict):
        raise ValueError(f"Set {name!r} missing 'affine' mapping")
    params = params_from_mapping(schira_cfg)
    affine = affine_from_mapping(affine_cfg)
    return name, params, affine, block


def load_session_retinotopy(
    path: Path,
    set_name: str | None = None,
) -> tuple[SchiraParams, CorticalAffine, dict]:
    """Backward-compatible wrapper: ``(params, affine, set_yaml)``."""
    _name, params, affine, block = load_schira_set(path, set_name=set_name)
    return params, affine, block
