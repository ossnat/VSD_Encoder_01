"""Resolve workspace-relative Data/ paths (see docs/DATA_LAYOUT.md)."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def workspace_root(project_root_path: Path | None = None) -> Path:
    """Parent of repo; contains Data/ and code repos."""
    root = project_root_path or project_root()
    return root.resolve().parent


def resolve_data_path(path: str | Path, project_root_path: Path | None = None) -> Path:
    """Resolve portable Data/... paths relative to workspace."""
    root = project_root_path or project_root()
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p.resolve()
    if str(path).startswith("Data/"):
        return (workspace_root(root) / path).resolve()
    return (root / path).resolve()
