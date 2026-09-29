"""Path helpers for Schira-based encoding artifacts."""

from __future__ import annotations

from pathlib import Path


def schira_encoding_root(root: Path, monkey: str) -> Path:
    return root / monkey


def lut_path(
    root: Path,
    monkey: str,
    *,
    schira_set: str,
    anchor_session: str,
    spatial_h: int,
    spatial_w: int,
    canvas_size: int,
    input_size: int,
) -> Path:
    """Static geometry LUT on the anchor camera grid."""
    name = (
        f"lut__set-{schira_set}__anchor-{anchor_session}"
        f"__vsd-{spatial_h}x{spatial_w}"
        f"__canvas-{canvas_size}__input-{input_size}.npz"
    )
    return schira_encoding_root(root, monkey) / "lut" / name


def warped_feature_dir(
    root: Path,
    monkey: str,
    *,
    model_slug: str,
    feature_layer: str,
    schira_set: str,
    anchor_session: str,
) -> Path:
    return (
        schira_encoding_root(root, monkey)
        / "features_warped"
        / model_slug
        / feature_layer
        / f"set-{schira_set}__anchor-{anchor_session}"
    )


def warped_feature_map_path(
    root: Path,
    monkey: str,
    *,
    model_slug: str,
    feature_layer: str,
    schira_set: str,
    anchor_session: str,
    h5_session: str,
    condition: str,
) -> Path:
    return (
        warped_feature_dir(
            root,
            monkey,
            model_slug=model_slug,
            feature_layer=feature_layer,
            schira_set=schira_set,
            anchor_session=anchor_session,
        )
        / "maps"
        / f"{h5_session}__{condition}.npy"
    )


def loo_protocol_leaf(
    protocol: str,
    loss_roi: str,
    date_prefix: str | None = None,
    run_tag: str | None = None,
) -> str:
    """Shared LOO directory leaf: ``protocol_B_noise_ceiling_hull`` (+ optional suffixes)."""
    leaf = f"protocol_{protocol}"
    if loss_roi and loss_roi not in {"none", "full"}:
        leaf = f"{leaf}_{loss_roi}"
    if date_prefix:
        leaf = f"{leaf}__dates-{date_prefix}"
    if run_tag:
        leaf = f"{leaf}__{run_tag}"
    return leaf


def _loo_set_dir(
    root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
    *,
    schira_set: str,
    anchor_session: str,
) -> Path:
    return (
        schira_encoding_root(root, monkey)
        / "loo"
        / window_id
        / model_slug
        / feature_layer
        / f"set-{schira_set}__anchor-{anchor_session}"
    )


def local_ridge_output_dir(
    root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
    *,
    schira_set: str,
    anchor_session: str,
    run_tag: str | None = None,
) -> Path:
    name = f"set-{schira_set}__anchor-{anchor_session}"
    if run_tag:
        name = f"{name}__{run_tag}"
    return (
        schira_encoding_root(root, monkey)
        / "local_ridge"
        / window_id
        / model_slug
        / feature_layer
        / name
    )


def local_ridge_loo_dir(
    root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
    *,
    schira_set: str,
    anchor_session: str,
    protocol: str,
    loss_roi: str,
    date_prefix: str | None = None,
    run_tag: str | None = None,
) -> Path:
    """LOO leaf for per-pixel local ridge."""
    return _loo_set_dir(
        root,
        monkey,
        window_id,
        model_slug,
        feature_layer,
        schira_set=schira_set,
        anchor_session=anchor_session,
    ) / loo_protocol_leaf(protocol, loss_roi, date_prefix, run_tag)


def global_channel_ridge_schira_opt_dir(
    root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
    *,
    schira_set: str,
    anchor_session: str,
    run_name: str = "schira_opt",
    optimize: str | None = None,
) -> Path:
    """Per-fold Schira fit plus global-channel ridge, beside the fixed-geometry LOO."""
    from src.schira_encoding.geometry_fit import schira_opt_run_leaf

    if optimize is None:
        leaf = str(run_name).strip() or "schira_opt"
    else:
        leaf = schira_opt_run_leaf(optimize, run_name)
    return (
        _loo_set_dir(
            root,
            monkey,
            window_id,
            model_slug,
            feature_layer,
            schira_set=schira_set,
            anchor_session=anchor_session,
        )
        / "global_channel_ridge"
        / leaf
    )


def global_channel_ridge_loo_dir(
    root: Path,
    monkey: str,
    window_id: str,
    model_slug: str,
    feature_layer: str,
    *,
    schira_set: str,
    anchor_session: str,
    protocol: str,
    loss_roi: str,
    date_prefix: str | None = None,
    run_tag: str | None = None,
) -> Path:
    """LOO leaf for global channel ridge (sibling of local-ridge LOO)."""
    return (
        _loo_set_dir(
            root,
            monkey,
            window_id,
            model_slug,
            feature_layer,
            schira_set=schira_set,
            anchor_session=anchor_session,
        )
        / "global_channel_ridge"
        / loo_protocol_leaf(protocol, loss_roi, date_prefix, run_tag)
    )
