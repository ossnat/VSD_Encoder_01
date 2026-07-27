"""Compare CORnet layer runs: metrics plus Ridge weight/alpha spatial summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.encoding.ridge import alpha_map, weight_norm_map
from src.encoding.schema import ridge_output_dir
from src.evaluation.mask import mask_from_eval_cfg
from src.paths import resolve_data_path


def center_periphery_masks(
    spatial_size: tuple[int, int],
    *,
    eval_mask: np.ndarray | None,
    center_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the evaluation disk into a central disk and peripheral annulus.

    ``center_frac`` is the radius of the center disk as a fraction of the
    evaluation-mask radius (or half the shorter image side if no mask).
    """
    height, width = spatial_size
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if eval_mask is not None:
        mask = eval_mask.astype(bool)
        if not mask.any():
            raise ValueError("evaluation mask is empty")
        # Approximate eval radius from masked pixels.
        radii = dist[mask]
        r_eval = float(np.percentile(radii, 98))
    else:
        mask = np.ones((height, width), dtype=bool)
        r_eval = 0.5 * min(height, width)

    r_center = max(1.0, float(center_frac) * r_eval)
    center = mask & (dist <= r_center)
    periphery = mask & (dist > r_center)
    return center, periphery


def center_periphery_radii(
    spatial_size: tuple[int, int],
    *,
    eval_mask: np.ndarray | None,
    center_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Return (cx, cy, r_eval, r_center) for definition overlays."""
    height, width = spatial_size
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if eval_mask is not None:
        mask = eval_mask.astype(bool)
        r_eval = float(np.percentile(dist[mask], 98))
    else:
        r_eval = 0.5 * min(height, width)
    r_center = max(1.0, float(center_frac) * r_eval)
    return cx, cy, r_eval, r_center


def plot_center_periphery_definition(
    spatial_size: tuple[int, int],
    output_path: Path,
    *,
    eval_mask: np.ndarray | None,
    center_frac: float = 0.5,
    underlay: np.ndarray | None = None,
) -> Path:
    """
    Draw the geometric center-disk vs peripheral-annulus definition.

    This is NOT an anatomical V1/V2/V4 ROI — only a split of the evaluation circle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = spatial_size
    center, periphery = center_periphery_masks(
        spatial_size, eval_mask=eval_mask, center_frac=center_frac
    )
    cx, cy, r_eval, r_center = center_periphery_radii(
        spatial_size, eval_mask=eval_mask, center_frac=center_frac
    )

    label_map = np.full((height, width), np.nan, dtype=float)
    label_map[periphery] = 1.0
    label_map[center] = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    if underlay is not None:
        axes[0].imshow(underlay, cmap="gray")
        axes[0].set_title("Underlay (optional VSD)")
    else:
        axes[0].imshow(np.zeros((height, width)), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Map geometry")
    circ_eval = plt.Circle((cx, cy), r_eval, fill=False, color="cyan", linewidth=2, label="eval disk")
    circ_ctr = plt.Circle(
        (cx, cy), r_center, fill=False, color="orange", linewidth=2, label="center disk"
    )
    axes[0].add_patch(circ_eval)
    axes[0].add_patch(circ_ctr)
    axes[0].plot([cx], [cy], "r+", markersize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlim(-0.5, width - 0.5)
    axes[0].set_ylim(height - 0.5, -0.5)
    axes[0].axis("off")

    im = axes[1].imshow(label_map, cmap="coolwarm", vmin=1, vmax=2)
    axes[1].set_title(
        f"Center (orange) vs periphery (blue)\n"
        f"center_frac={center_frac:.2f} | r_eval≈{r_eval:.1f} | r_center≈{r_center:.1f}"
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, ticks=[1, 2])
    cbar.ax.set_yticklabels(["periphery", "center"])

    fig.suptitle(
        "Center/periphery definition (geometric proxy, not anatomical V1/V2/V4)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def spatial_band_stats(
    spatial_map: np.ndarray,
    center: np.ndarray,
    periphery: np.ndarray,
) -> dict[str, float]:
    c_vals = spatial_map[center]
    p_vals = spatial_map[periphery]
    c_mean = float(np.mean(c_vals)) if c_vals.size else float("nan")
    p_mean = float(np.mean(p_vals)) if p_vals.size else float("nan")
    ratio = c_mean / p_mean if np.isfinite(p_mean) and p_mean != 0 else float("nan")
    return {
        "center_mean": c_mean,
        "periphery_mean": p_mean,
        "center_over_periphery": ratio,
        "n_center": int(center.sum()),
        "n_periphery": int(periphery.sum()),
    }


def load_ridge_spatial_maps(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug: str,
    feature_layer: str,
) -> dict[str, Any]:
    """Load weight-norm and (if available) alpha maps from a trained ridge run."""
    import joblib

    ridge_root = resolve_data_path(cfg["paths"]["ridge_encode_root"], repo)
    ridge_dir = ridge_output_dir(
        ridge_root, cfg["monkey"], window_id, model_slug, feature_layer
    )
    model_path = ridge_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ridge model: {model_path}")

    payload = joblib.load(model_path)
    result = payload["result"] if isinstance(payload, dict) and "result" in payload else payload
    spatial_size = tuple(result.spatial_size)
    wnorm = weight_norm_map(result, spatial_size)
    out: dict[str, Any] = {
        "ridge_dir": ridge_dir,
        "spatial_size": spatial_size,
        "weight_norm": wnorm,
        "alpha": None,
    }
    if bool(getattr(result, "alpha_per_target", False)):
        out["alpha"] = alpha_map(result, spatial_size)
    return out


def summarize_layer_spatial_maps(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug: str,
    feature_layers: list[str],
    ridge_cfg: dict | None = None,
    center_frac: float = 0.5,
) -> pd.DataFrame:
    """Per-layer center/periphery stats for weight-norm (and alpha if present)."""
    ridge_cfg = ridge_cfg or {}
    rows: list[dict[str, Any]] = []
    first_size: tuple[int, int] | None = None
    eval_mask = None

    for layer in feature_layers:
        maps = load_ridge_spatial_maps(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=model_slug,
            feature_layer=layer,
        )
        spatial_size = maps["spatial_size"]
        if first_size is None:
            first_size = spatial_size
            eval_mask = mask_from_eval_cfg(
                ridge_cfg.get("evaluation", {}), spatial_size
            )
        center, periphery = center_periphery_masks(
            spatial_size, eval_mask=eval_mask, center_frac=center_frac
        )
        w_stats = spatial_band_stats(maps["weight_norm"], center, periphery)
        row: dict[str, Any] = {
            "model_slug": model_slug,
            "feature_layer": layer,
            "weight_center_mean": w_stats["center_mean"],
            "weight_periphery_mean": w_stats["periphery_mean"],
            "weight_center_over_periphery": w_stats["center_over_periphery"],
        }
        if maps["alpha"] is not None:
            a_stats = spatial_band_stats(maps["alpha"], center, periphery)
            row.update(
                {
                    "alpha_center_mean": a_stats["center_mean"],
                    "alpha_periphery_mean": a_stats["periphery_mean"],
                    "alpha_center_over_periphery": a_stats["center_over_periphery"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_weight_alpha_grid(
    *,
    repo: Path,
    cfg: dict,
    window_id: str,
    model_slug: str,
    feature_layers: list[str],
    output_path: Path,
    ridge_cfg: dict | None = None,
) -> Path:
    """Side-by-side weight-norm (and alpha) maps across CORnet layers."""
    ridge_cfg = ridge_cfg or {}
    n = len(feature_layers)
    has_alpha = False
    loaded: list[dict[str, Any]] = []
    for layer in feature_layers:
        maps = load_ridge_spatial_maps(
            repo=repo,
            cfg=cfg,
            window_id=window_id,
            model_slug=model_slug,
            feature_layer=layer,
        )
        loaded.append(maps)
        if maps["alpha"] is not None:
            has_alpha = True

    nrows = 2 if has_alpha else 1
    fig, axes = plt.subplots(
        nrows, n, figsize=(max(3.2 * n, 6), 3.4 * nrows), squeeze=False
    )
    spatial_size = loaded[0]["spatial_size"]
    eval_mask = mask_from_eval_cfg(ridge_cfg.get("evaluation", {}), spatial_size)

    for col, (layer, maps) in enumerate(zip(feature_layers, loaded)):
        w = maps["weight_norm"].astype(float)
        if eval_mask is not None:
            w = np.where(eval_mask, w, np.nan)
        im0 = axes[0, col].imshow(w, cmap="magma")
        axes[0, col].set_title(f"{layer}\n||w||₂")
        axes[0, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        if has_alpha:
            a = maps["alpha"]
            if a is None:
                axes[1, col].axis("off")
                axes[1, col].set_title(f"{layer}\nalpha N/A")
            else:
                aa = a.astype(float)
                if eval_mask is not None:
                    aa = np.where(eval_mask, aa, np.nan)
                im1 = axes[1, col].imshow(np.log10(np.clip(aa, 1e-12, None)), cmap="viridis")
                axes[1, col].set_title(f"{layer}\nlog10(α)")
                axes[1, col].axis("off")
                fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"CORnet Ridge spatial maps | {model_slug} | {window_id}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_center_periphery_bars(
    stats_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Bar chart of center/periphery weight-norm ratio by feature layer."""
    if stats_df.empty or "weight_center_over_periphery" not in stats_df.columns:
        return output_path
    labels = [
        f"{row.model_slug}\n{row.feature_layer}"
        if "model_slug" in stats_df.columns
        else str(row.feature_layer)
        for row in stats_df.itertuples(index=False)
    ]
    vals = pd.to_numeric(
        stats_df["weight_center_over_periphery"], errors="coerce"
    ).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    ax.bar(range(len(labels)), np.where(np.isfinite(vals), vals, 0.0), color="steelblue")
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("center / periphery ||w||₂")
    ax.set_title(title)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
