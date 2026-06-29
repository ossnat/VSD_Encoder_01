#!/usr/bin/env python3
"""Visualize Gabor wavelet pyramid energy maps per scale (exploratory only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.DL_features.gabor_gwp import GaborWaveletPyramid
from src.DL_features.preprocess import preprocess_stimulus
from src.paths import project_root, resolve_data_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _repo_root() -> Path:
    return project_root()


def _exploratory_root() -> Path:
    return _repo_root() / "exploratory" / "gwp_visualization"


def _select_stimuli(manifest_path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    picks: list[dict] = []
    for shape_type, label in (
        ("filled_circle", "filled_circle_black"),
        ("circle_contour", "circle_contour_black"),
        ("triangle_contour", "triangle_contour_black"),
    ):
        sub = df[(df["shape_type"] == shape_type) & (df["color"] == "black")]
        if sub.empty:
            sub = df[df["shape_type"] == shape_type]
        if sub.empty:
            continue
        row = sub.sort_values("h5_session").iloc[0]
        picks.append(
            {
                "key": label,
                "shape_type": shape_type,
                "condition": row["condition"],
                "h5_session": row["h5_session"],
                "stimulus_text": row["stimulus_text"],
                "image_path": row["image_path"],
            }
        )
    if not picks:
        raise RuntimeError("No circle/triangle stimuli found in manifest")
    return picks


def _energy_tensor(image_path: Path, model_cfg: dict, repo: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (luminance H×W, energy C×H×W)."""
    abs_path = resolve_data_path(str(image_path), repo)
    rgb = np.asarray(Image.open(abs_path).convert("RGB"), dtype=np.uint8)
    input_size = int(model_cfg.get("input_size", 224))
    tensor = preprocess_stimulus(
        rgb,
        model_cfg=model_cfg,
        input_size=input_size,
        imagenet_normalize=False,
    )
    gwp_cfg = model_cfg.get("gwp", {}) or {}
    pyramid = GaborWaveletPyramid(
        n_scales=int(gwp_cfg.get("number_of_scales", 5)),
        n_orientations=int(gwp_cfg.get("number_of_directions", 8)),
        kernel_size=int(gwp_cfg.get("kernel_size", 31)),
        min_wavelength=float(gwp_cfg.get("min_wavelength", 3.0)),
        wavelength_factor=float(gwp_cfg.get("wavelength_factor", 2**0.5)),
    )
    pyramid.eval()
    with torch.no_grad():
        energy = pyramid(tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
    lum = tensor.squeeze(0).numpy()
    return lum, energy


def _scale_maps(energy: np.ndarray, n_scales: int, n_orientations: int) -> np.ndarray:
    """Reshape (C,H,W) -> (n_scales, n_orientations, H, W)."""
    c, h, w = energy.shape
    expected = n_scales * n_orientations
    if c != expected:
        raise ValueError(f"Expected {expected} channels, got {c}")
    return energy.reshape(n_scales, n_orientations, h, w)


def _plot_stimulus_overview(
    *,
    lum: np.ndarray,
    scale_max: np.ndarray,
    meta: dict,
    wavelengths: list[float],
    out_path: Path,
) -> None:
    n_scales = scale_max.shape[0]
    ncols = max(3, n_scales + 1)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 4.5))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(lum, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Luminance input")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    vmax = float(np.percentile(scale_max, 99.5))
    vmax = vmax if vmax > 1e-8 else 1.0
    for s in range(n_scales):
        ax = axes[0, s + 1]
        ax.imshow(scale_max[s], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"Scale {s}\nλ≈{wavelengths[s]:.1f}px")
        ax.axis("off")
        axes[1, s + 1].axis("off")

    for col in range(n_scales + 2, ncols):
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    fig.suptitle(
        f"{meta['key']} | {meta['h5_session']} {meta['condition']}\n"
        f"{meta['stimulus_text']}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_orientation_fan(
    scale_maps: np.ndarray,
    *,
    scale_idx: int,
    meta: dict,
    out_path: Path,
) -> None:
    n_orientations = scale_maps.shape[1]
    angles = np.linspace(0, 180, n_orientations, endpoint=False)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    vmax = float(np.percentile(scale_maps[scale_idx], 99.5))
    vmax = vmax if vmax > 1e-8 else 1.0
    for i, ax in enumerate(axes.ravel()):
        if i >= n_orientations:
            ax.axis("off")
            continue
        ax.imshow(scale_maps[scale_idx, i], cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{angles[i]:.0f}°")
        ax.axis("off")
    fig.suptitle(
        f"Orientations at scale {scale_idx} | {meta['key']}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _wavelengths_px(model_cfg: dict) -> list[float]:
    gwp_cfg = model_cfg.get("gwp", {}) or {}
    n_scales = int(gwp_cfg.get("number_of_scales", 5))
    min_wl = float(gwp_cfg.get("min_wavelength", 3.0))
    factor = float(gwp_cfg.get("wavelength_factor", 2**0.5))
    return [min_wl * (factor**s) for s in range(n_scales)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs/default.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_repo_root() / "configs/models/gabor_serre.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_exploratory_root() / "results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = _repo_root()
    base_cfg = _load_yaml(args.config)
    model_cfg = _load_yaml(args.model)
    monkey = base_cfg["monkey"]
    stimuli_root = resolve_data_path(base_cfg["paths"]["stimuli_root"], repo)
    manifest_path = stimuli_root / monkey / "manifest.parquet"

    gwp_cfg = model_cfg.get("gwp", {}) or {}
    n_scales = int(gwp_cfg.get("number_of_scales", 5))
    n_orientations = int(gwp_cfg.get("number_of_directions", 8))
    wavelengths = _wavelengths_px(model_cfg)

    picks = _select_stimuli(manifest_path)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# GWP layer visualization",
        "",
        f"Model: `{args.model.relative_to(repo)}`",
        f"Scales: {n_scales}, orientations: {n_orientations}",
        f"Wavelengths (px): {', '.join(f'{w:.1f}' for w in wavelengths)}",
        "",
        "## Stimuli",
    ]

    for meta in picks:
        lum, energy = _energy_tensor(meta["image_path"], model_cfg, repo)
        scale_maps = _scale_maps(energy, n_scales, n_orientations)
        scale_max = scale_maps.max(axis=1)

        stem = meta["key"]
        overview_path = out_dir / f"{stem}_scales.png"
        orient_path = out_dir / f"{stem}_scale2_orientations.png"
        mid_scale = min(2, n_scales - 1)

        _plot_stimulus_overview(
            lum=lum,
            scale_max=scale_max,
            meta=meta,
            wavelengths=wavelengths,
            out_path=overview_path,
        )
        _plot_orientation_fan(
            scale_maps,
            scale_idx=mid_scale,
            meta=meta,
            out_path=orient_path,
        )

        np.savez_compressed(
            out_dir / f"{stem}_energy.npz",
            luminance=lum,
            energy=energy,
            scale_maps=scale_maps,
            wavelengths=np.asarray(wavelengths, dtype=np.float32),
            **{k: str(v) for k, v in meta.items()},
        )

        summary_lines.append(
            f"- **{stem}**: `{overview_path.name}`, `{orient_path.name}` "
            f"({meta['h5_session']} {meta['condition']})"
        )
        print(f"Wrote {overview_path.relative_to(repo)}")
        print(f"Wrote {orient_path.relative_to(repo)}")

    (out_dir / "README.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Summary: {(out_dir / 'README.md').relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
