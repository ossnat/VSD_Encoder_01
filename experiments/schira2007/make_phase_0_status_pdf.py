#!/usr/bin/env python3
"""Short Phase 0 status PDF: goal, data, attempts, results, next steps + figures.

Usage:
  scripts/py experiments/schira2007/make_phase_0_status_pdf.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "phase_0_status_summary.pdf"

FIGURES = {
    "montage": HERE / "phase_0_register/overlays_pooled/montage__201118__pooled.png",
    "fig13": HERE / "phase_0_register/fig13_caption_schira_vs_thesis_vsd.png",
    "schira_f": HERE / "phase_0_schira/201118__letter_F_white_1.png",
    "residual": HERE / "phase_0_register/landmark_residual_diag__201118__letter_F_white_1.png",
}


def _show_image(ax, path: Path, title: str | None = None) -> None:
    if not path.exists():
        ax.text(0.5, 0.5, f"Missing: {path.name}", ha="center", va="center")
        ax.axis("off")
        return
    ax.imshow(imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8.5, pad=3)


def make_status_pdf(output_path: Path = DEFAULT_OUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page1_text = (
        "GOAL\n"
        "Validate Schira 2007 retinotopy on our VSDI data: forward-map stimulus ink\n"
        "to model cortex (w = u + iv), then register onto empirical VSD maps\n"
        "(thesis Fig. 13: thin black model overlay on mean raw VSD).\n\n"
        "DATA\n"
        "• Encoder trials, monkey; mean raw VSD 100×100, evoked frames [35, 46)\n"
        "• Session 20/11/18 (set 201118): letters D, F, L, N (white contours)\n"
        "• Schira YAML: a=0.72, α=1.5, k=1, sech_amp=0.1821, fa_combine=power\n"
        "• Thesis reference: Fig. 10 landmark guide, Fig. 11 ~1.5 px RMSD (chamber)\n\n"
        "WHAT WE TRIED\n"
        "1. Schira-only: catalog ink → (u,v), no camera affine (Step 1)\n"
        "2. Camera similarity: w → VSD (origin, scale, rotation, flips)\n"
        "   · Legacy GUI: stimulus image ↔ VSD clicks\n"
        "   · Current GUI: Schira forward ink (u,v) ↔ VSD clicks (--pick-schira)\n"
        "3. Pooled session fit: 17 landmarks, 4 letters → affine_fit__pooled.yaml\n"
        "4. Overlay QA: thin forward ink through pooled affine (figure →)\n\n"
        "RESULTS (201118, pooled affine)\n"
        "• Pooled RMSD ≈ 9.7 px  (thesis chamber ≈ 1.5 px)\n"
        "• Per letter: D 8.7 | F 9.4 | L 10.5 | N 10.1 px\n"
        "• Affine: origin (−10.5, 19.6), ppu 61.7, rot 127.4°, flip_u=True\n"
        "• Schira shapes qualitatively OK; corners ≠ VF 90° (log-polar mapping)\n"
        "• Global pose on VSD reasonable; local offset on activity blobs\n\n"
        "MISSING\n"
        "• Consistent landmark semantics (hotspot vs ridge/corner)\n"
        "• Dense / contour alignment (chamfer, ICP) not implemented\n"
        "• Pooled affine not yet in configs/schira/sets.yaml\n"
        "• 10/07/18 bars; L at thesis caption pos (1.4, −0.7) vs catalog\n\n"
        "NEXT\n"
        "• Overlay-first picking or automatic pose on ink cloud vs VSD ridge\n"
        "• Re-fit; copy affine → sets.yaml; run_phase_0.py full overlays\n"
        "• Compare at thesis Fig. 13 caption positions where needed"
    )

    with PdfPages(output_path) as pdf:
        # Page 1 — narrative left, pooled montage right
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig.text(
            0.025,
            0.975,
            page1_text,
            transform=fig.transFigure,
            fontsize=7.4,
            va="top",
            family="sans-serif",
            linespacing=1.32,
        )
        ax_r = fig.add_axes([0.52, 0.05, 0.46, 0.9])
        _show_image(
            ax_r,
            FIGURES["montage"],
            "Pooled overlays · thin black Schira ink on mean VSD",
        )
        fig.suptitle(
            "Schira Phase 0 status · session 20/11/18",
            fontsize=11,
            fontweight="bold",
            y=0.99,
        )
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 2 — supporting figures
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        ax_top = fig.add_axes([0.02, 0.52, 0.96, 0.44])
        _show_image(
            ax_top,
            FIGURES["fig13"],
            "Thesis VSD+model vs our Schira at Fig. 13 caption positions (side-by-side, unregistered)",
        )
        ax_bl = fig.add_axes([0.02, 0.05, 0.47, 0.42])
        _show_image(
            ax_bl,
            FIGURES["schira_f"],
            "Step 1 · Schira-only forward ink → (u,v), letter F",
        )
        ax_br = fig.add_axes([0.51, 0.05, 0.47, 0.42])
        _show_image(
            ax_br,
            FIGURES["residual"],
            "Landmark residual diagnostic (F, legacy stimulus↔VSD picks)",
        )
        fig.suptitle("Pipeline context & diagnostics", fontsize=11, y=0.99)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Schira Phase 0 Status Summary"
        d["Author"] = "VSD_Encoder_01 / schira2007"
        d["Subject"] = "Phase 0 registration status"
        d["CreationDate"] = date.today().isoformat()

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    make_status_pdf(args.output)


if __name__ == "__main__":
    main()
