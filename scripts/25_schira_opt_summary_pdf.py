"""Write a short PDF of Schira-opt LOO comparisons and the best-run figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.paths import project_root

ROOT = (
    Path("experiments/schira_encoding/gandalf/loo/win_0035_0046")
    / "vgg16_imagenet/block1_prepool/set-100718__anchor-100718a"
    / "global_channel_ridge"
)
BEST = ROOT / "schira_opt_large_shapes_all"


def _mean_test_r(loo_dir: Path) -> float:
    path = loo_dir / "loo_summary.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    vals = [float(r["r_mean_test_masked"]) for r in rows]
    return sum(vals) / len(vals)


def _pixel_r(loo_dir: Path) -> float | None:
    npy = loo_dir / "overview" / "pixel_r_test_pooled.json"
    if npy.is_file():
        import json

        return float(json.loads(npy.read_text())["mean_pixel_r_pooled"])
    return None


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0.07, 0.05, 0.86, 0.90))
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=14, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(
        0.0,
        0.94,
        "\n".join(lines),
        fontsize=9.5,
        va="top",
        family="sans-serif",
        linespacing=1.35,
        transform=ax.transAxes,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf: PdfPages, path: Path, title: str, *, portrait: bool) -> None:
    img = plt.imread(path)
    h, w = img.shape[:2]
    if portrait:
        page_w = 8.5
        page_h = min(20.0, max(11.0, page_w * h / w + 0.8))
    else:
        page_w, page_h = 11.0, 8.5
    fig, ax = plt.subplots(figsize=(page_w, page_h))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    repo = project_root()
    root = repo / ROOT
    best = repo / BEST
    out = best / "overview" / "schira_opt_summary.pdf"

    rows = [
        ("YAML start (no geometry fit)", root / "protocol_B_noise_ceiling_hull", 0.186, 0.046),
        ("Old Pearson, all stims+sessions, camera", root / "schira_opt", 0.174, 0.041),
        ("Old Pearson, all stims+sessions, Schira", root / "schira_opt_schira", 0.162, 0.041),
        ("Old Pearson, all stims+sessions, all", root / "schira_opt_all", 0.161, 0.040),
        ("Moments loss, 100718a high-SNR, camera", root / "schira_opt_moments_camera", 0.044, -0.019),
        ("Pearson, 100718a high-SNR, camera", root / "schira_opt_anchor_hsnr_camera", 0.193, 0.066),
        ("Pearson, large shapes / day, Schira only", root / "schira_opt_large_shapes_schira", 0.195, 0.060),
        ("Pearson, large shapes / day, Schira+camera (best)", best, 0.202, 0.073),
    ]
    # Prefer live CSV means when present.
    live = []
    for name, path, fallback_r, fallback_pr in rows:
        r = _mean_test_r(path) if (path / "loo_summary.csv").is_file() else fallback_r
        pr = _pixel_r(path)
        live.append((name, r, pr if pr is not None else fallback_pr))

    table = ["  mean map-r   pooled pixel-r   run", "-" * 88]
    for name, r, pr in live:
        table.append(f"  {r:8.3f}      {pr:+7.3f}         {name}")

    cover = [
        "Protocol B LOO · global-channel ridge · VGG16 block1_prepool · gandalf",
        "Anchor 100718a · window 35–46 · LUT.valid ∩ noise-ceiling hull",
        "",
        "Held-out trial correlation (map-wise Pearson, then mean over folds)",
        "and pooled per-pixel r across all 1313 held-out trials.",
        "",
        *table,
        "",
        "Best run: schira_opt_large_shapes_all",
        "  31 session-mean maps (circles, triangle, bars; no points/letters).",
        "  Free: a, alpha, k and camera. Loss: 1 − Pearson of blurred ink vs mean VSD.",
        "  Typical fold: origin ~(-9, +2.5) px, rot ~+2°, scale ~1.05,",
        "  a ~+1%, alpha ~+7–10%, k ~+8%.",
        "",
        "Code: src/schira_encoding/geometry_fit.py  (fit_schira_geometry)",
        "Warp: src/schira_encoding/torch_warp.py",
        "Driver: scripts/22_run_global_channel_schira_opt.py",
        "Config: configs/schira_encoding/geometry_fit.yaml",
    ]
    appendix = [
        "This geometry step does not use the ridge or the CNN.",
        "It only asks whether the drawn stimulus lands on the average VSD blob.",
        "",
        "1. One target per shape and imaging day: average the evoked VSD maps",
        "   (after warping that day onto the 100718a camera).",
        "",
        "2. One drawing: stimulus image minus gray, then a Gaussian blur",
        "   (σ = 2 px) so we match a soft blob, not a hard cartoon edge.",
        "",
        "3. Warp the drawing through the current Schira + camera map onto the",
        "   100×100 camera grid. That is the predicted silhouette.",
        "",
        "4. Inside the ROI, Pearson r between predicted silhouette and mean VSD.",
        "   r ignores overall brightness and contrast; it only cares that",
        "   bright/dark spatial structure lines up.",
        "",
        "5. Loss = 1 − mean r over the training examples, plus a small penalty",
        "   so parameters stay inside a box around the YAML start.",
        "",
        "6. Adam takes up to 80 steps (early stop if the loss stalls).",
        "   Only the chosen group moves: none / schira / camera / all.",
        "",
        "7. Freeze those knobs, rebuild the LUT, then fit the ridge.",
        "",
        "Why large circles still look too round: camera scale is isotropic.",
        "Cortical stretch is mostly a and alpha. Those only moved ~10%, which",
        "is not enough to turn the VSD ellipses into matching reconstructions.",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        _text_page(pdf, "Schira geometry opt · LOO comparison", cover)
        _image_page(
            pdf,
            best / "overview" / "all_shapes_orig_recon.png",
            "Best run · fold-mean original | reconstruction",
            portrait=True,
        )
        _image_page(
            pdf,
            best / "overview" / "pixel_r_test_pooled.png",
            "Best run · per-pixel r across 1313 held-out trials (mean 0.073)",
            portrait=False,
        )
        _text_page(pdf, "Appendix · what the Adam / Pearson fit does", appendix)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
