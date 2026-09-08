#!/usr/bin/env python3
"""Phase 0 registration ONLY — fit camera affine from landmark clicks.

Schira ``(a, α, k, fa)`` stay fixed. This script does **not** produce VSD
overlays; that is a later combine step.

Workflow:
  **``--pick-schira``** (recommended): left = forward Schira ink in ``(u, v)``,
  right = mean raw VSD. Click matching pairs on the **model ink** then VSD.
  Landmarks store ``schira_u`` / ``schira_v`` directly (no stimulus re-map).

  **``--pick``** (legacy): left = stimulus image, right = VSD. Stimulus clicks
  are mapped through Schira only at ``--fit``.

  Both modes: ``--fit`` solves similarity ``w`` → VSD pixels and writes
  ``affine_fit__*.yaml`` (does not auto-edit ``configs/schira/sets.yaml``).

Usage:
  # Schira ink ↔ VSD — all Fig. 13 letters (opens GUI once per letter, then pooled fit)
  scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 \\
      --pick-schira --stimuli letter_D_white_1 letter_F_white_1 letter_L_white_1 letter_N_white_1

  # Pooled fit from landmark YAMLs already on disk (does not open GUI)
  scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 \\
      --fit-merge experiments/schira2007/phase_0_register/landmarks_schira__201118__letter_*.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.data.splits import load_trial_table
from src.data.trial_frames import load_h5_mean_frame
from src.paths import project_root, resolve_data_path
from src.retinotopy.params import load_schira_set
from src.retinotopy.register import (
    affine_to_mapping,
    fit_affine_from_w_pixels,
    landmark_mode_from_pairs,
    landmarks_pairs_to_w,
)
from src.retinotopy.warp import forward_ink_cloud_w
from src.stimuli.catalog import (
    load_full_encoder_catalog,
    stimulus_spec_from_mapping,
)
from src.stimuli.exclusions import is_excluded_encoding_trial
from src.stimuli.identity import attach_stimulus_ids
from src.stimuli.render import RenderConfig, render_stimulus

HERE = Path(__file__).resolve().parent
DEFAULT_SCHIRA = project_root() / "configs/schira/sets.yaml"
DEFAULT_CONFIG = project_root() / "configs/default.yaml"
DEFAULT_WINDOW = project_root() / "configs/windows/evoked_35_46.yaml"
DEFAULT_STIMULI = project_root() / "configs/stimuli/default.yaml"
OUT_DIR = HERE / "phase_0_register"

# Fig. 13 letter panels on 20/11/18 — used for merge hints / batch pick.
DEFAULT_POOL_LETTERS_201118 = [
    "letter_D_white_1",
    "letter_F_white_1",
    "letter_L_white_1",
    "letter_N_white_1",
]


def _resolve_landmark_paths(paths: list[Path], repo: Path) -> list[Path]:
    """Resolve paths; expand shell-style globs when the shell did not."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        raw = str(path)
        if any(ch in raw for ch in "*?[]"):
            base = path if path.is_absolute() else (repo / path)
            parent = base.parent
            matches = sorted(parent.glob(base.name))
            if not matches:
                raise FileNotFoundError(f"No landmark files match {path}")
            for match in matches:
                m = match.resolve()
                if m not in seen:
                    seen.add(m)
                    resolved.append(m)
            continue
        p = path if path.is_absolute() else (repo / path).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if p not in seen:
            seen.add(p)
            resolved.append(p)
    return resolved


def _stimulus_ids_from_args(args) -> list[str]:
    if args.stimuli:
        return list(args.stimuli)
    return [str(args.stimulus)]


def _warn_missing_pool_landmarks(
    set_id: str,
    resolved: list[Path],
    out_dir: Path,
) -> None:
    if set_id != "201118":
        return
    found: set[str] = set()
    for path in resolved:
        for stim_id in DEFAULT_POOL_LETTERS_201118:
            if stim_id in path.name:
                found.add(stim_id)
    missing = [s for s in DEFAULT_POOL_LETTERS_201118 if s not in found]
    if not missing:
        return
    print(
        "\nNote: --fit-merge only fits from landmark YAMLs already on disk. "
        "It does not open the pick GUI for other letters.",
        flush=True,
    )
    for stim_id in missing:
        print(
            f"  missing: landmarks_schira__{set_id}__{stim_id}.yaml",
            flush=True,
        )
    print(
        "Pick missing letters then re-run merge:\n"
        "  scripts/py experiments/schira2007/run_phase_0_register.py --set 201118 "
        "--pick-schira --stimuli "
        + " ".join(missing),
        flush=True,
    )


def _run_pick_schira_for_stimulus(
    *,
    repo: Path,
    set_name: str,
    schira_config: Path,
    config_path: Path,
    window_path: Path,
    stimuli_yaml: Path,
    stimulus_id: str,
    n_trials: int,
    seed: int,
    schira_canvas_size: int,
    out_dir: Path,
) -> Path:
    (
        set_id,
        params,
        _aff,
        meta,
        stim_rgb,
        mean_vsd,
        render_cfg,
        _session,
    ) = _load_stimulus_and_mean_vsd(
        repo=repo,
        set_name=set_name,
        schira_config=schira_config,
        config_path=config_path,
        window_path=window_path,
        stimuli_yaml=stimuli_yaml,
        stimulus_id=stimulus_id,
        n_trials=n_trials,
        seed=seed,
        schira_canvas_size=schira_canvas_size,
    )
    fu, fv, _w = forward_ink_cloud_w(stim_rgb, params=params, render_cfg=render_cfg)
    print(
        f"\n=== Pick Schira↔VSD: {stimulus_id} ({meta['h5_session']} "
        f"{meta['condition']}, {fu.size} ink pts, canvas={render_cfg.canvas_size}) ===",
        flush=True,
    )
    print(
        "Fig.10 guide (F): 1=upper stem∩top, 2=stem∩middle, "
        "3=bottom of stem, 4=end of top bar, 5=end of middle bar.",
        flush=True,
    )
    pairs = pick_schira_landmarks_gui(
        fu,
        fv,
        mean_vsd,
        title=f"{meta['h5_session']} {stimulus_id}",
    )
    if len(pairs) < 2:
        raise SystemExit(f"Need at least 2 pairs for {stimulus_id}; aborting.")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"mean_vsd__{meta['h5_session']}__{meta['condition']}.npy", mean_vsd)
    landmarks_path = out_dir / f"landmarks_schira__{set_id}__{stimulus_id}.yaml"
    payload = {
        **meta,
        "landmark_mode": "schira_uv",
        "schira_canvas_size": render_cfg.canvas_size,
        "pixels_per_deg": render_cfg.pixels_per_deg,
        "pairs": pairs,
        "guide": (
            "Pairs are Schira (u,v)=Re/Im(w) on forward ink ↔ VSD (col,row). "
            "Pick on model ink, not stimulus image. w used directly at --fit."
        ),
    }
    landmarks_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {landmarks_path} ({len(pairs)} pairs)", flush=True)
    return landmarks_path


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _render_config(stimuli_cfg: dict, *, canvas_size: int | None = None) -> RenderConfig:
    base_canvas = int(stimuli_cfg.get("canvas_size", 224))
    canvas = int(canvas_size) if canvas_size is not None else base_canvas
    quadrant_extent_deg = float(stimuli_cfg.get("quadrant_extent_deg", 6.0))
    base_ppd = stimuli_cfg.get("pixels_per_deg")
    if base_ppd is None:
        base_ppd = base_canvas / quadrant_extent_deg
    pixels_per_deg = float(base_ppd) * (canvas / base_canvas)
    return RenderConfig(
        canvas_size=canvas,
        pixels_per_deg=pixels_per_deg,
        quadrant_extent_deg=quadrant_extent_deg,
        background_gray=int(stimuli_cfg.get("background_gray", 128)),
        bar_length_deg=float(stimuli_cfg.get("bar_length_deg", 1.0)),
        bar_width_px=max(
            1, int(round(int(stimuli_cfg.get("bar_width_px", 1)) * canvas / base_canvas))
        ),
        contour_width_px=max(
            1,
            int(round(int(stimuli_cfg.get("contour_width_px", 1)) * canvas / base_canvas)),
        ),
        assume_size_is_diameter=bool(stimuli_cfg.get("assume_size_is_diameter", True)),
        draw_fixation=bool(stimuli_cfg.get("draw_fixation", False)),
    )


def _load_stimulus_and_mean_vsd(
    *,
    repo: Path,
    set_name: str | None,
    schira_config: Path,
    config_path: Path,
    window_path: Path,
    stimuli_yaml: Path,
    stimulus_id: str,
    n_trials: int,
    seed: int,
    schira_canvas_size: int | None = None,
) -> tuple[str, object, object, dict, np.ndarray, np.ndarray, RenderConfig, dict]:
    cfg = _load_yaml(config_path)
    cfg.update(_load_yaml(window_path))
    stimuli_cfg = _load_yaml(stimuli_yaml)
    render_cfg = _render_config(stimuli_cfg, canvas_size=schira_canvas_size)
    set_id, params, _affine, session_raw = load_schira_set(
        schira_config, set_name=set_name
    )
    date_prefix = str(session_raw["date_prefix"])

    encoder_root = resolve_data_path(cfg["paths"]["encoder_data_root"], repo)
    catalog = load_full_encoder_catalog(
        encoder_root,
        monkey=str(cfg["monkey"]),
        bar_length_deg=render_cfg.bar_length_deg,
    )
    catalog = catalog[
        catalog["h5_session"].astype(str).str.startswith(date_prefix)
    ].copy()
    catalog = catalog[~catalog["is_blank"]].copy()
    catalog = attach_stimulus_ids(catalog)
    catalog = catalog[
        ~catalog.apply(
            lambda r: is_excluded_encoding_trial(
                str(r["h5_session"]),
                str(r["condition"]),
                shape_type=str(r.get("shape_type")),
            ),
            axis=1,
        )
    ]
    hit = catalog[catalog["stimulus_id"].astype(str) == stimulus_id]
    if hit.empty:
        raise RuntimeError(
            f"No catalog rows for {stimulus_id!r} under {date_prefix!r}"
        )
    # Prefer session letter a if present
    hit = hit.sort_values("h5_session").reset_index(drop=True)
    row = hit.iloc[0]
    spec = stimulus_spec_from_mapping(row.to_dict())
    stimulus_rgb = render_stimulus(spec, render_cfg)

    trials = load_trial_table(
        cfg["split_csv"],
        str(cfg["monkey"]),
        trials_index_csv=cfg.get("trials_index_csv"),
        project_root_path=repo,
    )
    trials = trials[trials["date"].astype(str) == str(row["h5_session"])].copy()
    trials = trials[trials["condition"].astype(str) == str(row["condition"])].copy()
    available = trials["target_file"].apply(
        lambda p: resolve_data_path(p, repo).exists()
    )
    trials = trials.loc[available].reset_index(drop=True)
    if trials.empty:
        raise RuntimeError(
            f"No local H5 trials for {row['h5_session']} {row['condition']}"
        )
    ordered = trials.sort_values("trial_global_id").reset_index(drop=True)
    if len(ordered) > n_trials:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(ordered), size=n_trials, replace=False))
        ordered = ordered.iloc[idx].reset_index(drop=True)

    spatial_size = tuple(int(x) for x in cfg["spatial_size"])
    maps = [
        load_h5_mean_frame(
            target_file=str(r.target_file),
            trial_global_id=int(r.trial_global_id),
            repo=repo,
            spatial_size=spatial_size,
            start_frame=int(cfg["start_frame"]),
            end_frame=int(cfg["end_frame"]),
            avg_method=str(cfg.get("avg_method", "mean")),
            normalization="none",
        )
        for r in ordered.itertuples(index=False)
    ]
    mean_vsd = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    meta = {
        "set_name": set_id,
        "date_prefix": date_prefix,
        "h5_session": str(row["h5_session"]),
        "condition": str(row["condition"]),
        "stimulus_id": stimulus_id,
        "trial_global_ids": [int(x) for x in ordered["trial_global_id"].tolist()],
        "schira": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "fa_combine": params.fa_combine,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp": params.sech_amp,
        },
    }
    return set_id, params, _affine, meta, stimulus_rgb, mean_vsd, render_cfg, session_raw


def _ensure_interactive_matplotlib() -> str:
    import matplotlib

    backend_candidates = ("MacOSX", "QtAgg", "Qt5Agg", "TkAgg")
    chosen = None
    errors: list[str] = []
    for name in backend_candidates:
        try:
            matplotlib.use(name, force=True)
            matplotlib.backends.backend_registry.load_backend_module(name)
            chosen = name
            break
        except Exception as exc:  # noqa: BLE001 — probe only
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if chosen is None:
        detail = "; ".join(errors) if errors else "no candidates tried"
        raise RuntimeError(
            "No interactive matplotlib backend available for landmark picking. "
            "On macOS, MacOSX should work with the system/framework Python; "
            "alternatively install a Qt binding (PyQt6/PySide6) or a Python "
            f"build with Tk. Tried: {detail}"
        )
    print(f"Using matplotlib backend: {chosen}", flush=True)
    return chosen


def pick_schira_landmarks_gui(
    schira_u: np.ndarray,
    schira_v: np.ndarray,
    mean_vsd: np.ndarray,
    *,
    title: str,
) -> list[dict]:
    """Click pairs: Schira ink ``(u, v)`` (left) then VSD peak (right).

    Keys: ``z`` undo last incomplete/complete pair, close figure when done.
    """
    _ensure_interactive_matplotlib()
    import matplotlib.pyplot as plt
    from src.plotting_colormaps import VSD_CMAP

    fu = np.asarray(schira_u, dtype=np.float64).ravel()
    fv = np.asarray(schira_v, dtype=np.float64).ravel()
    ok = np.isfinite(fu) & np.isfinite(fv)
    fu, fv = fu[ok], fv[ok]
    if fu.size == 0:
        raise RuntimeError("No finite Schira ink points to display")

    u0, u1 = float(fu.min()), float(fu.max())
    v0, v1 = float(fv.min()), float(fv.max())
    span = max(u1 - u0, v1 - v0, 1e-3)
    pad = 0.15 * span
    u_lo, u_hi = u0 - pad, u1 + pad
    v_lo, v_hi = v0 - pad, v1 + pad

    pairs: list[dict] = []
    pending_schira: tuple[float, float] | None = None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    fig.canvas.manager.set_window_title(
        "Schira↔VSD: click model ink then VSD peak — z=undo"
    )
    fig.suptitle(
        title
        + "\nClick SCHIRA ink (left, u,v) then matching VSD peak (right). "
        "Fig10 F: stem∩top, stem∩middle, stem bottom, top tip, mid tip. "
        "Prefer 5 pairs. Close when done. Key z = undo.",
        fontsize=9,
    )
    axes[0].set_facecolor("white")
    axes[0].scatter(fu, fv, s=3.0, c="black", marker="s", linewidths=0, alpha=1.0)
    axes[0].set_xlim(u_lo, u_hi)
    axes[0].set_ylim(v_lo, v_hi)
    axes[0].set_aspect("equal")
    axes[0].axhline(0.0, color="0.7", lw=0.7, ls="--")
    axes[0].axvline(0.0, color="0.7", lw=0.7, ls="--")
    axes[0].set_xlabel("u (Re(w))")
    axes[0].set_ylabel("v (Im(w))")
    axes[0].set_title("Schira forward ink (model cortex)")
    finite = mean_vsd[np.isfinite(mean_vsd)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    axes[1].imshow(mean_vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
    axes[1].set_title("Mean raw VSD")
    axes[1].axis("off")

    schira_sc = axes[0].scatter([], [], c="red", s=50, zorder=5)
    vsd_sc = axes[1].scatter([], [], c="black", s=40, zorder=5)
    texts: list = []

    def _redraw() -> None:
        us = [p["schira_u"] for p in pairs]
        vs = [p["schira_v"] for p in pairs]
        cs = [p["vsd_col"] for p in pairs]
        rs = [p["vsd_row"] for p in pairs]
        schira_sc.set_offsets(np.column_stack([us, vs]) if us else np.zeros((0, 2)))
        vsd_sc.set_offsets(np.column_stack([cs, rs]) if cs else np.zeros((0, 2)))
        for t in texts:
            t.remove()
        texts.clear()
        for i, p in enumerate(pairs, start=1):
            texts.append(
                axes[0].text(
                    p["schira_u"] + 0.01 * span,
                    p["schira_v"] + 0.01 * span,
                    str(i),
                    color="red",
                    fontsize=9,
                    fontweight="bold",
                )
            )
            texts.append(
                axes[1].text(
                    p["vsd_col"] + 1,
                    p["vsd_row"] + 1,
                    str(i),
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal pending_schira
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is axes[0]:
            pending_schira = (float(event.xdata), float(event.ydata))
            print(
                f"  schira click: u={pending_schira[0]:.4f} v={pending_schira[1]:.4f}"
            )
        elif event.inaxes is axes[1]:
            if pending_schira is None:
                print("  Click the Schira ink (left) first for this pair.")
                return
            col, row = float(event.xdata), float(event.ydata)
            pairs.append(
                {
                    "schira_u": pending_schira[0],
                    "schira_v": pending_schira[1],
                    "vsd_col": col,
                    "vsd_row": row,
                }
            )
            print(
                f"  pair {len(pairs)}: schira u,v={pending_schira} → "
                f"vsd (col={col:.1f}, row={row:.1f})"
            )
            pending_schira = None
            _redraw()

    def on_key(event) -> None:
        nonlocal pending_schira
        if event.key == "z":
            if pending_schira is not None:
                pending_schira = None
                print("  cleared pending Schira click")
            elif pairs:
                pairs.pop()
                print(f"  undone; {len(pairs)} pairs left")
                _redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return pairs


def pick_landmarks_gui(
    stimulus_rgb: np.ndarray,
    mean_vsd: np.ndarray,
    *,
    title: str,
) -> list[dict]:
    """Click pairs: stimulus (left) then VSD (right). Close window when done.

    Keys: ``z`` undo last incomplete/complete pair, close figure to finish.
    """
    _ensure_interactive_matplotlib()
    import matplotlib.pyplot as plt
    from src.plotting_colormaps import VSD_CMAP

    pairs: list[dict] = []
    pending_stim: tuple[float, float] | None = None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    # Window title: Fig. 10 tip order (cheap UX; also printed to console).
    fig.canvas.manager.set_window_title(
        "Fig10: (1) stem∩top (2) stem∩middle (3) stem bottom "
        "(4) top-bar tip (5) middle-bar tip — z=undo"
    )
    fig.suptitle(
        title
        + "\nClick STIMULUS (left) then matching VSD peak (right). "
        "Fig10 order: stem∩top, stem∩middle, stem bottom, top tip, mid tip. "
        "Prefer 5 pairs. Close when done. Key z = undo.",
        fontsize=9,
    )
    axes[0].imshow(stimulus_rgb)
    axes[0].set_title("Stimulus (visual field)")
    axes[0].axis("off")
    finite = mean_vsd[np.isfinite(mean_vsd)]
    lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    axes[1].imshow(mean_vsd, cmap=VSD_CMAP, vmin=lo, vmax=hi)
    axes[1].set_title("Mean raw VSD")
    axes[1].axis("off")

    stim_sc = axes[0].scatter([], [], c="red", s=40, zorder=5)
    vsd_sc = axes[1].scatter([], [], c="black", s=40, zorder=5)
    texts: list = []

    def _redraw() -> None:
        xs = [p["stim_xy_px"][0] for p in pairs]
        ys = [p["stim_xy_px"][1] for p in pairs]
        cs = [p["vsd_col"] for p in pairs]
        rs = [p["vsd_row"] for p in pairs]
        stim_sc.set_offsets(np.column_stack([xs, ys]) if xs else np.zeros((0, 2)))
        vsd_sc.set_offsets(np.column_stack([cs, rs]) if cs else np.zeros((0, 2)))
        for t in texts:
            t.remove()
        texts.clear()
        for i, p in enumerate(pairs, start=1):
            texts.append(
                axes[0].text(
                    p["stim_xy_px"][0] + 2,
                    p["stim_xy_px"][1] + 2,
                    str(i),
                    color="yellow",
                    fontsize=9,
                    fontweight="bold",
                )
            )
            texts.append(
                axes[1].text(
                    p["vsd_col"] + 1,
                    p["vsd_row"] + 1,
                    str(i),
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal pending_stim
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is axes[0]:
            pending_stim = (float(event.xdata), float(event.ydata))
            print(f"  stim click: col={pending_stim[0]:.1f} row={pending_stim[1]:.1f}")
        elif event.inaxes is axes[1]:
            if pending_stim is None:
                print("  Click the stimulus (left) first for this pair.")
                return
            col, row = float(event.xdata), float(event.ydata)
            pairs.append(
                {
                    "stim_xy_px": [pending_stim[0], pending_stim[1]],
                    "vsd_col": col,
                    "vsd_row": row,
                }
            )
            print(
                f"  pair {len(pairs)}: stim={pending_stim} → "
                f"vsd (col={col:.1f}, row={row:.1f})"
            )
            pending_stim = None
            _redraw()

    def on_key(event) -> None:
        nonlocal pending_stim
        if event.key == "z":
            if pending_stim is not None:
                pending_stim = None
                print("  cleared pending stimulus click")
            elif pairs:
                pairs.pop()
                print(f"  undone; {len(pairs)} pairs left")
                _redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return pairs


def _render_cfg_for_landmarks(data: dict, stimuli_cfg: dict) -> RenderConfig:
    canvas_size = data.get("schira_canvas_size")
    if canvas_size is None:
        canvas_size = data.get("canvas_size")
    return _render_config(
        stimuli_cfg,
        canvas_size=int(canvas_size) if canvas_size is not None else None,
    )


def _landmark_arrays_from_yaml(
    data: dict,
    *,
    params,
    render_cfg: RenderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    pairs = data.get("pairs") or []
    if not pairs:
        raise RuntimeError("landmarks YAML has no pairs")
    cols = np.array([p["vsd_col"] for p in pairs], dtype=np.float64)
    rows = np.array([p["vsd_row"] for p in pairs], dtype=np.float64)
    w = landmarks_pairs_to_w(pairs, params=params, render_cfg=render_cfg)
    if not np.all(np.isfinite(w)):
        raise RuntimeError("Some landmarks map to non-finite Schira w")
    return w, rows, cols, pairs


def fit_and_save(
    *,
    landmarks_path: Path,
    repo: Path,
    schira_config: Path,
    set_name: str | None,
    out_dir: Path,
) -> Path:
    data = _load_yaml(landmarks_path)
    pairs = data.get("pairs") or []
    if len(pairs) < 2:
        raise RuntimeError(f"Need ≥2 pairs in {landmarks_path}")

    set_id, params, _old_affine, session_raw = load_schira_set(
        schira_config, set_name=set_name or data.get("set_name")
    )
    stimuli_cfg = _load_yaml(DEFAULT_STIMULI)
    render_cfg = _render_cfg_for_landmarks(data, stimuli_cfg)

    w, rows, cols, pairs = _landmark_arrays_from_yaml(
        data, params=params, render_cfg=render_cfg
    )

    affine, fit_meta = fit_affine_from_w_pixels(w, rows, cols, params)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"affine_fit__{set_id}__{data.get('stimulus_id', 'stim')}.yaml"
    payload = {
        "set_name": set_id,
        "date_prefix": session_raw.get("date_prefix"),
        "stimulus_id": data.get("stimulus_id"),
        "landmark_mode": landmark_mode_from_pairs(pairs),
        "landmarks_file": str(landmarks_path),
        "schira_held_fixed": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "fa_combine": params.fa_combine,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp": params.sech_amp,
        },
        "affine": affine_to_mapping(affine),
        "fit": fit_meta,
        "note": (
            "Registration only. Copy affine into configs/schira/sets.yaml "
            "when satisfied; overlays are a separate later step."
        ),
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {out_path}")
    print(
        f"RMSD={fit_meta['rmsd_px']:.3f} px  "
        f"origin=({affine.origin_x:.2f},{affine.origin_y:.2f})  "
        f"ppu={affine.pixels_per_unit:.2f}  rot={affine.rotation_deg:.1f}°  "
        f"flips=({affine.flip_u},{affine.flip_v})"
    )
    return out_path


def fit_merge_and_save(
    *,
    landmarks_paths: list[Path],
    repo: Path,
    schira_config: Path,
    set_name: str | None,
    out_dir: Path,
) -> Path:
    """Fit one session camera affine from pooled landmark YAMLs."""
    if not landmarks_paths:
        raise RuntimeError("Need at least one landmarks file for --fit-merge")

    resolved = _resolve_landmark_paths(landmarks_paths, repo)
    print(f"Loading {len(resolved)} landmark file(s):", flush=True)
    for path in resolved:
        data = _load_yaml(path)
        n = len(data.get("pairs") or [])
        stim = data.get("stimulus_id", path.stem)
        print(f"  {path.name}  ({stim}, {n} pairs)", flush=True)

    stimuli_cfg = _load_yaml(DEFAULT_STIMULI)
    first = _load_yaml(resolved[0])
    set_id, params, _old_affine, session_raw = load_schira_set(
        schira_config, set_name=set_name or first.get("set_name")
    )
    _warn_missing_pool_landmarks(set_id, resolved, out_dir)

    w_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    per_file: list[dict] = []

    for path in resolved:
        data = _load_yaml(path)
        file_set = str(data.get("set_name") or "")
        if file_set and file_set != set_id:
            raise RuntimeError(
                f"Set mismatch: {path.name} has set_name={file_set!r}, expected {set_id!r}"
            )
        render_cfg = _render_cfg_for_landmarks(data, stimuli_cfg)
        w, rows, cols, pairs = _landmark_arrays_from_yaml(
            data, params=params, render_cfg=render_cfg
        )
        stim_id = str(data.get("stimulus_id") or path.stem)
        per_file.append(
            {
                "stimulus_id": stim_id,
                "landmarks_file": str(path),
                "landmark_mode": landmark_mode_from_pairs(pairs),
                "n_landmarks": int(w.size),
            }
        )
        w_parts.append(w)
        row_parts.append(rows)
        col_parts.append(cols)

    w_all = np.concatenate(w_parts)
    rows_all = np.concatenate(row_parts)
    cols_all = np.concatenate(col_parts)
    if w_all.size < 2:
        raise RuntimeError("Need ≥2 pooled landmark pairs across all files")

    affine, fit_meta = fit_affine_from_w_pixels(w_all, rows_all, cols_all, params)
    pred_r, pred_c = affine.w_to_pixel(w_all, params)
    err = np.hypot(pred_r - rows_all, pred_c - cols_all)

    offset = 0
    per_stimulus: dict[str, dict] = {}
    for block in per_file:
        n = int(block["n_landmarks"])
        sl = slice(offset, offset + n)
        block_err = err[sl]
        stim_id = block["stimulus_id"]
        per_stimulus[stim_id] = {
            "landmarks_file": block["landmarks_file"],
            "landmark_mode": block["landmark_mode"],
            "n_landmarks": n,
            "rmsd_px": float(np.sqrt(np.mean(block_err**2))),
            "residuals_px": block_err.tolist(),
        }
        offset += n

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"affine_fit__{set_id}__pooled.yaml"
    payload = {
        "set_name": set_id,
        "date_prefix": session_raw.get("date_prefix"),
        "stimulus_id": "pooled",
        "landmark_mode": "merged",
        "landmarks_files": [str(p) for p in resolved],
        "schira_held_fixed": {
            "a": params.a,
            "alpha": params.alpha,
            "k": params.k,
            "shear": params.shear,
            "fa_combine": params.fa_combine,
            "sech_ecc_k": params.sech_ecc_k,
            "sech_amp": params.sech_amp,
        },
        "affine": affine_to_mapping(affine),
        "fit": {
            **fit_meta,
            "n_landmarks_pooled": int(w_all.size),
            "per_stimulus": per_stimulus,
        },
        "note": (
            "Pooled session registration. Copy affine into configs/schira/sets.yaml "
            "when satisfied; overlays are a separate later step."
        ),
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {out_path}")
    print(
        f"Pooled RMSD={fit_meta['rmsd_px']:.3f} px  "
        f"({w_all.size} landmarks from {len(resolved)} files)"
    )
    print(
        f"origin=({affine.origin_x:.2f},{affine.origin_y:.2f})  "
        f"ppu={affine.pixels_per_unit:.2f}  rot={affine.rotation_deg:.1f}°  "
        f"flips=({affine.flip_u},{affine.flip_v})"
    )
    for stim_id, meta in per_stimulus.items():
        print(f"  {stim_id}: RMSD={meta['rmsd_px']:.3f} px  n={meta['n_landmarks']}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_name", default="201118")
    p.add_argument("--schira-config", type=Path, default=DEFAULT_SCHIRA)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--window", type=Path, default=DEFAULT_WINDOW)
    p.add_argument("--stimuli-yaml", type=Path, default=DEFAULT_STIMULI)
    p.add_argument("--stimulus", default="letter_F_white_1")
    p.add_argument(
        "--stimuli",
        nargs="+",
        default=None,
        metavar="STIMULUS",
        help="One or more stimulus_ids for --pick / --pick-schira (overrides --stimulus)",
    )
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--pick",
        action="store_true",
        help="Open GUI: stimulus image ↔ VSD landmark pairs (legacy)",
    )
    p.add_argument(
        "--pick-schira",
        action="store_true",
        help="Open GUI: Schira forward ink (u,v) ↔ VSD landmark pairs",
    )
    p.add_argument(
        "--schira-canvas-size",
        type=int,
        default=630,
        help="Stimulus render size for Schira ink cloud (--pick-schira; 210×3=630)",
    )
    p.add_argument(
        "--fit",
        type=Path,
        default=None,
        help="Fit affine from an existing landmarks YAML",
    )
    p.add_argument(
        "--fit-merge",
        nargs="+",
        type=Path,
        metavar="LANDMARKS",
        help="Fit one session affine from multiple landmark YAMLs (writes *__pooled.yaml)",
    )
    args = p.parse_args()
    repo = project_root()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()

    if args.pick and args.pick_schira:
        raise SystemExit("Use only one of --pick or --pick-schira")
    if args.fit is not None and args.fit_merge:
        raise SystemExit("Use only one of --fit or --fit-merge")
    if args.fit_merge and (args.pick or args.pick_schira):
        raise SystemExit("--fit-merge cannot be combined with --pick / --pick-schira")

    if args.fit_merge:
        fit_merge_and_save(
            landmarks_paths=args.fit_merge,
            repo=repo,
            schira_config=args.schira_config,
            set_name=args.set_name,
            out_dir=out_dir,
        )
        return

    if args.pick_schira:
        stimulus_ids = _stimulus_ids_from_args(args)
        saved_paths: list[Path] = []
        for stim_id in stimulus_ids:
            landmarks_path = _run_pick_schira_for_stimulus(
                repo=repo,
                set_name=args.set_name,
                schira_config=args.schira_config,
                config_path=args.config,
                window_path=args.window,
                stimuli_yaml=args.stimuli_yaml,
                stimulus_id=stim_id,
                n_trials=args.n_trials,
                seed=args.seed,
                schira_canvas_size=args.schira_canvas_size,
                out_dir=out_dir,
            )
            saved_paths.append(landmarks_path)
        if len(saved_paths) == 1:
            fit_and_save(
                landmarks_path=saved_paths[0],
                repo=repo,
                schira_config=args.schira_config,
                set_name=args.set_name,
                out_dir=out_dir,
            )
        else:
            print(
                f"\nPooled fit from {len(saved_paths)} landmark files...",
                flush=True,
            )
            fit_merge_and_save(
                landmarks_paths=saved_paths,
                repo=repo,
                schira_config=args.schira_config,
                set_name=args.set_name,
                out_dir=out_dir,
            )
        return

    if args.pick:
        (
            set_id,
            params,
            _aff,
            meta,
            stim_rgb,
            mean_vsd,
            render_cfg,
            _session,
        ) = _load_stimulus_and_mean_vsd(
            repo=repo,
            set_name=args.set_name,
            schira_config=args.schira_config,
            config_path=args.config,
            window_path=args.window,
            stimuli_yaml=args.stimuli_yaml,
            stimulus_id=args.stimulus,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        print(
            f"Picking landmarks for {meta['h5_session']} {meta['condition']} "
            f"{args.stimulus} (Schira held fixed)."
        )
        print(
            "Fig.10 guide (F): 1=upper stem∩top, 2=stem∩middle, "
            "3=bottom of stem, 4=end of top bar, 5=end of middle bar."
        )
        pairs = pick_landmarks_gui(
            stim_rgb,
            mean_vsd,
            title=f"{meta['h5_session']} {args.stimulus}",
        )
        if len(pairs) < 2:
            raise SystemExit("Need at least 2 pairs; aborting.")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Cache mean VSD for later combine step
        np.save(out_dir / f"mean_vsd__{meta['h5_session']}__{meta['condition']}.npy", mean_vsd)
        landmarks_path = out_dir / f"landmarks__{set_id}__{args.stimulus}.yaml"
        payload = {
            **meta,
            "landmark_mode": "stimulus_px",
            "canvas_size": render_cfg.canvas_size,
            "pixels_per_deg": render_cfg.pixels_per_deg,
            "pairs": pairs,
            "guide": (
                "Pairs are (stimulus image xy_px) ↔ (VSD col,row). "
                "Fig.10 F corners/endpoints. Schira not applied until --fit."
            ),
        }
        landmarks_path.write_text(yaml.safe_dump(payload, sort_keys=False))
        print(f"Wrote {landmarks_path} ({len(pairs)} pairs)")
        # Immediately fit (still registration-only output)
        fit_and_save(
            landmarks_path=landmarks_path,
            repo=repo,
            schira_config=args.schira_config,
            set_name=args.set_name,
            out_dir=out_dir,
        )
        return

    if args.fit is not None:
        fit_and_save(
            landmarks_path=args.fit,
            repo=repo,
            schira_config=args.schira_config,
            set_name=args.set_name,
            out_dir=out_dir,
        )
        return

    raise SystemExit("Specify --pick-schira, --pick, --fit PATH, or --fit-merge PATH ...")


if __name__ == "__main__":
    main()
