"""Fit Schira and/or camera parameters to anchor-aligned mean VSD maps.

``optimize`` selects which group is free::

    none    — YAML start, no update
    schira  — ``a``, ``alpha``, ``k``
    camera  — fovea, rotation, ``pixels_per_unit``
    all     — both groups

The two double-sech constants and the flips stay fixed. Each leave-one-out
fold starts from the same YAML values and drops its held-out stimulus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml

from src.encoding.cm_plotting import load_stimulus_rgb
from src.paths import resolve_data_path
from src.retinotopy.affine import CorticalAffine
from src.retinotopy.register import affine_to_mapping
from src.retinotopy.schira import SchiraParams
from src.schira_encoding.io import load_session_transform, load_target_map
from src.schira_encoding.targets import warp_target_to_anchor
from src.schira_encoding.torch_warp import sample_contrast_maps

OPTIMIZE_NONE = "none"
OPTIMIZE_SCHIRA = "schira"
OPTIMIZE_CAMERA = "camera"
OPTIMIZE_ALL = "all"
OPTIMIZE_MODES = (
    OPTIMIZE_NONE,
    OPTIMIZE_SCHIRA,
    OPTIMIZE_CAMERA,
    OPTIMIZE_ALL,
)
CAMERA_PARAM_NAMES = ("origin_x", "origin_y", "rotation_deg", "pixels_per_unit")
SCHIRA_PARAM_NAMES = ("a", "alpha", "k")


def normalize_optimize(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in OPTIMIZE_MODES:
        raise ValueError(f"optimize must be one of {OPTIMIZE_MODES}, got {value!r}")
    return mode


def free_param_names(optimize: str) -> tuple[str, ...]:
    mode = normalize_optimize(optimize)
    if mode == OPTIMIZE_NONE:
        return ()
    if mode == OPTIMIZE_SCHIRA:
        return SCHIRA_PARAM_NAMES
    if mode == OPTIMIZE_CAMERA:
        return CAMERA_PARAM_NAMES
    return CAMERA_PARAM_NAMES + SCHIRA_PARAM_NAMES


def schira_opt_run_leaf(optimize: str, run_name: str | None = None) -> str:
    """Directory leaf: ``schira_opt_{mode}`` (or ``{run_name}_{mode}``)."""
    mode = normalize_optimize(optimize)
    base = str(run_name or "schira_opt").strip() or "schira_opt"
    suffix = f"_{mode}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


@dataclass(frozen=True)
class GeometryFitConfig:
    """Box and optimizer settings. Distances are relative to the YAML start."""

    optimize: str = OPTIMIZE_CAMERA
    max_origin_px: float = 20.0
    max_rotation_deg: float = 20.0
    max_scale_frac: float = 0.20
    max_a_frac: float = 0.30
    max_alpha_frac: float = 0.30
    max_k_frac: float = 0.30
    blur_sigma_px: float = 2.0
    n_steps: int = 80
    lr: float = 0.05
    penalty_weight: float = 0.01
    patience: int = 15
    min_delta: float = 1.0e-4
    loss: str = "pearson"
    centroid_weight: float = 1.0
    moment_weight: float = 1.0
    exclude_points: bool = True
    exclude_letters: bool = True
    exclude_bars: bool = False
    anchor_session_only: bool = False
    per_session_examples: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimize", normalize_optimize(self.optimize))
        loss = str(self.loss).strip().lower()
        if loss not in {"pearson", "moments"}:
            raise ValueError(f"loss must be 'pearson' or 'moments', got {self.loss!r}")
        object.__setattr__(self, "loss", loss)

    @classmethod
    def from_mapping(cls, data: dict | None) -> GeometryFitConfig:
        raw = data or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = sorted(set(raw) - known)
        if extra:
            raise ValueError(f"Unknown geometry-fit keys: {extra}")
        return cls(**{k: raw[k] for k in known if k in raw})


def keep_geometry_fit_stimulus(stimulus_id: str, cfg: GeometryFitConfig) -> bool:
    """Drop points, letters, and optionally bars from the geometry-fit set."""
    sid = str(stimulus_id)
    if cfg.exclude_points and "_point_" in sid:
        return False
    if cfg.exclude_letters and sid.startswith("letter_"):
        return False
    if cfg.exclude_bars and "_bar_" in sid:
        return False
    return True


@dataclass(frozen=True)
class StimulusMean:
    """One stimulus: contrast image and its mean anchor-aligned VSD map."""

    stimulus_id: str
    contrast: np.ndarray
    vsd_mean: np.ndarray
    n_trials: int


@dataclass(frozen=True)
class GeometryFitResult:
    params: SchiraParams
    affine: CorticalAffine
    optimize: str
    correlation_before: float
    correlation_after: float
    contrast_sign: float
    n_stimuli: int
    n_steps: int
    per_stimulus_r_before: dict[str, float]
    per_stimulus_r_after: dict[str, float]

    def delta_mapping(
        self, start_affine: CorticalAffine, start_params: SchiraParams
    ) -> dict[str, float]:
        return {
            "origin_x": float(self.affine.origin_x - start_affine.origin_x),
            "origin_y": float(self.affine.origin_y - start_affine.origin_y),
            "rotation_deg": float(
                self.affine.rotation_deg - start_affine.rotation_deg
            ),
            "scale_ratio": float(
                self.affine.pixels_per_unit / start_affine.pixels_per_unit
            ),
            "a_ratio": float(self.params.a / start_params.a),
            "alpha_ratio": float(self.params.alpha / start_params.alpha),
            "k_ratio": float(self.params.k / start_params.k),
        }

    def to_yaml_block(
        self,
        *,
        start_affine: CorticalAffine,
        start_params: SchiraParams,
        heldout_stimulus_id: str,
    ) -> dict:
        return {
            "schira": schira_params_to_mapping(self.params),
            "affine": affine_to_mapping(self.affine),
            "fit": {
                "optimize": self.optimize,
                "heldout_stimulus_id": heldout_stimulus_id,
                "n_stimuli": int(self.n_stimuli),
                "n_steps": int(self.n_steps),
                "contrast_sign": float(self.contrast_sign),
                "correlation_before": float(self.correlation_before),
                "correlation_after": float(self.correlation_after),
                "delta_from_start": self.delta_mapping(start_affine, start_params),
                "per_stimulus_r_before": self.per_stimulus_r_before,
                "per_stimulus_r_after": self.per_stimulus_r_after,
            },
        }


def schira_params_to_mapping(params: SchiraParams) -> dict:
    return {
        "a": float(params.a),
        "alpha": float(params.alpha),
        "k": float(params.k),
        "shear": str(params.shear),
        "fa_combine": str(params.fa_combine),
        "sech_ecc_k": float(params.sech_ecc_k),
        "sech_amp": float(params.sech_amp),
    }


def frozen_geometry_result(
    params: SchiraParams, affine: CorticalAffine
) -> GeometryFitResult:
    """YAML start with no parameter update (``optimize=none``)."""
    return GeometryFitResult(
        params=params,
        affine=affine,
        optimize=OPTIMIZE_NONE,
        correlation_before=float("nan"),
        correlation_after=float("nan"),
        contrast_sign=1.0,
        n_stimuli=0,
        n_steps=0,
        per_stimulus_r_before={},
        per_stimulus_r_after={},
    )


def fit_config_to_yaml(cfg: GeometryFitConfig) -> str:
    data = {
        "optimize": cfg.optimize,
        "max_origin_px": cfg.max_origin_px,
        "max_rotation_deg": cfg.max_rotation_deg,
        "max_scale_frac": cfg.max_scale_frac,
        "max_a_frac": cfg.max_a_frac,
        "max_alpha_frac": cfg.max_alpha_frac,
        "max_k_frac": cfg.max_k_frac,
        "blur_sigma_px": cfg.blur_sigma_px,
        "n_steps": cfg.n_steps,
        "lr": cfg.lr,
        "penalty_weight": cfg.penalty_weight,
        "patience": cfg.patience,
        "min_delta": cfg.min_delta,
        "loss": cfg.loss,
        "centroid_weight": cfg.centroid_weight,
        "moment_weight": cfg.moment_weight,
        "exclude_points": cfg.exclude_points,
        "exclude_letters": cfg.exclude_letters,
        "exclude_bars": cfg.exclude_bars,
        "anchor_session_only": cfg.anchor_session_only,
        "per_session_examples": cfg.per_session_examples,
    }
    return yaml.safe_dump(data, sort_keys=False)


def build_anchor_stimulus_means(
    pairs: pd.DataFrame,
    *,
    repo,
    spatial_size: tuple[int, int],
    anchor_session: str,
    monkey: str,
    register_root,
    background_gray: float,
    canvas_size: int,
    sessions: set[str] | None = None,
    per_session: bool = False,
) -> dict[str, StimulusMean]:
    """Mean VSD per ``stimulus_id``, after session→anchor registration.

    The rendered image is the contrast (pixel minus background). Trials of the
    same stimulus on different sessions are averaged in the anchor frame unless
    ``per_session`` is set, in which case each imaging day is its own example.
    Pass ``sessions`` to keep only those imaging days (e.g. the anchor).
    """
    if "stimulus_id" not in pairs.columns:
        raise ValueError("pairs need stimulus_id")
    if "image_path" not in pairs.columns:
        raise ValueError("pairs need image_path")
    height, width = spatial_size
    grouped: dict[str, list] = {}
    for row in pairs.itertuples(index=False):
        if sessions is not None and str(row.date) not in sessions:
            continue
        sid = str(row.stimulus_id)
        key = f"{row.date}::{sid}" if per_session else sid
        grouped.setdefault(key, []).append(row)

    transforms: dict[str, object] = {}
    out: dict[str, StimulusMean] = {}
    for key, rows in grouped.items():
        sid = str(rows[0].stimulus_id)
        maps: list[np.ndarray] = []
        image = None
        for row in rows:
            session = str(row.date)
            if session not in transforms:
                transforms[session] = load_session_transform(
                    session=session,
                    anchor_session=anchor_session,
                    monkey=monkey,
                    register_root=register_root,
                )
            y = load_target_map(resolve_data_path(row.nc_path, repo), spatial_size)
            y = warp_target_to_anchor(y, transforms[session])
            if y.shape != (height, width):
                raise ValueError(f"{sid} map shape {y.shape} != {(height, width)}")
            maps.append(y)
            if image is None:
                rgb = load_stimulus_rgb(resolve_data_path(row.image_path, repo))
                if rgb.shape[0] != canvas_size or rgb.shape[1] != canvas_size:
                    raise ValueError(
                        f"{sid} image shape {rgb.shape[:2]} != canvas {canvas_size}"
                    )
                gray = rgb.astype(np.float32).mean(axis=2)
                image = gray - np.float32(background_gray)
        stack = np.stack(maps, axis=0)
        mean = np.nanmean(stack, axis=0).astype(np.float32)
        out[key] = StimulusMean(
            stimulus_id=sid,
            contrast=np.asarray(image, dtype=np.float32),
            vsd_mean=mean,
            n_trials=len(rows),
        )
    return out


def _pixel_grid(
    spatial_size: tuple[int, int], *, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = spatial_size
    rows = torch.arange(height, dtype=dtype).view(height, 1).expand(height, width)
    cols = torch.arange(width, dtype=dtype).view(1, width).expand(height, width)
    return rows, cols


def _scalar(value: float, raw: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=raw.dtype, device=raw.device)


def _shift_box(
    raw: torch.Tensor,
    index: dict[str, int],
    name: str,
    start: float,
    width: float,
) -> torch.Tensor:
    if name not in index:
        return _scalar(start, raw)
    return float(start) + float(width) * torch.tanh(raw[index[name]])


def _log_box(
    raw: torch.Tensor,
    index: dict[str, int],
    name: str,
    start: float,
    max_frac: float,
) -> torch.Tensor:
    if name not in index:
        return _scalar(start, raw)
    log_span = float(np.log1p(max_frac))
    return torch.exp(float(np.log(start)) + log_span * torch.tanh(raw[index[name]]))


def _free_from_raw(
    raw: torch.Tensor,
    start_affine: CorticalAffine,
    start_params: SchiraParams,
    cfg: GeometryFitConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Unpack the tanh box for the free names of ``cfg.optimize``."""
    names = free_param_names(cfg.optimize)
    if int(raw.numel()) != len(names):
        raise ValueError(f"raw has {int(raw.numel())} entries, expected {len(names)}")
    index = {name: i for i, name in enumerate(names)}
    origin_x = _shift_box(
        raw, index, "origin_x", start_affine.origin_x, cfg.max_origin_px
    )
    origin_y = _shift_box(
        raw, index, "origin_y", start_affine.origin_y, cfg.max_origin_px
    )
    rotation = _shift_box(
        raw, index, "rotation_deg", start_affine.rotation_deg, cfg.max_rotation_deg
    )
    ppu = _log_box(
        raw, index, "pixels_per_unit", start_affine.pixels_per_unit, cfg.max_scale_frac
    )
    a_t = _log_box(raw, index, "a", start_params.a, cfg.max_a_frac)
    alpha_t = _log_box(raw, index, "alpha", start_params.alpha, cfg.max_alpha_frac)
    k_t = _log_box(raw, index, "k", start_params.k, cfg.max_k_frac)
    return origin_x, origin_y, rotation, ppu, a_t, alpha_t, k_t


def _masked_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pearson r per stimulus. ``pred``/``target`` are ``(N, H, W)``."""
    a = pred[:, mask]
    b = target[:, mask]
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = a.norm(dim=1) * b.norm(dim=1)
    r = (a * b).sum(dim=1) / denom.clamp_min(1e-8)
    flat = (a.norm(dim=1) < 1e-8) | (b.norm(dim=1) < 1e-8) | ~torch.isfinite(r)
    return torch.where(flat, torch.zeros_like(r), r)


def _predict(
    contrast: torch.Tensor,
    target: torch.Tensor,
    raw: torch.Tensor,
    *,
    start: CorticalAffine,
    params: SchiraParams,
    rows: torch.Tensor,
    cols: torch.Tensor,
    mask: torch.Tensor,
    pixels_per_deg: float,
    cfg: GeometryFitConfig,
) -> torch.Tensor:
    pred = _warped_pred(
        contrast,
        raw,
        start=start,
        params=params,
        rows=rows,
        cols=cols,
        pixels_per_deg=pixels_per_deg,
        cfg=cfg,
    )
    return _masked_pearson(pred, target, mask)


def _energy_weights(
    maps: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Positive spatial weight from squared deviation inside ``mask``."""
    pix = maps[:, mask]
    pix = pix - pix.mean(dim=1, keepdim=True)
    energy = torch.zeros_like(maps)
    energy[:, mask] = pix * pix
    return energy


def _spatial_moments(
    weights: torch.Tensor,
    mask: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centroid ``(cy, cx)`` and second moments ``(yy, xx, xy)`` per map."""
    w = weights * mask.to(dtype=weights.dtype)
    mass = w.sum(dim=(1, 2)).clamp_min(1e-8)
    cy = (w * rows).sum(dim=(1, 2)) / mass
    cx = (w * cols).sum(dim=(1, 2)) / mass
    dy = rows.unsqueeze(0) - cy.view(-1, 1, 1)
    dx = cols.unsqueeze(0) - cx.view(-1, 1, 1)
    yy = (w * dy * dy).sum(dim=(1, 2)) / mass
    xx = (w * dx * dx).sum(dim=(1, 2)) / mass
    xy = (w * dy * dx).sum(dim=(1, 2)) / mass
    return cy, cx, yy, xx, xy


def _moment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    centroid_weight: float,
    moment_weight: float,
) -> torch.Tensor:
    """Mean squared centroid error (px²) plus second-moment error."""
    w_p = _energy_weights(pred, mask)
    w_t = _energy_weights(target, mask)
    cy_p, cx_p, yy_p, xx_p, xy_p = _spatial_moments(w_p, mask, rows, cols)
    cy_t, cx_t, yy_t, xx_t, xy_t = _spatial_moments(w_t, mask, rows, cols)
    cent = (cy_p - cy_t) ** 2 + (cx_p - cx_t) ** 2
    mom = (yy_p - yy_t) ** 2 + (xx_p - xx_t) ** 2 + 2.0 * (xy_p - xy_t) ** 2
    return (
        float(centroid_weight) * cent.mean() / 25.0
        + float(moment_weight) * mom.mean() / 2500.0
    )


def _warped_pred(
    contrast: torch.Tensor,
    raw: torch.Tensor,
    *,
    start: CorticalAffine,
    params: SchiraParams,
    rows: torch.Tensor,
    cols: torch.Tensor,
    pixels_per_deg: float,
    cfg: GeometryFitConfig,
) -> torch.Tensor:
    origin_x, origin_y, rotation, ppu, a_t, alpha_t, k_t = _free_from_raw(
        raw, start, params, cfg
    )
    return sample_contrast_maps(
        contrast,
        origin_x=origin_x,
        origin_y=origin_y,
        rotation_deg=rotation,
        pixels_per_unit=ppu,
        flip_u=bool(start.flip_u),
        flip_v=bool(start.flip_v),
        rows=rows,
        cols=cols,
        params=params,
        pixels_per_deg=pixels_per_deg,
        blur_sigma_px=float(cfg.blur_sigma_px),
        a=a_t,
        alpha=alpha_t,
        k=k_t,
    )


def fit_schira_geometry(
    stimuli: dict[str, StimulusMean],
    *,
    init_params: SchiraParams,
    init_affine: CorticalAffine,
    mask: np.ndarray,
    pixels_per_deg: float,
    cfg: GeometryFitConfig | None = None,
) -> GeometryFitResult:
    """Update the free group in ``cfg.optimize`` so warped stimuli match mean VSD.

    ``stimuli`` must already exclude the held-out id. The two double-sech
    constants stay at ``init_params``.
    """
    cfg = cfg or GeometryFitConfig()
    if cfg.optimize == OPTIMIZE_NONE:
        return frozen_geometry_result(init_params, init_affine)
    stimuli = {
        sid: stim
        for sid, stim in stimuli.items()
        if float(np.std(stim.contrast)) >= 0.3 and np.isfinite(stim.vsd_mean).any()
    }
    if not stimuli:
        raise ValueError("No stimuli to fit")
    if (
        cfg.max_origin_px <= 0
        or cfg.max_rotation_deg <= 0
        or cfg.max_scale_frac <= 0
        or cfg.max_a_frac <= 0
        or cfg.max_alpha_frac <= 0
        or cfg.max_k_frac <= 0
    ):
        raise ValueError("Geometry box limits must be positive")
    if init_affine.pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive")
    if init_params.a <= 0 or init_params.alpha <= 0 or init_params.k == 0:
        raise ValueError("Schira a, alpha must be > 0 and k must be non-zero")

    ids = sorted(stimuli)
    contrast_np = np.stack([stimuli[s].contrast for s in ids], axis=0)
    target_np = np.stack([stimuli[s].vsd_mean for s in ids], axis=0)
    spatial = target_np.shape[-2:]
    if tuple(mask.shape) != spatial:
        raise ValueError(f"mask shape {mask.shape} != VSD {spatial}")
    finite = np.isfinite(target_np).all(axis=0) & np.asarray(mask, dtype=bool)
    if int(finite.sum()) < 16:
        raise ValueError(f"Need at least 16 finite mask pixels, got {int(finite.sum())}")

    dtype = torch.float64
    contrast = torch.from_numpy(np.ascontiguousarray(contrast_np)).to(dtype).unsqueeze(1)
    target = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(target_np))).to(dtype)
    mask_t = torch.from_numpy(np.ascontiguousarray(finite))
    rows, cols = _pixel_grid(spatial, dtype=dtype)
    n_free = len(free_param_names(cfg.optimize))
    raw = torch.nn.Parameter(torch.zeros(n_free, dtype=dtype))
    opt = torch.optim.Adam([raw], lr=float(cfg.lr))

    with torch.no_grad():
        r0 = _predict(
            contrast,
            target,
            raw,
            start=init_affine,
            params=init_params,
            rows=rows,
            cols=cols,
            mask=mask_t,
            pixels_per_deg=pixels_per_deg,
            cfg=cfg,
        )
    signed = float(torch.nanmean(r0).item())
    # A clearly negative mean is an imaging-polarity flip. A value near zero
    # is a misplaced blob, and flipping it would push the fit the wrong way.
    contrast_sign = -1.0 if np.isfinite(signed) and signed < -0.05 else 1.0
    if contrast_sign < 0.0:
        contrast = contrast * contrast_sign

    def _loss_and_r(raw_param: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = _warped_pred(
            contrast,
            raw_param,
            start=init_affine,
            params=init_params,
            rows=rows,
            cols=cols,
            pixels_per_deg=pixels_per_deg,
            cfg=cfg,
        )
        r = _masked_pearson(pred, target, mask_t)
        usable = torch.isfinite(r)
        if not bool(usable.any()):
            raise RuntimeError("Geometry correlation is non-finite for every stimulus")
        penalty = float(cfg.penalty_weight) * (raw_param * raw_param).sum()
        if cfg.loss == "moments":
            data_loss = _moment_loss(
                pred,
                target,
                mask_t,
                rows,
                cols,
                centroid_weight=float(cfg.centroid_weight),
                moment_weight=float(cfg.moment_weight),
            )
        else:
            data_loss = 1.0 - r[usable].mean()
        return data_loss + penalty, r

    with torch.no_grad():
        loss0, r_before = _loss_and_r(raw)
    best_raw = raw.detach().clone()
    best_loss = float(loss0.item())
    stall = 0
    steps_run = 0
    for _step in range(int(cfg.n_steps)):
        opt.zero_grad(set_to_none=True)
        loss, _r = _loss_and_r(raw)
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], max_norm=5.0)
        opt.step()
        steps_run += 1
        value = float(loss.detach().item())
        if value < best_loss - float(cfg.min_delta):
            best_loss = value
            best_raw = raw.detach().clone()
            stall = 0
        else:
            stall += 1
            if stall >= int(cfg.patience):
                break
    with torch.no_grad():
        raw.copy_(best_raw)
        r_after = _predict(
            contrast,
            target,
            raw,
            start=init_affine,
            params=init_params,
            rows=rows,
            cols=cols,
            mask=mask_t,
            pixels_per_deg=pixels_per_deg,
            cfg=cfg,
        )
        origin_x, origin_y, rotation, ppu, a_t, alpha_t, k_t = _free_from_raw(
            raw, init_affine, init_params, cfg
        )

    def _r_dict(values: torch.Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        for sid, value in zip(ids, values.detach().cpu().tolist()):
            out[sid] = float(value)
        return out

    affine = CorticalAffine(
        origin_x=float(origin_x.item()),
        origin_y=float(origin_y.item()),
        pixels_per_unit=float(ppu.item()),
        rotation_deg=float(rotation.item()),
        flip_u=bool(init_affine.flip_u),
        flip_v=bool(init_affine.flip_v),
        needs_confirmation=False,
    )
    fitted_params = SchiraParams(
        a=float(a_t.item()),
        alpha=float(alpha_t.item()),
        k=float(k_t.item()),
        shear=str(init_params.shear),
        sech_ecc_k=float(init_params.sech_ecc_k),
        sech_amp=float(init_params.sech_amp),
        fa_combine=str(init_params.fa_combine),
    )
    return GeometryFitResult(
        params=fitted_params,
        affine=affine,
        optimize=cfg.optimize,
        correlation_before=float(torch.nanmean(r_before).item()),
        correlation_after=float(torch.nanmean(r_after).item()),
        contrast_sign=float(contrast_sign),
        n_stimuli=len(ids),
        n_steps=int(steps_run),
        per_stimulus_r_before=_r_dict(r_before),
        per_stimulus_r_after=_r_dict(r_after),
    )
