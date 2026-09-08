"""Session-camera vasculature registration (not Schira landmark fitting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.session_register.config import (
    MixedMonkeyError,
    RegistrationBoundsError,
    SessionRegisterConfig,
    check_same_monkey,
    load_config,
)
from src.session_register.fit import fit_transform, ncc, phase_cross_correlation
from src.session_register.transform import SessionTransform, default_center_xy
from src.session_register.vessels import emphasize_vessels


def _vessel_like(size: int = 100, seed: int = 0) -> np.ndarray:
    """Bright ridges on a dark field (already vessel-like)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    img[22:25, 12:88] = 1.0
    img[55:59, 18:82] = 1.0
    img[70:73, 20 : min(75, size - 1)] = 0.8
    for i in range(18, min(82, size)):
        c0 = min(size - 1, max(0, 28 + i // 5))
        c1 = min(size - 1, max(0, 29 + i // 5))
        c2 = min(size - 1, max(0, 72 - i // 6))
        img[i, c0] = 1.0
        img[i, c1] = 1.0
        img[i, c2] = 0.9
    img += 0.04 * rng.normal(size=img.shape)
    return gaussian_filter(img, 0.7)


def test_frame_window_end_exclusive():
    cfg = SessionRegisterConfig(start_frame=35, end_frame=40)
    assert cfg.n_window_frames == 5
    assert cfg.frame_window == [35, 40]
    with pytest.raises(ValueError, match="half-open"):
        SessionRegisterConfig(start_frame=35, end_frame=35)


def test_monkey_mismatch_raises():
    with pytest.raises(MixedMonkeyError, match="within-animal"):
        check_same_monkey("gandalf", "jimi")
    assert check_same_monkey("Gandalf", "gandalf") == "gandalf"


def test_transform_yaml_roundtrip(tmp_path: Path):
    t = SessionTransform(
        rotation_deg=8.25,
        translation_xy=(5.0, -3.0),
        transform_type="rigid",
        scale=1.0,
        center_xy=(49.5, 49.5),
        spatial_size=(100, 100),
    )
    path = tmp_path / "transform.yaml"
    t.save_yaml(
        path,
        extra={
            "fixed_session": "100718a",
            "moving_session": "240718a",
            "monkey": "gandalf",
        },
    )
    loaded, raw = SessionTransform.load_yaml(path)
    assert raw["fixed_session"] == "100718a"
    assert raw["moving_session"] == "240718a"
    assert raw["monkey"] == "gandalf"
    assert "inverse" in raw
    np.testing.assert_allclose(loaded.rotation_deg, t.rotation_deg)
    np.testing.assert_allclose(loaded.translation_xy, t.translation_xy)
    np.testing.assert_allclose(loaded.scale, t.scale)
    np.testing.assert_allclose(loaded.center_xy, t.center_xy)


def test_inverse_roundtrip_points():
    t = SessionTransform(
        rotation_deg=8.0,
        translation_xy=(5.0, -3.0),
        transform_type="rigid",
        center_xy=(49.5, 49.5),
        spatial_size=(100, 100),
    )
    rng = np.random.default_rng(1)
    rows = rng.uniform(10, 90, size=20)
    cols = rng.uniform(10, 90, size=20)
    r2, c2 = t.apply_to_points(rows, cols, inverse=False)
    r3, c3 = t.apply_to_points(r2, c2, inverse=True)
    np.testing.assert_allclose(r3, rows, atol=1e-6)
    np.testing.assert_allclose(c3, cols, atol=1e-6)
    r4, c4 = t.invert().invert().apply_to_points(rows, cols)
    np.testing.assert_allclose(r4, r2, atol=1e-6)
    np.testing.assert_allclose(c4, c2, atol=1e-6)


def test_apply_to_map_matches_points():
    t = SessionTransform(
        rotation_deg=8.0,
        translation_xy=(5.0, -3.0),
        transform_type="rigid",
        center_xy=default_center_xy((100, 100)),
        spatial_size=(100, 100),
    )
    img = np.zeros((100, 100), dtype=np.float64)
    img[40, 60] = 1.0
    img = gaussian_filter(img, 0.6)
    warped = t.apply_to_map(img, inverse=False, cval=0.0)
    r_fix, c_fix = t.apply_to_points(np.array([40.0]), np.array([60.0]))
    # Forward map sends moving (40, 60) to fixed; warped[fixed] samples moving
    # at T^{-1}(fixed), so the blob appears at T(40, 60).
    peak = np.unravel_index(int(np.argmax(warped)), warped.shape)
    assert abs(peak[0] - r_fix[0]) < 1.5
    assert abs(peak[1] - c_fix[0]) < 1.5


def test_synthetic_rigid_recover():
    fixed = _vessel_like(100, seed=2)
    true = SessionTransform(
        rotation_deg=8.0,
        translation_xy=(5.0, -3.0),
        transform_type="rigid",
        center_xy=default_center_xy(fixed.shape),
        spatial_size=fixed.shape,
    )
    moving = true.apply_to_map(fixed, inverse=True, cval=0.0)
    fitted, meta = fit_transform(
        fixed,
        moving,
        transform_type="rigid",
        metric="ncc",
        max_rotation_deg=25.0,
        max_translation_px=30.0,
        mask_border_px=4,
    )
    assert abs(fitted.rotation_deg - 8.0) < 0.5
    assert abs(fitted.dx_col - 5.0) < 0.5
    assert abs(fitted.dy_row - (-3.0)) < 0.5
    assert meta["metric"] == "ncc"
    assert meta["metric_value"] > 0.7


def test_hits_translation_wall():
    fixed = _vessel_like(100, seed=3)
    true = SessionTransform(
        rotation_deg=0.0,
        translation_xy=(18.0, 4.0),
        transform_type="translation",
        center_xy=default_center_xy(fixed.shape),
        spatial_size=fixed.shape,
    )
    moving = true.apply_to_map(fixed, inverse=True, cval=0.0)
    with pytest.raises(RegistrationBoundsError, match="hit search bounds"):
        fit_transform(
            fixed,
            moving,
            transform_type="translation",
            metric="ncc",
            max_rotation_deg=25.0,
            max_translation_px=8.0,
            mask_border_px=4,
        )


def test_emphasize_vessels_dark_on_white():
    rng = np.random.default_rng(4)
    raw = np.full((100, 100), 0.75, dtype=np.float64)
    raw += 0.01 * rng.normal(size=raw.shape)
    yy, xx = np.ogrid[:100, :100]
    radius = np.hypot(yy - 49.5, xx - 49.5)
    raw[(radius > 43) & (radius < 48)] = 0.05  # dark chamber circle (must ignore)
    raw[53:56, 18:82] = 0.12  # trunk a little below mid
    raw[74:77, 18:82] = 0.12  # trunk near bottom interior
    out = emphasize_vessels(
        raw,
        invert_intensity=False,
        vessel_method="meijering",
    )
    assert out.shape == raw.shape
    assert out.dtype == np.float32
    assert float(np.median(out)) > 0.85  # mostly white
    assert float(np.min(out[52:57, 30:70])) < 0.5
    assert float(np.min(out[73:78, 30:70])) < 0.5
    rim = (radius > 43) & (radius < 48)
    assert float(np.mean(out[rim])) > 0.85
    vmask = out < 0.5
    cols = np.where(vmask.any(axis=0))[0]
    assert cols.size > 20
    assert float(vmask.sum(axis=0)[cols].mean()) <= 3.0


def test_load_anchor_yaml_defaults():
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / "experiments/session_register/configs/gandalf_100718a_anchor.yaml"
    cfg = load_config(cfg_path, moving_session="240718a", repo=repo)
    assert cfg.monkey == "gandalf"
    assert cfg.fixed_session == "100718a"
    assert cfg.moving_session == "240718a"
    assert cfg.start_frame == 35 and cfg.end_frame == 40
    assert cfg.n_window_frames == 5
    assert cfg.normalization == "none"
    assert cfg.transform_type_name == "rigid"
    assert cfg.invert_intensity is False
    assert cfg.vessel_method == "meijering"
    assert cfg.n_vessel_objects == 2
    assert cfg.chamber_rim_px == 12
    assert cfg.roi_radius_px == 40
    assert cfg.vessel_trace_px == 1
    assert cfg.n_landmarks == 6
    assert cfg.vessel_start_frame == 5
    assert cfg.vessel_end_frame == 25
    assert cfg.save_qc is False
    assert cfg.verbose == 0


def test_phase_corr_pure_translation():
    img = _vessel_like(64, seed=5)
    t = SessionTransform(
        rotation_deg=0.0,
        translation_xy=(4.0, -2.0),
        transform_type="translation",
        center_xy=default_center_xy(img.shape),
        spatial_size=img.shape,
    )
    moving = t.apply_to_map(img, inverse=True, cval=0.0)
    dy, dx = phase_cross_correlation(img, moving)
    assert abs(dx - 4.0) < 0.6
    assert abs(dy - (-2.0)) < 0.6
    fitted, _ = fit_transform(
        img,
        moving,
        transform_type="translation",
        metric="ncc",
        max_rotation_deg=25.0,
        max_translation_px=30.0,
        mask_border_px=4,
    )
    assert abs(fitted.rotation_deg) < 1e-9
    assert abs(fitted.dx_col - 4.0) < 0.5
    assert abs(fitted.dy_row - (-2.0)) < 0.5
    assert ncc(img, t.apply_to_map(moving), np.ones(img.shape, dtype=bool)) > 0.85


def test_n_landmarks_config_default():
    cfg = SessionRegisterConfig()
    assert cfg.n_landmarks == 6
    cfg2 = SessionRegisterConfig.from_mapping({"n_landmarks": 10})
    assert cfg2.n_landmarks == 10


def test_rigid_from_landmark_pairs_recovers_pose():
    from src.session_register.fit import landmark_rmsd_px, rigid_from_point_pairs

    true = SessionTransform(
        rotation_deg=8.0,
        translation_xy=(5.0, -3.0),
        transform_type="rigid",
        center_xy=default_center_xy((100, 100)),
        spatial_size=(100, 100),
    )
    moving = np.array(
        [[50.0, 30.0], [55.0, 45.0], [62.0, 58.0], [48.0, 70.0], [66.0, 40.0], [58.0, 52.0]]
    )
    fr, fc = true.apply_to_points(moving[:, 0], moving[:, 1])
    fixed = np.stack([fr, fc], axis=1)
    fitted = rigid_from_point_pairs(
        moving, fixed, center_xy=true.center_xy, spatial_size=true.spatial_size
    )
    assert abs(fitted.rotation_deg - 8.0) < 0.05
    assert abs(fitted.dx_col - 5.0) < 0.05
    assert abs(fitted.dy_row - (-3.0)) < 0.05
    assert landmark_rmsd_px(moving, fixed, fitted) < 0.05


def test_homologous_stations_share_columns():
    from src.session_register.vessels import _homologous_stations

    upper = {c: 42 for c in range(20, 81)}
    lower = {c: 58 for c in range(20, 81)}
    # Pass lower first — ordering must still label 0=upper.
    pts, ids = _homologous_stations(
        [lower, upper], n_landmarks=6, row_lo=0, row_hi=100
    )
    assert len(pts) >= 6
    labels = {(int(a), int(b)) for a, b in ids}
    assert (0, 0) in labels and (1, 0) in labels
    by_id = {(int(a), int(b)): (int(r), int(c)) for (r, c), (a, b) in zip(pts, ids)}
    for k in range(3):
        assert by_id[(0, k)][1] == by_id[(1, k)][1]
        assert by_id[(0, k)][0] < by_id[(1, k)][0]


def test_homologous_stations_skip_interpolated_gap():
    from src.session_register.vessels import _homologous_stations

    upper = {c: 42 for c in range(20, 81)}
    lower = {c: 58 for c in range(20, 81)}
    detected_u = {c: 42 for c in range(32, 72)}
    detected_l = {c: 58 for c in range(32, 72)}
    pts, ids = _homologous_stations(
        [upper, lower],
        n_landmarks=6,
        row_lo=0,
        row_hi=100,
        detected_upper=detected_u,
        detected_lower=detected_l,
    )
    cols = pts[:, 1]
    assert cols.min() > 20
    assert cols.max() < 80


def test_tracks_from_column_troughs_two_rivers():
    from src.session_register.vessels import _tracks_from_column_troughs

    crest = np.zeros((80, 80), dtype=bool)
    for c in range(20, 70):
        crest[40, c] = True
        crest[55, c] = True
        if c % 5 == 0:
            crest[40, c] = False  # gap on upper
    tracks, _, _ = _tracks_from_column_troughs(crest, min_sep=6)
    assert len(tracks) == 2
    overlap = set(tracks[0]) & set(tracks[1])
    assert min(overlap) <= 25 and max(overlap) >= 65
    assert abs(np.mean(list(tracks[0].values())) - 40) < 2
    assert abs(np.mean(list(tracks[1].values())) - 55) < 2


def test_match_landmarks_by_id_ignores_nearest_neighbour():
    from src.session_register.fit import match_landmarks_by_id

    # Fixed: U0 near (40,20), L0 near (55,20). Moving L0 is closer to fixed U0.
    fixed = np.array([[40.0, 20.0], [40.0, 50.0], [55.0, 20.0], [55.0, 50.0]])
    fixed_ids = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    moving = np.array([[42.0, 21.0], [41.0, 51.0], [41.0, 22.0], [56.0, 51.0]])
    moving_ids = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    mf, mm, ids = match_landmarks_by_id(fixed, fixed_ids, moving, moving_ids)
    assert mf.shape[0] == 4
    paired = {(int(a), int(b)): (tuple(f), tuple(m)) for (a, b), f, m in zip(ids, mf, mm)}
    assert paired[(1, 0)][0] == (55.0, 20.0)
    assert paired[(1, 0)][1] == (41.0, 22.0)


def test_landmark_id_label():
    from src.session_register.fit import landmark_id_label

    assert landmark_id_label(0, 1) == "U1"
    assert landmark_id_label(1, 0) == "L0"
    assert landmark_id_label(-1, 0) == "J"


def test_run_registration_skips_existing_yaml(tmp_path: Path):
    from src.session_register.pipeline import load_registration, run_registration

    cfg = SessionRegisterConfig(
        monkey="gandalf",
        fixed_session="100718a",
        moving_session="240718a",
        output_dir=str(tmp_path),
        save_qc=False,
        verbose=0,
    )
    transform = SessionTransform(
        rotation_deg=1.5,
        translation_xy=(2.0, -1.0),
        transform_type="rigid",
        center_xy=default_center_xy((100, 100)),
        spatial_size=(100, 100),
    )
    yaml_path = cfg.pair_dir() / "transform.yaml"
    transform.save_yaml(yaml_path, extra={"moving_session": "240718a", "monkey": "gandalf"})
    out = run_registration(cfg)
    assert out["skipped"] is True
    assert "fixed_raw" not in out
    assert abs(out["transform"].rotation_deg - 1.5) < 1e-9
    assert list(Path(out["pair_dir"]).glob("*.png")) == []
    loaded = load_registration(cfg)
    assert loaded["yaml_path"] == str(yaml_path)


def _same_day_pair_available() -> bool:
    from src.paths import project_root
    from src.session_register.load import session_trial_rows

    repo = project_root()
    cfg = SessionRegisterConfig(monkey="gandalf", n_trials=4, start_frame=35, end_frame=40)
    try:
        a = session_trial_rows(cfg, "100718a", repo=repo)
        b = session_trial_rows(cfg, "100718b", repo=repo)
    except (ValueError, FileNotFoundError, OSError):
        return False
    return len(a) > 0 and len(b) > 0


@pytest.mark.skipif(not _same_day_pair_available(), reason="100718a/b H5 not mounted")
def test_extract_vessels_100718a_two_trunks(tmp_path: Path):
    """Optional: 100718a mean should isolate two dark trunks on a white field."""
    from src.paths import project_root
    from src.session_register.load import load_session_mean_map
    from src.session_register.vessels import extract_vessels
    from skimage.measure import label

    cfg = SessionRegisterConfig(
        monkey="gandalf",
        fixed_session="100718a",
        moving_session="100718b",
        start_frame=35,
        end_frame=40,
        n_trials=4,
        seed=17,
        vessel_method="meijering",
        n_vessel_objects=2,
        output_dir=str(tmp_path),
        overwrite=True,
    )
    mean, _meta = load_session_mean_map(cfg, "100718a", repo=project_root())
    maps = extract_vessels(mean, cfg=cfg)
    assert float(np.median(maps.display)) > 0.9
    n_obj = int(label(maps.vessel_mask, connectivity=2).max())
    assert 1 <= n_obj <= 80
    n_px = int(maps.vessel_mask.sum())
    assert 60 <= n_px <= 400
    cols = np.where(maps.vessel_mask.any(axis=0))[0]
    assert float(maps.vessel_mask.sum(axis=0)[cols].mean()) <= 3.0
    # Chamber circle (outer annulus) must not be labeled a vessel.
    h, w = mean.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    rim = (r > min(h, w) / 2.0 - 8) & (r <= min(h, w) / 2.0)
    assert not np.any(maps.vessel_mask & rim)
    # Trunks must stay two ordered bands (no X-crossing).
    upper, lower = [], []
    for col in range(w):
        rows = np.where(maps.vessel_mask[:, col])[0]
        if rows.size >= 2:
            upper.append(int(rows.min()))
            lower.append(int(rows.max()))
    if upper:
        assert float(np.median(lower)) - float(np.median(upper)) >= 8


def test_extract_vessels_cached_100718a_thin_no_rim():
    """If the mean cache exists, trunks must be thin and miss the chamber circle."""
    from src.paths import project_root
    from src.session_register.vessels import extract_vessels
    from skimage.measure import label

    path = (
        project_root()
        / "experiments/session_register/runs/cache/mean_vessel__100718a.npy"
    )
    if not path.is_file():
        pytest.skip("cached 100718a mean not present")
    mean = np.load(path)
    maps = extract_vessels(mean)
    assert float(np.median(maps.display)) > 0.9
    n_px = int(maps.vessel_mask.sum())
    assert 60 <= n_px <= 400
    assert int(label(maps.vessel_mask, connectivity=2).max()) <= 80
    h, w = mean.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    rim = r > (min(h, w) / 2.0 - 8)
    assert not np.any(maps.vessel_mask & rim)
    upper, lower = [], []
    for col in range(w):
        rows = np.where(maps.vessel_mask[:, col])[0]
        if rows.size >= 2:
            upper.append(int(rows.min()))
            lower.append(int(rows.max()))
    if upper:
        assert float(np.median(lower)) - float(np.median(upper)) >= 8
