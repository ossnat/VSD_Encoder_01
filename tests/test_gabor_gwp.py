from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.DL_features.backbone import build_feature_extractor
from src.DL_features.gabor_gwp import GaborWaveletPyramid, _build_gabor_bank
from src.DL_features.preprocess import preprocess_stimulus
from src.DL_features.schema import model_slug


def test_model_slug_resnet():
    cfg = {"type": "resnet", "name": "resnet18", "pretrained": True}
    assert model_slug(cfg) == "resnet18_imagenet"


def test_model_slug_gabor():
    cfg = {"type": "gabor_gwp", "name": "gabor_serre"}
    assert model_slug(cfg) == "gabor_serre_gwp"


def test_model_slug_gabor_variant():
    cfg = {"type": "gabor_gwp", "name": "gabor_serre", "variant": "wl14_pf2"}
    assert model_slug(cfg) == "gabor_serre_gwp_wl14_pf2"


def test_gabor_gamma_changes_bank():
    bank_lo = _build_gabor_bank(
        n_scales=1, n_orientations=1, kernel_size=15,
        min_wavelength=4.0, wavelength_factor=1.414, gamma=0.3,
    )
    bank_hi = _build_gabor_bank(
        n_scales=1, n_orientations=1, kernel_size=15,
        min_wavelength=4.0, wavelength_factor=1.414, gamma=0.7,
    )
    assert not np.allclose(bank_lo, bank_hi)


def test_gabor_sqrt_energy():
    model = GaborWaveletPyramid(n_scales=2, n_orientations=4, kernel_size=15, energy_mode="sqrt_contrast")
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert out.shape == (1, 8, 32, 32)
    assert (out >= 0).all()


def test_gabor_energy_shape():
    model = GaborWaveletPyramid(n_scales=5, n_orientations=8, kernel_size=31)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 40, 224, 224)


def test_gabor_pooled_extractor():
    cfg = {
        "type": "gabor_gwp",
        "name": "gabor_serre",
        "pool": "avg",
        "pool_size": 14,
        "gwp": {"number_of_scales": 3, "number_of_directions": 4},
    }
    model = build_feature_extractor(cfg)
    x = torch.randn(1, 1, 224, 224)
    out = model(x)
    assert out.shape == (1, 12, 14, 14)


def test_gabor_extractor_from_config():
    cfg = {
        "type": "gabor_gwp",
        "name": "gabor_serre",
        "gwp": {"number_of_scales": 3, "number_of_directions": 4},
    }
    model = build_feature_extractor(cfg)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 12, 64, 64)


def test_resnet_legacy_string_api():
    model = build_feature_extractor(
        "resnet18", pretrained=False, feature_layer="layer3"
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.ndim == 4
    assert out.shape[1] == 256


def test_preprocess_stimulus_grayscale():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., :] = 128
    cfg = {"preprocess": "grayscale_luminance"}
    tensor = preprocess_stimulus(rgb, model_cfg=cfg, input_size=32)
    assert tensor.shape == (1, 32, 32)
    assert float(tensor.mean()) > 0.4


def test_single_stimulus_end_to_end(tmp_path):
    """One synthetic RGB stimulus through Gabor extractor."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 28:36, :] = 255
    path = tmp_path / "bar.png"
    Image.fromarray(img).save(path)

    model_cfg = {
        "type": "gabor_gwp",
        "name": "gabor_serre",
        "preprocess": "grayscale_luminance",
        "gwp": {"number_of_scales": 2, "number_of_directions": 4, "kernel_size": 15},
    }
    model = build_feature_extractor(model_cfg)
    model.eval()
    arr = np.asarray(Image.open(path).convert("RGB"))
    tensor = preprocess_stimulus(arr, model_cfg=model_cfg, input_size=64)
    with torch.no_grad():
        feat = model(tensor.unsqueeze(0)).squeeze(0).numpy()
    assert feat.shape == (8, 64, 64)
    assert np.isfinite(feat).all()
