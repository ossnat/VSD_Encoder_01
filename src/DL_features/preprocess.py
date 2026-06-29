"""Preprocess averaged VSD frames for ImageNet backbones."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _scale_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image.astype(np.float32)
    if mode == "per_image_zscore":
        m = float(image.mean())
        s = float(image.std())
        s = s if s > 1e-8 else 1.0
        return ((image - m) / s).astype(np.float32)
    if mode == "minmax_01":
        mn = float(image.min())
        mx = float(image.max())
        if mx - mn < 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - mn) / (mx - mn)).astype(np.float32)
    raise ValueError(f"Unsupported input_scaling: {mode!r}")


def preprocess_image(
    image: np.ndarray,
    *,
    input_size: int = 224,
    input_channels: int = 3,
    input_scaling: str = "none",
    imagenet_normalize: bool = True,
) -> torch.Tensor:
    """
    Convert a single (H, W) averaged frame into model input tensor (C, S, S).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")
    scaled = _scale_image(image, input_scaling)
    tensor = torch.from_numpy(scaled).to(torch.float32).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    tensor = F.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    )
    if input_channels == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif input_channels != 1:
        raise ValueError(f"Unsupported input_channels: {input_channels}")

    out = tensor.squeeze(0)
    if imagenet_normalize:
        if out.shape[0] != 3:
            raise ValueError("imagenet_normalize requires 3-channel input")
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return out


def preprocess_stimulus_rgb(
    image: np.ndarray,
    *,
    input_size: int = 224,
    imagenet_normalize: bool = True,
) -> torch.Tensor:
    """
    Convert an RGB stimulus image (H, W, 3) uint8 into model input (3, S, S).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape={image.shape}")
    scaled = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(scaled).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    h, w = int(tensor.shape[2]), int(tensor.shape[3])
    if h != input_size or w != input_size:
        tensor = F.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
    out = tensor.squeeze(0)
    if imagenet_normalize:
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return out


def preprocess_stimulus_grayscale(
    image: np.ndarray,
    *,
    input_size: int = 224,
) -> torch.Tensor:
    """Convert RGB stimulus to luminance tensor (1, S, S) in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape={image.shape}")
    rgb = image.astype(np.float32) / 255.0
    luminance = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )
    tensor = torch.from_numpy(luminance).unsqueeze(0).unsqueeze(0)
    h, w = int(tensor.shape[2]), int(tensor.shape[3])
    if h != input_size or w != input_size:
        tensor = F.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
    return tensor.squeeze(0)


def preprocess_stimulus(
    image: np.ndarray,
    *,
    model_cfg: dict,
    input_size: int = 224,
    imagenet_normalize: bool = True,
) -> torch.Tensor:
    """Dispatch stimulus preprocessing based on model config."""
    mode = model_cfg.get("preprocess", "imagenet_rgb")
    if mode == "imagenet_rgb":
        return preprocess_stimulus_rgb(
            image,
            input_size=input_size,
            imagenet_normalize=imagenet_normalize,
        )
    if mode == "grayscale_luminance":
        return preprocess_stimulus_grayscale(image, input_size=input_size)
    raise ValueError(f"Unsupported preprocess mode: {mode!r}")
