"""Backbone registry for spatial feature-map extraction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    resnet18,
    resnet34,
    resnet50,
    vgg16,
)

from src.DL_features.gabor_gwp import build_gabor_gwp_extractor

# Activation maps after each ResNet stage (post-ReLU block output).
FEATURE_LAYERS: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4", "avgpool")
DEFAULT_FEATURE_LAYER = "layer3"

# VGG16 taps: after each stage MaxPool (maps are post-ReLU of the last conv in that block).
VGG_FEATURE_LAYERS: tuple[str, ...] = (
    "block1",
    "block2",
    "block3",
    "block4",
    "block5",
    "avgpool",
)
# Index of the MaxPool ending each VGG16 features stage.
_VGG16_BLOCK_CUTS: dict[str, int] = {
    "block1": 4,
    "block2": 9,
    "block3": 16,
    "block4": 23,
    "block5": 30,
}
DEFAULT_VGG_FEATURE_LAYER = "block4"


class ResNetFeatureExtractor(nn.Module):
    """Run ResNet forward and return the activation map at a chosen stage."""

    def __init__(self, backbone: nn.Module, feature_layer: str) -> None:
        super().__init__()
        if feature_layer not in FEATURE_LAYERS:
            raise ValueError(
                f"Unsupported feature_layer={feature_layer!r}. "
                f"Choose from: {', '.join(FEATURE_LAYERS)}"
            )
        self.feature_layer = feature_layer
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.feature_layer == "layer1":
            return x

        x = self.layer2(x)
        if self.feature_layer == "layer2":
            return x

        x = self.layer3(x)
        if self.feature_layer == "layer3":
            return x

        x = self.layer4(x)
        if self.feature_layer == "layer4":
            return x

        x = self.avgpool(x)
        return x


class VGGFeatureExtractor(nn.Module):
    """Run VGG forward and return the activation map at a chosen stage."""

    def __init__(self, backbone: nn.Module, feature_layer: str) -> None:
        super().__init__()
        if feature_layer not in VGG_FEATURE_LAYERS:
            raise ValueError(
                f"Unsupported VGG feature_layer={feature_layer!r}. "
                f"Choose from: {', '.join(VGG_FEATURE_LAYERS)}"
            )
        self.feature_layer = feature_layer
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_layer == "avgpool":
            x = self.features(x)
            return self.avgpool(x)

        cut = _VGG16_BLOCK_CUTS[self.feature_layer]
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == cut:
                return x
        raise RuntimeError(f"VGG cut index {cut} not reached")


def _load_resnet(name: str, pretrained: bool) -> nn.Module:
    key = name.lower().strip()
    if key == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    if key == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        return resnet34(weights=weights)
    if key == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    raise ValueError(f"Unsupported ResNet backbone: {name!r}")


def _load_vgg(name: str, pretrained: bool) -> nn.Module:
    key = name.lower().strip()
    if key == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        return vgg16(weights=weights)
    raise ValueError(f"Unsupported VGG backbone: {name!r}")


def feature_layers_for_type(backbone_type: str) -> tuple[str, ...]:
    """Return supported feature-layer names for a backbone type."""
    key = backbone_type.lower().strip()
    if key == "resnet":
        return FEATURE_LAYERS
    if key == "vgg":
        return VGG_FEATURE_LAYERS
    if key == "gabor_gwp":
        return ("energy",)
    raise ValueError(f"Unsupported backbone type: {backbone_type!r}")


def _resolve_feature_layer(model_cfg: dict[str, Any]) -> str:
    backbone_type = model_cfg.get("type", "resnet")
    layer = model_cfg.get("feature_layer")
    if backbone_type == "gabor_gwp" and layer in (None, "default", "energy"):
        return "energy"
    if backbone_type == "vgg":
        return str(layer or DEFAULT_VGG_FEATURE_LAYER)
    return str(layer or DEFAULT_FEATURE_LAYER)


def build_feature_extractor(
    model_cfg: dict[str, Any] | str,
    *,
    pretrained: bool = True,
    feature_layer: str | None = None,
) -> nn.Module:
    """
    Return a frozen feature-map extractor for the requested model config.

    Accepts either a full model config dict or legacy ResNet name string.
    """
    if isinstance(model_cfg, str):
        model_cfg = {
            "type": "resnet",
            "name": model_cfg,
            "pretrained": pretrained,
            "feature_layer": feature_layer or DEFAULT_FEATURE_LAYER,
        }
    else:
        model_cfg = dict(model_cfg)
        if feature_layer is not None:
            model_cfg["feature_layer"] = feature_layer

    backbone_type = model_cfg.get("type", "resnet")
    layer = _resolve_feature_layer(model_cfg)

    if backbone_type == "resnet":
        name = model_cfg["name"]
        pt = bool(model_cfg.get("pretrained", True))
        backbone = _load_resnet(name, pt)
        model = ResNetFeatureExtractor(backbone, feature_layer=layer)
    elif backbone_type == "vgg":
        name = model_cfg["name"]
        pt = bool(model_cfg.get("pretrained", True))
        backbone = _load_vgg(name, pt)
        model = VGGFeatureExtractor(backbone, feature_layer=layer)
    elif backbone_type == "gabor_gwp":
        model = build_gabor_gwp_extractor(model_cfg)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type!r}")

    model.eval()
    return model


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
