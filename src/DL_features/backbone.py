"""Backbone registry for spatial feature-map extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

# Activation maps after each ResNet stage (post-ReLU block output).
# avgpool = last map before the original FC classifier.
FEATURE_LAYERS: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4", "avgpool")
DEFAULT_FEATURE_LAYER = "layer3"


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
    raise ValueError(f"Unsupported backbone: {name!r}")


def build_feature_extractor(
    name: str,
    *,
    pretrained: bool = True,
    feature_layer: str = DEFAULT_FEATURE_LAYER,
) -> ResNetFeatureExtractor:
    """Return a frozen feature-map extractor for the requested ResNet stage."""
    backbone = _load_resnet(name, pretrained)
    model = ResNetFeatureExtractor(backbone, feature_layer=feature_layer)
    model.eval()
    return model


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
