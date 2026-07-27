"""Vendored CORnet-S architecture (DiCarlo Lab) for spatial feature extraction.

Source: https://github.com/dicarlolab/CORnet (cornet/cornet_s.py)
Pretrained ImageNet weights: https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

CORNET_S_HASH = "1d3f7974"
CORNET_S_WEIGHTS_URL = (
    f"https://s3.amazonaws.com/cornet-models/cornet_s-{CORNET_S_HASH}.pth"
)

# Cortical-area taps (spatial maps). Optional V1_pool{N} names are handled in the extractor.
CORNET_AREA_LAYERS: tuple[str, ...] = ("V1", "V2", "V4", "IT")
CORNET_FEATURE_LAYERS: tuple[str, ...] = (
    "V1",
    "V1_pool7",
    "V1_pool14",
    "V2",
    "V4",
    "IT",
    "avgpool",
)
DEFAULT_CORNET_FEATURE_LAYER = "V4"


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CORblock_S(nn.Module):
    scale = 4

    def __init__(self, in_channels: int, out_channels: int, times: int = 1) -> None:
        super().__init__()
        self.times = times
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=2, bias=False
        )
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            out_channels, out_channels * self.scale, kernel_size=1, bias=False
        )
        self.nonlin1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            out_channels * self.scale, out_channels, kernel_size=1, bias=False
        )
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = Identity()
        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv_input(inp)
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)
            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)
            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)
            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)
            x = x + skip
            x = self.nonlin3(x)
            x = self.output(x)
        return x


def build_cornet_s() -> nn.Sequential:
    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "V1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv1",
                                    nn.Conv2d(
                                        3,
                                        64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias=False,
                                    ),
                                ),
                                ("norm1", nn.BatchNorm2d(64)),
                                ("nonlin1", nn.ReLU(inplace=True)),
                                (
                                    "pool",
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                ),
                                (
                                    "conv2",
                                    nn.Conv2d(
                                        64,
                                        64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm2", nn.BatchNorm2d(64)),
                                ("nonlin2", nn.ReLU(inplace=True)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
                ("V2", CORblock_S(64, 128, times=2)),
                ("V4", CORblock_S(128, 256, times=4)),
                ("IT", CORblock_S(256, 512, times=2)),
                (
                    "decoder",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                ("flatten", Flatten()),
                                ("linear", nn.Linear(512, 1000)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def load_cornet_s(*, pretrained: bool = True, map_location: str | None = "cpu") -> nn.Module:
    model = build_cornet_s()
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            CORNET_S_WEIGHTS_URL,
            map_location=map_location,
            check_hash=False,
        )
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(_strip_module_prefix(state))
    return model


def parse_cornet_feature_layer(feature_layer: str) -> tuple[str, int | None]:
    """
    Parse ``V1``, ``V2``, ``V4``, ``IT``, ``avgpool``, or ``V1_pool{N}``.

    Returns (area_or_avgpool, pool_size_or_None).
    """
    layer = str(feature_layer).strip()
    if layer in CORNET_AREA_LAYERS or layer == "avgpool":
        return layer, None
    if layer.startswith("V1_pool"):
        size_str = layer[len("V1_pool") :]
        if not size_str.isdigit():
            raise ValueError(
                f"Unsupported CORnet feature_layer={feature_layer!r}. "
                f"Use V1_pool7 / V1_pool14 style names."
            )
        return "V1", int(size_str)
    raise ValueError(
        f"Unsupported CORnet feature_layer={feature_layer!r}. "
        f"Choose from: {', '.join(CORNET_FEATURE_LAYERS)}"
    )


class CORnetFeatureExtractor(nn.Module):
    """Forward through CORnet-S areas; optional adaptive avg-pool for V1 mega-pixels."""

    def __init__(self, backbone: nn.Module, feature_layer: str) -> None:
        super().__init__()
        area, pool_size = parse_cornet_feature_layer(feature_layer)
        self.feature_layer = feature_layer
        self.area = area
        self.pool_size = pool_size
        self.V1 = backbone.V1
        self.V2 = backbone.V2
        self.V4 = backbone.V4
        self.IT = backbone.IT
        self.decoder = backbone.decoder

    def _maybe_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_size is None:
            return x
        return F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.V1(x)
        if self.area == "V1":
            return self._maybe_pool(x)

        x = self.V2(x)
        if self.area == "V2":
            return x

        x = self.V4(x)
        if self.area == "V4":
            return x

        x = self.IT(x)
        if self.area == "IT":
            return x

        # decoder avgpool → (N, 512, 1, 1)
        x = self.decoder.avgpool(x)
        return x
