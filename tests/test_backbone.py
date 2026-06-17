from __future__ import annotations

import torch

from src.DL_features.backbone import build_feature_extractor


def test_resnet18_layer_shapes():
    model = build_feature_extractor("resnet18", pretrained=False, feature_layer="layer3")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 256, 14, 14)


def test_resnet18_avgpool_shape():
    model = build_feature_extractor("resnet18", pretrained=False, feature_layer="avgpool")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 512, 1, 1)
