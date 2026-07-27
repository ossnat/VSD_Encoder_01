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


def test_vgg16_block_shapes():
    model = build_feature_extractor(
        {"type": "vgg", "name": "vgg16", "pretrained": False, "feature_layer": "block4"},
        feature_layer="block4",
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 512, 14, 14)


def test_vgg16_avgpool_shape():
    model = build_feature_extractor(
        {"type": "vgg", "name": "vgg16", "pretrained": False},
        feature_layer="avgpool",
    )
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 512, 7, 7)


def test_vgg16_early_block_pooling():
    cfg = {"type": "vgg", "name": "vgg16", "pretrained": False}
    x = torch.randn(1, 3, 224, 224)
    expected = {
        "block1": (1, 64, 112, 112),
        "block1_pool7": (1, 64, 7, 7),
        "block1_pool14": (1, 64, 14, 14),
        "block2": (1, 128, 56, 56),
        "block2_pool7": (1, 128, 7, 7),
        "block2_pool14": (1, 128, 14, 14),
        "block3": (1, 256, 28, 28),
        "block3_pool14": (1, 256, 14, 14),
    }
    for layer, shape in expected.items():
        model = build_feature_extractor(cfg, feature_layer=layer)
        with torch.no_grad():
            out = model(x)
        assert out.shape == shape, f"{layer}: got {tuple(out.shape)}, expected {shape}"


def test_parse_vgg_feature_layer():
    from src.DL_features.backbone import parse_vgg_feature_layer

    assert parse_vgg_feature_layer("block4") == ("block4", None)
    assert parse_vgg_feature_layer("block1_pool7") == ("block1", 7)
    assert parse_vgg_feature_layer("block2_pool14") == ("block2", 14)


def test_cornet_s_layer_shapes_and_v1_pooling():
    cfg = {"type": "cornet", "name": "cornet_s", "pretrained": False}
    x = torch.randn(1, 3, 224, 224)
    expected = {
        "V1": (1, 64, 56, 56),
        "V1_pool7": (1, 64, 7, 7),
        "V1_pool14": (1, 64, 14, 14),
        "V2": (1, 128, 28, 28),
        "V4": (1, 256, 14, 14),
        "IT": (1, 512, 7, 7),
        "avgpool": (1, 512, 1, 1),
    }
    for layer, shape in expected.items():
        model = build_feature_extractor(cfg, feature_layer=layer)
        with torch.no_grad():
            out = model(x)
        assert out.shape == shape, f"{layer}: got {tuple(out.shape)}, expected {shape}"


def test_cornet_s_model_slug():
    from src.DL_features.schema import model_slug

    assert (
        model_slug({"type": "cornet", "name": "cornet_s", "pretrained": True})
        == "cornet_s_imagenet"
    )


def test_parse_cornet_feature_layer():
    from src.DL_features.cornet_s import parse_cornet_feature_layer

    assert parse_cornet_feature_layer("V1") == ("V1", None)
    assert parse_cornet_feature_layer("V1_pool7") == ("V1", 7)
    assert parse_cornet_feature_layer("V4") == ("V4", None)
