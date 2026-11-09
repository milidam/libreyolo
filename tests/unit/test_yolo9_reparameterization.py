"""YOLO9 export reparameterization: BN folding and RepConv branch merging."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.unit


def _randomize_batch_norm(block):
    with torch.no_grad():
        for conv in (block.conv1, block.conv2):
            conv.bn.running_mean.normal_()
            conv.bn.running_var.uniform_(0.5, 2.0)
            conv.bn.weight.normal_()
            conv.bn.bias.normal_()


def test_repconv_convert_to_deploy_matches_training_form():
    from libreyolo.models.yolo9.nn import RepConv

    torch.manual_seed(0)
    block = RepConv(8, 8, 3).eval()
    _randomize_batch_norm(block)

    x = torch.randn(2, 8, 12, 12)
    with torch.no_grad():
        expected = block(x)

    deployed = copy.deepcopy(block).convert_to_deploy()
    with torch.no_grad():
        actual = deployed(x)

    assert deployed.conv1 is None and deployed.conv2 is None
    assert deployed.conv.bias is not None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_repconv_convert_to_deploy_is_idempotent():
    from libreyolo.models.yolo9.nn import RepConv

    block = RepConv(4, 4, 3).eval()
    _randomize_batch_norm(block)
    x = torch.randn(1, 4, 8, 8)

    block.convert_to_deploy()
    with torch.no_grad():
        first = block(x)
    block.convert_to_deploy()
    with torch.no_grad():
        second = block(x)

    torch.testing.assert_close(second, first)


def test_yolo9_deploy_matches_training_form():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model, RepConv

    torch.manual_seed(0)
    model = LibreYOLO9Model(config="t", nb_classes=2).eval()
    x = torch.randn(1, 3, 128, 160)

    with torch.no_grad():
        expected = model(x)["predictions"]

    deployed = copy.deepcopy(model).deploy()
    with torch.no_grad():
        actual = deployed(x)["predictions"]

    assert all(
        module.conv1 is None
        for module in deployed.modules()
        if isinstance(module, RepConv)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_yolo9_deploy_leaves_the_source_model_untouched():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model, RepConv

    model = LibreYOLO9Model(config="t", nb_classes=2).eval()
    original_keys = set(model.state_dict())

    copy.deepcopy(model).deploy()

    assert set(model.state_dict()) == original_keys
    assert any(
        module.conv1 is not None
        for module in model.modules()
        if isinstance(module, RepConv)
    )
