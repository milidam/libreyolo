"""YOLO9 ``decode="host"`` exports: raw head tensors decode to the same boxes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def test_raw_head_outputs_decode_to_the_graph_decode():
    from libreyolo.export.exporter import (
        _YOLO9RawOutputWrapper,
        _YOLO9SplitOutputWrapper,
    )
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.postprocess.yolo9 import decode_raw_levels

    torch.manual_seed(0)
    model = LibreYOLO9Model(config="t", nb_classes=2).eval()
    canvas_hw = (128, 160)
    x = torch.randn(1, 3, *canvas_hw)

    split = _YOLO9SplitOutputWrapper(model, canvas_hw).eval()
    raw = _YOLO9RawOutputWrapper(model, canvas_hw).eval()
    with torch.no_grad():
        boxes, scores = split(x)
        raw_outputs = raw(x)

    levels = [
        {
            "stride": int(stride),
            "height": canvas_hw[0] // int(stride),
            "width": canvas_hw[1] // int(stride),
        }
        for stride in model.head.strides
    ]
    assert len(raw_outputs) == 2 * len(levels)

    decoded = decode_raw_levels(
        [tensor.numpy() for tensor in raw_outputs[0::2]],
        [tensor.numpy() for tensor in raw_outputs[1::2]],
        levels,
        canvas_hw,
        reg_max=model.head.reg_max,
    )
    expected = np.concatenate(
        [boxes.numpy(), np.transpose(scores.numpy(), (0, 2, 1))], axis=1
    )

    np.testing.assert_allclose(decoded, expected, rtol=1e-5, atol=1e-5)


def test_raw_decoder_accepts_nhwc_tensors():
    from libreyolo.postprocess.yolo9 import decode_raw_levels

    levels = [{"stride": 8, "height": 2, "width": 3}]
    box = np.zeros((1, 2, 3, 4 * 16), dtype=np.float32)
    cls = np.zeros((1, 2, 3, 2), dtype=np.float32)

    nchw = decode_raw_levels(
        [np.transpose(box, (0, 3, 1, 2))], [np.transpose(cls, (0, 3, 1, 2))],
        levels, (16, 24),
    )
    nhwc = decode_raw_levels([box], [cls], levels, (16, 24))

    np.testing.assert_allclose(nchw, nhwc)
    assert nchw.shape == (1, 4 + 2, 6)


def test_backend_pairs_raw_outputs_by_shape_not_order():
    from libreyolo.backends.tflite import TFLiteBackend
    from libreyolo.postprocess.yolo9 import decode_raw_levels

    rng = np.random.default_rng(0)
    levels = [{"stride": 8, "height": 4, "width": 6}]
    boxes = rng.random((1, 4, 6, 4 * 16), dtype=np.float32)
    classes = rng.random((1, 4, 6, 2), dtype=np.float32)

    backend = object.__new__(TFLiteBackend)
    backend._raw_levels = levels
    backend._reg_max = 16
    backend.nb_classes = 2
    backend._canvas_hw = (32, 48)

    # Class tensor first and both in NCHW: pairing must use the shapes.
    merged = backend._decode_raw_outputs(
        [
            np.transpose(classes, (0, 3, 1, 2)),
            np.transpose(boxes, (0, 3, 1, 2)),
        ]
    )
    expected = decode_raw_levels([boxes], [classes], levels, (32, 48), 16)

    np.testing.assert_allclose(merged, expected)
