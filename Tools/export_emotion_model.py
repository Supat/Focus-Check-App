#!/usr/bin/env python3
"""
Convert Microsoft's FER+ emotion-classification ONNX model to Core ML.

Produces EmotionFERPlus.mlpackage in the current directory. The model
classifies a cropped face into 8 emotion labels:

    [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]

Input: 1×1×64×64 grayscale, raw pixel intensities in [0, 255] (the ONNX
graph handles its own normalization). The Swift side pre-crops each
face rectangle from VNDetectFaceLandmarksRequest, converts to grayscale,
and resamples to 64² before running this model.

Install prerequisites:

    pip install onnx onnx-simplifier onnx2torch torch coremltools

Download the ONNX once:

    curl -L -o emotion-ferplus-8.onnx \\
      https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

then:

    python3 Tools/export_emotion_model.py

then compile + publish per CLAUDE.md:

    xcrun coremlcompiler compile EmotionFERPlus.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \\
          /tmp/EmotionFERPlus.mlmodelc EmotionFERPlus.mlmodelc.zip
    gh release create emotion-model-v1 EmotionFERPlus.mlmodelc.zip \\
        --repo Supat/Focus-Check-App

Notes on the auto_pad dance: FER+'s Conv / MaxPool nodes use
`auto_pad = SAME_UPPER`, which onnx2torch can't ingest. We let
onnx-simplifier override the input shape and propagate shapes
through the graph (its internal shape inference is friendlier than
calling onnx.shape_inference directly — the latter raises rank
mismatches on FER+'s dim_param-bearing output), then rewrite every
Conv / MaxPool / AveragePool's auto_pad into explicit pads using
the propagated value_info.
"""

import onnx
import onnxsim
import onnx2torch
import torch
import coremltools as ct
from onnx import helper, version_converter

ONNX_PATH  = "emotion-ferplus-8.onnx"
INPUT_SIZE = 64
TARGET_OPSET = 13   # onnx2torch's supported op versions sit in opset 11+.


def build_shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Collect every tensor's inferred shape from inputs + value_info
    + outputs. Only keeps dims with a concrete dim_value; dim_params
    are dropped (the caller bails out of auto_pad rewriting for any
    node whose input dims aren't fully concrete)."""
    shapes: dict[str, list[int]] = {}
    all_value_infos = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )
    for vi in all_value_infos:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                dims = []
                break
        if dims:
            shapes[vi.name] = dims
    return shapes


def resolve_auto_pad(model: onnx.ModelProto,
                    shapes: dict[str, list[int]]) -> None:
    """Rewrite every Conv / MaxPool / AveragePool's SAME_UPPER /
    SAME_LOWER auto_pad attribute into an explicit pads list using
    the provided shape map. Modifies `model.graph.node` in place."""
    for node in model.graph.node:
        if node.op_type not in ("Conv", "MaxPool", "AveragePool"):
            continue

        auto_pad, kernel, strides, dilations = None, None, None, None
        for attr in node.attribute:
            if attr.name == "auto_pad":     auto_pad = attr.s.decode()
            if attr.name == "kernel_shape": kernel = list(attr.ints)
            if attr.name == "strides":      strides = list(attr.ints)
            if attr.name == "dilations":    dilations = list(attr.ints)

        if auto_pad in (None, "NOTSET", "VALID"):
            continue

        in_shape = shapes.get(node.input[0])
        if not in_shape or len(in_shape) < 3 or not kernel:
            continue
        spatial = in_shape[2:]
        strides = strides or [1] * len(kernel)
        dilations = dilations or [1] * len(kernel)

        pad_before, pad_after = [], []
        for i, k in enumerate(kernel):
            s = strides[i]
            d = dilations[i]
            size = spatial[i]
            out = (size + s - 1) // s
            total = max(0, (out - 1) * s + (k - 1) * d + 1 - size)
            lo = total // 2
            hi = total - lo
            if auto_pad == "SAME_UPPER":
                pad_before.append(lo); pad_after.append(hi)
            else:  # SAME_LOWER
                pad_before.append(hi); pad_after.append(lo)

        kept = [a for a in node.attribute if a.name not in ("auto_pad", "pads")]
        node.ClearField("attribute")
        for a in kept:
            node.attribute.append(a)
        node.attribute.append(helper.make_attribute("pads", pad_before + pad_after))


def main() -> None:
    graph = onnx.load(ONNX_PATH)

    # Let onnx-simplifier override the input shape *and* propagate
    # shapes through the graph. Its internal shape inference tolerates
    # FER+'s dim_param output where onnx.shape_inference.infer_shapes
    # raises a rank mismatch.
    input_name = graph.graph.input[0].name
    simplified, passed = onnxsim.simplify(
        graph,
        overwrite_input_shapes={input_name: [1, 1, INPUT_SIZE, INPUT_SIZE]},
    )
    assert passed, "onnx-simplifier failed to verify the simplified graph."

    # Rewrite remaining auto_pad attributes using the propagated shapes.
    shapes = build_shape_map(simplified)
    resolve_auto_pad(simplified, shapes)

    # Upgrade to a modern opset so onnx2torch's converter registry
    # matches — FER+ ships at opset 7, which has older Dropout /
    # BatchNorm / etc. signatures that aren't registered.
    simplified = version_converter.convert_version(simplified, TARGET_OPSET)

    pt_model = onnx2torch.convert(simplified).eval()

    dummy = torch.rand(1, 1, INPUT_SIZE, INPUT_SIZE) * 255
    traced = torch.jit.trace(pt_model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 1, INPUT_SIZE, INPUT_SIZE),
            color_layout=ct.colorlayout.GRAYSCALE,
            # Pixel values arrive 0..255 directly; the ONNX graph does
            # its own mean subtraction internally.
            scale=1.0,
            bias=[0],
        )],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.save("EmotionFERPlus.mlpackage")
    print(f"Wrote EmotionFERPlus.mlpackage (from {ONNX_PATH} at {INPUT_SIZE}²)")


if __name__ == "__main__":
    main()
