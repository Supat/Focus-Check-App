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
`auto_pad = SAME_UPPER`, which onnx2torch can't ingest directly. We
first set concrete input shape + run shape inference, then rewrite
every Conv / MaxPool auto_pad into an explicit `pads = [...]`
attribute that onnx2torch accepts. onnx-simplifier is also run to
fold any remaining shape arithmetic, but the manual pad rewrite is
what actually unblocks onnx2torch.
"""

import onnx
import onnxsim
import onnx2torch
import torch
import coremltools as ct
from onnx import helper, shape_inference

ONNX_PATH  = "emotion-ferplus-8.onnx"
INPUT_SIZE = 64


def resolve_auto_pad(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run shape inference with the concrete input shape set, then
    rewrite every Conv / MaxPool / AveragePool's `auto_pad =
    SAME_UPPER / SAME_LOWER` attribute into an explicit `pads` list
    that onnx2torch accepts."""
    # Set concrete input shape on every graph input.
    for inp in model.graph.input:
        inp.type.tensor_type.shape.Clear()
        for d in (1, 1, INPUT_SIZE, INPUT_SIZE):
            inp.type.tensor_type.shape.dim.add().dim_value = d

    # Nuke stale `value_info` entries — they still reference the
    # pre-override symbolic batch dim, which makes shape_inference
    # raise "Inferred shape and existing shape differ in rank" when
    # the re-propagated shapes don't match what's cached.
    del model.graph.value_info[:]

    model = shape_inference.infer_shapes(model)

    shape_map: dict[str, list[int]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = dims

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

        in_shape = shape_map.get(node.input[0])
        if not in_shape or len(in_shape) < 3:
            continue
        spatial = in_shape[2:]
        if not kernel:
            continue
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

    return model


def main() -> None:
    graph = onnx.load(ONNX_PATH)

    graph = resolve_auto_pad(graph)

    # Constant-fold anything left over so onnx2torch gets a clean graph.
    graph, passed = onnxsim.simplify(graph)
    assert passed, "onnx-simplifier failed to verify the simplified graph."

    pt_model = onnx2torch.convert(graph).eval()

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
