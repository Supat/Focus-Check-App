#!/usr/bin/env python3
"""
Convert the NudeNet detector ONNX to Core ML (`.mlpackage`).

Produces NudeNet.mlpackage in the current directory. The raw YOLO
output tensor is preserved; `NudityDetector.parseYOLODetections` on the
Swift side handles anchor decode, class argmax, and NMS — so this
script doesn't need to graft Create ML's object-detector head on.

Defaults to the `640m` medium variant because its recall is
meaningfully better than the `320n` nano variant on small subjects
(20-ish px features at 320² become 40-ish px at 640², above the YOLO
anchor floor). Swap ONNX_PATH and INPUT_SIZE to use a different
variant.

Run once per maintainer release:

    python3 Tools/export_nudenet_model.py

then compile + bundle per CLAUDE.md:

    xcrun coremlcompiler compile NudeNet.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \
          /tmp/NudeNet.mlmodelc NudeNet.mlmodelc.zip
    gh release create nudenet-model-v2 NudeNet.mlmodelc.zip \
        --repo Supat/Focus-Check-App
"""

import onnx
import onnx2torch
import torch
import coremltools as ct

# Source ONNX — download from the NudeNet repo's python package or its
# release page. 640m gives much better recall than 320n at ~4x latency.
ONNX_PATH  = "640m.onnx"
INPUT_SIZE = 640


def main() -> None:
    graph = onnx.load(ONNX_PATH)
    pt_model = onnx2torch.convert(graph).eval()

    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced = torch.jit.trace(pt_model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
            scale=1 / 255.0,
            bias=[0, 0, 0],
        )],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.save("NudeNet.mlpackage")
    print(f"Wrote NudeNet.mlpackage (from {ONNX_PATH} at {INPUT_SIZE}²)")


if __name__ == "__main__":
    main()
