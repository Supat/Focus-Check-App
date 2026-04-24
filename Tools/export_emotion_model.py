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
"""

import onnx
import onnx2torch
import torch
import coremltools as ct

ONNX_PATH  = "emotion-ferplus-8.onnx"
INPUT_SIZE = 64


def main() -> None:
    graph = onnx.load(ONNX_PATH)
    pt_model = onnx2torch.convert(graph).eval()

    # FER+ expects single-channel (grayscale) input at (1, 1, 64, 64).
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
