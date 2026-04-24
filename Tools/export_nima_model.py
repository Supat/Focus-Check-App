#!/usr/bin/env python3
"""
Convert idealo/image-quality-assessment (NIMA, technical-quality
variant) to Core ML.

Produces NIMA.mlpackage in the current directory. The model takes a
[1, 224, 224, 3] RGB image in [0, 255] and returns a [1, 10] softmax
over rating bins 1…10. The Swift side takes the expectation
`Σ (i + 1) · p_i` for a single scalar quality score in [1, 10].

Architecturally: MobileNet-v1 backbone (ImageNet pretrained, then
fine-tuned on TID2013) → GlobalAveragePool → Dropout(0.75) →
Dense(10, softmax). Dropout is a no-op at inference so it doesn't
need weights from the checkpoint.

Preprocessing: MobileNet-v1 expects inputs in [-1, 1]
(`(x / 127.5) - 1`). That's NOT baked into Keras's MobileNet graph,
so the Core ML ImageType handles it via `scale = 2/255` and
`bias = [-1, -1, -1]` applied before the tensor hits the first
Conv.

LICENSE NOTE: code is Apache-2.0, and TID2013 is published for
"scientific and educational research". That's more permissive than
the IMDB-WIKI / BP4D / DISFA terms we flagged for other tiers, but
not unambiguously commercial-OK — treat shipping in a signed App
Store build as "check with a lawyer first".

Install conversion deps:

    pip install "tensorflow>=2.13,<2.16" "coremltools>=7.2,<9" "numpy<2" h5py pillow

Download the weights (~17 MB, MobileNet-technical-0.11 checkpoint):

    curl -L -O https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_technical_0.11.hdf5

then, from the Focus-Check-App root:

    python3 Tools/export_nima_model.py

then compile + publish per CLAUDE.md:

    xcrun coremlcompiler compile NIMA.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \\
          /tmp/NIMA.mlmodelc NIMA.mlmodelc.zip
    zip -d NIMA.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
    gh release create quality-model-v1 NIMA.mlmodelc.zip \\
        --repo Supat/Focus-Check-App
"""

import os
import sys

import numpy as np
import coremltools as ct
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

WEIGHTS_PATH = os.environ.get(
    "NIMA_WEIGHTS", "weights_mobilenet_technical_0.11.hdf5"
)
INPUT_SIZE = 224
NUM_BINS = 10

if not os.path.isfile(WEIGHTS_PATH):
    sys.exit(
        f"Weights not found at {WEIGHTS_PATH}. Download with:\n"
        "  curl -L -O https://github.com/idealo/image-quality-assessment/"
        "raw/master/models/MobileNet/weights_mobilenet_technical_0.11.hdf5\n"
        "or set NIMA_WEIGHTS to the downloaded file path."
    )


def build_model() -> Model:
    """Reconstruct idealo's NIMA architecture inline. Mirrors
    `src/handlers/model_builder.py::Nima.build` for the technical
    variant: MobileNet-v1 backbone → dropout → softmax head.
    Dropout rate is irrelevant at inference — TF zeroes the layer
    out when not training — so we set 0 to avoid needing the
    config."""
    base = applications.MobileNet(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        weights=None,        # we'll load the fine-tuned weights below
        include_top=False,
        pooling="avg",
    )
    x = Dropout(0)(base.output)
    x = Dense(units=NUM_BINS, activation="softmax")(x)
    return Model(base.inputs, x)


def main() -> None:
    model = build_model()
    model.load_weights(WEIGHTS_PATH)

    # Sanity on a neutral gray probe — real images range widely
    # (distorted/compressed images push mass toward low bins, clean
    # images toward high bins). A gray probe should produce a
    # plausible mid-range distribution.
    probe = np.full((1, INPUT_SIZE, INPUT_SIZE, 3), 128.0, dtype=np.float32)
    # Apply MobileNet's preprocess manually for the Keras sanity
    # call (the Core ML model will do this itself via ImageType
    # scale/bias at inference).
    probe_mnet = (probe / 127.5) - 1
    probs = model.predict(probe_mnet, verbose=0)[0]
    bins = np.arange(1, NUM_BINS + 1)
    score = float((bins * probs).sum())
    print(f"[sanity] probs={probs.tolist()}")
    print(f"[sanity] expected quality on gray probe: {score:.2f}")

    placeholder_name = model.input.name.split(":")[0]
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        inputs=[ct.ImageType(
            name=placeholder_name,
            shape=(1, INPUT_SIZE, INPUT_SIZE, 3),
            # MobileNet-v1 expects [-1, 1] RGB. 2/255 ≈ 0.00784.
            # applied BEFORE bias, so `y = x * (2/255) + (-1)`.
            scale=2.0 / 255.0,
            bias=[-1.0, -1.0, -1.0],
            color_layout=ct.colorlayout.RGB,
        )],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    # Rename I/O names the converter picked to the stable pair the
    # Swift side prefers. Auto-picked names are `input_1` /
    # `Identity` (or similar) depending on the graph.
    import coremltools.models.utils as ct_utils
    spec = mlmodel.get_spec()
    if placeholder_name != "image":
        ct_utils.rename_feature(spec, placeholder_name, "image")
    for out in list(spec.description.output):
        shape = list(out.type.multiArrayType.shape)
        n = 1
        for d in shape:
            n *= d
        if n == NUM_BINS and out.name != "quality_distribution":
            ct_utils.rename_feature(spec, out.name, "quality_distribution")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    mlmodel.save("NIMA.mlpackage")
    print(f"Wrote NIMA.mlpackage (weights: {WEIGHTS_PATH}, "
          f"input: {INPUT_SIZE}² RGB, bins: {NUM_BINS})")

    if os.environ.get("SKIP_COREML_SANITY"):
        return
    try:
        from PIL import Image
        test = Image.fromarray(
            np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        )
        out = mlmodel.predict({"image": test})
        dist = np.asarray(out["quality_distribution"]).flatten()
        cm_score = float((np.arange(1, NUM_BINS + 1) * dist).sum())
        print(f"[sanity-coreml] distribution {dist.tolist()}")
        print(f"[sanity-coreml] expected quality: {cm_score:.2f}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sanity-coreml] skipped — host predict raised "
            f"{type(exc).__name__}: {exc}. Compile + test the .mlmodelc "
            "on-device to verify."
        )


if __name__ == "__main__":
    main()
