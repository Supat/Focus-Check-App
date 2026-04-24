#!/usr/bin/env python3
"""
Convert yu4u/age-gender-estimation (Keras / TF) to Core ML.

Produces AgeGender.mlpackage in the current directory. The model takes a
[1, 224, 224, 3] RGB image in [0, 255] (EfficientNetB3's Rescaling +
Normalization layers are baked into the graph, so the Swift side just
feeds raw pixel bytes) and returns two softmax heads:

    pred_gender : [1, 2]   — [female, male]
    pred_age    : [1, 101] — probability over ages 0…100

The Swift side takes the expectation  `Σ i · p_i`  for a continuous age
estimate and  `sqrt(Σ (i-μ)² · p_i)`  for an uncertainty band.

LICENSE NOTE: the code is MIT, but the IMDB-WIKI training set is
"academic research only". Treat the compiled .mlmodelc the same way we
treat EmoNet / OpenGraphAU — fine for local dev + Playgrounds; do not
bundle in a signed App Store build.

Install conversion deps:

    pip install "tensorflow>=2.13,<2.16" "coremltools>=7" h5py numpy

Download the v0.6 weights (132 MB — EfficientNetB3, MAE 3.44) once:

    curl -L -O \\
      https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5

then, from the Focus-Check-App root:

    python3 Tools/export_age_gender_model.py

then compile + publish per CLAUDE.md:

    xcrun coremlcompiler compile AgeGender.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \\
          /tmp/AgeGender.mlmodelc AgeGender.mlmodelc.zip
    zip -d AgeGender.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
    gh release create age-model-v1 AgeGender.mlmodelc.zip \\
        --repo Supat/Focus-Check-App
"""

import os
import sys

import numpy as np
import coremltools as ct
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

WEIGHTS_PATH = os.environ.get(
    "AGE_GENDER_WEIGHTS", "EfficientNetB3_224_weights.11-3.44.hdf5"
)
INPUT_SIZE = 224
NUM_AGE_BINS = 101   # 0..100 inclusive
NUM_GENDER = 2       # [female, male] — yu4u's training ordering

if not os.path.isfile(WEIGHTS_PATH):
    sys.exit(
        f"Weights not found at {WEIGHTS_PATH}. Download with:\n"
        "  curl -L -O "
        "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/"
        "EfficientNetB3_224_weights.11-3.44.hdf5\n"
        "or set AGE_GENDER_WEIGHTS to the downloaded file path."
    )


def build_model() -> Model:
    """Reconstruct yu4u's two-head model inline. Mirrors
    `src/factory.py::get_model` exactly — EfficientNetB3 backbone with
    `include_top=False`, global-average pool, then one Dense(2) head for
    gender and one Dense(101) head for age. Inlining avoids needing a
    local clone of the upstream repo for this one function."""
    base = applications.EfficientNetB3(
        include_top=False,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling="avg",
    )
    features = base.output
    pred_gender = Dense(NUM_GENDER, activation="softmax",
                        name="pred_gender")(features)
    pred_age = Dense(NUM_AGE_BINS, activation="softmax",
                     name="pred_age")(features)
    return Model(inputs=base.input, outputs=[pred_gender, pred_age])


def main() -> None:
    model = build_model()
    model.load_weights(WEIGHTS_PATH)

    # Sanity check before conversion — EfficientNet's baked-in
    # normalization means a plausible neutral gray input should produce
    # a non-degenerate softmax. Expectation of the age head lands
    # somewhere in 20-40 for most trained checkpoints.
    probe = np.full((1, INPUT_SIZE, INPUT_SIZE, 3), 128.0, dtype=np.float32)
    gender_probe, age_probe = model.predict(probe, verbose=0)
    if not np.isfinite(gender_probe).all() or not np.isfinite(age_probe).all():
        raise RuntimeError(
            "Non-finite probabilities from the Keras model on a neutral "
            "probe input — weights probably didn't load."
        )
    exp_age = float(np.sum(np.arange(NUM_AGE_BINS) * age_probe[0]))
    print(
        f"[sanity] gender probs: {gender_probe[0].tolist()} | "
        f"age expectation: {exp_age:.2f}"
    )

    # Drop explicit `outputs=` — when a tf.keras model is handed
    # straight to coremltools' TF2 frontend, the concrete function's
    # internal output node names don't always match the Keras layer
    # names (`pred_gender` / `pred_age`), and forcing those names
    # trips an "is not in graph" assert inside `extract_sub_graph`.
    # Letting the converter auto-discover produces two outputs named
    # something like `Identity` / `Identity_1`; the Swift side binds
    # by shape (2-dim gender, 101-dim age) via the shape-based
    # fallback in `AgeEstimator`.
    # The TF2 frontend sees the graph's placeholder name, which for a
    # Keras-built EfficientNet is `input_1`, not the `image` label we
    # wanted to hand the ImageType. Match the placeholder here and
    # rename to `image` after conversion; same trick for the two
    # outputs (converter picks `Identity` / `Identity_1`, we rename
    # by head-size to `pred_age` / `pred_gender`).
    placeholder_name = model.input.name.split(":")[0]
    mlmodel = ct.convert(
        model,
        source="tensorflow",
        inputs=[ct.ImageType(
            name=placeholder_name,
            shape=(1, INPUT_SIZE, INPUT_SIZE, 3),
            # EfficientNetB3's Rescaling(1/255) + Normalization layers
            # are baked into the graph. Feed raw 0..255 pixel bytes.
            scale=1.0,
            bias=[0, 0, 0],
        )],
        # F32 keeps the 101-bin age softmax stable; the expectation-
        # over-bins reduction is sensitive to precision drift at the
        # tail (ages 80+ have tiny probabilities but large indices).
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    import coremltools.models.utils as ct_utils
    spec = mlmodel.get_spec()
    # Rename input placeholder → `image` to match the Swift side's
    # preferred name.
    if placeholder_name != "image":
        ct_utils.rename_feature(spec, placeholder_name, "image")
    # Rename whatever output names the converter picked to the stable
    # `pred_age` / `pred_gender` pair the Swift side prefers. 101-dim
    # head is age, 2-dim head is gender. Falls back to the shape-
    # based binding inside `AgeEstimator` if coremltools changes the
    # output shape ordering on us.
    for out in list(spec.description.output):
        shape = list(out.type.multiArrayType.shape)
        element_count = 1
        for dim in shape:
            element_count *= dim
        if element_count == NUM_AGE_BINS and out.name != "pred_age":
            ct_utils.rename_feature(spec, out.name, "pred_age")
        elif element_count == NUM_GENDER and out.name != "pred_gender":
            ct_utils.rename_feature(spec, out.name, "pred_gender")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    mlmodel.save("AgeGender.mlpackage")
    print(f"Wrote AgeGender.mlpackage (weights: {WEIGHTS_PATH}, "
          f"input: {INPUT_SIZE}² RGB, age bins: {NUM_AGE_BINS})")

    if os.environ.get("SKIP_COREML_SANITY"):
        return
    try:
        from PIL import Image
        test = Image.fromarray(
            np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        )
        out = mlmodel.predict({"image": test})
        g = np.asarray(out["pred_gender"]).flatten()
        a = np.asarray(out["pred_age"]).flatten()
        print(
            f"[sanity-coreml] gender {g.tolist()} | "
            f"age expectation {float((np.arange(a.size) * a).sum()):.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[sanity-coreml] skipped — host predict raised "
            f"{type(exc).__name__}: {exc}. Compile + test the .mlmodelc "
            "on-device to verify."
        )


if __name__ == "__main__":
    main()
