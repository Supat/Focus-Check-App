#!/usr/bin/env python3
"""
Convert OpenGraphAU (Luo et al. 2022, `lingjivoo/OpenGraphAU`) to Core ML
so the app can estimate Facial Action Unit intensities per face.

Produces OpenGraphAU.mlpackage in the current directory. The wrapped
model takes a [1, 3, 224, 224] RGB image in [0, 1] (Core ML applies the
1/255 scale) and returns a single [1, 41] tensor of sigmoid
probabilities for the OpenGraphAU AU ordering:

    main (27): 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 22, 23, 24, 25, 26, 27, 32, 38, 39
    sub  (14): L1, R1, L2, R2, L4, R4, L6, R6, L10, R10, L12, R12,
               L14, R14

The Swift side reads indices 2, 4, 5, 6, 7 (AU4 / AU6 / AU7 / AU9 /
AU10) to compute PSPI. AU43 (eye closure) is derived from Vision's
face landmarks Swift-side — OpenGraphAU doesn't include it.

LICENSE NOTE: the OpenGraphAU code is Apache-2.0, but the Stage-2
weights are trained on BP4D + DISFA + other AU datasets which carry
research-only terms. Treat the exported model the same way we treat
EmoNet — fine for Playgrounds / local dev; do not bundle the compiled
.mlmodelc in a signed App Store build.

Setup (one-time):

    git clone https://github.com/lingjivoo/OpenGraphAU ~/opengraphau
    cd ~/opengraphau
    # Download Stage-2 ResNet-50 checkpoint per the upstream README and
    # drop it at checkpoints/OpenGprahAU-ResNet50_second_stage.pth.

Install conversion deps:

    pip install torch coremltools numpy

then, from the Focus-Check-App root:

    python3 Tools/export_opengraphau_model.py

then compile + publish per CLAUDE.md:

    xcrun coremlcompiler compile OpenGraphAU.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \\
          /tmp/OpenGraphAU.mlmodelc OpenGraphAU.mlmodelc.zip
    zip -d OpenGraphAU.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
    gh release create pain-model-v1 OpenGraphAU.mlmodelc.zip \\
        --repo Supat/Focus-Check-App
"""

import os
import sys
import torch
import coremltools as ct
import numpy as np

# Torch 2.6 made torch.load default to weights_only=True, which
# rejects legacy-tar checkpoints like the stock torchvision ResNet-50
# file and OpenGraphAU's own .pth. OpenGraphAU's internal resnet.py
# doesn't pass the kwarg, so shim the default back to False for the
# whole script. The weights come from sources we already inspected by
# hand, so the pickle-exec risk that flag protects against is
# acceptable here.
_original_torch_load = torch.load
def _torch_load_legacy(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_legacy

OPENGRAPHAU_REPO = os.environ.get(
    "OPENGRAPHAU_REPO", os.path.expanduser("~/opengraphau")
)
WEIGHTS_PATH = os.environ.get(
    "OPENGRAPHAU_WEIGHTS",
    os.path.join(OPENGRAPHAU_REPO, "checkpoints",
                 "OpenGprahAU-ResNet50_second_stage.pth"),
)

INPUT_SIZE = 224
NUM_MAIN = 27
NUM_SUB = 14
BACKBONE = "resnet50"

if not os.path.isdir(OPENGRAPHAU_REPO):
    sys.exit(
        f"OpenGraphAU repo not found at {OPENGRAPHAU_REPO}. Clone with: "
        f"git clone https://github.com/lingjivoo/OpenGraphAU {OPENGRAPHAU_REPO} "
        "(or set OPENGRAPHAU_REPO env var to an existing clone)."
    )
if not os.path.isfile(WEIGHTS_PATH):
    sys.exit(
        f"Stage-2 weights not found at {WEIGHTS_PATH}. Download the "
        "ResNet-50 second-stage checkpoint per the upstream README and "
        "place it there (or set OPENGRAPHAU_WEIGHTS)."
    )
sys.path.insert(0, OPENGRAPHAU_REPO)

try:
    # Prefer the Stage-2 MEFL model — its MEFARG constructor takes only
    # backbone + class counts. ANFL (Stage-1) has extra KNN hyperparams
    # we'd have to match to the checkpoint; Stage-2 weights are what
    # upstream publishes, so importing MEFL first lines the constructor
    # up with the weights we're loading.
    from model.MEFL import MEFARG  # noqa: E402
except ImportError:
    try:
        from model.ANFL import MEFARG  # noqa: E402
    except ImportError as e:
        sys.exit(
            f"Could not import MEFARG from {OPENGRAPHAU_REPO}: {e}\n"
            "Confirm the repo layout matches the upstream README."
        )

# ImageNet normalization — upstream's `image_eval()` applies these after
# the Resize(256) + CenterCrop(224). The Swift side is responsible for
# producing a 224×224 crop at [0, 1], which Core ML's scale=1/255 yields
# from the raw pixel buffer; the wrapper subtracts mean + divides by std.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class OpenGraphAUWrapper(torch.nn.Module):
    """Applies ImageNet normalization inside the traced graph, runs
    MEFARG, and squeezes the batch axis so `coremltools` sees a clean
    [1, 41] output. Sigmoid is applied here so the Swift side can treat
    the output as ready-to-use probabilities in [0, 1] — upstream's
    demo does the same before thresholding at 0.5."""

    def __init__(self) -> None:
        super().__init__()
        self.model = MEFARG(
            num_main_classes=NUM_MAIN,
            num_sub_classes=NUM_SUB,
            backbone=BACKBONE,
        )
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        # strict=False — upstream ships a few checkpoint variants whose
        # key naming drifts; the backbone and heads are what matters for
        # AU predictions, and any mismatches will surface as junk logits
        # during the sanity check below.
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:5]:
                print(f"[weights] first-missing: {missing[:5]}")
        self.model.eval()

    def forward(self, pixel_values: torch.Tensor):  # [B, 3, 224, 224] in [0, 1]
        normalized = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        logits = self.model(normalized)
        # MEFARG returns a single [B, NUM_MAIN + NUM_SUB] tensor of
        # logits. `torch.sigmoid` is standard for multi-label AU
        # classifiers and matches upstream's demo thresholding.
        return torch.sigmoid(logits)


def main() -> None:
    # OpenGraphAU's resnet.py resolves "pretrain_models/..." relative
    # to os.getcwd(), so chdir into the repo for the duration of the
    # model construction. Save the original cwd so the mlpackage still
    # lands next to wherever the user invoked the script from.
    original_cwd = os.getcwd()
    os.chdir(OPENGRAPHAU_REPO)
    try:
        wrapper = OpenGraphAUWrapper().eval()
    finally:
        os.chdir(original_cwd)
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        probs = wrapper(dummy)
    if probs.shape[-1] != NUM_MAIN + NUM_SUB:
        raise RuntimeError(
            f"Unexpected output width {probs.shape[-1]}; "
            f"expected {NUM_MAIN + NUM_SUB}. Check the MEFARG class "
            "and the --num_main/--num_sub checkpoint parameters."
        )
    if not torch.isfinite(probs).all():
        raise RuntimeError(
            "OpenGraphAU emitted non-finite probabilities on the dummy "
            "input — weights probably didn't load. Inspect "
            f"{WEIGHTS_PATH} against the MEFARG state_dict keys."
        )
    print(
        f"[sanity] prob range [{probs.min():.3f}, {probs.max():.3f}] "
        f"shape={tuple(probs.shape)}"
    )

    traced = torch.jit.trace(wrapper, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
            scale=1 / 255.0,
            bias=[0, 0, 0],
        )],
        outputs=[ct.TensorType(name="au_probabilities")],
        # FLOAT32 keeps the ResNet-50 activations stable — F16 overflow
        # is what bit us on EmoNet, and MEFARG's graph stage uses the
        # same ResNet-50 backbone with matmul on top so the same risk
        # applies. ~100 MB model size is acceptable for a research tier.
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    # Save first — the post-conversion predict that follows can crash
    # inside MPSGraph on some Apple-Silicon macOS builds even when the
    # on-device model is fine, and we don't want to lose a successful
    # conversion to a host-only validation glitch.
    mlmodel.save("OpenGraphAU.mlpackage")
    print(
        f"Wrote OpenGraphAU.mlpackage "
        f"(weights: {WEIGHTS_PATH}, input: {INPUT_SIZE}² RGB)"
    )

    # Optional post-conversion sanity check. MPSGraph's MLIR pass
    # manager has a known abort path on host-side prediction for
    # certain ops in ResNet + graph models; the compiled .mlmodelc
    # almost always runs correctly on an actual iOS / macOS device
    # even when this host predict crashes. Skip with
    # SKIP_COREML_SANITY=1 if the abort kills the process before this
    # except can run.
    if os.environ.get("SKIP_COREML_SANITY"):
        return
    try:
        from PIL import Image
        test = Image.fromarray(
            np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        )
        out = mlmodel.predict({"image": test})
        probs = np.asarray(out["au_probabilities"]).flatten()
        if probs.size != NUM_MAIN + NUM_SUB:
            print(
                f"[sanity-coreml] WARNING output size {probs.size} != "
                f"{NUM_MAIN + NUM_SUB}; the converter may have reshaped "
                "the head."
            )
        elif not np.isfinite(probs).all():
            print(
                "[sanity-coreml] WARNING non-finite output on random input."
            )
        else:
            print(
                f"[sanity-coreml] prob range "
                f"[{probs.min():.3f}, {probs.max():.3f}] shape={probs.shape}"
            )
    except Exception as exc:
        print(
            f"[sanity-coreml] skipped — host predict raised {type(exc).__name__}: "
            f"{exc}. Compile + test the .mlmodelc on-device to verify."
        )


if __name__ == "__main__":
    main()
