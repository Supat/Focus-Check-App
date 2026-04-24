#!/usr/bin/env python3
"""
Convert EmoNet (Toisoul et al. 2021) to Core ML.

Produces EmoNet.mlpackage in the current directory. EmoNet outputs
three tensors per face:

    expression: [1, 8] logits (neutral, happy, sad, surprise, fear,
                 disgust, anger, contempt)
    valence   : [1, 1] continuous, [-1, 1]
    arousal   : [1, 1] continuous, [-1, 1]

The Swift side takes the argmax + softmax over `expression` for the
discrete label, uses `valence` and `arousal` directly as P and A in
the PAD vector, and Mehrabian-projects D from the softmax.

LICENSE WARNING: EmoNet is released under Imperial College's CPD
licence — **research use only**. Do not bundle the exported
.mlmodelc in a signed App Store build. Playgrounds / local dev is
fine because no signed distribution is involved.

Setup (one-time):

    git clone https://github.com/face-analysis/emonet ~/emonet
    cd ~/emonet
    pip install -e .
    # Weights ship in the repo under pretrained/, so no separate
    # download step — the script below reads emonet_8.pth from there.

Install conversion deps:

    pip install torch coremltools numpy

then, from the Focus-Check-App root:

    python3 Tools/export_emonet_model.py

then compile + publish per CLAUDE.md:

    xcrun coremlcompiler compile EmoNet.mlpackage /tmp/
    ditto -c -k --sequesterRsrc --keepParent \\
          /tmp/EmoNet.mlmodelc EmoNet.mlmodelc.zip
    zip -d EmoNet.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
    gh release create emotion-model-v2 EmoNet.mlmodelc.zip \\
        --repo Supat/Focus-Check-App
"""

import os
import sys
import torch
import coremltools as ct

# EmoNet's architecture lives in its own repo — point EMONET_REPO at
# wherever you cloned it, or leave the default ~/emonet. The weights
# ship inside that repo under pretrained/emonet_8.pth.
EMONET_REPO = os.environ.get("EMONET_REPO", os.path.expanduser("~/emonet"))
WEIGHTS_PATH = os.path.join(EMONET_REPO, "pretrained", "emonet_8.pth")

INPUT_SIZE = 256
N_EXPRESSION = 8

if not os.path.isdir(EMONET_REPO):
    sys.exit(
        f"EmoNet repo not found at {EMONET_REPO}. "
        "Clone with: git clone https://github.com/face-analysis/emonet "
        f"{EMONET_REPO}  (or set EMONET_REPO env var to an existing clone)."
    )
sys.path.insert(0, EMONET_REPO)

try:
    from emonet.models import EmoNet  # noqa: E402
except ImportError as e:
    sys.exit(
        f"Could not import emonet.models.EmoNet from {EMONET_REPO}: {e}\n"
        "Run `pip install -e .` inside the emonet repo first."
    )

# ImageNet normalization stats EmoNet expects — baked into the traced
# wrapper so the Core ML input takes raw [0, 1] pixel values.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class EmoNetWrapper(torch.nn.Module):
    """Normalizes [0, 1] RGB → ImageNet stats, runs EmoNet, and
    returns a tuple instead of a dict so `torch.jit.trace` doesn't
    collapse the structure."""

    def __init__(self) -> None:
        super().__init__()
        self.emonet = EmoNet(n_expression=N_EXPRESSION)

        # Load weights defensively — EmoNet's checkpoint sometimes
        # ships as a raw state_dict, sometimes nested under a
        # 'state_dict' key, and occasionally with 'module.' prefixes
        # left over from DataParallel training. Strict loading
        # surfaces any remaining mismatches so the exported model
        # isn't silently running on randomly-initialized weights
        # (the hallmark of that failure mode is NaN outputs).
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.emonet.load_state_dict(state, strict=True)
        self.emonet.eval()

    def forward(self, pixel_values: torch.Tensor):  # [B, 3, H, W] in [0, 1]
        normalized = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        out = self.emonet(normalized)
        return out["expression"], out["valence"], out["arousal"]


def main() -> None:
    wrapper = EmoNetWrapper().eval()
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Sanity check the raw PyTorch model before tracing. If weights
    # failed to load, the expression logits will be garbage (±inf,
    # NaN, or wildly divergent). Bail out early with a readable
    # error rather than exporting a silently-broken .mlpackage.
    with torch.no_grad():
        expr, val, ars = wrapper(dummy)
    if not torch.isfinite(expr).all():
        raise RuntimeError(
            "EmoNet emitted non-finite expression logits on the dummy "
            "input — weights probably didn't load. Inspect "
            f"{WEIGHTS_PATH} keys vs. EmoNet class attributes."
        )
    print(
        f"[sanity] expression range [{expr.min():.3f}, {expr.max():.3f}] "
        f"valence={val.item():.3f} arousal={ars.item():.3f}"
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
        outputs=[
            ct.TensorType(name="expression"),
            ct.TensorType(name="valence"),
            ct.TensorType(name="arousal"),
        ],
        # EmoNet's ResNet-50 backbone produces activations that overflow
        # F16 range on natural faces (post-conv + batchnorm can exceed
        # 65504), which propagates as NaN through the rest of the graph.
        # Convert at F32 — ~100 MB model but stable inference.
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    # Post-conversion sanity: run the compiled model on a dummy input
    # and fail loudly if Core ML introduces NaN/Inf. Lighter-weight than
    # asking the maintainer to notice it on-device.
    import numpy as np
    from PIL import Image
    test = Image.fromarray(
        np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    )
    out = mlmodel.predict({"image": test})
    exp = np.asarray(out["expression"])
    val = np.asarray(out["valence"]).flatten()
    ars = np.asarray(out["arousal"]).flatten()
    if not (np.isfinite(exp).all() and np.isfinite(val).all() and np.isfinite(ars).all()):
        raise RuntimeError(
            "Core ML model emits non-finite outputs on a random test image. "
            "Precision, input layout, or the converter itself may be at fault. "
            "Inspect coremltools warnings above."
        )
    print(
        f"[sanity-coreml] expression range [{exp.min():.3f}, {exp.max():.3f}] "
        f"valence={val[0]:.3f} arousal={ars[0]:.3f}"
    )

    mlmodel.save("EmoNet.mlpackage")
    print(f"Wrote EmoNet.mlpackage (weights: {WEIGHTS_PATH}, input: {INPUT_SIZE}² RGB)")


if __name__ == "__main__":
    main()
