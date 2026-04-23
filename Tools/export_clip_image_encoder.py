#!/usr/bin/env python3
"""
Convert OpenAI CLIP ViT-B/32's image encoder to Core ML.

Outputs CLIPImageEncoder.mlpackage in the current directory. The output
embedding is L2-normalized inside the traced graph so the Swift side
just computes dot products against the pre-normalized prompt embeddings.

Run once per maintainer release:

    python3 Tools/export_clip_image_encoder.py

then compile + bundle per CLAUDE.md:

    xcrun coremlcompiler compile CLIPImageEncoder.mlpackage /tmp/
    mkdir -p /tmp/CLIP
    mv /tmp/CLIPImageEncoder.mlmodelc /tmp/CLIP/
    cp clip-prompts.json              /tmp/CLIP/
    ditto -c -k --sequesterRsrc --keepParent /tmp/CLIP CLIP.zip
    gh release upload clip-model-v1 CLIP.zip
"""

import torch
import coremltools as ct
from transformers import CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
INPUT_SIZE = 224     # CLIP ViT-B/32's expected short-side length.

# Standard CLIP ImageNet-style normalization. Baking it inside the traced
# graph lets the Core ML input be an unnormalized RGB image, so the Swift
# side just feeds a center-cropped CIImage through `ciContext.render` into
# a 32BGRA CVPixelBuffer.
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip: CLIPModel) -> None:
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        normalized = (pixel_values - MEAN) / STD
        feats = self.clip.get_image_features(pixel_values=normalized)
        return feats / feats.norm(dim=-1, keepdim=True)


def main() -> None:
    clip = CLIPModel.from_pretrained(MODEL_ID).eval()
    encoder = ImageEncoder(clip).eval()

    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced = torch.jit.trace(encoder, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
            scale=1 / 255.0,        # coremltools feeds uint8 [0, 255] → [0, 1]
            bias=[0, 0, 0],
        )],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    mlmodel.save("CLIPImageEncoder.mlpackage")
    print("Wrote CLIPImageEncoder.mlpackage")


if __name__ == "__main__":
    main()
