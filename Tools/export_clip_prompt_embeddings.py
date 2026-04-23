#!/usr/bin/env python3
"""
Produce clip-prompts.json from a list of text prompts.

Runs OpenAI CLIP ViT-B/32's text encoder over the prompts defined below
(edit in place) and writes L2-normalized embeddings to a JSON file that
ships inside CLIP.zip alongside the image-encoder .mlmodelc.

    python3 Tools/export_clip_prompt_embeddings.py

The text encoder must match the image encoder from
`export_clip_image_encoder.py` — both come from the same MODEL_ID. If
you swap to a different CLIP variant (ViT-L/14, MobileCLIP, etc.), edit
MODEL_ID in both scripts.

Re-running this script to refine the prompt set is cheap and doesn't
require re-exporting the image encoder — just re-zip CLIP.zip and
re-upload.
"""

import json
import torch
from transformers import CLIPModel, CLIPTokenizer

MODEL_ID = "openai/clip-vit-base-patch32"
OUTPUT_PATH = "clip-prompts.json"

# Prompts are surfaced in the Context badge as-is, so phrase them how
# you'd want them to read. The set mixes nudity / context / safe-art /
# medical / minor anchors so the top match works as a coarse label.
PROMPTS = [
    "a photograph of a nude person",
    "a photograph of two people having sex",
    "a photograph of a person in lingerie",
    "a photograph of a person in a bedroom",
    "a photograph of a person at the beach",
    "a photograph of a fully clothed person",
    "a painting of a nude figure",
    "a medical photograph of human anatomy",
    "a photograph of children playing",
    "a photograph of a landscape",
    "a photograph of food",
    "a portrait photograph",
]


def main() -> None:
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)

    with torch.no_grad():
        inputs = tokenizer(PROMPTS, padding=True, return_tensors="pt")
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    payload = [
        {"prompt": prompt, "embedding": features[i].tolist()}
        for i, prompt in enumerate(PROMPTS)
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {len(payload)} prompts → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
