#!/usr/bin/env python3
"""
Pre-embed a set of text prompts via LAION-CLAP HTSAT-tiny's text
encoder and write them to `clap-prompts.json` in the same schema
as `clip-prompts.json`:

    [
      { "prompt": "...", "embedding": [Float, ...] },
      ...
    ]

Embeddings are L2-normalized so the Swift caller just dot-products
the audio embedding against each entry. The text encoder runs once
on the maintainer's Mac — no Core ML conversion needed.

Run once per maintainer release:

    python3 Tools/export_clap_prompt_embeddings.py

Outputs `clap-prompts.json` in the current directory; bundle into
the CLAPAudio.zip alongside `CLAPAudioEncoder.mlmodelc/` per
CLAUDE.md.
"""

import json
import torch
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"

# Prompt set: a mix of target audio-NSFW classes, plausibly-
# overlapping physiological sounds (so the model has to discriminate
# rather than just matching any human-vocalization), and obvious
# safe contrast classes. AudioCaps / Clotho-style natural-language
# captions land in CLAP's training distribution; one-word queries
# like "moaning" tend to score worse than full phrases.
PROMPTS = [
    # Explicit / intimate
    "a person moaning during sex",
    "the sound of sexual intercourse",
    "two people having sex",
    "explicit sexual audio with vocalization",
    "a person having an orgasm",

    # Physiological sounds (could plausibly overlap with explicit)
    "two people kissing",
    "a person breathing heavily",
    "a person grunting from physical effort",
    "a person crying",
    "a person screaming",

    # Safe contrast classes
    "people speaking in conversation",
    "people laughing together",
    "music playing in the background",
    "the ambient room tone of an empty room",
    "the sound of footsteps",
]


def main() -> None:
    print(f"[export] loading {MODEL_ID}")
    model = ClapModel.from_pretrained(MODEL_ID).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    print(f"[export] embedding {len(PROMPTS)} prompts")
    entries = []
    with torch.no_grad():
        for prompt in PROMPTS:
            tokens = tokenizer(
                prompt, padding=True, return_tensors="pt"
            )
            text_out = model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            # Same projection path as `model.get_text_features`.
            feats = model.text_projection(text_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            entries.append({
                "prompt": prompt,
                "embedding": feats.squeeze(0).tolist(),
            })

    out = "clap-prompts.json"
    with open(out, "w") as f:
        json.dump(entries, f)
    print(f"[export] wrote {out} — {len(entries)} prompts × {len(entries[0]['embedding'])}d")


if __name__ == "__main__":
    main()
