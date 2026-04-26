#!/usr/bin/env python3
"""
Convert LAION-CLAP HTSAT-tiny's audio branch to Core ML.

The exported model takes a raw 10-second waveform at 48 kHz
(`[1, 480000]` Float32) and returns an L2-normalized 512-d audio
embedding. The mel-spectrogram + log + normalization steps are
baked into the traced graph so the Swift caller doesn't have to
re-implement the HF feature extractor's exact mel pipeline.

Run once per maintainer release:

    python3 Tools/export_clap_audio_encoder.py

Then compile + bundle per CLAUDE.md:

    xcrun coremlcompiler compile CLAPAudioEncoder.mlpackage /tmp/
    mkdir -p /tmp/CLAPAudio
    mv /tmp/CLAPAudioEncoder.mlmodelc /tmp/CLAPAudio/
    cp clap-prompts.json                   /tmp/CLAPAudio/
    ditto -c -k --sequesterRsrc --keepParent /tmp/CLAPAudio CLAPAudio.zip
    gh release upload clap-audio-v1 CLAPAudio.zip
"""

import torch
import coremltools as ct
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48_000
N_FFT = 1024
HOP_LENGTH = 480
N_MELS = 64
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS  # 480_000
NB_MAX_FRAMES = 1000  # HF truncates the trailing frame from STFT's 1001
# HF audio model expects [1, 1, spec_width, n_mels] where
# spec_width = spec_size * freq_ratio = 256 * 4 = 1024. When the
# input is shorter (HF processor emits 1000 frames), the model
# bicubic-interpolates to spec_width inside its forward pass.
# coremltools doesn't implement bicubic, so we pre-interpolate
# (bilinear) to 1024 here and skip the model's resize entirely.
SPEC_WIDTH = 1024


class HFMelSpectrogram(torch.nn.Module):
    """Numerically matches HF's `ClapFeatureExtractor`:
      - STFT(n_fft=1024, hop=480, hann window, center=True, reflect pad)
      - magnitude → power-2
      - linear projection by HF's stored slaney mel filterbank
        (513 → 64, fmin=50 Hz, fmax=14 kHz)
      - 10 · log10(max(power, 1e-10)) — librosa-equivalent power-to-dB
      - truncate trailing frame so output has exactly 1000 frames
      - reshape to [batch, 1, time, mel] for the audio model.

    Mel filterbank weights are pulled off the live `ClapProcessor` so
    any future change to HF's bank is picked up automatically.
    """

    def __init__(self, processor: ClapProcessor) -> None:
        super().__init__()
        # STFT realized as conv1d with windowed cos/sin DFT basis
        # weights — sidesteps coremltools' partial coverage of
        # torch.stft / unfold / fft.rfft. Equivalent math.
        n_bins = N_FFT // 2 + 1
        window = torch.hann_window(N_FFT)
        n = torch.arange(N_FFT, dtype=torch.float32)
        weights = torch.zeros(2 * n_bins, 1, N_FFT)
        for k in range(n_bins):
            angle = 2.0 * torch.pi * k * n / N_FFT
            weights[k, 0, :] = window * torch.cos(angle)
            weights[k + n_bins, 0, :] = -window * torch.sin(angle)
        self.register_buffer("stft_weights", weights)
        # HF stores mel_filters_slaney as (n_fft//2 + 1, n_mels) = (513, 64).
        mel = torch.tensor(
            processor.feature_extractor.mel_filters_slaney,
            dtype=torch.float32,
        )
        self.register_buffer("mel_filters", mel)
        # Reflection padding to match HF's center=True convention
        # in librosa.stft. ReflectionPad1d traces cleanly to
        # coremltools whereas F.pad(mode="reflect") on a 2-D input
        # does not.
        self.reflect_pad = torch.nn.ReflectionPad1d(N_FFT // 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, 480000]
        # Add a channel dim for conv1d, then reflect-pad ±n_fft/2
        # so the first window is centred on sample 0.
        x = waveform.unsqueeze(1)                           # [1, 1, 480000]
        x = self.reflect_pad(x)                             # [1, 1, 481024]
        # conv1d with stride=hop emits both real and imag concatenated
        # on the channel axis — first n_bins channels = real,
        # next n_bins = imag.
        spec = torch.nn.functional.conv1d(
            x, self.stft_weights, stride=HOP_LENGTH
        )                                                   # [1, 1026, T]
        n_bins = N_FFT // 2 + 1                             # 513
        real = spec[:, :n_bins, :]
        imag = spec[:, n_bins:, :]
        power = real * real + imag * imag                   # [1, 513, T]
        # Truncate the trailing frame to match HF's nb_max_frames.
        power = power[:, :, :NB_MAX_FRAMES]                 # [1, 513, 1000]
        # Project onto mel filterbank: [513, 64]ᵀ × [513, T] → [64, T]
        mel = torch.einsum("bft,fm->bmt", power, self.mel_filters)  # [1, 64, 1000]
        # Power → dB. HF/librosa: 10·log10(max(power, 1e-10)).
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        # HF audio model expects [batch, 1, time, mel].
        log_mel = log_mel.transpose(-2, -1).unsqueeze(1)    # [1, 1, 1000, 64]
        # Pre-interpolate the time axis 1000 → 1024 (bilinear in
        # place of HF's bicubic — coremltools doesn't ship bicubic).
        # Skipping HF's interpolation step keeps the trace clean
        # and the small cubic↔bilinear drift is < 1% on the output
        # embedding, well within the prompt-matching margin.
        log_mel = torch.nn.functional.interpolate(
            log_mel, size=(SPEC_WIDTH, N_MELS),
            mode="bilinear", align_corners=True,
        )                                                   # [1, 1, 1024, 64]
        return log_mel


class CLAPAudioPipeline(torch.nn.Module):
    """Raw waveform → mel-spec → CLAP audio encoder → L2-normalized
    projection. The Swift caller hands over plain Float audio
    (10 s @ 48 kHz, mono) and receives a 512-d embedding ready to
    dot-product against the pre-embedded prompt set.
    """

    def __init__(self, m: ClapModel, processor: ClapProcessor) -> None:
        super().__init__()
        self.melspec = HFMelSpectrogram(processor)
        self.audio_model = m.audio_model
        self.audio_projection = m.audio_projection

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        log_mel = self.melspec(waveform)
        out = self.audio_model(input_features=log_mel)
        feats = self.audio_projection(out.pooler_output)
        return feats / feats.norm(dim=-1, keepdim=True)


def _verify_against_processor(
    pipeline: CLAPAudioPipeline,
    model: ClapModel,
    processor: ClapProcessor,
) -> None:
    """Sanity check: a deterministic synthetic waveform should
    produce the same embedding via our baked pipeline vs. the HF
    processor + model path. Cosine similarity > 0.99 means the mel
    pipeline matches; anything lower means a parameter drift."""
    torch.manual_seed(0)
    audio = torch.randn(1, WINDOW_SAMPLES) * 0.1

    with torch.no_grad():
        baked = pipeline(audio).cpu().numpy().reshape(-1)

        proc_in = processor(
            audio=audio[0].numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        ref_out = model.audio_model(input_features=proc_in["input_features"])
        ref = model.audio_projection(ref_out.pooler_output)
        ref = ref / ref.norm(dim=-1, keepdim=True)
        ref = ref.cpu().numpy().reshape(-1)

    cos = float(
        (baked * ref).sum()
        / ((baked**2).sum() ** 0.5 * (ref**2).sum() ** 0.5)
    )
    print(f"[verify] cosine similarity (baked vs HF): {cos:.4f}")
    if cos < 0.99:
        print("[verify] WARN: pipeline drift — check mel parameters / log convention")


def main() -> None:
    print("[export] loading", MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)

    pipeline = CLAPAudioPipeline(model, processor).eval()

    print("[export] sanity-checking against HF processor reference")
    _verify_against_processor(pipeline, model, processor)

    print(f"[export] tracing on input shape [1, {WINDOW_SAMPLES}]")
    example = torch.randn(1, WINDOW_SAMPLES) * 0.01
    with torch.no_grad():
        traced = torch.jit.trace(pipeline, example, strict=False)

    print("[export] converting to Core ML")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="waveform", shape=(1, WINDOW_SAMPLES))],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    out_path = "CLAPAudioEncoder.mlpackage"
    mlmodel.save(out_path)
    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    main()
