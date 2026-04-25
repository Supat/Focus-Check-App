# extract_genital_crops.py

Extract NudeNet detections from a macOS Photos Library into a
flat folder of crop thumbnails, ready for manual review and
**CreateML** image-classifier training.

This is a **maintainer tool**, not a runtime dependency of the
app. The output is a corpus of candidate training images for
a downstream Core ML model that re-classifies NudeNet's
genital-region detections into a finer-grained schema:

  - `MALE_GENITALIA_COVERED`
  - `MALE_GENITALIA_FLACCID`
  - `MALE_GENITALIA_AROUSAL`
  - `MALE_GENITALIA_ORGASM`
  - `OTHER` (NudeNet false positive, or anatomically not a
    male genital — buttocks, anus, miscellaneous skin / fabric)

Only the classifier is trained from this corpus; the detector
side stays as upstream-NudeNet, no fine-tuning. The classifier
runs after each NudeNet detection in the relevant input
classes and overrides its label with one of the five above.

Replaces the earlier `bootstrap_nudenet_dataset.py` workflow,
which tried to extend NudeNet's detector schema directly and
grew too large for the project. Nothing from that pipeline is
imported or shared here.

## What it does

For each photo in the requested scope of the Photos Library:

1. Exports a JPEG copy to a temporary path (HEIC / RAW
   auto-converted by osxphotos).
2. Runs NudeNet on the **whole image** — no body cropping, no
   segmentation gate. We don't need every detection; we need
   enough training-data volume across the relevant classes.
3. Filters detections to the six NudeNet classes that map onto
   the genital / buttocks / anus region:

   | NudeNet class             | Tag |
   |---------------------------|-----|
   | MALE_GENITALIA_EXPOSED    | MGE |
   | FEMALE_GENITALIA_COVERED  | FGC |
   | FEMALE_GENITALIA_EXPOSED  | FGE |
   | BUTTOCKS_EXPOSED          | BTE |
   | BUTTOCKS_COVERED          | BTC |
   | ANUS_EXPOSED              | ANE |

   On an all-male corpus, FGC / FGE detections are mostly
   misclassified male anatomy — ideal training signal for the
   classifier. BTE / BTC / ANE crops anchor the `OTHER`
   bucket so the classifier learns to distinguish genital
   from non-genital private-region anatomy at inference time.

4. Crops each surviving detection with 15 % outward padding,
   pads the result to a square with black borders (so
   CreateML's resize-to-square doesn't distort the aspect),
   and writes one JPEG per detection into the output's
   `unsorted/` directory.

The label .txt sidecars / classes.txt / data.yaml /
RUBRIC.md / README-review.md artifacts the bootstrap script
emitted are all gone — CreateML reads the five class folders
directly, no detector training metadata needed.

## Safety model

Read-only on the Photos Library. Same guarantees as the
earlier bootstrap script:

- `osxphotos` opens `Photos.sqlite` in read-only SQLite mode.
- The only file-system writes are to `--output` (and to a
  short-lived per-photo temp file via `osxphotos.PhotoExporter`,
  which copies originals — never mutates the library).
- `--output` is resolved and rejected if it overlaps the
  library path in either direction.
- `--dry-run` does zero I/O — enumerates UUIDs and exits.

If any of these guarantees ever break — osxphotos changes
semantics, a new flag gets added — that's a regression. Use
`--dry-run` on any unfamiliar library first.

## Install

```bash
pip install osxphotos nudenet pillow pillow-heif tqdm \
            opencv-python
```

For the larger `640m` NudeNet variant (better recall than the
bundled `320n`):

```bash
curl -L -O https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
```

## Usage

Always start with a dry-run on a small album:

```bash
python3 Tools/extract_genital_crops.py \
    --library  ~/Pictures/Outdoor\ Library.photoslibrary \
    --output   ~/Desktop/genital-crops \
    --album    "Training" \
    --dry-run
```

Real run on a 50-photo smoke test:

```bash
python3 Tools/extract_genital_crops.py \
    --library  ~/Pictures/Outdoor\ Library.photoslibrary \
    --output   ~/Desktop/genital-crops \
    --album    "Training" \
    --limit    50 \
    --model-path ~/Downloads/640m.onnx
```

Full library:

```bash
python3 Tools/extract_genital_crops.py \
    --library  ~/Pictures/Outdoor\ Library.photoslibrary \
    --output   ~/Desktop/genital-crops \
    --no-album-filter \
    --model-path /tmp/640m.onnx \
    --confidence 0.05
```

**One library at a time.** If you want to mix sources, run the
script multiple times with different `--library` values into
the **same** `--output` directory. Filenames include the source
photo's stem, so collisions across libraries are extremely
unlikely; if any do collide the script appends a numeric
suffix.

## Output layout

```
<output>/
├── unsorted/
│   ├── _MGE/        ← NudeNet called these MALE_GENITALIA_EXPOSED
│   ├── _FGC/        ← FEMALE_GENITALIA_COVERED
│   ├── _FGE/        ← FEMALE_GENITALIA_EXPOSED
│   ├── _BTE/        ← BUTTOCKS_EXPOSED
│   ├── _BTC/        ← BUTTOCKS_COVERED
│   └── _ANE/        ← ANUS_EXPOSED
│
├── MALE_GENITALIA_COVERED/    ← reviewer fills these five
├── MALE_GENITALIA_FLACCID/
├── MALE_GENITALIA_AROUSAL/
├── MALE_GENITALIA_ORGASM/
└── OTHER/
```

Filenames inside `unsorted/_*/` look like:

```
IMG_1234_det03_MGE_72.jpg
└─source─┘ └id┘ └NN┘└c%┘
```

- `IMG_1234` — source photo's filename stem.
- `det03` — detection index within that photo (zero-padded).
- `MGE` — NudeNet's original class tag.
- `72` — detection confidence as 2-digit percent.

The class tag and confidence stay in the filename even after
the file is moved into a class folder — useful provenance
when inspecting the training set later.

## Review workflow

In Finder:

1. Open `<output>/unsorted/` and switch to icon view at a
   reasonable thumbnail size. The six sub-folders correspond to
   the NudeNet class that surfaced each crop.
2. Process one sub-folder at a time. On an all-male corpus,
   most of `_FGC/` is `MALE_GENITALIA_COVERED` — multi-select
   the obvious ones, drag the whole batch into the `COVERED`
   class folder. Same for `_BTE/`, `_BTC/`, `_ANE/` → almost
   all into `OTHER`.
3. `_MGE/` is the work-heavy bucket. Sort by name (groups
   crops from the same photo together — useful for multi-
   detection contexts) or by score (right-click column header
   in List view) to triage. Drag each crop into one of
   `FLACCID` / `AROUSAL` / `ORGASM` / `OTHER` based on the
   rubric.
4. Anything ambiguous after a 2-second look: leave in
   `unsorted/` (excluded from training automatically) or
   delete. Don't bucket borderline cases — noisy training
   labels do more damage than fewer clean ones.

## Sub-class rubric

Same diagnostic criteria the earlier project used (which were
already a clinical-style differential):

### MALE_GENITALIA_COVERED
Pelvic pouch fully covered by an opaque garment such that no
genital skin is visible. Bulge through fabric is permitted;
visible tissue is not. Wet / sheer fabric where the penis or
scrotum colour and shape are individually distinguishable
beyond a generic bulge → not COVERED, route to FLACCID /
AROUSAL / ORGASM as appropriate.

### MALE_GENITALIA_FLACCID
Exposed male genitalia with no visible erection. Shaft hangs
parallel to gravity, no rigidity, no engorgement. Default for
non-erotic contexts (bathing, medical, locker-room) and the
post-ejaculatory resolution phase. Semi-erect / 20–40°
ambiguous cases default here unless rigidity is unambiguous.

### MALE_GENITALIA_AROUSAL
Visibly erect — shaft rigid, ≥ 45° from vertical-hanging
(often pointing toward the navel), glans engorged and
noticeably darker than its flaccid baseline, larger overall
circumference. Pre-ejaculate fluid at the meatus stays
AROUSAL.

### MALE_GENITALIA_ORGASM
Ejaculation in progress or just concluded with **visible
evidence**: opaque white / off-white fluid in mid-trajectory
or deposited on body / surroundings, in volume noticeably
larger than pre-ejaculate. The genitals will typically still
be erect; once the fluid is gone and the erection subsides,
the correct class becomes FLACCID. When in doubt between
AROUSAL and ORGASM on a still frame, prefer AROUSAL.

### OTHER
- NudeNet false positive (background, clothing fold, shadow,
  unrelated skin, the wrong body region entirely).
- Non-genital private-region anatomy (the buttocks / anus
  buckets land here so the classifier learns to *not* call
  these MGE-family at inference time).
- Heavily-blurred / occluded crops where no confident sub-
  class call is possible.

## CreateML training

Once the five class folders are populated:

1. Open **CreateML.app** → New Project → **Image
   Classification**.
2. Drop `<output>/` onto the Training Data well. CreateML
   reads the top-level folder names as classes; `unsorted/`
   is excluded automatically because it isn't part of the
   class list (its sub-folders are nested deeper than CreateML
   looks).
3. Default 80 / 20 train/val split.
4. Augmentations: leave **Crop** and **Rotate** on; turn
   **Flip** off (anatomical asymmetry could matter and a
   horizontal flip during training would teach the model to
   treat mirrored anatomy as identical, which it isn't —
   probably negligible at downscaled input but cheap to
   exclude).
5. Backbone: **MobileNetV2** for fastest on-device inference,
   or **BiT** for higher accuracy at the cost of model size
   and ANE compatibility.
6. Train. ~30 min on Apple Silicon for a few thousand images.
7. Inspect the confusion matrix; the FLACCID-vs-AROUSAL
   confusion will be the highest, since the diagnostic is
   continuous (rigidity / angle / engorgement on a spectrum
   rather than a hard threshold).
8. Export the trained model as `.mlmodel`.

## On-device integration sketch

In `Analysis/NudityDetector.swift`, after each NudeNet
detection whose class is one of `{MGE, FGC, FGE}`:

```swift
let crop = source.cropped(to: detection.rect)
if let subClass = genitalClassifier.classify(crop) {
    detection.label = subClass    // overrides the raw NudeNet class
}
```

`genitalClassifier` is a new `MLModel` wrapper following the
existing `EmotionClassifier` / `AgeEstimator` pattern. The
model archive lives in a new GitHub release (e.g.
`genital-classifier-v1`) following the same `ModelArchive`
convention.

The `NudityLevel` enum maps the new sub-classes:

| Sub-class                | NudityLevel |
|--------------------------|-------------|
| MALE_GENITALIA_COVERED   | `.covered`  |
| MALE_GENITALIA_FLACCID   | `.nude`     |
| MALE_GENITALIA_AROUSAL   | `.nude`     |
| MALE_GENITALIA_ORGASM    | `.nude`     |
| OTHER                    | (treat as if NudeNet hadn't fired) |

`BUTTOCKS_*` and `ANUS_*` detections from NudeNet are not
re-classified — they pass through with their original labels
and the existing `NudityLevel` mapping applies.

## Caveats

- **NudeNet is biased**: on an all-male corpus, ~40 % of male
  chests come back as `FEMALE_BREAST_EXPOSED`, lots of male
  groins land in `FEMALE_GENITALIA_*`. The extractor leans
  into that — those misclassifications are *useful training
  signal* for the classifier, since they're crops of the
  same anatomy NudeNet got the gender label wrong on.
- **Whole-image NudeNet under-recalls genital regions** (~36 %
  recall observed empirically on real-world photos). That's
  fine for *training-data volume*; it's a problem when the
  same model is used for *inference* on the device. The
  detector-side recall problem is now out of scope — we
  accept what NudeNet gives us and only fix the labels.
- **CreateML's image classifier has no localization output**.
  The classifier sees the crop NudeNet handed it; it doesn't
  emit a bounding box. The on-device pipeline keeps NudeNet's
  box and only swaps the label.
- **Class imbalance is real**. ORGASM is rarest; expect a
  10×–100× imbalance vs FLACCID. CreateML weights classes
  inversely to frequency by default, but the rare class will
  still have higher variance on the validation confusion
  matrix. Plan for a second pass once you see the matrix.
