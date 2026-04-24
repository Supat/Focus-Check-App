# bootstrap_nudenet_dataset.py

Extract photos from the macOS Photos Library and pre-label them with
NudeNet to seed a dataset for fine-tuning a custom detector.

This is a **maintainer tool**, not a runtime dependency of the app. It
exists because NudeNet's default weights have known weak spots
(under-detection of male anatomy, `FACE_MALE` ↔ `FACE_FEMALE` flips)
and the app's own `ModelArchive.nudenet` tier could be replaced by a
fine-tuned checkpoint if enough reviewed training data exists.

## What it does

1. Walks a subset of the Photos Library (album, UUID list, or the
   entire library).
2. Exports the originals to a user-specified output directory,
   converting HEIC/RAW to JPEG by default so NudeNet and downstream
   annotation tools can read them.
3. Runs NudeNet on each exported image and writes a YOLO-format
   `.txt` sidecar with its bounding-box predictions.
4. Emits a `data.yaml` and `classes.txt` wired for Ultralytics
   YOLOv8 fine-tuning, plus a `README-review.md` with the human-
   review checklist.

The labels it writes are **candidate annotations from a biased
model**. They are *not* the dataset. The dataset is what you have
after you review and correct them.

## Safety model

The script is read-only on the Photos Library. The guarantees are:

- **`osxphotos` opens `Photos.sqlite` in read-only SQLite mode.** There
  is no API path in `osxphotos` that writes back to the library.
  We import only `PhotosDB` and iterate `photos()`; no mutation
  methods are referenced.
- **The only file-system write is to `--output`.** Before any write,
  the script resolves both the library path and `--output` and
  aborts if either one is a subpath of the other. This catches the
  case where a user passes `--output ~/Pictures/Photos Library.photoslibrary/…`.
- **`photo.export()` is a copy.** It reads the original from the
  library and writes a new file into the output directory. It does
  not mutate or re-import anything.
- **`--dry-run` does zero I/O.** Enumerates photo UUIDs and exits.
- **Label writes are atomic.** Each `.txt` is written to a `.tmp`
  file and renamed over the final path, so an interrupted run
  doesn't leave partial sidecars.

If any of these guarantees ever break — osxphotos changes semantics,
a new flag gets added, the script learns to do deletes — that's a
regression. Prefer `--dry-run` on any unfamiliar library to confirm
the enumeration looks right before removing the flag.

## Install

```bash
pip install osxphotos nudenet pillow pillow-heif tqdm
```

For the larger `640m` NudeNet variant (higher recall than the
bundled `320n`):

```bash
curl -L -O https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx
```

## Usage

Start with a dry run on a small album:

```bash
python3 Tools/bootstrap_nudenet_dataset.py \
    --output ~/Desktop/nude-dataset \
    --album "Training" \
    --dry-run
```

The output is UUIDs + filenames, capped at the first 20. No files
are created anywhere.

Real smoke-test run — exports 50 photos from the same album and
writes pre-labels at confidence ≥ 0.25 using the 640m model:

```bash
python3 Tools/bootstrap_nudenet_dataset.py \
    --output ~/Desktop/nude-dataset \
    --album "Training" \
    --limit 50 \
    --confidence 0.25 \
    --model-path ~/Downloads/640m.onnx \
    --inference-resolution 640
```

If you want to target specific photos by UUID (e.g., a curated list
you exported from Photos' Get Info panel):

```bash
python3 Tools/bootstrap_nudenet_dataset.py \
    --output ~/Desktop/nude-dataset \
    --uuid-file ~/Desktop/selected-uuids.txt
```

Full-library runs are possible but require `--no-album-filter` to be
explicit about the scope:

```bash
python3 Tools/bootstrap_nudenet_dataset.py \
    --output ~/Desktop/nude-dataset \
    --no-album-filter \
    --limit 1000
```

### Non-default / external libraries

`--library` points at any `.photoslibrary` bundle, including ones on
external drives, backups, or a separate library you switch to in
Photos via `File ▸ Open Library…`.

```bash
python3 Tools/bootstrap_nudenet_dataset.py \
    --library "/Volumes/PhotoBackup/Archive.photoslibrary" \
    --output ~/Desktop/archive-dataset \
    --album "Training" \
    --dry-run
```

`~` is expanded, so `--library ~/Pictures/OldLibrary.photoslibrary`
works. The script also accepts a direct path to the internal
`database/Photos.sqlite` if you need to target that — same
safety guarantees apply. Passing anything that isn't a bundle or a
Photos SQLite file aborts with a specific error before any I/O.

## Output layout

```
<output>/
├── images/              — exported JPEGs (HEIC / RAW auto-converted)
├── labels/              — YOLO-format .txt sidecars, one per image
├── crops/               — per-class thumbnails of every detection
│   ├── MALE_GENITALIA_EXPOSED/
│   │   ├── IMG_1234_00_42.jpg   — image-stem, det-index, confidence%
│   │   └── ...
│   └── <other classes>/
├── classes.txt          — NudeNet labels, one per line; index = class_id
├── data.yaml            — Ultralytics dataset config
├── README-review.md     — review checklist (generated at runtime)
└── RUBRIC.md            — annotation rules for the three new
                          genital-state sub-classes (18/19/20)
```

`crops/` is a review convenience, not a training input. Scroll
`crops/<CLASS>/` in Finder's icon view to spot obvious NudeNet
errors and reclassify the `MALE_GENITALIA_EXPOSED` (class 14) boxes
per the rubric in `RUBRIC.md` before training.

## Review step (non-optional)

Before training, every pre-label must be reviewed. Using the raw
NudeNet output to train a NudeNet replacement would just bake in
NudeNet's biases. The review is where the value comes from.

Tools that load YOLO-format labels for inline correction:

- **CVAT** — web-based, best for multi-reviewer workflows.
- **Label Studio** — local or web, good for custom label hierarchies.
- **LabelImg** — simple desktop app, fastest for single-reviewer runs.

Prioritize the classes we know NudeNet handles poorly:

- `MALE_GENITALIA_EXPOSED` — add missing boxes.
- `FACE_MALE` vs `FACE_FEMALE` — fix the flips.
- Loose boxes — tighten to the subject.
- Duplicates — NMS can leave neighboring boxes; collapse them.

## Fine-tune

Once the labels are reviewed, carve `images/` and `labels/` into
`train/` and `val/` subsets (~80/20). Fill `data.yaml`'s `path:` to
point at the dataset root.

```bash
pip install ultralytics

# Start from NudeNet's published 640m.pt so the backbone is already
# anatomy-tuned; fine-tune for a few epochs on your reviewed set.
yolo detect train \
    model=/path/to/640m.pt \
    data=~/Desktop/nude-dataset/data.yaml \
    epochs=30 \
    imgsz=640 \
    patience=5
```

Freezing the backbone (`freeze=10` or higher in Ultralytics) is
safer on small datasets — only the detection head learns, so the
model doesn't catastrophically forget everything NudeNet's original
training gave it.

Export to ONNX when you're done:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```

Then feed the resulting `best.onnx` through the existing
`Tools/export_nudenet_model.py` to produce the Core ML bundle, and
bump `ModelArchive.nudenet` to the next version tag.

## Caveats

- **Personal libraries overfit.** One person's Photos library is not
  a representative sample. Mixing a public baseline (a random
  subsample of NudeNet's original training data) during fine-tune
  preserves generalization.
- **Class imbalance is severe.** Almost all personal photos are
  fully-clothed. Expect 1-2 orders of magnitude more negative
  examples than positive ones for the exposure classes. Use
  `class_weights` or targeted sampling during training.
- **NudeNet's blind spots are inherited by the pre-labels.** If
  NudeNet misses a subject, this script misses it too — the review
  step has to fill those in manually.
