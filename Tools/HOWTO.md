# HOWTO — train the genital sub-class classifier in CreateML.app

End-to-end workflow from a reviewed crop dataset to a `.mlmodelc.zip`
artifact ready for the on-device `ModelArchive.genitalClassifier`
download. Assumes extraction (via `extract_genital_crops.py`) and
manual review are already complete — every crop intended for
training lives inside one of the five class folders at the dataset
root, and `unsorted/` is empty (or contains only crops deliberately
excluded).

Companion docs:

- `README-extract-genital-crops.md` — extraction + review pipeline
  that produces the dataset CreateML consumes here.
- `FocusApp.swiftpm/Sources/App/Analysis/GenitalClassifier.swift`
  — Swift wrapper that loads the trained model on-device and runs
  it after every NudeNet detection in MGE / FGC / FGE.

## 0. Prerequisites — verify the dataset before opening CreateML

```bash
# Sanity-check class counts. Aim for ≥200 per class minimum, ≥1k each
# is comfortable. ORGASM will be the rarest; if it's <50 the model
# will memorize that bucket — consider folding it into FLACCID or
# downsampling AROUSAL/FLACCID to keep the imbalance under ~10×.
for c in MALE_GENITALIA_COVERED MALE_GENITALIA_FLACCID \
         MALE_GENITALIA_AROUSAL MALE_GENITALIA_ORGASM OTHER; do
    n=$(ls /Volumes/Narita/extraction-dataset/$c 2>/dev/null | wc -l)
    printf "%6d  %s\n" "$n" "$c"
done
```

CreateML refuses to train when any class is empty, and faithfully
memorizes whatever leftover junk is in the folders. Fix bucketing
mistakes now; not after the first confusion matrix.

If multiple library extractions have been run (e.g. an Outdoor run
plus a Hi-Res run), decide whether to merge into one training set
or train per-corpus. **Merging is recommended** — more variety
generalises better than per-corpus segregation:

```bash
# Move Hi-Res reviewed crops into the Outdoor dataset's class folders
for c in MALE_GENITALIA_COVERED MALE_GENITALIA_FLACCID \
         MALE_GENITALIA_AROUSAL MALE_GENITALIA_ORGASM OTHER; do
    src=/Volumes/Narita/extraction-hires-dataset/$c
    dst=/Volumes/Narita/extraction-dataset/$c
    [ -d "$src" ] && mv "$src"/*.jpg "$dst"/ 2>/dev/null
done
```

Filenames don't collide across libraries (each crop carries the
source photo's stem prefix), so `mv` is safe.

## 1. Open CreateML and start a new project

1. **CreateML.app** is bundled with Xcode — `Xcode.app → Open
   Developer Tool → Create ML`. Or `xcrun open -a CreateML` from
   terminal.
2. **File → New Project** (`⌘N`).
3. Pick the **Image Classification** template (top-left, camera
   icon labelled "Image Classification"). Click **Next**.
4. Project name: `GenitalClassifier-v1`. Set the project location
   somewhere outside `/Volumes/Narita/` — CreateML writes
   intermediate files; keep them on a fast local disk.
5. Click **Done**. CreateML opens the project window with empty
   Training Data / Testing Data wells.

## 2. Drop the dataset onto the Training Data well

1. In Finder, navigate to the dataset root (e.g.
   `/Volumes/Narita/extraction-dataset/`).
2. Drag the **dataset root folder itself** (not the inner class
   folders) onto the **Training Data** well in CreateML.
3. CreateML scans for top-level subfolders and reads their names
   as class labels. The summary should read:
   ```
   5 classes
   N images
   MALE_GENITALIA_COVERED, MALE_GENITALIA_FLACCID,
   MALE_GENITALIA_AROUSAL, MALE_GENITALIA_ORGASM, OTHER
   ```
4. **`unsorted/` will appear as a 6th "class" if it contains
   anything**. CreateML treats every direct subfolder as a class.
   Two options to exclude:
   - **Recommended**: empty `unsorted/` first
     (`rm -rf /Volumes/Narita/extraction-dataset/unsorted/`).
   - Or move it outside the dataset root.

   The buckets-by-NudeNet-tag (`unsorted/_MGE/`, etc.) sit two
   levels deep, so CreateML won't pick them up as classes — only
   `unsorted` itself appears as a sibling class, which is wrong
   for training.

5. Leave the **Testing Data** well empty. CreateML auto-splits a
   validation set from training (default 5%); for typical dataset
   sizes here that's around right.

## 3. Configure training parameters

Click the gear icon / **Settings** panel (right side of the
project window).

1. **Algorithm / Feature Extractor** — choose between two options:
   - **Image Feature Print (default)** — Apple's Vision-pretrained
     transformer. Very small inference graph, runs on ANE,
     ~1 MB output. Good first choice.
   - **BiT-M-R50x1** — Google Big Transfer, much larger,
     marginally better accuracy on hard classes. Use only if the
     first run's confusion matrix on FLACCID-vs-AROUSAL is
     unacceptable. ~100 MB inference graph.
   - Stick with **Image Feature Print** for the first run.

2. **Iterations**: 25 is the default. Bump to **50–100** for this
   dataset — class boundaries (FLACCID vs AROUSAL) are continuous,
   so a longer training schedule helps. Check the loss curve; if
   it plateaus by iteration 30 the longer schedule isn't earning
   its keep and the next run can revert.

3. **Validation split**: leave at **Auto** unless a manual
   hold-out has been prepared. CreateML samples ~5% from each
   class.

4. **Augmentations** — selective:
   - **✓ Crop** — random translation crops simulate the slight
     variance in NudeNet's box placement.
   - **✓ Rotate** — small angle variation; helpful for handheld-
     camera tilts.
   - **☐ Flip** — *unchecked*. Anatomical asymmetry (especially
     scrotum / penis orientation) is signal worth keeping. Plus
     padding-to-square is centred, so a horizontal flip changes
     which side has the black bar.
   - **✓ Expose** — exposure variation; helps generalisation
     across lighting.
   - **☐ Blur** — *unchecked*. Many genuine motion-blurred crops
     are already in the dataset; injecting more degrades the
     model's ability to distinguish detailed states (rigidity,
     fluid presence).
   - **☐ Noise** — *unchecked*. Same reasoning.

5. **Maximum Iterations** (BiT only) — leave default unless
   overriding.

## 4. Train

1. Click the **Train** button (top-left of the project window,
   blue play icon).
2. CreateML splits the data, builds the validation set, and
   starts iterating. Live metrics appear:
   - **Training Accuracy** — should rise rapidly and plateau by
     70–90%.
   - **Validation Accuracy** — what actually matters. If it's
     >5–10 points below training accuracy, the model is overfitting
     (more data needed, or fewer iterations).
3. Training time on Apple Silicon: ~10–30 min for Image Feature
   Print, ~1–2 h for BiT, depending on dataset size.
4. CreateML highlights the iteration with the **best validation
   accuracy** automatically. That's the snapshot it'll export.

**If training fails immediately:**
- *"Not enough images per class"* — check class minimums (CreateML
  wants ~10+ per class minimum).
- *"Folder X has no images"* — empty `unsorted/` per step 2.
- *Generic permission error* — output volume might be out of space
  or the project location isn't writable.

## 5. Evaluate the model

When training finishes, two new tabs appear: **Training** and
**Validation**.

1. Click **Validation**.
2. **Confusion matrix** — rows are true class, columns are
   predicted. The diagonal is correct predictions.
3. Things to look at:
   - **OTHER row**: false-positive rate. Off-diagonal numbers in
     the OTHER row (predicted as a genital sub-class) mean the
     classifier will leak NudeNet false positives through as
     genital labels on-device. Fix in step 7.
   - **COVERED ↔ FLACCID confusion**: rare; if present, garments
     may be ambiguous in some crops (sheer / tight white).
   - **FLACCID ↔ AROUSAL confusion**: expected to be highest. The
     diagnostic is continuous (rigidity, angle). If >15%, consider
     adding manual review of borderline cases or training BiT.
   - **ORGASM**: low recall is expected because it's the rarest
     class. If recall is <60% and the class matters, the answer is
     more training data, not more iterations.

4. **Precision** column: of the model's predictions in this class,
   how many were right. Care about high precision on COVERED,
   FLACCID, AROUSAL, ORGASM.
5. **Recall** column: of all true examples in this class, how
   many the model caught. Care about high recall on OTHER (don't
   want NudeNet false positives leaking through as genital labels)
   and COVERED.

If either matrix looks unacceptable: stop here, refine the
training data (see step 7), retrain.

## 6. Export the .mlmodel

When the validation matrix is acceptable:

1. Switch to the **Output** tab (or right-click the trained model
   snapshot in the left sidebar).
2. Click **Get** (or drag the model icon) — CreateML offers a
   `.mlmodel` file. Save it as `GenitalClassifier.mlmodel` in
   `~/Downloads/` or wherever convenient.
3. Verify the file's class list:
   ```bash
   python3 -c "
   import coremltools as ct
   m = ct.models.MLModel('~/Downloads/GenitalClassifier.mlmodel'.replace('~', '$HOME'))
   spec = m.get_spec()
   classes = list(spec.neuralNetworkClassifier.stringClassLabels.vector
                  or spec.pipelineClassifier.pipeline.models[-1].neuralNetworkClassifier.stringClassLabels.vector
                  or [])
   print('classes:', classes)
   print('inputs:',  [i.name for i in spec.description.input])
   print('outputs:', [o.name for o in spec.description.output])
   "
   ```
   Expected output:
   - `classes`: the five class strings exactly matching the
     `GenitalSubClass` enum raw values
     (`MALE_GENITALIA_COVERED` / `_FLACCID` / `_AROUSAL` /
     `_ORGASM` / `OTHER`).
   - `inputs`: one image input (CreateML names it `image`
     typically).
   - `outputs`: `classLabel` (String) + `classLabelProbs`
     (Dictionary).

If any class string is misspelled relative to
`GenitalSubClass.rawValue`, the on-device wrapper's
`GenitalSubClass(rawLabel:)` returns nil and the override silently
fails. Fix the folder name in the dataset and retrain — don't try
to rename in the .mlmodel after the fact.

## 7. If the model needs another pass

Common failures and fixes:

| Symptom | Fix |
|---|---|
| OTHER recall < 80% (false positives leaking through) | Add 200+ more OTHER crops from `unsorted/_BTE/`, `_BTC/`, `_ANE/`, and any obvious-junk thumbnails skipped during review. Retrain. |
| FLACCID vs AROUSAL confusion > 20% | Open `MALE_GENITALIA_AROUSAL/` in icon view; some borderline crops likely belong in FLACCID. Move them. Retrain. |
| ORGASM recall < 50% | Either (a) add more ORGASM examples, or (b) drop the class entirely and treat orgasm as AROUSAL on-device. Note: dropping requires changing `GenitalSubClass` enum + `aggregate()` in `NudityDetector.swift`. |
| Training accuracy 99%, validation 75% | Overfitting. Reduce iterations to ~25, or expand the dataset. |
| Validation accuracy 95%+ on first run | Suspicious. Make sure the validation set isn't accidentally photos from the same source as training — CreateML's Auto split is per-image, not per-source-photo, so two crops from the same source photo can land on opposite sides of the split. Usually fine but worth a sanity check. |

## 8. Compile to .mlmodelc and upload to GitHub

Produces the artifact `ModelArchive.genitalClassifier.sourceURL`
expects:

```bash
# 1. Compile to .mlmodelc
xcrun coremlcompiler compile \
    ~/Downloads/GenitalClassifier.mlmodel \
    /tmp/

# 2. Rename to match ModelArchive.genitalClassifier.directoryName
mv /tmp/GenitalClassifier.mlmodelc /tmp/GenitalClassifier-v1.mlmodelc

# 3. Ditto-zip preserving directory structure (NOT a regular zip;
# `ditto -c -k` produces an archive Apple's CIRAWFilter / CoreML
# can decompress directly).
ditto -c -k --sequesterRsrc --keepParent \
    /tmp/GenitalClassifier-v1.mlmodelc \
    /tmp/GenitalClassifier.mlmodelc.zip

# 4. Strip macOS metadata if present.
zip -d /tmp/GenitalClassifier.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true

# 5. Create the GitHub release + upload.
gh release create genital-classifier-v1 \
    --title "Genital sub-class classifier v1" \
    --notes "CreateML ImageFeaturePrint, $(stat -f '%z' /tmp/GenitalClassifier-v1.mlmodelc | numfmt --to=iec) compiled, 5 classes." \
    /tmp/GenitalClassifier.mlmodelc.zip
```

## 9. Install on-device

1. Run the app.
2. Scroll the install rows in OverlayControls; tap **Download**
   on the "Genital sub-class classifier not installed." row.
3. The download bar fills, then the row disappears (= installed).
4. Load any image with NudeNet detections. The label overlay
   (Labels toggle in OverlayControls) should now show the new
   sub-class strings instead of `MALE_GENITALIA_EXPOSED` /
   `FEMALE_GENITALIA_*`.
5. Per-subject NudityLevel reflects the override: a subject whose
   only genital detection got reclassified to MGC will now show
   `.covered` instead of `.nude`; FLACCID / AROUSAL / ORGASM stay
   `.nude`.

If the labels don't change, the most likely failure mode is the
model emitting class strings that don't match
`GenitalSubClass.rawValue` — check the Console for
`[GenitalClassifier] loaded …` logs and compare the model's
`classLabel` output against the five enum cases.

## 10. Iterating

Each retrain cycle:

- Refine the dataset (move crops, add more from later library
  extractions).
- Retrain in CreateML — same project, click **Train** again.
  CreateML keeps every snapshot; compare confusion matrices.
- Re-export, recompile, push as `genital-classifier-v2`. **Bump
  both halves of `ModelArchive.genitalClassifier`**:
  `directoryName` to `GenitalClassifier-v2.mlmodelc` and the URL
  to the new tag, so a previously-installed v1 on someone's
  device gets re-downloaded as v2 instead of resolving to a
  stale install.

That's the full loop. If anything in the pipeline breaks
(confusion matrix unacceptable, model load error, label
mismatch), the most common fix is "go back and fix the dataset,
then retrain" — the rest of the loop is mechanical.
