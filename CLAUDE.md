# Focus Check — CLAUDE.md

## Project Overview

A Swift Playgrounds app (`.swiftpm`) that imports a photo and renders
focus-analysis overlays on top of it. Beyond the original sharpness-peaking
idea, the app has grown into a general image-inspection tool: classical
sharpness + ML depth + motion-blur detection, on-device sensitive-content
classification, silhouette-aware mosaic redaction, EXIF badges, 2× zoom,
press-and-hold compare, and PNG export via the share sheet.

Target platforms: **iPadOS 18.1+** and **macOS 14+** (single codebase, UIKit
everywhere — the product type is `.iOSApplication`, so Mac runs via
*Designed for iPad*).

Distribution paths (sibling projects):
- `FocusApp.swiftpm/` — Swift Playgrounds on iPad / Mac. Unsigned, local.
- `Xcode/` — XcodeGen manifest → `.xcodeproj`. Same sources; signed builds
  for TestFlight / App Store + the `SensitiveContentAnalysis` capability.

---

## Architecture

### Package format
- Swift Package Manager project (`.swiftpm`)
- Product type: `.iOSApplication` (UIKit view system, iPad + Mac)
- Swift language mode: 5.9 (Playgrounds default)
- Round-trips with Xcode 16+ via the sibling XcodeGen project for
  debugging, Metal Frame Capture, and Instruments

### Dependency
- [`ZIPFoundation`](https://github.com/weichsel/ZIPFoundation) —
  pure-Swift unzip, used by `ModelArchiveInstaller` to unpack Core ML
  model archives downloaded at runtime.

### Capabilities (declared in `Package.swift`)
- `.photoLibrary` — select originals from Photos via `PHPickerViewController`
- `.fileAccess(.userSelectedFiles, .readOnly)` — `.fileImporter` path
- `.outgoingNetworkConnections` — required on macOS sandbox for the
  runtime model downloads

### Layer structure

```
FocusApp.swiftpm/
├── Package.swift
└── Sources/App/
    ├── FocusApp.swift                     # @main / WindowGroup
    ├── UI/
    │   ├── ContentView.swift              # NavigationStack, toolbar, badges
    │   ├── MetalView.swift                # UIViewRepresentable around MTKView
    │   ├── FocusViewModel.swift           # @MainActor ObservableObject
    │   ├── OverlayControls.swift          # style picker, slider, mosaic row
    │   ├── ImageImporter.swift            # PHPicker + fileImporter menu
    │   └── ExposureInfo.swift             # EXIF reader + formatters
    ├── Analysis/
    │   ├── FocusAnalyzer.swift            # actor — the GPU/ML workhorse
    │   ├── LaplacianVariance.swift        # MPS sharpness pipeline + AnalysisError
    │   ├── DepthEstimator.swift           # Depth Anything v2 wrapper
    │   ├── MotionBlurDetector.swift       # vDSP 2D FFT (global + tiled)
    │   ├── SensitiveContentChecker.swift  # SCA primary + NSFW fallback
    │   ├── ModelArchiveInstaller.swift    # shared downloader actor
    │   └── CIImage+Stretch.swift          # translate-to-origin + stretch helpers
    └── Rendering/
        └── FocusRenderer.swift            # Core Image compositor + MTKView driver
```

---

## Core technical stack

### Sharpness detection (classical path)
- Pipeline: `MPSImageGaussianBlur` (σ ≈ 1) → `MPSImageLaplacian` →
  `MPSImageMedian` (3×3, drops shadow-noise false positives)
- Compute at **1024 px long-side** analysis resolution (`LaplacianVariance.analysisLongSide`);
  upscale to source extent with `CILanczosScaleTransform`
- Output: single-channel `r16Float` `MTLTexture`, cached across slider scrub

### Depth estimation (ML path, optional)
- Model: `apple/coreml-depth-anything-v2-small` (F16, ~50 MB, ANE-friendly)
- **Runtime download**, not bundled. `ModelArchiveInstaller(.depthAnything)`
  fetches a zipped `.mlmodelc` from a GitHub release and installs it into
  `Application Support/`. The 50 MB directory-format artifact can't sit in
  the git repo and can't be compiled on iPad.
- Loaded via generic `MLModel` + `MLDictionaryFeatureProvider` (no
  auto-generated Swift class — SPM can't process `.mlpackage`)
- Input size is **queried from the model's image constraint** at load
  time (pick the largest enumerated size); don't hardcode 518×518
- Stretched (non-uniform) to model input — preserves source↔map
  spatial correspondence for the post-inference upscale

### Motion-blur detection
- Frequency-domain: 2D Accelerate FFT; motion PSF produces a directional
  dark band in the spectrum, defocus produces rings
- **Global reading** for the info badge: center-cropped 256² patch,
  ~10 ms on A14
- **Tiled confidence map** for the `.motion` overlay: 512² analysis
  image, 128² patches, 64-stride → 7×7 grid, ~100 ms total
- Badge floor: `confidence > 0.4` (`MotionBlurReport.isSignificant`) —
  keeps fences / waves / stripes from consistently triggering it; the
  overlay shows sub-threshold readings too so the user can inspect them

### Sensitive-content classification
Two backends stacked in `SensitiveContentChecker`:

1. **Primary: Apple's `SensitiveContentAnalysis` framework** — requires
   the SCA capability + user-enabled Communication Safety (iOS) or
   per-app Sensitive Content Warning (macOS). In Playgrounds /
   unsigned builds this reliably returns `.disabled`.
2. **Fallback: runtime-downloaded Core ML model** — lovoo/NSFWDetector
   CreateML-trained binary SFW/NSFW classifier, installed via
   `ModelArchiveInstaller(.nsfw)`.

The availability enum surfaces both paths plus "neither" so the UI can
show a targeted install prompt rather than silently hiding the feature.

### Per-subject nudity rating (NudeNet)
Third tier, additive to whichever primary classifier is active:
`NudityDetector` runs a downloaded NudeNet Core ML model (YOLO-style
object detector, Create ML-exported) over the whole image. Each
detection's bounding box is attributed to one of the Vision body
rectangles (≥ 50 % area overlap) and the bag of per-body labels maps
to a four-level rating — `.none` / `.covered` / `.partial` / `.nude`.

The renderer reads `nudityLevels` + `nudityGate` off the composite
inputs and skips any body whose level falls below the gate, so clothed
subjects in a mixed scene aren't mosaiced alongside the flagged ones.
Gate is user-controllable once the model is installed.

### Context scoring (CLIP, optional)
Fourth tier, complements the anatomy-centric detectors with
scene/context understanding: `CLIPScorer` runs a downloaded CLIP
image encoder once per analyze, then computes cosine similarity
against a set of pre-embedded text prompts shipped alongside the
model. Top match is surfaced as a "Context" badge; the scorer runs
inside the non-mode-dependent cache so mode switches don't re-run it.

- Model: OpenAI CLIP ViT-B/32 (Core ML, image-encoder only; ~150 MB F16)
- Archive layout: `CLIP.zip` containing both `CLIPImageEncoder.mlmodelc`
  and `clip-prompts.json` (prompt + matching text-encoder embedding).
  `ModelArchive.clip` uses `kind: .bundle` so the installer moves the
  whole tree instead of just the `.mlmodelc`.
- Prompts are whatever the maintainer chose at export time — typical
  set mixes nudity / scene / safe-art / medical / minor-presence
  anchors so the top match functions as a coarse context label.

### Whole-image quality judges (NIMA, optional)
Seventh tier. Unlike the per-face tiers above, the two NIMA
analyzers run once per image and each produce a single scalar
score in [1, 10]. Two independent axes:

- `QualityAnalyzer` (technical) — TID2013-trained. Captures
  sharpness / exposure / compression / noise / banding — the
  "did the camera capture this correctly?" axis.
- `AestheticAnalyzer` (aesthetic) — AVA-trained. Captures
  composition / subject interest / lighting mood — the "is this
  a good-looking photograph?" axis.

Both share `QualityScore` and the same architecture: MobileNet-v1
backbone (ImageNet pre-train → fine-tuned on the task) →
GlobalAveragePool → Dense(10, softmax). Swift-side reduction:

- score = `Σ (i+1) · p_i` over i ∈ [0, 9], clamped to [1, 10]
- stdev = `sqrt(Σ (i+1 − μ)² · p_i)` — distribution spread; low
  stdev = confident, high = mixed / uncertain

They're loaded by a single `NIMAModel` class parameterized by
`ModelArchive`; two shared statics (`technicalShared`,
`aestheticShared`) mean the inference code lives in one place.

Surfaced as two sibling capsules on the top badge row:
- `✓ Quality 6.4 ±0.8` (technical, checkmark-seal icon)
- `✨ Aesthetic 5.7 ±1.1` (aesthetic, sparkles icon)

Both use the same colour scale: red below 4, orange 4–6, green
above 6.

- Models: idealo/image-quality-assessment
  - `weights_mobilenet_technical_0.11.hdf5` (EMD 0.107,
    SROCC 0.675 on TID2013)
  - `weights_mobilenet_aesthetic_0.07.hdf5` (AVA-trained)
- Input: 224² RGB in [0, 255]. Core ML ImageType declares
  `scale = 2/255, bias = [-1, -1, -1]` so the network receives
  [-1, 1] MobileNet-normalized values — preprocessing is NOT
  baked into the graph, unlike EfficientNet-based tiers.
- License: Apache-2.0 code. TID2013 is "scientific and
  educational research"; AVA is research-license. Signed App
  Store build distribution needs a lawyer check for both.
- Known caveat: NIMA scores are ordinal-reliable (photo A > B
  most of the time) but not calibrated absolute — treat a 7.2
  vs 7.5 delta as "similar", a 3.0 vs 7.0 gap as real. The ±
  band next to each score nudges the user toward that reading.

### Per-face age estimation (SSR-Net, optional)
Sixth tier, complementary to the other per-face tiers: `AgeEstimator`
runs a downloaded SSR-Net model per face and emits a single scalar
age in [0, 100]. SSR-Net is age-only — gender comes from NudeNet's
FACE_* branch (`nudityGenders`) exclusively.

Architecturally this model is a stack of 3-stage MobileNet-like
blocks followed by a custom "Soft Stagewise Regression" combine
step. Upstream's coremltools converter couldn't inline the combine
math, so it's implemented as an `MLCustomLayer` (`SSRModule` at
the bottom of `AgeEstimator.swift`). Core ML loads the `.mlmodelc`
and resolves the custom class by `@objc` name — no registry call
needed on the Swift side.

- Model: shamangary/SSR-Net stage_num [3, 3, 3], 0.32 MB uncompiled
- Input: 64² BGR in [0, 255] (no baked-in preprocessing — feed raw
  pixel bytes). The Core ML spec's `colorSpace` was flipped from
  the upstream RGB default to BGR because SSR-Net's training
  pipeline reads with `cv2.imread` (BGR) and never swaps.
- Output: single scalar age. No distribution → no uncertainty band
- License: Apache-2.0 code, trained on MORPH2 / IMDB / WIKI —
  research-only training data. Treat the same as EmoNet /
  OpenGraphAU; do not ship in signed App Store builds.
- Replaces the earlier yu4u/age-gender-estimation v2 model which
  had a strong "adult female" bias on real-device inputs.

### Per-face pain detection (OpenGraphAU + PSPI, optional)
Fifth tier, complementary to the emotion classifier: `PainDetector`
runs a downloaded OpenGraphAU model per face to get multi-label
Action Unit probabilities, then computes a Prkachin-Solomon Pain
Intensity (PSPI) proxy Swift-side:

    PSPI ≈ AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

AUs 4 / 6 / 7 / 9 / 10 come straight from OpenGraphAU's 41-dim
sigmoid output. AU43 (eye closure) isn't one of OpenGraphAU's
classes, so we derive it from Vision's face-landmarks eye-aspect
ratio (bbox h/w of the 6-point eye cluster) and fold it into the
sum. The final value lives in [0, 4] (all four terms are [0, 1]
probabilities rather than FACS 0-5 intensities), binned into
none / mild / moderate / severe for a per-subject bar.

- Model: OpenGraphAU Stage-2 ResNet-50 (~100 MB F32), 224² RGB input
- License: Apache-2.0 code + BP4D/DISFA-trained weights (research-only
  in practice — treat as non-shippable in signed App Store builds)
- This is a **proxy** for PSPI, not the clinical metric. The
  detector wasn't trained on the UNBC-McMaster Pain dataset and the
  output ranges are probabilities, so presented levels are ordinal
  not medical-grade.

### Vision-derived regions
Run in one consolidated `VNImageRequestHandler.perform([...])` pass so the
CGImage decode and internal pyramid are shared:

- `VNDetectFaceLandmarksRequest` — face rects + eye-bar geometry
- `VNDetectHumanRectanglesRequest` — body bounding boxes
- `VNDetectHumanBodyPoseRequest` — 2D joints (derive groin + chest rects)
- `VNDetectHumanBodyPose3DRequest` — 3D hips for sideways-pose widening
- `VNGeneratePersonSegmentationRequest` (`.balanced`) — silhouette mask
  for the `.body` mosaic mode

### Overlay rendering
- `MTKView` via `UIViewRepresentable` (UIKit-only — product type is
  `.iOSApplication`; no `NSViewRepresentable` branch exists or is correct)
- `framebufferOnly = false`, `autoResizeDrawable = true`, paused off
- `CIContext(mtlDevice:)` shared across analyzer + renderer; never
  rasterize to `CGImage` mid-pipeline except at export and Vision decode
- Working color space: **extended linear Display P3**,
  working format: `RGBAh`, `cacheIntermediates = false`
- Drawable: `rgba16Float` on the `CAMetalLayer`,
  `wantsExtendedDynamicRangeContent = true` for EDR

### Overlay styles (`OverlayStyle`)
| Style | What it shows |
|---|---|
| `None` | Original image — disables the threshold slider + analysis mode |
| `Peaking` | Colored edges gated by the focus mask (FastRawViewer-style) |
| `Mask` | Flat 50%-opacity tint over in-focus pixels |
| `Heatmap` | Viridis LUT via `CIColorCube` (never jet) |
| `Error` | Red = too close / blue = too far, relative to focal plane; requires Hybrid mode |
| `Motion` | Tint where the per-patch motion-blur confidence exceeds the threshold |

### Mosaic modes (`MosaicMode`)
| Mode | Coverage |
|---|---|
| `Tabloid` | Eyes (black bar) + Groin (pixelate) |
| `Eyes` | Rotated black bar following head tilt |
| `Face` | Pixelate face landmark rect |
| `Chest` | Pixelate upper torso rect from pose joints |
| `Groin` | Pixelate pelvis rect; widens up to 3× for sideways bodies |
| `Body` | Pixelate along the person silhouette, snapped to the pixelation grid so the edge reads block-jagged |
| `Whole` | Pixelate the entire image |

Triggers: classifier flagged **and** the Mosaic toggle is on, **or**
`Force Censor` is on (bypasses the classifier). Both paths are honored
by the live renderer and the PNG export.

---

## Key implementation rules

### Threading
- `FocusAnalyzer` is an actor; MPS kernels, `CIContext`, `MLModel`
  calls all stay on it
- Analysis runs on `Task.detached(priority: .userInitiated)`; cancel
  the prior task on each new image load / mode change
- Slider scrub must **not** re-analyze — the renderer reads
  `@Published var threshold: Float` directly each frame
- Live MTKView resize (Stage Manager): only notify the renderer when
  drawable size delta > 1 px

### Memory
- Treat everything as `CIImage` (lazy recipe graph); never materialize
  `CGImage` mid-pipeline
- 50 MP RAW → fp16 ≈ 400 MB; needs the
  `com.apple.developer.kernel.increased-memory-limit` entitlement on
  8 GB iPads (signed Xcode builds only — Playgrounds can't declare it)

### Core ML in SwiftPM
- SwiftPM **cannot compile** `.mlpackage` / `.mlmodel` — artifacts must
  already be `.mlmodelc` when the app sees them. We solve this by
  downloading zipped `.mlmodelc` directories at runtime; nothing Core-ML
  needs to live in the repo.
- No auto-generated Swift class — load with `MLModel(contentsOf:)`,
  feed `MLDictionaryFeatureProvider`, iterate `featureNames` at output

### Metal shaders
- No custom `.metal` files — SwiftPM doesn't compile them. Rely on
  MPS kernels + Core Image built-in filters throughout.

### Color management
- RAW via `CIRAWFilter(imageURL:)` — don't bypass it
- `CIImage(contentsOf:)` with `.applyOrientationProperty: true` so EXIF
  orientation is normalized before any transform runs
- Working color space `extendedLinearDisplayP3`, working format `RGBAh`
- PNG export: render in linear P3, then `createCGImage` to `sRGB` for the
  file — downstream viewers overwhelmingly expect sRGB PNGs

---

## Platform notes

| Concern | iPad (`.iOSApplication`) | Mac (Designed for iPad) |
|---|---|---|
| View bridge | `UIViewRepresentable` | Same (UIKit via Catalyst-style) |
| File import | PHPicker + `.fileImporter` | `.fileImporter` (PHPicker also works) |
| SCA primary backend | Enabled via Screen Time → Comm Safety | Enabled via Settings → Sensitive Content Warning (per-app; requires capability + signing) |
| Pencil Pro | `.onPencilSqueeze` cycles Peaking ↔ Heatmap | N/A |
| Network egress | Default-allow | Needs `.outgoingNetworkConnections()` capability + `com.apple.security.network.client` in entitlements |

---

## File input

**Primary (Photos):** `PHPickerViewController` wrapped via
`UIViewControllerRepresentable` in `ImageImporter.swift`. Runs
out-of-process — no Photos permission dialog. `suggestedName` carries
the original camera filename, which SwiftUI's `PhotosPicker` doesn't
surface. `preferredAssetRepresentationMode = .current` preserves
ProRAW / RAW / HEIC originals.

**UTI fallback order** (empty / unexpected UTIs): `public.camera-raw-image`
→ `com.adobe.raw-image` → `public.heic` → `public.image`.

**Secondary (Files / disk):** `.fileImporter(allowedContentTypes: [.image, .rawImage])`.
Security-scoped URLs — call `startAccessingSecurityScopedResource()`
but read via `Data(contentsOf:)` regardless (`copyItem` is flakier
inside the macOS sandbox).

**Drag and drop:** `.dropDestination(for: URL.self)` on the empty-state
placeholder view.

---

## Export

`FocusViewModel.exportPNG()` snapshots the current state into a
`FocusCompositeInputs`, feeds it through the same `FocusRenderer.composite`
used by the live MTKView (at source resolution, zoom = 1), encodes sRGB
PNG via `CGImageDestination`, and writes to `FileManager.temporaryDirectory`.
`ContentView` presents the URL through a `UIActivityViewController`
(`ShareSheet`) so the user can save to Files, Photos, or any registered
share target.

Force Censor participates in the export: the mosaic condition is
`forceCensor || ((isSensitive == true) && mosaicEnabled)` in both the
live renderer and the export path, so the exported image always matches
what's on screen.

---

## Runtime model installs

Both installs use the same code path — `ModelArchiveInstaller` +
`ModelArchive` descriptor:

```swift
struct ModelArchive {
    let directoryName: String   // e.g. "DepthAnythingV2SmallF16.mlmodelc"
    let sourceURL: URL          // GitHub release asset (.zip)
    func installedURL() throws -> URL
    func isInstalled() -> Bool
}
```

- `ModelArchive.depthAnything` → `depth-model-v1` release tag
- `ModelArchive.nsfw` → `nsfw-model-v1` release tag
- `ModelArchive.nudenet` → `nudenet-model-v1` release tag (per-subject detector)
- `ModelArchive.clip` → `clip-model-v1` release tag (context scorer, bundle)
- `ModelArchive.emotion` → `emotion-model-v5` release tag (EmoNet per-face classifier, **research-only license** — do not ship in signed builds)
- `ModelArchive.openGraphAU` → `pain-model-v1` release tag (OpenGraphAU AU detector for PSPI pain proxy, **research-only license** — do not ship in signed builds)
- `ModelArchive.age` → `age-model-v3` release tag (SSR-Net — per-face age only, **research-only training data** — do not ship in signed builds; gender falls back to NudeNet's FACE_* branch)
- `ModelArchive.quality` → `quality-model-v1` release tag (idealo/NIMA MobileNet — whole-image technical-quality scalar, Apache-2.0 + TID2013 research-license — shipping in signed builds plausible but not vetted)
- `ModelArchive.aesthetic` → `aesthetic-model-v1` release tag (idealo/NIMA MobileNet aesthetic variant — whole-image composition / mood scalar, Apache-2.0 + AVA research-license)

Maintainer workflow (one-time, requires a Mac):

```bash
# Depth Anything v2 Small F16
xcrun coremlcompiler compile DepthAnythingV2SmallF16.mlpackage /tmp/
ditto -c -k --sequesterRsrc --keepParent \
      /tmp/DepthAnythingV2SmallF16.mlmodelc \
      DepthAnythingV2SmallF16.mlmodelc.zip
gh release upload depth-model-v1 DepthAnythingV2SmallF16.mlmodelc.zip

# NSFW fallback (lovoo)
curl -L -o NSFW.mlmodel \
  https://github.com/lovoo/NSFWDetector/releases/download/1.1.0/NSFW.mlmodel
xcrun coremlcompiler compile NSFW.mlmodel /tmp/
ditto -c -k --sequesterRsrc --keepParent /tmp/NSFW.mlmodelc NSFW.mlmodelc.zip
gh release upload nsfw-model-v1 NSFW.mlmodelc.zip

# CLIP context scorer — two-stage: export the image encoder + run the
# text encoder once to produce prompt embeddings, then ZIP both into
# the same archive.
python3 export_clip_image_encoder.py    # -> CLIPImageEncoder.mlpackage
python3 export_clip_prompt_embeddings.py \
        --prompts prompts.txt \
        --out     clip-prompts.json     # pre-normalized [{prompt, embedding}]
xcrun coremlcompiler compile CLIPImageEncoder.mlpackage /tmp/
mkdir -p /tmp/CLIP
mv /tmp/CLIPImageEncoder.mlmodelc /tmp/CLIP/
cp clip-prompts.json                    /tmp/CLIP/
ditto -c -k --sequesterRsrc --keepParent /tmp/CLIP CLIP.zip
gh release upload clip-model-v1 CLIP.zip

# OpenGraphAU Stage-2 ResNet-50 (pain / PSPI proxy). See
# Tools/export_opengraphau_model.py for the one-time clone +
# weight-download prereqs.
python3 Tools/export_opengraphau_model.py
xcrun coremlcompiler compile OpenGraphAU.mlpackage /tmp/
ditto -c -k --sequesterRsrc --keepParent \
      /tmp/OpenGraphAU.mlmodelc OpenGraphAU.mlmodelc.zip
zip -d OpenGraphAU.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
gh release upload pain-model-v1 OpenGraphAU.mlmodelc.zip

# SSR-Net (per-face age). Uses the pre-compiled mlmodel from the
# upstream conversion example; no Keras re-conversion needed.
curl -L -O https://github.com/shamangary/Keras-to-coreml-multiple-inputs-example/raw/master/ssrnet.mlmodel
# Flip the declared colorSpace from RGB (upstream default) to BGR
# so Core ML hands SSR-Net's BGR-trained conv layers the right bytes:
python3 -c 'import coremltools as ct; m=ct.models.MLModel("ssrnet.mlmodel"); s=m.get_spec(); [setattr(i.type.imageType,"colorSpace",30) for i in s.description.input if i.type.imageType.colorSpace==20]; open("ssrnet-bgr.mlmodel","wb").write(s.SerializeToString())'
xcrun coremlcompiler compile ssrnet-bgr.mlmodel /tmp/
mv /tmp/ssrnet-bgr.mlmodelc /tmp/SSRNet-v1.mlmodelc
ditto -c -k --sequesterRsrc --keepParent \
      /tmp/SSRNet-v1.mlmodelc SSRNet.mlmodelc.zip
zip -d SSRNet.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
gh release upload age-model-v3 SSRNet.mlmodelc.zip

# NIMA technical-quality (whole-image scalar). Fresh Keras →
# MLProgram via Tools/export_nima_model.py.
curl -L -O https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_technical_0.11.hdf5
python3 Tools/export_nima_model.py
xcrun coremlcompiler compile NIMA.mlpackage /tmp/
ditto -c -k --sequesterRsrc --keepParent \
      /tmp/NIMA.mlmodelc NIMA.mlmodelc.zip
zip -d NIMA.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
gh release upload quality-model-v1 NIMA.mlmodelc.zip

# NIMA aesthetic — same export script, different weights via env.
curl -L -O https://github.com/idealo/image-quality-assessment/raw/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5
NIMA_WEIGHTS=weights_mobilenet_aesthetic_0.07.hdf5 python3 Tools/export_nima_model.py
xcrun coremlcompiler compile NIMA.mlpackage /tmp/
mv /tmp/NIMA.mlmodelc /tmp/NIMA-Aesthetic.mlmodelc
ditto -c -k --sequesterRsrc --keepParent \
      /tmp/NIMA-Aesthetic.mlmodelc NIMA-Aesthetic.mlmodelc.zip
zip -d NIMA-Aesthetic.mlmodelc.zip '__MACOSX/*' 2>/dev/null || true
gh release upload aesthetic-model-v1 NIMA-Aesthetic.mlmodelc.zip
```

`OverlayControls` renders a dedicated install/progress/retry row for
each, gated by the analyzer's availability flags.

---

## Known limitations & non-goals

- No custom `.metal` authoring in Playgrounds — MPS / Core Image only
- No debugger breakpoints in Playgrounds — open `Xcode/FocusApp.xcodeproj`
  for Metal Frame Capture, Instruments, and source-level debugging
- **Motion blur** from the FFT path isn't perfectly distinguishable from
  strong directional texture; the 0.4 confidence floor tamps most false
  positives but not all
- **Smooth in-focus regions** (skin, sky) underread in the Laplacian
  path; Hybrid with Depth Anything recovers them
- Vision has **no spatial focus-map API** — all focus detection is custom
- `CIRAWFilter` supports Apple's [published camera list](https://support.apple.com/en-us/122870)
  only; cameras outside it would need LibRaw
- **SCA in Playgrounds** will always be `.disabled` — the NSFW fallback
  download is the only way sensitive-content flagging works in
  unsigned builds
- **Demographic inference is present but research-only.** The
  `.ageGender` tier (yu4u EfficientNetB3) covers age + gender; NudeNet's
  FACE_* branch provides a coarser gender fallback. Both backends are
  trained on datasets with "research only" terms — the compiled models
  are fine for Playgrounds / local dev but must not be bundled in
  signed App Store builds. No ethnicity classifier is planned; Vision
  doesn't expose one and we aren't shipping a third-party one
- **NudeNet detector format assumptions** — `NudityDetector` expects
  Create ML's object-detector output (`coordinates` Nx4 + `confidence` NxC).
  If a maintainer converts NudeNet with a different export shape (raw
  YOLO tensors, concatenated format), extend `parseDetections(from:)`

---

## References

- Apple Accelerate sample: [Finding the sharpest image in a sequence](https://developer.apple.com/documentation/accelerate/finding-the-sharpest-image-in-a-sequence-of-captured-images)
- Depth Anything v2 Core ML release: [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small)
- NSFW fallback: [lovoo/NSFWDetector](https://github.com/lovoo/NSFWDetector)
- Metal in Swift Playgrounds 4: [Better Programming](https://betterprogramming.pub/using-metal-in-swift-playgrounds-4-e100122d276a)
- Core ML in `.swiftpm`: [Apple Developer Forums thread 771598](https://developer.apple.com/forums/thread/771598)
- EDR rendering: WWDC22 session 10114 — *Display EDR content with Core Image, Metal, and SwiftUI*
- Person segmentation: `VNGeneratePersonSegmentationRequest` (iOS 15+)
- Classical focus operators benchmark: Pertuz, Puig, García — *Pattern Recognition* 46(5), 2013
