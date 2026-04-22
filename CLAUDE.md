# Focus Detection App — CLAUDE.md

## Project Overview

A Swift Playgrounds app (`.swiftpm`) that takes an image as input, detects in-focus vs out-of-focus regions using classical sharpness analysis and optional ML-based depth estimation, then renders a focus-highlight overlay for the user.

Target platforms: **iPadOS 18.1+ and macOS 14+** (single codebase, ~80% shared).
Distribution: App Store / TestFlight (not Swift Student Challenge — model exceeds 25 MB cap).

---

## Architecture

### Package Format
- Swift Package Manager project (`.swiftpm`)
- Product type: `iOSApplication` (also runs on Mac via Catalyst)
- Swift language mode: 5.9 (Playgrounds default); upgrade to Swift 6 with care around Sendable actors
- Round-trips cleanly with Xcode 16+ for debugging and profiling

### Layer Structure

```
FocusApp.swiftpm/
├── Package.swift
└── Sources/
    └── App/
        ├── App entry (@main, WindowGroup)
        ├── UI/
        │   ├── ContentView.swift          # root SwiftUI layout
        │   ├── MetalView.swift            # UIViewRepresentable wrapping MTKView
        │   ├── OverlayControls.swift      # threshold slider, color well, style toggle
        │   └── ImageImporter.swift        # PhotosPicker + fileImporter glue
        ├── Analysis/
        │   ├── FocusAnalyzer.swift        # actor — owns MTLDevice, CIContext, MPS kernels
        │   ├── LaplacianVariance.swift    # MPS sharpness pipeline
        │   └── DepthEstimator.swift       # Core ML Depth Anything v2 wrapper
        ├── Rendering/
        │   └── FocusRenderer.swift        # overlay compositing via Core Image + MTKView
        └── Resources/
            └── DepthAnythingV2SmallF16.mlmodelc/   # pre-compiled model (see setup)
```

---

## Core Technical Stack

### Sharpness Detection (classical path)
- **Pipeline**: `MPSImageGaussianBlur` (σ≈1) → `MPSImageLaplacian` → `MPSImageStatisticsMeanAndVariance`
- Compute at **1024 px long-side** (analysis resolution); upscale mask with `CILanczosScaleTransform`
- Cache sharpness map as `MTLTexture` (R16F); re-run only on new image load, not on slider scrub
- Target: **2–5 ms per frame** on Apple Silicon iPad/Mac

### Depth Estimation (ML path, optional)
- **Model**: `apple/coreml-depth-anything-v2-small` F16 variant (49.8 MB, ~30 ms on ANE)
- Load via `MLModel(contentsOf: Bundle.module.url(forResource:withExtension:)!)`
- Run with `MLComputeUnits.all` (targets Neural Engine)
- Output: relative monocular depth map → use as focal-plane prior
- **Hybrid mode**: Laplacian mask AND depth mask intersection (best quality)

### Overlay Rendering
- `MTKView` embedded via `UIViewRepresentable` (iOS) / `NSViewRepresentable` (macOS)
- `MTKView.framebufferOnly = false`, `autoResizeDrawable = true`
- `CIContext(mtlDevice:)` for all compositing — never rasterize to `CGImage` mid-pipeline
- Colorspace: **extended linear Display P3** (`extendedLinearDisplayP3`) throughout
- Pixel format: `rgba16Float` on the backing `CAMetalLayer`; `wantsExtendedDynamicRangeContent = true` for EDR

### Overlay Styles
1. **Focus peaking** — colored edges at high-gradient pixels (default; matches FastRawViewer)
2. **Heatmap** — viridis LUT via `CIColorCube` (never jet — perceptually non-uniform)
3. **Binary mask** — threshold + tint at 50% opacity
4. **Split view** — `DragGesture` divider, original vs overlay

---

## Key Implementation Rules

### Threading
- `FocusAnalyzer` is a Swift `actor` — all MPS/CIContext/MLModel calls stay on this actor
- `Task.detached(priority: .userInitiated)` for image analysis; cancel prior task on new load
- Slider scrub must NOT re-run analysis — update only a `@Published var threshold: Float`
- Stage Manager live-resize: reallocate `MTLTexture` only when drawable size changes by >1 px

### Memory
- Keep everything as `CIImage` (lazy recipe graph) — never `CGImage` intermediates
- 50 MP RAW decoded to fp16 = ~400 MB; declare `com.apple.developer.kernel.increased-memory-limit` entitlement
- On 4 GB iPads (A14 iPad 10th-gen): tile images through Core Image ROI
- `CIContext.cacheIntermediates = false` for large images

### Core ML in SwiftPM (critical gotcha)
- SPM **cannot** compile `.mlpackage` — must pre-compile externally (see Setup)
- Resource rule in `Package.swift` **must be `.copy()`**, never `.process()`:
  ```swift
  .copy("DepthAnythingV2SmallF16.mlmodelc")
  ```
- No auto-generated Swift class — use generic `MLModel` + `MLDictionaryFeatureProvider`

### Metal Shaders
- **No custom `.metal` files** in Playgrounds — SwiftPM doesn't compile them
- Use MPS kernels exclusively for sharpness (no shader authoring needed)
- If a custom CIKernel is ever needed: pre-compile a `.metallib` in Xcode and ship as `.copy()` resource, or use string-based `makeLibrary(source:)` at runtime

### Color Management
- Decode RAW with `CIRAWFilter(imageURL:)` — never bypass for manual decode
- Set `CIContext.workingColorSpace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!`
- Set `CIContext.workingFormat = CIFormat.RGBAh`
- Apply 3×3 median filter to sharpness map before thresholding (eliminates shadow noise / false positives)

---

## Platform Differences

| Concern | iPadOS | macOS |
|---|---|---|
| MTKView bridge | `UIViewRepresentable` | `NSViewRepresentable` |
| EDR headroom query | `UIScreen.currentEDRHeadroom` | `NSScreen.maximumExtendedDynamicRangeColorComponentValue` |
| File import | `PHPickerViewController` / `.fileImporter` | `.fileImporter` / `NSOpenPanel` (via SwiftUI) |
| LiDAR depth (iPad Pro) | Available via `ARKit` — use as depth prior when present | Not available |
| Apple Pencil Pro | `.onPencilSqueeze` for threshold toggle | Not applicable |
| Thermal throttling | Throttles after ~15 min sustained GPU load | Active cooling (MacBook Pro), similar on MacBook Air |

Use `#if os(iOS)` / `#if canImport(AppKit)` for platform-specific branches. Wrap both bridging types behind a `PlatformViewRepresentable` typealias.

---

## File Input

**Primary (photo library):** SwiftUI `PhotosPicker` with `Transferable` or `PHPickerViewController`.
- Set `preferredAssetRepresentationMode = .current` for ProRAW
- Load via `loadFileRepresentation(forTypeIdentifier:)` — not `loadObject(ofClass: UIImage.self)` (that decodes to rendered JPEG)
- If UTI lookup fails, inspect `registeredTypeIdentifiers` and fall back through: `public.camera-raw-image` → `com.adobe.raw-image` → vendor UTIs

**Secondary (Files / external):** SwiftUI `.fileImporter(allowedContentTypes: [.image, .rawImage])`. Security-scoped URLs — always call `startAccessingSecurityScopedResource()` before reading.

**Drag and drop:** `.dropDestination(for: URL.self)` (iOS 16+, `Transferable`-based). Accept `URL` to avoid copying large files into memory.

---

## Setup (one-time, requires Mac)

### 1. Pre-compile Depth Anything v2 model
```bash
# Download from https://huggingface.co/apple/coreml-depth-anything-v2-small
xcrun coremlcompiler compile DepthAnythingV2SmallF16.mlpackage /tmp/
cp -r /tmp/DepthAnythingV2SmallF16.mlmodelc \
  FocusApp.swiftpm/Sources/App/Resources/
```

### 2. Register in Package.swift
```swift
targets: [
    .executableTarget(
        name: "App",
        path: "Sources/App",
        resources: [
            .copy("Resources/DepthAnythingV2SmallF16.mlmodelc")
        ]
    )
]
```

### 3. Entitlements (via Playgrounds Capabilities UI)
- Photos Library (read access)
- Camera (if live capture added later)
- `com.apple.developer.kernel.increased-memory-limit` (required for 50 MP+ RAW on 8 GB iPads)

---

## Known Limitations & Non-Goals

- **No custom `.metal` source authoring** in Playgrounds — use MPS / Core Image exclusively
- **No breakpoint debugger in Playgrounds** — for GPU debugging, open `.swiftpm` in Xcode and use Metal Frame Capture / Instruments
- **Motion blur** is not distinguished from defocus by classical operators — known limitation, no fix without a dedicated DBD network
- **Smooth in-focus regions** (skin, sky) underestimate sharpness in Laplacian path — Depth Anything v2 hybrid mode corrects this
- Vision framework has **no spatial focus-map API** (confirmed as of iPadOS/macOS 26) — all focus detection is custom
- `CIRAWFilter` supports Apple's [published camera list](https://support.apple.com/en-us/122870) only; cameras outside it require LibRaw

---

## References

- Apple Accelerate sample: [Finding the sharpest image in a sequence](https://developer.apple.com/documentation/accelerate/finding-the-sharpest-image-in-a-sequence-of-captured-images)
- Depth Anything v2 Core ML model + Swift example: [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small)
- Metal in Swift Playgrounds 4: [Better Programming](https://betterprogramming.pub/using-metal-in-swift-playgrounds-4-e100122d276a)
- Core ML in `.swiftpm`: [Apple Developer Forums thread 771598](https://developer.apple.com/forums/thread/771598)
- EDR rendering: WWDC22 session 10114 — *Display EDR content with Core Image, Metal, and SwiftUI*
- Classical focus operators benchmark: Pertuz, Puig, García — *Pattern Recognition* 46(5), 2013
