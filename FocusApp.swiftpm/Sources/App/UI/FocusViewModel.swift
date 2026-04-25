import SwiftUI
import CoreImage
import Metal

enum OverlayStyle: String, CaseIterable, Identifiable {
    // `.off` rather than `.none` so we don't shadow Optional.none in call sites.
    case off        = "None"
    case peaking    = "Peaking"
    case mask       = "Mask"
    case heatmap    = "Heatmap"
    case focusError = "Error"
    case motion     = "Motion"
    case errorLevel = "ELA"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .off:        return "circle.slash"
        case .peaking:    return "sparkles"
        case .mask:       return "square.fill.on.square"
        case .heatmap:    return "thermometer.sun"
        case .focusError: return "scope"
        case .motion:     return "wind"
        case .errorLevel: return "rectangle.dashed"
        }
    }

    /// `.focusError` needs both sharpness and depth to estimate the focal plane.
    var requiresDepth: Bool {
        switch self {
        case .focusError: return true
        default:          return false
        }
    }

    /// When true, the renderer skips all overlay compositing and the UI
    /// disables any control that would otherwise be routed through the
    /// analysis pipeline (threshold slider, analysis-mode picker).
    var isOff: Bool { self == .off }
}

enum AnalysisMode: String, CaseIterable, Identifiable {
    case sharpness = "Sharpness"
    case depth     = "Depth"
    case hybrid    = "Hybrid"

    var id: String { rawValue }
}

enum DepthInstallState: Equatable {
    case notInstalled
    case downloading(progress: Double)
    case installed
    case failed(String)
}

/// Aggregate of every optional Core ML tier's install state. One
/// case per archive that ships under `ModelArchive`. Bundled as a
/// struct so the view model isn't paging eighteen flat published
/// properties (state + download-task per tier); SwiftUI consumers
/// reach in via `viewModel.installs.<tier>`.
struct InstallStates: Equatable {
    var depth: DepthInstallState = .notInstalled
    var nsfw: DepthInstallState = .notInstalled
    var nudenet: DepthInstallState = .notInstalled
    var clip: DepthInstallState = .notInstalled
    var emotion: DepthInstallState = .notInstalled
    var openGraphAU: DepthInstallState = .notInstalled
    var age: DepthInstallState = .notInstalled
    var quality: DepthInstallState = .notInstalled
    var aesthetic: DepthInstallState = .notInstalled
    var genitalClassifier: DepthInstallState = .notInstalled
}

/// Slot per archive for the in-flight download Task. Kept private +
/// off `@Published` because task lifecycle isn't user-visible state
/// — only the install enum is.
struct InstallTasks {
    var depth: Task<Void, Never>?
    var nsfw: Task<Void, Never>?
    var nudenet: Task<Void, Never>?
    var clip: Task<Void, Never>?
    var emotion: Task<Void, Never>?
    var openGraphAU: Task<Void, Never>?
    var age: Task<Void, Never>?
    var quality: Task<Void, Never>?
    var aesthetic: Task<Void, Never>?
    var genitalClassifier: Task<Void, Never>?
}

/// Snapshot of where the analysis pipeline is. Reported from
/// `FocusAnalyzer.analyze`'s progress callback and mirrored into
/// the view model so the main content view can render a determinate
/// progress bar with the currently-running stage's name.
struct AnalysisProgress: Equatable, Sendable {
    var fraction: Double
    var label: String
}

/// Which region the mosaic covers when the user has the mosaic toggle on.
/// .eyes uses a solid black bar rather than pixelate; the others pixelate.
enum MosaicMode: String, CaseIterable, Identifiable {
    case tabloid = "Tabloid"
    case jacket  = "Jacket"
    case eyes    = "Eyes"
    case face    = "Face"
    case chest   = "Chest"
    case groin   = "Groin"
    case body    = "Body"
    case privy   = "Privy"
    case nudity  = "Nudity"
    var id: String { rawValue }
}

@MainActor
final class FocusViewModel: ObservableObject {
    // Scrubbable display state — cheap, no re-analysis.
    @Published var threshold: Float = 0.35
    @Published var overlayColor: Color = .red
    @Published var style: OverlayStyle = .off {
        didSet {
            // Focus Error needs depth data. Auto-promote to hybrid analysis so
            // the user doesn't have to toggle two controls.
            if style.requiresDepth && mode != .hybrid && depthAvailable {
                mode = .hybrid
                reanalyze()
            }
        }
    }

    // Zoom display state — purely view-layer, no re-analysis.
    // `zoomAnchor` is in normalized view coordinates (0...1), Y-top origin.
    // `zoomPanOffset` is an additional translation in view-pixel coords
    // applied after the anchor-based scale, so the user can drag to
    // explore when zoomed in without re-anchoring.
    @Published var zoomScale: CGFloat = 1.0
    @Published var zoomAnchor: CGPoint = CGPoint(x: 0.5, y: 0.5)
    @Published var zoomPanOffset: CGSize = .zero

    /// Press-and-hold to compare against the original photo. While true, the
    /// renderer skips all overlay compositing and draws just the fitted source.
    @Published var overlayHidden: Bool = false

    private var zoomAnimationTask: Task<Void, Never>?

    /// Double-tap handler: toggle between fit-to-view and 2.5x centered at `normalized`,
    /// interpolating the transition over ~250 ms. SwiftUI's withAnimation doesn't apply
    /// here because MTKView reads the raw values in its render callback rather than
    /// through SwiftUI's transaction system, so we drive the animation manually.
    func toggleZoom(at normalized: CGPoint) {
        let zoomedIn = zoomScale > 1.001
        let targetScale: CGFloat = zoomedIn ? 1.0 : nativeZoomScale()
        // Anchor at the tap location for both directions — matches
        // Photos.app: the content under the finger stays under the
        // finger as the image scales. At the zoom-out endpoint
        // (scale=1) the renderer's fit() ignores anchor anyway, so
        // the final frame is centered regardless. Clamp first so a
        // tap inside a letterbox region (or near an edge of a non-
        // matching aspect source) doesn't pick a fixed point that
        // leaves a black gap on the opposite side after the zoom.
        let targetAnchor = clampedAnchor(normalized, forScale: targetScale)
        animateZoom(toScale: targetScale, toAnchor: targetAnchor)
    }

    /// Reconcile the current zoom + anchor against the renderer's
    /// latest drawable size. Called from `FocusRenderer.resize` so
    /// a layout change (full-screen toggle, Stage Manager resize,
    /// rotation) doesn't leave a previously-fine zoom showing
    /// black bands because the fit scale shifted under it.
    ///
    /// Two corrections applied in order:
    ///   1. Bump zoom to the minimum that still covers the new
    ///      drawable. The previous zoom may have been computed for
    ///      a different aspect; if the new drawable's letterbox
    ///      axis is wider relative, the post-zoom image can be
    ///      fundamentally narrower than the drawable and no anchor
    ///      eliminates the gap.
    ///   2. Re-clamp anchor against the (possibly new) zoom — the
    ///      valid range tightens when zoom drops or the letterbox
    ///      grows.
    /// Pan is zeroed since the previous offset was tuned to a
    /// drawable that no longer applies.
    ///
    /// No-op while a zoom animation is in flight — the animation
    /// loop owns those properties for its duration; a side-band
    /// edit would fight the easing for a frame.
    func reclampForDrawableChange() {
        guard zoomAnimationTask == nil else { return }
        guard zoomScale > 1.001 else { return }
        let cover = coveringZoomScale()
        if zoomScale < cover { zoomScale = cover }
        zoomAnchor = clampedAnchor(zoomAnchor, forScale: zoomScale)
        zoomPanOffset = .zero
    }

    /// Minimum zoom factor that keeps the fitted image covering
    /// the drawable on both axes. The fit-binding axis already
    /// covers exactly at zoom = 1, so cover comes from the other
    /// axis's drawable / fitted ratio.
    private func coveringZoomScale() -> CGFloat {
        guard let src = sourceImage,
              lastDrawableSize.width > 0,
              lastDrawableSize.height > 0
        else { return 1 }
        let fit = min(lastDrawableSize.width / src.extent.width,
                      lastDrawableSize.height / src.extent.height)
        guard fit > 0 else { return 1 }
        let coverX = lastDrawableSize.width / (src.extent.width * fit)
        let coverY = lastDrawableSize.height / (src.extent.height * fit)
        return max(coverX, coverY)
    }

    /// Clamp a normalized anchor (0..1, Y-top) to the range that
    /// keeps the post-zoom image fully covering the drawable —
    /// mirrors the `clampedPan` math in ContentView, just solved
    /// for the anchor instead of the pan.
    ///
    /// For axis with no letterbox (fitted dimension == drawable
    /// dimension), the valid range is the full [0, 1] — anchor is
    /// unrestricted. For an axis with letterbox, the range
    /// narrows as zoom approaches 1; if the post-zoom fitted
    /// dimension is still smaller than the drawable, no anchor
    /// can eliminate the gap so we pin to the centre.
    ///
    /// Skipped (returns the input unchanged) at scale ≤ 1.001
    /// since the renderer's fit() ignores anchor below that
    /// threshold.
    private func clampedAnchor(_ normalized: CGPoint, forScale zoom: CGFloat) -> CGPoint {
        guard zoom > 1.001,
              let src = sourceImage,
              lastDrawableSize.width > 0,
              lastDrawableSize.height > 0
        else { return normalized }
        let fit = min(lastDrawableSize.width / src.extent.width,
                      lastDrawableSize.height / src.extent.height)
        let fittedW = src.extent.width * fit
        let fittedH = src.extent.height * fit
        let marginX = zoom * (1 - fittedW / lastDrawableSize.width) / (2 * (zoom - 1))
        let marginY = zoom * (1 - fittedH / lastDrawableSize.height) / (2 * (zoom - 1))
        let x = marginX >= 0.5 ? 0.5 : max(marginX, min(1 - marginX, normalized.x))
        let y = marginY >= 0.5 ? 0.5 : max(marginY, min(1 - marginY, normalized.y))
        return CGPoint(x: x, y: y)
    }

    /// Zoom factor that maps one source pixel onto one drawable
    /// pixel for the constraining axis — i.e. native pixel-perfect
    /// view of the source. Floor at 1.5 so a sub-drawable thumbnail
    /// doesn't compute to a zoom-out. Falls back to 2.5x (the
    /// pre-native default) before the renderer has reported a
    /// drawable size or while no image is loaded.
    private func nativeZoomScale() -> CGFloat {
        guard let src = sourceImage,
              lastDrawableSize.width > 0,
              lastDrawableSize.height > 0
        else { return 2.5 }
        let extent = src.extent
        let sx = extent.width / lastDrawableSize.width
        let sy = extent.height / lastDrawableSize.height
        return max(1.5, max(sx, sy))
    }

    private func animateZoom(toScale target: CGFloat, toAnchor targetAnchor: CGPoint) {
        zoomAnimationTask?.cancel()
        // Capture the starting pan and interpolate it to zero
        // alongside scale + anchor. Without this, a non-zero pan
        // (user dragged to a corner while zoomed in) would snap to
        // zero on the first frame, making the zoom-out look like
        // it pivots from the centered image rather than from the
        // viewing position. Both toggle directions end at pan = 0.
        let startScale = zoomScale
        let startAnchor = zoomAnchor
        let startPan = zoomPanOffset
        let duration: Double = 0.25
        let stepDuration: Double = 1.0 / 60.0
        let steps = max(1, Int(duration / stepDuration))
        let stepNS = UInt64(stepDuration * 1_000_000_000)

        zoomAnimationTask = Task { @MainActor [weak self] in
            for step in 1...steps {
                if Task.isCancelled { return }

                // Broken into locals so the type checker doesn't have to solve
                // one big mixed-type expression, and so Swift 6's concurrency
                // checker doesn't flag a strongly-bound `self` inside a Sendable
                // closure.
                let progress = Double(step) / Double(steps)
                let easedValue: Double
                if progress < 0.5 {
                    easedValue = 2.0 * progress * progress
                } else {
                    let u = 1.0 - progress
                    easedValue = 1.0 - 2.0 * u * u
                }
                let eased = CGFloat(easedValue)

                let newScale = startScale + (target - startScale) * eased
                let ax = startAnchor.x + (targetAnchor.x - startAnchor.x) * eased
                let ay = startAnchor.y + (targetAnchor.y - startAnchor.y) * eased
                let panW = startPan.width + (0 - startPan.width) * eased
                let panH = startPan.height + (0 - startPan.height) * eased

                self?.zoomScale = newScale
                self?.zoomAnchor = CGPoint(x: ax, y: ay)
                self?.zoomPanOffset = CGSize(width: panW, height: panH)

                try? await Task.sleep(nanoseconds: stepNS)
            }
            if !Task.isCancelled {
                self?.zoomScale = target
                self?.zoomAnchor = targetAnchor
                self?.zoomPanOffset = .zero
            }
        }
    }

    // Analysis configuration — change triggers re-analysis.
    @Published var mode: AnalysisMode = .sharpness

    // Source + derived state published for renderer consumption.
    @Published var sourceImage: CIImage?
    @Published var sourceName: String?
    @Published var sharpnessOverlay: CIImage?
    @Published var depthOverlay: CIImage?
    @Published var focalPlane: Float?
    @Published var motionBlur: MotionBlurReport?
    @Published var motionOverlay: CIImage?
    /// Per-pixel ELA diff produced once per image load. Renderer
    /// applies threshold-driven gain at display time so the slider
    /// scrubs sensitivity. nil while ELA hasn't completed for the
    /// current source or if the round-trip failed.
    @Published var errorLevelOverlay: CIImage?
    @Published var exposureInfo: ExposureInfo?
    @Published var isSensitive: Bool?
    @Published var sensitiveLabel: String?
    @Published var sensitiveConfidence: Float?
    @Published var faceRectangles: [CGRect] = []
    @Published var bodyRectangles: [CGRect] = []
    @Published var groinRectangles: [CGRect] = []
    @Published var eyeBars: [EyeBar] = []
    @Published var chestRectangles: [CGRect] = []
    /// Person-silhouette mask stretched to source extent. Populated on
    /// analyze; consumed by the .body mosaic mode to pixelate along the
    /// actual outline instead of a loose bounding box.
    @Published var personMask: CIImage?
    /// Per-body nudity level from NudeNet, same index as `bodyRectangles`.
    /// Empty when the NudeNet model isn't installed.
    @Published var nudityLevels: [NudityLevel] = []
    /// Per-body inferred gender from NudeNet's FACE_* branch, same
    /// index as `bodyRectangles`. `.unknown` when no face detection
    /// attributed to that body.
    @Published var nudityGenders: [SubjectGender] = []
    /// Raw NudeNet per-part detections. Only consumed by the optional
    /// label overlay; mosaic pipeline uses `nudityLevels` instead.
    @Published var nudityDetections: [NudityDetection] = []
    /// CLIP zero-shot context matches for the current image, sorted
    /// highest-similarity first. Empty when the CLIP bundle isn't
    /// installed.
    @Published var clipMatches: [CLIPMatch] = []
    /// Per-face emotion predictions from FER+, indexed alongside
    /// `faceRectangles`. `nil` entries mean the classifier didn't
    /// meet its confidence floor for that face. Empty when the
    /// emotion model isn't installed.
    @Published var faceEmotions: [EmotionPrediction?] = []
    /// Per-face pain score from OpenGraphAU + Vision-derived AU43,
    /// indexed parallel to `faceRectangles`. Empty when the pain
    /// model isn't installed; `nil` entries for faces the detector
    /// couldn't score.
    @Published var painScores: [PainScore?] = []
    /// Per-face age prediction from SSR-Net, indexed parallel to
    /// `faceRectangles`. Empty when the age model isn't installed;
    /// `nil` entries for faces that couldn't be cropped. Age-only
    /// — gender comes from `nudityGenders` (NudeNet) exclusively.
    @Published var ageEstimations: [AgePrediction?] = []
    /// Whole-image technical quality score from NIMA. `nil` when
    /// the model isn't installed.
    @Published var qualityScore: QualityScore?
    /// Whole-image aesthetic quality score from NIMA. `nil` when
    /// the aesthetic variant isn't installed.
    @Published var aestheticScore: QualityScore?
    /// User toggle: draw NudeNet detection boxes + class labels on top
    /// of the image. Hidden by default so most users don't see the raw
    /// detector output.
    @Published var showNudityLabels: Bool = false
    /// User toggle: render the per-subject meter row (PAD bars plus
    /// the OpenGraphAU pain bar, when installed) under each head
    /// badge. Off by default — the head badge alone reads cleanly
    /// enough for most photos; flip on when you want the continuous
    /// emotion / pain detail.
    @Published var showPADMeter: Bool = false
    /// Minimum level that triggers the per-subject mosaic gating. Bodies
    /// whose level is below this are skipped even when the global mosaic
    /// condition is on. `.covered` leaves clothed subjects alone.
    @Published var nudityGate: NudityLevel = .none
    /// User-controlled mosaic toggle. Defaults on — protective default so
    /// sensitive content isn't displayed until the user explicitly opts in.
    @Published var mosaicEnabled: Bool = true
    @Published var mosaicMode: MosaicMode = .tabloid
    /// When true, the selected `mosaicMode` is applied to every image
    /// regardless of what the sensitive-content classifier returns — useful
    /// for redacting screenshots, preparing sample images for documentation,
    /// or working with photos the classifier doesn't flag but the user still
    /// wants covered.
    @Published var forceCensor: Bool = false
    @Published var sensitiveContentAvailability: SensitiveContentAvailability = .frameworkMissing
    /// All nine optional Core ML tier install states bundled into one
    /// struct so the view model isn't dragging eighteen flat
    /// properties (state + download-task per tier). The shape
    /// (notInstalled / downloading / installed / failed) is generic;
    /// each tier reuses `DepthInstallState`.
    @Published var installs = InstallStates()
    @Published var isAnalyzing: Bool = false
    /// Fractional progress + current-stage label from the analysis
    /// pipeline. nil while idle. The bar is fixed-weight per stage
    /// (each step counts equally regardless of timing), so the
    /// fraction is a rough position indicator, not calibrated time.
    @Published var analysisProgress: AnalysisProgress?
    @Published var errorMessage: String?
    @Published var depthAvailable: Bool = false

    let analyzer: FocusAnalyzer
    /// Latest MTKView drawable size in pixels, pushed by
    /// `FocusRenderer.resize(to:)` whenever the view's drawable
    /// dimensions change by more than a pixel. Plain var, not
    /// @Published — `toggleZoom` is the only consumer and it
    /// reads on demand, so observation would only churn invalidations.
    var lastDrawableSize: CGSize = .zero
    private var currentTask: Task<Void, Never>?
    /// URL of the file we copied to temporary storage in
    /// `ImageImporter`. Tracked so we can delete the previous one
    /// when a new image is loaded — otherwise every imported photo
    /// (5–100 MB each) leaks into the temp directory until the OS
    /// reclaims it, and the app's working set / disk footprint
    /// grows linearly with imports per session.
    private var previousLoadedURL: URL?
    /// Active download task slots, parallel to `installs`. Kept off
    /// `@Published` so spurious "task started / finished"
    /// invalidations don't churn SwiftUI views that only care about
    /// the user-visible install state.
    private var installTasks = InstallTasks()

    init() {
        self.analyzer = FocusAnalyzer()
        // Resolve the install-state flags synchronously so the UI never
        // briefly shows a "Download" button for an already-installed
        // model. `ModelArchive.isInstalled` is a cheap disk check that
        // doesn't need the actor — calling it through `await analyzer`
        // creates a race window where the user can re-trigger a
        // download that may now 404 against a deleted GitHub release,
        // wiping the perfectly-fine install they already had.
        let depthInstalled = ModelArchive.depthAnything.isInstalled()
        let nsfwInstalled = ModelArchive.nsfw.isInstalled()
        let nudenetInstalled = ModelArchive.nudenet.isInstalled()
        let clipInstalled = ModelArchive.clip.isInstalled()
        let emotionInstalled = ModelArchive.emotion.isInstalled()
        let openGraphAUInstalled = ModelArchive.openGraphAU.isInstalled()
        let ageInstalled = ModelArchive.age.isInstalled()
        let qualityInstalled = ModelArchive.quality.isInstalled()
        let aestheticInstalled = ModelArchive.aesthetic.isInstalled()
        let genitalClassifierInstalled = ModelArchive.genitalClassifier.isInstalled()
        self.depthAvailable = depthInstalled
        self.installs = InstallStates(
            depth: depthInstalled ? .installed : .notInstalled,
            nsfw: nsfwInstalled ? .installed : .notInstalled,
            nudenet: nudenetInstalled ? .installed : .notInstalled,
            clip: clipInstalled ? .installed : .notInstalled,
            emotion: emotionInstalled ? .installed : .notInstalled,
            openGraphAU: openGraphAUInstalled ? .installed : .notInstalled,
            age: ageInstalled ? .installed : .notInstalled,
            quality: qualityInstalled ? .installed : .notInstalled,
            aesthetic: aestheticInstalled ? .installed : .notInstalled,
            genitalClassifier: genitalClassifierInstalled ? .installed : .notInstalled
        )

        let analyzer = self.analyzer
        // Pre-compile installed Core ML models in the background after
        // launch so the first analyze doesn't block on cold compile.
        // Utility priority yields to anything the UI is doing; by the
        // time the user picks an image the models are usually ready.
        Task.detached(priority: .utility) {
            await analyzer.dumpInstallDirectory()
            await analyzer.prewarmModels()
        }
    }

    /// Re-query the analyzer for Communication Safety state. Call on app
    /// foreground / image load so a setting toggled while the app is running
    /// is picked up without a restart.
    func refreshSensitiveContentAvailability() {
        let analyzer = self.analyzer
        Task { [weak self] in
            let sensitive = await analyzer.sensitiveContentAvailability
            await MainActor.run { [weak self] in
                self?.sensitiveContentAvailability = sensitive
            }
        }
    }

    /// Generic download driver shared by every model tier. Each
    /// public `download<Model>Model()` is a thin wrapper that picks
    /// the right archive, state property, task slot, and install
    /// method to invoke. `onInstalled` runs on the MainActor right
    /// before `state = .installed` for tiers that need extra
    /// follow-up (depth flips `depthAvailable`; NSFW re-queries SCA).
    ///
    /// Failure semantics: re-downloads can fail (404 / network)
    /// without touching the existing on-disk install. The installer
    /// only wipes the destination after the new content has been
    /// unpacked, so when the install file is still on disk after a
    /// failure, treat it as `.installed` rather than `.failed` so
    /// the UI doesn't surface a misleading error for a model that's
    /// still working.
    private func download(
        archive: ModelArchive,
        state stateKP: ReferenceWritableKeyPath<FocusViewModel, DepthInstallState>,
        task taskKP: ReferenceWritableKeyPath<FocusViewModel, Task<Void, Never>?>,
        install: @Sendable @escaping (FocusAnalyzer, @Sendable @escaping (Double) -> Void) async throws -> Void,
        onInstalled: (@MainActor @Sendable (FocusViewModel) -> Void)? = nil
    ) {
        guard self[keyPath: taskKP] == nil else { return }
        self[keyPath: stateKP] = .downloading(progress: 0)
        let analyzer = self.analyzer
        self[keyPath: taskKP] = Task { [weak self] in
            do {
                try await install(analyzer) { [weak self] p in
                    Task { @MainActor in
                        self?[keyPath: stateKP] = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    onInstalled?(self)
                    self[keyPath: stateKP] = .installed
                }
            } catch {
                let stillInstalled = archive.isInstalled()
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    if stillInstalled {
                        onInstalled?(self)
                        self[keyPath: stateKP] = .installed
                    } else {
                        self[keyPath: stateKP] = .failed(error.localizedDescription)
                    }
                }
            }
            await MainActor.run { [weak self] in
                self?[keyPath: taskKP] = nil
            }
        }
    }

    func downloadDepthModel() {
        download(
            archive: .depthAnything,
            state: \.installs.depth,
            task: \.installTasks.depth,
            install: { try await $0.installDepthModel(progress: $1) },
            onInstalled: { $0.depthAvailable = true }
        )
    }

    func downloadNSFWModel() {
        download(
            archive: .nsfw,
            state: \.installs.nsfw,
            task: \.installTasks.nsfw,
            install: { try await $0.installNSFWModel(progress: $1) },
            // Re-query SCA so the row reflects that the NSFW
            // fallback is now usable.
            onInstalled: { $0.refreshSensitiveContentAvailability() }
        )
    }

    func downloadNudeNetModel() {
        download(
            archive: .nudenet,
            state: \.installs.nudenet,
            task: \.installTasks.nudenet,
            install: { try await $0.installNudeNetModel(progress: $1) }
        )
    }

    func downloadCLIPModel() {
        download(
            archive: .clip,
            state: \.installs.clip,
            task: \.installTasks.clip,
            install: { try await $0.installCLIPModel(progress: $1) }
        )
    }

    func downloadEmotionModel() {
        download(
            archive: .emotion,
            state: \.installs.emotion,
            task: \.installTasks.emotion,
            install: { try await $0.installEmotionModel(progress: $1) }
        )
    }

    func downloadOpenGraphAUModel() {
        download(
            archive: .openGraphAU,
            state: \.installs.openGraphAU,
            task: \.installTasks.openGraphAU,
            install: { try await $0.installOpenGraphAUModel(progress: $1) }
        )
    }

    func downloadAgeModel() {
        download(
            archive: .age,
            state: \.installs.age,
            task: \.installTasks.age,
            install: { try await $0.installAgeModel(progress: $1) }
        )
    }

    func downloadQualityModel() {
        download(
            archive: .quality,
            state: \.installs.quality,
            task: \.installTasks.quality,
            install: { try await $0.installQualityModel(progress: $1) }
        )
    }

    func downloadAestheticModel() {
        download(
            archive: .aesthetic,
            state: \.installs.aesthetic,
            task: \.installTasks.aesthetic,
            install: { try await $0.installAestheticModel(progress: $1) }
        )
    }

    func downloadGenitalClassifierModel() {
        download(
            archive: .genitalClassifier,
            state: \.installs.genitalClassifier,
            task: \.installTasks.genitalClassifier,
            install: { try await $0.installGenitalClassifierModel(progress: $1) }
        )
    }

    func load(url: URL, name: String) {
        print("[ViewModel] load url=\(url.path) name=\(name)")
        currentTask?.cancel()
        isAnalyzing = true
        analysisProgress = AnalysisProgress(fraction: 0, label: "Loading image")
        errorMessage = nil
        // Deliberately don't clear the derived overlay state here. The
        // previous image is still on screen while the new analysis
        // runs, and clearing would un-mosaic it mid-load — bad news
        // for a sensitive photo the user explicitly wanted covered.
        // The brief frame of new-overlays-on-old-image that SwiftUI +
        // MTKView can produce at the transition is the lesser of the
        // two evils; overlays and sourceImage are updated together at
        // the bottom of the completion block. Same reason applies to
        // exposureInfo and sourceName — they get set in the completion
        // block rather than synchronously here so the EXIF capsule and
        // the title don't flip to the new image's metadata while the
        // previous image is still on screen.
        refreshSensitiveContentAvailability()

        let mode = self.mode
        let analyzer = self.analyzer

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                print("[ViewModel] loadImage on analyzer…")
                let image = try await analyzer.loadImage(from: url)
                print("[ViewModel] loaded CIImage extent=\(image.extent)")
                try Task.checkCancellation()
                // Read EXIF off the source URL on the worker — quick
                // metadata-only read with no pixel decode. Hand the
                // result back to the main actor alongside the image.
                let exposure = ExposureInfo.read(from: url)
                print("[ViewModel] analyze mode=\(mode)")
                let overlays = try await analyzer.analyze(mode: mode) { [weak self] fraction, label in
                    Task { @MainActor in
                        self?.analysisProgress = AnalysisProgress(
                            fraction: fraction,
                            label: label ?? ""
                        )
                    }
                }
                print("[ViewModel] analyze done — sourceImage about to set")
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    self?.sourceImage = image
                    self?.sourceName = name
                    self?.exposureInfo = exposure
                    self?.sharpnessOverlay = overlays.sharpness
                    self?.depthOverlay = overlays.depth
                    self?.focalPlane = overlays.focalPlane
                    self?.motionBlur = overlays.motionBlur
                    self?.motionOverlay = overlays.motionOverlay
                    self?.isSensitive = overlays.isSensitive
                    self?.sensitiveLabel = overlays.sensitiveLabel
                    self?.sensitiveConfidence = overlays.sensitiveConfidence
                    self?.faceRectangles = overlays.faceRectangles
                    self?.bodyRectangles = overlays.bodyRectangles
                    self?.groinRectangles = overlays.groinRectangles
                    self?.eyeBars = overlays.eyeBars
                    self?.chestRectangles = overlays.chestRectangles
                    self?.personMask = overlays.personMask
                    self?.nudityLevels = overlays.nudityLevels
                    self?.nudityGenders = overlays.nudityGenders
                    self?.nudityDetections = overlays.nudityDetections
                    self?.clipMatches = overlays.clipMatches
                    self?.faceEmotions = overlays.faceEmotions
                    self?.painScores = overlays.painScores
                    self?.ageEstimations = overlays.ageEstimations
                    self?.qualityScore = overlays.quality
                    self?.aestheticScore = overlays.aesthetic
                    self?.errorLevelOverlay = overlays.errorLevel
                    self?.isAnalyzing = false
                    self?.analysisProgress = nil
                    // Delete the previous import's temp file once
                    // the new sourceImage has replaced it (which
                    // releases the old CIImage's hold on the
                    // backing file). Track this load's URL for the
                    // next swap.
                    self?.cleanupPreviousImport()
                    self?.previousLoadedURL = url
                    print("[ViewModel] sourceImage set, isAnalyzing=false")
                }
            } catch is CancellationError {
                print("[ViewModel] load cancelled")
                await MainActor.run { [weak self] in
                    self?.analysisProgress = nil
                }
            } catch {
                print("[ViewModel] load FAIL: \(error)")
                await MainActor.run { [weak self] in
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                    self?.analysisProgress = nil
                }
            }
        }
    }

    /// Best-effort delete of the previous import's temp file. Called
    /// after a new sourceImage has replaced the old one (so the old
    /// CIImage's mmap on the file is released first). Errors are
    /// silent — a stale temp file isn't a correctness issue, just a
    /// disk-space one.
    private func cleanupPreviousImport() {
        guard let url = previousLoadedURL else { return }
        let tmpRoot = FileManager.default.temporaryDirectory.path
        // Sanity: only delete if it's actually under the temp dir.
        // Refuse to touch anything else (e.g. the user picked a
        // photos-library URL that we accidentally ended up tracking).
        guard url.path.hasPrefix(tmpRoot) else { return }
        try? FileManager.default.removeItem(at: url)
        previousLoadedURL = nil
    }

    func clear() {
        currentTask?.cancel()
        currentTask = nil
        zoomAnimationTask?.cancel()
        zoomAnimationTask = nil
        // Drop the loaded image first so the CIImage releases its
        // hold on the backing temp file, then delete the file.
        sourceImage = nil
        cleanupPreviousImport()
        sourceName = nil
        sharpnessOverlay = nil
        depthOverlay = nil
        focalPlane = nil
        motionBlur = nil
        motionOverlay = nil
        isSensitive = nil
        sensitiveLabel = nil
        sensitiveConfidence = nil
        faceRectangles = []
        bodyRectangles = []
        groinRectangles = []
        eyeBars = []
        chestRectangles = []
        personMask = nil
        nudityLevels = []
        nudityGenders = []
        nudityDetections = []
        clipMatches = []
        faceEmotions = []
        painScores = []
        ageEstimations = []
        qualityScore = nil
        aestheticScore = nil
        exposureInfo = nil
        errorMessage = nil
        isAnalyzing = false
        analysisProgress = nil
        zoomScale = 1.0
        zoomAnchor = CGPoint(x: 0.5, y: 0.5)
        zoomPanOffset = .zero
    }

    /// Render the current image + overlays at source resolution and write
    /// a PNG file to the temp directory. Returns the URL so ContentView
    /// can feed it to a ShareLink. Throws if no image is loaded or the
    /// encode fails.
    func exportPNG() async throws -> URL {
        guard let source = sourceImage else { throw AnalysisError.imageDecodeFailed }

        let inputs = FocusCompositeInputs(
            source: source,
            style: style,
            threshold: threshold,
            tint: CIColor(color: overlayColor) ?? CIColor(red: 1, green: 0.85, blue: 0),
            focalPlane: focalPlane,
            sharpness: sharpnessOverlay,
            depth: depthOverlay,
            motion: motionOverlay,
            mosaic: forceCensor || ((isSensitive == true) && mosaicEnabled),
            mosaicMode: mosaicMode,
            faces: faceRectangles,
            bodies: bodyRectangles,
            groins: groinRectangles,
            eyes: eyeBars,
            chests: chestRectangles,
            personMask: personMask,
            nudityLevels: nudityLevels,
            nudityGate: nudityGate,
            nudityDetections: nudityDetections
        )

        let baseName: String
        if let name = sourceName, !name.isEmpty {
            let stripped = URL(fileURLWithPath: name)
                .deletingPathExtension()
                .lastPathComponent
            baseName = stripped.isEmpty ? "FocusCheck" : stripped
        } else {
            baseName = "FocusCheck"
        }
        let suffix = UUID().uuidString.prefix(8)
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(baseName)-\(suffix)")
            .appendingPathExtension("png")

        try await analyzer.exportPNG(inputs: inputs, destination: destination)
        return destination
    }

    func reanalyze() {
        guard sourceImage != nil else { return }
        currentTask?.cancel()
        isAnalyzing = true
        analysisProgress = AnalysisProgress(fraction: 0, label: "Re-analyzing")
        let mode = self.mode
        let analyzer = self.analyzer
        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let overlays = try await analyzer.analyze(mode: mode) { [weak self] fraction, label in
                    Task { @MainActor in
                        self?.analysisProgress = AnalysisProgress(
                            fraction: fraction,
                            label: label ?? ""
                        )
                    }
                }
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    self?.sharpnessOverlay = overlays.sharpness
                    self?.depthOverlay = overlays.depth
                    self?.focalPlane = overlays.focalPlane
                    self?.motionBlur = overlays.motionBlur
                    self?.motionOverlay = overlays.motionOverlay
                    self?.isSensitive = overlays.isSensitive
                    self?.sensitiveLabel = overlays.sensitiveLabel
                    self?.sensitiveConfidence = overlays.sensitiveConfidence
                    self?.faceRectangles = overlays.faceRectangles
                    self?.bodyRectangles = overlays.bodyRectangles
                    self?.groinRectangles = overlays.groinRectangles
                    self?.eyeBars = overlays.eyeBars
                    self?.chestRectangles = overlays.chestRectangles
                    self?.personMask = overlays.personMask
                    self?.nudityLevels = overlays.nudityLevels
                    self?.nudityGenders = overlays.nudityGenders
                    self?.nudityDetections = overlays.nudityDetections
                    self?.clipMatches = overlays.clipMatches
                    self?.faceEmotions = overlays.faceEmotions
                    self?.painScores = overlays.painScores
                    self?.ageEstimations = overlays.ageEstimations
                    self?.qualityScore = overlays.quality
                    self?.aestheticScore = overlays.aesthetic
                    self?.errorLevelOverlay = overlays.errorLevel
                    self?.isAnalyzing = false
                    self?.analysisProgress = nil
                }
            } catch is CancellationError {
                // Swallow — a newer analyze is in flight.
                await MainActor.run { [weak self] in
                    self?.analysisProgress = nil
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                    self?.analysisProgress = nil
                }
            }
        }
    }
}
