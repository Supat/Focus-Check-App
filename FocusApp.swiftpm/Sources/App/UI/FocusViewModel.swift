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

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .off:        return "circle.slash"
        case .peaking:    return "sparkles"
        case .mask:       return "square.fill.on.square"
        case .heatmap:    return "thermometer.sun"
        case .focusError: return "scope"
        case .motion:     return "wind"
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

/// Which region the mosaic covers when the user has the mosaic toggle on.
/// .eyes uses a solid black bar rather than pixelate; the others pixelate.
enum MosaicMode: String, CaseIterable, Identifiable {
    case tabloid = "Tabloid"
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
        let targetScale: CGFloat = zoomedIn ? 1.0 : 2.5
        let targetAnchor: CGPoint = zoomedIn
            ? CGPoint(x: 0.5, y: 0.5)
            : normalized
        animateZoom(toScale: targetScale, toAnchor: targetAnchor)
    }

    private func animateZoom(toScale target: CGFloat, toAnchor targetAnchor: CGPoint) {
        zoomAnimationTask?.cancel()
        // Starting (or ending) a new zoom wipes any pan from the prior
        // zoomed state — otherwise the transition would interpolate
        // scale/anchor while a stale pan offset stayed fixed, producing
        // a visual jump at either end.
        zoomPanOffset = .zero

        let startScale = zoomScale
        let startAnchor = zoomAnchor
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

                self?.zoomScale = newScale
                self?.zoomAnchor = CGPoint(x: ax, y: ay)

                try? await Task.sleep(nanoseconds: stepNS)
            }
            if !Task.isCancelled {
                self?.zoomScale = target
                self?.zoomAnchor = targetAnchor
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
    /// User toggle: draw NudeNet detection boxes + class labels on top
    /// of the image. Hidden by default so most users don't see the raw
    /// detector output.
    @Published var showNudityLabels: Bool = false
    /// User toggle: render the per-subject PAD bar stack (V/A/D)
    /// under each head badge. On by default — turning it off keeps
    /// the head badge but drops the meter capsule for a cleaner
    /// composition when emotion detail isn't needed.
    @Published var showPADMeter: Bool = true
    /// User toggle: render the per-subject PSPI pain meter under the
    /// head badge. On by default once the model is installed; hidden
    /// entirely when OpenGraphAU isn't downloaded.
    @Published var showPainMeter: Bool = true
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
    /// Install state for the NSFW fallback model. Reuses DepthInstallState —
    /// the shape (notInstalled / downloading / installed / failed) is generic.
    @Published var nsfwInstall: DepthInstallState = .notInstalled
    /// Install state for the NudeNet per-subject detector.
    @Published var nudenetInstall: DepthInstallState = .notInstalled
    /// Install state for the CLIP image encoder + prompt-embeddings bundle.
    @Published var clipInstall: DepthInstallState = .notInstalled
    /// Install state for the FER+ facial-emotion classifier.
    @Published var emotionInstall: DepthInstallState = .notInstalled
    /// Install state for the OpenGraphAU facial Action Unit detector
    /// (pain / PSPI proxy). Same state shape as the other optional
    /// model tiers.
    @Published var openGraphAUInstall: DepthInstallState = .notInstalled
    @Published var isAnalyzing: Bool = false
    @Published var errorMessage: String?
    @Published var depthAvailable: Bool = false
    @Published var depthInstall: DepthInstallState = .notInstalled

    let analyzer: FocusAnalyzer
    private var currentTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var nsfwInstallTask: Task<Void, Never>?
    private var nudenetInstallTask: Task<Void, Never>?
    private var clipInstallTask: Task<Void, Never>?
    private var emotionInstallTask: Task<Void, Never>?
    private var openGraphAUInstallTask: Task<Void, Never>?

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
        self.depthAvailable = depthInstalled
        self.depthInstall = depthInstalled ? .installed : .notInstalled
        self.nsfwInstall = nsfwInstalled ? .installed : .notInstalled
        self.nudenetInstall = nudenetInstalled ? .installed : .notInstalled
        self.clipInstall = clipInstalled ? .installed : .notInstalled
        self.emotionInstall = emotionInstalled ? .installed : .notInstalled
        self.openGraphAUInstall = openGraphAUInstalled ? .installed : .notInstalled

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

    /// Download the NSFW fallback model. Mirrors downloadDepthModel so the
    /// UI layer can reuse the same install-state rendering.
    func downloadNSFWModel() {
        guard nsfwInstallTask == nil else { return }
        nsfwInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        nsfwInstallTask = Task { [weak self] in
            do {
                try await analyzer.installNSFWModel { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.nsfwInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.nsfwInstall = .installed
                    // Re-query SCA availability so the row reflects that
                    // the NSFW fallback is now usable.
                    self?.refreshSensitiveContentAvailability()
                }
            } catch {
                // Re-downloads can fail (404, network) without touching
                // the existing on-disk install — the installer wipes
                // the destination only after the new content has been
                // unpacked. Preserve `.installed` when the file's
                // still there so the UI doesn't surface a misleading
                // failure for a model that's actually working.
                let stillInstalled = ModelArchive.nsfw.isInstalled()
                await MainActor.run { [weak self] in
                    self?.nsfwInstall = stillInstalled
                        ? .installed
                        : .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.nsfwInstallTask = nil }
        }
    }

    /// Download the FER+ emotion classifier. Same install-state pattern
    /// as the other optional-model install rows.
    func downloadEmotionModel() {
        guard emotionInstallTask == nil else { return }
        emotionInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        emotionInstallTask = Task { [weak self] in
            do {
                try await analyzer.installEmotionModel { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.emotionInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.emotionInstall = .installed
                }
            } catch {
                let stillInstalled = ModelArchive.emotion.isInstalled()
                await MainActor.run { [weak self] in
                    self?.emotionInstall = stillInstalled
                        ? .installed
                        : .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.emotionInstallTask = nil }
        }
    }

    /// Download the OpenGraphAU pain detector. Same install-state
    /// pattern as the other optional-model install rows.
    func downloadOpenGraphAUModel() {
        guard openGraphAUInstallTask == nil else { return }
        openGraphAUInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        openGraphAUInstallTask = Task { [weak self] in
            do {
                try await analyzer.installOpenGraphAUModel { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.openGraphAUInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.openGraphAUInstall = .installed
                }
            } catch {
                let stillInstalled = ModelArchive.openGraphAU.isInstalled()
                await MainActor.run { [weak self] in
                    self?.openGraphAUInstall = stillInstalled
                        ? .installed
                        : .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.openGraphAUInstallTask = nil }
        }
    }

    /// Download the CLIP bundle (image encoder + prompt embeddings).
    /// Mirrors the other model-install flows so the UI can reuse the
    /// same install-state row.
    func downloadCLIPModel() {
        guard clipInstallTask == nil else { return }
        clipInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        clipInstallTask = Task { [weak self] in
            do {
                try await analyzer.installCLIPModel { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.clipInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.clipInstall = .installed
                }
            } catch {
                let stillInstalled = ModelArchive.clip.isInstalled()
                await MainActor.run { [weak self] in
                    self?.clipInstall = stillInstalled
                        ? .installed
                        : .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.clipInstallTask = nil }
        }
    }

    /// Download the NudeNet per-subject detector. Mirrors the NSFW / depth
    /// download flow so the UI layer can reuse the same install-state row.
    func downloadNudeNetModel() {
        guard nudenetInstallTask == nil else { return }
        nudenetInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        nudenetInstallTask = Task { [weak self] in
            do {
                try await analyzer.installNudeNetModel { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.nudenetInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.nudenetInstall = .installed
                }
            } catch {
                let stillInstalled = ModelArchive.nudenet.isInstalled()
                await MainActor.run { [weak self] in
                    self?.nudenetInstall = stillInstalled
                        ? .installed
                        : .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.nudenetInstallTask = nil }
        }
    }

    func downloadDepthModel() {
        guard installTask == nil else { return }
        depthInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        installTask = Task { [weak self] in
            do {
                try await analyzer.installDepthModel { [weak self] p in
                    // Progress callback may arrive on any thread — hop to main.
                    Task { @MainActor [weak self] in
                        self?.depthInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run { [weak self] in
                    self?.depthAvailable = true
                    self?.depthInstall = .installed
                }
            } catch {
                let stillInstalled = ModelArchive.depthAnything.isInstalled()
                await MainActor.run { [weak self] in
                    if stillInstalled {
                        self?.depthAvailable = true
                        self?.depthInstall = .installed
                    } else {
                        self?.depthInstall = .failed(error.localizedDescription)
                    }
                }
            }
            await MainActor.run { [weak self] in self?.installTask = nil }
        }
    }

    func load(url: URL, name: String) {
        print("[ViewModel] load url=\(url.path) name=\(name)")
        currentTask?.cancel()
        isAnalyzing = true
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
                let overlays = try await analyzer.analyze(mode: mode)
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
                    self?.isAnalyzing = false
                    print("[ViewModel] sourceImage set, isAnalyzing=false")
                }
            } catch is CancellationError {
                print("[ViewModel] load cancelled")
            } catch {
                print("[ViewModel] load FAIL: \(error)")
                await MainActor.run { [weak self] in
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                }
            }
        }
    }

    func clear() {
        currentTask?.cancel()
        currentTask = nil
        zoomAnimationTask?.cancel()
        zoomAnimationTask = nil
        sourceImage = nil
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
        exposureInfo = nil
        errorMessage = nil
        isAnalyzing = false
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
        let mode = self.mode
        let analyzer = self.analyzer
        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let overlays = try await analyzer.analyze(mode: mode)
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
                    self?.isAnalyzing = false
                }
            } catch is CancellationError {
                // Swallow — a newer analyze is in flight.
            } catch {
                await MainActor.run { [weak self] in
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                }
            }
        }
    }
}
