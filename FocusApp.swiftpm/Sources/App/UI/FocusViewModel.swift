import SwiftUI
import CoreImage
import Metal

enum OverlayStyle: String, CaseIterable, Identifiable {
    case peaking    = "Peaking"
    case mask       = "Mask"
    case heatmap    = "Heatmap"
    case focusError = "Error"
    case motion     = "Motion"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
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
    case eyes  = "Eyes"
    case face  = "Face"
    case chest = "Chest"
    case groin = "Groin"
    case body  = "Body"
    case whole = "Whole"
    var id: String { rawValue }
}

@MainActor
final class FocusViewModel: ObservableObject {
    // Scrubbable display state — cheap, no re-analysis.
    @Published var threshold: Float = 0.35
    @Published var overlayColor: Color = .red
    @Published var style: OverlayStyle = .peaking {
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
    @Published var zoomScale: CGFloat = 1.0
    @Published var zoomAnchor: CGPoint = CGPoint(x: 0.5, y: 0.5)

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
    @Published var eyeRectangles: [CGRect] = []
    @Published var chestRectangles: [CGRect] = []
    /// User-controlled mosaic toggle. Defaults on — protective default so
    /// sensitive content isn't displayed until the user explicitly opts in.
    @Published var mosaicEnabled: Bool = true
    @Published var mosaicMode: MosaicMode = .face
    @Published var sensitiveContentAvailability: SensitiveContentAvailability = .frameworkMissing
    /// Install state for the NSFW fallback model. Reuses DepthInstallState —
    /// the shape (notInstalled / downloading / installed / failed) is generic.
    @Published var nsfwInstall: DepthInstallState = .notInstalled
    @Published var isAnalyzing: Bool = false
    @Published var errorMessage: String?
    @Published var depthAvailable: Bool = false
    @Published var depthInstall: DepthInstallState = .notInstalled

    let analyzer: FocusAnalyzer
    private var currentTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var nsfwInstallTask: Task<Void, Never>?

    init() {
        self.analyzer = FocusAnalyzer()
        let analyzer = self.analyzer
        Task { [weak self] in
            let depth = await analyzer.isDepthAvailable
            let sensitive = await analyzer.sensitiveContentAvailability
            let nsfwInstalled = await analyzer.isNSFWModelInstalled
            await MainActor.run { [weak self] in
                self?.depthAvailable = depth
                self?.depthInstall = depth ? .installed : .notInstalled
                self?.sensitiveContentAvailability = sensitive
                self?.nsfwInstall = nsfwInstalled ? .installed : .notInstalled
            }
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
                try await analyzer.installNSFWModel { p in
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
                await MainActor.run { [weak self] in
                    self?.nsfwInstall = .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.nsfwInstallTask = nil }
        }
    }

    func downloadDepthModel() {
        guard installTask == nil else { return }
        depthInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        installTask = Task { [weak self] in
            do {
                try await analyzer.installDepthModel { p in
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
                await MainActor.run { [weak self] in
                    self?.depthInstall = .failed(error.localizedDescription)
                }
            }
            await MainActor.run { [weak self] in self?.installTask = nil }
        }
    }

    func load(url: URL, name: String) {
        currentTask?.cancel()
        isAnalyzing = true
        errorMessage = nil
        // Deliberately don't clear sharpnessOverlay / depthOverlay / focalPlane
        // / motionBlur / motionOverlay / isSensitive / faceRectangles here.
        // The old source image keeps showing until the new analysis lands,
        // so clearing the derived state would cause the old image to briefly
        // lose its mosaic / overlays while the new analysis runs.
        exposureInfo = ExposureInfo.read(from: url)
        sourceName = name
        refreshSensitiveContentAvailability()

        let mode = self.mode
        let analyzer = self.analyzer

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let image = try await analyzer.loadImage(from: url)
                try Task.checkCancellation()
                let overlays = try await analyzer.analyze(mode: mode)
                try Task.checkCancellation()
                await MainActor.run { [weak self] in
                    self?.sourceImage = image
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
                    self?.eyeRectangles = overlays.eyeRectangles
                    self?.chestRectangles = overlays.chestRectangles
                    self?.isAnalyzing = false
                }
            } catch is CancellationError {
                // Superseded by a newer load — stay silent.
            } catch {
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
        eyeRectangles = []
        chestRectangles = []
        exposureInfo = nil
        errorMessage = nil
        isAnalyzing = false
        zoomScale = 1.0
        zoomAnchor = CGPoint(x: 0.5, y: 0.5)
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
                    self?.eyeRectangles = overlays.eyeRectangles
                    self?.chestRectangles = overlays.chestRectangles
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
