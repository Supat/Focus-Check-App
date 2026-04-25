import Metal
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import Vision
import simd

/// An oriented black-bar redaction over a detected pair of eyes. Center +
/// size + tilt angle (radians) — lets the renderer rotate the bar to follow
/// a tilted head instead of drawing a jarring axis-aligned strip.
struct EyeBar: Equatable {
    var center: CGPoint
    var size: CGSize
    var angleRadians: CGFloat
}

/// Owns all heavyweight GPU/ML resources and serializes access through actor isolation.
/// CLAUDE.md rule: MPS / CIContext / MLModel calls live on this actor.
actor FocusAnalyzer {
    struct Overlays {
        var sharpness: CIImage?
        var depth: CIImage?
        /// Median depth value over the high-sharpness pixels. `nil` unless hybrid
        /// analysis was requested and both signals were produced. Used by the
        /// Focus Error renderer to classify out-of-focus pixels as too-close vs.
        /// too-far relative to this scalar focal plane.
        var focalPlane: Float?
        /// Global motion-blur signature from FFT analysis. `nil` if detection
        /// failed or the image is sharp / isotropically blurred.
        var motionBlur: MotionBlurReport?
        /// Per-patch motion-blur confidence grid upscaled to source extent.
        /// Used by the `.motion` overlay style to highlight *where* motion
        /// blur is concentrated rather than just reporting a global scalar.
        var motionOverlay: CIImage?
        /// Apple's Sensitive Content Analysis result. nil when the framework
        /// is unavailable or Communication Safety is off in Screen Time.
        var isSensitive: Bool?
        /// Top class label from whichever backend answered ("Nudity" from
        /// SCA, or a class name like "NSFW" from the fallback model).
        var sensitiveLabel: String?
        /// Top class probability in [0, 1]. Only populated by the NSFW
        /// fallback; SCA doesn't expose a confidence number.
        var sensitiveConfidence: Float?
        /// Face bounding boxes in source-extent coordinates (CIImage Y-up).
        /// Empty array when no faces detected. Used by the renderer to mosaic
        /// only face regions when sensitive content is flagged.
        var faceRectangles: [CGRect] = []
        /// Full-body bounding boxes from VNDetectHumanRectanglesRequest, also
        /// in source-extent coordinates. Used for the .body mosaic mode.
        var bodyRectangles: [CGRect] = []
        /// Groin rectangles derived from body-pose hip joints, in source-
        /// extent coordinates. Used for the .groin mosaic mode.
        var groinRectangles: [CGRect] = []
        /// Oriented eye bars (center + size + tilt angle) derived from face-
        /// landmark eye points, in source-extent coordinates. Used for the
        /// Eyes black-bar mode and the Tabloid combined mode.
        var eyeBars: [EyeBar] = []
        /// Upper-torso rectangles derived from body-pose shoulder + hip
        /// joints, in source-extent coordinates. Used for the .chest mode.
        var chestRectangles: [CGRect] = []
        /// Person-silhouette mask from Vision's segmentation request, already
        /// stretched to the source image's extent so downstream compositing
        /// can use it directly as a blendWithMask input. White = person,
        /// black = background. nil when no person was detected or Vision
        /// declined to run segmentation.
        var personMask: CIImage?
        /// Per-body nudity level — same order as `bodyRectangles`. Empty
        /// when the NudeNet model isn't installed. Callers distinguish
        /// "model absent" (empty) from "model says none" (.none entries).
        var nudityLevels: [NudityLevel] = []
        /// Per-body inferred gender from NudeNet's FACE_* branch, same
        /// order / same empty-vs-`.unknown` semantics as `nudityLevels`.
        var nudityGenders: [SubjectGender] = []
        /// Raw per-part detections in source-extent coords. Used only by
        /// the debug label overlay; compositing paths use the aggregated
        /// `nudityLevels` instead.
        var nudityDetections: [NudityDetection] = []
        /// CLIP zero-shot context matches, sorted highest-similarity
        /// first. Empty when the CLIP bundle isn't installed.
        var clipMatches: [CLIPMatch] = []
        /// Per-face emotion predictions, indexed alongside
        /// `faceRectangles`. `nil` entries mean the classifier
        /// didn't meet its confidence floor for that face. Empty
        /// when the FER+ model isn't installed.
        var faceEmotions: [EmotionPrediction?] = []
        /// Per-face PSPI-proxy pain score from the OpenGraphAU
        /// detector + Vision-derived AU43. Same indexing + empty
        /// contract as `faceEmotions`. Read by the UI to render a
        /// per-subject pain badge under the head stack.
        var painScores: [PainScore?] = []
        /// Per-face age prediction from SSR-Net, indexed alongside
        /// `faceRectangles`. `nil` entries for faces the model
        /// couldn't crop; empty array when the age model isn't
        /// installed. Age-only — gender comes from `nudityGenders`
        /// (NudeNet's FACE_* branch) exclusively now.
        var ageEstimations: [AgePrediction?] = []
        /// NIMA technical-quality score for the whole image. `nil`
        /// when the model isn't installed. Scalar in [1, 10] with
        /// an uncertainty band from the softmax stdev.
        var quality: QualityScore?
        /// NIMA aesthetic-quality score for the whole image — same
        /// scalar + distribution shape as `quality`, different
        /// training set (AVA instead of TID2013).
        var aesthetic: QualityScore?
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let laplacian: LaplacianVariance
    private let motionBlur: MotionBlurDetector
    private let sensitiveContent = SensitiveContentChecker()
    private let nudityDetector = NudityDetector()
    private let clipScorer = CLIPScorer()
    private let emotionClassifier = EmotionClassifier()
    private let painDetector = PainDetector()
    private let ageEstimator = AgeEstimator()
    private let qualityAnalyzer = QualityAnalyzer()
    private let aestheticAnalyzer = AestheticAnalyzer()
    private let depthInstaller = ModelArchiveInstaller(.depthAnything)
    private let nsfwInstaller = ModelArchiveInstaller(.nsfw)
    private let nudenetInstaller = ModelArchiveInstaller(.nudenet)
    private let clipInstaller = ModelArchiveInstaller(.clip)
    private let emotionInstaller = ModelArchiveInstaller(.emotion)
    private let openGraphAUInstaller = ModelArchiveInstaller(.openGraphAU)
    private let ageInstaller = ModelArchiveInstaller(.age)
    private let qualityInstaller = ModelArchiveInstaller(.quality)
    private let aestheticInstaller = ModelArchiveInstaller(.aesthetic)

    private var source: CIImage?

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal is required.")
        }
        self.device = device
        self.commandQueue = queue

        self.ciContext = CIContext(mtlDevice: device, options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!,
            .workingFormat: CIFormat.RGBAh,
            .cacheIntermediates: false
        ])
        self.laplacian = LaplacianVariance(device: device, commandQueue: queue)
        self.motionBlur = MotionBlurDetector(ciContext: self.ciContext)
        // DepthEstimator and NudityClassifier eager-load their Core ML
        // models (50 MB+ each, ANE compile ~1–3 s). Skipping that at
        // init keeps app launch under Swift Playgrounds' 5-second
        // preview budget — both are loaded on first use instead.
    }

    var isDepthAvailable: Bool { ModelArchive.depthAnything.isInstalled() }

    /// Lazy-loaded depth estimator. Nil until either (a) the caller has
    /// invoked an analysis mode that needs depth, or (b) the model
    /// finishes downloading via `installDepthModel`. Returning nil here
    /// when the model isn't present lets `.depth` / `.hybrid` modes
    /// gracefully downgrade.
    private func depthEstimator() -> DepthEstimator? {
        if let cached = _depthEstimator { return cached }
        guard let created = try? DepthEstimator() else { return nil }
        _depthEstimator = created
        return created
    }
    private var _depthEstimator: DepthEstimator?

    /// Download + install the Depth Anything v2 `.mlmodelc` from the maintainer's
    /// release URL, then refresh the estimator so Depth/Hybrid modes become usable.
    /// Progress (0...1) is reported via the callback — may run on any thread.
    func installDepthModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await depthInstaller.install(progress: progress)
        // Eagerly load the freshly-installed model so the first analysis
        // after download doesn't pay the compile cost on the user's tap.
        _depthEstimator = try DepthEstimator()
    }

    /// Render the composite for `inputs` at source resolution via the
    /// analyzer's CIContext, encode as PNG, and write to `destination`.
    /// Uses sRGB for the output so the file looks right on generic image
    /// viewers — extended Display P3 is the working space but PNGs in the
    /// wild are overwhelmingly sRGB.
    func exportPNG(inputs: FocusCompositeInputs, destination: URL) throws {
        let composite = FocusRenderer.composite(
            inputs,
            drawableSize: inputs.source.extent.size,
            overlayHidden: false,
            zoomScale: 1,
            zoomAnchor: CGPoint(x: 0.5, y: 0.5)
        )
        let cropped = composite.cropped(to: inputs.source.extent)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let cgImage = ciContext.createCGImage(
            cropped, from: cropped.extent,
            format: .RGBA8,
            colorSpace: colorSpace
        ) else {
            throw AnalysisError.coreImageFailure
        }

        guard let dest = CGImageDestinationCreateWithURL(
            destination as CFURL,
            "public.png" as CFString,
            1, nil
        ) else {
            throw AnalysisError.coreImageFailure
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw AnalysisError.coreImageFailure
        }
    }

    /// Decode a URL into the working CIImage. RAW goes through `CIRAWFilter`; everything else
    /// loads via `CIImage(contentsOf:)` which respects EXIF orientation.
    func loadImage(from url: URL) throws -> CIImage {
        let image: CIImage
        if isRAW(url: url) {
            guard let raw = CIRAWFilter(imageURL: url),
                  let output = raw.outputImage else {
                throw AnalysisError.imageDecodeFailed
            }
            image = output
        } else {
            // CIImage(contentsOf:) passes through CGImageSource which reads EXIF orientation;
            // we normalize here so downstream transforms operate in natural orientation.
            let opts: [CIImageOption: Any] = [.applyOrientationProperty: true]
            guard let decoded = CIImage(contentsOf: url, options: opts) else {
                throw AnalysisError.imageDecodeFailed
            }
            image = decoded
        }
        self.source = image
        // Invalidate the non-mode cache — Vision / NudeNet / motion
        // blur / sensitive content are all bound to this specific
        // source. analyze() will repopulate on the next call.
        cachedNonMode = nil
        return image
    }

    /// Live availability — Communication Safety can be toggled at runtime so
    /// callers should query this each time they want to know the state.
    var sensitiveContentAvailability: SensitiveContentAvailability {
        sensitiveContent.availability
    }

    /// True when the NSFW fallback Core ML model is already installed.
    var isNSFWModelInstalled: Bool { ModelArchive.nsfw.isInstalled() }

    /// Download + install the NSFW fallback model, mirroring installDepthModel.
    func installNSFWModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await nsfwInstaller.install(progress: progress)
    }

    /// True when the NudeNet per-subject detector is installed + loaded.
    var isNudeNetInstalled: Bool { ModelArchive.nudenet.isInstalled() }

    /// Download + install NudeNet, same pattern as depth / NSFW. The wrapper
    /// initializes lazily on first `detect` call, so no explicit reload.
    func installNudeNetModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await nudenetInstaller.install(progress: progress)
    }

    /// True when the CLIP bundle (image encoder + prompt embeddings) is
    /// installed on disk.
    var isCLIPInstalled: Bool { ModelArchive.clip.isInstalled() }

    /// Download + install the CLIP image-encoder + prompt-embeddings
    /// bundle. Uses the same runtime-download pattern as the other
    /// models; see `ModelArchive.clip` for the expected ZIP layout.
    func installCLIPModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await clipInstaller.install(progress: progress)
    }

    /// True when the FER+ emotion classifier is installed on disk.
    var isEmotionInstalled: Bool { ModelArchive.emotion.isInstalled() }

    /// Download + install the FER+ emotion classifier. Same pattern as
    /// the other models — lazy MLModel load happens on first classify.
    func installEmotionModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await emotionInstaller.install(progress: progress)
    }

    /// True when the OpenGraphAU pain detector is installed on disk.
    var isOpenGraphAUInstalled: Bool { ModelArchive.openGraphAU.isInstalled() }

    /// Download + install the OpenGraphAU multi-label AU classifier.
    /// Paired with a Vision-derived AU43 proxy on the Swift side to
    /// produce a PSPI pain estimate per face.
    func installOpenGraphAUModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await openGraphAUInstaller.install(progress: progress)
    }

    /// True when the SSR-Net age estimator is installed on disk.
    var isAgeInstalled: Bool { ModelArchive.age.isInstalled() }

    /// Download + install the SSR-Net age model. Lazy MLModel load
    /// happens on first predict; no explicit reload needed.
    func installAgeModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await ageInstaller.install(progress: progress)
    }

    /// True when the NIMA quality analyzer is installed on disk.
    var isQualityInstalled: Bool { ModelArchive.quality.isInstalled() }

    /// Download + install NIMA. Lazy MLModel load happens on first
    /// analyze; no explicit reload needed.
    func installQualityModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await qualityInstaller.install(progress: progress)
    }

    /// True when the NIMA aesthetic analyzer is installed on disk.
    var isAestheticInstalled: Bool { ModelArchive.aesthetic.isInstalled() }

    /// Download + install the NIMA aesthetic variant.
    func installAestheticModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await aestheticInstaller.install(progress: progress)
    }

    /// Eagerly compile the installed Core ML models so the first analyze
    /// after launch doesn't pay the ~1–3 s compile cost. Safe to call
    /// from a background task once the app is idle. No-op for models
    /// that aren't on disk.
    func prewarmModels() {
        _ = depthEstimator()
        _ = nudityDetector.warm()
        _ = clipScorer.warm()
        _ = emotionClassifier.warm()
        _ = painDetector.warm()
        _ = ageEstimator.warm()
        _ = qualityAnalyzer.warm()
        _ = aestheticAnalyzer.warm()
    }

    /// Walk `Application Support/` and print every file + byte size.
    /// Intended for diagnosing install issues on iPad Swift Playgrounds
    /// where there's no Finder — output lands in the in-app console
    /// (toggle with the text-bubble icon at the bottom of the Run view).
    /// Resolves `/private` symlinks so relative paths print cleanly.
    func dumpInstallDirectory() {
        let fm = FileManager.default
        guard let appSupport = try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false
        ) else {
            print("[Install] Application Support unavailable")
            return
        }
        let rootPath = appSupport.resolvingSymlinksInPath().path
        print("[Install] \(rootPath)")
        guard let enumerator = fm.enumerator(
            at: appSupport,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey]
        ) else {
            print("[Install] (empty or unreadable)")
            return
        }
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
            let isDir = values?.isDirectory ?? false
            let size = values?.fileSize ?? 0
            let resolved = url.resolvingSymlinksInPath().path
            let relative = resolved.hasPrefix(rootPath + "/")
                ? String(resolved.dropFirst(rootPath.count + 1))
                : resolved
            if isDir {
                print("[Install]   \(relative)/")
            } else {
                print("[Install]   \(relative) (\(size) bytes)")
            }
        }
    }

    /// Everything analyze() computes that doesn't depend on the chosen
    /// AnalysisMode. Cached on first full run so mode switches
    /// (Sharpness ↔ Depth ↔ Hybrid) only recompute the sharpness /
    /// depth stage instead of redoing Vision, NudeNet, motion blur,
    /// and the sensitive-content classifier.
    private struct NonModeResults {
        var motion: MotionBlurReport?
        var motionOverlay: CIImage?
        var sensitive: SensitiveContentResult?
        var vision: VisionResults
        var nudity: NudityAnalysis
        var clipMatches: [CLIPMatch]
        var faceEmotions: [EmotionPrediction?]
        var painScores: [PainScore?]
        var ageEstimations: [AgePrediction?]
        var quality: QualityScore?
        var aesthetic: QualityScore?
    }
    private var cachedNonMode: NonModeResults?

    /// Run the analysis pipeline for the given mode. Returns display-ready overlay images
    /// already upscaled to the source's extent. `progress` (optional)
    /// reports fractional completion [0, 1] with a short label naming
    /// the stage about to run — fires before each stage so the UI
    /// label matches what the bar is about to wait on. Fixed-weight
    /// per stage: unequal real-world timing produces some jump, but
    /// the total is bounded.
    func analyze(mode: AnalysisMode,
                 progress: (@Sendable (Double, String?) -> Void)? = nil) async throws -> Overlays {
        guard let source else { throw AnalysisError.imageDecodeFailed }
        try Task.checkCancellation()

        // Step accounting — skips the non-mode stages entirely when
        // the cache hits (mode-only reanalyze).
        let needsNonMode = (cachedNonMode == nil)
        let nonModeSteps = needsNonMode ? 10 : 0
        let totalSteps = nonModeSteps + 1
        var stepsDone = 0
        func tick(_ label: String) {
            progress?(Double(stepsDone) / Double(totalSteps), label)
            stepsDone += 1
        }

        // Non-mode-dependent pipeline — cached across mode changes on the
        // same source so `reanalyze()` short-circuits to just sharpness /
        // depth. Invalidated on every `loadImage`.
        let nonMode: NonModeResults
        if let cached = cachedNonMode {
            nonMode = cached
        } else {
            let computed = await computeNonModeResults(source: source, tick: tick)
            cachedNonMode = computed
            nonMode = computed
        }
        try Task.checkCancellation()

        let modeLabel: String
        switch mode {
        case .sharpness: modeLabel = "Computing sharpness"
        case .depth:     modeLabel = "Estimating depth"
        case .hybrid:    modeLabel = "Computing sharpness + depth"
        }
        tick(modeLabel)
        let modeResult = try computeModeStage(mode: mode, source: source)
        let sharpness = modeResult.sharpness
        let depth = modeResult.depth
        let focalPlane = modeResult.focalPlane
        progress?(1.0, nil)

        let vision = nonMode.vision
        let nudity = nonMode.nudity
        let nudityLevels = nudity.levels
        let nudityGenders = nudity.genders
        let nudityDetections = nudity.detections
        let motionReport = nonMode.motion
        let motionOverlay = nonMode.motionOverlay
        let isSensitive = nonMode.sensitive?.isSensitive
        let sensitiveLabel = nonMode.sensitive?.topLabel
        let sensitiveConfidence = nonMode.sensitive?.confidence
        let clipMatches = nonMode.clipMatches
        let faceEmotions = nonMode.faceEmotions
        let painScores = nonMode.painScores
        let ageEstimations = nonMode.ageEstimations
        let quality = nonMode.quality
        let aesthetic = nonMode.aesthetic

        return Overlays(
            sharpness: sharpness,
            depth: depth,
            focalPlane: focalPlane,
            motionBlur: motionReport,
            motionOverlay: motionOverlay,
            isSensitive: isSensitive,
            sensitiveLabel: sensitiveLabel,
            sensitiveConfidence: sensitiveConfidence,
            faceRectangles: vision.faces,
            bodyRectangles: vision.bodies,
            groinRectangles: vision.groins,
            eyeBars: vision.eyes,
            chestRectangles: vision.chests,
            personMask: vision.personMask,
            nudityLevels: nudityLevels,
            nudityGenders: nudityGenders,
            nudityDetections: nudityDetections,
            clipMatches: clipMatches,
            faceEmotions: faceEmotions,
            painScores: painScores,
            ageEstimations: ageEstimations,
            quality: quality,
            aesthetic: aesthetic
        )
    }

    /// Run every analyze stage that doesn't depend on the chosen
    /// AnalysisMode. Result is cached on `self` so subsequent
    /// reanalyze() calls only repeat the sharpness / depth stage.
    /// Sensitive-content classification is launched first so its
    /// 100–300 ms latency overlaps with the rest of the pipeline.
    private func computeNonModeResults(
        source: CIImage,
        tick: (String) -> Void
    ) async -> NonModeResults {
        async let sensitiveFuture = sensitiveContent.check(image: source, ciContext: ciContext)

        tick("Detecting motion blur")
        let motionReport = motionBlur.detect(in: source)
        let motionOverlay = motionBlur.detectMap(in: source).flatMap {
            upscale(image: $0, toExtentOf: source)
        }
        tick("Running Vision")
        let vision = runVision(in: source)
        tick("Classifying nudity")
        let nudity = nudityDetector.analyze(
            image: source, bodies: vision.bodies, ciContext: ciContext
        )
        tick("Scoring context")
        // Zero-shot context scoring via CLIP. Returns [] when the
        // model archive isn't installed so downstream UI just hides
        // the Context badge.
        let clipMatches = clipScorer.score(image: source, ciContext: ciContext)
        tick("Classifying emotions")
        // Per-face emotion classification via FER+. Same caching +
        // empty-when-missing contract as the other optional tiers.
        let faceEmotions = emotionClassifier.classify(
            faces: vision.faces, rolls: vision.faceRolls,
            in: source, ciContext: ciContext
        )
        tick("Estimating pain")
        // Per-face PSPI via OpenGraphAU + Vision-derived AU43.
        let painScores = painDetector.classify(
            faces: vision.faces, rolls: vision.faceRolls,
            eyeOpenness: vision.faceEyeOpenness,
            in: source, ciContext: ciContext
        )
        tick("Estimating age")
        // Per-face age via SSR-Net.
        let ageEstimations = ageEstimator.estimate(
            faces: vision.faces, rolls: vision.faceRolls,
            in: source, ciContext: ciContext
        )
        tick("Judging technical quality")
        let quality = qualityAnalyzer.analyze(image: source, ciContext: ciContext)
        tick("Judging aesthetic quality")
        let aesthetic = aestheticAnalyzer.analyze(image: source, ciContext: ciContext)
        tick("Checking sensitive content")
        let sensitive = await sensitiveFuture

        return NonModeResults(
            motion: motionReport,
            motionOverlay: motionOverlay,
            sensitive: sensitive,
            vision: vision,
            nudity: nudity,
            clipMatches: clipMatches,
            faceEmotions: faceEmotions,
            painScores: painScores,
            ageEstimations: ageEstimations,
            quality: quality,
            aesthetic: aesthetic
        )
    }

    /// Compute the mode-specific overlays (sharpness, depth, focal
    /// plane). Hybrid degrades gracefully to sharpness-only when the
    /// depth model isn't installed; pure-Depth throws so the caller
    /// can surface the missing-model error.
    private func computeModeStage(
        mode: AnalysisMode, source: CIImage
    ) throws -> (sharpness: CIImage?, depth: CIImage?, focalPlane: Float?) {
        switch mode {
        case .sharpness:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            return (upscale(texture: tex, toExtentOf: source), nil, nil)
        case .depth:
            guard let estimator = depthEstimator() else { throw AnalysisError.modelMissing }
            let map = try estimator.depthMap(for: source, ciContext: ciContext)
            return (nil, upscale(image: map, toExtentOf: source), nil)
        case .hybrid:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            let sharpness = upscale(texture: tex, toExtentOf: source)
            guard let estimator = depthEstimator() else {
                return (sharpness, nil, nil)
            }
            let map = try estimator.depthMap(for: source, ciContext: ciContext)
            let depth = upscale(image: map, toExtentOf: source)
            let focalPlane: Float? = {
                guard let s = sharpness, let d = depth else { return nil }
                return computeFocalPlane(sharpness: s, depth: d)
            }()
            return (sharpness, depth, focalPlane)
        }
    }

    /// Bundle of all Vision-derived rectangles and bars produced in one pass.
    private struct VisionResults {
        var faces: [CGRect] = []
        /// Per-face roll angle (radians, Apple's convention: positive =
        /// clockwise head tilt viewed from the front). Parallel array
        /// to `faces`. Used by `EmotionClassifier` to rotate crops back
        /// to eye-level before running EmoNet, which was trained on
        /// FAN-aligned inputs.
        var faceRolls: [CGFloat] = []
        /// Per-face eye-aspect ratio averaged across both eyes — the
        /// bbox height/width of the 6-point eye-landmark clusters. In
        /// practice lands near 0.3 for a wide-open eye and trends
        /// toward 0 as the lids close. Consumed by `PainDetector` as
        /// an AU43 (eye closure) proxy. Defaults to 0.3 (fully open)
        /// when Vision couldn't give us landmarks for that face.
        var faceEyeOpenness: [Float] = []
        var eyes: [EyeBar] = []
        var bodies: [CGRect] = []
        var groins: [CGRect] = []
        var chests: [CGRect] = []
        var personMask: CIImage?
    }

    /// Run all five Vision-based detections in a single pass. The previous
    /// approach spent 5 × `createCGImage` + 5 × `VNImageRequestHandler` per
    /// analysis — one handler with three requests shares the image decode and
    /// any internal pyramid the framework builds. Face rectangles come from
    /// the landmarks request (`VNDetectFaceLandmarksRequest` is a subclass of
    /// the rectangle request); groin + chest share one body-pose request.
    private func runVision(in image: CIImage) -> VisionResults {
        // Downscale the Vision handler input — the full source (up to
        // 50 MP) is far more than any of the requests need. Face /
        // body / pose detectors internally resample anyway, and the
        // segmentation mask at `.balanced` is already sub-resolution
        // so the blockier output from a smaller input is
        // indistinguishable downstream after upscale. `createCGImage`
        // on a 50-MP CIImage is itself one of the slower steps in the
        // pipeline — capping the long side at 1600 px turns a 400 ms
        // decode into a 30 ms one on large photos. Boxes come back
        // normalized to input size, which equals normalized to source
        // size, so denormalize against `image.extent` is still correct.
        let extent = image.extent
        let analysisInput = visionDownsample(image)
        guard let cgImage = ciContext.createCGImage(
            analysisInput, from: analysisInput.extent
        ) else {
            return VisionResults()
        }

        let faceLandmarks = VNDetectFaceLandmarksRequest()
        let bodyRects = VNDetectHumanRectanglesRequest()
        bodyRects.upperBodyOnly = false
        let bodyPose = VNDetectHumanBodyPoseRequest()
        // 3D body pose (iOS 17+) — used for body-orientation inference so
        // we can widen the groin mosaic when the subject is turned
        // sideways. Runs alongside the 2D pose request; same CGImage.
        let bodyPose3D = VNDetectHumanBodyPose3DRequest()
        // Person segmentation — produces an alpha mask that follows the
        // subject's outline so the .body mosaic can pixelate along the
        // silhouette rather than inside a loose bounding box. .balanced is
        // the speed/quality sweet spot; .accurate is much slower and the
        // extra edge fidelity isn't perceptible under pixelation.
        let personSeg = VNGeneratePersonSegmentationRequest()
        personSeg.qualityLevel = .balanced
        personSeg.outputPixelFormat = kCVPixelFormatType_OneComponent8

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([faceLandmarks, bodyRects, bodyPose, bodyPose3D, personSeg])
        } catch {
            return VisionResults()
        }

        var results = VisionResults()

        // Face rects + eye bars from the single landmarks request. Vision's
        // `.results` is already `[VNFaceObservation]?` on modern SDKs — the
        // previous `as? [VNFaceObservation]` downcast was a no-op Swift 6
        // warned about.
        for obs in faceLandmarks.results ?? [] {
            results.faces.append(Self.denormalize(obs.boundingBox, in: extent))
            results.faceRolls.append(CGFloat(obs.roll?.doubleValue ?? 0))
            results.faceEyeOpenness.append(Self.eyeOpenness(from: obs))
            if let bar = Self.eyeBar(from: obs, in: extent) {
                results.eyes.append(bar)
            }
        }

        results.bodies = Self.mergedBodies(
            rects: bodyRects.results ?? [],
            poses: bodyPose.results ?? [],
            in: extent
        )

        // Groin + chest from the shared pose observations. Pair each 2D
        // observation with the corresponding 3D one by index so the
        // sideways-expansion can use 3D hip positions when available.
        let poses3D = bodyPose3D.results ?? []
        for (index, obs) in (bodyPose.results ?? []).enumerated() {
            let pose3D = index < poses3D.count ? poses3D[index] : nil
            if let g = Self.groinRect(from: obs, pose3D: pose3D, in: extent) {
                results.groins.append(g)
            }
            if let c = Self.chestRect(from: obs, in: extent) { results.chests.append(c) }
        }

        if let buffer = personSeg.results?.first?.pixelBuffer {
            results.personMask = Self.scalePersonMask(buffer, to: extent)
        }
        return results
    }

    /// Union the body-rect detector's results with pose-derived
    /// bodies, deduplicating using IoU against the smaller box
    /// (≥ 0.3) so tight crops of one person don't register as two.
    /// VNDetectHumanRectanglesRequest frequently misses subjects in
    /// images where it *should* catch them, so the pose-derived
    /// fallback covers NudeNet's per-subject list reliably.
    private static func mergedBodies(
        rects: [VNHumanObservation],
        poses: [VNHumanBodyPoseObservation],
        in extent: CGRect
    ) -> [CGRect] {
        var bodies: [CGRect] = rects.map { Self.denormalize($0.boundingBox, in: extent) }
        for obs in poses {
            guard let rect = Self.bodyRect(from: obs, in: extent) else { continue }
            let overlapsExisting = bodies.contains { existing in
                let inter = existing.intersection(rect)
                guard !inter.isNull else { return false }
                let interArea = inter.width * inter.height
                let smallerArea = min(existing.width * existing.height,
                                      rect.width * rect.height)
                return smallerArea > 0 && interArea / smallerArea > 0.3
            }
            if !overlapsExisting {
                bodies.append(rect)
            }
        }
        return bodies
    }

    /// Scale Vision's segmentation mask up to the source extent so the
    /// compositor can feed it straight into blendWithMask. The pixel
    /// buffer is smaller than the source at .balanced quality;
    /// `samplingNearest()` preserves the block-aligned boundary
    /// instead of bilinearly smoothing it into a feathered edge,
    /// which keeps the silhouette visibly jagged.
    private static func scalePersonMask(_ buffer: CVPixelBuffer, to extent: CGRect) -> CIImage {
        let maskImage = CIImage(cvPixelBuffer: buffer).samplingNearest()
        let sx = extent.width / maskImage.extent.width
        let sy = extent.height / maskImage.extent.height
        let scaled = maskImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let placed = scaled.transformed(
            by: CGAffineTransform(translationX: extent.minX - scaled.extent.minX,
                                  y: extent.minY - scaled.extent.minY)
        )
        return placed.cropped(to: extent)
    }

    /// Derive one oriented eye bar from a face landmarks observation.
    /// Centered on the midpoint between eye centroids, rotated to the head's
    /// tilt, sized proportional to the eye-to-eye distance.
    private static func eyeBar(from obs: VNFaceObservation, in extent: CGRect) -> EyeBar? {
        guard let landmarks = obs.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye
        else { return nil }

        // Face rect in image extent coords; eye-landmark points are
        // face-bbox-normalized, so scale by the face rect to lift them.
        let fx = extent.minX + obs.boundingBox.minX * extent.width
        let fy = extent.minY + obs.boundingBox.minY * extent.height
        let fw = obs.boundingBox.width * extent.width
        let fh = obs.boundingBox.height * extent.height

        func centroid(_ region: VNFaceLandmarkRegion2D) -> CGPoint? {
            let pts = region.normalizedPoints
            guard !pts.isEmpty else { return nil }
            var sx: CGFloat = 0, sy: CGFloat = 0
            for p in pts { sx += p.x; sy += p.y }
            let n = CGFloat(pts.count)
            // Lift from face-bbox space to image extent.
            return CGPoint(x: fx + (sx / n) * fw, y: fy + (sy / n) * fh)
        }
        guard let leftC = centroid(leftEye), let rightC = centroid(rightEye) else { return nil }

        let dx = rightC.x - leftC.x
        let dy = rightC.y - leftC.y
        let eyeDistance = hypot(dx, dy)
        guard eyeDistance > 0 else { return nil }

        return EyeBar(
            center: CGPoint(x: (leftC.x + rightC.x) / 2, y: (leftC.y + rightC.y) / 2),
            size: CGSize(width: eyeDistance * 2.1, height: eyeDistance * 0.396),
            angleRadians: atan2(dy, dx)
        )
    }

    /// Eye openness proxy (~EAR): average of each eye's bbox
    /// height/width over the 6 eye landmarks. Returns 0.3 when
    /// landmarks are missing so downstream callers don't interpret a
    /// detector miss as "eyes closed". Landmark coords are
    /// face-bbox-normalized — since we're taking a ratio, no
    /// denormalization is needed.
    private static func eyeOpenness(from obs: VNFaceObservation) -> Float {
        guard let landmarks = obs.landmarks,
              let left = landmarks.leftEye,
              let right = landmarks.rightEye
        else { return 0.3 }

        func ratio(_ region: VNFaceLandmarkRegion2D) -> Float? {
            let pts = region.normalizedPoints
            guard pts.count >= 4 else { return nil }
            var minX: CGFloat = .greatestFiniteMagnitude
            var maxX: CGFloat = -.greatestFiniteMagnitude
            var minY: CGFloat = .greatestFiniteMagnitude
            var maxY: CGFloat = -.greatestFiniteMagnitude
            for p in pts {
                minX = min(minX, p.x); maxX = max(maxX, p.x)
                minY = min(minY, p.y); maxY = max(maxY, p.y)
            }
            let w = maxX - minX
            guard w > 1e-5 else { return nil }
            return Float((maxY - minY) / w)
        }

        let l = ratio(left) ?? 0.3
        let r = ratio(right) ?? 0.3
        return (l + r) / 2
    }

    /// Derive a pelvis rect from a body-pose observation's hip joints.
    /// Nil when either hip is below the confidence floor. When a 3D pose
    /// observation is supplied, uses the hip joints' depth spread to
    /// detect sideways body orientation and widen the rect — the 2D hip
    /// distance collapses toward zero as the person turns away from the
    /// camera, but the groin area is still projected into roughly the
    /// same spatial band and needs the extra cover.
    private static func groinRect(from obs: VNHumanBodyPoseObservation,
                                  pose3D: VNHumanBodyPose3DObservation?,
                                  in extent: CGRect) -> CGRect? {
        guard let leftHip = try? obs.recognizedPoint(.leftHip),
              let rightHip = try? obs.recognizedPoint(.rightHip),
              leftHip.confidence > 0.3,
              rightHip.confidence > 0.3
        else { return nil }

        let cx = (leftHip.location.x + rightHip.location.x) / 2
        let cy = (leftHip.location.y + rightHip.location.y) / 2
        let hipDistance = max(abs(rightHip.location.x - leftHip.location.x), 0.04)
        let w = hipDistance * 1.5
        let h = hipDistance * 1.0
        let yOffset = -hipDistance * 0.35
        let normalized = CGRect(
            x: cx - w / 2,
            y: cy - h / 2 + yOffset,
            width: w,
            height: h
        )
        var rect = denormalize(normalized, in: extent)

        // Sideways expansion — when the subject faces the camera, the
        // hip vector lies mostly along the image X axis; when turned
        // sideways, it rotates into the Z (depth) axis. Scale width by
        // (1 + 2 * sidewaysFactor) so a full profile gets up to 3x the
        // original width, a three-quarter view gets ~2x, and a frontal
        // pose is unchanged. Take the max of the 3D and 2D signals —
        // VNDetectHumanBodyPose3DRequest declines on many sideways
        // images where the 2D hip-to-torso ratio still captures it.
        let sideways = max(
            Self.sidewaysFactor(from: pose3D),
            Self.sidewaysFactor2D(from: obs)
        )
        if sideways > 0.01 {
            let expandedWidth = rect.width * (1 + 2 * sideways)
            let extra = expandedWidth - rect.width
            rect = CGRect(
                x: rect.minX - extra / 2,
                y: rect.minY,
                width: expandedWidth,
                height: rect.height
            )
        }

        // Portrait / tall-aspect sources can still make the denormalized
        // rect taller than wide even after the sideways expansion. Pad to
        // 3x height around the same center x so the rect always reads as
        // a horizontal strip.
        guard rect.width < rect.height else { return rect }
        let newWidth = rect.height * 3
        let extra = newWidth - rect.width
        return CGRect(
            x: rect.minX - extra / 2,
            y: rect.minY,
            width: newWidth,
            height: rect.height
        )
    }

    /// 2D fallback when the 3D pose request can't solve. Uses the ratio of
    /// projected hip width to torso height: a frontal body has hip width
    /// close to 0.5 × torso height; as the subject turns to profile both
    /// hips project to roughly the same x and the ratio collapses toward
    /// zero. Normalized so 0 = frontal (≥ the reference ratio), 1 = pure
    /// profile (zero projected hip width). Returns 0 when any of the four
    /// required landmarks are low-confidence.
    private static func sidewaysFactor2D(from obs: VNHumanBodyPoseObservation) -> CGFloat {
        guard let leftHip = try? obs.recognizedPoint(.leftHip),
              let rightHip = try? obs.recognizedPoint(.rightHip),
              let leftShoulder = try? obs.recognizedPoint(.leftShoulder),
              let rightShoulder = try? obs.recognizedPoint(.rightShoulder),
              leftHip.confidence > 0.3, rightHip.confidence > 0.3,
              leftShoulder.confidence > 0.3, rightShoulder.confidence > 0.3
        else { return 0 }

        let hipWidth = abs(rightHip.location.x - leftHip.location.x)
        let shoulderY = (leftShoulder.location.y + rightShoulder.location.y) / 2
        let hipY = (leftHip.location.y + rightHip.location.y) / 2
        let torsoHeight = abs(shoulderY - hipY)
        guard torsoHeight > 0.02 else { return 0 }

        // Reference ratio tuned empirically — frontal poses land near
        // 0.45–0.55, three-quarter views near 0.25, full profiles near 0.
        let reference: CGFloat = 0.5
        let factor = 1 - min(1, hipWidth / torsoHeight / reference)
        return max(0, factor)
    }

    /// 0 = fully frontal, 1 = fully sideways. Computed from the 3D hip
    /// positions: frontal bodies have a wide hip spread along the image
    /// X axis and ~0 along Z; sideways bodies have ~0 along X and a wide
    /// spread along Z. Returns 0 when the 3D observation or either hip
    /// joint is unavailable.
    private static func sidewaysFactor(from pose3D: VNHumanBodyPose3DObservation?) -> CGFloat {
        guard let pose3D,
              let leftHip = try? pose3D.recognizedPoint(.leftHip),
              let rightHip = try? pose3D.recognizedPoint(.rightHip)
        else { return 0 }

        // VNHumanBodyRecognizedPoint3D.position is a simd_float4x4 where
        // column 3 holds the (x, y, z) translation in Vision's 3D space.
        let leftPos = leftHip.position.columns.3
        let rightPos = rightHip.position.columns.3
        let dx = Double(rightPos.x - leftPos.x)
        let dz = Double(rightPos.z - leftPos.z)
        let magnitude = sqrt(dx * dx + dz * dz)
        guard magnitude > 0 else { return 0 }
        return CGFloat(abs(dz) / magnitude)
    }

    /// Derive a full-body bounding rect from whatever joints the pose
    /// request returned with decent confidence. Used to fill in bodies
    /// that VNDetectHumanRectanglesRequest missed. A small pad around
    /// the joint cloud covers clothing beyond the skeleton.
    private static func bodyRect(from obs: VNHumanBodyPoseObservation,
                                 in extent: CGRect) -> CGRect? {
        guard let points = try? obs.recognizedPoints(.all) else { return nil }
        let valid = points.values.filter { $0.confidence > 0.3 }
        guard valid.count >= 4 else { return nil }

        var minX: CGFloat = .infinity
        var minY: CGFloat = .infinity
        var maxX: CGFloat = -.infinity
        var maxY: CGFloat = -.infinity
        for p in valid {
            minX = min(minX, p.location.x)
            minY = min(minY, p.location.y)
            maxX = max(maxX, p.location.x)
            maxY = max(maxY, p.location.y)
        }
        let padX = (maxX - minX) * 0.15
        let padY = (maxY - minY) * 0.15
        let normalized = CGRect(
            x: minX - padX,
            y: minY - padY,
            width: (maxX - minX) + 2 * padX,
            height: (maxY - minY) + 2 * padY
        )
        return denormalize(normalized, in: extent)
    }

    /// Derive a chest rect from a body-pose observation's shoulder + hip
    /// joints. Nil when any of the four required landmarks are low-confidence.
    private static func chestRect(from obs: VNHumanBodyPoseObservation, in extent: CGRect) -> CGRect? {
        guard let leftShoulder = try? obs.recognizedPoint(.leftShoulder),
              let rightShoulder = try? obs.recognizedPoint(.rightShoulder),
              let leftHip = try? obs.recognizedPoint(.leftHip),
              let rightHip = try? obs.recognizedPoint(.rightHip),
              leftShoulder.confidence > 0.3,
              rightShoulder.confidence > 0.3,
              leftHip.confidence > 0.3,
              rightHip.confidence > 0.3
        else { return nil }

        let shoulderY = (leftShoulder.location.y + rightShoulder.location.y) / 2
        let hipY = (leftHip.location.y + rightHip.location.y) / 2
        let centerX = (leftShoulder.location.x + rightShoulder.location.x) / 2
        let shoulderSpan = max(abs(rightShoulder.location.x - leftShoulder.location.x), 0.05)
        let torsoHeight = max(shoulderY - hipY, 0.05)
        let chestHeight = torsoHeight * 0.55
        let topPad = chestHeight * 0.15
        let w = shoulderSpan * 1.15
        let normalized = CGRect(
            x: centerX - w / 2,
            y: shoulderY - chestHeight,
            width: w,
            height: chestHeight + topPad
        )
        return denormalize(normalized, in: extent)
    }

    /// Static shape-transform helper so the per-observation extractors above
    /// can live at type scope (no implicit self capture).
    private static func denormalize(_ box: CGRect, in extent: CGRect) -> CGRect {
        CGRect(
            x: extent.minX + box.minX * extent.width,
            y: extent.minY + box.minY * extent.height,
            width: box.width * extent.width,
            height: box.height * extent.height
        )
    }

    /// Estimate the focal plane depth as the median of depth values at high-sharpness pixels.
    /// Reads both signals back to the CPU at a small analysis resolution (64×64) — median
    /// on GPU requires sorting and isn't worth the complexity for a once-per-analysis scalar.
    private func computeFocalPlane(sharpness: CIImage, depth: CIImage) -> Float? {
        let side = 64
        let rect = CGRect(x: 0, y: 0, width: side, height: side)
        let linearSRGB = CGColorSpace(name: CGColorSpace.linearSRGB)!

        // Normalize both images to the same small extent so the i-th pixel in each
        // array describes the same spatial location.
        let sharpSmall = sharpness.stretched(to: CGSize(width: side, height: side))
        let depthSmall = depth.stretched(to: CGSize(width: side, height: side))

        var sharpBuf = [Float](repeating: 0, count: side * side)
        var depthBuf = [Float](repeating: 0, count: side * side)

        sharpBuf.withUnsafeMutableBytes { ptr in
            ciContext.render(sharpSmall,
                             toBitmap: ptr.baseAddress!,
                             rowBytes: side * MemoryLayout<Float>.size,
                             bounds: rect,
                             format: .Rf,
                             colorSpace: linearSRGB)
        }
        depthBuf.withUnsafeMutableBytes { ptr in
            ciContext.render(depthSmall,
                             toBitmap: ptr.baseAddress!,
                             rowBytes: side * MemoryLayout<Float>.size,
                             bounds: rect,
                             format: .Rf,
                             colorSpace: linearSRGB)
        }

        // Collect depths where sharpness is above a cutoff. The cutoff is deliberately
        // loose — we want enough textured in-focus pixels to dominate the median, not
        // only the absolute peaks (which are noisy).
        let cutoff: Float = 0.1
        var selected: [Float] = []
        selected.reserveCapacity(side * side)
        for i in 0..<(side * side) where sharpBuf[i] > cutoff {
            selected.append(depthBuf[i])
        }
        guard selected.count >= 10 else { return nil }
        selected.sort()
        return selected[selected.count / 2]
    }

    // MARK: - Helpers

    /// Cap the image's long side at 1600 px for Vision consumption.
    /// Returns the original when it's already within budget so we skip
    /// the Core Image recipe extension for typical phone-sized photos.
    private func visionDownsample(_ image: CIImage) -> CIImage {
        let longSide = max(image.extent.width, image.extent.height)
        guard longSide > 1600 else { return image }
        let scale = 1600 / longSide
        let scaled = CIFilter.lanczosScaleTransform()
        scaled.inputImage = image.translatedToOrigin()
        scaled.scale = Float(scale)
        scaled.aspectRatio = 1
        return (scaled.outputImage ?? image).cropped(
            to: CGRect(origin: .zero,
                       size: CGSize(width: image.extent.width * scale,
                                    height: image.extent.height * scale))
        )
    }

    private func upscale(texture: MTLTexture, toExtentOf source: CIImage) -> CIImage? {
        let low = CIImage(mtlTexture: texture, options: [
            .colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!
        ]) ?? CIImage.empty()
        return upscale(image: low, toExtentOf: source)
    }

    private func upscale(image: CIImage, toExtentOf source: CIImage) -> CIImage? {
        // Non-uniform stretch with Lanczos: the low-res map may come from the
        // depth estimator at the model's aspect, which differs from the
        // source's. Uniform scaling with aspectRatio=1 placed content at
        // source.origin but only the cropped region's data was stretched to
        // fill the whole extent, producing a visible offset. Stretching both
        // axes independently makes coordinates map 1:1. Lanczos (rather than
        // the `.stretched` extension's default affine) matters here because
        // the depth map has meaningful gradient detail we don't want
        // bilinearly-smoothed away.
        let sx = source.extent.width / image.extent.width
        let sy = source.extent.height / image.extent.height
        let normalized = image.translatedToOrigin()
        let upscale = CIFilter.lanczosScaleTransform()
        upscale.inputImage = normalized
        upscale.scale = Float(sy)
        // CILanczos: output = input * scale, then x-axis additionally stretched by aspectRatio.
        upscale.aspectRatio = Float(sx / sy)
        guard let stretched = upscale.outputImage else { return nil }
        return stretched
            .cropped(to: CGRect(origin: .zero, size: source.extent.size))
            .transformed(by: CGAffineTransform(translationX: source.extent.minX, y: source.extent.minY))
    }

    private func isRAW(url: URL) -> Bool {
        let rawExtensions: Set<String> = [
            "dng", "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2",
            "raf", "rw2", "orf", "pef", "srw", "x3f"
        ]
        return rawExtensions.contains(url.pathExtension.lowercased())
    }
}
