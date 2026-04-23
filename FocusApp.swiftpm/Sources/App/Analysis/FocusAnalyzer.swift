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
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let laplacian: LaplacianVariance
    private let motionBlur: MotionBlurDetector
    private let sensitiveContent = SensitiveContentChecker()
    private let nudityDetector = NudityDetector()
    private let depthInstaller = ModelArchiveInstaller(.depthAnything)
    private let nsfwInstaller = ModelArchiveInstaller(.nsfw)
    private let nudenetInstaller = ModelArchiveInstaller(.nudenet)

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

    /// Eagerly compile the installed Core ML models so the first analyze
    /// after launch doesn't pay the ~1–3 s compile cost. Safe to call
    /// from a background task once the app is idle. No-op for models
    /// that aren't on disk.
    func prewarmModels() {
        _ = depthEstimator()
        _ = nudityDetector.warm()
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
    }
    private var cachedNonMode: NonModeResults?

    /// Run the analysis pipeline for the given mode. Returns display-ready overlay images
    /// already upscaled to the source's extent.
    func analyze(mode: AnalysisMode) async throws -> Overlays {
        guard let source else { throw AnalysisError.imageDecodeFailed }
        try Task.checkCancellation()

        // Non-mode-dependent pipeline — cached across mode changes on the
        // same source so `reanalyze()` short-circuits to just sharpness /
        // depth. Invalidated on every `loadImage`.
        let nonMode: NonModeResults
        if let cached = cachedNonMode {
            nonMode = cached
        } else {
            // Classifier runs on an independent compute path (SCA or the
            // NSFW MLModel); kick it off early so its 100–300 ms latency
            // overlaps with the rest of the non-mode pipeline.
            async let sensitiveFuture = sensitiveContent.check(image: source, ciContext: ciContext)

            let motionReport = motionBlur.detect(in: source)
            let motionOverlay = motionBlur.detectMap(in: source).flatMap {
                upscale(image: $0, toExtentOf: source)
            }
            let vision = runVision(in: source)
            let nudity = nudityDetector.analyze(
                image: source, bodies: vision.bodies, ciContext: ciContext
            )
            let sensitive = await sensitiveFuture

            let computed = NonModeResults(
                motion: motionReport,
                motionOverlay: motionOverlay,
                sensitive: sensitive,
                vision: vision,
                nudity: nudity
            )
            cachedNonMode = computed
            nonMode = computed
        }
        try Task.checkCancellation()

        var sharpness: CIImage?
        var depth: CIImage?
        var focalPlane: Float?

        switch mode {
        case .sharpness:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            sharpness = upscale(texture: tex, toExtentOf: source)

        case .depth:
            guard let estimator = depthEstimator() else { throw AnalysisError.modelMissing }
            let map = try estimator.depthMap(for: source, ciContext: ciContext)
            depth = upscale(image: map, toExtentOf: source)

        case .hybrid:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            sharpness = upscale(texture: tex, toExtentOf: source)
            if let estimator = depthEstimator() {
                let map = try estimator.depthMap(for: source, ciContext: ciContext)
                depth = upscale(image: map, toExtentOf: source)
                if let s = sharpness, let d = depth {
                    focalPlane = computeFocalPlane(sharpness: s, depth: d)
                }
            }
        }

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
            nudityDetections: nudityDetections
        )
    }

    /// Bundle of all Vision-derived rectangles and bars produced in one pass.
    private struct VisionResults {
        var faces: [CGRect] = []
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
            if let bar = Self.eyeBar(from: obs, in: extent) {
                results.eyes.append(bar)
            }
        }

        // Full-body rectangles. VNDetectHumanRectanglesRequest frequently
        // misses subjects in images where it *should* catch them — we
        // augment below using pose-derived bounding boxes so NudeNet
        // has a proper per-subject list.
        for obs in bodyRects.results ?? [] {
            results.bodies.append(Self.denormalize(obs.boundingBox, in: extent))
        }
        // Merge in pose-derived bodies that don't overlap an existing
        // rect — catches subjects the body-rect request dropped. Dedup
        // uses IoU against the smaller box (≥ 0.3) so tight crops of one
        // person don't register as two.
        for obs in bodyPose.results ?? [] {
            guard let rect = Self.bodyRect(from: obs, in: extent) else { continue }
            let overlapsExisting = results.bodies.contains { existing in
                let inter = existing.intersection(rect)
                guard !inter.isNull else { return false }
                let interArea = inter.width * inter.height
                let smallerArea = min(existing.width * existing.height,
                                      rect.width * rect.height)
                return smallerArea > 0 && interArea / smallerArea > 0.3
            }
            if !overlapsExisting {
                results.bodies.append(rect)
            }
        }

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

        // Scale the mask up to the source extent so the compositor can feed
        // it straight into blendWithMask. The pixel buffer is smaller than
        // the source at .balanced quality; `samplingNearest()` preserves
        // the block-aligned boundary instead of bilinearly smoothing it
        // into a feathered edge, which keeps the silhouette visibly jagged.
        if let mask = personSeg.results?.first?.pixelBuffer {
            let maskImage = CIImage(cvPixelBuffer: mask).samplingNearest()
            let sx = extent.width / maskImage.extent.width
            let sy = extent.height / maskImage.extent.height
            let scaled = maskImage.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            let placed = scaled.transformed(
                by: CGAffineTransform(translationX: extent.minX - scaled.extent.minX,
                                      y: extent.minY - scaled.extent.minY)
            )
            results.personMask = placed.cropped(to: extent)
        }
        return results
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
