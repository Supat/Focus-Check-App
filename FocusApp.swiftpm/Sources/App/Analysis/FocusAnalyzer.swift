import Metal
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import Vision

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
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let laplacian: LaplacianVariance
    private let motionBlur: MotionBlurDetector
    private let sensitiveContent = SensitiveContentChecker()
    private var depthEstimator: DepthEstimator?
    private let downloader = DepthModelDownloader()
    private let nsfwDownloader = NSFWModelDownloader()

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
        self.depthEstimator = try? DepthEstimator()
    }

    var isDepthAvailable: Bool { depthEstimator != nil }

    /// Download + install the Depth Anything v2 `.mlmodelc` from the maintainer's
    /// release URL, then refresh the estimator so Depth/Hybrid modes become usable.
    /// Progress (0...1) is reported via the callback — may run on any thread.
    func installDepthModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await downloader.install(progress: progress)
        depthEstimator = try DepthEstimator()
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
        return image
    }

    /// Live availability — Communication Safety can be toggled at runtime so
    /// callers should query this each time they want to know the state.
    var sensitiveContentAvailability: SensitiveContentAvailability {
        sensitiveContent.availability
    }

    /// True when the NSFW fallback Core ML model is already installed.
    var isNSFWModelInstalled: Bool { NSFWModelDownloader.isInstalled() }

    /// Download + install the NSFW fallback model, mirroring installDepthModel.
    func installNSFWModel(progress: @Sendable @escaping (Double) -> Void) async throws {
        try await nsfwDownloader.install(progress: progress)
    }

    /// Run the analysis pipeline for the given mode. Returns display-ready overlay images
    /// already upscaled to the source's extent.
    func analyze(mode: AnalysisMode) async throws -> Overlays {
        guard let source else { throw AnalysisError.imageDecodeFailed }
        try Task.checkCancellation()

        var sharpness: CIImage?
        var depth: CIImage?
        var focalPlane: Float?
        // Motion blur is cheap (~10 ms) and independent of the selected mode —
        // run it every time so the info badge stays accurate after mode changes.
        let motionReport = motionBlur.detect(in: source)

        // Per-patch motion blur map (~100 ms). Upscaled so downstream compositing
        // treats it the same as sharpness/depth overlays.
        let motionOverlay = motionBlur.detectMap(in: source).flatMap {
            upscale(image: $0, toExtentOf: source)
        }

        // Sensitive-content classification — binary + top class label.
        // Off-main async; returns nil if the stack couldn't answer.
        let sensitiveResult = await sensitiveContent.check(image: source, ciContext: ciContext)
        let isSensitive = sensitiveResult?.isSensitive
        let sensitiveLabel = sensitiveResult?.topLabel
        let sensitiveConfidence = sensitiveResult?.confidence

        // Face / body / pose detection — consolidated into a single Vision
        // pass so the source only gets decoded to a CGImage once and the
        // three requests share the handler's image pyramid.
        let vision = runVision(in: source)

        switch mode {
        case .sharpness:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            sharpness = upscale(texture: tex, toExtentOf: source)

        case .depth:
            guard let estimator = depthEstimator else { throw AnalysisError.modelMissing }
            let map = try estimator.depthMap(for: source, ciContext: ciContext)
            depth = upscale(image: map, toExtentOf: source)

        case .hybrid:
            let tex = try laplacian.sharpnessMap(from: source, ciContext: ciContext)
            sharpness = upscale(texture: tex, toExtentOf: source)
            if let estimator = depthEstimator {
                let map = try estimator.depthMap(for: source, ciContext: ciContext)
                depth = upscale(image: map, toExtentOf: source)
                if let s = sharpness, let d = depth {
                    focalPlane = computeFocalPlane(sharpness: s, depth: d)
                }
            }
        }

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
            chestRectangles: vision.chests
        )
    }

    /// Bundle of all Vision-derived rectangles and bars produced in one pass.
    private struct VisionResults {
        var faces: [CGRect] = []
        var eyes: [EyeBar] = []
        var bodies: [CGRect] = []
        var groins: [CGRect] = []
        var chests: [CGRect] = []
    }

    /// Run all five Vision-based detections in a single pass. The previous
    /// approach spent 5 × `createCGImage` + 5 × `VNImageRequestHandler` per
    /// analysis — one handler with three requests shares the image decode and
    /// any internal pyramid the framework builds. Face rectangles come from
    /// the landmarks request (`VNDetectFaceLandmarksRequest` is a subclass of
    /// the rectangle request); groin + chest share one body-pose request.
    private func runVision(in image: CIImage) -> VisionResults {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return VisionResults()
        }
        let extent = image.extent

        let faceLandmarks = VNDetectFaceLandmarksRequest()
        let bodyRects = VNDetectHumanRectanglesRequest()
        bodyRects.upperBodyOnly = false
        let bodyPose = VNDetectHumanBodyPoseRequest()

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([faceLandmarks, bodyRects, bodyPose])
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

        // Full-body rectangles.
        for obs in bodyRects.results ?? [] {
            results.bodies.append(Self.denormalize(obs.boundingBox, in: extent))
        }

        // Groin + chest from the shared pose observations.
        for obs in bodyPose.results ?? [] {
            if let g = Self.groinRect(from: obs, in: extent) { results.groins.append(g) }
            if let c = Self.chestRect(from: obs, in: extent) { results.chests.append(c) }
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
            size: CGSize(width: eyeDistance * 1.75, height: eyeDistance * 0.33),
            angleRadians: atan2(dy, dx)
        )
    }

    /// Derive a pelvis rect from a body-pose observation's hip joints.
    /// Nil when either hip is below the confidence floor.
    private static func groinRect(from obs: VNHumanBodyPoseObservation, in extent: CGRect) -> CGRect? {
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
        let rect = denormalize(normalized, in: extent)

        // Groin cover reads better as a wide horizontal band than as a
        // square or tall box, regardless of how the denormalized hip rect
        // comes out. Enforce a minimum width of 3x the rect's height:
        // rects already wider than that pass through; narrower ones get
        // padded symmetrically around the same center x.
        let minWidth = rect.height * 3
        guard rect.width < minWidth else { return rect }
        let extra = minWidth - rect.width
        return CGRect(
            x: rect.minX - extra / 2,
            y: rect.minY,
            width: minWidth,
            height: rect.height
        )
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
        let sharpSmall = shrinkToOrigin(sharpness, size: CGSize(width: side, height: side))
        let depthSmall = shrinkToOrigin(depth,     size: CGSize(width: side, height: side))

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

    /// Non-uniform stretch to `size` with origin at (0,0). Matches the CPU readback bounds.
    private func shrinkToOrigin(_ image: CIImage, size: CGSize) -> CIImage {
        let sx = size.width / image.extent.width
        let sy = size.height / image.extent.height
        let normalized = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)
        )
        return normalized
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: CGRect(origin: .zero, size: size))
    }

    // MARK: - Helpers

    private func upscale(texture: MTLTexture, toExtentOf source: CIImage) -> CIImage? {
        let low = CIImage(mtlTexture: texture, options: [
            .colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!
        ]) ?? CIImage.empty()
        return upscale(image: low, toExtentOf: source)
    }

    private func upscale(image: CIImage, toExtentOf source: CIImage) -> CIImage? {
        // Non-uniform stretch: the low-res map may come from the depth estimator at
        // the model's aspect, which differs from the source's. Uniform scaling with
        // aspectRatio=1 placed content at source.origin but only the cropped region's
        // data was stretched to fill the whole extent, producing a visible offset.
        // Stretching both axes independently makes coordinates map 1:1.
        let sx = source.extent.width / image.extent.width
        let sy = source.extent.height / image.extent.height
        let normalized = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)
        )
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
