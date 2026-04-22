import Metal
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import Vision

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
        /// Eye-bar rectangles derived from face-landmark eye points, in
        /// source-extent coordinates. Used for the .eyes black-bar mode.
        var eyeRectangles: [CGRect] = []
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

        // Face rectangles — used by the mosaic renderer. Detect on every
        // analysis so the data is ready when the user toggles mosaic on.
        let faceRectangles = detectFaces(in: source)
        let bodyRectangles = detectBodies(in: source)
        let groinRectangles = detectGroins(in: source)
        let eyeRectangles = detectEyes(in: source)
        let chestRectangles = detectChests(in: source)

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
            faceRectangles: faceRectangles,
            bodyRectangles: bodyRectangles,
            groinRectangles: groinRectangles,
            eyeRectangles: eyeRectangles,
            chestRectangles: chestRectangles
        )
    }

    /// Run Vision's face-rectangles request and convert the normalized
    /// bounding boxes back into the image's CIImage extent space. Returns
    /// an empty array if rendering or detection fails.
    private func detectFaces(in image: CIImage) -> [CGRect] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return []
        }
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let observations = request.results as? [VNFaceObservation] else { return [] }
        return observations.map { denormalize($0.boundingBox, in: image.extent) }
    }

    /// Run Vision's full-body human rectangles request. `upperBodyOnly = false`
    /// makes the boxes cover from head to feet rather than just torso/head.
    private func detectBodies(in image: CIImage) -> [CGRect] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return []
        }
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let observations = request.results as? [VNHumanObservation] else { return [] }
        return observations.map { denormalize($0.boundingBox, in: image.extent) }
    }

    /// Derive groin rectangles from body-pose hip joints. One rect per
    /// person whose `leftHip` / `rightHip` keypoints exceed a confidence
    /// floor. Centred slightly below the hip midpoint — the joint
    /// landmarks sit at the top of the femur, so "below" covers the
    /// pelvis/groin region.
    private func detectGroins(in image: CIImage) -> [CGRect] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return []
        }
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let observations = request.results as? [VNHumanBodyPoseObservation] else { return [] }
        let extent = image.extent

        return observations.compactMap { obs -> CGRect? in
            guard let leftHip = try? obs.recognizedPoint(.leftHip),
                  let rightHip = try? obs.recognizedPoint(.rightHip),
                  leftHip.confidence > 0.3,
                  rightHip.confidence > 0.3
            else { return nil }

            // Normalized 0..1, Y-up coordinates — same as bounding boxes.
            let cx = (leftHip.location.x + rightHip.location.x) / 2
            let cy = (leftHip.location.y + rightHip.location.y) / 2
            let hipDistance = max(abs(rightHip.location.x - leftHip.location.x), 0.04)

            // Rect padded 1.5x hip width, offset downward (lower Y in Y-up
            // coords) so the box covers the pelvis below the hip landmarks.
            let w = hipDistance * 1.5
            let h = hipDistance * 1.0
            let yOffset = -hipDistance * 0.35
            let normalized = CGRect(
                x: cx - w / 2,
                y: cy - h / 2 + yOffset,
                width: w,
                height: h
            )
            return denormalize(normalized, in: extent)
        }
    }

    /// Derive chest rectangles from body-pose shoulder + hip joints. One
    /// rect per person whose all four landmarks exceed the confidence floor.
    /// Covers the upper ~55% of the shoulder-to-hip span — clearly the
    /// 'chest' rather than the full torso — with a small top padding above
    /// the shoulder line so the collarbone / neckline is included.
    private func detectChests(in image: CIImage) -> [CGRect] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return []
        }
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let observations = request.results as? [VNHumanBodyPoseObservation] else { return [] }
        let extent = image.extent

        return observations.compactMap { obs -> CGRect? in
            guard let leftShoulder = try? obs.recognizedPoint(.leftShoulder),
                  let rightShoulder = try? obs.recognizedPoint(.rightShoulder),
                  let leftHip = try? obs.recognizedPoint(.leftHip),
                  let rightHip = try? obs.recognizedPoint(.rightHip),
                  leftShoulder.confidence > 0.3,
                  rightShoulder.confidence > 0.3,
                  leftHip.confidence > 0.3,
                  rightHip.confidence > 0.3
            else { return nil }

            // Vision coords are Y-up: shoulders sit at a larger y than hips.
            let shoulderY = (leftShoulder.location.y + rightShoulder.location.y) / 2
            let hipY = (leftHip.location.y + rightHip.location.y) / 2
            let centerX = (leftShoulder.location.x + rightShoulder.location.x) / 2
            let shoulderSpan = max(abs(rightShoulder.location.x - leftShoulder.location.x), 0.05)

            let torsoHeight = max(shoulderY - hipY, 0.05)
            let chestHeight = torsoHeight * 0.55
            let topPad = chestHeight * 0.15

            // Bottom edge in Y-up = shoulderY - chestHeight, height reaches
            // up past the shoulders by `topPad` to cover the collarbone.
            let w = shoulderSpan * 1.15
            let normalized = CGRect(
                x: centerX - w / 2,
                y: shoulderY - chestHeight,
                width: w,
                height: chestHeight + topPad
            )
            return denormalize(normalized, in: extent)
        }
    }

    /// Derive a single eye-bar rect per detected face. VNDetectFaceLandmarks
    /// gives eye landmark points in face-bbox-normalized coordinates; we
    /// convert to image-extent space and take the bounding rect of both
    /// eyes plus small horizontal padding and expanded height so the bar
    /// reads as a classic privacy redaction rather than a hairline strip.
    private func detectEyes(in image: CIImage) -> [CGRect] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return []
        }
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let observations = request.results as? [VNFaceObservation] else { return [] }
        let extent = image.extent

        return observations.compactMap { obs -> CGRect? in
            guard let landmarks = obs.landmarks,
                  let leftEye = landmarks.leftEye,
                  let rightEye = landmarks.rightEye
            else { return nil }

            // Face rect in image extent coords; landmark points are face-bbox-
            // normalized, so multiply by faceRect to lift them into extent space.
            let faceRect = denormalize(obs.boundingBox, in: extent)
            let eyePoints = (leftEye.normalizedPoints + rightEye.normalizedPoints).map { p in
                CGPoint(
                    x: faceRect.minX + p.x * faceRect.width,
                    y: faceRect.minY + p.y * faceRect.height
                )
            }
            guard !eyePoints.isEmpty else { return nil }
            let xs = eyePoints.map(\.x)
            let ys = eyePoints.map(\.y)
            guard let minX = xs.min(), let maxX = xs.max(),
                  let minY = ys.min(), let maxY = ys.max() else { return nil }

            let eyeWidth = maxX - minX
            let eyeHeight = maxY - minY
            let padW = eyeWidth * 0.2
            let barHeight = max(eyeHeight * 2.0, eyeWidth * 0.15)

            return CGRect(
                x: minX - padW,
                y: (minY + maxY) / 2 - barHeight / 2,
                width: eyeWidth + 2 * padW,
                height: barHeight
            )
        }
    }

    /// Vision boundingBoxes are normalized 0..1 with Y-up — the same
    /// convention as CIImage extent — so the conversion is just scale +
    /// translate by the source image's origin.
    private func denormalize(_ box: CGRect, in extent: CGRect) -> CGRect {
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
