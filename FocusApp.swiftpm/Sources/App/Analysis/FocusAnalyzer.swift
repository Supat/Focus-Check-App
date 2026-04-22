import Metal
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO

/// Owns all heavyweight GPU/ML resources and serializes access through actor isolation.
/// CLAUDE.md rule: MPS / CIContext / MLModel calls live on this actor.
actor FocusAnalyzer {
    struct Overlays {
        var sharpness: CIImage?
        var depth: CIImage?
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let laplacian: LaplacianVariance
    private var depthEstimator: DepthEstimator?
    private let downloader = DepthModelDownloader()

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

    /// Run the analysis pipeline for the given mode. Returns display-ready overlay images
    /// already upscaled to the source's extent.
    func analyze(mode: AnalysisMode) throws -> Overlays {
        guard let source else { throw AnalysisError.imageDecodeFailed }
        try Task.checkCancellation()

        var sharpness: CIImage?
        var depth: CIImage?

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
            }
        }

        return Overlays(sharpness: sharpness, depth: depth)
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
