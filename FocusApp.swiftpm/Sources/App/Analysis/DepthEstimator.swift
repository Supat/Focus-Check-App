import CoreML
import CoreImage
import Vision

/// Wraps the pre-compiled Depth Anything v2 Small F16 model.
///
/// The `.mlmodelc` is fetched at runtime by `DepthModelDownloader` and installed into
/// `Application Support/`. `init()` probes that location first, falling back to the
/// app bundle for dev setups that manually drop the model into `Sources/App/Resources/`.
/// When neither source has the model, it throws `AnalysisError.modelMissing` and the
/// analyzer degrades to sharpness-only.
struct DepthEstimator {
    private let model: MLModel

    init() throws {
        let url = try Self.locateModel()
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Prefer ANE; falls back to GPU then CPU.
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Probe the runtime install location first, then the app bundle. Returns the
    /// first `.mlmodelc` URL found, or throws `.modelMissing`.
    private static func locateModel() throws -> URL {
        if let installed = try? DepthModelDownloader.installedURL(),
           FileManager.default.fileExists(atPath: installed.path) {
            return installed
        }
        if let bundled = Bundle.main.url(forResource: "DepthAnythingV2SmallF16",
                                         withExtension: "mlmodelc") {
            return bundled
        }
        throw AnalysisError.modelMissing
    }

    /// Predict a relative monocular depth map at the model's native resolution.
    /// Output is a grayscale `CIImage` where brighter = closer.
    func depthMap(for image: CIImage, ciContext: CIContext) throws -> CIImage {
        // Depth Anything v2 expects 518×518 RGB input. The exact feature name comes from the model;
        // we probe the description so this stays model-agnostic if Apple ships a relabeled variant.
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "depth"

        let side: CGFloat = 518
        let fit = image
            .transformed(by: fitTransform(for: image.extent, target: CGSize(width: side, height: side)))
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        CVPixelBufferCreate(kCFAllocatorDefault,
                            Int(side), Int(side),
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { throw AnalysisError.coreImageFailure }
        ciContext.render(fit, to: pb)

        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(pixelBuffer: pb)
        ])
        let result = try model.prediction(from: features)

        if let buffer = result.featureValue(for: outputName)?.imageBufferValue {
            return CIImage(cvPixelBuffer: buffer)
        }
        if let array = result.featureValue(for: outputName)?.multiArrayValue {
            return try ciImage(fromDepthArray: array)
        }
        throw AnalysisError.coreImageFailure
    }

    private func fitTransform(for extent: CGRect, target: CGSize) -> CGAffineTransform {
        let sx = target.width / extent.width
        let sy = target.height / extent.height
        let s = min(sx, sy)
        return CGAffineTransform(scaleX: s, y: s)
    }

    /// Pack a Float32/Float16 MLMultiArray [H,W] into a luminance CIImage, normalized to [0,1].
    private func ciImage(fromDepthArray array: MLMultiArray) throws -> CIImage {
        // Handle shapes like [1, H, W] or [H, W].
        let shape = array.shape.map(\.intValue)
        let (h, w): (Int, Int)
        switch shape.count {
        case 2: (h, w) = (shape[0], shape[1])
        case 3: (h, w) = (shape[1], shape[2])
        case 4: (h, w) = (shape[2], shape[3])
        default: throw AnalysisError.coreImageFailure
        }

        let count = h * w
        var floats = [Float](repeating: 0, count: count)

        // Copy out as Float32 regardless of underlying dtype.
        for i in 0..<count {
            floats[i] = array[i].floatValue
        }

        // Normalize to [0,1]
        var lo: Float = .greatestFiniteMagnitude
        var hi: Float = -.greatestFiniteMagnitude
        for v in floats { if v < lo { lo = v }; if v > hi { hi = v } }
        let range = max(hi - lo, 1e-6)
        var bytes = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            bytes[i] = UInt8(clamping: Int(((floats[i] - lo) / range) * 255))
        }

        let data = Data(bytes)
        return CIImage(
            bitmapData: data,
            bytesPerRow: w,
            size: CGSize(width: w, height: h),
            format: .L8,
            colorSpace: CGColorSpaceCreateDeviceGray()
        )
    }
}
