import Foundation
import CoreImage
import CoreML

/// NIMA scalar-score result. Used by both the technical-quality
/// analyzer (TID2013) and the aesthetic-quality analyzer (AVA) —
/// both models share the same 10-bin softmax output shape.
/// `score` is `Σ (i+1) · p_i` in [1, 10]. Higher is better.
struct QualityScore: Hashable, Sendable {
    let score: Float
    /// Standard deviation of the distribution. Low = the network is
    /// confident, high = uncertain / mixed signal.
    let stdev: Float
    /// Full 10-bin distribution. Index 0 = P(rating=1), …, 9 = P(rating=10).
    let distribution: [Float]
}

/// Whole-image TECHNICAL quality from NIMA (MobileNet + TID2013).
/// Captures sharpness / exposure / compression / noise — the "did
/// the camera capture this correctly?" axis.
struct QualityAnalyzer {
    private var model: NIMAModel? { NIMAModel.technicalShared }
    var isReady: Bool { ModelArchive.quality.isInstalled() }
    func warm() -> Bool { model != nil }
    func analyze(image: CIImage, ciContext: CIContext) -> QualityScore? {
        model?.predict(image: image, ciContext: ciContext)
    }
}

/// Whole-image AESTHETIC quality from NIMA (MobileNet + AVA).
/// Captures composition / subject interest / lighting mood — the
/// "is this a good-looking photograph?" axis. Same architecture
/// as `QualityAnalyzer`, just a different training set.
struct AestheticAnalyzer {
    private var model: NIMAModel? { NIMAModel.aestheticShared }
    var isReady: Bool { ModelArchive.aesthetic.isInstalled() }
    func warm() -> Bool { model != nil }
    func analyze(image: CIImage, ciContext: CIContext) -> QualityScore? {
        model?.predict(image: image, ciContext: ciContext)
    }
}

// MARK: - Model wrapper (shared by technical + aesthetic)

private final class NIMAModel {
    /// Technical-quality (TID2013). Loaded lazily on first access.
    static let technicalShared: NIMAModel? = {
        try? NIMAModel(archive: .quality, label: "technical")
    }()
    /// Aesthetic-quality (AVA). Loaded lazily on first access.
    static let aestheticShared: NIMAModel? = {
        try? NIMAModel(archive: .aesthetic, label: "aesthetic")
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let label: String
    private let inputSize = CGSize(width: 224, height: 224)

    private static let numBins = 10

    init(archive: ModelArchive, label: String) throws {
        self.label = label
        let url = try archive.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // MobileNet-v1 is ANE-friendly and the 10-bin softmax has no
        // long-tail precision sensitivity (unlike the 101-bin age
        // head we retired), so `.all` is fine.
        config.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let input = inputs.first(where: { $0.value.type == .image })
                ?? inputs.first else {
            throw AnalysisError.modelLoadFailed(
                "NIMA \(label) model has no usable image input."
            )
        }
        self.inputName = input.key

        let outputs = model.modelDescription.outputDescriptionsByName
        if outputs["quality_distribution"] != nil {
            self.outputName = "quality_distribution"
        } else {
            let match = outputs.first { _, desc in
                desc.multiArrayConstraint?.shape
                    .map(\.intValue)
                    .reduce(1, *) == Self.numBins
            }
            guard let match else {
                throw AnalysisError.modelLoadFailed(
                    "NIMA \(label) model has no \(Self.numBins)-dim output."
                )
            }
            self.outputName = match.key
        }

        print("[NIMA/\(label)] loaded input=\(inputName) output=\(outputName)")
    }

    /// Aspect-preserving center-crop to 224² with a dedicated sRGB
    /// render context (same rationale as AgeEstimator — avoid
    /// extendedLinearDisplayP3 gamma drift on the way to an 8-bit
    /// buffer). Returns a scalar score + full distribution.
    func predict(image: CIImage, ciContext _: CIContext) -> QualityScore? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let side = min(extent.width, extent.height)
        let cropX = extent.minX + (extent.width - side) / 2
        let cropY = extent.minY + (extent.height - side) / 2
        let squareCrop = image.cropped(to: CGRect(
            x: cropX, y: cropY, width: side, height: side
        ))
        let scale = inputSize.width / side
        let transform = CGAffineTransform.identity
            .concatenating(CGAffineTransform(translationX: -cropX, y: -cropY))
            .concatenating(CGAffineTransform(scaleX: scale, y: scale))
        let resized = squareCrop
            .transformed(by: transform)
            .cropped(to: CGRect(origin: .zero, size: inputSize))

        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        let context = CIContext(options: [
            .workingColorSpace: sRGB,
            .outputColorSpace: sRGB,
        ])
        context.render(resized, to: pb,
                       bounds: CGRect(origin: .zero, size: inputSize),
                       colorSpace: sRGB)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let dist = result.featureValue(for: outputName)?.multiArrayValue,
                  dist.count >= Self.numBins else { return nil }

            var probs = [Float](repeating: 0, count: Self.numBins)
            var mean: Float = 0
            for i in 0..<Self.numBins {
                let p = dist[i].floatValue
                probs[i] = p.isFinite ? p : 0
                mean += Float(i + 1) * probs[i]
            }
            var variance: Float = 0
            for i in 0..<Self.numBins {
                let d = Float(i + 1) - mean
                variance += d * d * probs[i]
            }
            let stdev = variance.squareRoot()
            let score = max(1, min(10, mean))
            print(String(format: "[NIMA/%@] score=%.2f ± %.2f",
                         label, Double(score), Double(stdev)))
            return QualityScore(
                score: score,
                stdev: stdev.isFinite ? stdev : 0,
                distribution: probs
            )
        } catch {
            print("[NIMA/\(label)] predict failed: \(error)")
            return nil
        }
    }
}
