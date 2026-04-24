import Foundation
import CoreImage
import CoreML

/// NIMA technical-quality result. `score` is the expectation of the
/// 10-bin softmax over rating levels 1..10 — so it lives in [1, 10]
/// and higher is better. `distribution` is the full softmax, kept
/// for callers that want to compute mode, stdev, or display bars.
struct QualityScore: Hashable, Sendable {
    /// Expected quality, `Σ (i+1) · p_i`. Clamped to [1, 10].
    let score: Float
    /// Standard deviation of the distribution. Low stdev = the
    /// network is confident; high stdev = uncertain / mixed signal.
    let stdev: Float
    /// Full 10-bin distribution. Index 0 is P(rating=1), index 9 is
    /// P(rating=10).
    let distribution: [Float]
}

/// Thin wrapper around the NIMA MobileNet MLModel. Full-image (not
/// per-face). `analyze` returns nil when the model isn't installed
/// so `FocusAnalyzer` can skip the stage silently.
struct QualityAnalyzer {
    private var model: NIMAModel? { NIMAModel.shared }

    var isReady: Bool { ModelArchive.quality.isInstalled() }

    func warm() -> Bool { model != nil }

    func analyze(image: CIImage, ciContext: CIContext) -> QualityScore? {
        model?.predict(image: image, ciContext: ciContext)
    }
}

// MARK: - Model wrapper

private final class NIMAModel {
    static var shared: NIMAModel? = {
        try? NIMAModel()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputSize = CGSize(width: 224, height: 224)

    private static let numBins = 10

    init() throws {
        let url = try ModelArchive.quality.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // MobileNet-v1 is ANE-friendly — leave `.all` so Core ML
        // picks the fastest backend. The 10-bin softmax reduces to
        // a scalar via expectation and doesn't have the long-tail
        // precision sensitivity the 101-bin age head did.
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
                "NIMA model has no usable image input."
            )
        }
        self.inputName = input.key

        let outputs = model.modelDescription.outputDescriptionsByName
        if let preferred = outputs["quality_distribution"] {
            _ = preferred
            self.outputName = "quality_distribution"
        } else {
            let match = outputs.first { _, desc in
                desc.multiArrayConstraint?.shape
                    .map(\.intValue)
                    .reduce(1, *) == Self.numBins
            }
            guard let match else {
                throw AnalysisError.modelLoadFailed(
                    "NIMA model has no \(Self.numBins)-dim output."
                )
            }
            self.outputName = match.key
        }

        print("[NIMA] loaded input=\(inputName) output=\(outputName)")
    }

    func predict(image: CIImage, ciContext _: CIContext) -> QualityScore? {
        // Aspect-preserving resize to 224² via Lanczos, with a
        // center crop if the source aspect differs from square.
        // NIMA was trained on square crops of arbitrary photos, so
        // stretching anisotropically would bias the quality score.
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
        // Dedicated sRGB context — same rationale as AgeEstimator:
        // the shared analyzer CIContext uses extendedLinearDisplayP3
        // which can drift pixel values away from byte-accurate sRGB
        // on the way out. NIMA is MobileNet-backed and sensitive to
        // that drift because the scale/bias preprocessing (baked
        // into the Core ML ImageType) runs directly on pixel bytes.
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
            print(String(format: "[NIMA] score=%.2f ± %.2f", Double(score), Double(stdev)))
            return QualityScore(
                score: score,
                stdev: stdev.isFinite ? stdev : 0,
                distribution: probs
            )
        } catch {
            print("[NIMA] predict failed: \(error)")
            return nil
        }
    }
}
