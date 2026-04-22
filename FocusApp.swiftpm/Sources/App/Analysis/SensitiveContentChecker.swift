import Foundation
import CoreImage
import CoreML

#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

/// Current availability of the sensitive-content classifier stack on this
/// device and session. Tracked as an enum rather than a Bool so the UI can
/// explain *why* the feature is unavailable.
enum SensitiveContentAvailability: Equatable {
    /// SensitiveContentAnalysis framework couldn't be imported.
    case frameworkMissing
    /// SCA present but `.disabled` (per-app access off on macOS, or
    /// Communication Safety off on iOS).
    case disabled
    /// SCA ready — simple-intervention policy.
    case simpleInterventions
    /// SCA ready — descriptive-intervention policy.
    case descriptiveInterventions
    /// SCA unavailable, but a downloaded NSFW Core ML model is installed
    /// and will be used instead.
    case nsfwFallback

    var isReady: Bool {
        switch self {
        case .simpleInterventions, .descriptiveInterventions, .nsfwFallback:
            return true
        case .frameworkMissing, .disabled:
            return false
        }
    }

    var debugLabel: String {
        switch self {
        case .frameworkMissing:         return "SCA not compiled into build"
        case .disabled:                 return "analysisPolicy = .disabled"
        case .simpleInterventions:      return "analysisPolicy = .simpleInterventions"
        case .descriptiveInterventions: return "analysisPolicy = .descriptiveInterventions"
        case .nsfwFallback:             return "using downloaded NSFW model"
        }
    }
}

/// Combined result of a single sensitivity classification: whether the
/// image is flagged plus the top-confidence class label and probability
/// from the underlying model (used by the UI badge so users can see
/// *what* the classifier said and how confident it was).
struct SensitiveContentResult: Equatable {
    let isSensitive: Bool
    let topLabel: String?
    /// Probability of the top class in [0, 1]. SCA doesn't expose one, so
    /// this is nil for the primary backend; the NSFW fallback populates it.
    let confidence: Float?
}

/// On-device nudity / sensitive-imagery classifier.
///
/// Primary backend: Apple's `SensitiveContentAnalysis` framework (iOS 17+).
/// Gated per-app on macOS and by Communication Safety on iOS; unsigned
/// Playgrounds builds get `.disabled` and can't classify.
///
/// Fallback backend: a downloaded OpenNSFW-style Core ML model installed
/// via `NSFWModelDownloader`. Works in Playgrounds end-to-end once the
/// user has downloaded the model.
struct SensitiveContentChecker {

    /// Probability ≥ this flags the image as sensitive in the NSFW fallback.
    /// Lower = stricter (flags more content). 0.4 catches borderline cases the
    /// CreateML-trained lovoo classifier returns moderate confidence on.
    private let nsfwConfidenceThreshold: Float = 0.4

    var availability: SensitiveContentAvailability {
        #if canImport(SensitiveContentAnalysis)
        switch SCSensitivityAnalyzer().analysisPolicy {
        case .disabled:
            return NSFWModelDownloader.isInstalled() ? .nsfwFallback : .disabled
        case .simpleInterventions:      return .simpleInterventions
        case .descriptiveInterventions: return .descriptiveInterventions
        @unknown default:
            return NSFWModelDownloader.isInstalled() ? .nsfwFallback : .disabled
        }
        #else
        return NSFWModelDownloader.isInstalled() ? .nsfwFallback : .frameworkMissing
        #endif
    }

    /// Classify a CIImage. Returns the verdict + the top class label when the
    /// stack answered, `nil` when neither backend can run.
    func check(image: CIImage, ciContext: CIContext) async -> SensitiveContentResult? {
        if let scaResult = await checkSCA(image: image, ciContext: ciContext) {
            return scaResult
        }
        return checkNSFW(image: image, ciContext: ciContext)
    }

    // MARK: - SCA path

    private func checkSCA(image: CIImage, ciContext: CIContext) async -> SensitiveContentResult? {
        #if canImport(SensitiveContentAnalysis)
        let analyzer = SCSensitivityAnalyzer()
        guard analyzer.analysisPolicy != .disabled else { return nil }
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return nil
        }
        do {
            let result = try await analyzer.analyzeImage(cgImage)
            // SCA only exposes a `Sensitivity` struct (currently just .nudity)
            // when isSensitive is true. Surface "Nudity" / "Safe" rather than
            // a generic "Sensitive" so the badge text reads like a real
            // class label.
            let label = result.isSensitive ? "Nudity" : "Safe"
            return SensitiveContentResult(
                isSensitive: result.isSensitive,
                topLabel: label,
                confidence: nil
            )
        } catch {
            return nil
        }
        #else
        return nil
        #endif
    }

    // MARK: - NSFW Core ML path

    private func checkNSFW(image: CIImage, ciContext: CIContext) -> SensitiveContentResult? {
        guard let classifier = NSFWClassifier.shared,
              let top = classifier.topClass(for: image, ciContext: ciContext)
        else { return nil }
        let isSensitive = top.label.lowercased().contains("nsfw")
            && top.probability >= nsfwConfidenceThreshold
        return SensitiveContentResult(
            isSensitive: isSensitive,
            topLabel: top.label,
            confidence: top.probability
        )
    }
}

/// Minimal wrapper around a downloaded Core ML NSFW classifier.
/// Loaded lazily at first use; held statically so repeated invocations
/// reuse the MLModel instance.
private final class NSFWClassifier {
    static var shared: NSFWClassifier? = {
        try? NSFWClassifier()
    }()

    private let model: MLModel
    private let inputName: String
    private let inputSize: CGSize

    init() throws {
        let url = try NSFWModelDownloader.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }
        self.inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"

        // Read the model's expected input size from its image constraint.
        // CreateML models typically fix this at 299x299 or 224x224.
        var resolvedSize = CGSize(width: 224, height: 224)
        if let desc = model.modelDescription.inputDescriptionsByName[inputName],
           let constraint = desc.imageConstraint {
            resolvedSize = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolvedSize
    }

    /// Top-confidence class for an image: (label, probability). Returns nil
    /// on any failure.
    func topClass(for image: CIImage, ciContext: CIContext) -> (label: String, probability: Float)? {
        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        let resized = resize(image, to: inputSize)

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        ciContext.render(resized, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            return extractTopClass(from: result)
        } catch {
            return nil
        }
    }

    /// CreateML image classifiers expose two outputs — `classLabel` (String)
    /// and `classLabelProbs` (Dictionary<String, Double>) — so we have to
    /// iterate all output names to find the probability dictionary. Picks
    /// the class with the highest probability. Falls back to MLMultiArray
    /// outputs for non-CreateML models.
    private func extractTopClass(from result: MLFeatureProvider) -> (label: String, probability: Float)? {
        for name in result.featureNames {
            guard let value = result.featureValue(for: name) else { continue }
            if value.type == .dictionary {
                let dict = value.dictionaryValue
                var topKey: String?
                var topProb: Float = -1
                for (key, number) in dict {
                    let keyStr = (key as? String) ?? String(describing: key)
                    let prob = number.floatValue
                    if prob > topProb {
                        topProb = prob
                        topKey = keyStr
                    }
                }
                if let topKey { return (topKey, topProb) }
            }
        }
        // MLMultiArray fallback: OpenNSFW-style 2-element [SFW, NSFW] vector.
        for name in result.featureNames {
            if let array = result.featureValue(for: name)?.multiArrayValue, array.count >= 2 {
                let sfw = array[0].floatValue
                let nsfw = array[1].floatValue
                return nsfw >= sfw ? ("NSFW", nsfw) : ("SFW", sfw)
            }
        }
        return nil
    }

    /// Non-uniform stretch to `size` — matches the depth estimator's helper.
    private func resize(_ image: CIImage, to size: CGSize) -> CIImage {
        let sx = size.width / image.extent.width
        let sy = size.height / image.extent.height
        let normalized = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)
        )
        return normalized
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: CGRect(origin: .zero, size: size))
    }
}
