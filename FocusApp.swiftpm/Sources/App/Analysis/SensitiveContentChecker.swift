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

/// On-device nudity / sensitive-imagery classifier.
///
/// Primary backend: Apple's `SensitiveContentAnalysis` framework (iOS 17+).
/// Gated per-app on macOS and by Communication Safety on iOS; unsigned
/// Playgrounds builds get `.disabled` and can't classify.
///
/// Fallback backend: a downloaded OpenNSFW-style Core ML model installed
/// via `NSFWModelDownloader`. Works in Playgrounds end-to-end once the
/// user has downloaded the model. Less polished than SCA but useful
/// enough for a dev build.
struct SensitiveContentChecker {

    /// Probability ≥ this flags the image as sensitive in the NSFW fallback.
    private let nsfwConfidenceThreshold: Float = 0.6

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

    /// Classify a CIImage. Returns `true`/`false` when the stack answered,
    /// `nil` when neither backend can run.
    func check(image: CIImage, ciContext: CIContext) async -> Bool? {
        if let scaResult = await checkSCA(image: image, ciContext: ciContext) {
            return scaResult
        }
        return checkNSFW(image: image, ciContext: ciContext)
    }

    // MARK: - SCA path

    private func checkSCA(image: CIImage, ciContext: CIContext) async -> Bool? {
        #if canImport(SensitiveContentAnalysis)
        let analyzer = SCSensitivityAnalyzer()
        guard analyzer.analysisPolicy != .disabled else { return nil }
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return nil
        }
        do {
            let result = try await analyzer.analyzeImage(cgImage)
            return result.isSensitive
        } catch {
            return nil
        }
        #else
        return nil
        #endif
    }

    // MARK: - NSFW Core ML path

    private func checkNSFW(image: CIImage, ciContext: CIContext) -> Bool? {
        guard let classifier = NSFWClassifier.shared else { return nil }
        let probability = classifier.nsfwProbability(for: image, ciContext: ciContext)
        guard let probability else { return nil }
        return probability >= nsfwConfidenceThreshold
    }
}

/// Minimal wrapper around a downloaded OpenNSFW-style Core ML model.
/// Loaded lazily at first use; held statically so repeated invocations
/// reuse the MLModel instance.
private final class NSFWClassifier {
    static var shared: NSFWClassifier? = {
        try? NSFWClassifier()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String

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
        self.outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "classLabelProbs"
    }

    /// NSFW probability in [0, 1]. Returns nil on any failure.
    func nsfwProbability(for image: CIImage, ciContext: CIContext) -> Float? {
        let side = 224
        let resized = resize(image, to: CGSize(width: side, height: side))

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, side, side,
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
            return extractNSFWScore(from: result)
        } catch {
            return nil
        }
    }

    /// Probe common output shapes: dictionary {NSFW: p, SFW: q} or 2-element
    /// MLMultiArray where index 1 is the NSFW probability. Different
    /// checkpoints encode this differently.
    private func extractNSFWScore(from result: MLFeatureProvider) -> Float? {
        if let dict = result.featureValue(for: outputName)?.dictionaryValue {
            for (key, value) in dict {
                let keyStr = (key as? String) ?? String(describing: key)
                if keyStr.lowercased().contains("nsfw"),
                   let num = value as? NSNumber {
                    return num.floatValue
                }
            }
        }
        if let array = result.featureValue(for: outputName)?.multiArrayValue {
            if array.count >= 2 { return array[1].floatValue }
            if array.count == 1 { return array[0].floatValue }
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
