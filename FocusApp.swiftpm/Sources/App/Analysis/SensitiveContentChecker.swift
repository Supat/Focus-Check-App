import Foundation
import CoreImage
import Vision

#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

/// Current availability of Apple's Sensitive Content Analysis framework on
/// this device and session. Tracked as a richer enum than a simple Bool so
/// the UI can explain *why* the feature is unavailable.
enum SensitiveContentAvailability: Equatable {
    /// Framework couldn't be imported (SDK / platform mismatch).
    case frameworkMissing
    /// Framework present but Communication Safety / per-app access is off.
    case disabled
    /// Ready — system returns simple-intervention policy.
    case simpleInterventions
    /// Ready — system returns descriptive-intervention policy.
    case descriptiveInterventions
    /// SCA not usable; checker will fall back to Vision's general classifier.
    case visionFallback

    /// Checker can produce a meaningful answer (via SCA or the Vision fallback).
    var isReady: Bool {
        switch self {
        case .simpleInterventions, .descriptiveInterventions, .visionFallback:
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
        case .visionFallback:           return "using VNClassifyImageRequest"
        }
    }
}

/// On-device nudity / sensitive-imagery classifier wrapped behind a thin
/// Swift interface.
///
/// Primary backend: Apple's `SensitiveContentAnalysis` framework (iOS 17+).
/// That framework is gated per-app on macOS and by Communication Safety on
/// iOS — in both cases the gate is invisible to unsigned / development builds
/// and `analysisPolicy` stays `.disabled`.
///
/// Fallback backend: `VNClassifyImageRequest`, Vision's built-in general
/// image classifier. No download, no entitlement, no gating — works in
/// Swift Playgrounds. Noticeably less tuned for nudity than SCA, but catches
/// obvious cases. Used only when SCA is unavailable.
struct SensitiveContentChecker {

    /// Confidence threshold for the Vision fallback. Vision's classifier is
    /// broad; a higher threshold trades recall for fewer false positives
    /// (swimwear / underwear / art).
    private let visionConfidenceThreshold: Float = 0.6

    /// Re-query on each call — Communication Safety / per-app access can be
    /// toggled at runtime.
    var availability: SensitiveContentAvailability {
        #if canImport(SensitiveContentAnalysis)
        switch SCSensitivityAnalyzer().analysisPolicy {
        case .disabled:                 return .visionFallback
        case .simpleInterventions:      return .simpleInterventions
        case .descriptiveInterventions: return .descriptiveInterventions
        @unknown default:               return .visionFallback
        }
        #else
        return .visionFallback
        #endif
    }

    /// Classify a CIImage. Returns `true`/`false` when the system answered,
    /// `nil` on hard errors. Never throws — errors collapse to nil.
    func check(image: CIImage, ciContext: CIContext) async -> Bool? {
        if let scaResult = await checkSCA(image: image, ciContext: ciContext) {
            return scaResult
        }
        return await checkVision(image: image, ciContext: ciContext)
    }

    // MARK: - Primary (SensitiveContentAnalysis)

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

    // MARK: - Fallback (VNClassifyImageRequest)

    private func checkVision(image: CIImage, ciContext: CIContext) async -> Bool? {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return nil
        }
        return await withCheckedContinuation { continuation in
            let request = VNClassifyImageRequest { request, error in
                guard error == nil,
                      let observations = request.results as? [VNClassificationObservation]
                else {
                    continuation.resume(returning: nil)
                    return
                }
                let sensitive = observations.contains { obs in
                    obs.confidence >= self.visionConfidenceThreshold &&
                    self.isSensitiveLabel(obs.identifier)
                }
                continuation.resume(returning: sensitive)
            }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(returning: nil)
            }
        }
    }

    /// Match Vision identifiers that denote sensitive content. Vision's
    /// taxonomy evolves between OS versions, so match permissively rather
    /// than enumerating exact strings.
    private func isSensitiveLabel(_ identifier: String) -> Bool {
        let lower = identifier.lowercased()
        return lower.contains("nudit")           // "nudity", "explicit_nudity"
            || lower.contains("undressed")
            || lower.contains("explicit")
            || lower.contains("sexual")
    }
}
