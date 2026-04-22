import Foundation
import CoreImage

#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

/// Current availability of the sensitive-content classifier stack on this
/// device and session. Tracked as an enum rather than a Bool so the UI can
/// explain *why* the feature is unavailable.
enum SensitiveContentAvailability: Equatable {
    /// SensitiveContentAnalysis framework couldn't be imported.
    case frameworkMissing
    /// SCA is present but `.disabled` (per-app access off on macOS, or
    /// Communication Safety off on iOS). Classifier can't run.
    case disabled
    /// SCA ready — simple-intervention policy.
    case simpleInterventions
    /// SCA ready — descriptive-intervention policy.
    case descriptiveInterventions

    var isReady: Bool {
        self == .simpleInterventions || self == .descriptiveInterventions
    }

    var debugLabel: String {
        switch self {
        case .frameworkMissing:         return "SCA not compiled into build"
        case .disabled:                 return "analysisPolicy = .disabled"
        case .simpleInterventions:      return "analysisPolicy = .simpleInterventions"
        case .descriptiveInterventions: return "analysisPolicy = .descriptiveInterventions"
        }
    }
}

/// On-device nudity / sensitive-imagery classifier wrapped behind a thin
/// Swift interface. Uses Apple's `SensitiveContentAnalysis` framework
/// (iOS 17+) which returns a binary `isSensitive` signal — no bounding
/// boxes, no anatomy-specific labels.
///
/// The framework is gated per-app on macOS and by Communication Safety
/// on iOS. In Swift Playgrounds / unsigned builds those gates are not
/// lifted and `analysisPolicy` stays `.disabled`, which means `check(...)`
/// returns nil. Callers treat nil as "no decision" and skip the mosaic.
struct SensitiveContentChecker {

    /// Re-query on each call — Communication Safety / per-app access can
    /// be toggled at runtime.
    var availability: SensitiveContentAvailability {
        #if canImport(SensitiveContentAnalysis)
        switch SCSensitivityAnalyzer().analysisPolicy {
        case .disabled:                 return .disabled
        case .simpleInterventions:      return .simpleInterventions
        case .descriptiveInterventions: return .descriptiveInterventions
        @unknown default:               return .disabled
        }
        #else
        return .frameworkMissing
        #endif
    }

    /// Classify a CIImage. Returns `true`/`false` when the system answered,
    /// `nil` when the framework is unavailable or Communication Safety /
    /// per-app access is off. Never throws — errors collapse to nil.
    func check(image: CIImage, ciContext: CIContext) async -> Bool? {
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
}
