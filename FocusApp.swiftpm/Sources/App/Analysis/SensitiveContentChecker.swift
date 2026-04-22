import Foundation
import CoreImage

#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

/// Current availability of Apple's Sensitive Content Analysis framework on
/// this device and session. Tracked as a richer enum than a simple Bool so
/// the UI can explain *why* the feature is unavailable.
enum SensitiveContentAvailability: Equatable {
    /// Framework couldn't be imported (SDK / platform mismatch).
    case frameworkMissing
    /// Framework present but Communication Safety is off in Screen Time.
    case disabled
    /// Ready — system returns simple-intervention policy.
    case simpleInterventions
    /// Ready — system returns descriptive-intervention policy.
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
struct SensitiveContentChecker {

    /// Re-query the analyzer each call — Communication Safety can be toggled
    /// while the app is running and we want to pick that up without a restart.
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
    /// `nil` when the framework is unavailable or Communication Safety is
    /// disabled. Never throws — errors collapse to nil.
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
