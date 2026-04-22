import Foundation
import CoreImage

#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

/// On-device nudity / sensitive-imagery classifier, wrapped behind a thin
/// Swift interface. Uses Apple's `SensitiveContentAnalysis` framework
/// (iOS 17+) which returns a binary `isSensitive` signal — no bounding
/// boxes, no anatomy-specific labels.
///
/// The framework only runs when the user has enabled "Communication
/// Safety" in Settings → Screen Time. If that's off, `check(...)` returns
/// nil and the caller should treat the image as not-sensitive.
struct SensitiveContentChecker {

    /// Returns true if the framework is present and Communication Safety is
    /// enabled on this device. Callers can use this to hide the mosaic
    /// toggle entirely when the feature would be a no-op.
    var isAvailable: Bool {
        #if canImport(SensitiveContentAnalysis)
        return SCSensitivityAnalyzer().analysisPolicy != .disabled
        #else
        return false
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
