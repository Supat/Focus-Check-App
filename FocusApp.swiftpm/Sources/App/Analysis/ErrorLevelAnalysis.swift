import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO

/// Error Level Analysis (ELA): re-encode the source as JPEG at
/// a fixed quality and diff against the original. Regions with
/// different compression histories produce visibly different
/// magnitudes in the diff, surfacing pasted / edited areas in
/// what would otherwise look like a uniform photograph.
///
/// Classic ELA expects a JPEG input. For lossless inputs (PNG /
/// TIFF) the round-trip becomes a "JPEG susceptibility map" —
/// regions that originated from a prior JPEG (typical of pasted
/// edits) compress more faithfully than genuinely lossless
/// regions, so the diff still highlights history mismatches; it
/// just isn't strict ELA. HEIF behaves like JPEG since both are
/// lossy. Pure synthetic content (AI-generated PNG, fresh
/// screenshots) gives a flat result by design — there's no
/// prior compression history to mismatch against.
final class ErrorLevelAnalyzer {
    private let ciContext: CIContext
    /// JPEG quality used for the round-trip encode. 0.90 is the
    /// canonical ELA choice — high enough that genuine-photo
    /// content produces a smooth low-magnitude diff, low enough
    /// that the quantisation artefacts the diff measures are
    /// large enough to render visibly after the renderer's gain
    /// pass.
    private let jpegQuality: Double = 0.90
    /// Long-side cap for analysis resolution. ELA quality scales
    /// with resolution but 2K is enough to surface manipulation
    /// artefacts visually; finer detail rarely reveals anything
    /// a viewer can't already see at 2K. Capping keeps memory +
    /// time bounded on 50–60 MP sources where the round-trip
    /// would otherwise materialise hundreds of MB of pixel data.
    private let maxLongSide: CGFloat = 2048

    init(ciContext: CIContext) {
        self.ciContext = ciContext
    }

    /// Compute the raw |source - reencoded| diff image. Renderer
    /// applies gain + clamp at display time so the threshold
    /// slider can scrub the visualisation. Returns nil if
    /// encoding / decoding fails (rare; the caller treats nil
    /// the same as "ELA unavailable" and falls back to the
    /// source image unchanged).
    func analyze(_ source: CIImage) -> CIImage? {
        let extent = source.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        // Downscale large sources before the round-trip so the
        // encode + decode + diff stays bounded for 50 MP+ inputs.
        let analysisInput = downscaleIfNeeded(source.cropped(to: extent))

        // Round-trip encode at fixed quality. Pin sRGB so the
        // arithmetic isn't measuring colour-space gamut drift on
        // top of compression error; the source is rasterised
        // into sRGB before reaching the JPEG encoder.
        let qualityKey = CIImageRepresentationOption(
            rawValue: kCGImageDestinationLossyCompressionQuality as String
        )
        let opts: [CIImageRepresentationOption: Any] = [qualityKey: jpegQuality]
        guard let jpegData = ciContext.jpegRepresentation(
            of: analysisInput, colorSpace: sRGB, options: opts
        ) else { return nil }
        guard let reencoded = CIImage(data: jpegData) else { return nil }

        // CIImage(data:) decodes at origin (0, 0); shift so its
        // extent overlaps the analysis input's so the diff
        // operands occupy the same coordinate space.
        let target = analysisInput.extent
        let aligned = reencoded.transformed(
            by: CGAffineTransform(translationX: target.minX - reencoded.extent.minX,
                                  y: target.minY - reencoded.extent.minY)
        ).cropped(to: target)

        // Per-channel absolute difference — the classic ELA
        // primitive. Output is in [0, 1] but typically clusters
        // very low (compression error per pixel is small); the
        // renderer multiplies by a threshold-driven gain to
        // surface the structure visually.
        let diff = CIFilter.differenceBlendMode()
        diff.inputImage = analysisInput
        diff.backgroundImage = aligned
        return diff.outputImage?.cropped(to: target)
    }

    private func downscaleIfNeeded(_ image: CIImage) -> CIImage {
        let extent = image.extent
        let long = max(extent.width, extent.height)
        guard long > maxLongSide else { return image }
        let scale = maxLongSide / long
        return image.applyingFilter("CILanczosScaleTransform", parameters: [
            kCIInputScaleKey: scale,
            kCIInputAspectRatioKey: 1.0
        ])
    }
}
