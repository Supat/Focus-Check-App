import CoreImage

/// Shared geometry helpers for the analysis pipeline. Several call sites
/// (depth estimator, NSFW classifier, motion-blur detector, focal-plane
/// readback) need to feed a fixed-size buffer; this centralizes the
/// "translate-to-origin + non-uniform scale + crop" recipe so callers don't
/// re-derive it each time.
extension CIImage {
    /// Shift the image so its `extent.origin` is at (0, 0). Most downstream
    /// filters expect buffers rooted there; skipping this step is a frequent
    /// source of subtle-offset bugs when the original came from a cropped
    /// Core Image graph.
    func translatedToOrigin() -> CIImage {
        transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
    }

    /// Non-uniform stretch to `size` with origin at (0, 0) — ignores aspect.
    /// Use when downstream code operates on a fixed buffer and you just
    /// want coordinates to line up pixel-for-pixel.
    func stretched(to size: CGSize) -> CIImage {
        let sx = size.width / extent.width
        let sy = size.height / extent.height
        return translatedToOrigin()
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: CGRect(origin: .zero, size: size))
    }
}
