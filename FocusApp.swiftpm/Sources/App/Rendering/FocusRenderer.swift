import MetalKit
import CoreImage
import CoreImage.CIFilterBuiltins
import SwiftUI

/// Composites the source image with a focus overlay each frame.
///
/// Runs on the renderer thread (MTKView callback), reads display state from the view model
/// without re-running analysis — threshold and style are display-only knobs (CLAUDE.md rule).
final class FocusRenderer {
    private let viewModel: FocusViewModel
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let colorSpace: CGColorSpace

    init(viewModel: FocusViewModel) {
        self.viewModel = viewModel
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal is required.")
        }
        self.device = device
        self.commandQueue = queue
        self.colorSpace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)!
        self.ciContext = CIContext(mtlDevice: device, options: [
            .workingColorSpace: colorSpace,
            .workingFormat: CIFormat.RGBAh,
            .cacheIntermediates: false
        ])
    }

    func resize(to size: CGSize) {
        // Intentionally cheap — no backing store to reallocate; CIContext handles drawable binding.
    }

    // MARK: - Draw

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer()
        else { return }

        let drawableSize = CGSize(
            width: view.drawableSize.width,
            height: view.drawableSize.height
        )

        let content = buildFrame(drawableSize: drawableSize)
        // Composite over solid black so letterbox bars are defined pixels, not drawable garbage.
        let canvas = CIImage(color: CIColor.black)
            .cropped(to: CGRect(origin: .zero, size: drawableSize))
        let over = CIFilter.sourceOverCompositing()
        over.inputImage = content
        over.backgroundImage = canvas
        let frame = over.outputImage ?? content

        let destination = CIRenderDestination(
            width: Int(drawableSize.width),
            height: Int(drawableSize.height),
            pixelFormat: view.colorPixelFormat,
            commandBuffer: commandBuffer,
            mtlTextureProvider: { drawable.texture }
        )
        destination.colorSpace = colorSpace

        do {
            _ = try ciContext.startTask(toRender: frame, from: frame.extent,
                                        to: destination, at: .zero)
        } catch {
            // Swallow — transient drawable errors recover next frame.
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Compositing

    private func buildFrame(drawableSize: CGSize) -> CIImage {
        // MTKView delegate callbacks run on main — safe to read the view model directly.
        let snapshot: (source: CIImage?, style: OverlayStyle, threshold: Float, color: Color) =
            MainActor.assumeIsolated {
                (viewModel.sourceImage, viewModel.style, viewModel.threshold, viewModel.overlayColor)
            }

        guard let source = snapshot.source else {
            return CIImage(color: CIColor.black).cropped(to: CGRect(origin: .zero, size: drawableSize))
        }

        // Fit source to drawable, preserving aspect ratio.
        let fitted = fit(image: source, into: drawableSize)

        let threshold = CGFloat(snapshot.threshold)
        let tint = CIColor(color: snapshot.color) ?? CIColor(red: 1, green: 0.85, blue: 0)

        let (sharpnessOverlay, depthOverlay) = overlays(for: fitted)

        switch snapshot.style {
        case .peaking:
            return peakingComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay,
                                    threshold: threshold, tint: tint)
        case .heatmap:
            return heatmapComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay)
        case .mask:
            return maskComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay,
                                 threshold: threshold, tint: tint)
        }
    }

    private func overlays(for target: CIImage) -> (sharpness: CIImage?, depth: CIImage?) {
        // MTKView delegate callbacks run on the main thread; the view model is @MainActor-isolated
        // so direct reads here are safe without hopping actors.
        let sharp = MainActor.assumeIsolated { viewModel.sharpnessOverlay }
        let depth = MainActor.assumeIsolated { viewModel.depthOverlay }

        func align(_ image: CIImage?) -> CIImage? {
            guard let image else { return nil }
            let sx = target.extent.width / image.extent.width
            let sy = target.extent.height / image.extent.height
            let scale = min(sx, sy)
            return image
                .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                .transformed(by: CGAffineTransform(translationX: target.extent.minX,
                                                   y: target.extent.minY))
        }

        return (align(sharp), align(depth))
    }

    // MARK: - Style pipelines

    private func peakingComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                                  threshold: CGFloat, tint: CIColor) -> CIImage {
        guard let focusMask = maskForMode(sharpness: sharpness, depth: depth, threshold: threshold)
        else { return base }

        // Real camera-style peaking: extract Sobel edges from the source at display
        // resolution, then gate by the in-focus mask. `Mask` remains the
        // flood-filled variant.
        //
        // The CIContext works in extended-linear Display P3, where linearized pixel
        // values cluster near zero in dark/mid tones — so raw Sobel gradients are
        // tiny. Crank filter intensity to max and apply a hard luma gain before
        // clamping so real edges saturate to 1.0 while flat regions stay at 0.
        let sobel = CIFilter.edges()
        sobel.inputImage = base.clampedToExtent()
        sobel.intensity = 10.0
        let edgeImage = (sobel.outputImage ?? base).cropped(to: base.extent)

        let edgeGain: CGFloat = 20.0
        let edgeMask = edgeImage
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0.2126 * edgeGain, y: 0.7152 * edgeGain,
                                         z: 0.0722 * edgeGain, w: 0),
                "inputGVector": CIVector(x: 0.2126 * edgeGain, y: 0.7152 * edgeGain,
                                         z: 0.0722 * edgeGain, w: 0),
                "inputBVector": CIVector(x: 0.2126 * edgeGain, y: 0.7152 * edgeGain,
                                         z: 0.0722 * edgeGain, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
            ])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
            ])

        // Gate by in-focus mask — both are near-binary after their respective gains,
        // so multiplication yields a near-binary combined mask.
        let gated = CIFilter.multiplyCompositing()
        gated.inputImage = edgeMask
        gated.backgroundImage = focusMask
        let combined = gated.outputImage ?? edgeMask

        let colored = CIImage(color: tint).cropped(to: base.extent)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = colored
        blend.backgroundImage = base
        blend.maskImage = combined
        return blend.outputImage ?? base
    }

    private func heatmapComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?) -> CIImage {
        guard let magnitude = sharpness ?? depth else { return base }
        let cropped = magnitude.clampedToExtent().cropped(to: base.extent)

        // Sharpness/depth maps are single-channel (R-only); G and B are zero. Broadcast R into
        // all three channels so the viridis CIColorCube samples at (v, v, v) — otherwise the
        // lookup only probes one axis and every pixel collapses to the purple low-end of viridis.
        // Also apply a gain since raw Laplacian values typically sit in ~0..0.2.
        let gain: CGFloat = 6.0
        let broadcast = CIFilter.colorMatrix()
        broadcast.inputImage = cropped
        broadcast.rVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        broadcast.gVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        broadcast.bVector = CIVector(x: gain, y: 0, z: 0, w: 0)
        broadcast.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)

        let normalized = (broadcast.outputImage ?? cropped).applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])

        let mapped = viridis(normalized)

        // 60% opacity overlay — reads well without fully hiding the photo.
        let alpha = CIFilter.colorMatrix()
        alpha.inputImage = mapped
        alpha.aVector = CIVector(x: 0, y: 0, z: 0, w: 0.6)
        let blend = CIFilter.sourceOverCompositing()
        blend.inputImage = alpha.outputImage
        blend.backgroundImage = base
        return blend.outputImage ?? base
    }

    private func maskComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                               threshold: CGFloat, tint: CIColor) -> CIImage {
        // Mask mode runs with a gentler effective threshold than the slider shows —
        // Peaking's edge-gated display is naturally sparse, but Mask benefits from
        // wider coverage so the translucent fill communicates in-focus regions.
        let adjusted = threshold * 0.7
        guard let mask = maskForMode(sharpness: sharpness, depth: depth, threshold: adjusted)
        else { return base }
        let coloredTint = CIColor(red: tint.red, green: tint.green, blue: tint.blue, alpha: 0.5)
        let colored = CIImage(color: coloredTint).cropped(to: base.extent)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = colored
        blend.backgroundImage = base
        blend.maskImage = mask
        return blend.outputImage ?? base
    }

    // MARK: - Mask derivation

    private func maskForMode(sharpness: CIImage?, depth: CIImage?, threshold: CGFloat) -> CIImage? {
        switch (sharpness, depth) {
        case (let s?, nil):
            return thresholded(s, at: threshold)
        case (nil, let d?):
            return thresholded(d, at: threshold)
        case (let s?, let d?):
            // Hybrid = intersection — CLAUDE.md rule.
            let sm = thresholded(s, at: threshold)
            let dm = thresholded(d, at: threshold)
            let minComp = CIFilter.minimumCompositing()
            minComp.inputImage = sm
            minComp.backgroundImage = dm
            return minComp.outputImage
        case (nil, nil):
            return nil
        }
    }

    private func thresholded(_ image: CIImage, at t: CGFloat) -> CIImage {
        // mask = clamp((input * preGain - t) * steepness, 0, 1) per channel
        //      = clamp(input * scale - bias, 0, 1)
        //
        // preGain lifts raw Laplacian magnitudes (~0..0.2) into a usable 0..1 range;
        // steepness controls how hard the threshold edge is. Combined into a single
        // colorMatrix. We also broadcast R → R,G,B so CIBlendWithMask (which reads
        // the red channel) sees the same intensity as the rest of the pipeline.
        let preGain: CGFloat = 6.0
        let steepness: CGFloat = 4.0
        let scale = preGain * steepness
        let bias = -steepness * t

        let matrix = CIFilter.colorMatrix()
        matrix.inputImage = image
        matrix.rVector = CIVector(x: scale, y: 0, z: 0, w: 0)
        matrix.gVector = CIVector(x: scale, y: 0, z: 0, w: 0)
        matrix.bVector = CIVector(x: scale, y: 0, z: 0, w: 0)
        matrix.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
        matrix.biasVector = CIVector(x: bias, y: bias, z: bias, w: 0)

        return (matrix.outputImage ?? image).applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])
    }

    private func viridis(_ image: CIImage) -> CIImage {
        // Viridis LUT (5 control stops, bilinearly interpolated via CIColorCube).
        // Never jet — perceptually non-uniform (CLAUDE.md rule).
        let size = 16
        var cube = [Float](repeating: 0, count: size * size * size * 4)
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    let i = (b * size * size + g * size + r) * 4
                    // Use luma as the index into the viridis ramp.
                    let luma = 0.2126 * Float(r) + 0.7152 * Float(g) + 0.0722 * Float(b)
                    let t = luma / Float(size - 1)
                    let c = viridisColor(at: t)
                    cube[i + 0] = c.r
                    cube[i + 1] = c.g
                    cube[i + 2] = c.b
                    cube[i + 3] = 1
                }
            }
        }
        let filter = CIFilter.colorCube()
        filter.cubeDimension = Float(size)
        filter.cubeData = cube.withUnsafeBufferPointer {
            Data(buffer: $0)
        }
        filter.inputImage = image
        return filter.outputImage ?? image
    }

    private func viridisColor(at t: Float) -> (r: Float, g: Float, b: Float) {
        // Sampled control points for the viridis colormap.
        let stops: [(Float, Float, Float, Float)] = [
            (0.00, 0.267, 0.005, 0.329),
            (0.25, 0.229, 0.322, 0.545),
            (0.50, 0.127, 0.566, 0.551),
            (0.75, 0.369, 0.789, 0.383),
            (1.00, 0.993, 0.906, 0.144)
        ]
        let clamped = max(0, min(1, t))
        for i in 0..<(stops.count - 1) {
            let a = stops[i]; let b = stops[i + 1]
            if clamped >= a.0 && clamped <= b.0 {
                let u = (clamped - a.0) / (b.0 - a.0)
                return (a.1 + (b.1 - a.1) * u,
                        a.2 + (b.2 - a.2) * u,
                        a.3 + (b.3 - a.3) * u)
            }
        }
        return (stops.last!.1, stops.last!.2, stops.last!.3)
    }

    private func fit(image: CIImage, into size: CGSize) -> CIImage {
        let sx = size.width / image.extent.width
        let sy = size.height / image.extent.height
        let s = min(sx, sy)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: s, y: s))
        let offsetX = (size.width - scaled.extent.width) / 2 - scaled.extent.minX
        let offsetY = (size.height - scaled.extent.height) / 2 - scaled.extent.minY
        return scaled.transformed(by: CGAffineTransform(translationX: offsetX, y: offsetY))
    }
}

extension CIColor {
    convenience init?(color: Color) {
        #if canImport(AppKit)
        let platformColor = NSColor(color).usingColorSpace(.displayP3) ?? NSColor(color)
        self.init(color: platformColor)
        #else
        let platformColor = UIColor(color)
        self.init(color: platformColor)
        #endif
    }
}
