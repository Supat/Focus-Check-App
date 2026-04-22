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
        let snapshot: (source: CIImage?, style: OverlayStyle, threshold: Float,
                       color: Color, focalPlane: Float?,
                       zoomScale: CGFloat, zoomAnchor: CGPoint,
                       sharpness: CIImage?, depth: CIImage?, motion: CIImage?,
                       overlayHidden: Bool, mosaic: Bool, mosaicMode: MosaicMode,
                       faces: [CGRect], bodies: [CGRect], groins: [CGRect]) =
            MainActor.assumeIsolated {
                let applyMosaic = (viewModel.isSensitive == true) && viewModel.mosaicEnabled
                return (viewModel.sourceImage, viewModel.style, viewModel.threshold,
                        viewModel.overlayColor, viewModel.focalPlane,
                        viewModel.zoomScale, viewModel.zoomAnchor,
                        viewModel.sharpnessOverlay, viewModel.depthOverlay,
                        viewModel.motionOverlay,
                        viewModel.overlayHidden, applyMosaic, viewModel.mosaicMode,
                        viewModel.faceRectangles, viewModel.bodyRectangles,
                        viewModel.groinRectangles)
            }

        guard let source = snapshot.source else {
            return CIImage(color: CIColor.black).cropped(to: CGRect(origin: .zero, size: drawableSize))
        }

        // Sensitive content + mosaic toggle on → pixelate either just the
        // detected face regions or the whole frame, depending on mode.
        // Applied at source resolution so fit+zoom handles the result
        // without any rect-coordinate gymnastics.
        let baseSource: CIImage = {
            guard snapshot.mosaic else { return source }
            switch snapshot.mosaicMode {
            case .face:
                guard !snapshot.faces.isEmpty else { return source }
                return regionMosaic(source: source, regions: snapshot.faces, capDivisor: 32)
            case .groin:
                guard !snapshot.groins.isEmpty else { return source }
                return regionMosaic(source: source, regions: snapshot.groins, capDivisor: 32)
            case .body:
                // Prefer full-body regions; if Vision didn't find any, fall
                // back to face rects so flagged content still gets some cover.
                if !snapshot.bodies.isEmpty {
                    return regionMosaic(source: source, regions: snapshot.bodies, capDivisor: 64)
                }
                if !snapshot.faces.isEmpty {
                    return regionMosaic(source: source, regions: snapshot.faces, capDivisor: 32)
                }
                return source
            case .whole:
                return wholeMosaic(source: source)
            }
        }()

        // Apply fit + zoom to the base AND both overlays using identical parameters
        // so they stay spatially aligned. The overlays are already at source.extent
        // (upscaled by FocusAnalyzer), so the same transform maps them 1:1.
        let zoom = snapshot.zoomScale
        let anchor = snapshot.zoomAnchor
        let fitted = fit(image: baseSource, into: drawableSize, zoom: zoom, anchor: anchor)

        // Press-and-hold compare mode: return the base photo only (still
        // mosaiced if applicable).
        if snapshot.overlayHidden { return fitted }

        let sharpnessOverlay = snapshot.sharpness.map {
            fit(image: $0, into: drawableSize, zoom: zoom, anchor: anchor)
        }
        let depthOverlay = snapshot.depth.map {
            fit(image: $0, into: drawableSize, zoom: zoom, anchor: anchor)
        }
        let motionOverlay = snapshot.motion.map {
            fit(image: $0, into: drawableSize, zoom: zoom, anchor: anchor)
        }

        let threshold = CGFloat(snapshot.threshold)
        let tint = CIColor(color: snapshot.color) ?? CIColor(red: 1, green: 0.85, blue: 0)

        switch snapshot.style {
        case .peaking:
            return peakingComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay,
                                    threshold: threshold, tint: tint)
        case .heatmap:
            return heatmapComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay,
                                    threshold: threshold)
        case .mask:
            return maskComposite(base: fitted, sharpness: sharpnessOverlay, depth: depthOverlay,
                                 threshold: threshold, tint: tint)
        case .focusError:
            return focusErrorComposite(base: fitted, depth: depthOverlay,
                                       focalPlane: snapshot.focalPlane,
                                       threshold: threshold)
        case .motion:
            return motionComposite(base: fitted, motion: motionOverlay,
                                   threshold: threshold, tint: tint)
        }
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

    private func heatmapComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                                  threshold: CGFloat) -> CIImage {
        // Combine the two signals. In Hybrid mode both are non-nil; take the
        // per-pixel maximum so depth rescues smooth in-focus surfaces that the
        // Laplacian underestimates (skin, sky, walls). Other modes just use
        // whichever signal is present.
        let magnitude: CIImage
        switch (sharpness, depth) {
        case (let s?, let d?):
            let combine = CIFilter.maximumCompositing()
            combine.inputImage = s
            combine.backgroundImage = d
            magnitude = combine.outputImage ?? s
        case (let s?, nil):
            magnitude = s
        case (nil, let d?):
            magnitude = d
        case (nil, nil):
            return base
        }
        let cropped = magnitude.clampedToExtent().cropped(to: base.extent)

        // Sharpness/depth maps are single-channel (R-only); G and B are zero.
        // Broadcast R into all three channels so the viridis CIColorCube samples
        // the diagonal. Fixed gain (6x) lifts raw Laplacian magnitudes (~0..0.2)
        // into the 0..1 range; threshold controls a subtractive floor so the
        // slider trims off dim pixels without changing peak brightness.
        let gain: CGFloat = 6.0
        let bias: CGFloat = -3.0 * threshold

        let normalized = cropped
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: bias, y: bias, z: bias, w: 0)
            ])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
            ])

        let mapped = viridis(normalized)

        // Variable alpha: use the normalized magnitude as the mask so weak
        // regions let the photo show through and only strong regions are
        // fully overlaid. 0.75 peak alpha keeps some photo detail visible.
        let mask = normalized.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.75, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0.75, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0.75, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])

        let blend = CIFilter.blendWithMask()
        blend.inputImage = mapped
        blend.backgroundImage = base
        blend.maskImage = mask
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

    /// Tint regions where the per-patch motion-blur confidence exceeds a
    /// cutoff tied to the threshold slider. 50% tint alpha keeps the photo
    /// detail visible under the overlay.
    ///
    /// Motion confidence is already in [0,1] (unlike the ~0-0.2 Laplacian
    /// signal), so we use a dedicated linear threshold instead of reusing
    /// `thresholded()`'s 24x pre-gain which would saturate everything.
    private func motionComposite(base: CIImage, motion: CIImage?,
                                 threshold: CGFloat, tint: CIColor) -> CIImage {
        guard let motion else { return base }

        // Slider → effective confidence cutoff. Natural image variation usually
        // sits below 0.3, real motion blur above 0.5, so map [0,1] → [0.3, 0.8].
        let cutoff = 0.3 + 0.5 * threshold
        let steepness: CGFloat = 5.0
        let scale = steepness
        let bias = -steepness * cutoff

        let mask = motion
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: scale, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: scale, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: scale, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: bias, y: bias, z: bias, w: 0)
            ])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
            ])

        let coloredTint = CIColor(red: tint.red, green: tint.green, blue: tint.blue, alpha: 0.5)
        let colored = CIImage(color: coloredTint).cropped(to: base.extent)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = colored
        blend.backgroundImage = base
        blend.maskImage = mask
        return blend.outputImage ?? base
    }

    /// Two-color focus-error overlay:
    /// - Red where depth > focalPlane + ε (too close, foreground blur)
    /// - Blue where depth < focalPlane − ε (too far, background blur)
    /// - Untinted where |depth − focalPlane| < ε (within the user-tuned DoF band)
    ///
    /// DA v2's convention: higher depth value = closer to the camera.
    private func focusErrorComposite(base: CIImage, depth: CIImage?,
                                     focalPlane: Float?, threshold: CGFloat) -> CIImage {
        guard let depth, let focalPlane else { return base }

        // The threshold slider controls the DoF tolerance band. Larger threshold → wider
        // tolerance → less red/blue coverage. Range tuned tight so most out-of-focus
        // pixels land in the red/blue bins at the default slider position.
        let epsilon: CGFloat = 0.01 + 0.10 * threshold
        let gain: CGFloat = 6.0

        // Normalize depth to a grayscale where R = G = B = depth magnitude. Handles both
        // BGRA (from model's image output) and L8 (from MLMultiArray path).
        let gray = depth.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])

        // close mask = clamp((depth − (focal + ε)) * gain, 0, 1)
        let closeBias = -gain * (CGFloat(focalPlane) + epsilon)
        let closeMask = gray
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: gain, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: closeBias, y: closeBias, z: closeBias, w: 0)
            ])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
            ])

        // far mask = clamp(((focal − ε) − depth) * gain, 0, 1)
        let farBias = gain * (CGFloat(focalPlane) - epsilon)
        let farMask = gray
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: -gain, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: -gain, y: 0, z: 0, w: 0),
                "inputBVector": CIVector(x: -gain, y: 0, z: 0, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: farBias, y: farBias, z: farBias, w: 0)
            ])
            .applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
            ])

        // Red tint where too-close, composited over base.
        let red = CIImage(color: CIColor(red: 1, green: 0.1, blue: 0.1)).cropped(to: base.extent)
        let redStep = CIFilter.blendWithMask()
        redStep.inputImage = red
        redStep.backgroundImage = base
        redStep.maskImage = closeMask
        let afterClose = redStep.outputImage ?? base

        // Blue tint where too-far, stacked on top.
        let blue = CIImage(color: CIColor(red: 0.1, green: 0.4, blue: 1)).cropped(to: base.extent)
        let blueStep = CIFilter.blendWithMask()
        blueStep.inputImage = blue
        blueStep.backgroundImage = afterClose
        blueStep.maskImage = farMask
        return blueStep.outputImage ?? afterClose
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

    /// Pixelate the whole source. Block size is `longer_side / 64 × 1.5`
    /// — 50% larger than the per-region cap so whole-image mosaic reads as
    /// coarser and more unmistakable.
    private func wholeMosaic(source: CIImage) -> CIImage {
        let pixelate = CIFilter.pixellate()
        pixelate.inputImage = source.clampedToExtent()
        let longerSide = max(source.extent.width, source.extent.height)
        pixelate.scale = Float(longerSide / 64 * 1.5)
        pixelate.center = CGPoint(x: source.extent.midX, y: source.extent.midY)
        return (pixelate.outputImage ?? source).cropped(to: source.extent)
    }

    /// Pixelate only the supplied regions; everything else stays untouched.
    /// Operates on the source image so downstream fit+zoom doesn't have to
    /// transform the rectangles. Block size scales with the smallest region
    /// so blocks stay roughly proportional to what they're covering, capped
    /// to `longer_side / capDivisor` — a larger capDivisor = smaller blocks.
    private func regionMosaic(source: CIImage,
                              regions: [CGRect],
                              capDivisor: CGFloat) -> CIImage {
        guard let smallest = regions.min(by: {
            $0.width * $0.height < $1.width * $1.height
        }) else { return source }

        let pixelate = CIFilter.pixellate()
        pixelate.inputImage = source.clampedToExtent()
        let regionBased = min(smallest.width, smallest.height) * 0.08
        let cap = max(source.extent.width, source.extent.height) / capDivisor
        pixelate.scale = Float(max(min(regionBased, cap), 4))
        pixelate.center = CGPoint(x: source.extent.midX, y: source.extent.midY)
        let pixelated = (pixelate.outputImage ?? source).cropped(to: source.extent)

        // Mask: black canvas with white rects for each region. The blend
        // filter reads the red channel as the mask weight.
        var mask: CIImage = CIImage(color: CIColor.black).cropped(to: source.extent)
        for region in regions {
            // Expand 10% so edges (hair, hands, etc.) are also covered.
            let expanded = region.insetBy(dx: -region.width * 0.1, dy: -region.height * 0.1)
            let whiteBox = CIImage(color: CIColor.white).cropped(to: expanded)
            let stack = CIFilter.sourceOverCompositing()
            stack.inputImage = whiteBox
            stack.backgroundImage = mask
            mask = stack.outputImage ?? mask
        }

        let blend = CIFilter.blendWithMask()
        blend.inputImage = pixelated
        blend.backgroundImage = source
        blend.maskImage = mask
        return (blend.outputImage ?? source).cropped(to: source.extent)
    }

    private func fit(image: CIImage, into size: CGSize,
                     zoom: CGFloat = 1.0,
                     anchor: CGPoint = CGPoint(x: 0.5, y: 0.5)) -> CIImage {
        let sx = size.width / image.extent.width
        let sy = size.height / image.extent.height
        let s = min(sx, sy)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: s, y: s))
        let offsetX = (size.width - scaled.extent.width) / 2 - scaled.extent.minX
        let offsetY = (size.height - scaled.extent.height) / 2 - scaled.extent.minY
        let fitted = scaled.transformed(by: CGAffineTransform(translationX: offsetX, y: offsetY))

        guard zoom > 1.001 else { return fitted }

        // Anchor is in SwiftUI view coords (Y-top, 0...1). Convert to CIImage coords
        // (Y-bottom) in drawable space before composing the scale-about-anchor.
        let ax = anchor.x * size.width
        let ay = (1 - anchor.y) * size.height

        // Scale about (ax, ay): T(ax, ay) · S(zoom) · T(-ax, -ay).
        let transform = CGAffineTransform(translationX: -ax, y: -ay)
            .concatenating(CGAffineTransform(scaleX: zoom, y: zoom))
            .concatenating(CGAffineTransform(translationX: ax, y: ay))

        return fitted.transformed(by: transform)
            .cropped(to: CGRect(origin: .zero, size: size))
    }
}

extension CIColor {
    convenience init?(color: Color) {
        // iOSApplication product — UIKit is always the runtime view system,
        // even when running on macOS via Designed for iPad.
        self.init(color: UIColor(color))
    }
}
