import MetalKit
import CoreImage
import CoreImage.CIFilterBuiltins
import SwiftUI

/// Parameter bag for the compositing pipeline. Shared between the live
/// MTKView renderer and the PNG export path so both produce identical output.
struct FocusCompositeInputs {
    var source: CIImage
    var style: OverlayStyle
    var threshold: Float
    var tint: CIColor
    var focalPlane: Float?
    var sharpness: CIImage?
    var depth: CIImage?
    var motion: CIImage?
    var mosaic: Bool
    var mosaicMode: MosaicMode
    var faces: [CGRect]
    var bodies: [CGRect]
    var groins: [CGRect]
    var eyes: [EyeBar]
    var chests: [CGRect]
    /// Person-silhouette mask stretched to source extent (white = person).
    /// When present, .body mosaic uses the outline instead of a rectangle.
    var personMask: CIImage?
    /// Per-body NudeNet levels, aligned with `bodies`. Empty when the
    /// detector isn't installed. When populated, mosaic modes that
    /// render per body (Body / Chest / Groin / Face) can skip subjects
    /// whose level falls below `nudityGate`.
    var nudityLevels: [NudityLevel]
    /// Minimum level that triggers mosaic. Ignored when `nudityLevels`
    /// is empty (no detector) — in that case the original "every body"
    /// behavior is preserved.
    var nudityGate: NudityLevel
    /// Raw NudeNet detections in source-extent coords. Consumed by the
    /// `.nudity` mosaic mode, which pixelates each detection box
    /// directly rather than going through Vision body assignment.
    var nudityDetections: [NudityDetection]
}

/// Composites the source image with a focus overlay each frame for live
/// display in `MetalView`, and exposes a static `composite(...)` so the
/// export path can produce a matching CIImage without running a MTKView.
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
        // Intentionally cheap on the renderer side — no backing
        // store to reallocate. Push the drawable size onto the
        // view model so toggleZoom() can compute a native (1:1
        // source-to-drawable) zoom factor without round-tripping
        // through GeometryReader / display-scale guesses.
        // MTKViewDelegate callbacks land on the main thread, same
        // run loop as the @MainActor view model.
        MainActor.assumeIsolated {
            viewModel.lastDrawableSize = size
        }
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
        // MTKView.drawableSize is in pixels; DragGesture.translation (the
        // source of zoomPanOffset) is in points. Pass the ratio through
        // so the renderer can convert the pan into pixel space before
        // composing, otherwise the image drifts from the SwiftUI overlay
        // by the device scale factor.
        let pointsToPixels = view.bounds.width > 0
            ? drawableSize.width / view.bounds.width
            : 1

        let content = buildFrame(drawableSize: drawableSize, pointsToPixels: pointsToPixels)
        // Composite over solid black so letterbox bars are defined pixels.
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

    /// Viewport-only state captured alongside `FocusCompositeInputs` for a
    /// single frame. `inputs` is nil when the user hasn't loaded an image
    /// yet — callers render a black canvas in that case.
    private struct RenderSnapshot {
        var inputs: FocusCompositeInputs?
        var overlayHidden: Bool
        var zoomScale: CGFloat
        var zoomAnchor: CGPoint
        var zoomPan: CGSize
    }

    private func buildFrame(drawableSize: CGSize, pointsToPixels: CGFloat) -> CIImage {
        let snapshot: RenderSnapshot = MainActor.assumeIsolated {
            guard let source = viewModel.sourceImage else {
                return RenderSnapshot(
                    inputs: nil,
                    overlayHidden: viewModel.overlayHidden,
                    zoomScale: viewModel.zoomScale,
                    zoomAnchor: viewModel.zoomAnchor,
                    zoomPan: CGSize(
                        width: viewModel.zoomPanOffset.width * pointsToPixels,
                        height: viewModel.zoomPanOffset.height * pointsToPixels
                    )
                )
            }
            // Mosaic fires when the classifier flagged the image AND the
            // user has mosaic enabled, OR when Force Censor is on (which
            // bypasses classifier state entirely).
            let applyMosaic =
                viewModel.forceCensor ||
                ((viewModel.isSensitive == true) && viewModel.mosaicEnabled)
            let inputs = FocusCompositeInputs(
                source: source,
                style: viewModel.style,
                threshold: viewModel.threshold,
                tint: CIColor(color: viewModel.overlayColor) ?? CIColor(red: 1, green: 0.95, blue: 0),
                focalPlane: viewModel.focalPlane,
                sharpness: viewModel.sharpnessOverlay,
                depth: viewModel.depthOverlay,
                motion: viewModel.motionOverlay,
                mosaic: applyMosaic,
                mosaicMode: viewModel.mosaicMode,
                faces: viewModel.faceRectangles,
                bodies: viewModel.bodyRectangles,
                groins: viewModel.groinRectangles,
                eyes: viewModel.eyeBars,
                chests: viewModel.chestRectangles,
                personMask: viewModel.personMask,
                nudityLevels: viewModel.nudityLevels,
                nudityGate: viewModel.nudityGate,
                nudityDetections: viewModel.nudityDetections
            )
            return RenderSnapshot(
                inputs: inputs,
                overlayHidden: viewModel.overlayHidden,
                zoomScale: viewModel.zoomScale,
                zoomAnchor: viewModel.zoomAnchor,
                zoomPan: CGSize(
                    width: viewModel.zoomPanOffset.width * pointsToPixels,
                    height: viewModel.zoomPanOffset.height * pointsToPixels
                )
            )
        }

        guard let inputs = snapshot.inputs else {
            return CIImage(color: CIColor.black).cropped(to: CGRect(origin: .zero, size: drawableSize))
        }
        return Self.composite(
            inputs,
            drawableSize: drawableSize,
            overlayHidden: snapshot.overlayHidden,
            zoomScale: snapshot.zoomScale,
            zoomAnchor: snapshot.zoomAnchor,
            zoomPan: snapshot.zoomPan
        )
    }

    // MARK: - Public composite (shared with export)

    /// Build the composite CIImage for the given inputs. Pure Core Image
    /// recipe — no GPU work happens until the caller hands the result to a
    /// CIContext. The export path uses `drawableSize = source.extent.size`
    /// and `zoomScale = 1` so the PNG reflects the entire image at native
    /// resolution, not the zoomed viewport.
    static func composite(
        _ inputs: FocusCompositeInputs,
        drawableSize: CGSize,
        overlayHidden: Bool = false,
        zoomScale: CGFloat = 1,
        zoomAnchor: CGPoint = CGPoint(x: 0.5, y: 0.5),
        zoomPan: CGSize = .zero
    ) -> CIImage {
        // Mosaic is applied at source resolution before fit/zoom, so the fit
        // transform handles the rest without rect-coordinate gymnastics.
        //
        // When NudeNet has classified each body, per-subject modes skip
        // bodies below the gate. With no NudeNet (`nudityLevels` empty),
        // every body qualifies — preserving pre-NudeNet behavior.
        let gates = MosaicGates(inputs: inputs)
        let baseSource = inputs.mosaic
            ? mosaicked(inputs: inputs, gates: gates)
            : inputs.source

        let fitted = fit(image: baseSource, into: drawableSize,
                         zoom: zoomScale, anchor: zoomAnchor, pan: zoomPan)
        if overlayHidden { return fitted }

        let sharpnessOverlay = inputs.sharpness.map {
            fit(image: $0, into: drawableSize, zoom: zoomScale, anchor: zoomAnchor, pan: zoomPan)
        }
        let depthOverlay = inputs.depth.map {
            fit(image: $0, into: drawableSize, zoom: zoomScale, anchor: zoomAnchor, pan: zoomPan)
        }
        let motionOverlay = inputs.motion.map {
            fit(image: $0, into: drawableSize, zoom: zoomScale, anchor: zoomAnchor, pan: zoomPan)
        }

        let threshold = CGFloat(inputs.threshold)
        let tint = inputs.tint

        switch inputs.style {
        case .off:
            return fitted
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
                                       focalPlane: inputs.focalPlane,
                                       threshold: threshold)
        case .motion:
            return motionComposite(base: fitted, motion: motionOverlay,
                                   threshold: threshold, tint: tint)
        }
    }

    // MARK: - Mosaic dispatch

    /// Per-subject gating output. Each region array has been filtered
    /// against the per-body NudeNet level so subjects below the gate
    /// are omitted before any mosaic mode runs. With NudeNet absent
    /// (`inputs.nudityLevels` empty), the gate is a no-op and every
    /// region passes through.
    private struct MosaicGates {
        let bodies: [CGRect]
        let faces: [CGRect]
        let chests: [CGRect]
        let groins: [CGRect]
        let eyes: [EyeBar]

        init(inputs: FocusCompositeInputs) {
            self.bodies = filterBodies(bodies: inputs.bodies,
                                       levels: inputs.nudityLevels,
                                       gate: inputs.nudityGate)
            self.faces = filterByBodyLevel(regions: inputs.faces,
                                           bodies: inputs.bodies,
                                           levels: inputs.nudityLevels,
                                           gate: inputs.nudityGate)
            self.chests = filterByBodyLevel(regions: inputs.chests,
                                            bodies: inputs.bodies,
                                            levels: inputs.nudityLevels,
                                            gate: inputs.nudityGate)
            self.groins = filterByBodyLevel(regions: inputs.groins,
                                            bodies: inputs.bodies,
                                            levels: inputs.nudityLevels,
                                            gate: inputs.nudityGate)
            self.eyes = filterEyesByBodyLevel(eyes: inputs.eyes,
                                              bodies: inputs.bodies,
                                              levels: inputs.nudityLevels,
                                              gate: inputs.nudityGate)
        }
    }

    /// Apply the per-mode mosaic against `inputs.source`. Caller is
    /// responsible for the `inputs.mosaic` early-return — this is
    /// only invoked when mosaic should fire.
    private static func mosaicked(inputs: FocusCompositeInputs,
                                  gates: MosaicGates) -> CIImage {
        switch inputs.mosaicMode {
        case .tabloid:
            return mosaicTabloid(inputs: inputs, gates: gates)
        case .eyes:
            guard !gates.eyes.isEmpty else { return inputs.source }
            return blackBarOverlay(source: inputs.source, bars: gates.eyes)
        case .face:
            guard !gates.faces.isEmpty else { return inputs.source }
            return regionMosaic(source: inputs.source, regions: gates.faces)
        case .chest:
            guard !gates.chests.isEmpty else { return inputs.source }
            return regionMosaic(source: inputs.source, regions: gates.chests)
        case .groin:
            guard !gates.groins.isEmpty else { return inputs.source }
            return regionMosaic(source: inputs.source, regions: gates.groins)
        case .body:
            return mosaicBody(inputs: inputs, gates: gates)
        case .privy:
            return mosaicPrivy(inputs: inputs)
        case .nudity:
            return mosaicNudity(inputs: inputs)
        case .jacket:
            return mosaicJacket(inputs: inputs, gates: gates)
        }
    }

    /// Eye black bar + groin pixelation.
    ///
    /// Groin source priority is per-subject rather than global:
    /// pose-derived hip-joint estimates first (preferred — they're
    /// calibrated to anatomy and include clothed subjects), with
    /// NudeNet GENITALIA_* detections filling gaps for any subject
    /// the pose pass missed (occluded body, sideways framing, tight
    /// crop). A NudeNet detection is added only when no existing
    /// pose-derived groin already covers its area, so the priority
    /// isn't "all pose OR all NudeNet" — it's "pose where available,
    /// NudeNet to fill in".
    private static func mosaicTabloid(inputs: FocusCompositeInputs,
                                      gates: MosaicGates) -> CIImage {
        var result = inputs.source
        if !gates.eyes.isEmpty {
            result = blackBarOverlay(source: result, bars: gates.eyes)
        }
        var groinRegions = gates.groins
        let genitalDetections = gateDetections(
            inputs.nudityDetections,
            bodies: inputs.bodies,
            levels: inputs.nudityLevels,
            gate: inputs.nudityGate
        ).filter { $0.label.uppercased().contains("GENITALIA") }
        for det in genitalDetections {
            let alreadyCovered = groinRegions.contains { $0.intersects(det.rect) }
            if !alreadyCovered {
                groinRegions.append(det.rect)
            }
        }
        if !groinRegions.isEmpty {
            result = regionMosaic(source: result, regions: groinRegions)
        }
        return result
    }

    /// Body silhouette / body-rect mosaic with graceful fallbacks.
    /// Skips entirely when NudeNet excluded every body. Uses the
    /// person-segmentation mask when no per-subject filtering kicked
    /// in, otherwise falls back to body rectangles so clothed
    /// subjects don't end up under the silhouette.
    private static func mosaicBody(inputs: FocusCompositeInputs,
                                   gates: MosaicGates) -> CIImage {
        if gates.bodies.isEmpty && !inputs.nudityLevels.isEmpty {
            return inputs.source
        }
        if let mask = inputs.personMask,
           gates.bodies.count == inputs.bodies.count {
            return silhouetteMosaic(source: inputs.source, mask: mask)
        }
        if !gates.bodies.isEmpty {
            return regionMosaic(source: inputs.source, regions: gates.bodies)
        }
        if let mask = inputs.personMask {
            return silhouetteMosaic(source: inputs.source, mask: mask)
        }
        if !inputs.faces.isEmpty {
            return regionMosaic(source: inputs.source, regions: inputs.faces)
        }
        return inputs.source
    }

    /// Privy: pixelate every per-subject NudeNet detection except
    /// secondary regions the user explicitly spared — armpits,
    /// breasts, belly, feet. Genitalia / anus / buttocks / face-
    /// label detections still get mosaiced. Per-subject gate
    /// applies so detections from below-gate subjects are dropped
    /// the same way as the body / chest / groin modes.
    private static func mosaicPrivy(inputs: FocusCompositeInputs) -> CIImage {
        let keep = gateDetections(
            inputs.nudityDetections,
            bodies: inputs.bodies,
            levels: inputs.nudityLevels,
            gate: inputs.nudityGate
        ).filter { det in
            let upper = det.label.uppercased()
            return !(upper.contains("ARMPITS")
                  || upper.contains("BREAST")
                  || upper.contains("BELLY")
                  || upper.contains("FEET"))
        }
        guard !keep.isEmpty else { return inputs.source }
        return regionMosaic(source: inputs.source, regions: keep.map(\.rect))
    }

    /// Pixelate every per-subject NudeNet detection box. Per-subject
    /// gate is applied first so detections belonging to below-gate
    /// subjects are skipped alongside the body / chest / groin modes.
    private static func mosaicNudity(inputs: FocusCompositeInputs) -> CIImage {
        let gated = gateDetections(
            inputs.nudityDetections,
            bodies: inputs.bodies,
            levels: inputs.nudityLevels,
            gate: inputs.nudityGate
        )
        guard !gated.isEmpty else { return inputs.source }
        return regionMosaic(source: inputs.source, regions: gated.map(\.rect))
    }

    /// Eye black bar + genital mosaic. Covers anonymity at the head
    /// and explicit anatomy at the crotch while leaving the rest of
    /// the composition visible — useful for evidence / documentary
    /// frames where the subject must remain unidentifiable AND
    /// unexposed. Both paths respect the per-subject gate.
    private static func mosaicJacket(inputs: FocusCompositeInputs,
                                     gates: MosaicGates) -> CIImage {
        var result = inputs.source
        if !gates.eyes.isEmpty {
            // 95 % opacity — Jacket is a softer-anonymity preset than
            // Tabloid; the slightly translucent strip lets a hint of
            // eye/face structure through for context while still
            // breaking identifiability.
            result = blackBarOverlay(source: result, bars: gates.eyes, alpha: 0.95)
        }
        let genitalDetections = gateDetections(
            inputs.nudityDetections,
            bodies: inputs.bodies,
            levels: inputs.nudityLevels,
            gate: inputs.nudityGate
        ).filter { $0.label.uppercased().contains("GENITALIA_EXPOSED") }
        if !genitalDetections.isEmpty {
            result = regionMosaic(source: result, regions: genitalDetections.map(\.rect))
        }
        return result
    }

    // MARK: - Style pipelines

    private static func peakingComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                                         threshold: CGFloat, tint: CIColor) -> CIImage {
        guard let focusMask = maskForMode(sharpness: sharpness, depth: depth, threshold: threshold)
        else { return base }

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

    private static func heatmapComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                                         threshold: CGFloat) -> CIImage {
        let magnitude: CIImage
        switch (sharpness, depth) {
        case (let s?, let d?):
            let combine = CIFilter.maximumCompositing()
            combine.inputImage = s
            combine.backgroundImage = d
            magnitude = combine.outputImage ?? s
        case (let s?, nil): magnitude = s
        case (nil, let d?): magnitude = d
        case (nil, nil):    return base
        }
        let cropped = magnitude.clampedToExtent().cropped(to: base.extent)

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

    private static func maskComposite(base: CIImage, sharpness: CIImage?, depth: CIImage?,
                                      threshold: CGFloat, tint: CIColor) -> CIImage {
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

    private static func motionComposite(base: CIImage, motion: CIImage?,
                                        threshold: CGFloat, tint: CIColor) -> CIImage {
        guard let motion else { return base }
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

    private static func focusErrorComposite(base: CIImage, depth: CIImage?,
                                            focalPlane: Float?, threshold: CGFloat) -> CIImage {
        guard let depth, let focalPlane else { return base }
        let epsilon: CGFloat = 0.01 + 0.10 * threshold
        let gain: CGFloat = 6.0

        let gray = depth.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])

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

        let red = CIImage(color: CIColor(red: 1, green: 0.1, blue: 0.1)).cropped(to: base.extent)
        let redStep = CIFilter.blendWithMask()
        redStep.inputImage = red
        redStep.backgroundImage = base
        redStep.maskImage = closeMask
        let afterClose = redStep.outputImage ?? base

        let blue = CIImage(color: CIColor(red: 0.1, green: 0.4, blue: 1)).cropped(to: base.extent)
        let blueStep = CIFilter.blendWithMask()
        blueStep.inputImage = blue
        blueStep.backgroundImage = afterClose
        blueStep.maskImage = farMask
        return blueStep.outputImage ?? afterClose
    }

    // MARK: - Mask derivation

    private static func maskForMode(sharpness: CIImage?, depth: CIImage?, threshold: CGFloat) -> CIImage? {
        switch (sharpness, depth) {
        case (let s?, nil):
            return thresholded(s, at: threshold)
        case (nil, let d?):
            return thresholded(d, at: threshold)
        case (let s?, let d?):
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

    private static func thresholded(_ image: CIImage, at t: CGFloat) -> CIImage {
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

    /// 16³ RGBA Float cube for `CIFilter.colorCube`. Precomputed once at
    /// first access — the LUT depends only on compile-time constants, so
    /// rebuilding it every heatmap frame (4096 iterations + 64 KB alloc)
    /// was wasted work.
    private static let viridisCubeData: Data = {
        let size = 16
        var cube = [Float](repeating: 0, count: size * size * size * 4)
        for b in 0..<size {
            for g in 0..<size {
                for r in 0..<size {
                    let i = (b * size * size + g * size + r) * 4
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
        return cube.withUnsafeBufferPointer { Data(buffer: $0) }
    }()
    private static let viridisCubeDimension: Float = 16

    private static func viridis(_ image: CIImage) -> CIImage {
        let filter = CIFilter.colorCube()
        filter.cubeDimension = viridisCubeDimension
        filter.cubeData = viridisCubeData
        filter.inputImage = image
        return filter.outputImage ?? image
    }

    private static func viridisColor(at t: Float) -> (r: Float, g: Float, b: Float) {
        let stops: [(Float, Float, Float, Float)] = [
            (0.00, 0.267, 0.005, 0.329),
            (0.25, 0.229, 0.322, 0.545),
            (0.50, 0.127, 0.566, 0.551),
            (0.75, 0.369, 0.789, 0.383),
            (1.00, 0.993, 0.956, 0.144)
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

    // MARK: - NudeNet gating

    /// Returns bodies whose NudeNet level meets or exceeds `gate`. When
    /// `levels` is empty (no NudeNet model installed), every body passes —
    /// preserving behaviour from before the per-subject classifier existed.
    private static func filterBodies(bodies: [CGRect],
                                     levels: [NudityLevel],
                                     gate: NudityLevel) -> [CGRect] {
        guard !levels.isEmpty else { return bodies }
        return zip(bodies, levels)
            .compactMap { $1 >= gate ? $0 : nil }
    }

    /// Per-subject gate for NudeNet detections. Each detection gets
    /// attributed to whichever body it overlaps most (same rule
    /// NudityDetector uses internally), and survives only if that
    /// body's level meets the gate. Detections that don't attribute
    /// to any body pass through — same conservative default as
    /// `filterByBodyLevel`, since dropping an unattributed detection
    /// could skip a subject the body detector happened to miss.
    private static func gateDetections(_ detections: [NudityDetection],
                                       bodies: [CGRect],
                                       levels: [NudityLevel],
                                       gate: NudityLevel) -> [NudityDetection] {
        guard !levels.isEmpty, !bodies.isEmpty else { return detections }
        return detections.filter { det in
            var bestIdx: Int? = nil
            var bestArea: CGFloat = 0
            for (i, body) in bodies.enumerated() {
                let inter = body.intersection(det.rect)
                guard !inter.isNull else { continue }
                let area = inter.width * inter.height
                if area > bestArea {
                    bestArea = area
                    bestIdx = i
                }
            }
            guard let idx = bestIdx, idx < levels.count else { return true }
            return levels[idx] >= gate
        }
    }

    /// Assign each region (face / chest / groin) to whichever body
    /// contains its center; keep the region only when that body's level
    /// qualifies. Regions with no containing body are kept as a safety
    /// default — better to mosaic them than risk missing something the
    /// detector simply failed to attribute.
    private static func filterByBodyLevel(regions: [CGRect],
                                          bodies: [CGRect],
                                          levels: [NudityLevel],
                                          gate: NudityLevel) -> [CGRect] {
        guard !levels.isEmpty, !bodies.isEmpty else { return regions }
        return regions.filter { region in
            let center = CGPoint(x: region.midX, y: region.midY)
            guard let idx = bodies.firstIndex(where: { $0.contains(center) }),
                  idx < levels.count
            else { return true }
            return levels[idx] >= gate
        }
    }

    private static func filterEyesByBodyLevel(eyes: [EyeBar],
                                              bodies: [CGRect],
                                              levels: [NudityLevel],
                                              gate: NudityLevel) -> [EyeBar] {
        guard !levels.isEmpty, !bodies.isEmpty else { return eyes }
        return eyes.filter { bar in
            guard let idx = bodies.firstIndex(where: { $0.contains(bar.center) }),
                  idx < levels.count
            else { return true }
            return levels[idx] >= gate
        }
    }

    // MARK: - Mosaic primitives

    private static func blackBarOverlay(source: CIImage,
                                        bars: [EyeBar],
                                        alpha: CGFloat = 1.0) -> CIImage {
        // Pre-multiplied black at the given opacity. Default 1.0
        // matches the original tabloid behavior; the Jacket mode
        // overrides to 0.95 (slightly translucent) so the eye bar
        // lets a hint of face structure through — softer anonymity
        // than the solid Tabloid bar.
        let barColor = CIColor(red: 0, green: 0, blue: 0, alpha: alpha)
        var result = source
        for bar in bars {
            let axisAligned = CIImage(color: barColor).cropped(
                to: CGRect(origin: .zero, size: bar.size)
            )
            let transform = CGAffineTransform(
                translationX: -bar.size.width / 2,
                y: -bar.size.height / 2
            )
            .concatenating(CGAffineTransform(rotationAngle: bar.angleRadians))
            .concatenating(CGAffineTransform(
                translationX: bar.center.x,
                y: bar.center.y
            ))
            let oriented = axisAligned.transformed(by: transform)

            let over = CIFilter.sourceOverCompositing()
            over.inputImage = oriented
            over.backgroundImage = result
            result = over.outputImage ?? result
        }
        return result.cropped(to: source.extent)
    }

    /// Pixelate the source and composite it through Vision's silhouette
    /// mask. The mask is also pixelated on the same grid and thresholded
    /// so every pixelation block is either fully covered or fully clear —
    /// otherwise the mask clips blocks mid-pixel and the silhouette edge
    /// reads as smooth instead of block-jagged.
    private static func silhouetteMosaic(source: CIImage, mask: CIImage) -> CIImage {
        let blockSize = mosaicTileSize(source: source)
        let center = CGPoint(x: source.extent.midX, y: source.extent.midY)

        let pixelateImage = CIFilter.pixellate()
        pixelateImage.inputImage = source.clampedToExtent()
        pixelateImage.scale = blockSize
        pixelateImage.center = center
        let pixelated = (pixelateImage.outputImage ?? source).cropped(to: source.extent)

        let pixelateMask = CIFilter.pixellate()
        pixelateMask.inputImage = mask.clampedToExtent()
        pixelateMask.scale = blockSize
        pixelateMask.center = center
        let blockMask = (pixelateMask.outputImage ?? mask).cropped(to: source.extent)

        // Hard 0.5 threshold: any block whose average mask coverage exceeds
        // half gets fully pixelated. Gain 50 + bias -25 is a steep step
        // function around 0.5 that clamps to 0/1 in the following filter.
        let stepped = blockMask.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 50, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 50, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 50, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBiasVector": CIVector(x: -25, y: -25, z: -25, w: 0)
        ]).applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])

        let blend = CIFilter.blendWithMask()
        blend.inputImage = pixelated
        blend.backgroundImage = source
        blend.maskImage = stepped
        return (blend.outputImage ?? source).cropped(to: source.extent)
    }

    /// Unified tile-size base for every mosaic path. Returns the
    /// pixelation-block edge length in pixels, computed strictly from
    /// the source image's long side so the same photo produces the
    /// same tile size regardless of how many subjects are present or
    /// how big each is. Modes that want chunkier blocks (Nudity, Privy)
    /// pass a `multiplier` above 1.
    private static func mosaicTileSize(source: CIImage, multiplier: CGFloat = 1.0) -> Float {
        let longerSide = max(source.extent.width, source.extent.height)
        return max(Float(longerSide / 96 * multiplier), 4)
    }

    private static func regionMosaic(source: CIImage,
                                     regions: [CGRect],
                                     scaleMultiplier: CGFloat = 1.0) -> CIImage {
        guard !regions.isEmpty else { return source }

        let pixelate = CIFilter.pixellate()
        pixelate.inputImage = source.clampedToExtent()
        pixelate.scale = mosaicTileSize(source: source, multiplier: scaleMultiplier)
        pixelate.center = CGPoint(x: source.extent.midX, y: source.extent.midY)
        let pixelated = (pixelate.outputImage ?? source).cropped(to: source.extent)

        var mask: CIImage = CIImage(color: CIColor.black).cropped(to: source.extent)
        for region in regions {
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

    private static func fit(image: CIImage, into size: CGSize,
                            zoom: CGFloat = 1.0,
                            anchor: CGPoint = CGPoint(x: 0.5, y: 0.5),
                            pan: CGSize = .zero) -> CIImage {
        let sx = size.width / image.extent.width
        let sy = size.height / image.extent.height
        let s = min(sx, sy)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: s, y: s))
        let offsetX = (size.width - scaled.extent.width) / 2 - scaled.extent.minX
        let offsetY = (size.height - scaled.extent.height) / 2 - scaled.extent.minY
        let fitted = scaled.transformed(by: CGAffineTransform(translationX: offsetX, y: offsetY))

        guard zoom > 1.001 else { return fitted }

        let ax = anchor.x * size.width
        let ay = (1 - anchor.y) * size.height
        // Pan arrives in view-coords (SwiftUI Y-down); flip Y so it
        // composes with CIImage Y-up after the anchor-centred zoom.
        let transform = CGAffineTransform(translationX: -ax, y: -ay)
            .concatenating(CGAffineTransform(scaleX: zoom, y: zoom))
            .concatenating(CGAffineTransform(translationX: ax, y: ay))
            .concatenating(CGAffineTransform(translationX: pan.width, y: -pan.height))

        return fitted.transformed(by: transform)
            .cropped(to: CGRect(origin: .zero, size: size))
    }
}

extension CIColor {
    convenience init?(color: Color) {
        self.init(color: UIColor(color))
    }
}
