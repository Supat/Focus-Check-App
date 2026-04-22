import Accelerate
import CoreImage

/// Per-image motion-blur signal extracted from the frequency spectrum.
struct MotionBlurReport: Equatable {
    /// Estimated motion direction in degrees, [0, 180).
    /// Ambiguous by 180° because motion blur's PSF is symmetric about its midpoint.
    let angle: Float
    /// How strongly directional the spectrum is: 0 = isotropic (defocus or sharp),
    /// 1 = perfectly directional (strong motion blur along a single axis).
    let confidence: Float

    /// Heuristic cutoff for surfacing the badge to users. Set high enough that
    /// directional-texture false positives (fences, waves, stripes) don't
    /// consistently trigger it.
    var isSignificant: Bool { confidence > 0.4 }
}

/// Detects motion blur by analysing the image's 2D frequency spectrum.
///
/// Motion blur's point-spread function is a line segment; its Fourier transform
/// has periodic zeros perpendicular to the motion direction — so the spectrum
/// has a directional dark band. Defocus (disk PSF) produces concentric Bessel
/// rings, which are roughly isotropic.
///
/// `detect` reports a single global reading from a center-cropped square patch,
/// used for the info badge. `detectMap` tiles the image into overlapping patches
/// and returns a coarse grayscale confidence grid suitable for upscaling into
/// an overlay.
struct MotionBlurDetector {
    // Global-detect FFT side. 256 is a good balance of speed (~10 ms on A14)
    // and frequency resolution for the badge.
    static let globalSide = 256
    static let globalLog2: vDSP_Length = 8

    // Tiled map: resample the source into a fixed-size square, then slide a
    // smaller FFT window across it. 512 × 128-patch × 64-stride ≈ 49 patches.
    static let mapAnalysisSide = 512
    static let mapPatchSide = 128
    static let mapPatchLog2: vDSP_Length = 7
    static let mapPatchStride = 64

    private let ciContext: CIContext

    init(ciContext: CIContext) {
        self.ciContext = ciContext
    }

    // MARK: - Global detection (badge)

    func detect(in image: CIImage) -> MotionBlurReport? {
        let side = Self.globalSide
        guard let patch = squareLumaPatch(image, size: side) else { return nil }

        var luma = renderLumaBuffer(patch, size: side)
        subtractMean(&luma)
        applyHannWindow(&luma, side: side)

        guard let setup = vDSP_create_fftsetup(Self.globalLog2, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        defer { vDSP_destroy_fftsetup(setup) }

        return analyzeSpectrum(
            realInit: luma,
            sideSize: side,
            log2Size: Self.globalLog2,
            setup: setup
        )
    }

    // MARK: - Tiled map

    /// Tile the source, run motion detection per patch, pack per-patch confidence
    /// into a small grayscale CIImage (confidence ∈ [0,1] → luminance 0..255).
    /// Returns `nil` if rendering or FFT setup fails.
    func detectMap(in image: CIImage) -> CIImage? {
        let analysisSide = Self.mapAnalysisSide
        let patchSide = Self.mapPatchSide
        let stride = Self.mapPatchStride
        let patchesPerSide = (analysisSide - patchSide) / stride + 1
        guard patchesPerSide > 0 else { return nil }

        // Render a full analysis-sized luma buffer (stretched to square — mild
        // aspect distortion doesn't meaningfully affect frequency-domain analysis
        // and keeps the grid upscaling trivial).
        guard let squared = stretchedLumaPatch(image, size: analysisSide) else { return nil }
        let fullLuma = renderLumaBuffer(squared, size: analysisSide)

        guard let setup = vDSP_create_fftsetup(Self.mapPatchLog2, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        defer { vDSP_destroy_fftsetup(setup) }

        // Reusable per-patch buffers.
        var realBuf = [Float](repeating: 0, count: patchSide * patchSide)
        var hann = [Float](repeating: 0, count: patchSide)
        vDSP_hann_window(&hann, vDSP_Length(patchSide), Int32(vDSP_HANN_NORM))

        let gridCount = patchesPerSide * patchesPerSide
        var confidenceGrid = [Float](repeating: 0, count: gridCount)

        for py in 0..<patchesPerSide {
            for px in 0..<patchesPerSide {
                let x0 = px * stride
                let y0 = py * stride
                copyPatch(from: fullLuma,
                          fullSide: analysisSide,
                          at: (x0, y0),
                          size: patchSide,
                          into: &realBuf)
                subtractMean(&realBuf)
                applyWindow(&realBuf, window1D: hann, side: patchSide)

                let report = analyzeSpectrum(
                    realInit: realBuf,
                    sideSize: patchSide,
                    log2Size: Self.mapPatchLog2,
                    setup: setup
                )
                confidenceGrid[py * patchesPerSide + px] = report?.confidence ?? 0
            }
        }

        // Pack the grid into an 8-bit grayscale CIImage.
        var bytes = [UInt8](repeating: 0, count: gridCount)
        for i in 0..<gridCount {
            let v = min(1, max(0, confidenceGrid[i]))
            bytes[i] = UInt8(v * 255)
        }
        let data = Data(bytes)
        return CIImage(
            bitmapData: data,
            bytesPerRow: patchesPerSide,
            size: CGSize(width: patchesPerSide, height: patchesPerSide),
            format: .L8,
            colorSpace: CGColorSpaceCreateDeviceGray()
        )
    }

    // MARK: - Shared FFT + angular-histogram core

    /// Runs the 2D FFT on a pre-windowed, mean-subtracted real buffer and
    /// reduces the magnitude spectrum to (angle, confidence).
    private func analyzeSpectrum(realInit: [Float],
                                 sideSize: Int,
                                 log2Size: vDSP_Length,
                                 setup: FFTSetup) -> MotionBlurReport? {
        var real = realInit
        var imag = [Float](repeating: 0, count: sideSize * sideSize)

        let magnitudes: [Float] = real.withUnsafeMutableBufferPointer { realPtr in
            imag.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )
                vDSP_fft2d_zip(
                    setup, &split,
                    1, vDSP_Stride(sideSize),
                    log2Size, log2Size,
                    FFTDirection(FFT_FORWARD)
                )
                var mags = [Float](repeating: 0, count: sideSize * sideSize)
                vDSP_zvmags(&split, 1, &mags, 1, vDSP_Length(sideSize * sideSize))
                var n = Int32(sideSize * sideSize)
                vvsqrtf(&mags, mags, &n)
                return mags
            }
        }

        // Angular histogram over [0, 180) — FFT magnitude is conjugate symmetric.
        let numBins = 36
        var bins = [Float](repeating: 0, count: numBins)
        var counts = [Int](repeating: 0, count: numBins)

        let half = sideSize / 2
        let innerSkipSq = 5 * 5
        let outerLimitSq = (half - 4) * (half - 4)

        for y in 0..<sideSize {
            for x in 0..<sideSize {
                let sx = (x + half) % sideSize - half
                let sy = (y + half) % sideSize - half
                let r2 = sx * sx + sy * sy
                if r2 < innerSkipSq || r2 > outerLimitSq { continue }

                var angle = atan2(Float(sy), Float(sx)) * (180.0 / .pi)
                if angle < 0 { angle += 180 }
                if angle >= 180 { angle -= 180 }
                let bin = min(numBins - 1, Int(angle / 180.0 * Float(numBins)))
                bins[bin] += magnitudes[y * sideSize + x]
                counts[bin] += 1
            }
        }
        for i in 0..<numBins where counts[i] > 0 {
            bins[i] /= Float(counts[i])
        }

        let mean = bins.reduce(0, +) / Float(numBins)
        guard mean > 0,
              let minEntry = bins.enumerated().min(by: { $0.element < $1.element })
        else { return nil }

        let confidence = max(0, (mean - minEntry.element) / mean)
        let minAngle = (Float(minEntry.offset) + 0.5) * (180.0 / Float(numBins))
        let motionAngle = (minAngle + 90).truncatingRemainder(dividingBy: 180)

        return MotionBlurReport(angle: motionAngle, confidence: confidence)
    }

    // MARK: - Helpers

    private func renderLumaBuffer(_ patch: CIImage, size: Int) -> [Float] {
        var buf = [Float](repeating: 0, count: size * size)
        let rect = CGRect(x: 0, y: 0, width: size, height: size)
        buf.withUnsafeMutableBytes { ptr in
            ciContext.render(
                patch,
                toBitmap: ptr.baseAddress!,
                rowBytes: size * MemoryLayout<Float>.size,
                bounds: rect,
                format: .Rf,
                colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!
            )
        }
        return buf
    }

    private func subtractMean(_ buffer: inout [Float]) {
        var mean: Float = 0
        vDSP_meanv(buffer, 1, &mean, vDSP_Length(buffer.count))
        var neg = -mean
        vDSP_vsadd(buffer, 1, &neg, &buffer, 1, vDSP_Length(buffer.count))
    }

    private func applyHannWindow(_ buffer: inout [Float], side: Int) {
        var window1D = [Float](repeating: 0, count: side)
        vDSP_hann_window(&window1D, vDSP_Length(side), Int32(vDSP_HANN_NORM))
        applyWindow(&buffer, window1D: window1D, side: side)
    }

    /// Multiplies an in-memory side×side buffer by the outer product of `window1D`
    /// with itself (2D separable window).
    private func applyWindow(_ buffer: inout [Float], window1D: [Float], side: Int) {
        for y in 0..<side {
            let wy = window1D[y]
            for x in 0..<side {
                buffer[y * side + x] *= window1D[x] * wy
            }
        }
    }

    private func copyPatch(from full: [Float],
                           fullSide: Int,
                           at origin: (Int, Int),
                           size: Int,
                           into dest: inout [Float]) {
        let (x0, y0) = origin
        for y in 0..<size {
            let srcStart = (y0 + y) * fullSide + x0
            let dstStart = y * size
            for x in 0..<size {
                dest[dstStart + x] = full[srcStart + x]
            }
        }
    }

    /// Center-crop to a square, resample to `size`×`size`, project luma into R.
    private func squareLumaPatch(_ image: CIImage, size: Int) -> CIImage? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let squareSide = min(extent.width, extent.height)
        let cropRect = CGRect(
            x: extent.minX + (extent.width - squareSide) / 2,
            y: extent.minY + (extent.height - squareSide) / 2,
            width: squareSide, height: squareSide
        )
        let cropped = image.cropped(to: cropRect)
        let translated = cropped.transformed(
            by: CGAffineTransform(translationX: -cropped.extent.minX, y: -cropped.extent.minY)
        )
        let scale = CGFloat(size) / squareSide
        let scaled = translated.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return luma(scaled).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
    }

    /// Non-uniform stretch to `size`×`size` — used for the tiled map so the
    /// per-patch grid aligns with the source aspect after upscaling.
    private func stretchedLumaPatch(_ image: CIImage, size: Int) -> CIImage? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let translated = image.transformed(
            by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
        )
        let sx = CGFloat(size) / extent.width
        let sy = CGFloat(size) / extent.height
        let scaled = translated.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        return luma(scaled).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
    }

    private func luma(_ image: CIImage) -> CIImage {
        image.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])
    }
}
