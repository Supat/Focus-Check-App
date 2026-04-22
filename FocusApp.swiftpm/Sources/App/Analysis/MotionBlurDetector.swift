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

    /// Heuristic cutoff for surfacing the badge to users.
    var isSignificant: Bool { confidence > 0.25 }
}

/// Detects motion blur by analysing the image's 2D frequency spectrum.
///
/// Motion blur's point-spread function is a line segment; its Fourier transform
/// has periodic zeros perpendicular to the motion direction — so the spectrum
/// has a directional dark band. Defocus (disk PSF) produces concentric Bessel
/// rings, which are roughly isotropic.
///
/// We bin spectral magnitudes by angle over [0, 180). A sharp minimum in one
/// bin relative to the mean indicates directional suppression → motion blur.
/// The motion direction is perpendicular to that minimum.
struct MotionBlurDetector {
    /// FFT side length. Power of two for vDSP radix-2; 256 is a good balance of
    /// speed (~10 ms on A14) and frequency resolution.
    static let fftSide = 256
    static let log2Side: vDSP_Length = 8

    private let ciContext: CIContext

    init(ciContext: CIContext) {
        self.ciContext = ciContext
    }

    func detect(in image: CIImage) -> MotionBlurReport? {
        let side = Self.fftSide
        let rect = CGRect(x: 0, y: 0, width: side, height: side)

        // 1. Center-crop a square region of the source, resample to side×side luma.
        guard let patch = squareLumaPatch(image, size: side) else { return nil }

        var luma = [Float](repeating: 0, count: side * side)
        luma.withUnsafeMutableBytes { ptr in
            ciContext.render(
                patch,
                toBitmap: ptr.baseAddress!,
                rowBytes: side * MemoryLayout<Float>.size,
                bounds: rect,
                format: .Rf,
                colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!
            )
        }

        // 2. Subtract mean so the DC spike doesn't dominate.
        var mean: Float = 0
        vDSP_meanv(luma, 1, &mean, vDSP_Length(side * side))
        var negMean = -mean
        vDSP_vsadd(luma, 1, &negMean, &luma, 1, vDSP_Length(side * side))

        // 3. Apply a 2D Hann window (outer product of 1D windows) to kill edge
        // artifacts that would otherwise produce horizontal + vertical bright
        // crosses in the spectrum and false-trigger the directional detector.
        var window1D = [Float](repeating: 0, count: side)
        vDSP_hann_window(&window1D, vDSP_Length(side), Int32(vDSP_HANN_NORM))
        for y in 0..<side {
            let wy = window1D[y]
            for x in 0..<side {
                luma[y * side + x] *= window1D[x] * wy
            }
        }

        // 4. In-place complex 2D FFT (imaginary input = 0).
        guard let setup = vDSP_create_fftsetup(Self.log2Side, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        defer { vDSP_destroy_fftsetup(setup) }

        var real = luma
        var imag = [Float](repeating: 0, count: side * side)

        let magnitudes: [Float] = real.withUnsafeMutableBufferPointer { realPtr in
            imag.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(
                    realp: realPtr.baseAddress!,
                    imagp: imagPtr.baseAddress!
                )
                vDSP_fft2d_zip(
                    setup, &split,
                    1, vDSP_Stride(side),
                    Self.log2Side, Self.log2Side,
                    FFTDirection(FFT_FORWARD)
                )
                var mags = [Float](repeating: 0, count: side * side)
                vDSP_zvmags(&split, 1, &mags, 1, vDSP_Length(side * side))
                var n = Int32(side * side)
                vvsqrtf(&mags, mags, &n)
                return mags
            }
        }

        // 5. Angular histogram over [0, 180). FFT output is conjugate-symmetric,
        // so the top and bottom halves carry the same info — 180° is enough.
        let numBins = 36  // 5° each
        var bins = [Float](repeating: 0, count: numBins)
        var counts = [Int](repeating: 0, count: numBins)

        let half = side / 2
        let innerSkipSq = 5 * 5         // exclude DC neighborhood
        let outerLimitSq = (half - 4) * (half - 4)

        for y in 0..<side {
            for x in 0..<side {
                // Unshifted FFT coords → centred (DC at origin)
                let sx = (x + half) % side - half
                let sy = (y + half) % side - half
                let r2 = sx * sx + sy * sy
                if r2 < innerSkipSq || r2 > outerLimitSq { continue }

                var angle = atan2(Float(sy), Float(sx)) * (180.0 / .pi)
                if angle < 0 { angle += 180 }
                if angle >= 180 { angle -= 180 }

                let bin = min(numBins - 1, Int(angle / 180.0 * Float(numBins)))
                bins[bin] += magnitudes[y * side + x]
                counts[bin] += 1
            }
        }
        for i in 0..<numBins where counts[i] > 0 {
            bins[i] /= Float(counts[i])
        }

        // 6. Motion blur signature: one bin much dimmer than the mean.
        let mean = bins.reduce(0, +) / Float(numBins)
        guard mean > 0,
              let minEntry = bins.enumerated().min(by: { $0.element < $1.element })
        else { return nil }

        // Confidence = how much the minimum bin falls below the mean.
        let confidence = max(0, (mean - minEntry.element) / mean)

        // Motion direction is perpendicular to the dark band in the spectrum.
        let minAngle = (Float(minEntry.offset) + 0.5) * (180.0 / Float(numBins))
        let motionAngle = (minAngle + 90).truncatingRemainder(dividingBy: 180)

        return MotionBlurReport(angle: motionAngle, confidence: confidence)
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
        let luma = scaled.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])
        return luma.cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
    }
}
