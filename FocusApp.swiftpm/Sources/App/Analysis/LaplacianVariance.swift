import Metal
import MetalPerformanceShaders
import CoreImage
import CoreImage.CIFilterBuiltins

/// Classical sharpness pipeline:
///     Gaussian blur (σ≈1) → Laplacian → per-tile mean/variance
/// Emits a single-channel R16F texture where each pixel's magnitude encodes local sharpness.
/// All work happens on one command buffer; the caller owns the output texture's lifetime.
struct LaplacianVariance {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private let gaussian: MPSImageGaussianBlur
    private let laplacian: MPSImageLaplacian
    private let median: MPSImageMedian

    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        self.gaussian = MPSImageGaussianBlur(device: device, sigma: 1.0)
        self.laplacian = MPSImageLaplacian(device: device)
        // 3×3 median — CLAUDE.md rule: kills shadow/noise false positives.
        self.median = MPSImageMedian(device: device, kernelDiameter: 3)
    }

    /// Analysis resolution (long side). CLAUDE.md rule.
    static let analysisLongSide: CGFloat = 1024

    /// Compute a sharpness magnitude map at analysis resolution.
    ///
    /// - Parameter source: a full-resolution luminance-ready `CIImage`.
    /// - Returns: an `R16Float` texture sized to `analysisLongSide` on its longer dimension.
    func sharpnessMap(from source: CIImage, ciContext: CIContext) throws -> MTLTexture {
        let extent = source.extent
        let scale = Self.analysisLongSide / max(extent.width, extent.height)
        let scaled = source.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let outExtent = scaled.extent.integral

        let width = Int(outExtent.width)
        let height = Int(outExtent.height)

        let grayscaleDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        grayscaleDesc.usage = [.shaderRead, .shaderWrite]
        grayscaleDesc.storageMode = .private

        guard let input = device.makeTexture(descriptor: grayscaleDesc),
              let blurred = device.makeTexture(descriptor: grayscaleDesc),
              let edges = device.makeTexture(descriptor: grayscaleDesc),
              let denoised = device.makeTexture(descriptor: grayscaleDesc),
              let commandBuffer = commandQueue.makeCommandBuffer()
        else { throw AnalysisError.resourceAllocationFailed }

        // Render grayscale luma directly into the input texture using Core Image.
        let luma = CIFilter.colorMatrix()
        luma.inputImage = scaled
        // Rec. 709 luma weights in the R channel; zeros elsewhere — output is single-channel-like.
        luma.rVector = CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0)
        luma.gVector = CIVector(x: 0, y: 0, z: 0, w: 0)
        luma.bVector = CIVector(x: 0, y: 0, z: 0, w: 0)
        luma.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)

        guard let lumaImage = luma.outputImage else {
            throw AnalysisError.coreImageFailure
        }

        ciContext.render(
            lumaImage,
            to: input,
            commandBuffer: commandBuffer,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpace(name: CGColorSpace.linearSRGB)!
        )

        gaussian.encode(commandBuffer: commandBuffer, sourceTexture: input, destinationTexture: blurred)
        laplacian.encode(commandBuffer: commandBuffer, sourceTexture: blurred, destinationTexture: edges)
        median.encode(commandBuffer: commandBuffer, sourceTexture: edges, destinationTexture: denoised)

        commandBuffer.commit()
        // Caller may read via CIImage(mtlTexture:); no need to wait on the CPU here.
        return denoised
    }
}

enum AnalysisError: LocalizedError {
    case metalUnavailable
    case resourceAllocationFailed
    case coreImageFailure
    case imageDecodeFailed
    case modelMissing
    case modelLoadFailed(String)

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:         return "Metal is not available on this device."
        case .resourceAllocationFailed: return "Could not allocate analysis textures."
        case .coreImageFailure:         return "Core Image pipeline failed."
        case .imageDecodeFailed:        return "Could not decode the selected image."
        case .modelMissing:             return "Depth model is not bundled. Using sharpness-only mode."
        case .modelLoadFailed(let s):   return "Depth model failed to load: \(s)"
        }
    }
}
