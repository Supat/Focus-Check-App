import Foundation
import CoreImage
import CoreML

/// Per-face age prediction from shamangary/SSR-Net. SSR-Net is
/// age-only — gender comes from NudeNet's FACE_* branch via
/// `nudityGenders` at the UI layer.
struct AgePrediction: Hashable, Sendable {
    /// Predicted age in years, clamped to [0, 100]. SSR-Net outputs
    /// a single scalar (stage-sum soft regression), so unlike the
    /// earlier 101-bin classifier there's no distribution to
    /// integrate — what you see is what the model committed to.
    let age: Float
}

/// Thin wrapper around SSR-Net. Same shape as the other per-face
/// tiers — `FocusAnalyzer` iterates face rectangles and hands each
/// one in. Empty array out when the model isn't installed; nil
/// entries for faces the crop couldn't cover.
struct AgeEstimator {
    private var model: SSRNetModel? { SSRNetModel.shared }

    var isReady: Bool { ModelArchive.age.isInstalled() }

    func warm() -> Bool { model != nil }

    /// `rolls` is accepted for API parity with the sibling per-face
    /// tiers but is **not** applied — SSR-Net's training pipeline
    /// fed axis-aligned crops from MTCNN / LBP cascades and relied
    /// on random in-plane rotation augmentation for roll-invariance.
    func estimate(faces: [CGRect],
                  rolls _: [CGFloat],
                  in image: CIImage,
                  ciContext _: CIContext) -> [AgePrediction?] {
        guard let model else { return [] }
        guard !faces.isEmpty else { return [] }
        return faces.map { face -> AgePrediction? in
            model.predict(face: face, source: image)
        }
    }
}

// MARK: - Model wrapper

private final class SSRNetModel {
    static var shared: SSRNetModel? = {
        try? SSRNetModel()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    /// SSR-Net trained on 64² crops. Much smaller than EfficientNet's
    /// 224² but still captures enough of the face for a regression
    /// head after the three MaxPool/AvgPool stages.
    private let inputSize = CGSize(width: 64, height: 64)
    /// Dedicated sRGB-working-space CIContext so the rendered buffer
    /// bytes land as proper gamma-encoded sRGB — the shared analyzer
    /// context uses extendedLinearDisplayP3 which produced ~30 %
    /// darker-than-reality pixels on the retired yu4u model and
    /// pushed it into saturation. SSR-Net has less baked-in
    /// preprocessing so it's even more sensitive to this.
    private let renderContext: CIContext = {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        return CIContext(options: [
            .workingColorSpace: sRGB,
            .outputColorSpace: sRGB,
        ])
    }()

    init() throws {
        let url = try ModelArchive.age.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // SSR-Net is tiny and uses a custom layer. ANE can't take
        // custom layers, so `.all` silently falls back to CPU for
        // this model anyway — pinning `.cpuOnly` is explicit about
        // the resolved compute path.
        config.computeUnits = .cpuOnly
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let input = inputs.first(where: { $0.value.type == .image })
                ?? inputs.first else {
            throw AnalysisError.modelLoadFailed(
                "SSR-Net model has no usable image input."
            )
        }
        self.inputName = input.key

        // SSR-Net's single output is the custom SSR_module result:
        // a scalar (shape [1]) already in the 0..100 year range.
        let outputs = model.modelDescription.outputDescriptionsByName
        guard let output = outputs.first else {
            throw AnalysisError.modelLoadFailed(
                "SSR-Net model has no output."
            )
        }
        self.outputName = output.key

        print("[SSRNet] loaded input=\(inputName) output=\(outputName)")
    }

    /// Crop the face box with 40 % margin on each side, stretch
    /// anisotropically to 64², render, and run one inference.
    func predict(face: CGRect, source: CIImage) -> AgePrediction? {
        guard face.width >= 8, face.height >= 8 else { return nil }

        let margin: CGFloat = 0.4
        let cropRect = face.insetBy(dx: -margin * face.width,
                                    dy: -margin * face.height)

        let scaleX = inputSize.width / cropRect.width
        let scaleY = inputSize.height / cropRect.height
        let transform = CGAffineTransform.identity
            .concatenating(CGAffineTransform(translationX: -cropRect.minX,
                                             y: -cropRect.minY))
            .concatenating(CGAffineTransform(scaleX: scaleX, y: scaleY))
        let resized = source
            .clampedToExtent()
            .transformed(by: transform)
            .cropped(to: CGRect(origin: .zero, size: inputSize))

        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        renderContext.render(
            resized,
            to: pb,
            bounds: CGRect(origin: .zero, size: inputSize),
            colorSpace: sRGB
        )

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let out = result.featureValue(for: outputName)?.multiArrayValue,
                  out.count >= 1 else { return nil }
            let raw = out[0].floatValue
            guard raw.isFinite else { return nil }
            let age = max(0, min(100, raw))
            print(String(format: "[SSRNet] face=(%.0f,%.0f,%.0f,%.0f) age=%.1f",
                         Double(face.origin.x), Double(face.origin.y),
                         Double(face.width), Double(face.height), Double(age)))
            return AgePrediction(age: age)
        } catch {
            print("[SSRNet] predict failed: \(error)")
            return nil
        }
    }
}

// MARK: - Custom layer

/// Port of shamangary/Keras-to-coreml-multiple-inputs-example's
/// `SSR_module.swift`. Implements the soft-stagewise-regression
/// combine step that SSR-Net's original Keras Lambda layer performs
/// — coremltools' Keras 1.x converter couldn't inline the math, so
/// the author left it as a custom layer that the host app registers
/// at runtime.
///
/// The `@objc(SSR_module)` name **must** match the `className`
/// embedded in the compiled model (`ssrnet.mlmodelc`), which is
/// `"SSR_module"`. Core ML's runtime looks this class up by its
/// Objective-C name when loading the model.
///
/// Inputs (9 MLMultiArrays, all [1, 1, 3, 1, 1] or [1, 1, 1, 1, 1]):
///   0..2 — per-stage age probabilities  (pred_a_s{1,2,3})
///   3..5 — per-stage delta scalars      (delta_s{1,2,3})
///   6..8 — per-stage local shifts       (local_s{1,2,3})
/// Output: 5-dim shape [1, 1, 1, 1, 1] carrying the scalar age.
@objc(SSR_module)
final class SSRModule: NSObject, MLCustomLayer {
    private let s1: Double
    private let s2: Double
    private let s3: Double
    private let lambdaLocal: Double
    private let lambdaD: Double

    required init(parameters: [String: Any]) throws {
        self.s1 = (parameters["s1"] as? Double) ?? 3.0
        self.s2 = (parameters["s2"] as? Double) ?? 3.0
        self.s3 = (parameters["s3"] as? Double) ?? 3.0
        self.lambdaLocal = (parameters["lambda_local"] as? Double) ?? 1.0
        self.lambdaD = (parameters["lambda_d"] as? Double) ?? 1.0
        super.init()
    }

    func setWeightData(_ weights: [Data]) throws {
        // No trainable weights — the layer's "parameters" are the
        // five fixed scalars read from the custom-layer spec.
    }

    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        // Core ML's NN layout needs 5-rank outputs; the caller
        // extracts `.multiArrayValue[0]` for the scalar age.
        [[1, 1, 1, 1, 1]]
    }

    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        let zeroIdx: [NSNumber] = [0, 0, 0, 0, 0]

        // Stage 1: sum over i in 0..<s1 of (i + λ·local_s1[i]) · pred_a_s1[i]
        var a: Double = 0
        for i in 0..<Int(s1) {
            let idx: [NSNumber] = [0, 0, i as NSNumber, 0, 0]
            a += (Double(i) + lambdaLocal * inputs[6][idx].doubleValue)
                * inputs[0][idx].doubleValue
        }
        a /= s1 * (1.0 + lambdaD * inputs[3][zeroIdx].doubleValue)

        // Stage 2: same structure, divided by both stage-1 and
        // stage-2 denominators (nested soft regression).
        var b: Double = 0
        for j in 0..<Int(s2) {
            let idx: [NSNumber] = [0, 0, j as NSNumber, 0, 0]
            b += (Double(j) + lambdaLocal * inputs[7][idx].doubleValue)
                * inputs[1][idx].doubleValue
        }
        b /= s1 * (1.0 + lambdaD * inputs[3][zeroIdx].doubleValue)
        b /= s2 * (1.0 + lambdaD * inputs[4][zeroIdx].doubleValue)

        // Stage 3: all three denominators.
        var c: Double = 0
        for k in 0..<Int(s3) {
            let idx: [NSNumber] = [0, 0, k as NSNumber, 0, 0]
            c += (Double(k) + lambdaLocal * inputs[8][idx].doubleValue)
                * inputs[2][idx].doubleValue
        }
        c /= s1 * (1.0 + lambdaD * inputs[3][zeroIdx].doubleValue)
        c /= s2 * (1.0 + lambdaD * inputs[4][zeroIdx].doubleValue)
        c /= s3 * (1.0 + lambdaD * inputs[5][zeroIdx].doubleValue)

        // V = 101 matches the upstream constant in SSRNET_model.py:
        // the stage-sum is in [0, 1], V scales to the [0, 101] age
        // range the network was trained to emit.
        let age = (a + b + c) * 101.0
        outputs[0][zeroIdx] = NSNumber(value: age)
    }
}
