import Foundation
import CoreImage
import CoreML

/// Per-face age + gender prediction from yu4u/age-gender-estimation
/// (EfficientNetB3 trained on IMDB-WIKI). The model returns a softmax
/// over 101 age bins (0…100) and a 2-class gender softmax; the Swift
/// side turns those into a continuous age estimate via expectation
/// plus an uncertainty band via the same distribution's standard
/// deviation.
struct AgeGenderPrediction: Hashable, Sendable {
    /// Expected age (`Σ i · p_i`) clamped to [0, 100]. Presented to the
    /// UI as the center of the "28 ± 4" readout.
    let age: Float
    /// Standard deviation of the age distribution (`sqrt(Σ (i-μ)² · p_i)`).
    /// Functions as an uncertainty band — a confident prediction lands
    /// around 3–5, a hedge spreads into the teens.
    let ageStdev: Float
    /// Argmax gender. `.unknown` should not appear here — this model
    /// always commits to one of the two — but callers should treat low
    /// `genderConfidence` the same way they treat `.unknown` for UX.
    let gender: SubjectGender
    /// Top gender probability in [0, 1]. Below-threshold calls
    /// (< 0.6) should be downgraded to `.unknown` at the display
    /// layer; keep the raw value here so callers can choose.
    let genderConfidence: Float
}

/// Thin wrapper around the yu4u age-gender MLModel. Same shape as
/// `PainDetector` / `EmotionClassifier` — the analyzer iterates face
/// rectangles and hands each one in. Empty array out when the model
/// isn't installed; nil entries for faces the crop couldn't cover.
struct AgeEstimator {
    private var model: AgeGenderModel? { AgeGenderModel.shared }

    var isReady: Bool { ModelArchive.ageGender.isInstalled() }

    func warm() -> Bool { model != nil }

    /// Run the estimator for every face. `rolls` is parallel to
    /// `faces` and lets us de-rotate the crop before resizing, matching
    /// the axis-aligned inputs the model saw during training.
    func estimate(faces: [CGRect],
                  rolls: [CGFloat],
                  in image: CIImage,
                  ciContext: CIContext) -> [AgeGenderPrediction?] {
        guard let model else { return [] }
        guard !faces.isEmpty else { return [] }
        return faces.enumerated().map { idx, face -> AgeGenderPrediction? in
            let roll = idx < rolls.count ? rolls[idx] : 0
            return model.predict(face: face, roll: roll,
                                 source: image, ciContext: ciContext)
        }
    }
}

// MARK: - Model wrapper

private final class AgeGenderModel {
    static var shared: AgeGenderModel? = {
        try? AgeGenderModel()
    }()

    private let model: MLModel
    private let inputName: String
    private let ageOutputName: String
    private let genderOutputName: String
    private let inputSize = CGSize(width: 224, height: 224)

    private static let numAgeBins = 101  // 0…100 inclusive.

    init() throws {
        let url = try ModelArchive.ageGender.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // EfficientNetB3 is ANE-friendly — the MobileNet-style depthwise
        // conv blocks map cleanly. Leave `.all` so Core ML picks the
        // best compute unit at runtime.
        config.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let input = inputs.first(where: { $0.value.type == .image })
                ?? inputs.first else {
            throw AnalysisError.modelLoadFailed(
                "AgeGender model has no usable image input."
            )
        }
        self.inputName = input.key

        // Two named outputs: `pred_age` (101-dim) and `pred_gender`
        // (2-dim). Prefer the names the converter emitted; fall back
        // to shape-based lookup so a renamed head still binds.
        let outputs = model.modelDescription.outputDescriptionsByName
        func outputName(preferred: String, elementCount: Int) throws -> String {
            if outputs[preferred] != nil { return preferred }
            let match = outputs.first { _, desc in
                desc.multiArrayConstraint?.shape
                    .map(\.intValue)
                    .reduce(1, *) == elementCount
            }
            guard let match else {
                throw AnalysisError.modelLoadFailed(
                    "AgeGender model has no \(elementCount)-dim output "
                    + "(available: \(Array(outputs.keys)))."
                )
            }
            return match.key
        }
        self.ageOutputName = try outputName(preferred: "pred_age",
                                            elementCount: Self.numAgeBins)
        self.genderOutputName = try outputName(preferred: "pred_gender",
                                               elementCount: 2)

        print("[AgeGender] loaded input=\(inputName) "
              + "age=\(ageOutputName) gender=\(genderOutputName)")
    }

    /// Rotate + square-crop + resize the face into the model's 224²
    /// input, run inference twice (original + horizontal flip) so we
    /// can average the two probability distributions, and compose the
    /// age/gender result. Horizontal-flip TTA costs one extra
    /// inference per face — EfficientNetB3 is ~15 ms on the ANE so
    /// per-face latency lands around 30–40 ms, well under the
    /// budget a user perceives for a one-shot analyze.
    func predict(face: CGRect,
                 roll: CGFloat,
                 source: CIImage,
                 ciContext: CIContext) -> AgeGenderPrediction? {
        guard face.width >= 8, face.height >= 8 else { return nil }

        // yu4u's training recipe crops faces with `--margin 0.4` around
        // the detector box — 40 % extra on *each* side, i.e. a square
        // crop whose side is `1 + 2·margin = 1.8` times the face box.
        // v1 used 1.4 which is the wrong interpretation (40 % total,
        // 20 % each side); 1.8 matches what the pretrained weights
        // were trained on — hair + jawline + neck context included.
        let marginFactor: CGFloat = 1.8
        let halfSide = max(face.width, face.height) * marginFactor / 2
        let faceCenter = CGPoint(x: face.midX, y: face.midY)
        let outputHalf = inputSize.width / 2
        let scale = outputHalf / halfSide

        let transform = CGAffineTransform.identity
            .concatenating(CGAffineTransform(translationX: -faceCenter.x,
                                             y: -faceCenter.y))
            .concatenating(CGAffineTransform(rotationAngle: -roll))
            .concatenating(CGAffineTransform(scaleX: scale, y: scale))
            .concatenating(CGAffineTransform(translationX: outputHalf,
                                             y: outputHalf))
        let resized = source
            .clampedToExtent()
            .transformed(by: transform)
            .cropped(to: CGRect(origin: .zero, size: inputSize))

        // Horizontally flipped twin — reflect around the x center of
        // the output square. Face classification + regression tasks
        // are near-symmetric under this transform, so averaging the
        // two prediction distributions smooths over whatever side-of-
        // face bias the single crop picked up.
        let flipped = resized
            .transformed(by: CGAffineTransform(scaleX: -1, y: 1)
                .translatedBy(x: -inputSize.width, y: 0))
            .cropped(to: CGRect(origin: .zero, size: inputSize))

        guard let agePrimary = rawProbabilities(for: resized, ciContext: ciContext),
              let ageFlipped = rawProbabilities(for: flipped, ciContext: ciContext)
        else { return nil }

        // Average the two heads element-wise. The softmax property is
        // preserved: averaging two probability vectors stays on the
        // simplex (each value in [0, 1], sum = 1).
        let ageProbs = zip(agePrimary.age, ageFlipped.age).map { 0.5 * ($0 + $1) }
        let genderProbs = zip(agePrimary.gender, ageFlipped.gender).map { 0.5 * ($0 + $1) }

        // Age distribution reductions — expectation for the point
        // estimate, standard deviation for the uncertainty band.
        var mean: Float = 0
        for i in 0..<Self.numAgeBins {
            let p = ageProbs[i]
            if p.isFinite { mean += Float(i) * p }
        }
        var variance: Float = 0
        for i in 0..<Self.numAgeBins {
            let p = ageProbs[i]
            if p.isFinite {
                let d = Float(i) - mean
                variance += d * d * p
            }
        }
        let stdev = variance.squareRoot()

        // yu4u's training label ordering is [female=0, male=1].
        let pFemale = max(0, min(1, genderProbs[0]))
        let pMale   = max(0, min(1, genderProbs[1]))
        let (gender, confidence): (SubjectGender, Float) =
            pMale >= pFemale ? (.male, pMale) : (.female, pFemale)

        return AgeGenderPrediction(
            age: max(0, min(100, mean)),
            ageStdev: stdev.isFinite ? stdev : 0,
            gender: gender,
            genderConfidence: confidence
        )
    }

    /// Render `crop` into a 224² BGRA pixel buffer, run one inference,
    /// and return the two softmax heads as plain `[Float]`. Returning
    /// raw arrays (not MLMultiArray) lets the caller average two
    /// runs element-wise without juggling NSNumber conversions.
    private func rawProbabilities(for crop: CIImage,
                                  ciContext: CIContext)
                                  -> (age: [Float], gender: [Float])? {
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
        ciContext.render(
            crop,
            to: pb,
            bounds: CGRect(origin: .zero, size: inputSize),
            colorSpace: sRGB
        )

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let ageMA = result.featureValue(for: ageOutputName)?.multiArrayValue,
                  ageMA.count >= Self.numAgeBins,
                  let genderMA = result.featureValue(for: genderOutputName)?.multiArrayValue,
                  genderMA.count >= 2
            else { return nil }

            var age = [Float](repeating: 0, count: Self.numAgeBins)
            for i in 0..<Self.numAgeBins { age[i] = ageMA[i].floatValue }
            let gender = [genderMA[0].floatValue, genderMA[1].floatValue]
            return (age: age, gender: gender)
        } catch {
            print("[AgeGender] predict failed: \(error)")
            return nil
        }
    }
}
