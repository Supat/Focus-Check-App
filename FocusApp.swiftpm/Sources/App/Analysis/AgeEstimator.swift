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

    /// Run the estimator for every face. `rolls` is accepted for API
    /// parity with the sibling per-face tiers but is **not** applied
    /// — yu4u's training pipeline feeds axis-aligned crops straight
    /// from dlib and relies on `albumentations.ShiftScaleRotate` for
    /// rotation invariance. De-rotating in Swift shifts the crop
    /// composition in a way the model wasn't specifically trained
    /// for, so we leave the rectangle axis-aligned.
    func estimate(faces: [CGRect],
                  rolls _: [CGFloat],
                  in image: CIImage,
                  ciContext: CIContext) -> [AgeGenderPrediction?] {
        guard let model else { return [] }
        guard !faces.isEmpty else { return [] }
        return faces.map { face -> AgeGenderPrediction? in
            model.predict(face: face, source: image, ciContext: ciContext)
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
        // `.cpuAndGPU` rather than `.all` — the ANE runs EfficientNetB3
        // in F16, which is fine for the argmax-flavored emotion / pain
        // tiers but bites this one. The age head is a 101-bin softmax
        // whose expectation (`Σ i · p_i`) weights the long-tail bins
        // (ages 80+) heavily, and F16 underflow on their tiny
        // probabilities shifts the mean by 2–5 years on real faces.
        // GPU F32 keeps the distribution faithful; one-shot per-photo
        // inference makes the latency cost irrelevant.
        config.computeUnits = .cpuAndGPU
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

    /// Crop + anisotropically resize the face into the model's 224²
    /// input, run inference twice (original + horizontal flip) so we
    /// can average the two probability distributions, and compose the
    /// age/gender result. Horizontal-flip TTA costs one extra
    /// inference per face — EfficientNetB3 is ~15 ms on the ANE so
    /// per-face latency lands around 30–40 ms, well under the
    /// budget a user perceives for a one-shot analyze.
    ///
    /// Preprocessing exactly mirrors yu4u's demo.py:
    ///
    ///     xw1 = x1 - margin·w;  yw1 = y1 - margin·h
    ///     xw2 = x2 + margin·w;  yw2 = y2 + margin·h
    ///     face = cv2.resize(img[yw1:yw2+1, xw1:xw2+1], (224, 224))
    ///
    /// i.e. crop a rectangle expanded by 40 % of the face box's
    /// width and height on each side, then stretch **non-uniformly**
    /// to 224×224. A square crop (what v1/v2 did) over-pads the
    /// shorter axis and the model reads that as "wrong scale".
    func predict(face: CGRect,
                 source: CIImage,
                 ciContext: CIContext) -> AgeGenderPrediction? {
        guard face.width >= 8, face.height >= 8 else { return nil }

        // yu4u's `--margin 0.4` — 40 % of the face box's width/height
        // extended on each side. Total crop = face · (1 + 2 · margin).
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

        // Horizontally flipped twin — reflect around the x center of
        // the output square. Face classification + regression tasks
        // are near-symmetric under this transform, so averaging the
        // two prediction distributions smooths over whatever side-of-
        // face bias the single crop picked up. yu4u's training used
        // `albumentations.HorizontalFlip(p=0.5)` so the model is
        // explicitly flip-invariant.
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

        let prediction = AgeGenderPrediction(
            age: max(0, min(100, mean)),
            ageStdev: stdev.isFinite ? stdev : 0,
            gender: gender,
            genderConfidence: confidence
        )
        print(String(
            format: "[AgeGender] face=(%.0f,%.0f,%.0f,%.0f) age=%.1f±%.1f gender=%@ (P[F,M]=%.2f,%.2f)",
            Double(face.origin.x), Double(face.origin.y),
            Double(face.width), Double(face.height),
            Double(prediction.age), Double(prediction.ageStdev),
            prediction.gender == .male ? "M" : "F",
            Double(pFemale), Double(pMale)
        ))
        return prediction
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
