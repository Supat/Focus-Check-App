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
    /// Dedicated CIContext for crop rendering. The shared analyzer
    /// CIContext has `workingColorSpace = extendedLinearDisplayP3`,
    /// which produces pixel buffer bytes that don't exactly match
    /// sRGB-encoded values — on this model that manifests as a
    /// ~30 % darkening of the input, pushing the prediction into
    /// the model's "dark input" hallucination. A dedicated sRGB-
    /// working-space context renders byte-accurate sRGB pixels.
    private let renderContext: CIContext = {
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        return CIContext(options: [
            .workingColorSpace: sRGB,
            .outputColorSpace: sRGB,
        ])
    }()

    private static let numAgeBins = 101  // 0…100 inclusive.

    init() throws {
        let url = try ModelArchive.ageGender.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // `.cpuOnly` is the only compute unit that reliably keeps the
        // full F32 precision the 101-bin age softmax needs. `.all`
        // delegates to the ANE (F16), `.cpuAndGPU` delegates to
        // MPSGraph (also effectively F16 for matmul / conv on
        // Apple-silicon), and both underflow the tail logits enough
        // to collapse the softmax to (1.0, 0.0) on plausible face
        // inputs. Pure-CPU EfficientNetB3 at 224² per face lands in
        // the low hundreds of milliseconds — fine for the one-shot
        // per-photo analyze path, not fine for video.
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

        // Self-test on synthetic solid colors. Compare against
        // expected macOS Keras predictions so we can tell whether
        // iOS Core ML is handling the BGR pixel buffer correctly.
        //
        // Expected (macOS host Core ML with the same .mlpackage):
        //   BGR(0,0,0)     age≈50.5  P[F]≈0.91
        //   BGR(128,128,128) age≈42.0 P[F]≈0.28
        //   BGR(220,100,30) age≈44.2 P[F]≈0.51   ← our asymmetric probe
        for (label, b, g, r) in [
            ("black", UInt8(0),   UInt8(0),   UInt8(0)),
            ("gray",  UInt8(128), UInt8(128), UInt8(128)),
            ("probe", UInt8(220), UInt8(100), UInt8(30)),
        ] {
            guard let pb = makeSolidBGRA(width: 224, height: 224,
                                         b: b, g: g, r: r) else { continue }
            if let (age, gender) = rawInference(pixelBuffer: pb) {
                var mean: Float = 0
                for i in 0..<age.count { mean += Float(i) * age[i] }
                print(String(
                    format: "[AgeGender] self-test %@ BGR(%d,%d,%d): age=%.1f P[F,M]=%.3f,%.3f",
                    label, Int(b), Int(g), Int(r),
                    Double(mean),
                    Double(gender[0]), Double(gender[1])
                ))
            }
        }
    }

    /// Allocate a 32BGRA CVPixelBuffer of the given size filled with
    /// a constant BGR colour. Alpha is 255. Only used by the init-
    /// time self-test.
    private func makeSolidBGRA(width w: Int, height h: Int,
                               b: UInt8, g: UInt8, r: UInt8) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary, &pb)
        guard let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            for y in 0..<h {
                let row = base.advanced(by: y * bpr)
                    .assumingMemoryBound(to: UInt8.self)
                for x in 0..<w {
                    row[x * 4 + 0] = b
                    row[x * 4 + 1] = g
                    row[x * 4 + 2] = r
                    row[x * 4 + 3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        return pb
    }

    /// Run one inference directly on a ready-made pixel buffer,
    /// bypassing the crop / render path. Used only by the self-test.
    private func rawInference(pixelBuffer pb: CVPixelBuffer) -> (age: [Float], gender: [Float])? {
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
            return (age, [genderMA[0].floatValue, genderMA[1].floatValue])
        } catch {
            print("[AgeGender] self-test inference failed: \(error)")
            return nil
        }
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

        // Margin controls how much context around the detected face
        // the model sees. yu4u's training default was 0.4 (40 % on
        // each side) but empirically that was scooping up too much
        // of the surrounding frame on photos with dark backgrounds
        // or dark clothing — the model saturates to its "dark input"
        // hallucination (age ≈ 50, P[F] ≈ 0.91). A tighter 0.1
        // margin still gives a little hair + jawline context while
        // keeping the frame dominated by actual face pixels. If this
        // helps, revisit 0.2 as a middle ground.
        let margin: CGFloat = 0.1
        let cropRect = face.insetBy(dx: -margin * face.width,
                                    dy: -margin * face.height)

        // Diagnostic: source extent vs crop rect. If the crop extends
        // outside the extent, clampedToExtent should replicate edge
        // pixels but we're seeing dark borders in the rendered crop
        // that suggest otherwise.
        print(String(format: "[AgeGender] source extent=(%.0f,%.0f,%.0f,%.0f) "
                     + "face=(%.0f,%.0f,%.0f,%.0f) crop=(%.0f,%.0f,%.0f,%.0f)",
                     Double(source.extent.origin.x), Double(source.extent.origin.y),
                     Double(source.extent.width), Double(source.extent.height),
                     Double(face.origin.x), Double(face.origin.y),
                     Double(face.width), Double(face.height),
                     Double(cropRect.origin.x), Double(cropRect.origin.y),
                     Double(cropRect.width), Double(cropRect.height)))

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

        // Use the dedicated sRGB-working-space render context rather
        // than the caller's shared extendedLinearDisplayP3 one —
        // see `renderContext` property comment for why.
        guard let agePrimary = rawProbabilities(for: resized, ciContext: renderContext),
              let ageFlipped = rawProbabilities(for: flipped, ciContext: renderContext)
        else { return nil }

        // Diagnostic: top-3 age bins of the primary (un-flipped) pass,
        // plus raw gender probs. Distinguishes "saturated at one bin"
        // (precision bug) from "spread distribution but wrong mean"
        // (preprocessing bug).
        let topAge = agePrimary.age.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(3)
            .map { "\($0.offset):\(String(format: "%.2f", $0.element))" }
            .joined(separator: " ")
        print("[AgeGender] primary topAges=\(topAge) "
              + "gender=[\(String(format: "%.3f", agePrimary.gender[0])),"
              + "\(String(format: "%.3f", agePrimary.gender[1]))]")

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

        // Diagnostic: sample the rendered buffer's mean BGR channel
        // values over a 16×16 grid. Near-zero means the CIContext
        // render produced black (input-space mismatch, empty crop
        // extent, or a silent render failure). Near-normal (50–200)
        // means the face reached the buffer and the issue is model-
        // internal. Remove once the accuracy gap is root-caused.
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
            func luma(x: Int, y: Int) -> Int {
                let p = base.advanced(by: y * bytesPerRow + x * 4)
                    .assumingMemoryBound(to: UInt8.self)
                return (Int(p[0]) + Int(p[1]) + Int(p[2])) / 3
            }
            // 5x5 grid of luma values — face regions (eyes, cheeks,
            // mouth) should produce clearly different values at
            // different positions; uniform darkness at all 25 points
            // means the face isn't actually in the buffer.
            var rows: [String] = []
            for gy in 0..<5 {
                let y = min(h - 1, Int(Double(gy) * Double(h - 1) / 4.0))
                var row: [String] = []
                for gx in 0..<5 {
                    let x = min(w - 1, Int(Double(gx) * Double(w - 1) / 4.0))
                    row.append(String(format: "%3d", luma(x: x, y: y)))
                }
                rows.append(row.joined(separator: " "))
            }
            print("[AgeGender] 5x5 luma grid (top-down):")
            for r in rows { print("[AgeGender]   \(r)") }
        }
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)

        // Dump the rendered crop to the app's documents directory so
        // the user can retrieve via Files.app and we can actually see
        // what's reaching the model. One file per inference, timestamped.
        if let docs = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask,
            appropriateFor: nil, create: false
        ) {
            let ts = Int(Date().timeIntervalSince1970 * 1000) % 1_000_000
            let png = docs.appendingPathComponent("agegender-crop-\(ts).png")
            let ci = CIImage(cvPixelBuffer: pb)
            if let cg = ciContext.createCGImage(ci, from: ci.extent),
               let dest = CGImageDestinationCreateWithURL(
                   png as CFURL, "public.png" as CFString, 1, nil
               ) {
                CGImageDestinationAddImage(dest, cg, nil)
                if CGImageDestinationFinalize(dest) {
                    // Print the absolute path so the user can navigate
                    // via Finder → ⌘⇧G on macOS Designed-for-iPad. On
                    // iOS the file also shows in Files.app under
                    // "On My iPad" → FocusApp.
                    print("[AgeGender] crop dumped to \(png.path)")
                }
            }
        }

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
