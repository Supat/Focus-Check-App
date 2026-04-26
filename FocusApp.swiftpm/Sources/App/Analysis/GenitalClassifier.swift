import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML

/// Output classes from the CreateML image classifier. The strings here
/// must match the folder names used during CreateML training (the
/// classifier emits its top class as a literal string from that set).
/// `OTHER` is the "drop this detection" verdict — NudeNet false
/// positive or non-genital private-region anatomy.
///
/// v3 schema: every COVERED / EXPOSED label encodes the gating
/// prefix as a substring so the mosaic dispatch's existing
/// `contains("COVERED")` / `contains("EXPOSED")` filters work
/// without per-label enumeration. The five EXPOSED variants form
/// an anatomical state ladder; COVERED has a hybrid
/// COVERED_STIMULATION variant for clothed-but-stimulated cases.
enum GenitalSubClass: String, CaseIterable, Sendable, Hashable {
    case covered             = "MALE_GENITALIA_COVERED"
    case coveredStimulation  = "MALE_GENITALIA_COVERED_STIMULATION"
    case exposedLatent       = "MALE_GENITALIA_EXPOSED_LATENT"
    case exposedTumescent    = "MALE_GENITALIA_EXPOSED_TUMESCENT"
    case exposedArousal      = "MALE_GENITALIA_EXPOSED_AROUSAL"
    case exposedOrgasm       = "MALE_GENITALIA_EXPOSED_ORGASM"
    case exposedDetumescent  = "MALE_GENITALIA_EXPOSED_DETUMESCENT"
    case other               = "OTHER"

    init?(rawLabel: String) {
        let trimmed = rawLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        self.init(rawValue: trimmed)
    }
}

/// One classifier verdict on a NudeNet genital-region crop.
struct GenitalClassification: Hashable, Sendable {
    let subClass: GenitalSubClass
    let confidence: Float
}

/// Thin wrapper around the downloaded CreateML image classifier.
/// Lazy-loaded; runs after each NudeNet detection in
/// {MGE, FGC, FGE} and overrides the detection label with one of
/// the five sub-classes (or drops the detection when the verdict
/// is `.other`).
struct GenitalClassifier {
    private var model: GenitalClassifierModel? { GenitalClassifierModel.shared }

    var isReady: Bool { ModelArchive.genitalClassifier.isInstalled() }

    /// Trigger the lazy MLModel load without running a prediction.
    /// `FocusAnalyzer.prewarmModels` calls this so the first
    /// analyze tap doesn't absorb compile cost.
    func warm() -> Bool { model != nil }

    /// Run the classifier on the source-extent rect. Returns nil
    /// when the model isn't installed, the crop is too small, or
    /// the top-class confidence falls below the floor (in which
    /// case the caller should keep the original NudeNet label
    /// rather than overriding to a low-confidence guess).
    func classify(rect: CGRect,
                  in source: CIImage,
                  ciContext: CIContext) -> GenitalClassification? {
        guard let model else { return nil }
        return model.predict(rect: rect, source: source, ciContext: ciContext)
    }
}

// MARK: - Model wrapper

private final class GenitalClassifierModel {
    static var shared: GenitalClassifierModel? = {
        try? GenitalClassifierModel()
    }()

    private let model: MLModel
    private let inputName: String
    private let inputSize: CGSize
    /// CreateML image classifiers emit two outputs:
    ///   * `classLabel` — String, the predicted top class.
    ///   * `classLabelProbs` — [String: Double] across all classes.
    /// We probe at load time so a future export tweak (rename to
    /// e.g. `target` or numeric-prefix outputs) doesn't silently
    /// break inference.
    private let labelOutput: String
    private let probsOutput: String?

    /// Reject overrides below this confidence — NudeNet's original
    /// label stays. Tuned on a small validation set; raise if the
    /// classifier shows up too aggressive on borderline crops.
    private let confidenceFloor: Float = 0.40

    /// Outward padding around the NudeNet box before crop, matching
    /// `Tools/extract_genital_crops.py::CROP_PADDING_FRAC`. The
    /// classifier was trained on padded + square-padded crops; this
    /// keeps the inference distribution aligned with training.
    private let cropPaddingFrac: CGFloat = 0.15

    init() throws {
        let url = try ModelArchive.genitalClassifier.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let input = inputs.first(where: { $0.value.type == .image }) ?? inputs.first
        else {
            throw AnalysisError.modelLoadFailed(
                "Genital classifier has no usable image input."
            )
        }
        self.inputName = input.key

        let outputs = model.modelDescription.outputDescriptionsByName
        // CreateML names the predicted-class output `classLabel`. If
        // that exact name is present use it, otherwise fall back to
        // any String output (CreateML guarantees there's exactly one).
        if outputs["classLabel"] != nil {
            self.labelOutput = "classLabel"
        } else if let firstString = outputs.first(where: { $0.value.type == .string })?.key {
            self.labelOutput = firstString
        } else {
            throw AnalysisError.modelLoadFailed(
                "Genital classifier has no String output."
            )
        }
        // The probabilities dict is optional — used only for the
        // confidence floor. If the export stripped it we still emit
        // overrides at default-confidence semantics.
        if outputs["classLabelProbs"] != nil {
            self.probsOutput = "classLabelProbs"
        } else if let firstDict = outputs.first(where: { $0.value.type == .dictionary })?.key {
            self.probsOutput = firstDict
        } else {
            self.probsOutput = nil
        }

        var resolved = CGSize(width: 299, height: 299)
        if let constraint = input.value.imageConstraint {
            resolved = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolved

        print("[GenitalClassifier] loaded "
              + "input=\(inputName) "
              + "label=\(labelOutput) "
              + "probs=\(probsOutput ?? "(absent)") "
              + "inputSize=\(Int(inputSize.width))×\(Int(inputSize.height))")
    }

    func predict(rect: CGRect,
                 source: CIImage,
                 ciContext: CIContext) -> GenitalClassification? {
        // Reject obviously degenerate detections — the training
        // pipeline skipped anything <32 px on the short side, so the
        // model has no learned response below that resolution.
        guard rect.width >= 16, rect.height >= 16 else { return nil }

        // Build the same crop the training pipeline produced:
        //   1. Pad outward by 15 % of the longer side.
        //   2. Pad to a square with black borders, centred.
        //   3. Resize to model input size.
        let pad = max(rect.width, rect.height) * cropPaddingFrac
        let padded = rect.insetBy(dx: -pad, dy: -pad)
            .intersection(source.extent)
        guard padded.width >= 16, padded.height >= 16 else { return nil }

        let cropped = source.cropped(to: padded)
            .transformed(by: CGAffineTransform(
                translationX: -padded.minX, y: -padded.minY
            ))
        let side = max(padded.width, padded.height)
        let dx = (side - padded.width) / 2
        let dy = (side - padded.height) / 2
        let centered = cropped.transformed(by: CGAffineTransform(
            translationX: dx, y: dy
        ))
        let blackCanvas = CIImage(color: CIColor.black)
            .cropped(to: CGRect(origin: .zero, size: CGSize(width: side, height: side)))
        let composed = centered.composited(over: blackCanvas)
            .cropped(to: CGRect(origin: .zero, size: CGSize(width: side, height: side)))

        let scale = inputSize.width / side
        let resized = composed.transformed(
            by: CGAffineTransform(scaleX: scale, y: scale)
        ).cropped(to: CGRect(origin: .zero, size: inputSize))

        // Render to a CGImage in sRGB so the model sees consumer-style
        // gamma-encoded pixels rather than the working linear-Display-P3
        // space the rest of the analyzer uses.
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB),
              let cgImage = ciContext.createCGImage(
                resized, from: resized.extent,
                format: .RGBA8, colorSpace: sRGB
              )
        else { return nil }

        let pixelBufferOptions: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        var maybeBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(inputSize.width), Int(inputSize.height),
            kCVPixelFormatType_32BGRA,
            pixelBufferOptions as CFDictionary,
            &maybeBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybeBuffer else { return nil }
        ciContext.render(resized, to: pixelBuffer)

        let featureValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let provider: MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                inputName: featureValue
            ])
        } catch {
            return nil
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: provider)
        } catch {
            return nil
        }

        guard let labelString = result.featureValue(for: labelOutput)?.stringValue,
              let subClass = GenitalSubClass(rawLabel: labelString)
        else { return nil }

        var confidence: Float = 1.0
        if let probsName = probsOutput,
           let probs = result.featureValue(for: probsName)?.dictionaryValue {
            // CreateML stringifies the keys; look up by the literal
            // class name. The dictionary value is already a NSNumber
            // bridge in MLFeatureValue; pull out the float directly.
            if let raw = probs[labelString as NSString] {
                confidence = raw.floatValue
            }
        }
        guard confidence >= confidenceFloor else { return nil }
        // Suppress the cgImage capture warning — kept for diagnostic
        // bridging in case we want to dump misclassified crops later.
        _ = cgImage
        return GenitalClassification(subClass: subClass, confidence: confidence)
    }
}
