import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML

/// FER+ emotion label. Reflects the ONNX model's class order: the raw
/// index in the 8-dim softmax output lines up with `allCases`.
enum EmotionLabel: String, CaseIterable, Sendable, Hashable {
    case neutral
    case happy
    case surprise
    case sad
    case anger
    case disgust
    case fear
    case contempt

    /// Emoji glyph for the head-badge display. Text-based rather than
    /// an SF Symbol because the SF Symbols catalog doesn't cover the
    /// full FER+ set (no distinct disgust / fear / contempt glyphs).
    var emoji: String {
        switch self {
        case .neutral:  return "😐"
        case .happy:    return "😊"
        case .surprise: return "😲"
        case .sad:      return "😢"
        case .anger:    return "😠"
        case .disgust:  return "🤢"
        case .fear:     return "😨"
        case .contempt: return "😒"
        }
    }

    /// Anchor coordinates in Mehrabian's Pleasure-Arousal-Dominance
    /// (PAD) space, from his own mapping of discrete emotion
    /// categories. Each axis is in [-1, 1] by convention. Callers
    /// project FER+'s full softmax onto this table to produce a
    /// continuous PAD estimate.
    var padAnchor: PADVector {
        switch self {
        case .happy:    return PADVector(pleasure:  0.81, arousal:  0.51, dominance:  0.46)
        case .surprise: return PADVector(pleasure:  0.40, arousal:  0.67, dominance: -0.13)
        case .neutral:  return PADVector(pleasure:  0.00, arousal:  0.00, dominance:  0.00)
        case .sad:      return PADVector(pleasure: -0.63, arousal: -0.27, dominance: -0.33)
        case .fear:     return PADVector(pleasure: -0.64, arousal:  0.60, dominance: -0.43)
        case .disgust:  return PADVector(pleasure: -0.60, arousal:  0.35, dominance:  0.11)
        case .anger:    return PADVector(pleasure: -0.51, arousal:  0.59, dominance:  0.25)
        case .contempt: return PADVector(pleasure: -0.55, arousal:  0.43, dominance:  0.39)
        }
    }
}

/// Three-axis affective projection: Pleasure (negative to positive
/// affect), Arousal (calm to excited), Dominance (submissive to
/// controlling). Each axis in [-1, 1]. This app computes PAD as a
/// confidence-weighted blend of FER+'s softmax over
/// `EmotionLabel.padAnchor`, not via a dedicated regressor — see
/// CLAUDE.md for the trade-offs.
struct PADVector: Hashable, Sendable {
    let pleasure: Float
    let arousal: Float
    let dominance: Float

    static let zero = PADVector(pleasure: 0, arousal: 0, dominance: 0)
}

/// One per-face emotion record produced by `EmotionClassifier`. Top
/// emotion + its confidence drives the discrete-label UI; the full
/// PAD projection is kept so views can surface continuous affect
/// without another inference pass.
struct EmotionPrediction: Hashable, Sendable {
    let label: EmotionLabel
    let confidence: Float
    let pad: PADVector
}

/// Thin wrapper around the EmoNet Core ML model. Runs per face — the
/// analyzer iterates `VNDetectFaceLandmarksRequest` rectangles, crops
/// each, resizes to 256² RGB, and calls `classify`. EmoNet regresses
/// valence + arousal directly (no anchor-table projection), so P/A
/// come from the model; Mehrabian's lookup still drives D.
struct EmotionClassifier {
    private var model: EmoNetModel? { EmoNetModel.shared }

    var isReady: Bool { ModelArchive.emotion.isInstalled() }

    /// Trigger the lazy MLModel load without a prediction — used by
    /// `FocusAnalyzer.prewarmModels` so the first analyze tap doesn't
    /// absorb compile cost.
    func warm() -> Bool { model != nil }

    /// Classify each face rect in `faces` and return predictions in the
    /// same order. Returns an empty array when the model isn't
    /// installed; individual faces that fail (low-confidence top class,
    /// crop out-of-bounds, etc.) become nil entries so the indexing
    /// stays aligned.
    func classify(faces: [CGRect],
                  in image: CIImage,
                  ciContext: CIContext) -> [EmotionPrediction?] {
        guard let model else { return [] }
        guard !faces.isEmpty else { return [] }
        return faces.map { face -> EmotionPrediction? in
            model.predict(face: face, source: image, ciContext: ciContext)
        }
    }
}

// MARK: - Model wrapper

private final class EmoNetModel {
    static var shared: EmoNetModel? = {
        try? EmoNetModel()
    }()

    /// EmoNet's 8-class head outputs logits in this order. Our
    /// `EmotionLabel` enum uses a different ordering, so mapping the
    /// model's index to our label goes through this array.
    private static let classOrder: [EmotionLabel] = [
        .neutral, .happy, .sad, .surprise, .fear, .disgust, .anger, .contempt
    ]

    private let model: MLModel
    private let inputName: String
    private let inputSize: CGSize
    /// Resolved output names for each head, probed at load time so
    /// the model works regardless of whether coremltools preserved
    /// the `ct.TensorType(name:)` labels we asked for.
    private let expressionOutput: String
    private let valenceOutput: String
    private let arousalOutput: String

    /// Minimum top-class softmax score required to surface a
    /// prediction. Below this the face is returned as nil so the UI
    /// hides its glyph instead of asserting a guess. Lower for
    /// EmoNet than FER+ because EmoNet's expression head is
    /// softer-distributed on natural faces — a 0.35 floor dropped
    /// most real images.
    private let confidenceFloor: Float = 0.20

    init() throws {
        let url = try ModelArchive.emotion.installedURL()
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
            throw AnalysisError.modelLoadFailed("Emotion model has no usable image input.")
        }
        self.inputName = input.key

        // Resolve output names. We'd like the converter to have
        // preserved `expression` / `valence` / `arousal` literally,
        // but coremltools occasionally renames outputs to `var_N`
        // when converting traced modules. Fall back to shape-based
        // identification: an 8-dim multiarray is expression, a
        // 1-dim multiarray is either valence or arousal.
        let outputs = model.modelDescription.outputDescriptionsByName
        let names = Array(outputs.keys)
        let orderedShapes: [(name: String, count: Int)] = names.compactMap { name in
            let desc = outputs[name]
            guard let multi = desc?.multiArrayConstraint else { return nil }
            let totalCount = multi.shape.map(\.intValue).reduce(1, *)
            return (name, totalCount)
        }
        func firstMatching(_ hint: String, countPredicate: (Int) -> Bool) -> String? {
            if outputs[hint] != nil,
               let match = orderedShapes.first(where: { $0.name == hint && countPredicate($0.count) }) {
                return match.name
            }
            return orderedShapes.first(where: { countPredicate($0.count) })?.name
        }
        let expressionCandidates = orderedShapes
            .filter { $0.count == EmoNetModel.classOrder.count }
            .map(\.name)
        let scalarCandidates = orderedShapes
            .filter { $0.count == 1 }
            .map(\.name)

        guard let exprName = outputs["expression"] != nil
                ? "expression"
                : expressionCandidates.first
        else {
            throw AnalysisError.modelLoadFailed(
                "EmoNet model has no 8-dim expression output (available: \(names))."
            )
        }
        // Prefer exact-name matches for V/A; fall back to the two
        // remaining scalar outputs in declared order. EmoNet's
        // traced output order is expression → valence → arousal,
        // so the same order works when names were scrubbed.
        let valName: String = outputs["valence"] != nil ? "valence"
            : (scalarCandidates.first ?? "")
        let arsName: String = outputs["arousal"] != nil ? "arousal"
            : (scalarCandidates.count >= 2 ? scalarCandidates[1] : "")
        guard !valName.isEmpty, !arsName.isEmpty else {
            throw AnalysisError.modelLoadFailed(
                "EmoNet model missing valence/arousal scalar outputs (available: \(names))."
            )
        }
        self.expressionOutput = exprName
        self.valenceOutput = valName
        self.arousalOutput = arsName

        print("[EmoNet] loaded input=\(inputName) expression=\(exprName) valence=\(valName) arousal=\(arsName)")

        var resolvedSize = CGSize(width: 256, height: 256)
        if let constraint = input.value.imageConstraint {
            resolvedSize = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolvedSize
    }

    /// Crop the face rect from the source, resize to the model's
    /// input size as RGB, run inference, and return the combined
    /// EmotionPrediction. `expression` drives the discrete label +
    /// confidence; `valence` and `arousal` are used directly for the
    /// P and A axes of PAD; D is Mehrabian-projected from the
    /// softmax because EmoNet doesn't regress dominance.
    func predict(face: CGRect,
                 source: CIImage,
                 ciContext: CIContext) -> EmotionPrediction? {
        let extent = source.extent
        // Pad the face rect outward so the crop includes forehead /
        // chin — EmoNet was trained on AffectNet / AFEW-VA crops that
        // typically enclose the whole face, not just the landmarks
        // bounding box.
        let pad = face.width * 0.15
        let padded = face.insetBy(dx: -pad, dy: -pad)
        let clamped = padded.intersection(extent)
        guard !clamped.isNull, clamped.width >= 8, clamped.height >= 8 else {
            return nil
        }

        let resized = source.cropped(to: clamped).stretched(to: inputSize)

        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        ciContext.render(resized, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let expr = result.featureValue(for: expressionOutput)?.multiArrayValue,
                  expr.count >= Self.classOrder.count
            else {
                print("[EmoNet] predict: missing/short expression output (\(expressionOutput))")
                return nil
            }

            // Log the raw first two values so we can tell NaN-from-model
            // apart from NaN-from-Swift-softmax-bug.
            let sample0 = expr[0].floatValue
            let sample1 = expr.count > 1 ? expr[1].floatValue : 0
            if !sample0.isFinite || !sample1.isFinite {
                print(String(format: "[EmoNet] predict: raw logits non-finite (e0=%.4f e1=%.4f) — Core ML overflow / bad weights", sample0, sample1))
            }

            // Stable softmax over the expression logits.
            var logits = [Float](repeating: 0, count: Self.classOrder.count)
            for i in 0..<logits.count {
                logits[i] = expr[i].floatValue
            }
            let maxL = logits.max() ?? 0
            var expSum: Float = 0
            var probs = [Float](repeating: 0, count: logits.count)
            for i in 0..<logits.count {
                probs[i] = exp(logits[i] - maxL)
                expSum += probs[i]
            }
            if expSum > 0 {
                for i in 0..<probs.count { probs[i] /= expSum }
            }

            var topIdx = 0
            var topScore: Float = -1
            for i in 0..<probs.count where probs[i] > topScore {
                topScore = probs[i]
                topIdx = i
            }
            guard topScore >= confidenceFloor, topIdx < Self.classOrder.count else {
                print(String(format: "[EmoNet] predict: top softmax %.3f < floor %.2f — skipping", topScore, confidenceFloor))
                return nil
            }
            let topLabel = Self.classOrder[topIdx]

            // Valence + Arousal come straight from EmoNet's regression
            // heads — this is the whole point of using it instead of
            // FER+. Scalars sit in [-1, 1] by design; clamp defensively
            // in case the export introduced numerical slop.
            let valence = (result.featureValue(for: valenceOutput)?
                .multiArrayValue?[0].floatValue) ?? 0
            let arousal = (result.featureValue(for: arousalOutput)?
                .multiArrayValue?[0].floatValue) ?? 0

            // EmoNet doesn't regress dominance, so project D from the
            // 8-class softmax via Mehrabian's anchors. Same trick we
            // used for FER+, applied only to the dominance axis now.
            var d: Float = 0
            for i in 0..<probs.count where i < Self.classOrder.count {
                d += probs[i] * Self.classOrder[i].padAnchor.dominance
            }

            return EmotionPrediction(
                label: topLabel,
                confidence: topScore,
                pad: PADVector(
                    pleasure: max(-1, min(1, valence)),
                    arousal: max(-1, min(1, arousal)),
                    dominance: max(-1, min(1, d))
                )
            )
        } catch {
            return nil
        }
    }
}
