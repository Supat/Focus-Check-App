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

/// Thin wrapper around the FER+ Core ML model. Runs per face — the
/// analyzer iterates `VNDetectFaceLandmarksRequest` rectangles, crops
/// each, resizes to 64² grayscale, and calls `classify`.
struct EmotionClassifier {
    private var model: FERPlusModel? { FERPlusModel.shared }

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

private final class FERPlusModel {
    static var shared: FERPlusModel? = {
        try? FERPlusModel()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputSize: CGSize

    /// Minimum top-class softmax score required to surface a
    /// prediction. Below this the face is returned as nil so the UI
    /// hides its glyph instead of asserting a guess.
    private let confidenceFloor: Float = 0.35

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
        let outputs = model.modelDescription.outputDescriptionsByName
        guard let input = inputs.first(where: { $0.value.type == .image }) ?? inputs.first,
              let output = outputs.first
        else {
            throw AnalysisError.modelLoadFailed("Emotion model has no usable input/output.")
        }
        self.inputName = input.key
        self.outputName = output.key

        var resolvedSize = CGSize(width: 64, height: 64)
        if let constraint = input.value.imageConstraint {
            resolvedSize = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolvedSize
    }

    /// Crop the face rect from the source, convert to grayscale + the
    /// model's input size, run inference, and map the top softmax
    /// index to an `EmotionLabel`. Returns nil on any failure or when
    /// the top score falls under `confidenceFloor`.
    func predict(face: CGRect,
                 source: CIImage,
                 ciContext: CIContext) -> EmotionPrediction? {
        let extent = source.extent
        // Pad the face rect outward so the crop includes forehead / chin
        // — FER+ was trained on AFFECT/FER2013 crops that typically
        // enclose the whole face, not just the landmarks bounding box.
        let pad = face.width * 0.15
        let padded = face.insetBy(dx: -pad, dy: -pad)
        let clamped = padded.intersection(extent)
        guard !clamped.isNull, clamped.width >= 8, clamped.height >= 8 else {
            return nil
        }

        let cropped = source.cropped(to: clamped)
        let gray = cropped.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputGVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputBVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
        ])
        let resized = gray.stretched(to: inputSize)

        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_OneComponent8,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        ciContext.render(resized, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let scores = result.featureValue(for: outputName)?.multiArrayValue,
                  scores.count >= EmotionLabel.allCases.count
            else { return nil }

            // Raw scores may be logits (pre-softmax) or probabilities.
            // Normalize via stable softmax before thresholding so the
            // confidenceFloor works uniformly.
            var logits = [Float](repeating: 0, count: EmotionLabel.allCases.count)
            for i in 0..<logits.count {
                logits[i] = scores[i].floatValue
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
            guard topScore >= confidenceFloor else { return nil }
            let labels = EmotionLabel.allCases
            guard topIdx < labels.count else { return nil }

            // Confidence-weighted PAD projection — use the full
            // softmax distribution rather than the top class so a
            // mixture like "mostly neutral with some sad" lands
            // between the anchor points instead of snapping onto
            // neutral's (0,0,0).
            var p: Float = 0
            var a: Float = 0
            var d: Float = 0
            for i in 0..<probs.count where i < labels.count {
                let anchor = labels[i].padAnchor
                p += probs[i] * anchor.pleasure
                a += probs[i] * anchor.arousal
                d += probs[i] * anchor.dominance
            }
            return EmotionPrediction(
                label: labels[topIdx],
                confidence: topScore,
                pad: PADVector(pleasure: p, arousal: a, dominance: d)
            )
        } catch {
            return nil
        }
    }
}
