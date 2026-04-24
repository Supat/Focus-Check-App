import Foundation
import CoreImage
import CoreML

/// Per-face pain score computed from OpenGraphAU's Action Unit
/// probabilities plus a Vision-landmark-derived AU43 (eye closure).
///
/// Follows Prkachin & Solomon's PSPI formula:
///
///     PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
///
/// AUs 4 / 6 / 7 / 9 / 10 come from the Core ML model and sit in
/// [0, 1] (sigmoid probabilities, not FACS 0-5 intensities), and AU43
/// is an eye-aspect-ratio proxy also in [0, 1]. The resulting PSPI
/// therefore lives in [0, 4] — roughly a quarter of the clinical
/// 0-15 range. The UI bins it into no / mild / moderate / severe
/// using proportionally scaled thresholds.
struct PainScore: Hashable, Sendable {
    /// Summed PSPI proxy, clamped to [0, 4].
    let pspi: Float
    /// Individual AU probabilities used to form the sum — handy for
    /// debugging and for a future AU-level readout.
    let au4: Float
    let au6: Float
    let au7: Float
    let au9: Float
    let au10: Float
    let au43: Float

    /// Four-level bucket for a discrete UI label. Thresholds are
    /// scaled from the clinical PSPI bins (0 / 3 / 7 / 12 on the 0-15
    /// scale) down to our [0, 4] probability-sum range.
    enum Level: Sendable, Hashable {
        case none, mild, moderate, severe
    }
    var level: Level {
        switch pspi {
        case ..<0.8:  return .none
        case ..<1.9:  return .mild
        case ..<3.2:  return .moderate
        default:      return .severe
        }
    }
}

/// Thin wrapper around OpenGraphAU. Runs per face — the analyzer
/// iterates `VNDetectFaceLandmarksRequest` rectangles, crops each,
/// resizes to 224² RGB, and calls `classify`.
struct PainDetector {
    private var model: OpenGraphAUModel? { OpenGraphAUModel.shared }

    var isReady: Bool { ModelArchive.openGraphAU.isInstalled() }

    func warm() -> Bool { model != nil }

    /// Compute PSPI proxies for every face. `rolls` and `eyeOpenness`
    /// are parallel to `faces`; both come from Vision's face-landmarks
    /// pass. Returns an empty array when the model isn't installed;
    /// individual faces that fail (crop out-of-bounds, low
    /// probability on every AU) become nil entries so the indexing
    /// stays aligned with the face list.
    func classify(faces: [CGRect],
                  rolls: [CGFloat],
                  eyeOpenness: [Float],
                  in image: CIImage,
                  ciContext: CIContext) -> [PainScore?] {
        guard let model else { return [] }
        guard !faces.isEmpty else { return [] }
        return faces.enumerated().map { idx, face -> PainScore? in
            let roll = idx < rolls.count ? rolls[idx] : 0
            let openness = idx < eyeOpenness.count ? eyeOpenness[idx] : 0.3
            return model.predict(
                face: face,
                roll: roll,
                eyeOpenness: openness,
                source: image,
                ciContext: ciContext
            )
        }
    }
}

// MARK: - Model wrapper

private final class OpenGraphAUModel {
    static var shared: OpenGraphAUModel? = {
        try? OpenGraphAUModel()
    }()

    /// Upstream AU_ids order (main 27 + lateralized 14 = 41). The only
    /// indices we actually read for PSPI are the main-AU ones at
    /// positions 2, 4, 5, 6, 7 — matching '4', '6', '7', '9', '10'.
    /// Keeping the full list here as documentation for future callers
    /// that might want to surface other AUs.
    private static let auIDs: [String] = [
        "1", "2", "4", "5", "6", "7", "9", "10", "11", "12", "13",
        "14", "15", "16", "17", "18", "19", "20", "22", "23", "24",
        "25", "26", "27", "32", "38", "39",
        "L1", "R1", "L2", "R2", "L4", "R4", "L6", "R6",
        "L10", "R10", "L12", "R12", "L14", "R14"
    ]

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputSize = CGSize(width: 224, height: 224)

    init() throws {
        let url = try ModelArchive.openGraphAU.installedURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        // OpenGraphAU's graph head (per-AU linear projections +
        // einsum + topk neighbor aggregation) trips the GPU's MPSGraph
        // MLIR pass manager — same crash shape we hit during
        // coremltools' host-side sanity predict at export time. ANE
        // crashes on `.all`, GPU crashes on `.cpuAndGPU`, so pin the
        // model to CPU only. ResNet-50 on-CPU per face is ~100–200 ms
        // on an M1 iPad; acceptable given Focus Check runs analysis
        // once per photo load, not per frame.
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
                "OpenGraphAU model has no usable image input."
            )
        }
        self.inputName = input.key

        // Single 41-dim output. Prefer the exact name we asked the
        // converter to use; fall back to whichever output has 41
        // elements in case coremltools renamed it.
        let outputs = model.modelDescription.outputDescriptionsByName
        if outputs["au_probabilities"] != nil {
            self.outputName = "au_probabilities"
        } else {
            let match = outputs.first { _, desc in
                desc.multiArrayConstraint?.shape
                    .map(\.intValue)
                    .reduce(1, *) == Self.auIDs.count
            }
            guard let match else {
                throw AnalysisError.modelLoadFailed(
                    "OpenGraphAU model has no \(Self.auIDs.count)-dim output (available: \(Array(outputs.keys)))."
                )
            }
            self.outputName = match.key
        }

        print("[OpenGraphAU] loaded input=\(inputName) output=\(outputName)")
    }

    /// Crop + rotate + resize the face rect into 224², run inference,
    /// and compose the PSPI proxy. Mirrors the `EmotionClassifier`
    /// crop pipeline for consistency — a square, roll-corrected,
    /// aspect-preserving crop at ~25 % padding.
    func predict(face: CGRect,
                 roll: CGFloat,
                 eyeOpenness: Float,
                 source: CIImage,
                 ciContext: CIContext) -> PainScore? {
        guard face.width >= 8, face.height >= 8 else { return nil }

        let marginFactor: CGFloat = 1.5
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
            resized,
            to: pb,
            bounds: CGRect(origin: .zero, size: inputSize),
            colorSpace: sRGB
        )

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            print("[OpenGraphAU] predict begin face=(\(Int(face.origin.x)),\(Int(face.origin.y)),\(Int(face.width)),\(Int(face.height)))")
            let result = try model.prediction(from: features)
            print("[OpenGraphAU] predict returned")
            guard let probs = result.featureValue(for: outputName)?.multiArrayValue,
                  probs.count >= Self.auIDs.count
            else {
                print("[OpenGraphAU] predict: missing/short output (\(outputName))")
                return nil
            }

            // Index → AU mapping for PSPI inputs (positions in auIDs):
            //  4 → 2, 6 → 4, 7 → 5, 9 → 6, 10 → 7.
            func p(_ i: Int) -> Float {
                let v = probs[i].floatValue
                return v.isFinite ? max(0, min(1, v)) : 0
            }
            let au4 = p(2)
            let au6 = p(4)
            let au7 = p(5)
            let au9 = p(6)
            let au10 = p(7)
            // AU43 proxy: 1 - (eye openness / typical-open EAR).
            // Vision's eye landmarks give us roughly h/w ≈ 0.3 on a
            // wide-open eye, dropping toward 0 as the lids close.
            // Clamp so partially-open values don't saturate the sum.
            let au43 = max(0, min(1, 1 - eyeOpenness / 0.3))
            let pspi = au4 + max(au6, au7) + max(au9, au10) + au43

            return PainScore(
                pspi: max(0, min(4, pspi)),
                au4: au4, au6: au6, au7: au7,
                au9: au9, au10: au10, au43: au43
            )
        } catch {
            return nil
        }
    }
}
