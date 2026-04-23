import Foundation
import CoreImage
import CoreML
import Vision

/// Coarse per-subject sensitivity rating derived from NudeNet detections.
/// Ordered from least → most exposed; use `>=` for threshold checks.
enum NudityLevel: Int, Comparable, Equatable, Sendable {
    /// No flagged detections land on this subject — fully clothed or none
    /// of the classifier's labels fired.
    case none = 0
    /// Only `COVERED` labels (breasts covered, genitalia covered, feet, etc).
    /// Visible but clothed — swimwear, lingerie, bare arms/midriff.
    case covered = 1
    /// One `EXPOSED` label on a secondary region (breast, buttocks, belly,
    /// armpits). Partial nudity.
    case partial = 2
    /// Multiple `EXPOSED` labels or any genital exposure. Full nudity.
    case nude = 3

    static func < (lhs: NudityLevel, rhs: NudityLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var label: String {
        switch self {
        case .none:    return "Clothed"
        case .covered: return "Covered"
        case .partial: return "Partial"
        case .nude:    return "Nude"
        }
    }
}

/// One NudeNet detection at source-extent coordinates.
private struct NudityDetection {
    let rect: CGRect        // In source CIImage extent (Y-up).
    let label: String       // NudeNet class name, e.g. "FEMALE_BREAST_EXPOSED".
    let confidence: Float
}

/// NudeNet v3-style detector. Runs one Core ML inference over the whole
/// image, then attributes each detection to a body rectangle via
/// intersection area so we can surface a per-subject rating without a
/// classifier inference per person.
struct NudityDetector {
    private let classifier: NudityClassifier?

    init() {
        self.classifier = NudityClassifier.shared
    }

    /// True when the NudeNet model is available (installed + loaded).
    var isReady: Bool { classifier != nil }

    /// Run the detector and return one `NudityLevel` per body in `bodies`,
    /// preserving array order. Returns an empty array when the model is
    /// unavailable or detection failed — callers should treat that as
    /// "unknown" rather than "clothed".
    func levels(for image: CIImage,
                bodies: [CGRect],
                ciContext: CIContext) -> [NudityLevel] {
        guard let classifier else { return [] }
        guard !bodies.isEmpty else { return [] }
        let detections = classifier.detect(in: image, ciContext: ciContext)
        guard !detections.isEmpty else {
            return Array(repeating: .none, count: bodies.count)
        }

        return bodies.map { body -> NudityLevel in
            // Assign each detection to this body when ≥ 50 % of the
            // detection's area falls inside the body box — stricter than
            // a "center inside" test but still tolerant of the loose
            // VNDetectHumanRectangles boxes.
            let assigned = detections.filter { detection in
                let overlap = body.intersection(detection.rect)
                guard !overlap.isNull, detection.rect.width > 0, detection.rect.height > 0 else {
                    return false
                }
                let detArea = detection.rect.width * detection.rect.height
                return (overlap.width * overlap.height) / detArea >= 0.5
            }
            return aggregate(assigned)
        }
    }

    /// Map a bag of detections attributed to one subject into a level.
    /// Rules:
    /// - genitalia / anus exposure → `.nude`
    /// - ≥ 2 exposed labels → `.nude`
    /// - 1 exposed label → `.partial`
    /// - only covered labels → `.covered`
    /// - nothing → `.none`
    private func aggregate(_ detections: [NudityDetection]) -> NudityLevel {
        guard !detections.isEmpty else { return .none }

        var exposedCount = 0
        var hasCovered = false
        var hasCritical = false

        for det in detections {
            let upper = det.label.uppercased()
            if upper.contains("COVERED") {
                hasCovered = true
            } else if upper.contains("EXPOSED") {
                exposedCount += 1
                if upper.contains("GENITALIA") || upper.contains("ANUS") {
                    hasCritical = true
                }
            }
        }

        if hasCritical || exposedCount >= 2 { return .nude }
        if exposedCount == 1 { return .partial }
        if hasCovered { return .covered }
        return .none
    }
}

// MARK: - Model wrapper

/// Minimal wrapper around a downloaded NudeNet Core ML model. The model is
/// expected to follow Create ML's object-detector output convention
/// (`coordinates` Nx4 of normalized [cx, cy, w, h]; `confidence` NxC of
/// per-label probabilities) — this is what `coremltools.convert` produces
/// when fed a YOLO-style export with `ClassifierConfig` / object-detector
/// mode. If the artifact uses a different output shape, extend
/// `parseDetections(from:)`.
private final class NudityClassifier {
    static var shared: NudityClassifier? = {
        try? NudityClassifier()
    }()

    private let model: MLModel
    private let inputName: String
    private let coordinatesName: String
    private let confidenceName: String
    private let inputSize: CGSize

    /// Per-detection confidence floor — below this the detection is
    /// dropped. Create ML-exported object detectors already apply an
    /// internal threshold; this is a secondary guard.
    private let scoreThreshold: Float = 0.25

    /// Class labels in confidence-column order. NudeNet v3's default
    /// label set; override if the maintainer trained a different set.
    private static let defaultLabels: [String] = [
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED"
    ]
    private let labels: [String]

    init() throws {
        let url = try ModelArchive.nudenet.installedURL()
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

        // Pick the first image input — we don't care about the exact name,
        // just that it resolves at prediction time.
        guard let input = inputs.first(where: { $0.value.type == .image }) ?? inputs.first else {
            throw AnalysisError.modelLoadFailed("Model has no inputs")
        }
        self.inputName = input.key

        // Create ML object-detector exports name outputs "coordinates" and
        // "confidence"; allow common alternates too.
        self.coordinatesName = outputs.keys.first(where: {
            $0.lowercased().contains("coord") || $0.lowercased().contains("box")
        }) ?? "coordinates"
        self.confidenceName = outputs.keys.first(where: {
            $0.lowercased().contains("confidence") || $0.lowercased().contains("score")
        }) ?? "confidence"

        // Model image size — fixed input for YOLO-style detectors.
        var resolvedSize = CGSize(width: 320, height: 320)
        if let constraint = input.value.imageConstraint {
            resolvedSize = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolvedSize

        // Prefer embedded class labels when the model declares them.
        let meta = model.modelDescription.classLabels as? [String] ?? []
        self.labels = meta.isEmpty ? Self.defaultLabels : meta
    }

    /// Run one detector pass over the full image. Returns per-detection
    /// records in source-extent coordinates.
    fileprivate func detect(in image: CIImage,
                            ciContext: CIContext) -> [NudityDetection] {
        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        let resized = image.stretched(to: inputSize)

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return [] }
        ciContext.render(resized, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            return parseDetections(from: result, sourceExtent: image.extent)
        } catch {
            return []
        }
    }

    private func parseDetections(from result: MLFeatureProvider,
                                 sourceExtent: CGRect) -> [NudityDetection] {
        guard let coords = result.featureValue(for: coordinatesName)?.multiArrayValue,
              let conf = result.featureValue(for: confidenceName)?.multiArrayValue
        else { return [] }

        let count = coords.shape.first?.intValue ?? 0
        guard count > 0, coords.shape.count >= 2, conf.shape.count >= 2 else { return [] }
        let classCount = conf.shape[1].intValue
        let strideBox = coords.strides[0].intValue
        let strideConf = conf.strides[0].intValue

        var detections: [NudityDetection] = []
        detections.reserveCapacity(count)

        for i in 0..<count {
            // Pick the top class for this box.
            var topClass = 0
            var topScore: Float = -1
            for c in 0..<classCount {
                let score = conf[i * strideConf + c].floatValue
                if score > topScore {
                    topScore = score
                    topClass = c
                }
            }
            guard topScore >= scoreThreshold else { continue }
            guard topClass < labels.count else { continue }

            // Create ML box format: normalized [cx, cy, w, h] in image coords
            // with Y measured from the top. Convert to the CIImage Y-up
            // origin the rest of the pipeline uses.
            let cx = CGFloat(coords[i * strideBox + 0].floatValue)
            let cy = CGFloat(coords[i * strideBox + 1].floatValue)
            let bw = CGFloat(coords[i * strideBox + 2].floatValue)
            let bh = CGFloat(coords[i * strideBox + 3].floatValue)

            let xMin = (cx - bw / 2) * sourceExtent.width + sourceExtent.minX
            let yTop = (cy - bh / 2) * sourceExtent.height
            let yFlipped = sourceExtent.height - yTop - bh * sourceExtent.height + sourceExtent.minY
            let rect = CGRect(
                x: xMin,
                y: yFlipped,
                width: bw * sourceExtent.width,
                height: bh * sourceExtent.height
            )
            detections.append(NudityDetection(
                rect: rect,
                label: labels[topClass],
                confidence: topScore
            ))
        }
        return detections
    }
}
