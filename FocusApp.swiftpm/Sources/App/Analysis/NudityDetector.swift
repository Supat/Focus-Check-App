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
        // Path 1: Create ML object-detector format — the clean case, NMS
        // already applied inside the model. Outputs are paired multi-arrays
        // `coordinates` (Nx4, normalized [cx,cy,w,h]) + `confidence` (NxC).
        if let coords = result.featureValue(for: coordinatesName)?.multiArrayValue,
           let conf = result.featureValue(for: confidenceName)?.multiArrayValue,
           coords.shape.count >= 2, conf.shape.count >= 2 {
            return parseCreateMLDetections(coords: coords, conf: conf, sourceExtent: sourceExtent)
        }

        // Path 2: raw YOLO tensor from coremltools-converted ONNX — one
        // multi-array output, no NMS baked in. Iterate all model outputs
        // and accept the first one whose shape matches a YOLO layout.
        for name in result.featureNames {
            guard let array = result.featureValue(for: name)?.multiArrayValue else { continue }
            if let detections = parseYOLODetections(array: array, sourceExtent: sourceExtent) {
                return classAgnosticNMS(detections, iouThreshold: 0.45)
            }
        }
        return []
    }

    // MARK: - Create ML object-detector path

    private func parseCreateMLDetections(coords: MLMultiArray,
                                         conf: MLMultiArray,
                                         sourceExtent: CGRect) -> [NudityDetection] {
        let count = coords.shape.first?.intValue ?? 0
        guard count > 0 else { return [] }
        let classCount = conf.shape[1].intValue
        let strideBox = coords.strides[0].intValue
        let strideConf = conf.strides[0].intValue

        var detections: [NudityDetection] = []
        detections.reserveCapacity(count)

        for i in 0..<count {
            var topClass = 0
            var topScore: Float = -1
            for c in 0..<classCount {
                let score = conf[i * strideConf + c].floatValue
                if score > topScore {
                    topScore = score
                    topClass = c
                }
            }
            guard topScore >= scoreThreshold, topClass < labels.count else { continue }

            // Create ML emits normalized [cx, cy, w, h] with Y measured
            // from the top. CGRect uses min-y with Y-up — flip via
            // y_ciimage_min = 1 - (cy + h/2).
            let cx = CGFloat(coords[i * strideBox + 0].floatValue)
            let cy = CGFloat(coords[i * strideBox + 1].floatValue)
            let bw = CGFloat(coords[i * strideBox + 2].floatValue)
            let bh = CGFloat(coords[i * strideBox + 3].floatValue)

            let rect = CGRect(
                x: (cx - bw / 2) * sourceExtent.width + sourceExtent.minX,
                y: (1 - cy - bh / 2) * sourceExtent.height + sourceExtent.minY,
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

    // MARK: - Raw YOLO path

    /// Decode a raw YOLOv5 or YOLOv8 output tensor. Returns nil when the
    /// shape doesn't look like either layout; the caller then moves on
    /// to the next output. Returns detections *before* NMS — the caller
    /// runs `classAgnosticNMS` afterward.
    ///
    /// Supported layouts (shape is `[1, a, b]`):
    ///   * YOLOv5: `[1, N, 5+C]` — per-anchor row [cx, cy, w, h, obj, cls0..clsC-1]
    ///   * YOLOv8: `[1, 4+C, N]` — channels-first, no objectness column
    ///
    /// Coords are in model-input pixel space (0...inputSize); this decoder
    /// normalizes to [0,1] and then maps to source extent.
    private func parseYOLODetections(array: MLMultiArray,
                                     sourceExtent: CGRect) -> [NudityDetection]? {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1 else { return nil }

        let classCount = labels.count
        let v5Feats = 5 + classCount
        let v8Feats = 4 + classCount

        // Layout probing. Prefer YOLOv5 when the last dim matches, else
        // YOLOv8 (channels-first). Reject anything else so we don't
        // accidentally "decode" a classifier head.
        let numAnchors: Int
        let numFeats: Int
        let channelsFirst: Bool
        let hasObjectness: Bool
        if shape[2] == v5Feats {
            numAnchors = shape[1]; numFeats = v5Feats
            channelsFirst = false;  hasObjectness = true
        } else if shape[2] == v8Feats {
            numAnchors = shape[1]; numFeats = v8Feats
            channelsFirst = false;  hasObjectness = false
        } else if shape[1] == v5Feats {
            numAnchors = shape[2]; numFeats = v5Feats
            channelsFirst = true;   hasObjectness = true
        } else if shape[1] == v8Feats {
            numAnchors = shape[2]; numFeats = v8Feats
            channelsFirst = true;   hasObjectness = false
        } else {
            return nil
        }

        let classOffset = hasObjectness ? 5 : 4
        let invInputW = 1 / Float(inputSize.width)
        let invInputH = 1 / Float(inputSize.height)

        // Index helper. For `[1, F, N]` (YOLOv8), stride over anchors.
        // For `[1, N, F]` (YOLOv5), stride over features.
        func read(anchor a: Int, feat f: Int) -> Float {
            let flat = channelsFirst ? (f * numAnchors + a) : (a * numFeats + f)
            return array[flat].floatValue
        }

        var detections: [NudityDetection] = []
        detections.reserveCapacity(min(numAnchors, 256))

        for a in 0..<numAnchors {
            // Max-class scan first — many anchors have all-low class scores,
            // so early-reject before computing the combined score avoids
            // the objectness read for most rows. Not a huge saving but
            // meaningful over 25k anchors.
            var topClass = 0
            var topScore: Float = -1
            for c in 0..<classCount {
                let s = read(anchor: a, feat: classOffset + c)
                if s > topScore {
                    topScore = s
                    topClass = c
                }
            }
            // YOLOv5: final score = obj * cls. YOLOv8: score = cls only
            // (no objectness column). The upstream exporter is expected
            // to have applied the sigmoids already — coremltools does this
            // by default when the ONNX graph includes them.
            let score: Float = {
                if hasObjectness {
                    let obj = read(anchor: a, feat: 4)
                    return obj * topScore
                }
                return topScore
            }()
            guard score >= scoreThreshold, topClass < classCount else { continue }

            let cx = read(anchor: a, feat: 0) * invInputW
            let cy = read(anchor: a, feat: 1) * invInputH
            let bw = read(anchor: a, feat: 2) * invInputW
            let bh = read(anchor: a, feat: 3) * invInputH

            // YOLO boxes are top-down; flip to CIImage Y-up.
            let rect = CGRect(
                x: CGFloat(cx - bw / 2) * sourceExtent.width + sourceExtent.minX,
                y: CGFloat(1 - cy - bh / 2) * sourceExtent.height + sourceExtent.minY,
                width: CGFloat(bw) * sourceExtent.width,
                height: CGFloat(bh) * sourceExtent.height
            )
            detections.append(NudityDetection(
                rect: rect,
                label: labels[topClass],
                confidence: score
            ))
        }
        return detections
    }

    /// Class-agnostic non-maximum suppression. NudeNet labels often overlap
    /// spatially (e.g. a breast detection and a body-part-exposed detection
    /// on the same pixel); suppressing across classes keeps the result
    /// clean when attributing back to a subject.
    private func classAgnosticNMS(_ detections: [NudityDetection],
                                  iouThreshold: Float) -> [NudityDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [NudityDetection] = []
        kept.reserveCapacity(sorted.count)
        for det in sorted {
            let overlaps = kept.contains { Self.iou(det.rect, $0.rect) > iouThreshold }
            if !overlaps { kept.append(det) }
        }
        return kept
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }
}
