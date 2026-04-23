import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
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

/// One NudeNet detection at source-extent coordinates. Exposed beyond
/// the detector so the UI can draw per-box overlays when the user
/// enables the label toggle.
struct NudityDetection: Hashable, Sendable {
    let rect: CGRect        // In source CIImage extent (Y-up).
    let label: String       // NudeNet class name, e.g. "FEMALE_BREAST_EXPOSED".
    let confidence: Float
}

/// Inferred subject gender from NudeNet's `FACE_MALE` / `FACE_FEMALE`
/// detections attributed to a body. This is a classifier label, not
/// ground truth — it reads exactly as well as NudeNet's face branch
/// does. `.unknown` when no face detection landed on the body.
enum SubjectGender: Sendable, Equatable {
    case unknown
    case male
    case female

    /// Unicode glyph for the overhead badge — Mars / Venus. Returns nil
    /// for `.unknown` so the badge stays compact when NudeNet didn't
    /// commit to either label.
    var glyph: String? {
        switch self {
        case .unknown: return nil
        case .male:    return "♂"
        case .female:  return "♀"
        }
    }
}

/// Combined result of one detection pass: per-subject levels (same
/// order as the bodies argument), inferred gender per body, plus the
/// raw per-part detections in source-extent coordinates. Keeping all
/// three in one return value lets callers skip repeat detection runs.
struct NudityAnalysis: Sendable {
    let levels: [NudityLevel]
    let genders: [SubjectGender]
    let detections: [NudityDetection]
}

/// NudeNet v3-style detector. Runs one Core ML inference over the whole
/// image, then attributes each detection to a body rectangle via
/// intersection area so we can surface a per-subject rating without a
/// classifier inference per person.
struct NudityDetector {
    /// Defer touching `NudityClassifier.shared` until the first detection
    /// run — the shared static loads a 10-MB+ Core ML model, which would
    /// otherwise synchronously block app launch when the analyzer is
    /// constructed during `FocusViewModel.init`.
    private var classifier: NudityClassifier? { NudityClassifier.shared }

    /// True when the NudeNet model is installed on disk. Mirrors the
    /// disk-presence check used by the other models — resolves without
    /// forcing the MLModel load.
    var isReady: Bool { ModelArchive.nudenet.isInstalled() }

    /// Run the detector and return per-subject levels plus the raw
    /// per-part detections (for optional UI overlay). Returns empty
    /// levels when the model is unavailable or detection failed —
    /// callers should treat that as "unknown" rather than "clothed".
    func analyze(image: CIImage,
                 bodies: [CGRect],
                 ciContext: CIContext) -> NudityAnalysis {
        guard let classifier else {
            return NudityAnalysis(levels: [], genders: [], detections: [])
        }
        guard !bodies.isEmpty else {
            return NudityAnalysis(levels: [], genders: [], detections: [])
        }
        let detections = classifier.detect(in: image, ciContext: ciContext)
        guard !detections.isEmpty else {
            return NudityAnalysis(
                levels: Array(repeating: .none, count: bodies.count),
                genders: Array(repeating: .unknown, count: bodies.count),
                detections: []
            )
        }

        // Bucket detections per body by maximum intersection area —
        // each detection lands with whichever body it overlaps most.
        // Any positive overlap is enough to attribute (the earlier
        // "≥ 50 % inside the body box" rule dropped valid detections
        // when the Vision body rect was tight or loose in the wrong
        // direction). Detections with no overlap against any body are
        // still returned for the label overlay.
        var bags: [[NudityDetection]] = Array(repeating: [], count: bodies.count)
        for det in detections {
            var bestIdx: Int? = nil
            var bestArea: CGFloat = 0
            for (i, body) in bodies.enumerated() {
                let inter = body.intersection(det.rect)
                guard !inter.isNull else { continue }
                let area = inter.width * inter.height
                if area > bestArea {
                    bestArea = area
                    bestIdx = i
                }
            }
            if let idx = bestIdx {
                bags[idx].append(det)
            }
        }
        return NudityAnalysis(
            levels: bags.map(aggregate),
            genders: bags.map(inferGender),
            detections: detections
        )
    }

    /// Pick the gender implied by the FACE_MALE / FACE_FEMALE detections
    /// in a body's bag. Highest-confidence face wins when NudeNet fires
    /// both labels on the same subject (rare but possible when the face
    /// is partially obscured). Returns `.unknown` when no face-branch
    /// detection was attributed to this body.
    private func inferGender(from detections: [NudityDetection]) -> SubjectGender {
        var best: (gender: SubjectGender, confidence: Float) = (.unknown, 0)
        for det in detections {
            let upper = det.label.uppercased()
            let gender: SubjectGender?
            if upper == "FACE_MALE"        { gender = .male }
            else if upper == "FACE_FEMALE" { gender = .female }
            else                           { gender = nil }
            if let g = gender, det.confidence > best.confidence {
                best = (g, det.confidence)
            }
        }
        return best.gender
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
    /// dropped. Matches NudeNet's own Python default (0.2) rather than
    /// YOLO's generic 0.25; higher values dropped mid-confidence
    /// matches on obvious subjects.
    private let scoreThreshold: Float = 0.2

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

    /// Aspect-preserving letterbox parameters shared by the detect path
    /// and the coordinate-mapping helpers. Stretching the image to the
    /// model's square input distorts non-square photos enough to drop
    /// valid subject detections; letterboxing matches how YOLOv5/v8 were
    /// trained, so the detector behaves the way it was intended to.
    private struct Letterbox {
        let scale: CGFloat      // input-pixel per source-pixel
        let offsetX: CGFloat    // pad at left / right in input-pixel space
        let offsetY: CGFloat    // pad at top / bottom in input-pixel space
        let inputSize: CGSize
        let sourceExtent: CGRect

        /// Build a source-extent CGRect (CIImage Y-up) from a detection
        /// expressed in input-pixel top-down space (cx, cy, w, h).
        func sourceRect(cx: CGFloat, cy: CGFloat, w: CGFloat, h: CGFloat) -> CGRect {
            let srcCx = (cx - offsetX) / scale
            let srcCyTopDown = (cy - offsetY) / scale
            let srcW = w / scale
            let srcH = h / scale
            return CGRect(
                x: srcCx - srcW / 2 + sourceExtent.minX,
                y: sourceExtent.height - srcCyTopDown - srcH / 2 + sourceExtent.minY,
                width: srcW,
                height: srcH
            )
        }

        /// Create ML variant: same conversion but from normalized [0, 1]
        /// inputs by first scaling to input-pixel space.
        func sourceRectFromNormalized(cx: CGFloat, cy: CGFloat, w: CGFloat, h: CGFloat) -> CGRect {
            sourceRect(
                cx: cx * inputSize.width,
                cy: cy * inputSize.height,
                w: w * inputSize.width,
                h: h * inputSize.height
            )
        }
    }

    /// Run one detector pass over the full image. Returns per-detection
    /// records in source-extent coordinates.
    fileprivate func detect(in image: CIImage,
                            ciContext: CIContext) -> [NudityDetection] {
        let w = Int(inputSize.width)
        let h = Int(inputSize.height)
        let letterboxedInput = letterbox(image: image, into: inputSize)

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return [] }
        ciContext.render(letterboxedInput.image, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            return parseDetections(from: result, letterbox: letterboxedInput.box)
        } catch {
            return []
        }
    }

    /// Fit `image` into a centered region of a `size × size` canvas and
    /// pad with YOLO's customary 114/255 gray. Returns the composed CIImage
    /// plus the Letterbox params used, so the caller can undo the transform
    /// when mapping detection boxes back to source coordinates.
    private func letterbox(image: CIImage, into size: CGSize) -> (image: CIImage, box: Letterbox) {
        let src = image.extent
        let scale = min(size.width / src.width, size.height / src.height)
        let scaledW = src.width * scale
        let scaledH = src.height * scale
        let offsetX = (size.width - scaledW) / 2
        let offsetY = (size.height - scaledH) / 2

        let scaled = image.translatedToOrigin()
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let placed = scaled
            .transformed(by: CGAffineTransform(translationX: offsetX, y: offsetY))

        let pad = CIImage(color: CIColor(red: 114/255, green: 114/255, blue: 114/255))
            .cropped(to: CGRect(origin: .zero, size: size))
        let over = CIFilter.sourceOverCompositing()
        over.inputImage = placed
        over.backgroundImage = pad
        let composed = (over.outputImage ?? placed)
            .cropped(to: CGRect(origin: .zero, size: size))

        let box = Letterbox(
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY,
            inputSize: size,
            sourceExtent: src
        )
        return (composed, box)
    }

    private func parseDetections(from result: MLFeatureProvider,
                                 letterbox: Letterbox) -> [NudityDetection] {
        // Path 1: Create ML object-detector format — the clean case, NMS
        // already applied inside the model. Outputs are paired multi-arrays
        // `coordinates` (Nx4, normalized [cx,cy,w,h]) + `confidence` (NxC).
        if let coords = result.featureValue(for: coordinatesName)?.multiArrayValue,
           let conf = result.featureValue(for: confidenceName)?.multiArrayValue,
           coords.shape.count >= 2, conf.shape.count >= 2 {
            return parseCreateMLDetections(coords: coords, conf: conf, letterbox: letterbox)
        }

        // Path 2: raw YOLO tensor from coremltools-converted ONNX — one
        // multi-array output, no NMS baked in. Iterate all model outputs
        // and accept the first one whose shape matches a YOLO layout.
        for name in result.featureNames {
            guard let array = result.featureValue(for: name)?.multiArrayValue else { continue }
            if let detections = parseYOLODetections(array: array, letterbox: letterbox) {
                return classAgnosticNMS(detections, iouThreshold: 0.45)
            }
        }
        return []
    }

    // MARK: - Create ML object-detector path

    private func parseCreateMLDetections(coords: MLMultiArray,
                                         conf: MLMultiArray,
                                         letterbox: Letterbox) -> [NudityDetection] {
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

            // Create ML emits normalized [cx, cy, w, h] in the letterboxed
            // input's coordinate space. Undo the letterbox to land in
            // source-extent CIImage Y-up coords.
            let cx = CGFloat(coords[i * strideBox + 0].floatValue)
            let cy = CGFloat(coords[i * strideBox + 1].floatValue)
            let bw = CGFloat(coords[i * strideBox + 2].floatValue)
            let bh = CGFloat(coords[i * strideBox + 3].floatValue)

            let rect = letterbox.sourceRectFromNormalized(cx: cx, cy: cy, w: bw, h: bh)
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
    /// undoes the letterbox to land in source-extent CIImage coords.
    private func parseYOLODetections(array: MLMultiArray,
                                     letterbox: Letterbox) -> [NudityDetection]? {
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

            // YOLO coords are input-pixel, top-down — hand to the
            // letterbox helper to undo the aspect-fit transform and flip
            // to source-extent CIImage Y-up.
            let rect = letterbox.sourceRect(
                cx: CGFloat(read(anchor: a, feat: 0)),
                cy: CGFloat(read(anchor: a, feat: 1)),
                w: CGFloat(read(anchor: a, feat: 2)),
                h: CGFloat(read(anchor: a, feat: 3))
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
