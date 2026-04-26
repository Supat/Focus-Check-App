import Foundation
import CoreImage
import CoreMedia

/// Bundle of per-time analyzer signals — every field a video-mode
/// frame composite needs. Stored in `VideoSmoother` so per-display-
/// refresh interpolation can produce intermediate frames that glide
/// between two analysis pulses.
struct VideoSnapshot: Sendable {
    let time: CMTime
    var faceRectangles: [CGRect]
    var bodyRectangles: [CGRect]
    var groinRectangles: [CGRect]
    var chestRectangles: [CGRect]
    var eyeBars: [EyeBar]
    let personMask: CIImage?
    var nudityLevels: [NudityLevel]
    var nudityGenders: [SubjectGender]
    var nudityDetections: [NudityDetection]
}

/// Two-snapshot ring: holds the most recent analyzer pulse and the
/// one before it so per-display-refresh interpolation can produce
/// intermediate snapshots that ease the per-subject overlays from
/// one analyzed position to the next. Without this the mosaic and
/// head-badge stack snap visibly at every analysis pulse (~500ms).
///
/// Identity is reassigned on each `consume` via greedy IoU matching
/// — the curr snapshot's tracks are reordered so positions
/// `0..<min(prev.count, curr.count)` correspond to the same
/// real-world subject across pulses. This keeps the head badge for
/// "subject 1" attached to the same person across each pulse, and
/// gives the rect interpolator a meaningful pair to lerp between.
///
/// A subject in `curr` with no IoU match in `prev` is appended at
/// the end and shows at its raw curr position (no interpolation —
/// they "appear" at full position rather than gliding from
/// off-screen). A subject in `prev` with no match in `curr` simply
/// vanishes; the smoother does not render ghosts.
struct VideoSmoother {
    private(set) var prev: VideoSnapshot?
    private(set) var curr: VideoSnapshot?

    /// Greedy IoU threshold for declaring two rectangles "the same
    /// subject" across pulses. 0.3 is loose enough to follow a
    /// running subject between 500ms-spaced samples, tight enough
    /// to avoid swapping when two people are close.
    private static let matchThreshold: CGFloat = 0.3

    /// Per-pulse confidence multiplier on a carried-forward genital
    /// detection. 0.6 → after 2 missed pulses (1s) the carried
    /// detection sits at 0.36; after 3 (1.5s) at 0.22. Drops below
    /// `detectionPersistenceFloor` around pulse 4–5, so a detection
    /// the model can't re-see for ~2.5s vanishes naturally.
    private static let detectionPersistenceDecay: Float = 0.6
    /// Confidence floor below which a carried-forward detection is
    /// dropped instead of held. Aligns with NudeNet's own
    /// per-detection threshold so we don't surface signals weaker
    /// than the analyzer would have on its own.
    private static let detectionPersistenceFloor: Float = 0.05

    mutating func consume(_ raw: VideoSnapshot) {
        let aligned: VideoSnapshot
        if let prior = curr {
            let result = Self.alignBodies(curr: raw, to: prior)
            aligned = Self.persistGenitalDetections(
                curr: result.snapshot,
                prev: prior,
                outputIdxForPrev: result.outputIdxForPrev
            )
        } else {
            aligned = raw
        }
        prev = curr
        curr = aligned
    }

    mutating func reset() {
        prev = nil
        curr = nil
    }

    /// Snapshot to render for a given playhead `time`. Returns nil
    /// before any analysis pulse has been consumed; returns `curr`
    /// directly when only one pulse exists; otherwise interpolates
    /// between `prev` and `curr` using the linear factor derived
    /// from the time gap.
    func snapshot(at time: CMTime) -> VideoSnapshot? {
        guard let curr else { return nil }
        guard let prev else { return curr }
        let factor = Self.lerpFactor(time: time, prevTime: prev.time, currTime: curr.time)
        return Self.interpolated(prev: prev, curr: curr, factor: factor)
    }

    // MARK: - Identity matching

    /// Reorder `curr.bodyRectangles` (and the parallel `nudityLevels`
    /// / `nudityGenders` arrays) so that body indices that match a
    /// `prev` body via IoU sit at the same position in `prev`. New
    /// subjects (no IoU match) are appended at the end. Subjects
    /// that vanished from prev → curr aren't represented in the
    /// output — they don't get a slot, since the smoother has no
    /// curr position to interpolate them toward.
    ///
    /// `outputIdxForPrev[p]` is the index in the output snapshot
    /// where prev body `p` lives, or nil if prev body `p` had no
    /// IoU match in curr. Consumed by `persistGenitalDetections`
    /// to translate carried-forward detections from prev's
    /// coordinates into the output snapshot's body positions.
    private static func alignBodies(
        curr raw: VideoSnapshot, to prev: VideoSnapshot
    ) -> (snapshot: VideoSnapshot, outputIdxForPrev: [Int?]) {
        let prevCount = prev.bodyRectangles.count
        let currCount = raw.bodyRectangles.count
        guard currCount > 0 else {
            return (raw, Array(repeating: nil, count: prevCount))
        }

        let assignment = greedyIoUAssignment(
            curr: raw.bodyRectangles,
            prev: prev.bodyRectangles,
            iou: rectIoU,
            threshold: matchThreshold
        )

        // Build a `prevSlot → currIdx` lookup so we can reorder in
        // prev order, then append new tracks at the end.
        var currForPrev = Array<Int?>(repeating: nil, count: prevCount)
        var matchedCurr = Array<Bool>(repeating: false, count: currCount)
        for c in 0..<currCount {
            if let p = assignment[c] {
                currForPrev[p] = c
                matchedCurr[c] = true
            }
        }

        var order: [Int] = []
        order.reserveCapacity(currCount)
        var outputIdxForPrev = Array<Int?>(repeating: nil, count: prevCount)
        var nextOutputIdx = 0
        for p in 0..<prevCount {
            if let c = currForPrev[p] {
                order.append(c)
                outputIdxForPrev[p] = nextOutputIdx
                nextOutputIdx += 1
            }
        }
        for c in 0..<currCount where !matchedCurr[c] {
            order.append(c)
            nextOutputIdx += 1
        }

        let bodies = order.map { raw.bodyRectangles[$0] }
        let levels = order.map {
            $0 < raw.nudityLevels.count ? raw.nudityLevels[$0] : .none
        }
        let genders = order.map {
            $0 < raw.nudityGenders.count ? raw.nudityGenders[$0] : .unknown
        }

        var out = raw
        out.bodyRectangles = bodies
        out.nudityLevels = levels
        out.nudityGenders = genders
        return (out, outputIdxForPrev)
    }

    /// Carry forward genital detections from `prev` when the matching
    /// body track persists into `curr` but NudeNet didn't re-detect
    /// them this pulse. The carried rect is translated by the body
    /// displacement (curr_body.center − prev_body.center) and its
    /// confidence is multiplied by `detectionPersistenceDecay`. After
    /// confidence drops below `detectionPersistenceFloor` (typically
    /// after 4–5 missed pulses, ~2.5s at 2Hz) the carry stops.
    ///
    /// Scoped to GENITALIA detections because the genital warning
    /// chip is the most user-visible source of frame-to-frame
    /// flicker — non-genital detections aggregate into the per-body
    /// nudity level which absorbs single-pulse misses better. The
    /// pattern would extend cleanly to other labels if needed.
    ///
    /// The body-shield colour (driven by per-body NudityLevel) is
    /// intentionally NOT re-aggregated to include carried
    /// detections — it follows whatever NudeNet actually saw on
    /// this pulse. Brief "shield dimmed but chip still red" reads
    /// as honest; the chip carry-forward is what the user actually
    /// notices.
    private static func persistGenitalDetections(
        curr: VideoSnapshot,
        prev: VideoSnapshot,
        outputIdxForPrev: [Int?]
    ) -> VideoSnapshot {
        var out = curr
        for prevDet in prev.nudityDetections
        where prevDet.label.uppercased().contains("GENITALIA") {
            // Find prev body the detection attributed to (max area
            // intersection; mirrors NudityDetector.analyze).
            var bestPrevBody: Int? = nil
            var bestArea: CGFloat = 0
            for (i, body) in prev.bodyRectangles.enumerated() {
                let inter = body.intersection(prevDet.rect)
                guard !inter.isNull else { continue }
                let area = inter.width * inter.height
                if area > bestArea {
                    bestArea = area
                    bestPrevBody = i
                }
            }
            guard let prevBodyIdx = bestPrevBody,
                  prevBodyIdx < outputIdxForPrev.count,
                  let currBodyIdx = outputIdxForPrev[prevBodyIdx],
                  currBodyIdx < out.bodyRectangles.count
            else { continue }

            let currBody = out.bodyRectangles[currBodyIdx]
            // Skip if curr already has a GENITALIA detection on this
            // body — NudeNet found it this pulse, no need to carry.
            let alreadyHas = out.nudityDetections.contains { d in
                d.label.uppercased().contains("GENITALIA")
                    && currBody.intersects(d.rect)
            }
            if alreadyHas { continue }

            // Translate the prev detection to the curr body's frame
            // by the body-centre displacement.
            let prevBody = prev.bodyRectangles[prevBodyIdx]
            let dx = currBody.midX - prevBody.midX
            let dy = currBody.midY - prevBody.midY
            let translated = prevDet.rect.offsetBy(dx: dx, dy: dy)

            let newConfidence = prevDet.confidence * detectionPersistenceDecay
            guard newConfidence >= detectionPersistenceFloor else { continue }

            out.nudityDetections.append(NudityDetection(
                rect: translated,
                label: prevDet.label,
                confidence: newConfidence
            ))
        }
        return out
    }

    // MARK: - Interpolation

    private static func interpolated(
        prev: VideoSnapshot, curr: VideoSnapshot, factor: CGFloat
    ) -> VideoSnapshot {
        // Bodies are aligned (alignBodies above) so position-i in
        // both refers to the same subject. Lerp directly.
        let bodies = lerpRects(prev.bodyRectangles, curr.bodyRectangles, factor)

        // Faces / groins / chests / eyeBars aren't aligned across
        // snapshots — Vision returns them as flat arrays whose
        // order isn't stable. Run independent IoU matching per
        // array so each smooths separately. The head-badge solver
        // anchors on faces by center-in-body, so as long as each
        // array is internally smooth, the per-subject UI lands
        // consistently.
        let faces = lerpAndMatch(prev.faceRectangles, curr.faceRectangles, factor)
        let groins = lerpAndMatch(prev.groinRectangles, curr.groinRectangles, factor)
        let chests = lerpAndMatch(prev.chestRectangles, curr.chestRectangles, factor)
        let eyes = lerpEyeBars(prev.eyeBars, curr.eyeBars, factor)

        return VideoSnapshot(
            time: curr.time,
            faceRectangles: faces,
            bodyRectangles: bodies,
            groinRectangles: groins,
            chestRectangles: chests,
            eyeBars: eyes,
            personMask: curr.personMask,
            nudityLevels: curr.nudityLevels,
            nudityGenders: curr.nudityGenders,
            nudityDetections: curr.nudityDetections
        )
    }

    /// Aligned rect lerp — assumes `prev[i]` and `curr[i]` describe
    /// the same subject. Used for bodyRectangles after `alignBodies`.
    /// New tracks (i ≥ prev.count) appear at curr's raw position.
    private static func lerpRects(
        _ prev: [CGRect], _ curr: [CGRect], _ factor: CGFloat
    ) -> [CGRect] {
        var out: [CGRect] = []
        out.reserveCapacity(curr.count)
        for i in 0..<curr.count {
            if i < prev.count {
                out.append(rectLerp(prev[i], curr[i], factor))
            } else {
                out.append(curr[i])
            }
        }
        return out
    }

    /// Match-then-lerp for arrays where `prev` and `curr` aren't
    /// already aligned. Greedy IoU pairs each curr entry with its
    /// best prev match; matched pairs lerp, unmatched curr entries
    /// land at full curr position (no extrapolation).
    private static func lerpAndMatch(
        _ prev: [CGRect], _ curr: [CGRect], _ factor: CGFloat
    ) -> [CGRect] {
        guard !prev.isEmpty, !curr.isEmpty else { return curr }
        let assignment = greedyIoUAssignment(
            curr: curr, prev: prev, iou: rectIoU, threshold: matchThreshold
        )
        var out: [CGRect] = []
        out.reserveCapacity(curr.count)
        for c in 0..<curr.count {
            if let p = assignment[c] {
                out.append(rectLerp(prev[p], curr[c], factor))
            } else {
                out.append(curr[c])
            }
        }
        return out
    }

    /// EyeBar smoother. Match by center distance — IoU isn't well
    /// defined for the rotated bar geometry. Threshold is
    /// generous (~½ bar width) since eye bars move with heads
    /// which can shift more than IoU's intersection assumes.
    private static func lerpEyeBars(
        _ prev: [EyeBar], _ curr: [EyeBar], _ factor: CGFloat
    ) -> [EyeBar] {
        guard !prev.isEmpty, !curr.isEmpty else { return curr }
        // For each curr, find the closest prev by center distance,
        // bounded by half the curr bar's width.
        var out: [EyeBar] = []
        out.reserveCapacity(curr.count)
        var usedPrev = Set<Int>()
        for c in curr {
            var bestIdx: Int? = nil
            var bestD: CGFloat = c.size.width  // distance ≤ bar width counts as same eye
            for (pIdx, p) in prev.enumerated() where !usedPrev.contains(pIdx) {
                let dx = c.center.x - p.center.x
                let dy = c.center.y - p.center.y
                let d = (dx * dx + dy * dy).squareRoot()
                if d < bestD { bestD = d; bestIdx = pIdx }
            }
            if let p = bestIdx {
                usedPrev.insert(p)
                out.append(eyeBarLerp(prev[p], c, factor))
            } else {
                out.append(c)
            }
        }
        return out
    }

    // MARK: - Matching helpers

    /// Returns `prev` index that matches each `curr` entry, or nil
    /// when no `prev` entry passes the IoU threshold and isn't
    /// already claimed. Greedy: highest-IoU pair claimed first.
    private static func greedyIoUAssignment<T>(
        curr: [T], prev: [T],
        iou: (T, T) -> CGFloat,
        threshold: CGFloat
    ) -> [Int?] {
        var assignment = Array<Int?>(repeating: nil, count: curr.count)
        guard !prev.isEmpty else { return assignment }
        var pairs: [(c: Int, p: Int, iou: CGFloat)] = []
        pairs.reserveCapacity(curr.count * prev.count)
        for c in 0..<curr.count {
            for p in 0..<prev.count {
                let v = iou(curr[c], prev[p])
                if v > threshold { pairs.append((c, p, v)) }
            }
        }
        pairs.sort { $0.iou > $1.iou }
        var usedPrev = Set<Int>()
        for pair in pairs {
            if assignment[pair.c] != nil { continue }
            if usedPrev.contains(pair.p) { continue }
            assignment[pair.c] = pair.p
            usedPrev.insert(pair.p)
        }
        return assignment
    }

    private static func rectIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return interArea / unionArea
    }

    // MARK: - Lerp primitives

    private static func rectLerp(_ a: CGRect, _ b: CGRect, _ t: CGFloat) -> CGRect {
        CGRect(
            x: a.minX + (b.minX - a.minX) * t,
            y: a.minY + (b.minY - a.minY) * t,
            width: a.width + (b.width - a.width) * t,
            height: a.height + (b.height - a.height) * t
        )
    }

    private static func eyeBarLerp(_ a: EyeBar, _ b: EyeBar, _ t: CGFloat) -> EyeBar {
        EyeBar(
            center: CGPoint(
                x: a.center.x + (b.center.x - a.center.x) * t,
                y: a.center.y + (b.center.y - a.center.y) * t
            ),
            size: CGSize(
                width: a.size.width + (b.size.width - a.size.width) * t,
                height: a.size.height + (b.size.height - a.size.height) * t
            ),
            angleRadians: angleLerp(a.angleRadians, b.angleRadians, t)
        )
    }

    /// Angle interp on the shorter arc — picks the wraparound that
    /// avoids a 359° → 1° spin through the full circle.
    private static func angleLerp(_ a: CGFloat, _ b: CGFloat, _ t: CGFloat) -> CGFloat {
        var delta = b - a
        let twoPi = CGFloat.pi * 2
        if delta > .pi { delta -= twoPi }
        if delta < -.pi { delta += twoPi }
        return a + delta * t
    }

    /// Linear factor in [0, 1]: 0 means "render prev", 1 means
    /// "render curr". Outside the prev → curr window the result
    /// is clamped, so scrubbing past curr stays at curr (no
    /// extrapolation) and scrubbing before prev stays at prev.
    private static func lerpFactor(
        time: CMTime, prevTime: CMTime, currTime: CMTime
    ) -> CGFloat {
        let dt = CMTimeGetSeconds(currTime) - CMTimeGetSeconds(prevTime)
        let progress = CMTimeGetSeconds(time) - CMTimeGetSeconds(prevTime)
        guard dt > 0 else { return 1 }
        return CGFloat(max(0, min(1, progress / dt)))
    }
}
