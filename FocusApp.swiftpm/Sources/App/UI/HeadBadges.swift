import SwiftUI

// MARK: - Per-subject overlays

/// Per-subject overlay graph rendered on top of the Metal view: the
/// debug-style NudeNet rect overlay, the floating per-body head-badge
/// stack, and the helpers each one consumes. Lifted out of
/// ContentView.swift so the toolbar / gesture / EXIF code there isn't
/// drowned in classifier-readout details. Methods stay as
/// `extension ContentView` so they retain access to `viewModel`,
/// `viewRect`, `subjectBoxFlash`, etc. via self.
extension ContentView {
    /// Debug-style overlay: draws one outlined rect + text label per
    /// NudeNet detection on top of the rendered photo. Hidden when the
    /// user toggles it off, when no detections exist, or during the
    /// press-and-hold-to-toggle "compare with original" gesture so the
    /// original photo is visible unadorned. The coordinate math
    /// mirrors `FocusRenderer.fit` so the boxes track the same
    /// aspect-fit + zoom transform the Metal view uses.
    @ViewBuilder
    func nudityLabelOverlay(in size: CGSize) -> some View {
        if viewModel.showNudityLabels,
           !viewModel.overlayHidden,
           !viewModel.nudityDetections.isEmpty,
           let extent = viewModel.sourceImage?.extent,
           extent.width > 0, extent.height > 0 {
            ZStack {
                // Subject (body) rects in green — only flashed for
                // 1 s when the Labels toggle flips on, so they read
                // as a momentary "here are the subjects I attributed
                // detections to" cue rather than persistent clutter.
                if subjectBoxFlash {
                    ForEach(Array(viewModel.bodyRectangles.enumerated()), id: \.offset) { _, body in
                        let r = viewRect(for: body, source: extent, in: size)
                        Rectangle()
                            .strokeBorder(.green, lineWidth: 2)
                            .frame(width: r.width, height: r.height)
                            .position(x: r.midX, y: r.midY)
                    }
                    .transition(.opacity)
                }
                ForEach(Array(viewModel.nudityDetections.enumerated()), id: \.offset) { _, det in
                    let r = viewRect(for: det.rect, source: extent, in: size)
                    // Orange for "*_EXPOSED" labels, yellow otherwise —
                    // surfaces the high-signal detections at a glance.
                    let tint: Color = det.label.uppercased().contains("EXPOSED")
                        ? .orange : .yellow
                    Rectangle()
                        .strokeBorder(tint, lineWidth: 2)
                        .frame(width: r.width, height: r.height)
                        .overlay(alignment: .topLeading) {
                            Text("\(det.label) \(Int((det.confidence * 100).rounded()))%")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.black)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(tint.opacity(0.9),
                                            in: RoundedRectangle(cornerRadius: 3))
                                .fixedSize()
                                .alignmentGuide(.top) { _ in 14 }
                        }
                        .position(x: r.midX, y: r.midY)
                }
            }
            .allowsHitTesting(false)
        }
    }

    /// Floating per-subject warning badge placed above each body whose
    /// NudeNet level is `.covered` or higher. Colour matches the
    /// counter's legend — yellow / orange / red for covered / partial /
    /// nude. Independent of the Labels overlay and the per-subject
    /// gate: this is a persistent per-person summary so the viewer can
    /// see at a glance which subjects are flagged and how severely.
    @ViewBuilder
    func nudeSubjectHeadBadges(in size: CGSize) -> some View {
        // Suppress the per-subject head stack whenever the NudeNet
        // label overlay is actually painting rects — the two annotate
        // the same subjects and stacking them gets noisy. The guard
        // mirrors `nudityLabelOverlay`'s visibility condition so a
        // Labels toggle with no detections still lets the head badges
        // through.
        let labelsActive = viewModel.showNudityLabels
            && !viewModel.nudityDetections.isEmpty
        if !viewModel.overlayHidden,
           !labelsActive,
           let extent = viewModel.sourceImage?.extent,
           extent.width > 0, extent.height > 0,
           viewModel.nudityLevels.count == viewModel.bodyRectangles.count {
            let badgeLayout = computeHeadBadgeLayout(
                bodies: viewModel.bodyRectangles,
                faces: viewModel.faceRectangles,
                viewSize: size
            ) { viewRect(for: $0, source: extent, in: size) }
            ForEach(Array(viewModel.bodyRectangles.enumerated()), id: \.offset) { index, body in
                let level = viewModel.nudityLevels[index]
                let prediction = predictionForBody(body)
                let emotion = prediction?.label
                let age = ageForBody(body)
                // Gender comes from NudeNet's FACE_* branch exclusively.
                // The age tier (SSR-Net) is age-only now.
                let gender = index < viewModel.nudityGenders.count
                    ? viewModel.nudityGenders[index]
                    : .unknown
                let warning = genitalWarning(forBodyAt: index)
                // Badge renders when any of the per-subject signals
                // have something to say — covered+ nudity, an emotion
                // prediction, a pain score, an age estimate, or a
                // genital detection attributed to this body. Safe
                // clothed photos without any of these stay unadorned.
                if level >= .covered || prediction != nil
                    || painForBody(body) != nil || age != nil
                    || warning != nil {
                    let pain = painForBody(body)
                    // When PAD/Pain is hidden and this body has an
                    // orgasm-level genital detection (uniquely the
                    // 1.00-bar case in `genitalWarning`), claim the
                    // emoji slot so the orgasm signal still surfaces
                    // somewhere on the badge.
                    let overrideEmoji: String? =
                        (!viewModel.showPADMeter
                         && (warning?.bars ?? 0) >= 0.99)
                            ? "😫"
                            : nil
                    VStack(spacing: 4) {
                        SubjectHeadBadge(
                            level: level,
                            gender: gender,
                            emotion: emotion,
                            overrideEmoji: overrideEmoji,
                            age: age
                        )
                        // Genital warning chip — sits directly under
                        // the head badge so the severity readout for
                        // the most explicit anatomy on this subject
                        // is visible before the optional PAD meter.
                        if let warning {
                            genitalWarningChip(
                                bars: warning.bars,
                                color: warning.color
                            )
                        }
                        // PAD + Pain live in the same meter row and
                        // share the single showPADMeter toggle.
                        if viewModel.showPADMeter
                            && (prediction?.pad != nil || pain != nil) {
                            HStack(spacing: 4) {
                                if let pad = prediction?.pad {
                                    subjectPADBars(for: pad)
                                }
                                if let pain {
                                    subjectPainBar(for: pain)
                                }
                            }
                        }
                    }
                    // Position resolved by `computeHeadBadgeLayout` —
                    // tries above-the-body first, falls back to
                    // below / left / right when the default would
                    // overlap another stack or push out of the
                    // viewport. Face overlap is allowed only when
                    // no in-frame, non-overlapping option exists.
                    .position(badgeLayout[index])
                    .allowsHitTesting(false)
                }
            }
        }
    }

    // MARK: - Per-subject readouts

    /// Highest-severity genital warning style for the body at
    /// `bodyIndex`, or nil when no genital detection is best-
    /// attributed to that body. Severity ladder maps the v3
    /// GenitalClassifier sub-classes onto cellularbars's discrete
    /// 4-bar scale:
    ///
    ///   Bars   Colour   Source label
    ///   ────   ──────   ──────────────────────────────────────
    ///   0/4    grey     COVERED (subject not nude), raw EXPOSED
    ///                   (no sub-class), FEMALE_* (any state)
    ///   1/4    grey     COVERED on a `.nude` subject (low-grade
    ///                   signal: genitals clothed, other regions
    ///                   aren't)
    ///   1/4    yellow   EXPOSED_LATENT (resting)
    ///   2/4    yellow   EXPOSED_TUMESCENT (rising) /
    ///                   EXPOSED_DETUMESCENT (subsiding)
    ///   3/4    orange   EXPOSED_AROUSAL OR
    ///                   COVERED_STIMULATION  ← mapped to AROUSAL
    ///   4/4    red      EXPOSED_ORGASM
    ///
    /// Body attribution mirrors NudityDetector.analyze() — max
    /// intersection area wins. When multiple genital detections
    /// land on the same body the highest-severity verdict is
    /// returned so the badge encodes the worst-case state.
    fileprivate func genitalWarning(
        forBodyAt bodyIndex: Int
    ) -> (bars: Double, color: Color)? {
        guard !viewModel.bodyRectangles.isEmpty,
              bodyIndex < viewModel.bodyRectangles.count
        else { return nil }
        let subjectLevel = bodyIndex < viewModel.nudityLevels.count
            ? viewModel.nudityLevels[bodyIndex]
            : .none

        // Order matters — checks run highest-severity first so a
        // single-keyword match short-circuits before the broader
        // COVERED / EXPOSED rules. COVERED_STIMULATION is matched
        // before plain COVERED so it gets the AROUSAL bump.
        func style(for label: String) -> (bars: Double, color: Color) {
            let upper = label.uppercased()
            if upper.contains("ORGASM")             { return (1.00, .red) }
            if upper.contains("COVERED_STIMULATION") { return (0.75, .orange) }
            if upper.contains("AROUSAL")            { return (0.75, .orange) }
            if upper.contains("TUMESCENT") {
                // Catches both EXPOSED_TUMESCENT (rising) and
                // EXPOSED_DETUMESCENT (subsiding) — symmetric
                // about the AROUSAL peak.
                return (0.50, .yellow)
            }
            if upper.contains("LATENT") { return (0.25, .yellow) }
            if upper.contains("COVERED") && subjectLevel == .nude {
                return (0.25, .gray)
            }
            return (0.00, .gray)
        }

        var top: (bars: Double, color: Color)? = nil
        for det in viewModel.nudityDetections
        where det.label.uppercased().contains("GENITALIA") {
            var bestIdx: Int? = nil
            var bestArea: CGFloat = 0
            for (i, b) in viewModel.bodyRectangles.enumerated() {
                let inter = b.intersection(det.rect)
                guard !inter.isNull else { continue }
                let area = inter.width * inter.height
                if area > bestArea {
                    bestArea = area
                    bestIdx = i
                }
            }
            guard bestIdx == bodyIndex else { continue }
            let s = style(for: det.label)
            if (top?.bars ?? -1) < s.bars { top = s }
        }
        return top
    }

    /// Compact warning chip surfaced inside `nudeSubjectHeadBadges`
    /// — `waveform.path.ecg` (ECG-style heart-rate trace, reads
    /// as physiological arousal) + cellularbars whose fill
    /// encodes severity. See `genitalWarning(forBodyAt:)` for the
    /// ladder.
    ///
    /// Background uses the same `Color.black.opacity(0.45)` tint
    /// as `SubjectHeadBadge` so the chip's capsule reads as a
    /// continuation of the head badge it sits beneath rather
    /// than a lighter-coloured detached pill.
    fileprivate func genitalWarningChip(
        bars: Double, color: Color
    ) -> some View {
        HStack(spacing: 2) {
            Image(systemName: "waveform.path.ecg")
            Image(systemName: "cellularbars", variableValue: bars)
        }
        .font(.caption2)
        .foregroundStyle(color)
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .liquidBadgeBackground(tint: Color.black.opacity(0.45), in: Capsule())
    }

    /// EmoNet-style P / A / D readout: three stacked horizontal bars
    /// with a center-origin tick. Positive values extend right in
    /// green, negative extend left in red. V and A come straight from
    /// EmoNet's regression heads; D is projected from the expression
    /// softmax via Mehrabian's anchor table, so its confidence band is
    /// weaker than V/A — we still surface it because it's the third
    /// standard PAD axis and callers asked for the full triple.
    fileprivate func subjectPADBars(for pad: PADVector) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            padBar(label: "V", value: pad.pleasure)
            padBar(label: "A", value: pad.arousal)
            padBar(label: "D", value: pad.dominance)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .liquidBadgeBackground(
            tint: Color.black.opacity(0.45),
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
        )
    }

    /// One V / A / D row: tiny label, then a bipolar fill bar. Value
    /// is clamped to [-1, 1] before mapping so the fill never exceeds
    /// the track on numeric slop (and so stale pre-clamp diagnostics
    /// during the EmoNet bring-up don't push the bar out of bounds).
    fileprivate func padBar(label: String, value: Float) -> some View {
        let clamped = CGFloat(max(-1, min(1, value)))
        let barWidth: CGFloat = 45
        let barHeight: CGFloat = 4
        let half = barWidth / 2
        let fillWidth = abs(clamped) * half
        return HStack(spacing: 3) {
            Text(label)
                .font(.system(size: 7, weight: .bold).monospacedDigit())
                .foregroundStyle(.white.opacity(0.9))
                .frame(width: 6, alignment: .leading)
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.white.opacity(0.18))
                    .frame(width: barWidth, height: barHeight)
                Capsule()
                    .fill(clamped >= 0 ? Color.green : Color.red)
                    .frame(width: fillWidth, height: barHeight)
                    .offset(x: clamped >= 0 ? half : half - fillWidth)
                Rectangle()
                    .fill(Color.white.opacity(0.5))
                    .frame(width: 1, height: barHeight)
                    .offset(x: half - 0.5)
            }
            .frame(width: barWidth, height: barHeight)
        }
    }

    /// Compact pain readout rendered under the PAD stack for a
    /// subject. Bandage prefix glyph + a gauge glyph whose
    /// dial position and tint encode the PSPI level:
    ///
    ///   none      gauge.low     green
    ///   mild      gauge.low     yellow
    ///   moderate  gauge.medium  orange
    ///   severe    gauge.high    red
    ///
    /// The continuous PSPI value (0..4) is intentionally
    /// collapsed onto the four-step ladder — matches the way
    /// `PainScore.Level` is displayed elsewhere.
    fileprivate func subjectPainBar(for pain: PainScore) -> some View {
        let (gaugeName, tint): (String, Color) = {
            switch pain.level {
            case .none:     return ("gauge.low",    .green)
            case .mild:     return ("gauge.low",    .yellow)
            case .moderate: return ("gauge.medium", .orange)
            case .severe:   return ("gauge.high",   .red)
            }
        }()
        return HStack(spacing: 6) {
            Image(systemName: "bandage.fill")
                .font(.system(size: 7, weight: .bold))
                .foregroundStyle(.white.opacity(0.9))
                .frame(width: 8, alignment: .leading)
            Image(systemName: gaugeName)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(tint)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .liquidBadgeBackground(
            tint: Color.black.opacity(0.45),
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
        )
    }

    // MARK: - Body → face attribution

    /// Find the face rect whose center lies inside `body` and return
    /// its FER+ prediction (carries label + PAD) when the classifier
    /// met its confidence floor. Returns nil when the emotion model
    /// isn't installed, no face matches the body, or the top
    /// prediction was filtered out.
    fileprivate func predictionForBody(_ body: CGRect) -> EmotionPrediction? {
        guard !viewModel.faceEmotions.isEmpty else { return nil }
        for (i, face) in viewModel.faceRectangles.enumerated()
            where i < viewModel.faceEmotions.count {
            let center = CGPoint(x: face.midX, y: face.midY)
            if body.contains(center) {
                return viewModel.faceEmotions[i]
            }
        }
        return nil
    }

    /// Find the PSPI pain score for the face matched to `body`. Same
    /// index-alignment contract as `predictionForBody`.
    fileprivate func painForBody(_ body: CGRect) -> PainScore? {
        guard !viewModel.painScores.isEmpty else { return nil }
        for (i, face) in viewModel.faceRectangles.enumerated()
            where i < viewModel.painScores.count {
            let center = CGPoint(x: face.midX, y: face.midY)
            if body.contains(center) {
                return viewModel.painScores[i]
            }
        }
        return nil
    }

    /// Find the age prediction for the face matched to `body`.
    /// Same index-alignment contract as `predictionForBody`.
    fileprivate func ageForBody(_ body: CGRect) -> AgePrediction? {
        guard !viewModel.ageEstimations.isEmpty else { return nil }
        for (i, face) in viewModel.faceRectangles.enumerated()
            where i < viewModel.ageEstimations.count {
            let center = CGPoint(x: face.midX, y: face.midY)
            if body.contains(center) {
                return viewModel.ageEstimations[i]
            }
        }
        return nil
    }
}

// MARK: - Layout solver

/// Resolve a `.position` for each subject's head-badge stack
/// against four constraints, in priority order:
///   0. The stack lives in the *area around the subject's
///      face* — candidates are above / below / left / right
///      of the matched face rect, not the whole body. Falls
///      back to the body's top-region candidates only when
///      no face was attributed to that body.
///   1. The stack must stay fully inside the viewport.
///   2. The stack must not overlap any other subject's stack.
///   3. The stack should not overlap any subject's face — but
///      face overlap is permitted when (1) and (2) leave no
///      face-clear option.
///
/// Per body, four face-relative candidate positions are
/// evaluated, each clamped into the viewport. The candidate
/// with the fewest stack collisions wins; ties on stack
/// collisions break by fewer face overlaps; further ties
/// favour "above" so the layout stays predictable when
/// nothing conflicts. Subjects are processed top-to-bottom by
/// face-Y so the uppermost subject claims the prime above-
/// the-face slot first.
///
/// Pure function: takes everything it needs as inputs and
/// produces SwiftUI view-coord centres. `mapToView` lifts a
/// source-extent `CGRect` (CIImage Y-up) into the same Y-down
/// view-coord space `.position` consumes — caller plugs in the
/// aspect-fit + zoom transform that mirrors `FocusRenderer.fit`.
func computeHeadBadgeLayout(
    bodies: [CGRect],
    faces: [CGRect],
    viewSize: CGSize,
    mapToView: (CGRect) -> CGRect
) -> [CGPoint] {
    let count = bodies.count
    guard count > 0 else { return [] }

    // Estimate size for the stack — typical head badge with
    // warning chip + the occasional PAD row. Tightened from
    // an earlier 160×80 because the over-wide footprint was
    // making collision-detection pessimistic, kicking the
    // chosen candidate out of the prime "above the face" slot
    // and putting the badge visibly far from the head. 130×72
    // is closer to the actual rendered size at typical content.
    let badgeSize = CGSize(width: 130, height: 72)
    let halfW = badgeSize.width / 2
    let halfH = badgeSize.height / 2
    // Gap between the badge edge and the anchor edge. Smaller
    // = stack hovers closer to the face.
    let gap: CGFloat = 2

    // View-rect copies of bodies + faces, computed once.
    let bodyViewRects = bodies.map(mapToView)
    let faceViewRects = faces.map(mapToView)

    // Match each body to its face (face whose source-extent
    // centre is inside the body). Mirrors `predictionForBody`
    // / `painForBody` / `ageForBody` so attribution stays
    // consistent across the per-subject UI surfaces. Returns
    // nil when no face matched — caller falls back to a
    // body-relative anchor.
    func faceViewRect(forBody bodyIndex: Int) -> CGRect? {
        let body = bodies[bodyIndex]
        for (i, face) in faces.enumerated() {
            let centre = CGPoint(x: face.midX, y: face.midY)
            if body.contains(centre) { return faceViewRects[i] }
        }
        return nil
    }

    // Anchor rect for each body: face when available, else
    // body's top region (so the fallback still hovers near
    // the head rather than the centre of the body box).
    let anchors: [CGRect] = (0..<count).map { i in
        if let f = faceViewRect(forBody: i) { return f }
        let b = bodyViewRects[i]
        // Top quartile of the body — a reasonable proxy for
        // "where the head probably is" when face detection
        // missed the subject.
        return CGRect(
            x: b.minX, y: b.minY,
            width: b.width, height: max(b.height * 0.25, 40)
        )
    }

    // Process top-to-bottom by anchor-Y so the uppermost
    // subject claims the prime above-the-face slot first.
    let order = (0..<count).sorted {
        anchors[$0].minY < anchors[$1].minY
    }

    var placed: [CGRect] = []
    var positions = Array(repeating: CGPoint.zero, count: count)

    for i in order {
        let anchor = anchors[i]
        let anchorCentre = CGPoint(x: anchor.midX, y: anchor.midY)
        // Face-adjacent candidate centres, before clamp.
        // "above" first so it wins ties when distances match.
        let raw: [CGPoint] = [
            CGPoint(x: anchor.midX, y: anchor.minY - gap - halfH),
            CGPoint(x: anchor.midX, y: anchor.maxY + gap + halfH),
            CGPoint(x: anchor.minX - gap - halfW, y: anchor.midY),
            CGPoint(x: anchor.maxX + gap + halfW, y: anchor.midY),
        ]
        // Clamp into viewport (priority 1).
        let candidates = raw.map { p in
            CGPoint(
                x: max(halfW + 4, min(viewSize.width - halfW - 4, p.x)),
                y: max(halfH + 8, min(viewSize.height - halfH - 8, p.y))
            )
        }

        // Score each candidate. Stack collisions trump face
        // overlaps (priority 2 vs 3); distance from the
        // anchor centre breaks remaining ties so the picker
        // hugs the face when nothing forces it away.
        var bestIdx = 0
        var bestStackHits = Int.max
        var bestFaceHits = Int.max
        var bestDistance: CGFloat = .greatestFiniteMagnitude
        for (idx, c) in candidates.enumerated() {
            let r = CGRect(
                x: c.x - halfW, y: c.y - halfH,
                width: badgeSize.width, height: badgeSize.height
            )
            let stackHits = placed.reduce(0) {
                $0 + (r.intersects($1) ? 1 : 0)
            }
            let faceHits = faceViewRects.reduce(0) {
                $0 + (r.intersects($1) ? 1 : 0)
            }
            let dx = c.x - anchorCentre.x
            let dy = c.y - anchorCentre.y
            let distance = (dx * dx + dy * dy).squareRoot()
            let better =
                stackHits < bestStackHits ||
                (stackHits == bestStackHits && faceHits < bestFaceHits) ||
                (stackHits == bestStackHits && faceHits == bestFaceHits
                    && distance < bestDistance)
            if better {
                bestStackHits = stackHits
                bestFaceHits = faceHits
                bestDistance = distance
                bestIdx = idx
            }
        }

        let chosen = candidates[bestIdx]
        placed.append(CGRect(
            x: chosen.x - halfW, y: chosen.y - halfH,
            width: badgeSize.width, height: badgeSize.height
        ))
        positions[i] = chosen
    }

    return positions
}

// MARK: - SubjectHeadBadge

/// Floating glyph pair rendered above each flagged subject's head:
/// a gender-inferred figure symbol (when NudeNet's FACE_* branch fired
/// on this body) next to the level-coloured warning shield. Colour is
/// driven by the aggregated NudeNet level — yellow / orange / red for
/// covered / partial / nude — matching the per-subject counter badge
/// legend. Gender symbol is omitted when unknown, so the badge stays
/// compact for unidentified subjects.
struct SubjectHeadBadge: View {
    let level: NudityLevel
    let gender: SubjectGender
    /// Optional FER+ emotion label rendered as an emoji before the
    /// shield/gender cluster. nil when the classifier wasn't run on
    /// this subject's face or fell under its confidence floor.
    let emotion: EmotionLabel?
    /// Optional emoji that replaces `emotion.emoji` for this subject.
    /// Used when an external signal (e.g. orgasm-level genital
    /// detection with the PAD meter off) wants to commandeer the
    /// emoji slot. When nil, falls through to the FER+ emoji.
    let overrideEmoji: String?
    /// Optional age prediction from SSR-Net. When present, the age
    /// is appended after the glyph cluster and the gender glyph
    /// lights up regardless of nudity level (so clothed subjects
    /// with an age estimate still show their NudeNet gender).
    let age: AgePrediction?

    var body: some View {
        HStack(spacing: 2) {
            if level >= .covered {
                Image(systemName: "exclamationmark.shield.fill")
                    .font(.caption)
                    .foregroundStyle(tint)
            }
            // Emoji slot — override wins, otherwise FER+ emoji.
            // Sits between shield and gender so the colored nudity /
            // gender glyphs stay adjacent to each other. Outside the
            // coloured foreground so it keeps its native emoji colour
            // rather than getting tinted.
            if let overrideEmoji {
                Text(overrideEmoji)
                    .font(.caption)
            } else if let emotion {
                Text(emotion.emoji)
                    .font(.caption)
            }
            // Show the gender glyph when either the nudity level
            // crossed the "covered" threshold (original rule) or the
            // age/gender model committed a confident prediction for
            // this subject. Pick the tint based on whichever signal
            // is stronger — level colour wins when present, white
            // otherwise so clothed-subject glyphs stay readable.
            if let glyph = gender.glyph,
               level >= .covered || age != nil {
                Text(glyph)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(level >= .covered ? tint : .white)
            }
            if let age {
                Text(Self.ageText(for: age))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.white)
            }
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .liquidBadgeBackground(tint: Color.black.opacity(0.45), in: Capsule())
    }

    /// SSR-Net outputs a scalar (no distribution) so we render just
    /// the integer age. Prior yu4u-based UI showed "mean ± stdev";
    /// that band isn't available here.
    private static func ageText(for age: AgePrediction) -> String {
        "\(Int(age.age.rounded()))"
    }

    private var tint: Color {
        switch level {
        case .covered: return .yellow
        case .partial: return .orange
        case .nude:    return .red
        case .none:    return .clear
        }
    }
}
