import SwiftUI

private struct ExportedImage: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

struct ContentView: View {
    @StateObject private var viewModel = FocusViewModel()
    @State private var exportedImage: ExportedImage?
    @State private var isExporting = false
    /// Hides the toolbar + bottom control panel so the image occupies
    /// the whole window. Toggled from the toolbar button and from the
    /// floating close button that appears over the image while active.
    @State private var isFullScreen = false
    /// Drag-baseline for the pan gesture — captures the VM's pan at
    /// gesture start so successive drags accumulate rather than snap
    /// back to zero. Re-seeded on zoom toggle.
    @State private var panStart: CGSize = .zero
    /// In-flight timer that fires the badge-toggle if the user keeps
    /// holding for the full duration. Cancelled when the touch ends
    /// or moves out of range. SwiftUI's LongPressGesture callbacks
    /// fire at touch-down (not at minimumDuration), so we measure
    /// the hold duration ourselves.
    @State private var badgeHoldTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(viewModel.sourceName ?? "Focus Check")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    if viewModel.sourceImage != nil {
                        ToolbarItem(placement: .topBarLeading) {
                            Button(role: .destructive) {
                                viewModel.clear()
                            } label: {
                                Label("Remove photo", systemImage: "xmark.circle")
                            }
                        }
                        // Full-screen sits in front of the
                        // export / import I/O pair on the trailing
                        // side, with a fixed-width spacer between
                        // them so it reads as a separate group
                        // (view-state action vs. file I/O).
                        // ToolbarSpacer is iOS 26+; older OSes
                        // fall back to plain adjacency, which keeps
                        // the order correct but loses the visual
                        // gap.
                        ToolbarItem(placement: .primaryAction) {
                            fullScreenButton
                        }
                        if #available(iOS 26.0, macOS 26.0, *) {
                            ToolbarSpacer(.fixed, placement: .primaryAction)
                        }
                        ToolbarItem(placement: .primaryAction) {
                            exportButton
                        }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        ImageImporter(
                            onPick: { url, name in viewModel.load(url: url, name: name) },
                            onError: { message in viewModel.errorMessage = message }
                        )
                    }
                    // Custom principal title: Explicit badge (when the
                    // classifier flagged the image) followed by the file
                    // name. Lives in the toolbar instead of an image
                    // overlay so the press-and-hold-to-toggle compare gesture
                    // doesn't hide it — the flag state should remain
                    // visible at all times.
                    ToolbarItem(placement: .principal) {
                        principalTitle
                    }
                }
                .toolbar(isFullScreen ? .hidden : .automatic, for: .navigationBar)
                #if os(iOS)
                .statusBarHidden(isFullScreen)
                #endif
                .sheet(item: $exportedImage) { item in
                    ShareSheet(url: item.url)
                }
                .overlay(alignment: .top) {
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.callout)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .liquidBadgeBackground(in: Capsule())
                            .padding(.top, 8)
                            .transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
                .animation(.spring, value: viewModel.errorMessage)
        }
    }

    @ViewBuilder
    private var content: some View {
        VStack(spacing: 0) {
            ZStack {
                if viewModel.sourceImage == nil {
                    placeholder
                } else {
                    GeometryReader { geo in
                        ZStack {
                            MetalView(viewModel: viewModel)
                                .ignoresSafeArea(edges: .horizontal)
                                .contentShape(Rectangle())
                                .gesture(
                                    SpatialTapGesture(count: 2)
                                        .onEnded { event in
                                            let normalized = CGPoint(
                                                x: max(0, min(1, event.location.x / geo.size.width)),
                                                y: max(0, min(1, event.location.y / geo.size.height))
                                            )
                                            viewModel.toggleZoom(at: normalized)
                                        }
                                )
                                // Drag to pan when zoomed in. Gesture is
                                // recognized simultaneously with the tap /
                                // long-press so double-tap still fires and
                                // short drags don't hijack the compare
                                // hold. Only updates the VM's pan offset
                                // while zoomScale > 1 — a no-op at 1x so
                                // the unzoomed view isn't translatable.
                                .simultaneousGesture(
                                    DragGesture(minimumDistance: 6)
                                        .onChanged { value in
                                            guard viewModel.zoomScale > 1.001 else { return }
                                            let proposed = CGSize(
                                                width: panStart.width + value.translation.width,
                                                height: panStart.height + value.translation.height
                                            )
                                            viewModel.zoomPanOffset = clampedPan(
                                                proposed,
                                                viewSize: geo.size
                                            )
                                        }
                                        .onEnded { _ in
                                            panStart = viewModel.zoomPanOffset
                                        }
                                )
                                // Press-and-hold for 1.0 s on the image
                                // toggles the overlay / badges. Use a
                                // zero-distance DragGesture as a
                                // touch-began signal — its onChanged
                                // fires immediately at touch-down with
                                // translation ≈ 0, which LongPressGesture
                                // does not emit reliably in a
                                // simultaneousGesture context. Movement
                                // beyond 20 pt cancels the hold (the
                                // user is panning or scrolling, not
                                // deliberately holding still).
                                .simultaneousGesture(
                                    DragGesture(minimumDistance: 0)
                                        .onChanged { value in
                                            if badgeHoldTask == nil {
                                                badgeHoldTask = Task { @MainActor in
                                                    try? await Task.sleep(for: .milliseconds(1000))
                                                    if !Task.isCancelled {
                                                        viewModel.overlayHidden.toggle()
                                                    }
                                                    badgeHoldTask = nil
                                                }
                                            }
                                            let moved = hypot(
                                                value.translation.width,
                                                value.translation.height
                                            )
                                            if moved > 20 {
                                                badgeHoldTask?.cancel()
                                                badgeHoldTask = nil
                                            }
                                        }
                                        .onEnded { _ in
                                            badgeHoldTask?.cancel()
                                            badgeHoldTask = nil
                                        }
                                )
                            nudityLabelOverlay(in: geo.size)
                            nudeSubjectHeadBadges(in: geo.size)
                        }
                        // Contain any zoomed overlays (label boxes, head
                        // badges) to the image area — without this, a
                        // body rect mapped through the 2.5x zoom can
                        // place a badge above the viewport and paint
                        // over the toolbar.
                        .clipped()
                        .onChange(of: viewModel.zoomScale) { _, newScale in
                            // Re-seed the drag baseline any time the zoom
                            // toggles — otherwise the next drag picks up
                            // from a stale prior-zoom pan.
                            if newScale <= 1.001 { panStart = .zero }
                            else { panStart = viewModel.zoomPanOffset }
                        }
                    }
                }
                if let progress = viewModel.analysisProgress {
                    VStack(spacing: 8) {
                        ProgressView(value: progress.fraction)
                            .frame(width: 220)
                        if !progress.label.isEmpty {
                            Text(progress.label)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                    }
                    .padding()
                    .liquidBadgeBackground(in: RoundedRectangle(cornerRadius: 12))
                    .animation(.easeOut(duration: 0.15), value: progress.fraction)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .bottomLeading) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        nudeSubjectsBadge
                        contextBadge
                    }
                    HStack(spacing: 8) {
                        qualityBadge
                        aestheticBadge
                    }
                    HStack(spacing: 8) {
                        megapixelsBadge
                        exposureBadge
                        provenanceBadge
                        motionBlurBadge
                    }
                }
                .padding([.leading, .bottom], 12)
            }

            if !isFullScreen {
                Divider()

                OverlayControls(viewModel: viewModel)
                    .padding()
                    .background(.bar)
            }
        }
        .overlay(alignment: .topTrailing) {
            // Floating dismiss control only appears while full-screen is
            // active — the regular toolbar is hidden, so the user needs
            // an on-image way back.
            if isFullScreen {
                Button { toggleFullScreen() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.45))
                }
                .padding(16)
            }
        }
    }

    /// Toggles the full-screen mode: hides the navigation bar, bottom
    /// controls, and status bar so the image occupies the whole window.
    /// Animates lightly so the chrome fades instead of snapping.
    private var fullScreenButton: some View {
        Button { toggleFullScreen() } label: {
            Label(
                isFullScreen ? "Exit full screen" : "Full screen",
                systemImage: isFullScreen
                    ? "arrow.down.right.and.arrow.up.left"
                    : "arrow.up.left.and.arrow.down.right"
            )
        }
    }

    private func toggleFullScreen() {
        withAnimation(.easeOut(duration: 0.2)) {
            isFullScreen.toggle()
        }
    }

    private var exportButton: some View {
        Button {
            guard !isExporting else { return }
            isExporting = true
            Task {
                defer { Task { @MainActor in isExporting = false } }
                do {
                    let url = try await viewModel.exportPNG()
                    await MainActor.run {
                        exportedImage = ExportedImage(url: url)
                    }
                } catch {
                    await MainActor.run {
                        viewModel.errorMessage = error.localizedDescription
                    }
                }
            }
        } label: {
            if isExporting {
                ProgressView().controlSize(.small)
            } else {
                Label("Export PNG", systemImage: "square.and.arrow.up")
            }
        }
        .disabled(isExporting)
    }

    /// Integer megapixel count of the loaded source. Sits in front
    /// of the EXIF badge so resolution reads at a glance even when
    /// the photo carries no EXIF (e.g. screenshots, exported PNGs).
    /// Glows silver at 4K-or-above (≥ 8 MP) and gold beyond 8K
    /// (> 33 MP) — gives the viewer a quick visual cue for
    /// high-resolution sources.
    @ViewBuilder
    private var megapixelsBadge: some View {
        if let source = viewModel.sourceImage,
           !viewModel.overlayHidden {
            let pixels = source.extent.width * source.extent.height
            let mp = max(1, Int((pixels / 1_000_000).rounded()))
            // Reference resolutions:
            //   4K UHD = 3840×2160 ≈ 8.3 MP
            //   8K UHD = 7680×4320 ≈ 33.2 MP
            // Use rounded-MP integer thresholds so the badge text
            // and the glow are driven by the same number.
            let glow: Color? = {
                if mp > 33 { return Color(red: 1.00, green: 0.84, blue: 0.20) }
                if mp >= 8 { return Color(red: 0.80, green: 0.82, blue: 0.88) }
                return nil
            }()
            HStack(spacing: 6) {
                Image(systemName: "photo")
                Text("\(mp) MP")
                    .font(.caption.monospacedDigit())
            }
            // Shadow applied *before* the capsule background so it
            // forms around the icon + text alpha shapes (the
            // "content") rather than around the capsule's outer
            // edge. phaseAnimator cycles a breathing opacity on
            // the shadow so the glow pulses without any @State or
            // animation plumbing on our side.
            .modifier(MegapixelGlow(color: glow))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    private struct MegapixelGlow: ViewModifier {
        let color: Color?
        @ViewBuilder
        func body(content: Content) -> some View {
            if let color {
                content.phaseAnimator([0.35, 0.95]) { c, phase in
                    c.shadow(color: color.opacity(phase), radius: 6)
                } animation: { _ in
                    .easeInOut(duration: 1.2)
                }
            } else {
                content
            }
        }
    }

    @ViewBuilder
    private var exposureBadge: some View {
        if let info = viewModel.exposureInfo,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            let parts = [
                info.formattedFocalLength,
                info.formattedShutter,
                info.formattedFocusDistance
            ].compactMap { $0 }
            if !parts.isEmpty || info.flashFired == true {
                HStack(spacing: 6) {
                    Image(systemName: "camera.aperture")
                    if !parts.isEmpty {
                        Text(parts.joined(separator: " · "))
                            .font(.caption.monospacedDigit())
                    }
                    // Show only when EXIF Flash tag's bit 0 was set — i.e.
                    // the flash actually fired, not just 'flash mode auto'.
                    if info.flashFired == true {
                        Image(systemName: "bolt.fill")
                            .foregroundStyle(.yellow)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .liquidBadgeBackground(in: Capsule())
            }
        }
    }

    /// Provenance: surfaces the TIFF Software field (originating
    /// firmware or the editor that touched the file last) and a
    /// green checkmark.seal when a C2PA / Content Credentials
    /// manifest is embedded. Adobe and several major editors stamp
    /// this on export now; phone cameras and older editors
    /// generally don't, so absence isn't proof of authenticity.
    @ViewBuilder
    private var provenanceBadge: some View {
        if let info = viewModel.exposureInfo,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden,
           info.software != nil || info.hasContentCredentials {
            HStack(spacing: 6) {
                if info.hasContentCredentials {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                } else {
                    Image(systemName: "wand.and.stars")
                }
                if let sw = info.software {
                    Text(shortenedSoftware(sw))
                        .font(.caption.monospacedDigit())
                        .lineLimit(1)
                } else {
                    // C2PA manifest present but Software field
                    // stripped — surface the credentials state alone.
                    Text("Credentials")
                        .font(.caption)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    /// Trim verbose Software strings ("Adobe Photoshop 25.5.1
    /// (Macintosh)" → "Adobe Photoshop") so the badge stays
    /// readable in the bottom row without horizontal scrolling.
    /// Strips trailing version tokens (anything starting with a
    /// digit) and parenthesised platform tags.
    private func shortenedSoftware(_ s: String) -> String {
        var trimmed = s
        if let paren = trimmed.firstIndex(of: "(") {
            trimmed = String(trimmed[..<paren])
        }
        let parts = trimmed.split(separator: " ", omittingEmptySubsequences: true)
        let words = parts.prefix { token in
            !(token.first?.isNumber ?? false)
        }
        let joined = words.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        return joined.isEmpty ? trimmed.trimmingCharacters(in: .whitespaces) : joined
    }

    /// Navigation-bar principal item: the Explicit badge (when the
    /// classifier flagged the image) followed by the file name. Replaces
    /// the previous top-of-image overlay so the flag stays visible
    /// during the press-and-hold-to-toggle compare and doesn't clutter the photo.
    /// Colour keeps the original rule: red when NSFW confidence > 0.6,
    /// orange otherwise (or when SCA drove the verdict, which doesn't
    /// expose a confidence number).
    @ViewBuilder
    private var principalTitle: some View {
        HStack(spacing: 8) {
            if viewModel.isSensitive == true {
                let isHighConfidence = (viewModel.sensitiveConfidence ?? 0) > 0.6
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.shield.fill")
                    Text(viewModel.sensitiveLabel ?? "Sensitive")
                        .font(.caption)
                }
                .foregroundStyle(isHighConfidence ? Color.red : Color.orange)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .liquidBadgeBackground(tint: Color.black.opacity(0.4), in: Capsule())
            }
            Text(viewModel.sourceName ?? "Focus Check")
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    /// Debug-style overlay: draws one outlined rect + text label per
    /// NudeNet detection on top of the rendered photo. Hidden when the
    /// user toggles it off, when no detections exist, or during the
    /// press-and-hold-to-toggle "compare with original" gesture so the original
    /// photo is visible unadorned. The coordinate math mirrors
    /// `FocusRenderer.fit` so the boxes track the same aspect-fit +
    /// zoom transform the Metal view uses.
    @ViewBuilder
    private func nudityLabelOverlay(in size: CGSize) -> some View {
        if viewModel.showNudityLabels,
           !viewModel.overlayHidden,
           !viewModel.nudityDetections.isEmpty,
           let extent = viewModel.sourceImage?.extent,
           extent.width > 0, extent.height > 0 {
            ZStack {
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

    /// Highest-severity genital warning style for the body at
    /// `bodyIndex`, or nil when no genital detection is best-
    /// attributed to that body. Severity ladder:
    ///
    ///   Bars   Colour   Source label
    ///   ────   ──────   ──────────────────────────────────────
    ///   0/4    grey     COVERED (subject not nude) or raw
    ///                   EXPOSED (no sub-class assigned — also
    ///                   covers FEMALE_*, since the classifier
    ///                   doesn't refine female detections)
    ///   1/4    grey     COVERED on a subject whose aggregated
    ///                   NudityLevel is `.nude` (genitals clothed,
    ///                   other regions aren't — low-grade signal)
    ///   2/4    yellow   MALE_GENITALIA_FLACCID
    ///   3/4    orange   MALE_GENITALIA_AROUSAL
    ///   4/4    red      MALE_GENITALIA_ORGASM
    ///
    /// Body attribution mirrors NudityDetector.analyze() — max
    /// intersection area wins. When multiple genital detections
    /// land on the same body the highest-severity verdict is
    /// returned so the badge encodes the worst-case state.
    private func genitalWarning(
        forBodyAt bodyIndex: Int
    ) -> (bars: Double, color: Color)? {
        guard !viewModel.bodyRectangles.isEmpty,
              bodyIndex < viewModel.bodyRectangles.count
        else { return nil }
        let subjectLevel = bodyIndex < viewModel.nudityLevels.count
            ? viewModel.nudityLevels[bodyIndex]
            : .none

        func style(for label: String) -> (bars: Double, color: Color) {
            let upper = label.uppercased()
            if upper.contains("ORGASM")  { return (1.00, .red) }
            if upper.contains("AROUSAL") { return (0.75, .orange) }
            if upper.contains("FLACCID") { return (0.50, .yellow) }
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
    /// — exclamation octagon + cellularbars whose fill encodes
    /// severity. See `genitalWarning(forBodyAt:)` for the ladder.
    private func genitalWarningChip(
        bars: Double, color: Color
    ) -> some View {
        HStack(spacing: 2) {
            Image(systemName: "exclamationmark.octagon.fill")
            Image(systemName: "cellularbars", variableValue: bars)
        }
        .font(.caption2)
        .foregroundStyle(color)
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .liquidBadgeBackground(in: Capsule())
    }

    /// Floating per-subject warning badge placed above each body whose
    /// NudeNet level is `.covered` or higher. Colour matches the
    /// counter's legend — yellow / orange / red for covered / partial /
    /// nude. Independent of the Labels overlay and the per-subject
    /// gate: this is a persistent per-person summary so the viewer can
    /// see at a glance which subjects are flagged and how severely.
    @ViewBuilder
    private func nudeSubjectHeadBadges(in size: CGSize) -> some View {
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
                    let rect = viewRect(for: body, source: extent, in: size)
                    let pain = painForBody(body)
                    VStack(spacing: 4) {
                        SubjectHeadBadge(
                            level: level,
                            gender: gender,
                            emotion: emotion,
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
                    // Anchor the stack's top near the body's top edge
                    // so the head badge lands where it used to and the
                    // optional PAD capsule hangs below it. Guard against
                    // going above the viewport on very-near-top bodies.
                    .position(x: rect.midX, y: max(rect.minY - 4, 36))
                    .allowsHitTesting(false)
                }
            }
        }
    }

    /// EmoNet-style P / A / D readout: three stacked horizontal bars
    /// with a center-origin tick. Positive values extend right in
    /// green, negative extend left in red. V and A come straight from
    /// EmoNet's regression heads; D is projected from the expression
    /// softmax via Mehrabian's anchor table, so its confidence band is
    /// weaker than V/A — we still surface it because it's the third
    /// standard PAD axis and callers asked for the full triple.
    private func subjectPADBars(for pad: PADVector) -> some View {
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
    private func padBar(label: String, value: Float) -> some View {
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

    /// Find the face rect whose center lies inside `body` and return
    /// its FER+ prediction (carries label + PAD) when the classifier
    /// met its confidence floor. Returns nil when the emotion model
    /// isn't installed, no face matches the body, or the top
    /// prediction was filtered out.
    private func predictionForBody(_ body: CGRect) -> EmotionPrediction? {
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
    private func painForBody(_ body: CGRect) -> PainScore? {
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
    private func ageForBody(_ body: CGRect) -> AgePrediction? {
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
    private func subjectPainBar(for pain: PainScore) -> some View {
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

    /// Map a source-extent CIImage rect (Y-up) into a SwiftUI view rect
    /// (Y-down) that reflects the same aspect-fit + zoom transform the
    /// Metal renderer applies. Must stay in sync with `FocusRenderer.fit`.
    private func viewRect(for sourceRect: CGRect,
                          source: CGRect,
                          in viewSize: CGSize) -> CGRect {
        let scale = min(viewSize.width / source.width,
                        viewSize.height / source.height)
        let fittedW = source.width * scale
        let fittedH = source.height * scale
        let offsetX = (viewSize.width - fittedW) / 2
        let offsetY = (viewSize.height - fittedH) / 2

        // CIImage Y-up → SwiftUI Y-down, relative to the source origin.
        let localX = (sourceRect.minX - source.minX) * scale + offsetX
        let localY = (source.height - (sourceRect.maxY - source.minY)) * scale + offsetY
        var rect = CGRect(
            x: localX,
            y: localY,
            width: sourceRect.width * scale,
            height: sourceRect.height * scale
        )

        // Apply the same zoom transform the Metal view uses. The anchor
        // is stored in normalized view coordinates with Y-down origin,
        // so no flip needed here. The pan offset is in SwiftUI view
        // coords already — add it directly after the scale.
        let zoom = viewModel.zoomScale
        if zoom > 1.001 {
            let ax = viewModel.zoomAnchor.x * viewSize.width
            let ay = viewModel.zoomAnchor.y * viewSize.height
            rect = CGRect(
                x: (rect.minX - ax) * zoom + ax + viewModel.zoomPanOffset.width,
                y: (rect.minY - ay) * zoom + ay + viewModel.zoomPanOffset.height,
                width: rect.width * zoom,
                height: rect.height * zoom
            )
        }
        return rect
    }

    /// Constrain a proposed pan offset so the zoomed image still covers
    /// the viewport — prevents dragging into the letterbox void. The
    /// clamp derives the image's post-zoom bounds from the fit scale
    /// and the current zoom anchor, then caps pan so `image-min + pan
    /// ≤ 0` and `image-max + pan ≥ viewSize`. When the image is
    /// narrower / shorter than the view in a given axis (possible when
    /// zoomScale * fit < view size), pan for that axis is pinned to 0.
    private func clampedPan(_ pan: CGSize, viewSize: CGSize) -> CGSize {
        guard let extent = viewModel.sourceImage?.extent,
              extent.width > 0, extent.height > 0
        else { return .zero }
        let zoom = viewModel.zoomScale
        guard zoom > 1.001 else { return .zero }

        let fit = min(viewSize.width / extent.width,
                      viewSize.height / extent.height)
        let fittedW = extent.width * fit
        let fittedH = extent.height * fit
        let ax = viewModel.zoomAnchor.x * viewSize.width
        let ay = viewModel.zoomAnchor.y * viewSize.height

        // Image bounds in view coords after the anchor-centred zoom,
        // before the pan is added.
        let imageMinX = ax * (1 - zoom) + zoom * (viewSize.width - fittedW) / 2
        let imageMaxX = ax * (1 - zoom) + zoom * (viewSize.width + fittedW) / 2
        let imageMinY = ay * (1 - zoom) + zoom * (viewSize.height - fittedH) / 2
        let imageMaxY = ay * (1 - zoom) + zoom * (viewSize.height + fittedH) / 2

        // Pan limits that still fully cover the viewport. When the post-
        // zoom image is smaller than the view in an axis, we disable pan
        // on that axis (both bounds collapse so the clamp becomes 0).
        let panMaxX = max(0, -imageMinX)
        let panMinX = min(0, viewSize.width - imageMaxX)
        let panMaxY = max(0, -imageMinY)
        let panMinY = min(0, viewSize.height - imageMaxY)

        return CGSize(
            width: max(panMinX, min(panMaxX, pan.width)),
            height: max(panMinY, min(panMaxY, pan.height))
        )
    }

    /// CLIP zero-shot top context match. Shown as a tag icon + the
    /// prompt's display label + its similarity percent, so the viewer
    /// can see what the scene classifier thinks this image is about
    /// ("a photograph of a person in a bedroom" 34 %). Hidden when the
    /// CLIP bundle isn't installed or the classifier returned nothing.
    @ViewBuilder
    private var contextBadge: some View {
        if let top = viewModel.clipMatches.first,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            HStack(spacing: 6) {
                Image(systemName: "sparkle.magnifyingglass")
                Text(Self.sentenceCased(top.prompt))
                    .font(.caption)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                Text("\(Int((top.similarity * 100).rounded()))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    /// Upper-case the first letter of `s` while leaving the rest
    /// untouched — sentence case rather than title case, so prompts
    /// like "a photograph of a nude person" read as
    /// "A photograph of a nude person" without Title-Casing Every Word.
    private static func sentenceCased(_ s: String) -> String {
        guard let first = s.first else { return s }
        return first.uppercased() + s.dropFirst()
    }

    /// Per-subject counts from NudeNet, split by level so the user can
    /// see at a glance how many subjects fall into each bucket. Yellow =
    /// covered (swimwear / lingerie), orange = partial (one exposed
    /// region), red = nude (genital exposure or multiple exposed
    /// regions). Each bucket also respects the per-subject gate; the
    /// whole badge hides when the resulting total is zero so safe
    /// photos don't get a clutter capsule.
    @ViewBuilder
    private var nudeSubjectsBadge: some View {
        let levels = viewModel.nudityLevels
        let gate = viewModel.nudityGate
        let coveredCount = levels.filter { $0 == .covered && $0 >= gate }.count
        let partialCount = levels.filter { $0 == .partial && $0 >= gate }.count
        let nudeCount    = levels.filter { $0 == .nude    && $0 >= gate }.count
        let total = coveredCount + partialCount + nudeCount
        if total > 0,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            HStack(spacing: 10) {
                if coveredCount > 0 {
                    Label("\(coveredCount)", systemImage: "person.fill")
                        .foregroundStyle(.yellow)
                }
                if partialCount > 0 {
                    Label("\(partialCount)", systemImage: "person.fill")
                        .foregroundStyle(.orange)
                }
                if nudeCount > 0 {
                    Label("\(nudeCount)", systemImage: "person.fill")
                        .foregroundStyle(.red)
                }
            }
            .font(.caption.monospacedDigit())
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    @ViewBuilder
    private var motionBlurBadge: some View {
        // Normally hide the badge for sub-threshold readings, but when the user
        // has explicitly selected the Motion overlay, always surface the
        // number so the angle and confidence are visible even on mild cases.
        if let mb = viewModel.motionBlur,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden,
           mb.isSignificant || viewModel.style == .motion {
            HStack(spacing: 6) {
                Image(systemName: "arrow.left.and.right")
                    // Math angle (CCW from east) → SwiftUI rotation (CW positive).
                    .rotationEffect(.degrees(-Double(mb.angle)))
                Text("Motion blur ≈ \(Int(mb.angle.rounded()))°")
                    .font(.caption.monospacedDigit())
                Text("\(Int((mb.confidence * 100).rounded()))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    /// Whole-image technical-quality readout from NIMA/TID2013.
    /// Colour-codes the score: red below 4, orange 4–6, green
    /// above 6 — quick glance for "did this photo come out OK".
    /// Hidden when the model isn't installed.
    @ViewBuilder
    private var qualityBadge: some View {
        if let q = viewModel.qualityScore,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            nimaBadge(q, label: "Quality",
                      icon: "checkmark.seal.fill")
        }
    }

    /// Whole-image aesthetic-quality readout from NIMA/AVA. Same
    /// colour scale as the technical badge but different icon so
    /// they're distinguishable at a glance when sitting adjacent.
    @ViewBuilder
    private var aestheticBadge: some View {
        if let a = viewModel.aestheticScore,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            nimaBadge(a, label: "Aesthetic",
                      icon: "sparkles")
        }
    }

    /// Shared badge body for both NIMA variants — same layout and
    /// colour scale, different label + icon. Keeps the two capsules
    /// visually consistent while still distinguishable.
    private func nimaBadge(_ q: QualityScore, label: String,
                           icon: String) -> some View {
        let tint: Color = q.score < 4 ? .red
            : q.score < 6 ? .orange : .green
        return HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(tint)
            Text(String(format: "\(label) %.1f", Double(q.score)))
                .font(.caption.monospacedDigit())
            Text(String(format: "±%.1f", Double(q.stdev)))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .liquidBadgeBackground(in: Capsule())
    }

    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 64, weight: .thin))
                .foregroundStyle(.secondary)
            Text("Import a photo to begin")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text("RAW, HEIC, JPEG, and PNG are supported.")
                .font(.footnote)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            viewModel.load(url: url, name: url.lastPathComponent)
            return true
        }
    }
}

/// Thin UIKit wrapper around UIActivityViewController — SwiftUI's ShareLink
/// doesn't play well with URLs produced asynchronously after a user tap,
/// since it expects the item to be known at tap time.
private struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController,
                                context: Context) {}
}

/// Floating glyph pair rendered above each flagged subject's head:
/// a gender-inferred figure symbol (when NudeNet's FACE_* branch fired
/// on this body) next to the level-coloured warning shield. Colour is
/// driven by the aggregated NudeNet level — yellow / orange / red for
/// covered / partial / nude — matching the per-subject counter badge
/// legend. Gender symbol is omitted when unknown, so the badge stays
/// compact for unidentified subjects.
private struct SubjectHeadBadge: View {
    let level: NudityLevel
    let gender: SubjectGender
    /// Optional FER+ emotion label rendered as an emoji before the
    /// shield/gender cluster. nil when the classifier wasn't run on
    /// this subject's face or fell under its confidence floor.
    let emotion: EmotionLabel?
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
            if let emotion {
                // Emoji sits between shield and gender so the colored
                // nudity/gender glyphs stay adjacent to each other.
                // Outside the coloured foreground so it keeps its
                // native emoji colour rather than getting tinted.
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

/// Badge background that adopts Liquid Glass on iOS / iPadOS 26+ and
/// falls back to the pre-26 material (or a tint fill, if supplied) on
/// earlier systems. `compiler(>=6.2)` gates the `.glassEffect` symbol
/// so the code still builds on older Xcode / Swift Playgrounds SDKs —
/// Liquid Glass lights up automatically when built against an iOS 26+
/// SDK and run on a device that supports it.
extension View {
    @ViewBuilder
    func liquidBadgeBackground<S: Shape>(tint: Color? = nil, in shape: S) -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, *) {
            if let tint {
                self.glassEffect(.regular.tint(tint), in: shape)
            } else {
                self.glassEffect(.regular, in: shape)
            }
        } else {
            legacyBadgeBackground(tint: tint, in: shape)
        }
        #else
        legacyBadgeBackground(tint: tint, in: shape)
        #endif
    }

    @ViewBuilder
    private func legacyBadgeBackground<S: Shape>(tint: Color?, in shape: S) -> some View {
        if let tint {
            self.background(tint, in: shape)
        } else {
            self.background(.regularMaterial, in: shape)
        }
    }
}
