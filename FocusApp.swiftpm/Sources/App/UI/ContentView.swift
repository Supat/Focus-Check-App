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
                        ToolbarItem(placement: .primaryAction) {
                            fullScreenButton
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
                    // overlay so press-and-hold compare doesn't hide it —
                    // the flag state should remain visible at all times.
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
                                // Press-and-hold hides the overlay so the user can
                                // compare against the original photo; release
                                // re-reveals it.
                                .onLongPressGesture(
                                    minimumDuration: 0.2,
                                    maximumDistance: 100,
                                    perform: {},
                                    onPressingChanged: { pressing in
                                        viewModel.overlayHidden = pressing
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
                if viewModel.isAnalyzing {
                    ProgressView("Analyzing…")
                        .padding()
                        .liquidBadgeBackground(in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .bottomLeading) {
                HStack(spacing: 8) {
                    exposureBadge
                    motionBlurBadge
                    nudeSubjectsBadge
                    contextBadge
                    moodBadge
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
                Button {
                    withAnimation(.easeOut(duration: 0.2)) {
                        isFullScreen = false
                    }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.45))
                }
                .padding(16)
            }
        }
    }

    /// Triggers an async composite + PNG encode on the view model, then
    /// presents the system share sheet so the user can save to Files,
    /// Photos, or send via any registered share target.
    /// Toggles the full-screen mode: hides the navigation bar, bottom
    /// controls, and status bar so the image occupies the whole window.
    /// Animates lightly so the chrome fades instead of snapping.
    private var fullScreenButton: some View {
        Button {
            withAnimation(.easeOut(duration: 0.2)) {
                isFullScreen.toggle()
            }
        } label: {
            Label(
                isFullScreen ? "Exit full screen" : "Full screen",
                systemImage: isFullScreen
                    ? "arrow.down.right.and.arrow.up.left"
                    : "arrow.up.left.and.arrow.down.right"
            )
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

    @ViewBuilder
    private var exposureBadge: some View {
        if let info = viewModel.exposureInfo,
           viewModel.sourceImage != nil {
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

    /// Navigation-bar principal item: the Explicit badge (when the
    /// classifier flagged the image) followed by the file name. Replaces
    /// the previous top-of-image overlay so the flag stays visible
    /// during press-and-hold compare and doesn't clutter the photo.
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
    /// press-and-hold "compare with original" gesture so the original
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

    /// Floating per-subject warning badge placed above each body whose
    /// NudeNet level is `.covered` or higher. Colour matches the
    /// counter's legend — yellow / orange / red for covered / partial /
    /// nude. Independent of the Labels overlay and the per-subject
    /// gate: this is a persistent per-person summary so the viewer can
    /// see at a glance which subjects are flagged and how severely.
    @ViewBuilder
    private func nudeSubjectHeadBadges(in size: CGSize) -> some View {
        if !viewModel.overlayHidden,
           let extent = viewModel.sourceImage?.extent,
           extent.width > 0, extent.height > 0,
           viewModel.nudityLevels.count == viewModel.bodyRectangles.count {
            ForEach(Array(viewModel.bodyRectangles.enumerated()), id: \.offset) { index, body in
                let level = viewModel.nudityLevels[index]
                let gender = index < viewModel.nudityGenders.count
                    ? viewModel.nudityGenders[index]
                    : .unknown
                let emotion = emotionForBody(body)
                // Badge renders when *either* the nudity level crosses
                // covered OR the face classifier returned an emotion —
                // so clothed group photos still get per-face emotion
                // glyphs without us forcing a "safe" shield alongside.
                if level >= .covered || emotion != nil {
                    let rect = viewRect(for: body, source: extent, in: size)
                    SubjectHeadBadge(level: level, gender: gender, emotion: emotion)
                        .position(x: rect.midX, y: max(rect.minY - 18, 20))
                        .allowsHitTesting(false)
                }
            }
        }
    }

    /// Find the face rect whose center lies inside `body` and return
    /// its FER+ emotion if the classifier met its confidence floor.
    /// Returns nil when the emotion model isn't installed, no face
    /// matches the body, or the top prediction was filtered out.
    private func emotionForBody(_ body: CGRect) -> EmotionLabel? {
        guard !viewModel.faceEmotions.isEmpty else { return nil }
        for (i, face) in viewModel.faceRectangles.enumerated()
            where i < viewModel.faceEmotions.count {
            let center = CGPoint(x: face.midX, y: face.midY)
            if body.contains(center) {
                return viewModel.faceEmotions[i]?.label
            }
        }
        return nil
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

    /// Pleasure / Arousal / Dominance readout for the highest-
    /// confidence face's emotion. Derived from FER+'s softmax via
    /// Mehrabian's published anchor points — no separate regressor
    /// runs. Hides when no face met the confidence floor or press-
    /// and-hold compare is active.
    @ViewBuilder
    private var moodBadge: some View {
        let top = viewModel.faceEmotions
            .compactMap { $0 }
            .max(by: { $0.confidence < $1.confidence })
        if let top,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            HStack(spacing: 6) {
                Image(systemName: "heart.text.square")
                Text("P\(Self.signedPAD(top.pad.pleasure)) A\(Self.signedPAD(top.pad.arousal)) D\(Self.signedPAD(top.pad.dominance))")
                    .font(.caption2.monospacedDigit())
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .liquidBadgeBackground(in: Capsule())
        }
    }

    /// Format a PAD axis value as "+0.81" / "-0.63" / "0.00" —
    /// consistent width regardless of sign, two decimals.
    private static func signedPAD(_ v: Float) -> String {
        let clamped = max(-1, min(1, v))
        return String(format: "%+.2f", clamped)
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

    var body: some View {
        HStack(spacing: 4) {
            if let emotion {
                // Emoji outside the coloured foreground so it keeps
                // its native emoji colour rather than getting tinted.
                Text(emotion.emoji)
                    .font(.title3)
            }
            if level >= .covered {
                Image(systemName: "exclamationmark.shield.fill")
                    .font(.title3)
                    .foregroundStyle(tint)
                if let glyph = gender.glyph {
                    // Unicode Mars/Venus glyphs — rendered via Text
                    // instead of Image(systemName:) because they
                    // aren't SF Symbols.
                    Text(glyph)
                        .font(.title3.weight(.bold))
                        .foregroundStyle(tint)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .liquidBadgeBackground(tint: Color.black.opacity(0.45), in: Capsule())
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
