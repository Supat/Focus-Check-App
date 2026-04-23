import SwiftUI

private struct ExportedImage: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

struct ContentView: View {
    @StateObject private var viewModel = FocusViewModel()
    @State private var exportedImage: ExportedImage?
    @State private var isExporting = false

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
                            exportButton
                        }
                    }
                    ToolbarItem(placement: .primaryAction) {
                        ImageImporter(
                            onPick: { url, name in viewModel.load(url: url, name: name) },
                            onError: { message in viewModel.errorMessage = message }
                        )
                    }
                }
                .sheet(item: $exportedImage) { item in
                    ShareSheet(url: item.url)
                }
                .overlay(alignment: .top) {
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.callout)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.regularMaterial, in: Capsule())
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
                    }
                }
                if viewModel.isAnalyzing {
                    ProgressView("Analyzing…")
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .top) {
                sensitiveContentBadge
                    .padding(.top, 8)
            }
            .overlay(alignment: .bottomLeading) {
                HStack(spacing: 8) {
                    exposureBadge
                    motionBlurBadge
                    nudeSubjectsBadge
                }
                .padding([.leading, .bottom], 12)
            }

            Divider()

            OverlayControls(viewModel: viewModel)
                .padding()
                .background(.bar)
        }
    }

    /// Triggers an async composite + PNG encode on the view model, then
    /// presents the system share sheet so the user can save to Files,
    /// Photos, or send via any registered share target.
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
                .background(.regularMaterial, in: Capsule())
            }
        }
    }

    /// Shown whenever the classifier flagged the image, independent of the
    /// mosaic toggle — lets the user know the image was flagged even when
    /// they've chosen to view it uncovered. Uses the top class label from
    /// the classifier (e.g. "Nudity" from SCA, "NSFW" from the fallback).
    /// Red when the NSFW classifier's confidence exceeds 0.6 — a stronger
    /// visual signal for high-confidence matches. Orange for borderline
    /// cases or SCA results (which don't expose a numeric confidence).
    @ViewBuilder
    private var sensitiveContentBadge: some View {
        if viewModel.isSensitive == true,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            let isHighConfidence = (viewModel.sensitiveConfidence ?? 0) > 0.6
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.shield.fill")
                Text(viewModel.sensitiveLabel ?? "Sensitive")
                    .font(.caption)
            }
            .foregroundStyle(isHighConfidence ? Color.red : Color.orange)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            // .ultraThinMaterial still renders mostly opaque on this layer;
            // swap for an explicit 40%-black fill so the image clearly shows
            // through the capsule while the coloured text stays legible.
            .background(Color.black.opacity(0.4), in: Capsule())
        }
    }

    /// Per-subject count from NudeNet, shown alongside the exposure and
    /// motion-blur capsules. Hides itself when the count is zero (no
    /// subject crossed the gate) so it doesn't clutter safe photos.
    @ViewBuilder
    private var nudeSubjectsBadge: some View {
        let flagged = viewModel.nudityLevels
            .filter { $0 >= viewModel.nudityGate }.count
        if flagged > 0,
           viewModel.sourceImage != nil,
           !viewModel.overlayHidden {
            HStack(spacing: 6) {
                Image(systemName: "person.2.fill")
                Text("\(flagged)/\(viewModel.nudityLevels.count)")
                    .font(.caption.monospacedDigit())
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())
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
            .background(.regularMaterial, in: Capsule())
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
