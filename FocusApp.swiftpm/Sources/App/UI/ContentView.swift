import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = FocusViewModel()

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
                    }
                    ToolbarItem(placement: .primaryAction) {
                        ImageImporter { url, name in
                            viewModel.load(url: url, name: name)
                        }
                    }
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

            Divider()

            OverlayControls(viewModel: viewModel)
                .padding()
                .background(.bar)
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
