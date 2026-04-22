import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = FocusViewModel()

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Focus Check")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        ImageImporter { url in
                            viewModel.load(url: url)
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
                    MetalView(viewModel: viewModel)
                        .ignoresSafeArea(edges: .horizontal)
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
            viewModel.load(url: url)
            return true
        }
    }
}
