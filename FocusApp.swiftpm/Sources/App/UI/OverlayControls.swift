import SwiftUI

struct OverlayControls: View {
    @ObservedObject var viewModel: FocusViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Picker("Style", selection: $viewModel.style) {
                    ForEach(OverlayStyle.allCases) { style in
                        Label(style.rawValue, systemImage: style.systemImage).tag(style)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                ColorPicker("Overlay color",
                            selection: $viewModel.overlayColor,
                            supportsOpacity: false)
                    .labelsHidden()
                    .frame(width: 44)
            }

            HStack(spacing: 8) {
                Image(systemName: "slider.horizontal.below.sun.max")
                    .foregroundStyle(.secondary)
                Slider(value: $viewModel.threshold, in: 0...1)
                Text(String(format: "%.2f", viewModel.threshold))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 40, alignment: .trailing)
            }

            HStack {
                Picker("Mode", selection: $viewModel.mode) {
                    ForEach(AnalysisMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(viewModel.sourceImage == nil)
                .onChange(of: viewModel.mode) { _, _ in
                    viewModel.reanalyze()
                }

                if !viewModel.depthAvailable {
                    Label("No depth model", systemImage: "exclamationmark.triangle")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .labelStyle(.titleAndIcon)
                }
            }
        }
        #if os(iOS)
        .onPencilSqueeze { phase in
            // Apple Pencil Pro squeeze: cycle between peaking and heatmap on release.
            if case .ended = phase {
                viewModel.style = viewModel.style == .peaking ? .heatmap : .peaking
            }
        }
        #endif
    }
}
