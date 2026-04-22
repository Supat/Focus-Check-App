import SwiftUI

struct OverlayControls: View {
    @ObservedObject var viewModel: FocusViewModel

    // Local mirror of the threshold as a string so the TextField is edited freely;
    // we commit (parse + clamp) on submit / focus loss. Without this the numeric
    // `value:format:` binding only committed on unfocus and silently dropped input
    // from hardware keyboards.
    @State private var thresholdText: String = ""
    @FocusState private var thresholdFocused: Bool

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
                TextField("", text: $thresholdText)
                    .focused($thresholdFocused)
                    .multilineTextAlignment(.trailing)
                    .font(.caption.monospacedDigit())
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 64)
                    .submitLabel(.done)
                    .onSubmit { commitThresholdText() }
                    .onChange(of: thresholdFocused) { _, isFocused in
                        if !isFocused { commitThresholdText() }
                    }
                    .onChange(of: viewModel.threshold) { _, new in
                        // Keep the field text in sync when the slider moves — but only
                        // while the field isn't actively being edited.
                        if !thresholdFocused {
                            thresholdText = formatted(new)
                        }
                    }
                    .onAppear { thresholdText = formatted(viewModel.threshold) }
            }

            HStack {
                Picker("Mode", selection: $viewModel.mode) {
                    ForEach(AnalysisMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                // Error style requires both signals — lock the picker to Hybrid so
                // the user can't accidentally strand the overlay without depth data.
                .disabled(viewModel.sourceImage == nil || viewModel.style.requiresDepth)
                .onChange(of: viewModel.mode) { _, _ in
                    viewModel.reanalyze()
                }
            }

            depthInstallRow
            mosaicToggleRow
            nsfwInstallRow
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

    private func commitThresholdText() {
        if let parsed = Float(thresholdText.trimmingCharacters(in: .whitespaces)) {
            viewModel.threshold = min(max(parsed, 0), 1)
        }
        // Normalize display regardless of whether parse succeeded.
        thresholdText = formatted(viewModel.threshold)
    }

    private func formatted(_ value: Float) -> String {
        String(format: "%.2f", value)
    }

    @ViewBuilder
    private var mosaicToggleRow: some View {
        HStack {
            Spacer()
            Label("Mosaic sensitive content", systemImage: "eye.slash")
                .font(.caption)
            Toggle("", isOn: $viewModel.mosaicEnabled)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                .disabled(!viewModel.sensitiveContentAvailability.isReady)
        }
    }

    @ViewBuilder
    private var depthInstallRow: some View {
        switch viewModel.depthInstall {
        case .installed:
            // Depth mode is ready — no row needed.
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Depth model (~50 MB) not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadDepthModel() }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
            }

        case .downloading(let progress):
            HStack(spacing: 8) {
                ProgressView(value: progress)
                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 44, alignment: .trailing)
            }

        case .failed(let message):
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer()
                Button("Retry") { viewModel.downloadDepthModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// NSFW fallback install row. Only visible when SCA is not .simpleInterventions
    /// or .descriptiveInterventions — there's no point prompting for the fallback
    /// model if the primary classifier is already available.
    @ViewBuilder
    private var nsfwInstallRow: some View {
        let sca = viewModel.sensitiveContentAvailability
        let scaActive = sca == .simpleInterventions || sca == .descriptiveInterventions
        if !scaActive {
            switch viewModel.nsfwInstall {
            case .installed:
                EmptyView()

            case .notInstalled:
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle")
                        .foregroundStyle(.secondary)
                    Text("NSFW fallback model not installed.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Download") { viewModel.downloadNSFWModel() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                }

            case .downloading(let progress):
                HStack(spacing: 8) {
                    ProgressView(value: progress)
                    Text("\(Int(progress * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 44, alignment: .trailing)
                }

            case .failed(let message):
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                    Text(message)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    Spacer()
                    Button("Retry") { viewModel.downloadNSFWModel() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }
        }
    }
}
