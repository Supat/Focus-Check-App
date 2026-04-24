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
            // Slider + numeric field drive overlay compositing only; None mode
            // renders the original image untouched so there's nothing to scrub.
            .disabled(viewModel.style.isOff)

            HStack {
                Picker("Mode", selection: $viewModel.mode) {
                    ForEach(AnalysisMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                // Error style requires both signals — lock the picker to Hybrid so
                // the user can't accidentally strand the overlay without depth data.
                // None mode has no overlay, so analysis mode choice is moot.
                .disabled(
                    viewModel.sourceImage == nil ||
                    viewModel.style.requiresDepth ||
                    viewModel.style.isOff
                )
                .onChange(of: viewModel.mode) { _, _ in
                    viewModel.reanalyze()
                }
            }

            depthInstallRow
            mosaicToggleRow
            nsfwInstallRow
            nudenetInstallRow
            clipInstallRow
            emotionInstallRow
            painInstallRow
            ageInstallRow
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
        // NudeNet absent: one row, mosaic cluster only. Installed: try one
        // row first, fall back to stacked rows when the viewport is too
        // narrow (portrait iPad, Stage Manager, split view). ViewThatFits
        // picks the first variant whose intrinsic width fits the available
        // horizontal space.
        if viewModel.nudenetInstall == .installed {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    Spacer()
                    perSubjectCluster
                    mosaicCluster
                }
                VStack(alignment: .trailing, spacing: 8) {
                    HStack(spacing: 8) {
                        Spacer()
                        mosaicCluster
                    }
                    HStack(spacing: 8) {
                        Spacer()
                        perSubjectCluster
                    }
                }
            }
        } else {
            HStack(spacing: 8) {
                Spacer()
                mosaicCluster
            }
        }
    }

    /// Per-subject NudeNet gate. Caller is responsible for only rendering
    /// this when NudeNet is installed — the picker still needs a valid
    /// `nudityGate` binding either way, but the label is meaningless
    /// without detections to gate.
    @ViewBuilder
    private var perSubjectCluster: some View {
        // One toggle controls both the PAD bars and the Pain bar
        // sitting beside them. They're a single visual row and a
        // single conceptual "per-subject meter".
        Label("PAD meter", systemImage: "chart.bar.xaxis")
            .labelStyle(.iconOnly)
            .font(.caption)
        Toggle("", isOn: $viewModel.showPADMeter)
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
        Label("Per subject", systemImage: "person.crop.square.filled.and.at.rectangle")
            .labelStyle(.iconOnly)
            .font(.caption)
        Picker("Per-subject gate", selection: $viewModel.nudityGate) {
            Text("All").tag(NudityLevel.none)
            Text("Covered").tag(NudityLevel.covered)
            Text("Partial").tag(NudityLevel.partial)
            Text("Nude").tag(NudityLevel.nude)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .controlSize(.small)
        .frame(width: 280)
        Label("Labels", systemImage: "tag")
            .labelStyle(.iconOnly)
            .font(.caption)
        Toggle("", isOn: $viewModel.showNudityLabels)
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
    }

    /// The mosaic mode picker + enable + force toggles, extracted so the
    /// row-layout switch above can reuse it in both the single-row and
    /// stacked-row variants.
    @ViewBuilder
    private var mosaicCluster: some View {
        // The mode picker is usable whenever at least one of the mosaic paths
        // can fire — either the classifier-gated one (toggle + SCA ready) or
        // Force Censor (which bypasses the classifier entirely).
        let classifierPath =
            viewModel.mosaicEnabled && viewModel.sensitiveContentAvailability.isReady
        let pickerActive = classifierPath || viewModel.forceCensor

        Label("Mosaic", systemImage: "eye.slash")
            .labelStyle(.iconOnly)
            .font(.caption)
        Picker("Mosaic mode", selection: $viewModel.mosaicMode) {
            ForEach(MosaicMode.allCases) { mode in
                Text(mode.rawValue).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .controlSize(.small)
        .frame(width: 528)
        .disabled(!pickerActive)
        Toggle("", isOn: $viewModel.mosaicEnabled)
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
            .disabled(!viewModel.sensitiveContentAvailability.isReady)
        Label("Force", systemImage: "eye.slash.circle.fill")
            .labelStyle(.iconOnly)
            .font(.caption)
        Toggle("", isOn: $viewModel.forceCensor)
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
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

    /// NudeNet per-subject detector install row — parallels `nsfwInstallRow`
    /// but unconditional: per-subject rating is additive to whichever
    /// primary classifier is active.
    @ViewBuilder
    private var nudenetInstallRow: some View {
        switch viewModel.nudenetInstall {
        case .installed:
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Per-subject detector (NudeNet) not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadNudeNetModel() }
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
                Button("Retry") { viewModel.downloadNudeNetModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// CLIP context-scorer install row. Same shape as the other three
    /// install rows. Unconditional — CLIP is additive to every other
    /// signal in the pipeline, whether or not SCA / NSFW / NudeNet are
    /// active.
    @ViewBuilder
    private var clipInstallRow: some View {
        switch viewModel.clipInstall {
        case .installed:
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Context scorer (CLIP) not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadCLIPModel() }
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
                Button("Retry") { viewModel.downloadCLIPModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// FER+ emotion classifier install row. Drawn with the same
    /// download / progress / retry variants as the other optional
    /// models. Unconditional — emotion scoring is additive and
    /// independent of the other tiers.
    @ViewBuilder
    private var emotionInstallRow: some View {
        switch viewModel.emotionInstall {
        case .installed:
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Emotion classifier (FER+) not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadEmotionModel() }
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
                Button("Retry") { viewModel.downloadEmotionModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// OpenGraphAU / pain detector install row. Same download /
    /// progress / retry pattern as the other optional model tiers.
    /// The resulting per-face PSPI score is additive to whatever the
    /// emotion classifier contributes.
    @ViewBuilder
    private var painInstallRow: some View {
        switch viewModel.openGraphAUInstall {
        case .installed:
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Pain detector (OpenGraphAU) not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadOpenGraphAUModel() }
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
                Button("Retry") { viewModel.downloadOpenGraphAUModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// Age estimator install row. Same download / progress / retry
    /// pattern as the other optional model tiers. Output is a
    /// per-face age readout; gender comes from NudeNet's FACE_*
    /// branch (not from this model) — SSR-Net is age-only.
    @ViewBuilder
    private var ageInstallRow: some View {
        switch viewModel.ageInstall {
        case .installed:
            EmptyView()

        case .notInstalled:
            HStack(spacing: 8) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.secondary)
                Text("Age estimator not installed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Download") { viewModel.downloadAgeModel() }
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
                Button("Retry") { viewModel.downloadAgeModel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }
}
