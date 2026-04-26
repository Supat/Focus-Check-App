import SwiftUI

/// Sheet listing every optional Core ML tier the app can install,
/// with a single per-row Install / Uninstall control plus a live
/// progress bar while a download is mid-flight.
///
/// The list itself is computed by `FocusViewModel.modelEntries`,
/// which pairs each `ModelArchive` with the install / uninstall
/// triggers and a key-path to the `@Published` install state. This
/// view never reaches into the analyzer directly — all routing
/// goes through the view model so the existing per-row download
/// helpers (`downloadDepthModel()` / etc.) stay the single source
/// of truth for install lifecycle.
struct ModelManagerView: View {
    @ObservedObject var viewModel: FocusViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(viewModel.modelEntries) { entry in
                        ModelManagerRow(viewModel: viewModel, entry: entry)
                    }
                } footer: {
                    // Heads-up: the in-memory MLModel persists for
                    // the rest of the session even after Uninstall
                    // — Core ML reads the file once at construction.
                    // Reinstalling does *not* hot-swap the weights;
                    // the next launch picks up the fresh file.
                    Text("Uninstalling removes the model file from disk. "
                         + "Models already loaded in memory keep working "
                         + "until the app restarts.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Model Manager")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

/// One row in the Model Manager list. Renders the archive's
/// display name + version + on-disk install status, then a
/// trailing control whose shape switches by state:
///
///   .notInstalled   →  "Install" button
///   .downloading    →  inline ProgressView
///   .installed      →  "Uninstall" button (destructive)
///   .failed         →  "Retry" button + error footnote
private struct ModelManagerRow: View {
    @ObservedObject var viewModel: FocusViewModel
    let entry: ModelEntry

    var body: some View {
        let state = viewModel[keyPath: entry.state]
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.archive.displayName)
                        .font(.body)
                    Text(entry.archive.version)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                trailingControl(for: state)
            }
            if case .failed(let message) = state {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            if case .downloading(let progress) = state {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func trailingControl(for state: DepthInstallState) -> some View {
        switch state {
        case .notInstalled:
            Button("Install", action: entry.install)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        case .downloading(let progress):
            Text("\(Int(progress * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .trailing)
        case .installed:
            Button(role: .destructive,
                   action: entry.uninstall) {
                Text("Uninstall")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        case .failed:
            Button("Retry", action: entry.install)
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }
}
