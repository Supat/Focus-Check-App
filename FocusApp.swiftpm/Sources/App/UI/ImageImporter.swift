import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Toolbar control offering both PhotosPicker (library) and fileImporter (Files / disk) entry points.
struct ImageImporter: View {
    /// Passes the local tmp URL and a human-readable display name (original
    /// filename from `NSItemProvider.suggestedName` for library picks; the
    /// original `url.lastPathComponent` for file imports).
    let onPick: (URL, String) -> Void
    /// Surface failures back to the caller (view model) so the user sees
    /// why a pick didn't land instead of the picker silently doing nothing.
    let onError: (String) -> Void

    @State private var showingPhotosPicker = false
    @State private var showingFileImporter = false
    @State private var isLoading = false

    var body: some View {
        Menu {
            Button {
                showingPhotosPicker = true
            } label: {
                Label("Photo Library", systemImage: "photo.stack")
            }

            Button {
                showingFileImporter = true
            } label: {
                Label("Choose File…", systemImage: "folder")
            }
        } label: {
            if isLoading {
                ProgressView().controlSize(.small)
            } else {
                Label("Import", systemImage: "square.and.arrow.down")
            }
        }
        // Hidden shortcut hosts — Buttons inside `Menu { }` register
        // their .keyboardShortcut modifiers only while the menu is
        // open on iOS, which is why the inline approach never
        // fired. Mirror the actions onto invisible Buttons that
        // live in the view tree at all times so iPadOS / Mac pick
        // them up globally and surface them in the ⌘-hold
        // discoverability sheet.
        .background {
            Group {
                Button("Open Photo Library") {
                    showingPhotosPicker = true
                }
                .keyboardShortcut("o", modifiers: .command)
                Button("Open File…") {
                    showingFileImporter = true
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            }
            .opacity(0)
            .frame(width: 0, height: 0)
            .accessibilityHidden(true)
        }
        .sheet(isPresented: $showingPhotosPicker) {
            PHPickerSheet(
                isPresented: $showingPhotosPicker,
                onPick: { url, name in
                    isLoading = false
                    onPick(url, name)
                },
                onStart: { isLoading = true },
                onError: { message in
                    isLoading = false
                    onError(message)
                }
            )
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.image, .rawImage],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { deliver(url: url, fromScoped: true) }
            case .failure(let error):
                onError("Couldn't open file picker: \(error.localizedDescription)")
            }
        }
    }

    private func deliver(url: URL, fromScoped: Bool) {
        print("[Importer] deliver url=\(url.path) scoped=\(fromScoped)")
        let didStart = fromScoped && url.startAccessingSecurityScopedResource()
        defer { if didStart { url.stopAccessingSecurityScopedResource() } }
        print("[Importer] startAccessing=\(didStart) file exists=\(FileManager.default.fileExists(atPath: url.path))")

        do {
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            let ext = url.pathExtension.isEmpty ? "img" : url.pathExtension
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(ext)
            try data.write(to: tmp, options: .atomic)
            print("[Importer] read \(data.count) bytes → wrote \(tmp.lastPathComponent) → onPick")
            onPick(tmp, url.lastPathComponent)
        } catch {
            let message = "Couldn't read “\(url.lastPathComponent)”: \(error.localizedDescription)"
            print("[Importer] FAIL \(message)")
            onError(message)
        }
    }
}

/// Wraps PHPickerViewController directly so we can read `NSItemProvider.suggestedName`
/// (the original camera filename like `IMG_4217.HEIC`), which SwiftUI's `PhotosPicker`
/// doesn't surface. Runs out-of-process, so no photo-library permission is required.
private struct PHPickerSheet: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    let onPick: (URL, String) -> Void
    let onStart: () -> Void
    let onError: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.selectionLimit = 1
        config.filter = .images
        // `.current` preserves ProRAW / RAW / HEIC originals instead of rasterizing to JPEG.
        config.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PHPickerSheet

        init(_ parent: PHPickerSheet) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.isPresented = false
            guard let result = results.first else { return }

            // Prefer original-format UTIs so RAW stays RAW, HEIC stays HEIC.
            let preferredUTIs = [
                "public.camera-raw-image",
                "com.adobe.raw-image",
                UTType.heic.identifier,
                UTType.image.identifier
            ]
            let registered = result.itemProvider.registeredTypeIdentifiers
            let uti = preferredUTIs.first(where: { registered.contains($0) })
                ?? registered.first
                ?? UTType.image.identifier

            parent.onStart()
            let suggested = result.itemProvider.suggestedName
            let onPick = parent.onPick
            let onError = parent.onError

            result.itemProvider.loadFileRepresentation(forTypeIdentifier: uti) { tempURL, error in
                if let error {
                    Task { @MainActor in
                        onError("Couldn't load photo: \(error.localizedDescription)")
                    }
                    return
                }
                guard let tempURL else { return }
                do {
                    // URL.pathExtension is a non-optional String ("" when absent),
                    // so chain via a local fallback rather than `??` with a String literal.
                    let utiExt = UTType(uti)?.preferredFilenameExtension
                    let urlExt = tempURL.pathExtension
                    let ext = utiExt ?? (urlExt.isEmpty ? "img" : urlExt)
                    let tmp = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                        .appendingPathExtension(ext)
                    try FileManager.default.copyItem(at: tempURL, to: tmp)

                    // `suggestedName` typically lacks the extension (e.g. "IMG_4217").
                    let displayName: String
                    if let s = suggested, !s.isEmpty {
                        displayName = s.contains(".") ? s : "\(s).\(ext)"
                    } else {
                        displayName = "Photo.\(ext)"
                    }

                    DispatchQueue.main.async {
                        onPick(tmp, displayName)
                    }
                } catch {
                    // swallow
                }
            }
        }
    }
}
