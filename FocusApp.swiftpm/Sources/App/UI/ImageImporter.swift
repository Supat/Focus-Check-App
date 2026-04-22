import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Toolbar control offering both PhotosPicker (library) and fileImporter (Files / disk) entry points.
struct ImageImporter: View {
    /// Passes the local tmp URL and a human-readable display name (original
    /// filename from `NSItemProvider.suggestedName` for library picks; the
    /// original `url.lastPathComponent` for file imports).
    let onPick: (URL, String) -> Void

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
        .sheet(isPresented: $showingPhotosPicker) {
            PHPickerSheet(isPresented: $showingPhotosPicker) { url, name in
                isLoading = false
                onPick(url, name)
            } onStart: {
                isLoading = true
            }
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.image, .rawImage],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first { deliver(url: url, fromScoped: true) }
            case .failure:
                break
            }
        }
    }

    private func deliver(url: URL, fromScoped: Bool) {
        // Security-scoped URLs from .fileImporter need explicit start/stop. We copy into a
        // tmp file so the downstream analyzer doesn't have to hold the scope open.
        let didStart = fromScoped ? url.startAccessingSecurityScopedResource() : false
        defer { if didStart { url.stopAccessingSecurityScopedResource() } }
        do {
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(url.pathExtension)
            try FileManager.default.copyItem(at: url, to: tmp)
            onPick(tmp, url.lastPathComponent)
        } catch {
            // swallow
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

            result.itemProvider.loadFileRepresentation(forTypeIdentifier: uti) { tempURL, _ in
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
