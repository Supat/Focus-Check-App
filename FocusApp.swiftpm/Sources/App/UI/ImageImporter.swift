import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Toolbar control offering both PhotosPicker (library) and fileImporter (Files / disk) entry points.
struct ImageImporter: View {
    /// Passes the URL, a human-readable display name, and a flag
    /// indicating whether the URL is a still-live security-scoped
    /// resource (true for video imports from `.fileImporter` that
    /// stream directly without a temp copy). When true, the
    /// downstream consumer is responsible for releasing the scope
    /// when the resource is no longer needed.
    let onPick: (URL, String, Bool) -> Void
    /// Switch the active source to the live back-camera feed.
    /// Doesn't go through `onPick` because the camera path doesn't
    /// produce a URL — the viewModel handles authorization and
    /// session lifetime directly.
    let onPickCamera: () -> Void
    /// Surface failures back to the caller (view model) so the user sees
    /// why a pick didn't land instead of the picker silently doing nothing.
    let onError: (String) -> Void

    @State private var showingPhotosPicker = false
    @State private var showingFileImporter = false
    @State private var showingImportOptions = false
    @State private var isLoading = false

    var body: some View {
        // Plain Button + confirmationDialog instead of Menu — Menu's
        // internal state machine has been observed to drop tap
        // events when its parent rebuilds during video playback (the
        // viewModel's @Published `sourceImage` updates rebuild
        // ContentView, the toolbar gets a fresh ImageImporter
        // closure capture, and the Menu's open/closed state can
        // race with that). confirmationDialog is system-presented
        // and survives parent rebuilds without re-rendering the
        // Menu chrome each time.
        Button {
            showingImportOptions = true
        } label: {
            if isLoading {
                ProgressView().controlSize(.small)
            } else {
                Label("Import", systemImage: "square.and.arrow.down")
            }
        }
        .confirmationDialog(
            "Import", isPresented: $showingImportOptions,
            titleVisibility: .hidden
        ) {
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

            Button {
                onPickCamera()
            } label: {
                Label("Live Camera", systemImage: "camera")
            }

            Button("Cancel", role: .cancel) { }
        }
        // Pick up the keyboard shortcuts (⌘O / ⌘⇧O) that the
        // app-level `.commands` block in FocusApp.swift posts.
        // Buttons inside `Menu { }` and hidden Buttons in the
        // view tree both fail to surface in iPadOS 26's menu bar;
        // routing via NotificationCenter from a Scene-level
        // CommandGroup is the path that does.
        .onReceive(NotificationCenter.default.publisher(
            for: .openPhotoLibrary
        )) { _ in showingPhotosPicker = true }
        .onReceive(NotificationCenter.default.publisher(
            for: .openFileImporter
        )) { _ in showingFileImporter = true }
        .sheet(isPresented: $showingPhotosPicker) {
            PHPickerSheet(
                isPresented: $showingPhotosPicker,
                onPick: { url, name in
                    // PHPicker temp files are NSItemProvider-managed and
                    // already in our sandbox — never security-scoped.
                    isLoading = false
                    onPick(url, name, false)
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
            allowedContentTypes: [.image, .rawImage, .movie, .audio],
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

        // Video / audio files: skip the copy. AVAsset can stream
        // directly from the security-scoped URL — keeping the scope
        // open beats the disk-double cost of pulling a multi-GB
        // file through `Data(contentsOf:)` + `data.write(to:)`. Hand
        // the scope ownership to the downstream consumer so it
        // survives the rest of the playback session.
        if let type = UTType(filenameExtension: url.pathExtension),
           type.conforms(to: .movie)
            || type.conforms(to: .video)
            || type.conforms(to: .audio) {
            let didStart = fromScoped && url.startAccessingSecurityScopedResource()
            print("[Importer] video startAccessing=\(didStart) → onPick (no copy)")
            onPick(url, url.lastPathComponent, didStart)
            return
        }

        // Image path: read + write a temp copy so we don't depend on
        // the original URL or the security scope outliving the picker
        // sheet. Cheap on photos (≤ 100 MB), fine even for RAW.
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
            onPick(tmp, url.lastPathComponent, false)
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
        // Allow movies alongside images so the same picker covers
        // both pipelines. Detection happens downstream — `FocusViewModel.load`
        // forks based on UTType.
        config.filter = .any(of: [.images, .videos])
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
            // QuickTime / MPEG-4 first so videos don't fall through to a
            // generic image UTI that PHPicker can't satisfy.
            let preferredUTIs = [
                "public.camera-raw-image",
                "com.adobe.raw-image",
                UTType.heic.identifier,
                UTType.quickTimeMovie.identifier,
                UTType.mpeg4Movie.identifier,
                UTType.movie.identifier,
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
