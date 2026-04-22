import SwiftUI
import PhotosUI
import Photos
import UniformTypeIdentifiers

/// Toolbar control offering both PhotosPicker (library) and fileImporter (Files / disk) entry points.
struct ImageImporter: View {
    /// Passes the local tmp URL and a human-readable display name (original filename
    /// for file imports; a `Photo.<ext>` fallback for library picks since PhotosPicker
    /// doesn't expose the original filename without a PhotoKit permission flow).
    let onPick: (URL, String) -> Void

    @State private var photoItem: PhotosPickerItem?
    @State private var showingPhotosPicker = false
    @State private var showingFileImporter = false
    @State private var isLoading = false

    var body: some View {
        Menu {
            // `PhotosPicker` nested inside a `Menu` is a known SwiftUI bug:
            // the menu dismisses on tap but the picker sheet never presents.
            // Flip a flag here and present via `.photosPicker(isPresented:...)` below.
            //
            // Also: request photo-library access *before* presenting the picker.
            // PhotosPickerItem.itemIdentifier is the PHAsset.localIdentifier only
            // when the app has Photos access at pick-time; otherwise it's an
            // opaque string that PHAsset.fetchAssets can't resolve.
            Button {
                presentPhotosPickerAfterAuth()
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
        .photosPicker(
            isPresented: $showingPhotosPicker,
            selection: $photoItem,
            matching: .images,
            preferredItemEncoding: .current
        )
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
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            loadFromPicker(item: item)
        }
    }

    private func loadFromPicker(item: PhotosPickerItem) {
        isLoading = true
        Task {
            defer {
                Task { @MainActor in isLoading = false }
            }
            // Prefer the ORIGINAL representation: ProRAW / HEIC / RAW rather than a rasterized JPEG.
            let preferredUTIs = [
                "public.camera-raw-image",
                "com.adobe.raw-image",
                UTType.heic.identifier,
                UTType.image.identifier
            ]
            let available = item.supportedContentTypes.map(\.identifier)
            let uti = preferredUTIs.first(where: { available.contains($0) })
                ?? available.first
                ?? UTType.image.identifier

            do {
                guard let data = try await item.loadTransferable(type: Data.self) else { return }
                let ext = UTType(uti)?.preferredFilenameExtension ?? "img"
                let tmp = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension(ext)
                try data.write(to: tmp, options: .atomic)
                let displayName = await resolvedFilename(for: item, fallback: "Photo.\(ext)")
                await MainActor.run { onPick(tmp, displayName) }
            } catch {
                // Silent failure — the view model surfaces analysis errors.
            }
        }
    }

    /// Ensure photo-library access is granted before presenting the picker. If
    /// already authorized, present immediately; if undetermined, prompt and wait
    /// for the user's answer. If denied, present anyway — the picker itself still
    /// works, we just fall back to a generic display name.
    private func presentPhotosPickerAfterAuth() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .authorized, .limited:
            showingPhotosPicker = true
        case .notDetermined:
            Task { @MainActor in
                _ = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
                showingPhotosPicker = true
            }
        default:
            showingPhotosPicker = true
        }
    }

    /// Look up the Photo asset's original filename via PhotoKit. The caller must
    /// already have requested photo-library access — otherwise `itemIdentifier`
    /// isn't a PHAsset.localIdentifier and the fetch returns empty. Falls back
    /// to the generic name whenever the chain breaks (no identifier, no access,
    /// asset not in limited-access allowlist, missing resources).
    private func resolvedFilename(for item: PhotosPickerItem, fallback: String) async -> String {
        guard let id = item.itemIdentifier else { return fallback }

        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .authorized, .limited: break
        default: return fallback
        }

        let fetch = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil)
        guard let asset = fetch.firstObject else { return fallback }

        let resources = PHAssetResource.assetResources(for: asset)
        return resources.first?.originalFilename ?? fallback
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
