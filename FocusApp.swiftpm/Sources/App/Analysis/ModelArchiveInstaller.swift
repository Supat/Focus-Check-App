import Foundation
import ZIPFoundation

/// Descriptor for a Core ML model archive hosted as a GitHub release asset.
/// Identifies both the expected installed-directory name and the ZIP URL to
/// fetch — the pair together is enough to install, query, or uninstall a
/// model without caring about which model it is.
struct ModelArchive: Sendable {
    /// How the installer interprets the extracted ZIP contents.
    enum Kind: Sendable {
        /// ZIP contains a single `.mlmodelc` directory at the root (or
        /// one level deep). Only that directory is moved to the install
        /// location — extras are discarded. Default for single-model
        /// archives (depth, NSFW, NudeNet).
        case mlmodelc
        /// ZIP contains a whole directory tree — an `.mlmodelc` plus
        /// sibling files (e.g. a prompt-embedding JSON for CLIP). The
        /// entire unpacked tree is moved to the install location so
        /// callers can reference any member by name.
        case bundle
    }

    let directoryName: String
    let sourceURL: URL
    let kind: Kind

    init(directoryName: String, sourceURL: URL, kind: Kind = .mlmodelc) {
        self.directoryName = directoryName
        self.sourceURL = sourceURL
        self.kind = kind
    }

    /// Depth Anything v2 Small (F16) — Apple's Core ML release. Fetched
    /// on demand so the 50 MB `.mlmodelc` doesn't inflate the git repo.
    /// Maintainer publishes `<directoryName>.zip` at the tag below.
    static let depthAnything = ModelArchive(
        directoryName: "DepthAnythingV2SmallF16.mlmodelc",
        sourceURL: URL(string:
            "https://github.com/Supat/Focus-Check-App/releases/download/depth-model-v1/DepthAnythingV2SmallF16.mlmodelc.zip"
        )!
    )

    /// lovoo/NSFWDetector — CreateML-trained binary SFW/NSFW classifier.
    /// Acts as the sensitive-content fallback when Apple's SCA framework
    /// is unavailable (Playgrounds / unsigned builds).
    static let nsfw = ModelArchive(
        directoryName: "NSFW.mlmodelc",
        sourceURL: URL(string:
            "https://github.com/Supat/Focus-Check-App/releases/download/nsfw-model-v1/NSFW.mlmodelc.zip"
        )!
    )

    /// NudeNet v3 detector — YOLO-style object detector with 18 labels for
    /// exposed/covered body parts, enabling per-subject nudity scoring
    /// instead of the whole-image SFW/NSFW binary the `.nsfw` archive
    /// provides. Expected output: Create ML object-detector format
    /// (`coordinates` Nx4 + `confidence` NxC). Maintainer converts the
    /// upstream ONNX via coremltools + compiles with `xcrun coremlcompiler`.
    static let nudenet = ModelArchive(
        directoryName: "NudeNet.mlmodelc",
        sourceURL: URL(string:
            "https://github.com/Supat/Focus-Check-App/releases/download/nudenet-model-v1/NudeNet.mlmodelc.zip"
        )!
    )

    /// CLIP image encoder + pre-computed text-prompt embeddings for
    /// context-aware sensitive-content scoring. The ZIP at the tag
    /// below is expected to contain two siblings:
    ///   - `CLIPImageEncoder.mlmodelc/` — standard Core ML image encoder
    ///     (OpenAI CLIP ViT-B/32 works; outputs a 512-d image embedding).
    ///   - `clip-prompts.json` — array of `{prompt: String, embedding:
    ///     [Float]}` records, embeddings normalized and produced by the
    ///     *matching* text encoder at conversion time.
    /// The installer uses `kind: .bundle` to keep both files together.
    ///
    /// **Version**: `CLIP-v4` — bump the directory name + tag together
    /// when the prompt set or encoder variant changes so existing
    /// installs get orphaned and the Download button reappears for a
    /// clean re-pull. Old `CLIP-v3/` / `CLIP-v2/` / `CLIP/` stay on
    /// disk until the user offloads the app.
    static let clip = ModelArchive(
        directoryName: "CLIP-v4",
        sourceURL: URL(string:
            "https://github.com/Supat/Focus-Check-App/releases/download/clip-model-v4/CLIP.zip"
        )!,
        kind: .bundle
    )

    /// Persistent install path: `Application Support/<directoryName>`.
    /// Application Support is user-data, not purged on low-disk like Caches.
    func installedURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport.appendingPathComponent(directoryName, isDirectory: true)
    }

    /// True when the unpacked install directory exists on disk.
    func isInstalled() -> Bool {
        guard let url = try? installedURL() else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }
}

/// Downloads + unzips + atomic-installs a `ModelArchive`. One actor covers
/// both depth and NSFW models — the earlier two-file, near-duplicate
/// implementation was collapsed into this.
actor ModelArchiveInstaller {
    let archive: ModelArchive

    init(_ archive: ModelArchive) {
        self.archive = archive
    }

    /// Download + unzip + atomic install. Progress (0...1) reports via the
    /// callback, which may be invoked on any thread — the caller dispatches
    /// to the UI layer.
    func install(progress: @Sendable @escaping (Double) -> Void) async throws {
        progress(0)

        let tempZIP = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).zip")
        try await download(from: archive.sourceURL, to: tempZIP, progress: progress)
        defer { try? FileManager.default.removeItem(at: tempZIP) }

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.unzipItem(at: tempZIP, to: scratch)

        let source: URL
        switch archive.kind {
        case .mlmodelc:
            guard let mlmodelc = Self.locateMLModelC(in: scratch) else {
                throw AnalysisError.modelLoadFailed("ZIP does not contain a .mlmodelc directory.")
            }
            source = mlmodelc
        case .bundle:
            // Bundle archives may either put the contents at the root
            // (scratch/CLIPImageEncoder.mlmodelc + scratch/prompts.json)
            // or one level down (scratch/CLIP/...). Collapse the single-
            // top-dir case so the caller's install path always contains
            // the files directly instead of an extra wrapping layer.
            source = Self.collapseSingleRootDirectory(in: scratch) ?? scratch
        }

        let dest = try archive.installedURL()
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: source, to: dest)

        progress(1.0)
    }

    /// If `root` contains exactly one top-level directory that isn't
    /// itself a `.mlmodelc`, return that inner directory so the
    /// caller moves its contents to the install path without an
    /// extra wrapping layer. Filters out dotfiles and `__MACOSX`
    /// (zip metadata folder Finder / some ditto invocations add),
    /// so a single-real-dir archive still unwraps correctly. Uses
    /// `.isDirectoryKey` instead of `hasDirectoryPath` because the
    /// latter depends on whether the trailing slash made it through
    /// `contentsOfDirectory`.
    private static func collapseSingleRootDirectory(in root: URL) -> URL? {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]
        )) ?? []
        let ignored: Set<String> = ["__MACOSX"]
        let visible = entries.filter {
            let name = $0.lastPathComponent
            return !name.hasPrefix(".") && !ignored.contains(name)
        }
        guard visible.count == 1,
              let only = visible.first,
              only.pathExtension != "mlmodelc"
        else { return nil }
        let isDir = (try? only.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
        return isDir ? only : nil
    }

    /// Remove the installed model. Useful for a "re-download" UX or tests.
    func uninstall() throws {
        let url = try archive.installedURL()
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Private

    private func download(from source: URL, to destination: URL,
                          progress: @Sendable @escaping (Double) -> Void) async throws {
        let delegate = ProgressObservingDelegate(progress: progress)
        let (tempURL, response) = try await URLSession.shared.download(
            from: source, delegate: delegate
        )

        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw AnalysisError.modelLoadFailed("Download failed (HTTP \(code)).")
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    /// The zip may contain the `.mlmodelc` at the root or nested one level
    /// deep — handle both forms without a manifest file.
    private static func locateMLModelC(in root: URL) -> URL? {
        let rootContents = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]
        )) ?? []
        for entry in rootContents {
            if entry.pathExtension == "mlmodelc" { return entry }
            if entry.hasDirectoryPath {
                let sub = (try? FileManager.default.contentsOfDirectory(
                    at: entry, includingPropertiesForKeys: nil
                )) ?? []
                if let m = sub.first(where: { $0.pathExtension == "mlmodelc" }) {
                    return m
                }
            }
        }
        return nil
    }
}

/// Observes `URLSessionTask.progress.fractionCompleted` via KVO. `didCreateTask`
/// (iOS 16+) gives us the task reference without juggling continuations.
/// `@unchecked Sendable` is safe: `callback` is @Sendable; `observation` is
/// only written from `didCreateTask` and released by deinit.
private final class ProgressObservingDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let callback: @Sendable (Double) -> Void
    private var observation: NSKeyValueObservation?

    init(progress: @escaping @Sendable (Double) -> Void) {
        self.callback = progress
        super.init()
    }

    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        observation = task.progress.observe(\.fractionCompleted, options: [.new]) { [callback] prog, _ in
            callback(prog.fractionCompleted)
        }
    }
}
