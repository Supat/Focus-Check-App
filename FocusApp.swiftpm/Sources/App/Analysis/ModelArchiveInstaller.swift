import Foundation
import ZIPFoundation

/// Descriptor for a Core ML model archive hosted as a GitHub release asset.
/// Identifies both the expected installed-directory name and the ZIP URL to
/// fetch — the pair together is enough to install, query, or uninstall a
/// model without caring about which model it is.
struct ModelArchive: Sendable {
    let directoryName: String
    let sourceURL: URL

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

    /// True when the unpacked `.mlmodelc` directory exists on disk.
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

        guard let extracted = Self.locateMLModelC(in: scratch) else {
            throw AnalysisError.modelLoadFailed("ZIP does not contain a .mlmodelc directory.")
        }

        let dest = try archive.installedURL()
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: extracted, to: dest)

        progress(1.0)
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
