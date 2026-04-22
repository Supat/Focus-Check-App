import Foundation
import ZIPFoundation

/// Downloads and unpacks the compiled Depth Anything v2 model at runtime.
///
/// The `.mlmodelc` artifact can't live in the git repo (50 MB, directory form) and
/// can't be compiled on iPad (`.mlpackage` → `.mlmodelc` needs `xcrun coremlcompiler`,
/// macOS-only). Instead a maintainer compiles once on a Mac, zips the result, and
/// uploads it as a GitHub release asset. The app fetches the zip on first depth-mode
/// use and installs it into `Application Support/` where it persists across launches.
actor DepthModelDownloader {

    /// Filename of the model directory once installed.
    static let modelDirectoryName = "DepthAnythingV2SmallF16.mlmodelc"

    /// Where the compiled model is hosted. Maintainers: create a GitHub release at
    /// this tag on `Supat/Focus-Check-App` and upload `DepthAnythingV2SmallF16.mlmodelc.zip`
    /// (e.g. `ditto -c -k --sequesterRsrc DepthAnythingV2SmallF16.mlmodelc …`).
    static let defaultZIPURL = URL(string:
        "https://github.com/Supat/Focus-Check-App/releases/download/depth-model-v1/DepthAnythingV2SmallF16.mlmodelc.zip"
    )!

    /// Persistent destination: `Application Support/DepthAnythingV2SmallF16.mlmodelc`.
    /// Application Support is user-data; not purged on low disk like Caches.
    static func installedURL() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport.appendingPathComponent(modelDirectoryName, isDirectory: true)
    }

    /// True if the model is already installed and loadable.
    static func isInstalled() -> Bool {
        guard let url = try? installedURL() else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Download + unzip + atomic install. Progress (0...1) is reported via the callback,
    /// which may be invoked on any thread — the caller is responsible for UI dispatch.
    func install(from source: URL = DepthModelDownloader.defaultZIPURL,
                 progress: @Sendable @escaping (Double) -> Void) async throws {
        progress(0)

        // 1. Download the ZIP to a temp file with progress observation.
        let tempZIP = FileManager.default.temporaryDirectory
            .appendingPathComponent("depth-\(UUID().uuidString).zip")
        try await download(from: source, to: tempZIP, progress: progress)
        defer { try? FileManager.default.removeItem(at: tempZIP) }

        // 2. Unzip into a scratch directory.
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("depth-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.unzipItem(at: tempZIP, to: scratch)

        // 3. Locate the `.mlmodelc` directory inside the extracted content.
        guard let extracted = locateMLModelC(in: scratch) else {
            throw AnalysisError.modelLoadFailed("ZIP does not contain a .mlmodelc directory.")
        }

        // 4. Atomic install — replace any prior copy.
        let dest = try Self.installedURL()
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: extracted, to: dest)

        progress(1.0)
    }

    /// Remove the installed model. Useful for a "re-download" UX or testing.
    func uninstall() throws {
        let url = try Self.installedURL()
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Private

    private func download(from source: URL, to destination: URL,
                          progress: @Sendable @escaping (Double) -> Void) async throws {
        let delegate = ProgressObservingDelegate(progress: progress)
        let (tempURL, response) = try await URLSession.shared.download(from: source, delegate: delegate)

        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw AnalysisError.modelLoadFailed("Download failed (HTTP \(code)).")
        }

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    private func locateMLModelC(in root: URL) -> URL? {
        // The zip may contain the .mlmodelc directly at root, or nested one level deep.
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

/// Observes `URLSessionTask.progress.fractionCompleted` via KVO and reports on the
/// main actor. `didCreateTask` is iOS 16+ and gives us the task reference without
/// needing to juggle continuations ourselves.
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
