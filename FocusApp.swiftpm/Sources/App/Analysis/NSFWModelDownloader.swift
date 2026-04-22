import Foundation
import ZIPFoundation

/// Downloads and installs a Core ML NSFW classifier at runtime.
///
/// Parallel to `DepthModelDownloader` — Swift Playgrounds doesn't compile
/// `.mlmodel` / `.mlpackage` assets, so the artifact must already be in
/// `.mlmodelc` form when we fetch it.
///
/// Recommended source model: lovoo/NSFWDetector (BSD-licensed, 17 KB,
/// CreateML-trained binary SFW/NSFW classifier).
///
///   Download URL:
///     https://github.com/lovoo/NSFWDetector/releases/download/1.1.0/NSFW.mlmodel
///
/// Maintainer setup (one-time, Mac required):
///   curl -L -o NSFW.mlmodel \
///     https://github.com/lovoo/NSFWDetector/releases/download/1.1.0/NSFW.mlmodel
///   xcrun coremlcompiler compile NSFW.mlmodel /tmp/
///   ditto -c -k --sequesterRsrc --keepParent \
///         /tmp/NSFW.mlmodelc NSFW.mlmodelc.zip
///   gh release create nsfw-model-v1 NSFW.mlmodelc.zip \
///       --repo Supat/Focus-Check-App
actor NSFWModelDownloader {

    static let modelDirectoryName = "NSFW.mlmodelc"

    /// Where the compiled model is hosted. Maintainer creates release tag
    /// `nsfw-model-v1` on `Supat/Focus-Check-App` and attaches
    /// `NSFW.mlmodelc.zip` as an asset.
    static let defaultZIPURL = URL(string:
        "https://github.com/Supat/Focus-Check-App/releases/download/nsfw-model-v1/NSFW.mlmodelc.zip"
    )!

    /// Persistent destination: `Application Support/OpenNSFW.mlmodelc`.
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

    /// Download + unzip + atomic install. Progress (0...1) reports via the
    /// callback, which may be invoked on any thread.
    func install(from source: URL = NSFWModelDownloader.defaultZIPURL,
                 progress: @Sendable @escaping (Double) -> Void) async throws {
        progress(0)

        let tempZIP = FileManager.default.temporaryDirectory
            .appendingPathComponent("nsfw-\(UUID().uuidString).zip")
        try await download(from: source, to: tempZIP, progress: progress)
        defer { try? FileManager.default.removeItem(at: tempZIP) }

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("nsfw-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.unzipItem(at: tempZIP, to: scratch)

        guard let extracted = locateMLModelC(in: scratch) else {
            throw AnalysisError.modelLoadFailed("ZIP does not contain a .mlmodelc directory.")
        }

        let dest = try Self.installedURL()
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: extracted, to: dest)

        progress(1.0)
    }

    func uninstall() throws {
        let url = try Self.installedURL()
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Private

    private func download(from source: URL, to destination: URL,
                          progress: @Sendable @escaping (Double) -> Void) async throws {
        let delegate = NSFWProgressDelegate(progress: progress)
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

private final class NSFWProgressDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
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
