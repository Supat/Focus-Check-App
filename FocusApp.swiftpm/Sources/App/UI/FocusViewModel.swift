import SwiftUI
import CoreImage
import Metal

enum OverlayStyle: String, CaseIterable, Identifiable {
    case peaking = "Peaking"
    case heatmap = "Heatmap"
    case mask    = "Mask"

    var id: String { rawValue }

    var systemImage: String {
        switch self {
        case .peaking: return "sparkles"
        case .heatmap: return "thermometer.sun"
        case .mask:    return "square.fill.on.square"
        }
    }
}

enum AnalysisMode: String, CaseIterable, Identifiable {
    case sharpness = "Sharpness"
    case depth     = "Depth"
    case hybrid    = "Hybrid"

    var id: String { rawValue }
}

enum DepthInstallState: Equatable {
    case notInstalled
    case downloading(progress: Double)
    case installed
    case failed(String)
}

@MainActor
final class FocusViewModel: ObservableObject {
    // Scrubbable display state — cheap, no re-analysis.
    @Published var threshold: Float = 0.35
    @Published var overlayColor: Color = .yellow
    @Published var style: OverlayStyle = .peaking

    // Analysis configuration — change triggers re-analysis.
    @Published var mode: AnalysisMode = .sharpness

    // Source + derived state published for renderer consumption.
    @Published var sourceImage: CIImage?
    @Published var sharpnessOverlay: CIImage?
    @Published var depthOverlay: CIImage?
    @Published var isAnalyzing: Bool = false
    @Published var errorMessage: String?
    @Published var depthAvailable: Bool = false
    @Published var depthInstall: DepthInstallState = .notInstalled

    let analyzer: FocusAnalyzer
    private var currentTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?

    init() {
        self.analyzer = FocusAnalyzer()
        Task { [weak self] in
            let available = await self?.analyzer.isDepthAvailable ?? false
            await MainActor.run {
                self?.depthAvailable = available
                self?.depthInstall = available ? .installed : .notInstalled
            }
        }
    }

    func downloadDepthModel() {
        guard installTask == nil else { return }
        depthInstall = .downloading(progress: 0)
        let analyzer = self.analyzer
        installTask = Task { [weak self] in
            do {
                try await analyzer.installDepthModel { p in
                    // Progress callback may arrive on any thread — hop to main.
                    Task { @MainActor in
                        self?.depthInstall = .downloading(progress: p)
                    }
                }
                await MainActor.run {
                    self?.depthAvailable = true
                    self?.depthInstall = .installed
                }
            } catch {
                await MainActor.run {
                    self?.depthInstall = .failed(error.localizedDescription)
                }
            }
            await MainActor.run { self?.installTask = nil }
        }
    }

    func load(url: URL) {
        currentTask?.cancel()
        isAnalyzing = true
        errorMessage = nil
        sharpnessOverlay = nil
        depthOverlay = nil

        let mode = self.mode
        let analyzer = self.analyzer

        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let image = try await analyzer.loadImage(from: url)
                try Task.checkCancellation()
                let overlays = try await analyzer.analyze(mode: mode)
                try Task.checkCancellation()
                await MainActor.run {
                    self?.sourceImage = image
                    self?.sharpnessOverlay = overlays.sharpness
                    self?.depthOverlay = overlays.depth
                    self?.isAnalyzing = false
                }
            } catch is CancellationError {
                // Superseded by a newer load — stay silent.
            } catch {
                await MainActor.run {
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                }
            }
        }
    }

    func clear() {
        currentTask?.cancel()
        currentTask = nil
        sourceImage = nil
        sharpnessOverlay = nil
        depthOverlay = nil
        errorMessage = nil
        isAnalyzing = false
    }

    func reanalyze() {
        guard sourceImage != nil else { return }
        currentTask?.cancel()
        isAnalyzing = true
        let mode = self.mode
        let analyzer = self.analyzer
        currentTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let overlays = try await analyzer.analyze(mode: mode)
                try Task.checkCancellation()
                await MainActor.run {
                    self?.sharpnessOverlay = overlays.sharpness
                    self?.depthOverlay = overlays.depth
                    self?.isAnalyzing = false
                }
            } catch is CancellationError {
                // Swallow — a newer analyze is in flight.
            } catch {
                await MainActor.run {
                    self?.errorMessage = error.localizedDescription
                    self?.isAnalyzing = false
                }
            }
        }
    }
}
