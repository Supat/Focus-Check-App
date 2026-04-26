import Foundation
import AVFoundation
import CoreImage
import UIKit

/// Pulls decoded frames from an `AVAsset` at the screen refresh rate
/// and republishes them as `CIImage` so the existing image-centric
/// renderer can consume them without modification. The renderer
/// reads from `currentImage`; SwiftUI observes the same property to
/// trigger MetalView redraws.
///
/// Pipeline:
///   AVAsset
///     → AVPlayer + AVPlayerItem
///       → AVPlayerItemVideoOutput (BGRA, IOSurface-backed)
///         → CADisplayLink ticks at display rate
///           → copyPixelBuffer(forItemTime:) → CIImage
///
/// Transport (`play`, `pause`, `seek`) maps directly onto AVPlayer;
/// the display link runs continuously while the source is `start()`-ed
/// so a paused asset still re-renders the same frame whenever the
/// renderer needs to (e.g. a mosaic-mode change updates the overlay
/// graph and the MetalView pulls a fresh composite).
///
/// Color: BGRA8 + sRGB tag is the SDR baseline. HDR (BT.2020 PQ/HLG)
/// would need a 10-bit pixel format and the right `CGColorSpace` —
/// deferred until SDR works end-to-end.
@MainActor
final class VideoFrameSource: ObservableObject {
    /// Most recently decoded frame, ready for the renderer.
    /// Nil before the first display-link tick that finds a new frame
    /// in the output, and right after `stop()`.
    @Published private(set) var currentImage: CIImage?
    /// Playhead in the asset's timebase. Mirrors what AVPlayerItem
    /// thinks the current item time is — the analyzer keys its
    /// TrackStore on this so playback and analysis sample the same
    /// timeline.
    @Published private(set) var currentTime: CMTime = .zero
    /// Whether AVPlayer is currently playing. Toggled by `play()` /
    /// `pause()`; surfaced for the toolbar transport button.
    @Published private(set) var isPlaying: Bool = false
    /// Total duration in seconds. Populated once the asset finishes
    /// loading its `.duration` property in `init`.
    let duration: TimeInterval

    private let asset: AVURLAsset
    private let item: AVPlayerItem
    private let player: AVPlayer
    private let output: AVPlayerItemVideoOutput
    private var displayLink: CADisplayLink?
    /// True when this source took ownership of a security scope on
    /// the URL (fileImporter video path that streams without a temp
    /// copy). Released in `deinit` and on early throw paths.
    private let didStartSecurityScope: Bool
    private let scopedURL: URL

    /// `isSecurityScoped` should be true only when the caller has
    /// already started the security scope on `url` (i.e. an
    /// `.fileImporter` video that wants to stream directly). The
    /// init takes ownership and releases the scope on deinit.
    init(url: URL, isSecurityScoped: Bool = false) async throws {
        // The caller already started the scope before handing the
        // URL across; we just record that we're responsible for
        // releasing it. If init throws below, we release explicitly
        // since deinit doesn't run on partially-initialized objects.
        self.scopedURL = url
        self.didStartSecurityScope = isSecurityScoped

        do {
            let asset = AVURLAsset(url: url)

            // Validate up front that the file actually has a video
            // track — image / audio / text imports would silently
            // fall through to a black playback otherwise.
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard !tracks.isEmpty else {
                throw VideoFrameSourceError.noVideoTrack
            }
            let durationCM = try await asset.load(.duration)

            self.asset = asset
            self.duration = CMTimeGetSeconds(durationCM)

            let item = AVPlayerItem(asset: asset)
            // BGRA8 is the Core Image / Metal-friendly default;
            // IOSurface backing keeps the buffer eligible for
            // zero-copy texture use when the renderer materializes
            // via the shared CIContext.
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA,
                String(kCVPixelBufferIOSurfacePropertiesKey): [:] as CFDictionary,
            ])
            item.add(output)

            self.item = item
            self.output = output
            self.player = AVPlayer(playerItem: item)
        } catch {
            // Init throw path — Swift skips deinit since the object
            // never finished initializing, so release the scope now
            // or it leaks.
            if isSecurityScoped {
                url.stopAccessingSecurityScopedResource()
            }
            throw error
        }
    }

    deinit {
        if didStartSecurityScope {
            scopedURL.stopAccessingSecurityScopedResource()
        }
    }

    /// Begin sampling the output at the screen refresh rate. Idempotent
    /// — calling twice in a row reuses the existing display link.
    func start() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        self.displayLink = link
    }

    /// Tear down the display link and pause AVPlayer. Safe to call
    /// before `start()`.
    func stop() {
        displayLink?.invalidate()
        displayLink = nil
        player.pause()
        isPlaying = false
        currentImage = nil
    }

    func play() {
        player.play()
        isPlaying = true
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    /// Frame-accurate seek — the analyzer reads `currentTime` to key
    /// its TrackStore lookups, so step-imprecise seeks would let
    /// overlays land on the wrong sampled snapshot.
    func seek(to seconds: TimeInterval) async {
        let target = CMTime(
            seconds: seconds,
            preferredTimescale: CMTimeScale(NSEC_PER_SEC)
        )
        await player.seek(
            to: target,
            toleranceBefore: .zero,
            toleranceAfter: .zero
        )
    }

    /// Display-link callback. Asks the output whether a new pixel
    /// buffer is ready for the upcoming refresh; if so, wraps it in
    /// a `CIImage` and republishes. Skips silently when the output
    /// has nothing new — common during seeks or when the player
    /// hasn't reached `.readyToPlay` yet.
    @objc private func tick() {
        let hostTime = displayLink?.targetTimestamp ?? CACurrentMediaTime()
        let itemTime = output.itemTime(forHostTime: hostTime)
        guard output.hasNewPixelBuffer(forItemTime: itemTime),
              let pixelBuffer = output.copyPixelBuffer(
                forItemTime: itemTime,
                itemTimeForDisplay: nil
              )
        else { return }
        let image = CIImage(
            cvPixelBuffer: pixelBuffer,
            options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]
        )
        currentImage = image
        currentTime = itemTime
    }
}

enum VideoFrameSourceError: Error, LocalizedError {
    case noVideoTrack

    var errorDescription: String? {
        switch self {
        case .noVideoTrack:
            return "The selected file doesn't contain a video track."
        }
    }
}
