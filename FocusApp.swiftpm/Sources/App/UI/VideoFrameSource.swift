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
/// Also handles audio-only assets (.m4a, .mp3, .wav, etc.) — when
/// the loaded asset has no video track, no frame pump runs, but the
/// AVPlayer still plays the audio out the speaker and transport
/// (play / pause / scrub) keeps working. UI consumers branch on
/// `hasVideoTrack` to switch between the rendered MetalView and the
/// audio-only placeholder.
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
    /// True when the loaded asset has at least one video track. False
    /// for audio-only files — the player will still play audio but
    /// `currentImage` stays nil and the display link no-ops.
    let hasVideoTrack: Bool

    private let asset: AVURLAsset
    private let item: AVPlayerItem
    private let player: AVPlayer
    private let output: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?
    /// Periodic AVPlayer time observer used in audio-only mode to
    /// drive the transport bar's scrubber. Video assets get their
    /// time tick for free in the display-link `tick()` since the
    /// frame pump runs there.
    private var audioTimeObserver: Any?
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

            // Validate that the asset has at least one playable
            // track — video OR audio. Reject only when both are
            // missing. Audio-only files are valid; we'll skip the
            // frame pump but keep AVPlayer + transport working.
            let videoTracks = try await asset.loadTracks(withMediaType: .video)
            let audioTracks = try await asset.loadTracks(withMediaType: .audio)
            guard !videoTracks.isEmpty || !audioTracks.isEmpty else {
                throw VideoFrameSourceError.noVideoTrack
            }
            let hasVideo = !videoTracks.isEmpty
            let durationCM = try await asset.load(.duration)

            self.asset = asset
            self.hasVideoTrack = hasVideo
            self.duration = CMTimeGetSeconds(durationCM)

            let item = AVPlayerItem(asset: asset)
            // Only attach the video data output when there's
            // actually a video track to sample. AVPlayer plays the
            // audio out the speaker either way.
            if hasVideo {
                let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                    String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA,
                    String(kCVPixelBufferIOSurfacePropertiesKey): [String: Any](),
                ])
                item.add(output)
                self.output = output
            } else {
                self.output = nil
            }

            self.item = item
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

    /// Begin time updates. For video assets this is a screen-refresh
    /// display link sampling the frame pump; for audio-only it's a
    /// 10 Hz periodic time observer that drives the transport bar's
    /// scrubber. Idempotent.
    func start() {
        if hasVideoTrack {
            guard displayLink == nil else { return }
            let link = CADisplayLink(target: self, selector: #selector(tick))
            // Cap to 30 fps. M-series iPad ProMotion displays
            // refresh at 120 Hz, but we don't need a new pixel
            // buffer that often — most video is 30 fps natively
            // and any extra ticks just thrash main-thread @Published
            // updates (sourceImage + 10 smoother fields per tick),
            // which has been observed to starve toolbar Menu state
            // and freeze sheet presentation. 30 fps is the minimum
            // of "matches typical source rate" and "keeps the UI
            // responsive".
            link.preferredFramesPerSecond = 30
            link.add(to: .main, forMode: .common)
            self.displayLink = link
        } else {
            guard audioTimeObserver == nil else { return }
            let interval = CMTime(seconds: 0.1, preferredTimescale: 600)
            audioTimeObserver = player.addPeriodicTimeObserver(
                forInterval: interval, queue: .main
            ) { [weak self] time in
                self?.currentTime = time
            }
        }
    }

    /// Tear down the display link / time observer and pause AVPlayer.
    /// Safe to call before `start()`.
    func stop() {
        displayLink?.invalidate()
        displayLink = nil
        if let observer = audioTimeObserver {
            player.removeTimeObserver(observer)
            audioTimeObserver = nil
        }
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
        guard let output else { return }
        let hostTime = displayLink?.targetTimestamp ?? CACurrentMediaTime()
        let itemTime = output.itemTime(forHostTime: hostTime)
        // Even for audio-only this could be reached if hasVideoTrack
        // races, but the early `output` guard above handles it.
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
