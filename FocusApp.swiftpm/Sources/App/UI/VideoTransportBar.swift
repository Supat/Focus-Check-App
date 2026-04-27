import SwiftUI
import AVFoundation

/// Compact playback control row surfaced above the existing
/// `OverlayControls` whenever the loaded source is a video.
/// Play/pause toggle, scrubber, and a `mm:ss / mm:ss` readout —
/// nothing else. Wraps the small bit of state needed to debounce
/// the slider against live `currentTime` updates while the user is
/// dragging.
struct VideoTransportBar: View {
    @ObservedObject var source: VideoFrameSource

    /// Position the user is currently dragging towards. nil while
    /// not scrubbing — the slider reads from `source.currentTime`
    /// in that case. While set, the slider shows the user's intent
    /// rather than fighting with AVPlayer's per-frame
    /// `currentTime` republishes.
    @State private var scrubbingPosition: Double?
    /// Was playback running before the user grabbed the scrubber?
    /// Tracked so the bar resumes on release.
    @State private var wasPlayingBeforeScrub: Bool = false
    /// Back-pressure flag for in-drag seekFast calls. If a fast
    /// seek is in flight, drag updates skip queueing another one
    /// — AVPlayer drops queued seeks anyway, and skipping keeps
    /// the most recent drag position from getting stuck behind a
    /// stale seek that hadn't completed.
    @State private var fastSeekInFlight: Bool = false

    var body: some View {
        HStack(spacing: 12) {
            // Step-by-frame buttons sit either side of play/pause.
            // Audio-only sources hide them — `step(byCount:)` is a
            // video-track-only API.
            if source.hasVideoTrack {
                Button { source.stepFrame(by: -1) } label: {
                    Image(systemName: "backward.frame.fill")
                        .font(.title3)
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.borderless)
            }

            Button {
                if source.isPlaying {
                    source.pause()
                } else {
                    source.play()
                }
            } label: {
                Image(systemName: source.isPlaying ? "pause.fill" : "play.fill")
                    .font(.title3)
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderless)

            if source.hasVideoTrack {
                Button { source.stepFrame(by: 1) } label: {
                    Image(systemName: "forward.frame.fill")
                        .font(.title3)
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.borderless)
            }

            Text(Self.timeString(displayedSeconds))
                .font(.caption.monospacedDigit())
                .frame(width: 44, alignment: .trailing)

            Slider(
                value: scrubBinding,
                in: 0...max(source.duration, 0.001)
            ) { editing in
                if editing {
                    wasPlayingBeforeScrub = source.isPlaying
                    source.pause()
                    // Tells the renderer + per-subject overlays
                    // to hide while we drag — anchor data lags
                    // the playhead by up to 500 ms (analyzer
                    // pulse interval) so showing them on a fast
                    // scrub pins mosaic boxes on stale frames.
                    source.setSeeking(true)
                } else {
                    let target = scrubbingPosition ?? displayedSeconds
                    let resume = wasPlayingBeforeScrub
                    scrubbingPosition = nil
                    Task { @MainActor in
                        // Final landing seek is precise (zero
                        // tolerances) so the user's exact target
                        // frame ends up on screen.
                        await source.seek(to: target)
                        source.setSeeking(false)
                        if resume { source.play() }
                    }
                }
            }

            Text(Self.timeString(source.duration))
                .font(.caption.monospacedDigit())
                .frame(width: 44, alignment: .leading)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var displayedSeconds: Double {
        scrubbingPosition ?? CMTimeGetSeconds(source.currentTime)
    }

    private var scrubBinding: Binding<Double> {
        Binding(
            get: { displayedSeconds },
            set: { newValue in
                scrubbingPosition = newValue
                // Live preview: keyframe-tolerance seek as the
                // user drags. Throttled by `fastSeekInFlight` so
                // we don't queue seeks faster than AVPlayer can
                // complete them — without this, fast drags pile
                // up old positions and the preview lags the
                // gesture by hundreds of ms.
                guard !fastSeekInFlight else { return }
                fastSeekInFlight = true
                Task { @MainActor in
                    await source.seekFast(to: newValue)
                    fastSeekInFlight = false
                }
            }
        )
    }

    private static func timeString(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
