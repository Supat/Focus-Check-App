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

    var body: some View {
        HStack(spacing: 12) {
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
                } else {
                    let target = scrubbingPosition ?? displayedSeconds
                    let resume = wasPlayingBeforeScrub
                    scrubbingPosition = nil
                    Task { @MainActor in
                        await source.seek(to: target)
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
            set: { scrubbingPosition = $0 }
        )
    }

    private static func timeString(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
