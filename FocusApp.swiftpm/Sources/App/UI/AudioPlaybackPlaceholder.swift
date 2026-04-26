import SwiftUI
import AVFoundation

/// Rendered in place of the MetalView when the loaded source is an
/// audio-only file — `VideoFrameSource` doesn't produce frames in
/// that case, but the AVPlayer is still active and the user wants
/// some visual feedback that something is playing.
///
/// Shows: a large audio-spectrum glyph, the source filename, and a
/// pulsing animation tied to playback state. Transport (play /
/// pause / scrub) lives in the bottom panel as for video, so this
/// view is purely decorative.
struct AudioPlaybackPlaceholder: View {
    @ObservedObject var source: VideoFrameSource
    let name: String?

    /// Drives the breathing pulse on the glyph while playing.
    /// Stays at 1.0 when paused so the visual matches the audio.
    @State private var pulseScale: CGFloat = 1.0

    var body: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 220, height: 220)
                Image(systemName: "waveform")
                    .font(.system(size: 96, weight: .semibold))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.tint)
                    .scaleEffect(source.isPlaying ? pulseScale : 1.0)
                    .animation(
                        source.isPlaying
                            ? .easeInOut(duration: 0.85).repeatForever(autoreverses: true)
                            : .default,
                        value: source.isPlaying
                    )
            }
            .onAppear {
                // Kick the pulse target away from 1.0 so the
                // repeatForever animation has somewhere to go.
                pulseScale = 1.08
            }

            VStack(spacing: 4) {
                Text(name ?? "Audio")
                    .font(.headline)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .multilineTextAlignment(.center)
                Text(Self.timeReadout(
                    current: CMTimeGetSeconds(source.currentTime),
                    duration: source.duration
                ))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    /// Format the current/total seconds as `m:ss / m:ss`. Used as a
    /// subtitle under the file name.
    private static func timeReadout(
        current: Double, duration: Double
    ) -> String {
        "\(format(current)) / \(format(duration))"
    }

    private static func format(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let s = Int(seconds.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
