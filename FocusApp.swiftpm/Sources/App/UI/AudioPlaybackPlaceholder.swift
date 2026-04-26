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
    /// Top CLAP audio-context matches for the loaded file, sorted
    /// descending. Empty until scoring completes (or stays empty
    /// when the CLAP archive isn't installed).
    let audioMatches: [CLAPMatch]
    /// True when the CLAP archive is installed — controls whether
    /// the placeholder shows a "scoring…" indicator while waiting
    /// for the result, or just hides the row entirely.
    let clapAvailable: Bool

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

            audioContextSection
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    /// CLAP audio-context readout. Three states:
    ///   - archive not installed: hidden (the Model Manager is the
    ///     prompt to install it, no need to nag from here).
    ///   - installed but matches empty: "Analyzing audio…" indicator.
    ///   - installed and matches present: top-3 prompt rows with
    ///     similarity %.
    @ViewBuilder
    private var audioContextSection: some View {
        if clapAvailable {
            VStack(spacing: 6) {
                Text("Audio Context")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                if audioMatches.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Analyzing audio…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    ForEach(audioMatches.prefix(3), id: \.self) { match in
                        HStack(spacing: 12) {
                            Text(match.prompt)
                                .font(.caption)
                                .lineLimit(1)
                                .truncationMode(.tail)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text("\(Int((match.similarity * 100).rounded()))%")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .frame(maxWidth: 360)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial,
                        in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
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
