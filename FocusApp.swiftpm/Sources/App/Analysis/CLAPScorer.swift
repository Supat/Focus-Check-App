import Foundation
import AVFoundation
import CoreML

/// One CLAP zero-shot match for a window of the loaded audio.
/// Mirror of `CLIPMatch` for images: a prompt + its cosine
/// similarity against the audio embedding for some 10-second
/// window, plus the window's start time in seconds so the UI can
/// surface "the strongest match was at 1:23 in the file".
struct CLAPMatch: Hashable, Sendable {
    let prompt: String
    let similarity: Float
    let windowStart: TimeInterval
}

/// CLAP-based audio context scorer. Reads the audio track with
/// AVAssetReader, slices into overlapping fixed-length windows, runs
/// the audio encoder once per window, and ranks pre-embedded text
/// prompts by the highest cosine similarity any window produced.
///
/// One-shot: called once on audio/video import (currently audio-only;
/// video integration is a downstream extension), result is cached
/// on the view model and surfaced as a static badge. No live update
/// during playback — reaches its verdict in a single pass.
///
/// Returns `[]` when the model archive isn't installed; UI treats
/// absent and empty as "no audio context signal".
struct CLAPScorer {
    private var encoder: CLAPEncoder? { CLAPEncoder.shared }

    /// True when the CLAP audio bundle is installed on disk.
    var isReady: Bool { ModelArchive.clapAudio.isInstalled() }

    /// Trigger lazy MLModel load without running an inference —
    /// called from `FocusAnalyzer.prewarmModels` so the first import
    /// after launch doesn't pay the ~1 s ANE compile cost.
    func warm() -> Bool { encoder != nil }

    /// Score the audio at `audioURL` against the bundled prompts.
    /// Returns the top `topK` prompts (one entry per prompt — the
    /// best-scoring window for that prompt), sorted descending.
    /// Returns `[]` on any failure (model missing, decode error,
    /// no audio track, dimension mismatch).
    func score(audioURL: URL, topK: Int = 3) async -> [CLAPMatch] {
        guard let encoder, let prompts = CLAPPromptStore.shared else {
            return []
        }
        let sampleRate = encoder.sampleRate
        let windowSamples = encoder.windowSamples
        guard let samples = await Self.extractSamples(
            from: audioURL, sampleRate: sampleRate
        ) else { return [] }
        // Reject very short clips — anything below a quarter of a
        // window can't produce a meaningful embedding.
        guard samples.count >= windowSamples / 4 else { return [] }

        // 50%-overlap windows so a salient sound that straddles a
        // window boundary still lands centred in some other window.
        let stride = max(1, windowSamples / 2)
        var allMatches: [CLAPMatch] = []
        var windowStart = 0
        while windowStart + windowSamples <= samples.count {
            let slice = Array(samples[windowStart..<windowStart + windowSamples])
            scoreWindow(
                slice: slice,
                start: TimeInterval(windowStart) / sampleRate,
                encoder: encoder,
                prompts: prompts,
                into: &allMatches
            )
            windowStart += stride
        }
        // Tail: zero-pad the final partial window so a clip that
        // ends mid-event is still represented.
        if windowStart < samples.count,
           samples.count - windowStart >= windowSamples / 4 {
            var padded = Array(samples[windowStart...])
            padded.append(
                contentsOf: [Float](
                    repeating: 0, count: windowSamples - padded.count
                )
            )
            scoreWindow(
                slice: padded,
                start: TimeInterval(windowStart) / sampleRate,
                encoder: encoder,
                prompts: prompts,
                into: &allMatches
            )
        }
        // Group by prompt, keep the best-scoring window per prompt.
        var bestPerPrompt: [String: CLAPMatch] = [:]
        for match in allMatches {
            if let existing = bestPerPrompt[match.prompt] {
                if match.similarity > existing.similarity {
                    bestPerPrompt[match.prompt] = match
                }
            } else {
                bestPerPrompt[match.prompt] = match
            }
        }
        return Array(bestPerPrompt.values)
            .sorted { $0.similarity > $1.similarity }
            .prefix(topK)
            .map { $0 }
    }

    private func scoreWindow(
        slice: [Float],
        start: TimeInterval,
        encoder: CLAPEncoder,
        prompts: CLAPPromptStore,
        into matches: inout [CLAPMatch]
    ) {
        guard let embedding = encoder.embed(samples: slice) else { return }
        let normalized = Self.l2Normalize(embedding)
        for entry in prompts.entries {
            guard entry.embedding.count == normalized.count else { continue }
            var dot: Float = 0
            for i in 0..<normalized.count {
                dot += normalized[i] * entry.embedding[i]
            }
            matches.append(CLAPMatch(
                prompt: entry.prompt,
                similarity: dot,
                windowStart: start
            ))
        }
    }

    private static func l2Normalize(_ v: [Float]) -> [Float] {
        var sum: Float = 0
        for x in v { sum += x * x }
        let norm = sum > 0 ? sqrt(sum) : 1
        return v.map { $0 / norm }
    }

    /// Decode the asset's first audio track into a flat `[Float]` of
    /// mono samples at `sampleRate`. AVAssetReader resamples to the
    /// requested rate and downmixes to mono via the output settings,
    /// so the caller doesn't need vDSP / AVAudioConverter plumbing.
    /// Returns nil when the asset has no audio track or decode fails.
    private static func extractSamples(
        from url: URL, sampleRate: Double
    ) async -> [Float]? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .audio).first
        else { return nil }

        return await Task.detached(priority: .userInitiated) {
            do {
                let reader = try AVAssetReader(asset: asset)
                let outputSettings: [String: Any] = [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVLinearPCMBitDepthKey: 32,
                    AVLinearPCMIsFloatKey: true,
                    AVLinearPCMIsBigEndianKey: false,
                    AVLinearPCMIsNonInterleaved: false,
                    AVSampleRateKey: sampleRate,
                    AVNumberOfChannelsKey: 1,
                ]
                let output = AVAssetReaderTrackOutput(
                    track: track, outputSettings: outputSettings
                )
                output.alwaysCopiesSampleData = false
                guard reader.canAdd(output) else { return nil }
                reader.add(output)
                guard reader.startReading() else { return nil }

                var samples: [Float] = []
                samples.reserveCapacity(Int(sampleRate) * 30)  // 30s budget

                while reader.status == .reading,
                      let buffer = output.copyNextSampleBuffer(),
                      let blockBuffer = CMSampleBufferGetDataBuffer(buffer) {
                    var totalLength = 0
                    var dataPointer: UnsafeMutablePointer<Int8>?
                    let result = CMBlockBufferGetDataPointer(
                        blockBuffer,
                        atOffset: 0,
                        lengthAtOffsetOut: nil,
                        totalLengthOut: &totalLength,
                        dataPointerOut: &dataPointer
                    )
                    guard result == kCMBlockBufferNoErr,
                          let pointer = dataPointer else {
                        CMSampleBufferInvalidate(buffer)
                        continue
                    }
                    let count = totalLength / MemoryLayout<Float>.size
                    pointer.withMemoryRebound(
                        to: Float.self, capacity: count
                    ) { typed in
                        samples.append(contentsOf:
                            UnsafeBufferPointer(start: typed, count: count)
                        )
                    }
                    CMSampleBufferInvalidate(buffer)
                }
                return reader.status == .completed ? samples : nil
            } catch {
                return nil
            }
        }.value
    }
}

// MARK: - Audio encoder wrapper

/// Thin wrapper around the CLAP audio-encoder Core ML model. Inspects
/// the input multi-array shape at load time so the maintainer can
/// swap HTSAT-tiny (10 s @ 48 kHz = 480 000 samples) for any other
/// LAION-CLAP variant without code changes — `windowSamples` reads
/// off the model's declared input length.
private final class CLAPEncoder {
    static var shared: CLAPEncoder? = {
        try? CLAPEncoder()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    /// Number of samples per inference call. Matches the model's
    /// declared 1-D / 2-D multi-array length; caller pads or trims
    /// each window to exactly this count.
    let windowSamples: Int
    /// Sample rate the encoder was trained on. Hardcoded to 48 kHz —
    /// LAION-CLAP's default. Override here if a maintainer ships a
    /// 16 kHz Microsoft CLAP variant instead; the model archive
    /// could carry the rate as a metadata key in a future revision.
    let sampleRate: Double = 48000

    init() throws {
        let installRoot = try ModelArchive.clapAudio.installedURL()
        let modelURL = installRoot
            .appendingPathComponent("CLAPAudioEncoder.mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AnalysisError.modelMissing
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: config)
        } catch {
            throw AnalysisError.modelLoadFailed(error.localizedDescription)
        }

        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName
        guard let input = inputs.first(where: {
            $0.value.type == .multiArray
        }) ?? inputs.first,
              let output = outputs.first
        else {
            throw AnalysisError.modelLoadFailed(
                "CLAP model has no usable input/output."
            )
        }
        self.inputName = input.key
        self.outputName = output.key

        // The waveform axis is the longest dim in the input shape.
        // For [1, 480000] or just [480000] this picks 480000.
        var shapeSamples = 480000
        if let constraint = input.value.multiArrayConstraint {
            let dims = constraint.shape.map { $0.intValue }
            shapeSamples = dims.max() ?? 480000
        }
        self.windowSamples = max(shapeSamples, 1)
    }

    /// Run one encoder pass over `samples` (which must be
    /// `windowSamples` long) and return the embedding as `[Float]`.
    /// Returns nil on length mismatch or prediction failure.
    func embed(samples: [Float]) -> [Float]? {
        guard samples.count == windowSamples else { return nil }
        do {
            // [1, N] is the most common input shape for LAION-CLAP
            // audio encoders. If the model declares [N] (no batch
            // dim) the runtime broadcasts; if it declares [1, N]
            // we line up exactly. Either way works.
            let array = try MLMultiArray(
                shape: [1, NSNumber(value: windowSamples)],
                dataType: .float32
            )
            samples.withUnsafeBufferPointer { src in
                let dst = array.dataPointer.assumingMemoryBound(to: Float.self)
                dst.update(from: src.baseAddress!, count: windowSamples)
            }
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(multiArray: array)
            ])
            let result = try model.prediction(from: features)
            guard let outArray = result
                .featureValue(for: outputName)?.multiArrayValue
            else { return nil }
            let count = outArray.count
            var out = [Float](repeating: 0, count: count)
            outArray.dataPointer.assumingMemoryBound(to: Float.self)
                .withMemoryRebound(to: Float.self, capacity: count) { src in
                    out.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress?.update(from: src, count: count)
                    }
                }
            return out
        } catch {
            return nil
        }
    }
}

// MARK: - Prompt embeddings

/// Loaded `clap-prompts.json` — each entry pairs a human-readable
/// text prompt with the matching text encoder's pre-computed,
/// pre-normalized embedding. Same shape as CLIP's prompts file but
/// embeddings come from CLAP's text encoder so they're guaranteed
/// to align with the audio encoder's output space.
private final class CLAPPromptStore {
    struct Entry {
        let prompt: String
        let embedding: [Float]
    }

    static var shared: CLAPPromptStore? = {
        try? CLAPPromptStore()
    }()

    let entries: [Entry]

    init() throws {
        let installRoot = try ModelArchive.clapAudio.installedURL()
        let promptsURL = installRoot
            .appendingPathComponent("clap-prompts.json")
        guard FileManager.default.fileExists(atPath: promptsURL.path) else {
            throw AnalysisError.modelMissing
        }
        let data = try Data(contentsOf: promptsURL)

        struct JSONEntry: Decodable {
            let prompt: String
            let embedding: [Float]
        }
        let raw = try JSONDecoder().decode([JSONEntry].self, from: data)
        self.entries = raw.map { Entry(prompt: $0.prompt, embedding: $0.embedding) }
    }
}
