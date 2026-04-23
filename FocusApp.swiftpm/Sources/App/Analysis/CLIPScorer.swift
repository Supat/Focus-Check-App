import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML

/// One CLIP zero-shot match for the current image. Prompts are supplied
/// at model-conversion time (see `ModelArchive.clip` docs); the score
/// is the cosine similarity between the image embedding and the
/// prompt's pre-computed text embedding, normalized into [-1, 1].
struct CLIPMatch: Hashable, Sendable {
    let prompt: String
    let similarity: Float
}

/// CLIP-based context scorer. Runs one image-encoder pass per analyze
/// call and ranks a set of pre-embedded text prompts by cosine
/// similarity. Returns an empty array when the model archive isn't
/// installed — the UI treats absent and empty as "no context signal".
struct CLIPScorer {
    /// Lazy singleton. Accessed only at first `score(...)` call so app
    /// launch doesn't pay the MLModel compile cost until the first
    /// analysis has actually been requested.
    private var encoder: CLIPEncoder? { CLIPEncoder.shared }

    /// True when the CLIP bundle is installed on disk. Cheap — no
    /// MLModel load.
    var isReady: Bool { ModelArchive.clip.isInstalled() }

    /// Trigger the lazy MLModel load without running a prediction —
    /// called from `FocusAnalyzer.prewarmModels` so the first analyze
    /// tap doesn't absorb the ~1 s compile cost.
    func warm() -> Bool { encoder != nil }

    /// Score the image against the bundled prompts. Returns the top
    /// `topK` matches sorted by similarity (descending). Returns an
    /// empty array when the model or prompts couldn't be loaded.
    func score(image: CIImage, ciContext: CIContext, topK: Int = 3) -> [CLIPMatch] {
        guard let encoder, let prompts = CLIPPromptStore.shared else {
            print("[CLIP] scorer not ready — encoder=\(encoder == nil ? "nil" : "ok") prompts=\(CLIPPromptStore.shared == nil ? "nil" : "ok")")
            return []
        }
        guard let imageEmbedding = encoder.embed(image: image, ciContext: ciContext) else {
            print("[CLIP] embed() returned nil")
            return []
        }

        let imgNorm = l2Normalize(imageEmbedding)
        var matches: [CLIPMatch] = []
        matches.reserveCapacity(prompts.entries.count)
        var skipped = 0
        for entry in prompts.entries {
            guard entry.embedding.count == imgNorm.count else {
                skipped += 1
                continue
            }
            var dot: Float = 0
            for i in 0..<imgNorm.count {
                dot += imgNorm[i] * entry.embedding[i]
            }
            matches.append(CLIPMatch(prompt: entry.prompt, similarity: dot))
        }
        if matches.isEmpty {
            let promptDim = prompts.entries.first?.embedding.count ?? 0
            print("[CLIP] no matches — image dim=\(imgNorm.count) prompt dim=\(promptDim) skipped=\(skipped) (dimension mismatch means the image encoder and text encoder came from different CLIP variants)")
        }
        matches.sort { $0.similarity > $1.similarity }
        return Array(matches.prefix(topK))
    }

    private func l2Normalize(_ v: [Float]) -> [Float] {
        var sum: Float = 0
        for x in v { sum += x * x }
        let norm = sum > 0 ? sqrt(sum) : 1
        return v.map { $0 / norm }
    }
}

// MARK: - Image encoder wrapper

/// Thin wrapper around the CLIP image-encoder Core ML model. Resolves
/// the input image size from the model's constraint at load time, so
/// a maintainer can swap ViT-B/32 (224²) for ViT-L/14 (224² or 336²)
/// without code changes. Output is a 1-D multiarray of floats —
/// embedding dimension depends on the model variant (512 for B/32,
/// 768 for L/14).
private final class CLIPEncoder {
    static var shared: CLIPEncoder? = {
        try? CLIPEncoder()
    }()

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputSize: CGSize

    init() throws {
        let installRoot = try ModelArchive.clip.installedURL()
        let modelURL = installRoot.appendingPathComponent("CLIPImageEncoder.mlmodelc")
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
        guard let input = inputs.first(where: { $0.value.type == .image }) ?? inputs.first,
              let output = outputs.first
        else {
            throw AnalysisError.modelLoadFailed("CLIP model has no usable input/output.")
        }
        self.inputName = input.key
        self.outputName = output.key

        var resolvedSize = CGSize(width: 224, height: 224)
        if let constraint = input.value.imageConstraint {
            resolvedSize = CGSize(
                width: constraint.pixelsWide,
                height: constraint.pixelsHigh
            )
        }
        self.inputSize = resolvedSize
    }

    /// Run one image-encoder pass and return the embedding as a `[Float]`.
    /// Uses the same aspect-preserving centre-crop-and-resize convention
    /// CLIP was trained with (short side → inputSize, centre crop to
    /// square). Returns nil on any failure.
    func embed(image: CIImage, ciContext: CIContext) -> [Float]? {
        let prepared = centerCrop(image, to: inputSize)
        let w = Int(inputSize.width)
        let h = Int(inputSize.height)

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                            kCVPixelFormatType_32BGRA,
                            attrs as CFDictionary,
                            &pixelBuffer)
        guard let pb = pixelBuffer else { return nil }
        ciContext.render(prepared, to: pb)

        do {
            let features = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(pixelBuffer: pb)
            ])
            let result = try model.prediction(from: features)
            guard let array = result.featureValue(for: outputName)?.multiArrayValue else {
                return nil
            }
            let count = array.count
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count {
                out[i] = array[i].floatValue
            }
            return out
        } catch {
            return nil
        }
    }

    /// Scale `image` so its shorter side matches the CLIP input size,
    /// then center-crop to a square. Matches the official CLIP
    /// preprocessing so image embeddings align with the text encoder's
    /// distribution.
    private func centerCrop(_ image: CIImage, to size: CGSize) -> CIImage {
        let src = image.extent
        let scale = max(size.width / src.width, size.height / src.height)
        let scaled = image.translatedToOrigin()
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let scaledExtent = scaled.extent
        let cropX = (scaledExtent.width - size.width) / 2
        let cropY = (scaledExtent.height - size.height) / 2
        return scaled
            .cropped(to: CGRect(x: cropX, y: cropY,
                                width: size.width, height: size.height))
            .transformed(by: CGAffineTransform(translationX: -cropX, y: -cropY))
    }
}

// MARK: - Prompt embeddings

/// Loaded `clip-prompts.json` — each entry pairs a human-readable text
/// prompt with the text encoder's embedding. Shipped inside the CLIP
/// archive so the text encoder's output (which must match the image
/// encoder's output space) doesn't drift between releases.
private final class CLIPPromptStore {
    struct Entry {
        let prompt: String
        let embedding: [Float]   // pre-normalized at export time
    }

    static var shared: CLIPPromptStore? = {
        try? CLIPPromptStore()
    }()

    let entries: [Entry]

    init() throws {
        let installRoot = try ModelArchive.clip.installedURL()
        let promptsURL = installRoot.appendingPathComponent("clip-prompts.json")
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
