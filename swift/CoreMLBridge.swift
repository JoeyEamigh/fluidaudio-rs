import CoreML
import Foundation

// MARK: - CoreML Embedding Bridge
//
// Standalone CoreML model loading + speaker embedding inference.
// Independent of FluidAudio SDK — wraps Apple's MLModel API directly.
// Gated behind COREML_EMBEDDING Swift compiler flag.

#if COREML_EMBEDDING

@_cdecl("coreml_load_model")
func coreml_load_model(_ path: UnsafePointer<CChar>, _ computeUnits: Int32) -> UnsafeMutableRawPointer? {
    let url = URL(fileURLWithPath: String(cString: path))
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits == 1 ? .all : .cpuOnly
    guard let model = try? MLModel(contentsOf: url, configuration: config) else { return nil }
    return Unmanaged.passRetained(model as AnyObject).toOpaque()
}

@_cdecl("coreml_predict_embedding")
func coreml_predict_embedding(
    _ model: UnsafeMutableRawPointer,
    _ feats: UnsafePointer<Float>,
    _ numFrames: Int32,
    _ featDim: Int32,
    _ outEmbedding: UnsafeMutablePointer<Float>,
    _ embeddingDim: Int32
) -> Int32 {
    let mlModel = Unmanaged<AnyObject>.fromOpaque(model).takeUnretainedValue() as! MLModel

    let shape = [1 as NSNumber, NSNumber(value: numFrames), NSNumber(value: featDim)]

    let inputArray: MLMultiArray
    do {
        inputArray = try MLMultiArray(shape: shape, dataType: .float32)
    } catch {
        print("CoreML embedding: MLMultiArray creation failed: \(error)")
        return -1
    }

    let ptr = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
    memcpy(ptr, feats, Int(numFrames) * Int(featDim) * MemoryLayout<Float>.size)

    let provider: MLDictionaryFeatureProvider
    do {
        provider = try MLDictionaryFeatureProvider(dictionary: [
            "feats": MLFeatureValue(multiArray: inputArray)
        ])
    } catch {
        print("CoreML embedding: feature provider creation failed: \(error)")
        return -2
    }

    let output: MLFeatureProvider
    do {
        output = try mlModel.prediction(from: provider)
    } catch {
        print("CoreML embedding: prediction failed (numFrames=\(numFrames), featDim=\(featDim)): \(error)")
        return -3
    }

    guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else { return -4 }

    let embPtr = embedding.dataPointer.assumingMemoryBound(to: Float.self)
    memcpy(outEmbedding, embPtr, Int(embeddingDim) * MemoryLayout<Float>.size)
    return 0
}

@_cdecl("coreml_free_model")
func coreml_free_model(_ model: UnsafeMutableRawPointer) {
    Unmanaged<AnyObject>.fromOpaque(model).release()
}

#endif
