import CoreML
import Foundation

// MARK: - CoreML Generic Predictor Bridge
//
// Generic CoreML model loading + arbitrary tensor inference.
// Independent of FluidAudio SDK — wraps Apple's MLModel API directly.
// Gated behind COREML_PREDICTOR Swift compiler flag.

#if COREML_PREDICTOR

private class CoreMLPredictorHandle {
    let model: MLModel
    let inputName: String
    let outputName: String

    init(model: MLModel, inputName: String, outputName: String) {
        self.model = model
        self.inputName = inputName
        self.outputName = outputName
    }
}

@_cdecl("coreml_predictor_load")
func coreml_predictor_load(
    _ path: UnsafePointer<CChar>,
    _ computeUnits: Int32,
    _ inputName: UnsafePointer<CChar>,
    _ outputName: UnsafePointer<CChar>
) -> UnsafeMutableRawPointer? {
    let url = URL(fileURLWithPath: String(cString: path))
    let config = MLModelConfiguration()
    switch computeUnits {
    case 0: config.computeUnits = .cpuOnly
    case 1: config.computeUnits = .cpuAndNeuralEngine
    default: config.computeUnits = .all
    }
    guard let model = try? MLModel(contentsOf: url, configuration: config) else { return nil }
    let handle = CoreMLPredictorHandle(
        model: model,
        inputName: String(cString: inputName),
        outputName: String(cString: outputName)
    )
    return Unmanaged.passRetained(handle as AnyObject).toOpaque()
}

@_cdecl("coreml_predictor_predict")
func coreml_predictor_predict(
    _ handle: UnsafeMutableRawPointer,
    _ inputData: UnsafePointer<Float>,
    _ inputShape: UnsafePointer<Int32>,
    _ inputDims: Int32,
    _ outputData: UnsafeMutablePointer<Float>,
    _ outputCapacity: Int32
) -> Int32 {
    let predictorHandle = Unmanaged<AnyObject>.fromOpaque(handle).takeUnretainedValue() as! CoreMLPredictorHandle

    let dims = Int(inputDims)
    var shape = [NSNumber]()
    var totalElements = 1
    for i in 0..<dims {
        let dim = Int(inputShape[i])
        shape.append(NSNumber(value: dim))
        totalElements *= dim
    }

    let inputArray: MLMultiArray
    do {
        inputArray = try MLMultiArray(shape: shape, dataType: .float32)
    } catch {
        print("CoreML predictor: MLMultiArray creation failed: \(error)")
        return -1
    }

    // Copy Float32 input data into the MLMultiArray.
    // CoreML may internally convert to Float16 for models with FP16 precision.
    let ptr = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
    memcpy(ptr, inputData, totalElements * MemoryLayout<Float>.size)

    let provider: MLDictionaryFeatureProvider
    do {
        provider = try MLDictionaryFeatureProvider(dictionary: [
            predictorHandle.inputName: MLFeatureValue(multiArray: inputArray)
        ])
    } catch {
        print("CoreML predictor: feature provider creation failed: \(error)")
        return -2
    }

    let output: MLFeatureProvider
    do {
        output = try predictorHandle.model.prediction(from: provider)
    } catch {
        print("CoreML predictor: prediction failed: \(error)")
        return -3
    }

    guard let outputArray = output.featureValue(for: predictorHandle.outputName)?.multiArrayValue else {
        return -4
    }

    let outputCount = outputArray.count
    if outputCount > Int(outputCapacity) {
        print("CoreML predictor: output size \(outputCount) exceeds capacity \(outputCapacity)")
        return -5
    }

    // Handle both Float32 and Float16 output data types.
    // The caller always expects Float32, so convert if needed.
    if outputArray.dataType == .float32 {
        let outPtr = outputArray.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(outputData, outPtr, outputCount * MemoryLayout<Float>.size)
    } else if outputArray.dataType == .float16 {
        let f16Ptr = outputArray.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<outputCount {
            outputData[i] = Float(Float16(bitPattern: f16Ptr[i]))
        }
    } else {
        print("CoreML predictor: unsupported output data type: \(outputArray.dataType.rawValue)")
        return -6
    }
    return Int32(outputCount)
}

/// Predict with multiple named inputs. Each input is a flat f32 array with its own shape.
///
/// Layout in memory:
///   inputNames:  [ptr0, ptr1, ..., ptrN]    (C strings)
///   inputDatas:  [ptr0, ptr1, ..., ptrN]    (Float* pointers)
///   inputShapes: [ptr0, ptr1, ..., ptrN]    (Int32* pointers)
///   inputDims:   [dim0, dim1, ..., dimN]    (Int32 values)
@_cdecl("coreml_predictor_predict_multi")
func coreml_predictor_predict_multi(
    _ handle: UnsafeMutableRawPointer,
    _ numInputs: Int32,
    _ inputNames: UnsafePointer<UnsafePointer<CChar>>,
    _ inputDatas: UnsafePointer<UnsafePointer<Float>>,
    _ inputShapes: UnsafePointer<UnsafePointer<Int32>>,
    _ inputDims: UnsafePointer<Int32>,
    _ outputData: UnsafeMutablePointer<Float>,
    _ outputCapacity: Int32
) -> Int32 {
    let predictorHandle = Unmanaged<AnyObject>.fromOpaque(handle).takeUnretainedValue() as! CoreMLPredictorHandle
    let n = Int(numInputs)

    var featureDict = [String: MLFeatureValue]()
    for i in 0..<n {
        let name = String(cString: inputNames[i])
        let dims = Int(inputDims[i])
        var shape = [NSNumber]()
        var totalElements = 1
        for d in 0..<dims {
            let dim = Int(inputShapes[i][d])
            shape.append(NSNumber(value: dim))
            totalElements *= dim
        }
        let arr: MLMultiArray
        do {
            arr = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            print("CoreML predictor multi: MLMultiArray creation failed for '\(name)': \(error)")
            return -1
        }
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(ptr, inputDatas[i], totalElements * MemoryLayout<Float>.size)
        featureDict[name] = MLFeatureValue(multiArray: arr)
    }

    let provider: MLDictionaryFeatureProvider
    do {
        provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
    } catch {
        print("CoreML predictor multi: feature provider creation failed: \(error)")
        return -2
    }

    let output: MLFeatureProvider
    do {
        output = try predictorHandle.model.prediction(from: provider)
    } catch {
        print("CoreML predictor multi: prediction failed: \(error)")
        return -3
    }

    guard let outputArray = output.featureValue(for: predictorHandle.outputName)?.multiArrayValue else {
        return -4
    }

    let outputCount = outputArray.count
    if outputCount > Int(outputCapacity) {
        print("CoreML predictor multi: output size \(outputCount) exceeds capacity \(outputCapacity)")
        return -5
    }

    if outputArray.dataType == .float32 {
        let outPtr = outputArray.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(outputData, outPtr, outputCount * MemoryLayout<Float>.size)
    } else if outputArray.dataType == .float16 {
        let f16Ptr = outputArray.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<outputCount {
            outputData[i] = Float(Float16(bitPattern: f16Ptr[i]))
        }
    } else {
        print("CoreML predictor multi: unsupported output data type: \(outputArray.dataType.rawValue)")
        return -6
    }
    return Int32(outputCount)
}

@_cdecl("coreml_predictor_free")
func coreml_predictor_free(_ handle: UnsafeMutableRawPointer) {
    Unmanaged<AnyObject>.fromOpaque(handle).release()
}

#endif
