import Foundation
import AVFoundation
import FluidAudio
import Darwin

// MARK: - Bridge Class

/// Internal bridge class that wraps FluidAudio
class FluidAudioBridgeInternal {
    private var asrManager: AsrManager?
    private var asrModels: AsrModels?
    private var vadManager: VadManager?

    init() {}

    func initializeAsr() throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let models = try await AsrModels.downloadAndLoad()
                self.asrModels = models

                let manager = AsrManager()
                try await manager.initialize(models: models)
                self.asrManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func transcribeFile(_ path: String) throws -> (String, Float, Double, Double, Float) {
        guard let manager = asrManager else {
            throw BridgeError.notInitialized
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: ASRResult?
        var transcribeError: Error?

        Task {
            do {
                let url = URL(fileURLWithPath: path)
                result = try await manager.transcribe(url)
            } catch {
                transcribeError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = transcribeError {
            throw error
        }

        guard let r = result else {
            throw BridgeError.noResult
        }

        return (r.text, r.confidence, r.duration, r.processingTime, r.rtfx)
    }

    func isAsrAvailable() -> Bool {
        return asrManager != nil
    }

    func initializeVad(_ threshold: Float) throws {
        let semaphore = DispatchSemaphore(value: 0)
        var initError: Error?

        Task {
            do {
                let config = VadConfig(defaultThreshold: threshold)
                let manager = try await VadManager(config: config)
                self.vadManager = manager
            } catch {
                initError = error
            }
            semaphore.signal()
        }

        semaphore.wait()

        if let error = initError {
            throw error
        }
    }

    func isVadAvailable() -> Bool {
        return vadManager != nil
    }

    func cleanup() {
        asrManager = nil
        asrModels = nil
        vadManager = nil
    }
}

enum BridgeError: Error {
    case notInitialized
    case noResult
}

// MARK: - C FFI Functions

/// Storage for bridge instances (simple approach - use a single global for now)
private var globalBridge: FluidAudioBridgeInternal?

@_cdecl("fluidaudio_bridge_create")
public func fluidaudio_bridge_create() -> UnsafeMutableRawPointer? {
    let bridge = FluidAudioBridgeInternal()
    globalBridge = bridge
    return Unmanaged.passRetained(bridge).toOpaque()
}

@_cdecl("fluidaudio_bridge_destroy")
public func fluidaudio_bridge_destroy(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeRetainedValue()
    bridge.cleanup()
    if globalBridge === bridge {
        globalBridge = nil
    }
}

@_cdecl("fluidaudio_initialize_asr")
public func fluidaudio_initialize_asr(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeAsr()
        return 0
    } catch {
        print("ASR init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_transcribe_file")
public func fluidaudio_transcribe_file(
    _ ptr: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ outText: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    _ outConfidence: UnsafeMutablePointer<Float>?,
    _ outDuration: UnsafeMutablePointer<Double>?,
    _ outProcessingTime: UnsafeMutablePointer<Double>?,
    _ outRtfx: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let ptr = ptr, let path = path else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let pathString = String(cString: path)

    do {
        let (text, confidence, duration, processingTime, rtfx) = try bridge.transcribeFile(pathString)

        // Allocate and copy text
        if let outText = outText {
            let cString = strdup(text)
            outText.pointee = cString
        }

        outConfidence?.pointee = confidence
        outDuration?.pointee = duration
        outProcessingTime?.pointee = processingTime
        outRtfx?.pointee = rtfx

        return 0
    } catch {
        print("Transcribe error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_asr_available")
public func fluidaudio_is_asr_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isAsrAvailable() ? 1 : 0
}

@_cdecl("fluidaudio_initialize_vad")
public func fluidaudio_initialize_vad(_ ptr: UnsafeMutableRawPointer?, _ threshold: Float) -> Int32 {
    guard let ptr = ptr else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    do {
        try bridge.initializeVad(threshold)
        return 0
    } catch {
        print("VAD init error: \(error)")
        return -1
    }
}

@_cdecl("fluidaudio_is_vad_available")
public func fluidaudio_is_vad_available(_ ptr: UnsafeMutableRawPointer?) -> Int32 {
    guard let ptr = ptr else { return 0 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    return bridge.isVadAvailable() ? 1 : 0
}

@_cdecl("fluidaudio_get_platform")
public func fluidaudio_get_platform(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    #if os(macOS)
    let platform = "macOS"
    #elseif os(iOS)
    let platform = "iOS"
    #else
    let platform = "unknown"
    #endif

    out?.pointee = strdup(platform)
}

@_cdecl("fluidaudio_get_chip_name")
public func fluidaudio_get_chip_name(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) {
    var size: size_t = 0
    var chipName = "Unknown"

    if sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 {
        var buffer = [CChar](repeating: 0, count: Int(size))
        if sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 {
            chipName = String(cString: buffer)
        }
    }

    out?.pointee = strdup(chipName)
}

@_cdecl("fluidaudio_get_memory_gb")
public func fluidaudio_get_memory_gb() -> Double {
    return Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
}

@_cdecl("fluidaudio_is_apple_silicon")
public func fluidaudio_is_apple_silicon() -> Int32 {
    return SystemInfo.isAppleSilicon ? 1 : 0
}

@_cdecl("fluidaudio_cleanup")
public func fluidaudio_cleanup(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr = ptr else { return }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    bridge.cleanup()
}

@_cdecl("fluidaudio_free_string")
public func fluidaudio_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}
