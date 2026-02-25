//! # fluidaudio-rs
//!
//! Rust bindings for [FluidAudio](https://github.com/FluidInference/FluidAudio) -
//! a Swift library for ASR, VAD, Speaker Diarization, and TTS on Apple platforms.
//!
//! ## Features
//!
//! - **ASR (Automatic Speech Recognition)** - High-quality speech-to-text using Parakeet TDT models
//! - **VAD (Voice Activity Detection)** - Detect speech segments in audio
//!
//! ## Requirements
//!
//! - macOS 14+ or iOS 17+
//! - Apple Silicon (M1/M2/M3) recommended
//!
//! ## Example
//!
//! ```rust,no_run
//! use fluidaudio_rs::FluidAudio;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = FluidAudio::new()?;
//!
//!     // Transcribe an audio file
//!     audio.init_asr()?;
//!     let result = audio.transcribe_file("audio.wav")?;
//!     println!("Text: {}", result.text);
//!     println!("Confidence: {:.2}%", result.confidence * 100.0);
//!
//!     Ok(())
//! }
//! ```

mod ffi;

use std::path::Path;
use thiserror::Error;

// Re-export FFI types
pub use ffi::{AsrResult, SystemInfo};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenTiming {
    pub token: String,
    #[serde(rename = "tokenId")]
    pub token_id: i32,
    #[serde(rename = "startTime")]
    pub start_time: f64,
    #[serde(rename = "endTime")]
    pub end_time: f64,
    pub confidence: f32,
}

pub struct AsrResultWithTimings {
    pub text: String,
    pub confidence: f32,
    pub duration: f64,
    pub processing_time: f64,
    pub rtfx: f32,
    pub token_timings: Vec<TokenTiming>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DiarizationSegment {
    #[serde(rename = "speakerId")]
    pub speaker_id: String,
    #[serde(rename = "startTimeSeconds")]
    pub start_time: f64,
    #[serde(rename = "endTimeSeconds")]
    pub end_time: f64,
}

/// Errors that can occur when using FluidAudio
#[derive(Error, Debug)]
pub enum FluidAudioError {
    #[error("FluidAudio not initialized: {0}")]
    NotInitialized(String),

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Audio file not found: {0}")]
    FileNotFound(String),

    #[error("Swift bridge error: {0}")]
    BridgeError(String),
}

impl From<String> for FluidAudioError {
    fn from(s: String) -> Self {
        FluidAudioError::BridgeError(s)
    }
}

// SAFETY: FluidAudio wraps a Swift bridge pointer that is only accessed
// from one thread at a time. The ML worker takes exclusive ownership via
// take()/put-back pattern — no concurrent access occurs.
unsafe impl Send for FluidAudio {}

/// Main FluidAudio interface for Rust
///
/// Provides access to ASR and VAD functionality.
pub struct FluidAudio {
    bridge: ffi::FluidAudioBridge,
}

impl FluidAudio {
    /// Create a new FluidAudio instance
    pub fn new() -> Result<Self, FluidAudioError> {
        let bridge = ffi::FluidAudioBridge::new()
            .ok_or_else(|| FluidAudioError::BridgeError("Failed to create bridge".to_string()))?;
        Ok(Self { bridge })
    }

    // ========== ASR Methods ==========

    /// Initialize the ASR (Automatic Speech Recognition) engine
    ///
    /// This downloads and loads the ASR models. First run may take 20-30 seconds
    /// as models are compiled for the Neural Engine.
    pub fn init_asr(&self) -> Result<(), FluidAudioError> {
        self.bridge.initialize_asr().map_err(FluidAudioError::from)
    }

    /// Initialize ASR from a local directory containing CoreML model bundles.
    ///
    /// The directory must contain `Preprocessor.mlmodelc/`, `Encoder.mlmodelc/`,
    /// `Decoder.mlmodelc/`, `JointDecision.mlmodelc/`, and `parakeet_vocab.json`.
    pub fn init_asr_from_directory(
        &self,
        path: &Path,
        version: u32,
    ) -> Result<(), FluidAudioError> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_ref())
            .map_err(|_| FluidAudioError::BridgeError("Invalid path".to_string()))?;
        self.bridge
            .initialize_asr_from_directory(&c_path, version as i32)
            .map_err(FluidAudioError::from)
    }

    /// Transcribe an audio file, returning text with word-level token timings.
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<AsrResultWithTimings, FluidAudioError> {
        let path_str = path.as_ref().to_string_lossy();

        if !path.as_ref().exists() {
            return Err(FluidAudioError::FileNotFound(path_str.to_string()));
        }

        let raw = self
            .bridge
            .transcribe_file(&path_str)
            .map_err(FluidAudioError::from)?;

        parse_result_with_timings(raw)
    }

    /// Transcribe raw f32 PCM audio samples, returning text with word-level
    /// token timings.
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AsrResultWithTimings, FluidAudioError> {
        let raw = self
            .bridge
            .transcribe_samples(samples, sample_rate)
            .map_err(FluidAudioError::from)?;

        parse_result_with_timings(raw)
    }

    /// Check if ASR is initialized and ready
    pub fn is_asr_available(&self) -> bool {
        self.bridge.is_asr_available()
    }

    // ========== VAD Methods ==========

    /// Initialize the VAD (Voice Activity Detection) engine
    ///
    /// # Arguments
    /// * `threshold` - Detection threshold (0.0-1.0, default 0.85)
    pub fn init_vad(&self, threshold: f32) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_vad(threshold)
            .map_err(FluidAudioError::from)
    }

    /// Check if VAD is initialized and ready
    pub fn is_vad_available(&self) -> bool {
        self.bridge.is_vad_available()
    }

    // ========== Diarization Methods ==========

    /// Initialize the diarization engine
    ///
    /// Downloads and loads diarization models. First run may take time
    /// as models are downloaded from HuggingFace and compiled.
    pub fn init_diarizer(&self) -> Result<(), FluidAudioError> {
        self.bridge
            .initialize_diarizer()
            .map_err(FluidAudioError::from)
    }

    /// Diarize raw f32 PCM audio samples (16kHz mono)
    ///
    /// Returns a list of segments, each with a speaker ID and time range.
    pub fn diarize_samples(
        &self,
        samples: &[f32],
    ) -> Result<Vec<DiarizationSegment>, FluidAudioError> {
        let json = self
            .bridge
            .diarize_samples(samples)
            .map_err(FluidAudioError::from)?;

        if json.is_empty() {
            return Ok(Vec::new());
        }

        serde_json::from_str(&json).map_err(|e| {
            FluidAudioError::ProcessingFailed(format!("failed to parse diarization JSON: {e}"))
        })
    }

    /// Check if diarizer is initialized and ready
    pub fn is_diarizer_available(&self) -> bool {
        self.bridge.is_diarizer_available()
    }

    // ========== Model Loading Methods ==========

    /// Initialize ASR from a local model directory
    ///
    /// # Arguments
    /// * `path` - Path to directory containing CoreML model bundles and vocab
    /// * `version` - Model version (2 for v2, 3 for v3)
    pub fn init_asr_from_directory(
        &self,
        path: &Path,
        version: u32,
    ) -> Result<(), FluidAudioError> {
        let path_str = path.to_string_lossy();
        self.bridge
            .initialize_asr_from_directory(&path_str, version as i32)
            .map_err(FluidAudioError::from)
    }

    // ========== System Info ==========

    /// Get system information
    pub fn system_info(&self) -> SystemInfo {
        self.bridge.system_info()
    }

    /// Check if running on Apple Silicon
    pub fn is_apple_silicon(&self) -> bool {
        self.bridge.is_apple_silicon()
    }

    // ========== Cleanup ==========

    /// Release all resources
    pub fn cleanup(&self) {
        self.bridge.cleanup()
    }
}

fn parse_result_with_timings(
    raw: ffi::AsrResultWithTimings,
) -> Result<AsrResultWithTimings, FluidAudioError> {
    let token_timings: Vec<TokenTiming> = if raw.tokens_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&raw.tokens_json).map_err(|e| {
            FluidAudioError::ProcessingFailed(format!("failed to parse token timings JSON: {e}"))
        })?
    };

    Ok(AsrResultWithTimings {
        text: raw.text,
        confidence: raw.confidence,
        duration: raw.duration,
        processing_time: raw.processing_time,
        rtfx: raw.rtfx,
        token_timings,
    })
}

impl Drop for FluidAudio {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_instance() {
        // Note: This test will fail until Swift bridge is properly linked
        // For now, just test the types exist
        let _ = FluidAudioError::NotInitialized("test".to_string());
    }

    #[test]
    fn test_parse_token_timings_json() {
        let json = r#"[
            {"token": "Hello", "tokenId": 42, "startTime": 0.08, "endTime": 0.32, "confidence": 0.95},
            {"token": "world", "tokenId": 99, "startTime": 0.40, "endTime": 0.72, "confidence": 0.91}
        ]"#;
        let timings: Vec<TokenTiming> = serde_json::from_str(json).unwrap();
        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].token, "Hello");
        assert_eq!(timings[0].token_id, 42);
        assert!((timings[0].start_time - 0.08).abs() < 0.001);
        assert!((timings[0].end_time - 0.32).abs() < 0.001);
        assert!((timings[0].confidence - 0.95).abs() < 0.001);
        assert_eq!(timings[1].token, "world");
        assert_eq!(timings[1].token_id, 99);
    }

    #[test]
    fn test_parse_diarization_segments_json() {
        let json = r#"[
            {"speakerId": "speaker_0", "startTimeSeconds": 0.0, "endTimeSeconds": 2.5},
            {"speakerId": "speaker_1", "startTimeSeconds": 2.8, "endTimeSeconds": 5.1}
        ]"#;
        let segments: Vec<DiarizationSegment> = serde_json::from_str(json).unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker_id, "speaker_0");
        assert!((segments[0].start_time - 0.0).abs() < 0.001);
        assert!((segments[0].end_time - 2.5).abs() < 0.001);
        assert_eq!(segments[1].speaker_id, "speaker_1");
        assert!((segments[1].start_time - 2.8).abs() < 0.001);
        assert!((segments[1].end_time - 5.1).abs() < 0.001);
    }

    #[test]
    fn test_parse_empty_token_timings() {
        let raw = ffi::AsrResultWithTimings {
            text: "test".to_string(),
            confidence: 0.9,
            duration: 1.0,
            processing_time: 0.5,
            rtfx: 2.0,
            tokens_json: String::new(),
        };
        let result = parse_result_with_timings(raw).unwrap();
        assert!(result.token_timings.is_empty());
        assert_eq!(result.text, "test");
    }
}
