//! Error types for BitNet tools and utilities.
//!
//! Defines the BitNetError enum for common error cases across the BitNet toolchain.

use thiserror::Error;

/// Common error type for BitNet tools and utilities.
#[derive(Error, Debug)]
pub enum BitNetError {
    /// I/O error, typically from file or network operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/Deserialization error using serde.
    #[error("Serialization/Deserialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Error from safetensors library operations.
    #[error("Safetensor error: {0}")]
    Safetensor(#[from] safetensors::SafeTensorError),

    /// Error from Hugging Face Hub API operations.
    #[error("Hugging Face Hub API error: {0}")]
    Api(#[from] hf_hub::api::sync::ApiError),

    /// Model configuration error, such as invalid or missing settings.
    #[error("Model configuration error: {0}")]
    Config(String),
} 