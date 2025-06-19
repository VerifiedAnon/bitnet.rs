//! Shared error types for the BitNet project.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitNetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization/Deserialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Safetensor error: {0}")]
    Safetensor(#[from] safetensors::SafeTensorError),

    #[error("Hugging Face Hub API error: {0}")]
    Api(#[from] hf_hub::api::sync::ApiError),

    #[error("Model configuration error: {0}")]
    Config(String),
} 