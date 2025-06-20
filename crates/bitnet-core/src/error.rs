use thiserror::Error;
use wgpu::RequestDeviceError;

/// The primary error type for all operations in the `bitnet-core` crate.
#[derive(Error, Debug)]
pub enum BitNetError {
    /// I/O error, typically from file or network operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/Deserialization error using bincode.
    #[error("Serialization/Deserialization error (bincode): {0}")]
    Bincode(#[from] bincode::error::DecodeError),

    /// Error when WGPU fails to request a device.
    #[error("WGPU request device error: {0}")]
    RequestDeviceError(#[from] RequestDeviceError),

    /// Error when no suitable WGPU adapter is found.
    #[error("No suitable WGPU adapter found")]
    NoSuitableAdapter,

    /// Configuration error, typically for invalid or missing settings.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Inference error, for issues during model inference.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Error when WGPU compute operation times out.
    #[error("WGPU compute operation timed out")]
    ComputeTimeout,

    /// Error when WGPU compute operation fails.
    #[error("WGPU compute operation failed")]
    ComputeError,

    /// Error when WGPU buffer mapping fails.
    #[error("WGPU buffer mapping failed")]
    BufferMapError,

    /// Error when WGPU shader compilation fails.
    #[error("WGPU shader compilation failed: {0}")]
    ShaderCompilationError(String),

    /// Error when WGPU pipeline creation fails.
    #[error("WGPU pipeline creation failed: {0}")]
    PipelineCreationError(String),

    /// Error when WGPU bind group creation fails.
    #[error("WGPU bind group creation failed: {0}")]
    BindGroupCreationError(String),

    /// Error when WGPU buffer creation fails.
    #[error("WGPU buffer creation failed: {0}")]
    BufferCreationError(String),

    /// Error when WGPU command buffer submission fails.
    #[error("WGPU command buffer submission failed: {0}")]
    CommandBufferSubmissionError(String),

    /// Error when matrix dimensions are incompatible.
    #[error("Matrix dimension error: {0}")]
    DimensionError(String),

    /// Error when weight values are invalid (not -1, 0, or +1).
    #[error("Invalid weight value: {0}. Must be -1, 0, or 1.")]
    InvalidWeightValue(i8),

    /// Error when a requested buffer size would exceed device limits.
    #[error("Requested buffer size ({0} bytes) exceeds device limits.")]
    BufferSizeExceeded(u64),
} 