#![doc(html_root_url = "https://docs.rs/bitnet-core")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![deny(rustdoc::missing_crate_level_docs)]

//! # BitNet Core
//! 
//! A high-performance BitNet inference engine written in pure Rust, supporting both CPU (SIMD) and GPU (WGSL) backends.
//! 
//! ## Overview
//! 
//! `bitnet-core` provides the core functionality for running BitNet models, with a focus on:
//! 
//! - High-performance inference using CPU SIMD and GPU compute
//! - Streaming-friendly model loading and execution
//! - Pure Rust implementation with no Python/C++ dependencies
//! - Comprehensive test coverage and validation
//! 
//! ## Features
//! 
//! - `gpu` - Enables GPU support via wgpu/WGSL (disabled by default)
//! - `core-gui` - Enables developer visualization tools (disabled by default)
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use bitnet_core::{model::Transformer, settings::InferenceSettings};
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a model
//! let model = Transformer::from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")?;
//! 
//! // Configure inference settings
//! let settings = InferenceSettings::default()
//!     .with_temperature(0.7)
//!     .with_top_p(0.9);
//! 
//! // Run inference
//! let input = "The quick brown fox";
//! let output = model.generate(input, &settings)?;
//! println!("Generated: {}", output);
//! # Ok(())
//! # }
//! ```
//! 
//! ## GPU Support
//! 
//! To enable GPU support, add the `gpu` feature:
//! 
//! ```toml
//! [dependencies]
//! bitnet-core = { version = "0.1", features = ["gpu"] }
//! ```
//! 
//! Then the model will automatically use GPU acceleration when available:
//! 
//! ```rust,no_run
//! # use bitnet_core::{model::Transformer, settings::InferenceSettings};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Transformer::from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")?;
//! // GPU will be used automatically if the feature is enabled and hardware is available
//! # Ok(())
//! # }
//! ```
//! 
//! ## Module Overview
//! 
//! - [`model`] - Core transformer model implementation
//! - [`attention`] - Multi-head attention with RoPE
//! - [`feed_forward`] - Feed-forward network with SwiGLU
//! - [`rms_norm`] - RMSNorm implementation
//! - [`bitnet_linear`] - Quantized linear layer
//! - [`kernels`] - CPU (SIMD) and GPU kernel implementations
//! - [`tokenizer`] - Text tokenization and chat templates
//! - [`settings`] - Inference and generation settings
//! - [`embedding`] - Token embedding layer
//! - [`error`] - Error types and handling
//! 
//! ## Performance Notes
//! 
//! - CPU backend uses SIMD acceleration when available (AVX2 on x86, NEON on ARM)
//! - GPU backend uses wgpu/WGSL for cross-platform compute
//! - Model weights are loaded per-block for efficient memory usage
//! - KV cache is managed automatically for faster generation
//! 
//! ## Safety
//! 
//! This crate uses `unsafe` code in the following places:
//! 
//! - SIMD intrinsics in CPU kernels (x86/ARM)
//! - Memory mapping for efficient model loading
//! - GPU memory management via wgpu
//! 
//! All unsafe code is thoroughly tested and validated against scalar reference implementations.
//! 
//! ## Examples
//! 
//! See the [examples directory](https://github.com/microsoft/BitNet/tree/main/examples) for more usage examples.

// Re-export commonly used types
pub use crate::model::Transformer;
pub use crate::settings::InferenceSettings;
pub use crate::error::BitNetError;

// Public modules
pub mod attention;
pub mod feed_forward;
pub mod model;
pub mod rms_norm;

/// Error types and handling for BitNet operations.
pub mod error;
pub mod wgpu_context;
pub mod kernels;
pub mod settings;
pub mod tokenizer;
pub mod bitnet_linear;
pub mod embedding;
pub mod rope;

#[cfg(feature = "core-gui")]
#[cfg_attr(docsrs, doc(cfg(feature = "core-gui")))]
pub mod visualization;

// Training module is a work in progress
#[cfg(feature = "training")]
#[cfg_attr(docsrs, doc(cfg(feature = "training")))]
pub mod training;

pub mod pipeline;
pub use pipeline::{Pipeline, PipelineOptions, PipelineResult, PipelineError};

pub mod bitnetcore_test_utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {
        // Basic test to ensure the crate compiles
        assert!(true);
    }
}
