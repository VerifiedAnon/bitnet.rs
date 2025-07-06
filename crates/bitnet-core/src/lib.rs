#![doc(html_root_url = "https://docs.rs/bitnet-core")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![deny(rustdoc::missing_crate_level_docs)]

//! # BitNet Core
//! 
//! A high-performance BitNet inference engine written in pure Rust, supporting both CPU (SIMD) and GPU (WGSL) backends.

// Re-export commonly used types
pub use crate::model::Transformer;
pub use crate::settings::InferenceSettings;
pub use crate::error::BitNetError;

// Public modules
pub mod attention;
pub mod bitnet_linear;
pub mod embedding;
/// Error types and handling for BitNet core operations.
pub mod error;
pub mod feed_forward;
pub mod kernels;
pub mod model;
pub mod pipeline;
pub mod rms_norm;
pub mod rope;
pub mod settings;
pub mod tokenizer;
pub mod wgpu_context;
pub mod bitnetcore_test_utils;

#[cfg(feature = "core-gui")]
#[cfg_attr(docsrs, doc(cfg(feature = "core-gui")))]
pub mod visualization;

// Training module is a work in progress
#[cfg(feature = "training")]
#[cfg_attr(docsrs, doc(cfg(feature = "training")))]
pub mod training;

pub use pipeline::{Pipeline, PipelineOptions, PipelineResult, PipelineError};

#[cfg(test)]
mod tests {
    #[test]
    fn it_compiles() {
        assert!(true);
    }
}