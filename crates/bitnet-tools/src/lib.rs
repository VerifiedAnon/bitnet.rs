#![deny(missing_docs)]
//! Essential utilities for BitNet projects: file combining, model download, logging, and more.
//!
//! This crate provides shared tools and utilities for all BitNet-related crates and binaries.
//!
//! - Model download and management (Hugging Face integration)
//! - File combining and manipulation
//! - Test reporting and logging utilities
//! - Centralized constants and error types
//!
//! # Usage
//!
//! Add `bitnet-tools` as a dependency in your Cargo.toml and use the provided modules.

mod error;
pub use error::BitNetError;

/// File combining and manipulation utilities.
pub mod combine;
/// Hugging Face model download and management utilities.
pub mod hf_loader;
// pub use hf_loader::download_model_and_tokenizer; // removed, use get_model instead
/// Centralized constants for file paths, model names, and more.
pub mod constants;
/// Test reporting and logging utilities for test harnesses.
pub mod test_utils;
/// Unified logging initialization and utilities for all BitNet crates.
pub mod logging;
