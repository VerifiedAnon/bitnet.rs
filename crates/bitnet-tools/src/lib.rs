//! Shared tools for BitNet, including Hugging Face model download utilities.

mod error;
pub use error::BitNetError;

pub mod combine;
pub mod hf_loader;
// pub use hf_loader::download_model_and_tokenizer; // removed, use get_model instead
pub mod constants;
pub mod test_utils;  // New module for test utilities
