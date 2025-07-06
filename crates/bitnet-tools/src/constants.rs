//! Centralized constants for file paths, model names, and configuration in BitNet projects.
//!
//! This module provides shared constants and helper functions for managing file and directory paths, model IDs, and configuration file names.
/// Directory name for original model files.
pub const ORIGINAL_DIR: &str = "Original";
/// Directory name for converted model files.
pub const CONVERTED_DIR: &str = "Converted";
/// Directory name for models.
pub const MODELS_DIR: &str = "models";
use std::path::{PathBuf};
use std::env;

/// Returns the absolute path to the workspace root (where .workspace_root is found).
/// This enforces a single canonical root, even if there are nested workspaces.
pub fn workspace_root() -> PathBuf {
    let mut dir = env::current_dir().expect("Failed to get current dir");
    loop {
        if dir.join(".workspace_root").exists() {
            return dir;
        }
        if !dir.pop() {
            panic!("Could not find workspace root marker (.workspace_root)");
        }
    }
}

/// Returns the root directory for models.
pub fn models_root() -> PathBuf {
    workspace_root().join(MODELS_DIR)
}
/// Returns the directory for original model files.
pub fn original_dir() -> PathBuf {
    models_root().join(ORIGINAL_DIR)
}
/// Returns the directory for converted model files.
pub fn converted_dir() -> PathBuf {
    models_root().join(CONVERTED_DIR)
}

/// Default Hugging Face model id for BitNet 1.58 2B4T (BF16 master weights).
pub const DEFAULT_MODEL_ID: &str = "microsoft/bitnet-b1.58-2B-4T-bf16";
/// File name for the model configuration JSON file.
pub const CONFIG_JSON: &str = "config.json";
/// File extension for safetensors files.
pub const SAFETENSORS_EXT: &str = ".safetensors";
/// File name for the tokenizer model used by BitNet (Llama tokenizer).
pub const TOKENIZER_MODEL: &str = "tokenizer.model";
/// File name for the main safetensors model file.
pub const SAFETENSORS_FILE: &str = "model.safetensors";
/// File name for the packed model file.
pub const PACKED_MODEL_FILE: &str = "bitnet_model_packed.bin";

/// Tensor key for the embedding weights in BitNet single-file safetensors.
pub const EMBEDDING_KEY: &str = "embedding.weight";
/// Tensor key for the final normalization weights in BitNet single-file safetensors.
pub const NORM_KEY: &str = "norm.weight";
/// Tensor key for the output projection (LM head) weights in BitNet single-file safetensors.
pub const LM_HEAD_KEY: &str = "lm_head.weight";

/// Format string for per-block attention norm weights in single-file safetensors.
pub const BLOCK_ATTENTION_NORM_KEY: &str = "block_{}.attention_norm.weight";
/// Format string for per-block feedforward norm weights in single-file safetensors.
pub const BLOCK_FFN_NORM_KEY: &str = "block_{}.ffn_norm.weight";

/// List of all required files for a complete BitNet model directory.
pub const REQUIRED_MODEL_FILES: &[&str] = &[
    CONFIG_JSON,
    SAFETENSORS_FILE, // fallback for single-shard
    "model-00001-of-00002.safetensors", // multi-shard (add more if needed)
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    TOKENIZER_MODEL,
];

