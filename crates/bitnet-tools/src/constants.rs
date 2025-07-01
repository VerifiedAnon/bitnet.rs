/// Centralized constants for model management and file structure.
///
/// NOTE: The workspace root is determined by the presence of a `.workspace_root` marker file.
/// Place this file at the main project root to ensure all model files are created at the correct location.

pub const ORIGINAL_DIR: &str = "Original";
pub const CONVERTED_DIR: &str = "Converted"; 
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

pub fn models_root() -> PathBuf {
    workspace_root().join(MODELS_DIR)
}
pub fn original_dir() -> PathBuf {
    models_root().join(ORIGINAL_DIR)
}
pub fn converted_dir() -> PathBuf {
    models_root().join(CONVERTED_DIR)
}

/// Default Hugging Face model id for BitNet 1.58 2B4T (BF16 master weights).
pub const DEFAULT_MODEL_ID: &str = "microsoft/bitnet-b1.58-2B-4T-bf16";
pub const CONFIG_JSON: &str = "config.json";
pub const SAFETENSORS_EXT: &str = ".safetensors";
/// File name for the tokenizer model used by BitNet (Llama tokenizer).
pub const TOKENIZER_MODEL: &str = "tokenizer.model";
pub const SAFETENSORS_FILE: &str = "model.safetensors";
pub const PACKED_MODEL_FILE: &str = "bitnet_model_packed.bin";

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

