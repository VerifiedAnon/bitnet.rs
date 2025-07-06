#![deny(missing_docs)]
//! BitNet Converter: Convert Hugging Face safetensors to BitNet's optimized, quantized format.
//!
//! This crate provides a robust, streaming-friendly Rust tool for converting standard model weights (e.g., Hugging Face safetensors) into the optimized, quantized format required by the BitNet engine.
//!
//! # Features
//! - Burn-free, pure Rust conversion
//! - Streaming output (per-block and single-file)
//! - Parallelized for speed
//! - Robust loader and error handling
//! - Comprehensive test coverage
//!
//! # Usage
//! Add `bitnet-converter` as a dependency and use the provided API or CLI for model conversion.

use std::path::PathBuf;
use serde::Deserialize;
use bincode::serde::encode_to_vec;
use bincode::config::standard;
use bitnet_tools::constants::{workspace_root, CONFIG_JSON, SAFETENSORS_FILE, EMBEDDING_KEY, NORM_KEY, LM_HEAD_KEY};
use std::time::Instant;
use rayon::prelude::*;
use crate::packer::BitNetModelRecord;
use safetensors::{tensor::TensorView, Dtype, serialize_to_file};
use half::bf16;
use std::collections::BTreeMap;

/// Model quantization, packing, and serialization utilities.
pub mod packer;
/// Robust safetensors loader and tensor source utilities.
pub mod source;

#[derive(Deserialize)]
struct PartialConfig {
    num_hidden_layers: usize,
}

/// Write the full model as a single safetensors file (model.safetensors), including all quantized weights and scales
pub fn write_single_file_model_safetensors(model: &BitNetModelRecord, output_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let single_file_path = output_path.join("model.safetensors");
    // Ensure parent directory exists
    if let Some(parent) = single_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Use the robust quantized safetensors export (writes all weights/scales)
    crate::packer::export_quantized_to_safetensors(model, &single_file_path)?;
    println!("[CONVERT] Wrote quantized single-file model to: {}", single_file_path.display());
    Ok(())
}

/// Programmatic API to run the BitNet conversion pipeline.
/// Returns the path to the output packed model file on success.
pub fn convert_model_on_disk(input_dir: &str, output_dir: &str, write_streamable: bool) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let input_path = workspace_root().join(input_dir);
    let output_path = workspace_root().join(output_dir);
    std::fs::create_dir_all(&output_path)?;
    println!("[CONVERT] Loading config from: {}", input_path.join(CONFIG_JSON).display());
    let t0 = Instant::now();
    let config_str = std::fs::read_to_string(input_path.join(CONFIG_JSON))?;
    let config: PartialConfig = serde_json::from_str(&config_str)?;
    println!("[CONVERT] Loaded config in {:.2?}", t0.elapsed());
    println!("[CONVERT] Loading safetensors from: {}", input_path.join(SAFETENSORS_FILE).display());
    let t1 = Instant::now();
    let source = source::ModelSource::SafetensorsFile(
        input_path.join(SAFETENSORS_FILE).to_str().unwrap().to_string(),
    );
    let tensor_map = source.load_tensors()?;
    println!("[CONVERT] Loaded safetensors in {:.2?}", t1.elapsed());
    println!("[CONVERT] Packing weights...");
    let t2 = Instant::now();
    let record = packer::convert_model(tensor_map, config.num_hidden_layers, true)?;
    println!("[CONVERT] Packed weights in {:.2?}", t2.elapsed());
    println!("[CONVERT] Writing model as per-block files to: {}", output_path.display());
    let t3 = Instant::now();
    // Save embedding
    let embedding_path = output_path.join("embedding.bin");
    if let Some(parent) = embedding_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let embedding_bytes = encode_to_vec(&record.embedding, standard())?;
    std::fs::write(&embedding_path, &embedding_bytes)?;
    // Save norm
    let norm_path = output_path.join("norm.bin");
    if let Some(parent) = norm_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let norm_bytes = encode_to_vec(&record.norm, standard())?;
    std::fs::write(&norm_path, &norm_bytes)?;
    // Save lm_head
    let lm_head_path = output_path.join("lm_head.bin");
    if let Some(parent) = lm_head_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let lm_head_bytes = encode_to_vec(&record.lm_head, standard())?;
    std::fs::write(&lm_head_path, &lm_head_bytes)?;
    // --- Always write single-file model as safetensors ---
    write_single_file_model_safetensors(&record, &output_path)?;
    // --- Optionally write per-block streaming output ---
    if write_streamable {
        record.blocks.par_iter().enumerate().for_each(|(i, block)| {
            let block_path = output_path.join(format!("block_{}.bin", i));
            if let Some(parent) = block_path.parent() {
                std::fs::create_dir_all(parent).expect("Failed to create block file parent directory");
            }
            let block_bytes = encode_to_vec(block, standard()).expect("Failed to encode block");
            std::fs::write(&block_path, &block_bytes).expect("Failed to write block file");
        });
        println!("[CONVERT] Wrote all model parts in {:.2?}", t3.elapsed());
    }
    // --- Write a future-proof config for the packed model ---
    let orig_config_path = input_path.join(CONFIG_JSON);
    let packed_config_path = output_path.join("bitnet_packed_config.json");
    let mut packed_config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&orig_config_path)?)?;
    packed_config["bitnet_packed"] = serde_json::Value::Bool(true);
    packed_config["_note"] = serde_json::Value::String("All packed tensor shapes are as per BitNet's quantization. QKV: [3*hidden_size, hidden_size/16], etc. Use this config for inference with the packed model.".to_string());
    // --- Robust fix: always set hidden_size and vocab_size to match packed model ---
    packed_config["hidden_size"] = serde_json::Value::from(record.embedding.shape[1]);
    packed_config["vocab_size"] = serde_json::Value::from(record.embedding.shape[0]);
    std::fs::write(&packed_config_path, serde_json::to_string_pretty(&packed_config)?)?;
    println!("[CONVERT] Wrote packed model config to: {}", packed_config_path.display());
    println!("[CONVERT] Total conversion time: {:.2?}", start.elapsed());
    Ok(output_path)
} 