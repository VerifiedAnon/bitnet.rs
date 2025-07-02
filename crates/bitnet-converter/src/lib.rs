use std::path::PathBuf;
use serde::Deserialize;
use bincode::serde::encode_to_vec;
use bincode::config::standard;
use bitnet_tools::constants::{workspace_root, CONFIG_JSON, SAFETENSORS_FILE};
use std::time::Instant;
use rayon::prelude::*;
use crate::packer::BitNetModelRecord;
use safetensors::{tensor::TensorView, Dtype, serialize_to_file};
use half::bf16;
use std::collections::BTreeMap;

pub mod packer;
pub mod source;

#[derive(Deserialize)]
struct PartialConfig {
    num_hidden_layers: usize,
}

/// Write the full model as a single safetensors file (model.safetensors)
pub fn write_single_file_model_safetensors(model: &BitNetModelRecord, output_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let single_file_path = output_path.join("model.safetensors");
    // Ensure parent directory exists
    if let Some(parent) = single_file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tensors = BTreeMap::new();
    // Helper to convert Vec<f32> to Vec<u8> as bf16
    fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
        let bf16_vec: Vec<bf16> = data.iter().map(|&f| bf16::from_f32(f)).collect();
        let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
        bytemuck::cast_slice(&u16_vec).to_vec()
    }
    // Embedding
    let embedding_bytes = f32_to_bf16_bytes(&model.embedding.weight);
    tensors.insert("embedding.weight".to_string(), TensorView::new(Dtype::BF16, model.embedding.shape.clone(), &embedding_bytes)?);
    // Norm
    let norm_bytes = f32_to_bf16_bytes(&model.norm.weight);
    tensors.insert("norm.weight".to_string(), TensorView::new(Dtype::BF16, model.norm.shape.clone(), &norm_bytes)?);
    // LM Head
    let lm_head_bytes = f32_to_bf16_bytes(&model.lm_head.weight);
    tensors.insert("lm_head.weight".to_string(), TensorView::new(Dtype::BF16, model.lm_head.shape.clone(), &lm_head_bytes)?);
    // Blocks
    let mut block_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    for (i, block) in model.blocks.iter().enumerate() {
        let prefix = format!("block_{}.", i);
        let attn_norm_bytes = f32_to_bf16_bytes(&block.attention_norm.weight);
        let ffn_norm_bytes = f32_to_bf16_bytes(&block.ffn_norm.weight);
        block_bytes.push((format!("{}attention_norm.weight", prefix), attn_norm_bytes, block.attention_norm.shape.clone()));
        block_bytes.push((format!("{}ffn_norm.weight", prefix), ffn_norm_bytes, block.ffn_norm.shape.clone()));
        // Add more tensors as needed
    }
    for (name, bytes, shape) in &block_bytes {
        tensors.insert(name.clone(), TensorView::new(Dtype::BF16, shape.clone(), bytes)?);
    }
    serialize_to_file(&tensors, &None, &single_file_path)?;
    println!("[CONVERT] Wrote single-file model to: {}", single_file_path.display());
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
    println!("[CONVERT] Total conversion time: {:.2?}", start.elapsed());
    Ok(output_path)
} 