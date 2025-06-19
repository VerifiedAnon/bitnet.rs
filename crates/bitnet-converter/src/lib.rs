use std::path::PathBuf;
use serde::Deserialize;
use std::fs::File;
use std::io::Write;
use bincode::serde::encode_to_vec;
use bincode::config::standard;
use bitnet_tools::constants::{workspace_root, CONFIG_JSON, SAFETENSORS_FILE, PACKED_MODEL_FILE};
use std::time::Instant;
use rayon::prelude::*;

pub mod packer;
pub mod source;

#[derive(Deserialize)]
struct PartialConfig {
    num_hidden_layers: usize,
}

/// Programmatic API to run the BitNet conversion pipeline.
/// Returns the path to the output packed model file on success.
pub fn convert_model_on_disk(input_dir: &str, output_dir: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
    let embedding_bytes = encode_to_vec(&record.embedding, standard())?;
    std::fs::write(&embedding_path, &embedding_bytes)?;
    // Save norm
    let norm_path = output_path.join("norm.bin");
    let norm_bytes = encode_to_vec(&record.norm, standard())?;
    std::fs::write(&norm_path, &norm_bytes)?;
    // Save lm_head
    let lm_head_path = output_path.join("lm_head.bin");
    let lm_head_bytes = encode_to_vec(&record.lm_head, standard())?;
    std::fs::write(&lm_head_path, &lm_head_bytes)?;
    // Save each block in parallel
    record.blocks.par_iter().enumerate().for_each(|(i, block)| {
        let block_path = output_path.join(format!("block_{}.bin", i));
        let block_bytes = encode_to_vec(block, standard()).expect("Failed to encode block");
        std::fs::write(&block_path, &block_bytes).expect("Failed to write block file");
    });
    println!("[CONVERT] Wrote all model parts in {:.2?}", t3.elapsed());
    println!("[CONVERT] Total conversion time: {:.2?}", start.elapsed());
    Ok(output_path)
} 