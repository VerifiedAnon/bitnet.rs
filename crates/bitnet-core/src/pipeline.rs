//! BitNet Pipeline: Orchestrates model download, conversion, loading, and inference in a robust, async fashion.

use std::path::PathBuf;
use bitnet_tools::hf_loader;
use bitnet_tools::constants::{self, DEFAULT_MODEL_ID, original_dir, converted_dir};
use bitnet_converter::convert_model_on_disk;
use crate::model::{ModelConfig, Transformer};
use crate::attention::KVCache;
use crate::tokenizer::Tokenizer;
use crate::wgpu_context::WgpuContext;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

/// Options for configuring the BitNet Pipeline.
pub struct PipelineOptions {
    /// Model ID to use (defaults to official BitNet-2B if None)
    pub model_id: Option<String>,
    /// Directory containing original model files (optional)
    pub input_dir: Option<PathBuf>,
    /// Directory to store converted model files (optional)
    pub output_dir: Option<PathBuf>,
    /// Optional closure for progress reporting: (step, message)
    pub reporter: Option<Arc<dyn Fn(usize, &str) + Send + Sync>>,
    // Add more options as needed
}

impl std::fmt::Debug for PipelineOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineOptions")
            .field("model_id", &self.model_id)
            .field("input_dir", &self.input_dir)
            .field("output_dir", &self.output_dir)
            .finish()
    }
}

impl Clone for PipelineOptions {
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id.clone(),
            input_dir: self.input_dir.clone(),
            output_dir: self.output_dir.clone(),
            reporter: self.reporter.clone(),
        }
    }
}

/// Orchestrates the full BitNet workflow: download, convert, load, and inference.
pub struct Pipeline {
    /// Pipeline configuration options
    pub options: PipelineOptions,
    /// Path to the original model directory
    pub model_dir: PathBuf,
    /// Path to the converted model directory
    pub converted_dir: PathBuf,
}

/// Result of running inference through the pipeline.
#[derive(Debug)]
pub struct PipelineResult {
    /// Output logits from the model
    pub logits: Vec<f32>,
    /// Decoded token with the highest logit
    pub top_token: String,
    /// Token ID with the highest logit
    pub top_token_id: usize,
    /// Value of the highest logit
    pub top_logit: f32,
}

/// Errors that can occur during pipeline execution.
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// Model download or file error
    #[error("Model download or file error: {0}")]
    ModelFile(String),
    /// Model conversion error
    #[error("Model conversion error: {0}")]
    Conversion(String),
    /// Model load error
    #[error("Model load error: {0}")]
    ModelLoad(String),
    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),
    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// Other error
    #[error("Other: {0}")]
    Other(String),
}

impl Pipeline {
    /// Create a new BitNet Pipeline with the given options.
    pub async fn new(options: PipelineOptions) -> Result<Self, PipelineError> {
        let model_id = options.model_id.clone().unwrap_or(DEFAULT_MODEL_ID.to_string());
        let model_dir = options.input_dir.clone().unwrap_or_else(|| original_dir().join(&model_id));
        let converted_dir = options.output_dir.clone().unwrap_or_else(|| converted_dir().join(&model_id));
        Ok(Self { options, model_dir, converted_dir })
    }

    /// Ensures model files are present and converted. Runs blocking code in spawn_blocking.
    pub async fn ensure_model_ready(&self) -> Result<(), PipelineError> {
        let reporter = self.options.reporter.as_ref();
        let t0 = Instant::now();
        if let Some(cb) = reporter { cb(1, "Step 1: Downloading/Verifying original model files..."); }
        let model_id = self.options.model_id.clone().unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
        let model_dir = self.model_dir.clone();
        let converted_dir = self.converted_dir.clone();
        // Download model files if needed
        let model_id_for_blocking = model_id.clone();
        tokio::task::spawn_blocking(move || {
            hf_loader::get_model(Some(&model_id_for_blocking)).map_err(|e| PipelineError::ModelFile(format!("{e}")))
        }).await.map_err(|e| PipelineError::Other(format!("Join error: {e}")))??;
        if let Some(cb) = reporter { cb(1, "Step 1: Complete."); }
        if let Some(cb) = reporter { cb(1, "Step 2: Converting model to BitNet format..."); }
        // Convert if needed
        if !converted_dir.join("block_0.bin").exists() {
            let input_dir = model_dir.to_str().unwrap().to_string();
            let output_dir = converted_dir.to_str().unwrap().to_string();
            tokio::task::spawn_blocking(move || {
                convert_model_on_disk(&input_dir, &output_dir).map_err(|e| PipelineError::Conversion(format!("{e}")))
            }).await.map_err(|e| PipelineError::Other(format!("Join error: {e}")))??;
        }
        if let Some(cb) = reporter { cb(1, "Step 2: Complete."); }
        if let Some(cb) = reporter { cb(1, "Step 3: Running inference..."); }
        // Timing reporting can be added similarly if needed
        Ok(())
    }

    /// Loads the model, tokenizer, and runs a forward pass on the given prompt.
    pub async fn run_inference(&self, prompt: &str) -> Result<PipelineResult, PipelineError> {
        let reporter = self.options.reporter.as_ref();
        let t0 = Instant::now();
        if let Some(cb) = reporter { cb(2, "Step 1: Loading model..."); }
        // Load config
        let config_path = self.model_dir.join(constants::CONFIG_JSON);
        let config_str = fs::read_to_string(&config_path).map_err(|e| PipelineError::ModelFile(format!("{e}")))?;
        let hf_config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| PipelineError::ModelFile(format!("{e}")))?;
        let model_config = ModelConfig {
            hidden_size: hf_config["hidden_size"].as_u64().unwrap() as usize,
            intermediate_size: hf_config["intermediate_size"].as_u64().unwrap() as usize,
            num_hidden_layers: hf_config["num_hidden_layers"].as_u64().unwrap() as usize,
            num_attention_heads: hf_config["num_attention_heads"].as_u64().unwrap() as usize,
            num_key_value_heads: hf_config["num_key_value_heads"].as_u64().unwrap() as usize,
            vocab_size: hf_config["vocab_size"].as_u64().unwrap() as usize,
            max_seq_len: 4096,
            rms_norm_eps: hf_config["rms_norm_eps"].as_f64().unwrap() as f32,
            dropout: 0.0,
        };
        let mut model = Transformer::from_dir(&self.converted_dir, model_config)
            .map_err(|e| PipelineError::ModelLoad(format!("{e}")))?;
        if let Some(cb) = reporter { cb(2, "Step 1: Complete."); }
        if let Some(cb) = reporter { cb(2, "Step 2: Loading tokenizer..."); }
        // Tokenizer
        let tokenizer_path = self.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
        if let Some(cb) = reporter { cb(2, "Step 2: Complete."); }
        if let Some(cb) = reporter { cb(2, "Step 3: Running inference..."); }
        let context = WgpuContext::new().await.map_err(|e| PipelineError::Other(format!("{e}")))?;
        let tokens = tokenizer.encode(prompt).map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
        let token_id = tokens[0] as usize;
        let pos = 0;
        let mut kv_caches = vec![KVCache::default(); model.layers.len()];
        let result = model.forward(&context, token_id, pos, &mut kv_caches).await
            .map_err(|e| PipelineError::Inference(format!("{e}")))?;
        let logits = result;
        if let Some(cb) = reporter { cb(2, "Step 3: Complete."); }
        // Timing reporting can be added similarly if needed
        // Validate output
        if logits.len() != model.config.vocab_size {
            return Err(PipelineError::Inference("Logits vector has incorrect size".to_string()));
        }
        if !logits.iter().all(|&l| l.is_finite()) {
            return Err(PipelineError::Inference("Logits contain NaN or Infinity".to_string()));
        }
        let (top_token_id, top_logit) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let top_token = tokenizer.decode(&[top_token_id as u32]).unwrap_or("<UNK>".to_string());
        Ok(PipelineResult {
            logits: logits.clone(),
            top_token,
            top_token_id,
            top_logit: *top_logit,
        })
    }
} 