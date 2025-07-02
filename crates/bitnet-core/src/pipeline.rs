//! BitNet Pipeline: Orchestrates model download, conversion, loading, and inference in a robust, async fashion.

use std::path::PathBuf;
use bitnet_tools::hf_loader;
use bitnet_tools::constants::{DEFAULT_MODEL_ID, original_dir, converted_dir, CONFIG_JSON};
use bitnet_converter::convert_model_on_disk;
use std::sync::Arc;
use rayon::ThreadPoolBuilder;

/// Backend device selection for the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// Use the pure Rust CPU backend.
    Cpu,
    /// Use the GPU backend (WGPU/WGSL).
    Gpu,
    /// Automatically select the best available backend.
    Auto,
}

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
    /// Backend device to use for inference (CPU, GPU, or Auto).
    pub backend: PipelineBackend,
    /// Optional inference/generation settings
    pub settings: Option<crate::settings::InferenceSettings>,
    /// If true, load model from a single safetensors file instead of multi-file
    pub use_single_file: bool,
    // Add more options as needed
}

impl std::fmt::Debug for PipelineOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineOptions")
            .field("model_id", &self.model_id)
            .field("input_dir", &self.input_dir)
            .field("output_dir", &self.output_dir)
            .field("backend", &self.backend)
            .field("settings", &self.settings)
            .field("use_single_file", &self.use_single_file)
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
            backend: self.backend,
            settings: self.settings.clone(),
            use_single_file: self.use_single_file,
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
    /// Inference/generation settings
    pub settings: Option<crate::settings::InferenceSettings>,
    /// Backend device to use for inference (CPU, GPU, or Auto).
    pub backend: PipelineBackend,
    /// Persistently loaded model (CPU or GPU)
    pub model: Option<crate::model::Transformer>,
    /// Persistently loaded tokenizer
    pub tokenizer: Option<crate::tokenizer::Tokenizer>,
    /// Persistently loaded WGPU context (for GPU)
    pub wgpu_context: Option<crate::wgpu_context::WgpuContext>,
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
    /// True if generation stopped by EOS token
    pub stopped_by_eos: bool,
    /// True if generation stopped by max_tokens
    pub stopped_by_max_tokens: bool,
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
        if options.use_single_file {
            // Use the new single-file loader
            return Pipeline::from_safetensors(options).await;
        }
        let model_id = options.model_id.clone().unwrap_or(DEFAULT_MODEL_ID.to_string());
        let model_dir = options.input_dir.clone().unwrap_or_else(|| original_dir().join(&model_id));
        let converted_dir = options.output_dir.clone().unwrap_or_else(|| converted_dir().join(&model_id));
        Ok(Self {
            options: options.clone(),
            model_dir,
            converted_dir,
            settings: options.settings.clone(),
            backend: options.backend,
            model: None,
            tokenizer: None,
            wgpu_context: None,
        })
    }

    /// Load model/tokenizer from a single safetensors file (stub for now)
    pub async fn from_safetensors(options: PipelineOptions) -> Result<Self, PipelineError> {
        println!("[BitNet] [EXPERIMENTAL] Loading model from single .safetensors file (not yet implemented)");
        Err(PipelineError::Other("Single-file safetensors loading not yet implemented".to_string()))
    }

    /// Ensures model files are present and converted. Runs blocking code in spawn_blocking.
    pub async fn ensure_model_ready(&mut self) -> Result<(), PipelineError> {
        let reporter = self.options.reporter.as_ref();
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
        if let Some(cb) = reporter { cb(1, "Step 3: Loading model/tokenizer/context..."); }
        // Only load if not already loaded
        if self.model.is_none() || self.tokenizer.is_none() || (self.backend == PipelineBackend::Gpu && self.wgpu_context.is_none()) {
            // Load config
            let config_path = self.model_dir.join(CONFIG_JSON);
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| PipelineError::ModelFile(format!("{e}")))?;
            let hf_config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| PipelineError::ModelFile(format!("{e}")))?;
            let model_config = crate::model::ModelConfig {
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
            let model = crate::model::Transformer::from_dir(&self.converted_dir, model_config)
                .map_err(|e| PipelineError::ModelLoad(format!("{e}")))?;
            let tokenizer_path = self.model_dir.join("tokenizer.json");
            let tokenizer = crate::tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
            self.model = Some(model);
            self.tokenizer = Some(tokenizer);
            if self.backend == PipelineBackend::Gpu {
                let wgpu_context = crate::wgpu_context::WgpuContext::new().await.map_err(|e| PipelineError::Other(e.to_string()))?;
                self.wgpu_context = Some(wgpu_context);
            }
        }
        if let Some(cb) = reporter { cb(1, "Step 3: Complete."); }
        Ok(())
    }

    /// Loads the model, tokenizer, and runs a forward pass on the given prompt.
    pub async fn run_inference(&mut self, prompt: &str) -> Result<PipelineResult, PipelineError> {
        let reporter = self.options.reporter.as_ref();
        match self.backend {
            PipelineBackend::Cpu => {
                // --- Pure Rust, multi-threaded CPU inference path ---
                // Set Rayon thread pool size only once at process start (not per inference)
                static INIT_RAYON: std::sync::Once = std::sync::Once::new();
                let settings = self.settings.clone().unwrap_or_default();
                INIT_RAYON.call_once(|| {
                    if settings.threads > 0 {
                        let _ = ThreadPoolBuilder::new().num_threads(settings.threads).build_global();
                    }
                });
                use std::time::Instant;
                println!("[BitNet] [CPU] Starting inference for prompt: '{}'.", prompt);
                let t0 = Instant::now();
                if let Some(cb) = reporter { cb(2, "Step 1: Model/tokenizer already loaded."); }
                // Use already-loaded model/tokenizer
                let model = self.model.as_mut().ok_or(PipelineError::ModelLoad("Model not loaded".to_string()))?;
                let tokenizer = self.tokenizer.as_ref().ok_or(PipelineError::Tokenizer("Tokenizer not loaded".to_string()))?;
                let t1 = Instant::now();
                if let Some(cb) = reporter { cb(2, "Step 2: Tokenizing input..."); }
                let tokens = tokenizer.encode(prompt).map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
                println!("[BitNet] [CPU] Tokenized input: {:?}", tokens);
                let token_id = tokens[0] as usize;
                let pos = 0;
                let mut kv_caches = vec![crate::attention::KVCache::default(); model.layers.len()];
                // Log Rayon thread count
                let num_threads = rayon::current_num_threads();
                println!("[BitNet] [CPU] Rayon thread pool size: {}", num_threads);
                let t3 = Instant::now();
                let logits = model.cpu_forward(token_id, pos, &mut kv_caches);
                println!("[BitNet] [CPU] Inference completed in {:.2?}", t3.elapsed());
                if let Some(cb) = reporter { cb(2, "Step 3: Complete."); }
                if logits.len() != model.config.vocab_size {
                    return Err(PipelineError::Inference("Logits vector has incorrect size".to_string()));
                }
                if !logits.iter().all(|&l| l.is_finite()) {
                    return Err(PipelineError::Inference("Logits contain NaN or Infinity".to_string()));
                }
                let (top_token_id, top_logit) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                let top_token = tokenizer.decode(&[top_token_id as u32]).unwrap_or("<UNK>".to_string());
                println!("[BitNet] [CPU] Top token: '{}' (id: {}), logit: {}", top_token, top_token_id, top_logit);
                Ok(PipelineResult {
                    logits: logits.clone(),
                    top_token,
                    top_token_id,
                    top_logit: *top_logit,
                    stopped_by_eos: false,
                    stopped_by_max_tokens: true,
                })
            }
            PipelineBackend::Gpu => {
                // --- GPU inference path (persistently loaded model/context/tokenizer) ---
                let model = self.model.as_mut().ok_or(PipelineError::ModelLoad("Model not loaded".to_string()))?;
                let tokenizer = self.tokenizer.as_ref().ok_or(PipelineError::Tokenizer("Tokenizer not loaded".to_string()))?;
                let wgpu_context = self.wgpu_context.as_ref().ok_or(PipelineError::Other("WGPU context not loaded".to_string()))?;
                let tokens = tokenizer.encode(prompt).map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
                println!("[BitNet] [GPU] Tokenized input: {:?}", tokens);
                let token_id = tokens[0] as usize;
                let pos = 0;
                let mut kv_caches = vec![crate::attention::KVCache::default(); model.layers.len()];
                let logits = model.forward(wgpu_context, token_id, pos, &mut kv_caches).await.map_err(|e| PipelineError::Inference(e.to_string()))?;
                if logits.len() != model.config.vocab_size {
                    return Err(PipelineError::Inference("Logits vector has incorrect size".to_string()));
                }
                if !logits.iter().all(|&l| l.is_finite()) {
                    return Err(PipelineError::Inference("Logits contain NaN or Infinity".to_string()));
                }
                let (top_token_id, top_logit) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                let top_token = tokenizer.decode(&[top_token_id as u32]).unwrap_or("<UNK>".to_string());
                println!("[BitNet] [GPU] Top token: '{}' (id: {}), logit: {}", top_token, top_token_id, top_logit);
                Ok(PipelineResult {
                    logits: logits.clone(),
                    top_token,
                    top_token_id,
                    top_logit: *top_logit,
                    stopped_by_eos: false,
                    stopped_by_max_tokens: true,
                })
            }
            PipelineBackend::Auto => {
                unimplemented!("Auto backend selection not yet implemented");
            }
        }
    }
} 