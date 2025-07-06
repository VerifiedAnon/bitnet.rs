//! BitNet Pipeline: Orchestrates model download, conversion, loading, and inference in a robust, async fashion.

use std::path::PathBuf;
use bitnet_tools::hf_loader;
use bitnet_tools::constants::{DEFAULT_MODEL_ID, converted_dir, CONFIG_JSON};
use std::sync::Arc;
use rayon::ThreadPoolBuilder;
use crate::model::ModelFormat;
use log;
use bitnet_tools::logging;

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
    /// Optional log level for this pipeline ("error", "warn", "info", "debug", "trace"). If None, uses global/default.
    pub log_level: Option<String>,
    /// If true, enables verbose logging for debugging (overrides log_level to "debug").
    pub verbose: bool,
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
            .field("log_level", &self.log_level)
            .field("verbose", &self.verbose)
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
            log_level: self.log_level.clone(),
            verbose: self.verbose,
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
        // --- Logging setup ---
        logging::init_logging(options.log_level.as_deref(), options.verbose, None);
        let model_id = options.model_id.clone().unwrap_or(DEFAULT_MODEL_ID.to_string());
        let model_id_for_loader = model_id.clone();
        // Always use hf_loader to get the correct original model directory
        let model_files = tokio::task::spawn_blocking(move || {
            hf_loader::get_model(Some(&model_id_for_loader)).map_err(|e| PipelineError::ModelFile(format!("{e}")))
        }).await.map_err(|e| PipelineError::Other(format!("Join error: {e}")))??;
        let orig_model_dir = model_files.model_dir.clone();
        let model_dir = options.input_dir.clone().unwrap_or_else(|| orig_model_dir.clone());
        let converted_dir = options.output_dir.clone().unwrap_or_else(|| converted_dir().join(&model_id));
        println!("[BitNet][Pipeline][DEBUG] model_dir: {}", model_dir.display());
        match std::fs::metadata(&model_dir) {
            Ok(meta) => println!("[BitNet][Pipeline][DEBUG] model_dir exists: {} (is_dir: {})", model_dir.display(), meta.is_dir()),
            Err(e) => println!("[BitNet][Pipeline][DEBUG] model_dir does not exist: {} (error: {})", model_dir.display(), e),
        }
        // Load config
        // --- Use bitnet_packed_config.json if present ---
        let packed_config_path = model_dir.join("bitnet_packed_config.json");
        let config_path = if packed_config_path.exists() {
            println!("[BitNet][Pipeline][DEBUG] Loading config from: {}", packed_config_path.display());
            packed_config_path
        } else {
            let fallback = model_dir.join("config.json");
            println!("[BitNet][Pipeline][DEBUG] Loading config from: {}", fallback.display());
            fallback
        };
        let config: crate::model::ModelConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)
            .map_err(|e| PipelineError::ModelFile(format!("Config IO error: {e}")))?)
            .map_err(|e| PipelineError::ModelFile(format!("Config JSON error: {e}")))?;
        if options.use_single_file {
            // Single safetensors file path
            let safetensors_path = converted_dir.join("model.safetensors");
            if !safetensors_path.exists() {
                // Run the converter to create model.safetensors
                let input_dir = orig_model_dir.to_str().unwrap().to_string();
                let output_dir = converted_dir.to_str().unwrap().to_string();
                tokio::task::spawn_blocking(move || {
                    bitnet_converter::convert_model_on_disk(&input_dir, &output_dir, true)
                        .map_err(|e| PipelineError::Conversion(format!("{e}")))
                }).await.map_err(|e| PipelineError::Other(format!("Join error: {e}")))??;
            }
            println!("[BitNet][Pipeline][DEBUG] Attempting to open model.safetensors at: {}", model_dir.join("model.safetensors").display());
            // Now load from safetensors using the unified loader
            let model = crate::model::Transformer::load(&converted_dir, config, ModelFormat::Safetensors)
                .map_err(|e| PipelineError::ModelLoad(format!("{e}")))?;
            let tokenizer_path = model_dir.join("tokenizer.json");
            let tokenizer = crate::tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
            return Ok(Self {
                options: options.clone(),
                model_dir,
                converted_dir,
                settings: options.settings.clone(),
                backend: options.backend,
                model: Some(model),
                tokenizer: Some(tokenizer),
                wgpu_context: None,
            });
        } else {
            // Streaming .bin path (default)
            if !converted_dir.join("block_0.bin").exists() {
                let input_dir = orig_model_dir.to_str().unwrap().to_string();
                let output_dir = converted_dir.to_str().unwrap().to_string();
                tokio::task::spawn_blocking(move || {
                    bitnet_converter::convert_model_on_disk(&input_dir, &output_dir, false)
                        .map_err(|e| PipelineError::Conversion(format!("{e}")))
                }).await.map_err(|e| PipelineError::Other(format!("Join error: {e}")))??;
            }
            let model = crate::model::Transformer::load(&converted_dir, config, ModelFormat::Bin)
                .map_err(|e| PipelineError::ModelLoad(format!("{e}")))?;
            let tokenizer_path = model_dir.join("tokenizer.json");
            let tokenizer = crate::tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
            return Ok(Self {
                options: options.clone(),
                model_dir,
                converted_dir,
                settings: options.settings.clone(),
                backend: options.backend,
                model: Some(model),
                tokenizer: Some(tokenizer),
                wgpu_context: None,
            });
        }
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
                log::info!("[BitNet] [CPU] Starting inference for prompt: '{}'.", prompt);
                let _t0 = Instant::now();
                if let Some(cb) = reporter { cb(2, "Step 1: Model/tokenizer already loaded."); }
                // Use already-loaded model/tokenizer
                let model = self.model.as_mut().ok_or(PipelineError::ModelLoad("Model not loaded".to_string()))?;
                let tokenizer = self.tokenizer.as_ref().ok_or(PipelineError::Tokenizer("Tokenizer not loaded".to_string()))?;
                let _t1 = Instant::now();
                if let Some(cb) = reporter { cb(2, "Step 2: Tokenizing input..."); }
                let tokens = tokenizer.encode(prompt).map_err(|e| PipelineError::Tokenizer(format!("{e}")))?;
                log::debug!("[BitNet] [CPU] Tokenized input: {:?}", tokens);
                let token_id = tokens[0] as usize;
                let pos = 0;
                let mut kv_caches = vec![crate::attention::KVCache::default(); model.layers.len()];
                // Log Rayon thread count
                let num_threads = rayon::current_num_threads();
                log::info!("[BitNet] [CPU] Rayon thread pool size: {}", num_threads);
                let _t3 = Instant::now();
                let logits = model.cpu_forward(token_id, pos, &mut kv_caches);
                log::info!("[BitNet] [CPU] Inference completed.");
                if let Some(cb) = reporter { cb(2, "Step 3: Complete."); }
                if logits.len() != model.config.vocab_size {
                    return Err(PipelineError::Inference("Logits vector has incorrect size".to_string()));
                }
                if !logits.iter().all(|&l| l.is_finite()) {
                    return Err(PipelineError::Inference("Logits contain NaN or Infinity".to_string()));
                }
                let (top_token_id, top_logit) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                let top_token = tokenizer.decode(&[top_token_id as u32]).unwrap_or("<UNK>".to_string());
                log::info!("[BitNet] [CPU] Top token: '{}' (id: {}), logit: {}", top_token, top_token_id, top_logit);
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
                log::debug!("[BitNet] [GPU] Tokenized input: {:?}", tokens);
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
                log::info!("[BitNet] [GPU] Top token: '{}' (id: {}), logit: {}", top_token, top_token_id, top_logit);
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