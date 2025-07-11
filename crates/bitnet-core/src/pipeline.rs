//! BitNet Pipeline: Orchestrates model loading and inference.

use crate::model::{ModelFormat, Transformer};
use crate::settings::InferenceSettings;
use bitnet_tools::constants::{converted_dir, DEFAULT_MODEL_ID};
use bitnet_tools::{hf_loader, logging};
use log;
// Use rand::Uniform and rand::thread_rng for compatibility
use rand::Rng;
use rand::thread_rng;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Backend for pipeline execution.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum PipelineBackend {
    /// CPU backend.
    Cpu,
    /// GPU backend.
    Gpu,
    /// Auto backend selection.
    Auto,
}

/// Options for pipeline creation.
pub struct PipelineOptions {
    /// Model ID.
    pub model_id: Option<String>,
    /// Input directory.
    pub input_dir: Option<PathBuf>,
    /// Output directory.
    pub output_dir: Option<PathBuf>,
    /// Reporter callback.
    pub reporter: Option<Arc<dyn Fn(usize, &str) + Send + Sync>>,
    /// Backend to use.
    pub backend: PipelineBackend,
    /// Inference settings.
    pub settings: Option<InferenceSettings>,
    /// Use single safetensor file.
    pub use_single_file: bool,
    /// Log level.
    pub log_level: Option<String>,
    /// Verbose output.
    pub verbose: bool,
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

/// Pipeline for model inference.
pub struct Pipeline {
    /// Pipeline options.
    pub options: PipelineOptions,
    /// Loaded model.
    pub model: Option<Transformer>,
    /// Tokenizer.
    pub tokenizer: Option<crate::tokenizer::Tokenizer>,
    /// WGPU context.
    pub wgpu_context: Option<crate::wgpu_context::WgpuContext>,
}

/// Error type for pipeline operations.
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// Model file error.
    #[error("Model file error: {0}")] ModelFile(String),
    /// Conversion error.
    #[error("Conversion error: {0}")] Conversion(String),
    /// Model load error.
    #[error("Model load error: {0}")] ModelLoad(String),
    /// Tokenizer error.
    #[error("Tokenizer error: {0}")] Tokenizer(String),
    /// Other error.
    #[error("Other error: {0}")] Other(String),
}

/// Sampling settings for generation.
pub struct SampleSettings {
    /// Temperature.
    pub temperature: f32,
    /// Top-k.
    pub top_k: usize,
    /// Top-p.
    pub top_p: f32,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// No repeat ngram size.
    pub no_repeat_ngram_size: usize,
    /// Whether to sample.
    pub do_sample: bool,
}

impl Default for SampleSettings {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 0, top_p: 0.0, repetition_penalty: 1.0, no_repeat_ngram_size: 0, do_sample: true }
    }
}

/// Stateless processor for sampling next token from logits.
pub struct LogitsProcessor;

impl LogitsProcessor {
    /// Sample a token index from logits, given the generated token history and settings.
    pub fn sample(logits: &[f32], generated_ids: &[usize], settings: &SampleSettings) -> usize {
        if logits.is_empty() { return 0; }
        let mut processed_logits = logits.to_vec();

        // 1. Repetition penalty
        if (settings.repetition_penalty - 1.0).abs() > f32::EPSILON {
            let penalized: std::collections::HashSet<_> = generated_ids.iter().copied().collect();
            for &token_id in &penalized {
                if token_id < processed_logits.len() {
                    if processed_logits[token_id] > 0.0 {
                        processed_logits[token_id] /= settings.repetition_penalty;
                    } else {
                        processed_logits[token_id] *= settings.repetition_penalty;
                    }
                }
            }
        }

        // 2. No-repeat n-gram
        if settings.no_repeat_ngram_size > 0 && generated_ids.len() >= settings.no_repeat_ngram_size {
            let n = settings.no_repeat_ngram_size;
            let last_tokens = &generated_ids[generated_ids.len() + 1 - n..];
            let mut banned = std::collections::HashSet::new();
            for i in 0..=generated_ids.len().saturating_sub(n) {
                if &generated_ids[i..i + n - 1] == &last_tokens[..n-1] {
                    banned.insert(generated_ids[i + n - 1]);
                }
            }
            for &token in &banned {
                if token < processed_logits.len() {
                    processed_logits[token] = f32::NEG_INFINITY;
                }
            }
        }

        // 3. Temperature
        if (settings.temperature - 1.0).abs() > f32::EPSILON && settings.temperature > 0.0 {
            for l in &mut processed_logits {
                *l /= settings.temperature;
            }
        }

        // 4. Top-k and top-p filtering
        let mut logits_with_indices: Vec<(f32, usize)> = processed_logits
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_i, v)| v.is_finite())
            .map(|(i, v)| (v, i))
            .collect();
        logits_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Top-k
        if settings.top_k > 0 && settings.top_k < logits_with_indices.len() {
            logits_with_indices.truncate(settings.top_k);
        }

        // Top-p (nucleus)
        if settings.top_p > 0.0 && settings.top_p < 1.0 {
            let sorted_logits: Vec<f32> = logits_with_indices.iter().map(|x| x.0).collect();
            let probabilities = softmax(&sorted_logits);
            let mut cumulative = 0.0;
            let mut nucleus = logits_with_indices.len();
            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if cumulative > settings.top_p {
                    nucleus = i + 1;
                    break;
                }
            }
            logits_with_indices.truncate(nucleus);
        }
        if logits_with_indices.is_empty() { return 0; }

        // 5. Sampling or argmax
        if settings.do_sample && logits_with_indices.len() > 1 {
            let logits_vec: Vec<f32> = logits_with_indices.iter().map(|x| x.0).collect();
            let probabilities = softmax(&logits_vec);
            let mut rng = thread_rng();
            let rand_val: f32 = rng.gen();
            let mut cumulative = 0.0;
            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if rand_val <= cumulative {
                    return logits_with_indices[i].1;
                }
            }
            logits_with_indices.last().unwrap().1
        } else {
            // Argmax
            logits_with_indices.first().map(|x| x.1).unwrap_or(0)
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0/ exps.len() as f32; exps.len()] // Fallback to uniform distribution
    }
}


impl Pipeline {
    /// Create a new pipeline.
    pub async fn new(options: PipelineOptions) -> Result<Self, PipelineError> {
        logging::init_logging(options.log_level.as_deref(), options.verbose, None);

        let model_id = options.model_id.clone().unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
        let hf_files = hf_loader::get_model(Some(&model_id)).map_err(|e| PipelineError::ModelFile(e.to_string()))?;
        // Always use a converted/output directory for BitNet safetensors
        let converted_dir = options.output_dir.clone().unwrap_or_else(|| converted_dir().join(&model_id));
        std::fs::create_dir_all(&converted_dir).map_err(|e| PipelineError::Other(e.to_string()))?;
        let safetensors_path = converted_dir.join("model.safetensors");
        let config_path = converted_dir.join("config.json");

        // If the BitNet safetensors or config is missing, run the converter
        if !safetensors_path.exists() || !config_path.exists() {
            log::warn!("BitNet model files not found in {:?}. Running converter.", converted_dir);
            let input_str = hf_files.model_dir.to_str().unwrap().to_string();
            let output_str = converted_dir.to_str().unwrap().to_string();
            // Fix: set write_streamable = false to produce only safetensors
            bitnet_converter::convert_model_on_disk(&input_str, &output_str, false)
                .map_err(|e| PipelineError::Conversion(e.to_string()))?;
        }

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| PipelineError::ModelFile(e.to_string()))?;
        let config: crate::model::ModelConfig = serde_json::from_str(&config_str).map_err(|e| PipelineError::ModelFile(e.to_string()))?;

        log::info!("Loading BitNet model from safetensor file in {:?}...", converted_dir);
        let model = Transformer::load(&converted_dir, config, ModelFormat::Safetensors)
            .map_err(|e| PipelineError::ModelLoad(e.to_string()))?;

        // --- Load tokenizer ---
        let tokenizer_path = converted_dir.join("tokenizer.json");
        let tokenizer_path_fallback = hf_files.model_dir.join("tokenizer.json");
        let tokenizer_path_to_use = if tokenizer_path.exists() {
            tokenizer_path.clone()
        } else if tokenizer_path_fallback.exists() {
            tokenizer_path_fallback.clone()
        } else {
            return Err(PipelineError::Tokenizer("Could not find tokenizer.json in either converted or original directory".to_string()));
        };
        let tokenizer = crate::tokenizer::Tokenizer::from_file(tokenizer_path_to_use.to_str().unwrap())
            .map_err(|e| PipelineError::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        let wgpu_context = if options.backend == PipelineBackend::Gpu {
            Some(crate::wgpu_context::WgpuContext::new().await.map_err(|e| PipelineError::Other(format!("Failed to init WGPU: {e}")))?)
        } else {
            None
        };
        Ok(Self {
            options: options.clone(),
            model: Some(model),
            tokenizer: Some(tokenizer),
            wgpu_context,
        })
    }

    /// Generate text from a prompt.
    pub async fn generate_text(&mut self, prompt: &str, settings: &InferenceSettings) -> Result<String, PipelineError> {
        let tokenizer = self.tokenizer.as_ref().ok_or(PipelineError::Tokenizer("Tokenizer not loaded".to_string()))?;
        let model = self.model.as_mut().ok_or(PipelineError::ModelLoad("Model not loaded".to_string()))?;
        let use_gpu = self.options.backend == PipelineBackend::Gpu;
        let wgpu_context = self.wgpu_context.as_ref();
        // 1. Reset state and tokenize prompt
        model.reset_kv_cache();
        let mut all_tokens = tokenizer.encode(prompt).map_err(|e| PipelineError::Tokenizer(e.to_string()))?;
        if all_tokens.is_empty() {
            all_tokens.push(1); // Default to BOS token if prompt is empty
        }
        let prompt_tokens_usize: Vec<usize> = all_tokens.iter().map(|&t| t as usize).collect();
        let mut logits = if use_gpu {
            let ctx = wgpu_context.ok_or(PipelineError::Other("WGPU context missing".to_string()))?;
            model.gpu_forward_batch(ctx, &prompt_tokens_usize, 0).await
        } else {
            model.cpu_forward_batch(&prompt_tokens_usize, 0)
        };
        // 3. Generation loop
        let mut generated_tokens = Vec::new();
        for _ in 0..settings.max_new_tokens {
            let sample_settings = SampleSettings {
                temperature: settings.temperature as f32,
                top_k: settings.top_k,
                top_p: settings.top_p as f32,
                repetition_penalty: settings.repetition_penalty as f32,
                no_repeat_ngram_size: settings.no_repeat_ngram_size,
                do_sample: settings.do_sample,
            };
            let all_tokens_usize: Vec<usize> = all_tokens.iter().map(|&t| t as usize).collect();
            let next_token = LogitsProcessor::sample(&logits, &all_tokens_usize, &sample_settings) as u32;
            if Some(next_token) == settings.eos_token_id {
                break;
            }
            generated_tokens.push(next_token);
            all_tokens.push(next_token);
            logits = if use_gpu {
                let ctx = wgpu_context.ok_or(PipelineError::Other("WGPU context missing".to_string()))?;
                model.gpu_forward_single_token(ctx, next_token as usize, all_tokens.len() - 1).await
            } else {
                model.cpu_forward_single_token(next_token as usize, all_tokens.len() - 1)
            };
        }
        tokenizer.decode(&generated_tokens).map_err(|e| PipelineError::Tokenizer(e.to_string()))
    }
} 