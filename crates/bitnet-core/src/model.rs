//! BitNet transformer model implementation.

use crate::{
    attention::{Attention, AttentionConfig, KVCache},
    bitnet_linear::BitLinear,
    error::BitNetError,
    feed_forward::{FeedForward},
    rms_norm::BitnetRmsNorm,
};
use bitnet_converter::packer::{
    BitLinearRecord, EmbeddingRecord, RmsNormRecord, TransformerBlockRecord,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Model file format.
pub enum ModelFormat {
    /// Binary format.
    Bin,
    /// Safetensors format.
    Safetensors,
}

fn default_max_seq_len() -> usize { 4096 }

/// Model configuration.
#[derive(serde::Deserialize, serde::Serialize)]
pub struct ModelConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// Intermediate size.
    pub intermediate_size: usize,
    /// Number of hidden layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads.
    pub num_key_value_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Dropout rate (optional, defaults to 0.0).
    #[serde(default)]
    pub dropout: Option<f32>,
    /// Maximum sequence length (accepts Hugging Face's max_position_embeddings).
    #[serde(default, alias = "max_position_embeddings")]
    pub max_seq_len: usize,
}

/// Transformer layer.
#[derive(Clone)]
pub struct Layer {
    /// Attention layer.
    pub attn: Attention,
    /// Feed-forward layer.
    pub ffn: FeedForward,
    /// Attention normalization.
    pub attention_norm: BitnetRmsNorm,
    /// Feed-forward normalization.
    pub ffn_norm: BitnetRmsNorm,
}

impl Layer {
    /// Forward pass for the layer (CPU).
    pub fn cpu_forward(&mut self, x: &[f32], pos_offset: usize) -> Vec<f32> {
        let x_norm = self.attention_norm.forward(x);
        let attn_output = self.attn.cpu_forward(&x_norm, pos_offset);
        let residual_after_attn: Vec<f32> = x.iter().zip(attn_output.iter()).map(|(a, b)| a + b).collect();
        
        let x_norm2 = self.ffn_norm.forward(&residual_after_attn);
        let batch_size = x.len() / self.attention_norm.weight.len();
        let ffn_output = self.ffn.cpu_forward(&x_norm2, batch_size);
        
        residual_after_attn.iter().zip(ffn_output.iter()).map(|(a, b)| a + b).collect()
    }

    /// Forward pass for the layer (GPU).
    pub async fn gpu_forward(&mut self, context: &crate::wgpu_context::WgpuContext, x: &[f32], pos_offset: usize) -> Vec<f32> {
        let x_norm = self.attention_norm.forward(x);
        let attn_output = self.attn.gpu_forward(context, &x_norm, pos_offset).await;
        let residual_after_attn: Vec<f32> = x.iter().zip(attn_output.iter()).map(|(a, b)| a + b).collect();
        let x_norm2 = self.ffn_norm.forward(&residual_after_attn);
        let batch_size = x.len() / self.attention_norm.weight.len();
        let ffn_output = self.ffn.forward(context, &x_norm2, batch_size).await;
        residual_after_attn.iter().zip(ffn_output.iter()).map(|(a, b)| a + b).collect()
    }
}

/// Transformer model.
pub struct Transformer {
    /// Embedding weights.
    pub embedding: Vec<f32>,
    /// Embedding shape.
    pub embedding_shape: Vec<usize>,
    /// Model layers.
    pub layers: Vec<Layer>,
    /// Final normalization.
    pub norm: BitnetRmsNorm,
    /// Output projection.
    pub output: BitLinear,
    /// Model configuration.
    pub config: ModelConfig,
}

impl Transformer {
    /// Load a Transformer model from disk.
    pub fn load(dir: &Path, config: ModelConfig, format: ModelFormat) -> Result<Self, BitNetError> {
        match format {
            ModelFormat::Safetensors => {
                use bitnet_converter::source::ModelSource;

                let safetensors_path = dir.join("model.safetensors");
                if !safetensors_path.exists() {
                    return Err(BitNetError::Config(format!("File not found: {}", safetensors_path.display())));
                }

                let source = ModelSource::SafetensorsFile(safetensors_path.to_string_lossy().to_string());
                let tensor_map = source.load_tensors().map_err(|e| BitNetError::Config(format!("Failed to load safetensors: {e}")))?;

                let key = "tok_embeddings.weight";
                if let Some((_data, shape)) = tensor_map.get(key) {
                    println!("[DEBUG] Found key '{}', shape = {:?}", key, shape);
                } else {
                    println!("[DEBUG] Key '{}' not found in tensor_map", key);
                }
                let (embedding_data, embedding_shape) = tensor_map.get(key).cloned().ok_or_else(|| BitNetError::Config("Missing 'tok_embeddings.weight'".to_string()))?;
                let embedding = embedding_data.as_f32_vec().ok_or_else(|| BitNetError::Config("Embedding not f32".to_string()))?.clone();
                
                let (norm_data, _) = tensor_map.get("norm.weight").cloned().ok_or_else(|| BitNetError::Config("Missing 'norm.weight'".to_string()))?;
                let norm = BitnetRmsNorm::new(norm_data.as_f32_vec().ok_or_else(|| BitNetError::Config("Norm not f32".to_string()))?.clone(), config.rms_norm_eps);

                let (output_weights, output_shape) = tensor_map.get("output.weight").cloned().ok_or_else(|| BitNetError::Config("Missing 'output.weight'".to_string()))?;
                let (output_scales, _) = tensor_map.get("output.weight_scale").cloned().ok_or_else(|| BitNetError::Config("Missing 'output.weight_scale'".to_string()))?;
                let output = BitLinear {
                    packed_weights: output_weights.as_u32_vec().unwrap().clone(),
                    weight_scales: output_scales.as_f32_vec().unwrap().clone(),
                    in_features: output_shape[1] * 16,
                    out_features: output_shape[0],
                };

                let attn_config = AttentionConfig::new(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.max_seq_len);
                let mut layers = Vec::with_capacity(config.num_hidden_layers);

                let load_bitlinear = |w_name: &str, s_name: &str| -> Result<BitLinearRecord, BitNetError> {
                    let (weights_data, shape) = tensor_map.get(w_name).cloned().ok_or_else(|| BitNetError::Config(format!("Missing tensor '{}'", w_name)))?;
                    let (scales_data, _) = tensor_map.get(s_name).cloned().ok_or_else(|| BitNetError::Config(format!("Missing tensor '{}'", s_name)))?;
                    Ok(BitLinearRecord {
                        packed_weights: weights_data.as_u32_vec().unwrap().clone(),
                        weight_scales: scales_data.as_f32_vec().unwrap().clone(),
                        in_features: shape[1] * 16,
                        out_features: shape[0],
                    })
                };
                for i in 0..config.num_hidden_layers {
                    // Attention WQKV
                    let wqkv_weight = format!("layers.{}.attention.wqkv.weight", i);
                    let wqkv_weight_scale = format!("layers.{}.attention.wqkv.weight_scale", i);
                    let wqkv_record = load_bitlinear(&wqkv_weight, &wqkv_weight_scale)?;
                    let (q_proj, k_proj, v_proj) = split_wqkv_record(&wqkv_record, &attn_config);

                    // Attention WO
                    let wo_weight = format!("layers.{}.attention.wo.weight", i);
                    let wo_weight_scale = format!("layers.{}.attention.wo.weight_scale", i);
                    let o_proj = load_bitlinear(&wo_weight, &wo_weight_scale)?;

                    // Feed Forward W13
                    let w13_weight = format!("layers.{}.feed_forward.w13.weight", i);
                    let w13_weight_scale = format!("layers.{}.feed_forward.w13.weight_scale", i);
                    let w13_record = load_bitlinear(&w13_weight, &w13_weight_scale)?;

                    // Feed Forward W2
                    let w2_weight = format!("layers.{}.feed_forward.w2.weight", i);
                    let w2_weight_scale = format!("layers.{}.feed_forward.w2.weight_scale", i);
                    let w2_record = load_bitlinear(&w2_weight, &w2_weight_scale)?;

                    let attn_norm_key = format!("layers.{}.attention_norm.weight", i);
                    let ffn_norm_key = format!("layers.{}.ffn_norm.weight", i);
                    let (attn_norm_data, _) = tensor_map.get(&attn_norm_key).cloned().unwrap();
                    let (ffn_norm_data, _) = tensor_map.get(&ffn_norm_key).cloned().unwrap();

                    layers.push(Layer {
                        attn: Attention::from_records(q_proj, k_proj, v_proj, o_proj, &attn_config),
                        ffn: FeedForward::from_records(w13_record, w2_record),
                        attention_norm: BitnetRmsNorm::new(attn_norm_data.as_f32_vec().unwrap().clone(), config.rms_norm_eps),
                        ffn_norm: BitnetRmsNorm::new(ffn_norm_data.as_f32_vec().unwrap().clone(), config.rms_norm_eps),
                    });
                }

                Ok(Transformer { embedding, embedding_shape, layers, norm, output, config })
            }
            ModelFormat::Bin => Err(BitNetError::Config("Binary format loading is deprecated. Please use single-file safetensors.".to_string())),
        }
    }

    /// Processes a batch of tokens (e.g., a prompt) efficiently.
    pub fn cpu_forward_batch(&mut self, tokens: &[usize], pos_offset: usize) -> Vec<f32> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return vec![];
        }
        let hidden_size = self.embedding_shape[1];
        
        let mut x: Vec<f32> = Vec::with_capacity(seq_len * hidden_size);
        for &token_id in tokens {
            let start = token_id * hidden_size;
            x.extend_from_slice(&self.embedding[start..start + hidden_size]);
        }
        
        for layer in self.layers.iter_mut() {
            x = layer.cpu_forward(&x, pos_offset);
        }

        let x_norm = self.norm.forward(&x);
        let logits_batch = self.output.forward_cpu(&x_norm, seq_len);
        
        let vocab_size = self.config.vocab_size;
        logits_batch[(seq_len - 1) * vocab_size..].to_vec()
    }

    /// Processes a single token for auto-regressive generation.
    pub fn cpu_forward_single_token(&mut self, token_id: usize, pos: usize) -> Vec<f32> {
        let hidden_size = self.embedding_shape[1];
        let start = token_id * hidden_size;
        let mut x = self.embedding[start..start + hidden_size].to_vec();

        for layer in self.layers.iter_mut() {
            x = layer.cpu_forward(&x, pos);
        }
        
        let x_norm = self.norm.forward(&x);
        self.output.forward_cpu(&x_norm, 1)
    }

    /// Processes a batch of tokens (prompt) on GPU.
    pub async fn gpu_forward_batch(&mut self, context: &crate::wgpu_context::WgpuContext, tokens: &[usize], pos_offset: usize) -> Vec<f32> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return vec![];
        }
        let hidden_size = self.embedding_shape[1];
        let mut x: Vec<f32> = Vec::with_capacity(seq_len * hidden_size);
        for &token_id in tokens {
            let start = token_id * hidden_size;
            x.extend_from_slice(&self.embedding[start..start + hidden_size]);
        }
        for layer in self.layers.iter_mut() {
            x = layer.gpu_forward(context, &x, pos_offset).await;
        }
        let x_norm = self.norm.forward(&x);
        let logits_batch = self.output.forward(context, &x_norm, seq_len).await;
        let vocab_size = self.config.vocab_size;
        logits_batch[(seq_len - 1) * vocab_size..].to_vec()
    }
    /// Processes a single token for auto-regressive generation on GPU.
    pub async fn gpu_forward_single_token(&mut self, context: &crate::wgpu_context::WgpuContext, token_id: usize, pos: usize) -> Vec<f32> {
        let hidden_size = self.embedding_shape[1];
        let start = token_id * hidden_size;
        let mut x = self.embedding[start..start + hidden_size].to_vec();
        for layer in self.layers.iter_mut() {
            x = layer.gpu_forward(context, &x, pos).await;
        }
        let x_norm = self.norm.forward(&x);
        self.output.forward(context, &x_norm, 1).await
    }

    /// Resets the internal KV cache for all attention layers.
    pub fn reset_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn.kv_cache = KVCache::default();
        }
    }
}

pub(crate) fn split_wqkv_record(wqkv: &BitLinearRecord, attn_config: &AttentionConfig) -> (BitLinearRecord, BitLinearRecord, BitLinearRecord) {
    let head_dim = attn_config.hidden_size / attn_config.num_heads;
    let q_dim = attn_config.num_heads * head_dim;
    let k_dim = attn_config.num_kv_heads * head_dim;
    let v_dim = attn_config.num_kv_heads * head_dim;
    let in_features = wqkv.in_features;
    let packed_per_row = (in_features + 15) / 16;

    let q_packed_rows = q_dim;
    let k_packed_rows = k_dim;
    
    let q_packed_end = q_packed_rows * packed_per_row;
    let k_packed_end = q_packed_end + k_packed_rows * packed_per_row;

    let q_scale_end = q_dim;
    let k_scale_end = q_scale_end + k_dim;

    let q_rec = BitLinearRecord {
        packed_weights: wqkv.packed_weights[0..q_packed_end].to_vec(),
        weight_scales: wqkv.weight_scales[0..q_scale_end].to_vec(),
        in_features, out_features: q_dim,
    };
    let k_rec = BitLinearRecord {
        packed_weights: wqkv.packed_weights[q_packed_end..k_packed_end].to_vec(),
        weight_scales: wqkv.weight_scales[q_scale_end..k_scale_end].to_vec(),
        in_features, out_features: k_dim,
    };
    let v_rec = BitLinearRecord {
        packed_weights: wqkv.packed_weights[k_packed_end..].to_vec(),
        weight_scales: wqkv.weight_scales[k_scale_end..].to_vec(),
        in_features, out_features: v_dim,
    };
    (q_rec, k_rec, v_rec)
}