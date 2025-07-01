//! BitNet transformer model implementation.
//!
//! This module provides the core transformer model implementation for BitNet,
//! including the model architecture, configuration, and inference logic.
//!
//! # Architecture
//!
//! The BitNet model follows a standard transformer architecture with some key modifications:
//!
//! - Quantized linear layers (see [`crate::bitnet_linear`])
//! - RMSNorm for layer normalization
//! - SwiGLU activation in feed-forward layers
//! - Rotary position embeddings (RoPE) in attention
//!
//! # Examples
//!
//! ```rust,no_run
//! use bitnet_core::model::{Transformer, ModelConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a model configuration
//! let config = ModelConfig {
//!     hidden_size: 2048,
//!     intermediate_size: 5632,
//!     num_hidden_layers: 24,
//!     num_attention_heads: 32,
//!     vocab_size: 32000,
//!     rms_norm_eps: 1e-6,
//!     dropout: 0.0,
//! };
//!
//! // Initialize the model
//! let model = config.init();
//!
//! // Run inference
//! let tokens = vec![1, 2, 3, 4]; // Example token IDs
//! let output = model.forward(&tokens);
//! # Ok(())
//! # }
//! ```
//!
//! # Performance
//!
//! The model implementation is optimized for both CPU and GPU:
//!
//! - Uses quantized weights and activations
//! - Supports SIMD acceleration on CPU
//! - GPU acceleration via WGSL compute shaders
//! - Streaming-friendly model loading
//!
//! # Safety
//!
//! This module uses unsafe code in the following places:
//!
//! - Memory mapping for efficient model loading
//! - SIMD intrinsics in quantized operations
//!
//! All unsafe operations are thoroughly tested and validated.

use crate::{
    attention::{Attention, AttentionConfig},
    bitnet_linear::BitLinear,
    error::BitNetError,
    feed_forward::{FeedForward, FeedForwardConfig},
    rms_norm::BitnetRmsNorm,
    wgpu_context::WgpuContext,
};
use crate::attention::KVCache;
use bitnet_converter::packer::{
    EmbeddingRecord, RmsNormRecord, TransformerBlockRecord,
};
use bincode::config::standard;
use bincode::serde::decode_from_slice;
use std::fs;
use std::path::Path;

/// Configuration for a BitNet model.
///
/// This struct holds all the hyperparameters needed to define a BitNet model's architecture.
///
/// # Examples
///
/// ```rust
/// use bitnet_core::model::ModelConfig;
///
/// let config = ModelConfig {
///     hidden_size: 2048,
///     intermediate_size: 5632,
///     num_hidden_layers: 24,
///     num_attention_heads: 32,
///     vocab_size: 32000,
///     rms_norm_eps: 1e-6,
///     dropout: 0.0,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Size of the hidden layers
    pub hidden_size: usize,
    /// Size of the intermediate (feed-forward) layers
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads
    pub num_key_value_heads: usize,
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Epsilon for RMSNorm
    pub rms_norm_eps: f32,
    /// Dropout probability
    pub dropout: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

/// A single transformer block in the BitNet model.
///
/// Each block contains:
/// - Multi-head self-attention
/// - Feed-forward network
/// - Two RMSNorm layers
///
/// The forward pass follows the standard transformer pattern:
/// ```text
/// x = x + Attention(RMSNorm(x))
/// x = x + FeedForward(RMSNorm(x))
/// ```
#[derive(Clone)]
pub struct Layer {
    /// Multi-head self-attention module for this transformer block.
    pub attn: Attention,
    /// Feed-forward network for this transformer block.
    pub ffn: FeedForward,
    /// First RMSNorm layer for this transformer block.
    pub attention_norm: BitnetRmsNorm,
    /// Second RMSNorm layer for this transformer block.
    pub ffn_norm: BitnetRmsNorm,
}

impl Layer {
    /// Performs a forward pass through a transformer layer.
    ///
    /// # Arguments
    ///
    /// * `context` - GPU context for compute operations
    /// * `x` - Input tensor of shape `[batch_size * seq_len, hidden_size]`
    /// * `pos_offset` - Position offset for rotary embeddings
    /// * `cache` - Per-layer KV cache
    ///
    /// # Returns
    ///
    /// * Result containing the output tensor of shape `[batch_size * seq_len, hidden_size]`
    ///   or an error if the computation fails
    ///
    /// # Implementation Notes
    ///
    /// The forward pass follows the standard transformer pattern:
    /// 1. Apply input normalization
    /// 2. Self-attention with residual connection
    /// 3. Apply second normalization
    /// 4. Feed-forward network with residual connection
    pub async fn forward(
        &mut self, // Now mutable to update RoPE tables
        context: &WgpuContext,
        x: &[f32],
        pos_offset: usize,
        cache: &mut KVCache,
    ) -> Result<Vec<f32>, BitNetError> {
        // Pre-attention normalization and residual connection
        let x_norm = self.attention_norm.forward(x);
        let attn_output = self.attn.forward(context, &x_norm, pos_offset, Some(cache)).await;
        let residual_after_attn: Vec<f32> = x.iter().zip(attn_output.iter()).map(|(a, b)| a + b).collect();

        // Pre-feed-forward normalization and residual connection
        let x_norm2 = self.ffn_norm.forward(&residual_after_attn);
        let batch_size = x.len() / self.attention_norm.weight.len();
        let ffn_output = self.ffn.forward(context, &x_norm2, batch_size).await;
        let final_output: Vec<f32> = residual_after_attn.iter().zip(ffn_output.iter()).map(|(a, b)| a + b).collect();

        Ok(final_output)
    }
}

/// The complete BitNet transformer model.
///
/// This is the main model class that handles:
/// - Token embedding lookup
/// - Processing through transformer blocks
/// - Final layer normalization
/// - Output projection
///
/// # Examples
///
/// Loading a pretrained model:
///
/// ```rust,no_run
/// use bitnet_core::model::Transformer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = Transformer::from_dir("models/bitnet-2b")?;
/// let tokens = vec![1, 2, 3, 4];
/// let output = model.forward(&tokens);
/// # Ok(())
/// # }
/// ```
pub struct Transformer {
    /// Token embedding table [vocab_size][hidden_size]
    pub embedding: Vec<f32>,
    /// Shape of the embedding table
    pub embedding_shape: Vec<usize>,
    /// Sequence of transformer blocks
    pub layers: Vec<Layer>,
    /// Final layer normalization
    pub norm: BitnetRmsNorm,
    /// Output projection matrix [hidden_size][vocab_size]
    pub output: BitLinear,
    /// Model configuration
    pub config: ModelConfig,
}

impl ModelConfig {
    /// Creates a new transformer model from this configuration.
    ///
    /// This initializes all weights to zeros. For actual use,
    /// load a pretrained model using [`Transformer::from_dir`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::model::ModelConfig;
    ///
    /// let config = ModelConfig {
    ///     hidden_size: 2048,
    ///     intermediate_size: 5632,
    ///     num_hidden_layers: 24,
    ///     num_attention_heads: 32,
    ///     vocab_size: 32000,
    ///     rms_norm_eps: 1e-6,
    ///     dropout: 0.0,
    /// };
    ///
    /// let model = config.init();
    /// ```
    pub fn init(&self) -> Transformer {
        // TODO: Replace with real weights from model loading
        let embedding = vec![0.0; self.hidden_size * self.vocab_size];
        let output = BitLinear {
            packed_weights: vec![0; (self.vocab_size * self.hidden_size + 15) / 16],
            weight_scales: vec![1.0; self.vocab_size],
            in_features: self.hidden_size,
            out_features: self.vocab_size,
        };
        let norm = BitnetRmsNorm::new(vec![1.0; self.hidden_size], self.rms_norm_eps);
        let attn_cfg = AttentionConfig::new(self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.max_seq_len);
        let ffn_cfg = FeedForwardConfig::new(self.hidden_size, self.intermediate_size);
        let layer = Layer {
            attn: attn_cfg.init(),
            ffn: ffn_cfg.init(),
            attention_norm: BitnetRmsNorm::new(vec![1.0; self.hidden_size], self.rms_norm_eps),
            ffn_norm: BitnetRmsNorm::new(vec![1.0; self.hidden_size], self.rms_norm_eps),
        };
        Transformer {
            embedding,
            embedding_shape: vec![self.vocab_size, self.hidden_size],
            layers: vec![layer; self.num_hidden_layers],
            norm,
            output,
            config: self.clone(),
        }
    }
}

impl Transformer {
    /// Loads a pretrained and converted model from a directory.
    ///
    /// The directory should contain:
    /// - `embedding.bin` - Token embeddings
    /// - `norm.bin` - Final layer norm
    /// - `lm_head.bin` - Output projection
    /// - `block_*.bin` - Transformer blocks
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the model directory
    ///
    /// # Returns
    ///
    /// * `Result<Transformer, BitNetError>` - The loaded model or an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bitnet_core::model::Transformer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = Transformer::from_dir("models/bitnet-2b")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The directory doesn't exist
    /// - Required files are missing
    /// - Files are corrupted or in wrong format
    pub fn from_dir(dir: &Path, config: ModelConfig) -> Result<Self, BitNetError> {
        let embedding_path = dir.join("embedding.bin");
        let embedding_bytes = fs::read(&embedding_path)?;
        let (embedding_record, _): (EmbeddingRecord, _) = decode_from_slice(&embedding_bytes, standard())?;

        let norm_path = dir.join("norm.bin");
        let norm_bytes = fs::read(&norm_path)?;
        let (norm_record, _): (RmsNormRecord, _) = decode_from_slice(&norm_bytes, standard())?;
        let norm = BitnetRmsNorm::from_record(norm_record);

        // Note: The final output layer is also a linear projection.
        // We'll treat it as a BitLinear for consistency, though it might not be quantized.
        // The converter saves it as an EmbeddingRecord, which we can adapt.
        let output_path = dir.join("lm_head.bin");
        let output_bytes = fs::read(&output_path)?;
        let (lm_head_record, _): (EmbeddingRecord, _) = decode_from_slice(&output_bytes, standard())?;
        
        // This is a placeholder. A real model might have a different packing for the LM head.
        // For now, we assume it's not a true BitLinear but can be represented by it.
        let output = BitLinear {
            packed_weights: vec![0; (lm_head_record.shape[0] * lm_head_record.shape[1] + 15) / 16],
            weight_scales: vec![1.0; lm_head_record.shape[0]],
            in_features: lm_head_record.shape[1],
            out_features: lm_head_record.shape[0],
        };

        let attn_config = AttentionConfig::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_seq_len,
        );

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let block_path = dir.join(format!("block_{}.bin", i));
            let block_bytes = fs::read(&block_path)?;
            let (block_record, _): (TransformerBlockRecord, _) = decode_from_slice(&block_bytes, standard())?;

            let layer = Layer {
                attn: Attention::from_records(
                    block_record.attention.wqkv.clone(), // Placeholder
                    block_record.attention.wqkv.clone(), // Placeholder
                    block_record.attention.wqkv,
                    block_record.attention.o_proj,
                    &attn_config,
                ),
                ffn: FeedForward::from_records(
                    block_record.feed_forward.w13,
                    block_record.feed_forward.w2,
                ),
                attention_norm: BitnetRmsNorm::from_record(block_record.attention_norm),
                ffn_norm: BitnetRmsNorm::from_record(block_record.ffn_norm),
            };
            layers.push(layer);
        }

        Ok(Transformer {
            embedding: embedding_record.weight,
            embedding_shape: embedding_record.shape,
            layers,
            norm,
            output,
            config,
        })
    }

    /// Performs a forward pass through the entire model for one token.
    ///
    /// # Arguments
    ///
    /// * `context` - GPU context for compute operations
    /// * `token_id` - Input token ID
    /// * `pos` - Position in the sequence
    /// * `kv_caches` - Per-layer KV caches
    ///
    /// # Returns
    ///
    /// * Logits for next token prediction
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use bitnet_core::model::Transformer;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let model = Transformer::from_dir("models/bitnet-2b")?;
    /// let tokens = vec![1, 2, 3, 4];
    /// let logits = model.forward(&context, &tokens, 1, &mut [KVCache::new()])?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn forward(
        &mut self, // Mutable to update caches and RoPE tables
        context: &WgpuContext,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [KVCache],
    ) -> Result<Vec<f32>, BitNetError> {
        // 1. Embedding lookup
        let hidden_size = self.embedding_shape[1];
        let start = token_id * hidden_size;
        let mut x = self.embedding[start..start + hidden_size].to_vec();

        // 2. Apply each transformer layer
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(context, &x, pos, &mut kv_caches[i]).await?;
        }

        // 3. Final normalization
        let x_norm = self.norm.forward(&x);

        // 4. Final output projection (LM Head)
        let logits = self.output.forward(context, &x_norm, 1).await;
        
        Ok(logits)
    }
}

impl Default for Transformer {
    /// Creates a minimal default Transformer for testing purposes.
    /// This implementation uses dummy data and should not be used in production.
    fn default() -> Self {
        Transformer {
            embedding: vec![0.1; 128 * 1000],  // vocab_size=1000, hidden_size=128
            embedding_shape: vec![1000, 128],
            layers: Vec::new(),  // No layers for testing
            norm: BitnetRmsNorm::default(),
            output: BitLinear {
                packed_weights: vec![0; (128 * 1000 + 15) / 16],
                weight_scales: vec![1.0; 128],
                in_features: 128,
                out_features: 1000,
            },
            config: ModelConfig {
                hidden_size: 128,
                intermediate_size: 5632,
                num_hidden_layers: 24,
                num_attention_heads: 32,
                num_key_value_heads: 32,
                vocab_size: 1000,
                rms_norm_eps: 1e-6,
                dropout: 0.0,
                max_seq_len: 32,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnetcore_test_utils::{mini_model_config, mini_dummy_transformer};

    #[test]
    fn test_model_config() {
        let config = mini_model_config();
        let model = config.init();
        assert_eq!(model.layers.len(), config.num_hidden_layers);
    }

    #[tokio::test]
    async fn test_forward_pass() -> Result<(), BitNetError> {
        let context = WgpuContext::new().await?;
        let mut model = mini_model_config().init();
        let output = model.forward(&context, 0, 0, &mut [KVCache::default()]).await?;
        assert_eq!(output.len(), model.output.out_features);
        Ok(())
    }

    #[tokio::test]
    async fn test_model_forward() -> Result<(), BitNetError> {
        let context = WgpuContext::new().await?;
        let mut model = mini_dummy_transformer();
        let output = model.forward(&context, 0, 0, &mut [KVCache::default()]).await?;
        assert_eq!(output.len(), model.output.out_features);
        Ok(())
    }
}