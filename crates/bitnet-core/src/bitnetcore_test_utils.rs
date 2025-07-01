//! Centralized test utilities for BitNet core.
//!
//! Provides a consistent, robust mini-model config and dummy model generator for all tests.

use crate::model::{ModelConfig, Transformer};
use crate::attention::AttentionConfig;
use crate::feed_forward::FeedForwardConfig;
use crate::rms_norm::BitnetRmsNorm;
use crate::bitnet_linear::BitLinear;
use crate::model::Layer;

/// Returns a small, realistic ModelConfig for testing.
pub fn mini_model_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        vocab_size: 8,
        rms_norm_eps: 1e-6,
        dropout: 0.0,
        max_seq_len: 8,
    }
}

/// Returns a Transformer with all weights, shapes, and configs matching the mini config, using robust dummy initializations.
pub fn mini_dummy_transformer() -> Transformer {
    let config = mini_model_config();
    let embedding = vec![0.0; config.hidden_size * config.vocab_size];
    let output = BitLinear {
        packed_weights: vec![0; (config.vocab_size * config.hidden_size + 15) / 16],
        weight_scales: vec![1.0; config.vocab_size],
        in_features: config.hidden_size,
        out_features: config.vocab_size,
    };
    let norm = BitnetRmsNorm::new(vec![1.0; config.hidden_size], config.rms_norm_eps);
    let attn_cfg = AttentionConfig::new(config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.max_seq_len);
    let ffn_cfg = FeedForwardConfig::new(config.hidden_size, config.intermediate_size);
    let layer = Layer {
        attn: attn_cfg.init(),
        ffn: ffn_cfg.init(),
        attention_norm: BitnetRmsNorm::new(vec![1.0; config.hidden_size], config.rms_norm_eps),
        ffn_norm: BitnetRmsNorm::new(vec![1.0; config.hidden_size], config.rms_norm_eps),
    };
    Transformer {
        embedding,
        embedding_shape: vec![config.vocab_size, config.hidden_size],
        layers: vec![layer; config.num_hidden_layers],
        norm,
        output,
        config,
    }
} 