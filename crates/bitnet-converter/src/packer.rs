//! Model quantization, packing, and serialization utilities for BitNet Converter.
//!
//! Provides optimized routines for quantizing, packing, and serializing model weights and metadata.

use crate::source::{RichTensorMap, TensorData};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use thiserror::Error;
use std::time::Instant;
use half::bf16;
use bytemuck;
use safetensors::{tensor::TensorView, serialize_to_file};
use std::collections::BTreeMap;


// ================================================================================================
// Error Handling
// ================================================================================================

/// Error type for model conversion and packing operations.
#[derive(Error, Debug)]
pub enum ConversionError {
    /// Required tensor is missing from the input.
    #[error("Missing required tensor: {tensor_name}")]
    MissingTensor {
        /// Name of the missing tensor.
        tensor_name: String,
    },
    /// Tensor shape does not match expected dimensions.
    #[error("Invalid tensor shape for {tensor_name}: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        /// Name of the tensor with invalid shape.
        tensor_name: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape found.
        actual: Vec<usize>,
    },
    /// Weight count is not divisible by 16 (required for packing).
    #[error("Weight count {count} is not divisible by 16")]
    InvalidWeightCount {
        /// Number of weights found.
        count: usize,
    },
    /// Ternary weight value is invalid (must be -1, 0, or 1).
    #[error("Invalid ternary weight value: {value}")]
    InvalidTernaryWeight {
        /// The invalid ternary value encountered.
        value: i8,
    },
}

/// Result type for conversion operations.
pub type ConvResult<T> = std::result::Result<T, ConversionError>;

// ================================================================================================
// Core Data Structures (with better validation)
// ================================================================================================

/// Packed linear layer weights and scales for BitNet.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitLinearRecord {
    /// Packed weights (ternary, 16 per u32).
    pub packed_weights: Vec<u32>,
    /// Per-row scaling factors.
    pub weight_scales: Vec<f32>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl BitLinearRecord {
    /// Validate the packed weights and shape.
    fn validate(&self) -> ConvResult<()> {
        let expected_packed_len = (self.out_features * self.in_features + 15) / 16;
        if self.packed_weights.len() != expected_packed_len {
            return Err(ConversionError::InvalidShape {
                tensor_name: "packed_weights".to_string(),
                expected: vec![expected_packed_len],
                actual: vec![self.packed_weights.len()],
            });
        }
        Ok(())
    }
}

/// Packed RMSNorm weights and shape.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RmsNormRecord {
    /// Norm weights.
    pub weight: Vec<f32>,
    /// Shape of the norm tensor.
    pub shape: Vec<usize>,
}

/// Packed embedding weights and shape.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingRecord {
    /// Embedding weights.
    pub weight: Vec<f32>,
    /// Shape of the embedding tensor.
    pub shape: Vec<usize>,
}

/// Packed attention layer record.
#[derive(Serialize, Deserialize, Debug)]
pub struct AttentionRecord {
    /// Packed QKV weights and scales.
    pub wqkv: BitLinearRecord,
    /// Packed output projection weights and scales.
    pub o_proj: BitLinearRecord,
}

/// Packed feed-forward layer record.
#[derive(Serialize, Deserialize, Debug)]
pub struct FeedForwardRecord {
    /// Packed gate and up projection weights and scales.
    pub w13: BitLinearRecord,
    /// Packed down projection weights and scales.
    pub w2: BitLinearRecord,
}

/// Packed transformer block record.
#[derive(Serialize, Deserialize, Debug)]
pub struct TransformerBlockRecord {
    /// Attention layer record.
    pub attention: AttentionRecord,
    /// Feed-forward layer record.
    pub feed_forward: FeedForwardRecord,
    /// Attention norm record.
    pub attention_norm: RmsNormRecord,
    /// Feed-forward norm record.
    pub ffn_norm: RmsNormRecord,
}

/// Top-level model record for BitNet.
#[derive(Serialize, Deserialize, Debug)]
pub struct BitNetModelRecord {
    /// Embedding record.
    pub embedding: EmbeddingRecord,
    /// All transformer blocks.
    pub blocks: Vec<TransformerBlockRecord>,
    /// Final norm record.
    pub norm: RmsNormRecord,
    /// LM head record.
    pub lm_head: EmbeddingRecord,
    /// Model metadata.
    pub metadata: ModelMetadata,
}

/// Metadata for a BitNet model.
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelMetadata {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Timestamp of conversion.
    pub conversion_timestamp: u64,
}

// ================================================================================================
// SIMD-Optimized Quantization (with fallback)
// ================================================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-optimized quantization for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_simd(row: &[f32], scale: f32, output: &mut [i8]) {
    if is_x86_feature_detected!("avx2") && row.len() >= 8 {
        let scale_vec = _mm256_set1_ps(1.0 / scale);
        let chunks = row.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let data = _mm256_loadu_ps(row.as_ptr().add(offset));
            let scaled = _mm256_mul_ps(data, scale_vec);
            let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
            
            // Convert to i32 and clamp to [-1, 1]
            let as_i32 = _mm256_cvtps_epi32(rounded);
            let clamped = _mm256_max_epi32(_mm256_set1_epi32(-1), 
                         _mm256_min_epi32(as_i32, _mm256_set1_epi32(1)));
            
            // Extract and store as i8
            let result = [
                _mm256_extract_epi32(clamped, 0) as i8,
                _mm256_extract_epi32(clamped, 1) as i8,
                _mm256_extract_epi32(clamped, 2) as i8,
                _mm256_extract_epi32(clamped, 3) as i8,
                _mm256_extract_epi32(clamped, 4) as i8,
                _mm256_extract_epi32(clamped, 5) as i8,
                _mm256_extract_epi32(clamped, 6) as i8,
                _mm256_extract_epi32(clamped, 7) as i8,
            ];
            output[offset..offset + 8].copy_from_slice(&result);
        }
        
        // Handle remainder
        for i in (chunks * 8)..row.len() {
            let v = (row[i] / scale).round().clamp(-1.0, 1.0);
            output[i] = v as i8;
        }
    } else {
        quantize_row_fallback(row, scale, output);
    }
}

// Fallback implementation
fn quantize_row_fallback(row: &[f32], scale: f32, output: &mut [i8]) {
    for (i, &val) in row.iter().enumerate() {
        let v = (val / scale).round().clamp(-1.0, 1.0);
        output[i] = v as i8;
    }
}

// Cross-platform quantization function
fn quantize_row(row: &[f32], scale: f32, output: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { quantize_row_simd(row, scale, output) };
            return;
        }
    }
    quantize_row_fallback(row, scale, output);
}

/// Optimized quantization with better memory management
///
/// Quantizes a tensor to 1.58-bit ternary weights and returns the quantized values and per-row scales.
pub fn quantize_to_1_58_bit_optimized(tensor: &[f32], shape: &[usize]) -> (Vec<i8>, Vec<f32>) {
    let n = shape[0];
    let k = shape[1];
    
    // Pre-allocate with exact capacity
    let mut all_quantized = vec![0i8; n * k];
    let mut scales = Vec::with_capacity(n);
    
    // Process in parallel if beneficial
    if n >= 4 && k >= 64 {
        tensor.par_chunks(k)
            .zip(all_quantized.par_chunks_mut(k))
            .enumerate()
            .for_each(|(_, (row, output_row))| {
                // Calculate scale with better numerical stability
                let mean_abs = row.iter().map(|x| x.abs()).sum::<f32>() / k as f32;
                let scale = if mean_abs < f32::EPSILON { 1.0 } else { mean_abs };
                
                quantize_row(row, scale, output_row);
                
                // Store scale (need to handle thread safety)
                // This is a compromise - we'll collect scales after
            });
        
        // Recalculate scales (could be optimized further with thread-local storage)
        for i in 0..n {
            let row = &tensor[i * k..(i + 1) * k];
            let mean_abs = row.iter().map(|x| x.abs()).sum::<f32>() / k as f32;
            scales.push(if mean_abs < f32::EPSILON { 1.0 } else { mean_abs });
        }
    } else {
        // Sequential processing for small tensors
        for i in 0..n {
            let row = &tensor[i * k..(i + 1) * k];
            let output_row = &mut all_quantized[i * k..(i + 1) * k];
            
            let mean_abs = row.iter().map(|x| x.abs()).sum::<f32>() / k as f32;
            let scale = if mean_abs < f32::EPSILON { 1.0 } else { mean_abs };
            
            quantize_row(row, scale, output_row);
            scales.push(scale);
        }
    }
    
    (all_quantized, scales)
}

// ================================================================================================
// Optimized Packing with Better Error Handling
// ================================================================================================

/// Packs ternary weights into u32s for BitNet kernels.
pub fn pack_ternary_weights_optimized(weights: &[i8]) -> ConvResult<Vec<u32>> {
    if weights.len() % 16 != 0 {
        // To make this more robust for models where K is not a multiple of 16,
        // we can pad the input weights. For now, we keep the strict check.
        // A more robust implementation would pad `weights` with 0s to a multiple of 16 here.
        return Err(ConversionError::InvalidWeightCount { count: weights.len() });
    }
    
    let mut packed = Vec::with_capacity(weights.len() / 16);
    
    for chunk in weights.chunks_exact(16) {
        let mut packed_val = 0u32;
        
        // Unroll loop for better performance
        for i in 0..16 {
            // CORRECTED: This mapping now matches the WGSL kernel's decoding logic.
            // WGSL decode: 1 -> +1, 2 -> -1, 0/3 -> 0
            let encoded = match chunk[i] {
                1 => 1u32,  // 01
               -1 => 2u32,  // 10
                0 => 0u32,  // 00
                invalid => return Err(ConversionError::InvalidTernaryWeight { value: invalid }),
            };
            // Pack with LSB-first ordering to match the kernel
            packed_val |= encoded << (i * 2);
        }
        
        packed.push(packed_val);
    }
    
    Ok(packed)
}

// ================================================================================================
// Layer Processing with Better Resource Management
// ================================================================================================

struct LayerTensors {
    q: Vec<f32>, q_shape: Vec<usize>,
    k: Vec<f32>, k_shape: Vec<usize>,
    v: Vec<f32>, v_shape: Vec<usize>,
    wo: Vec<f32>, wo_shape: Vec<usize>,
    gate: Vec<f32>, gate_shape: Vec<usize>,
    up: Vec<f32>, up_shape: Vec<usize>,
    w2: Vec<f32>, w2_shape: Vec<usize>,
    attn_norm: Vec<f32>, attn_norm_shape: Vec<usize>,
    ffn_norm: Vec<f32>, ffn_norm_shape: Vec<usize>,
}

impl LayerTensors {
    fn extract_from_map(tensor_map: &mut RichTensorMap, layer_idx: usize) -> ConvResult<Self> {
        let keys = [
            ("q_proj", "self_attn.q_proj.weight"),
            ("k_proj", "self_attn.k_proj.weight"), 
            ("v_proj", "self_attn.v_proj.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("gate_proj", "mlp.gate_proj.weight"),
            ("up_proj", "mlp.up_proj.weight"),
            ("down_proj", "mlp.down_proj.weight"),
            ("attn_norm", "input_layernorm.weight"),
            ("ffn_norm", "post_attention_layernorm.weight"),
        ];
        let mut tensors = Vec::new();
        for (_, suffix) in &keys {
            let key = format!("model.layers.{}.{}", layer_idx, suffix);
            let (data, shape) = tensor_map.remove(&key)
                .ok_or_else(|| ConversionError::MissingTensor { tensor_name: key })?;
            tensors.push((data, shape));
        }
        Ok(LayerTensors {
            q: tensors[0].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[0].1.to_string(), expected: vec![], actual: vec![] })?.clone(), q_shape: tensors[0].1.clone(),
            k: tensors[1].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[1].1.to_string(), expected: vec![], actual: vec![] })?.clone(), k_shape: tensors[1].1.clone(),
            v: tensors[2].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[2].1.to_string(), expected: vec![], actual: vec![] })?.clone(), v_shape: tensors[2].1.clone(),
            wo: tensors[3].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[3].1.to_string(), expected: vec![], actual: vec![] })?.clone(), wo_shape: tensors[3].1.clone(),
            gate: tensors[4].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[4].1.to_string(), expected: vec![], actual: vec![] })?.clone(), gate_shape: tensors[4].1.clone(),
            up: tensors[5].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[5].1.to_string(), expected: vec![], actual: vec![] })?.clone(), up_shape: tensors[5].1.clone(),
            w2: tensors[6].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[6].1.to_string(), expected: vec![], actual: vec![] })?.clone(), w2_shape: tensors[6].1.clone(),
            attn_norm: tensors[7].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[7].1.to_string(), expected: vec![], actual: vec![] })?.clone(), attn_norm_shape: tensors[7].1.clone(),
            ffn_norm: tensors[8].0.as_f32_vec().ok_or_else(|| ConversionError::InvalidShape { tensor_name: keys[8].1.to_string(), expected: vec![], actual: vec![] })?.clone(), ffn_norm_shape: tensors[8].1.clone(),
        })
    }
}

fn create_bit_linear_record(
    weights: Vec<f32>, 
    shape: &[usize]
) -> ConvResult<BitLinearRecord> {
    let (quantized, scales) = quantize_to_1_58_bit_optimized(&weights, shape);
    let packed = pack_ternary_weights_optimized(&quantized)?;
    
    let record = BitLinearRecord {
        packed_weights: packed,
        weight_scales: scales,
        in_features: shape[1],
        out_features: shape[0],
    };
    
    record.validate()?;
    Ok(record)
}

fn pack_layer_optimized(tensors: LayerTensors) -> ConvResult<TransformerBlockRecord> {
    // Concatenate Q, K, V efficiently
    let total_qkv_rows = tensors.q_shape[0] + tensors.k_shape[0] + tensors.v_shape[0];
    let k_dim = tensors.q_shape[1];
    
    let mut wqkv = Vec::with_capacity(total_qkv_rows * k_dim);
    wqkv.extend_from_slice(&tensors.q);
    wqkv.extend_from_slice(&tensors.k);
    wqkv.extend_from_slice(&tensors.v);
    
    let wqkv_record = create_bit_linear_record(wqkv, &[total_qkv_rows, k_dim])?;
    
    // Output projection
    let o_proj_record = create_bit_linear_record(tensors.wo, &tensors.wo_shape)?;
    
    // Concatenate gate and up projections
    let total_w13_rows = tensors.gate_shape[0] + tensors.up_shape[0];
    let mut w13 = Vec::with_capacity(total_w13_rows * tensors.gate_shape[1]);
    w13.extend_from_slice(&tensors.gate);
    w13.extend_from_slice(&tensors.up);
    
    let w13_record = create_bit_linear_record(w13, &[total_w13_rows, tensors.gate_shape[1]])?;
    
    // Down projection
    let w2_record = create_bit_linear_record(tensors.w2, &tensors.w2_shape)?;
    
    // Layer norms (no quantization needed)
    let attn_norm_record = RmsNormRecord { 
        weight: tensors.attn_norm, 
        shape: tensors.attn_norm_shape 
    };
    let ffn_norm_record = RmsNormRecord { 
        weight: tensors.ffn_norm, 
        shape: tensors.ffn_norm_shape 
    };
    
    Ok(TransformerBlockRecord {
        attention: AttentionRecord { 
            wqkv: wqkv_record, 
            o_proj: o_proj_record 
        },
        feed_forward: FeedForwardRecord { 
            w13: w13_record, 
            w2: w2_record 
        },
        attention_norm: attn_norm_record,
        ffn_norm: ffn_norm_record,
    })
}

// ================================================================================================
// Main Conversion Functions with Better Architecture
// ================================================================================================

/// Configuration for model conversion.
pub struct ConversionConfig {
    /// Whether to use parallel processing for layer conversion.
    pub use_parallel: bool,
    /// Whether to validate tensors during conversion.
    pub validate_tensors: bool,
    /// Optional callback for reporting progress: (current_layer, total_layers).
    pub progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            validate_tensors: true,
            progress_callback: None,
        }
    }
}

/// Converts a tensor map into a BitNetModelRecord using advanced options.
pub fn convert_model_advanced(
    mut tensor_map: RichTensorMap,
    num_layers: usize,
    config: ConversionConfig,
) -> ConvResult<BitNetModelRecord> {
    // Extract top-level tensors with better error handling
    let embedding = extract_embedding(&mut tensor_map)?;
    let norm = extract_norm(&mut tensor_map)?;
    let lm_head = extract_lm_head(&mut tensor_map, &embedding)?;
    
    let metadata = ModelMetadata {
        num_layers,
        vocab_size: embedding.shape[0],
        hidden_size: embedding.shape[1],
        conversion_timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    // Process layers
    let blocks = if config.use_parallel && num_layers > 1 {
        convert_layers_parallel(&mut tensor_map, num_layers, &config)?
    } else {
        convert_layers_sequential(&mut tensor_map, num_layers, &config)?
    };
    
    Ok(BitNetModelRecord {
        embedding,
        blocks,
        norm,
        lm_head,
        metadata,
    })
}

fn extract_embedding(tensor_map: &mut RichTensorMap) -> ConvResult<EmbeddingRecord> {
    let (data, shape) = tensor_map.remove("model.embed_tokens.weight")
        .ok_or_else(|| ConversionError::MissingTensor { 
            tensor_name: "model.embed_tokens.weight".to_string() 
        })?;
    let weight = match data {
        TensorData::F32(v) | TensorData::BF16(v) => v,
        _ => return Err(ConversionError::InvalidShape {
            tensor_name: "model.embed_tokens.weight".to_string(),
            expected: vec![],
            actual: vec![],
        }),
    };
    Ok(EmbeddingRecord { weight, shape })
}

fn extract_norm(tensor_map: &mut RichTensorMap) -> ConvResult<RmsNormRecord> {
    let (data, shape) = tensor_map.remove("model.norm.weight")
        .ok_or_else(|| ConversionError::MissingTensor { 
            tensor_name: "model.norm.weight".to_string() 
        })?;
    let weight = match data {
        TensorData::F32(v) | TensorData::BF16(v) => v,
        _ => return Err(ConversionError::InvalidShape {
            tensor_name: "model.norm.weight".to_string(),
            expected: vec![],
            actual: vec![],
        }),
    };
    Ok(RmsNormRecord { weight, shape })
}

fn extract_lm_head(tensor_map: &mut RichTensorMap, embedding: &EmbeddingRecord) -> ConvResult<EmbeddingRecord> {
    if let Some((data, shape)) = tensor_map.remove("lm_head.weight") {
        let weight = match data {
            TensorData::F32(v) | TensorData::BF16(v) => v,
            _ => return Err(ConversionError::InvalidShape {
                tensor_name: "lm_head.weight".to_string(),
                expected: vec![],
                actual: vec![],
            }),
        };
        Ok(EmbeddingRecord { weight, shape })
    } else if let Some((data, shape)) = tensor_map.remove("output.weight") {
        let weight = match data {
            TensorData::F32(v) | TensorData::BF16(v) => v,
            _ => return Err(ConversionError::InvalidShape {
                tensor_name: "output.weight".to_string(),
                expected: vec![],
                actual: vec![],
            }),
        };
        Ok(EmbeddingRecord { weight, shape })
    } else {
        log::warn!("No lm_head.weight or output.weight found; using tied embeddings");
        Ok(embedding.clone())
    }
}

fn convert_layers_sequential(
    tensor_map: &mut RichTensorMap,
    num_layers: usize,
    config: &ConversionConfig,
) -> ConvResult<Vec<TransformerBlockRecord>> {
    let mut blocks = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if let Some(ref callback) = config.progress_callback {
            callback(i + 1, num_layers);
        }
        let t_layer = Instant::now();
        let tensors = LayerTensors::extract_from_map(tensor_map, i)?;
        let block = pack_layer_optimized(tensors)?;
        println!("[CONVERT] Layer {} packed in {:.2?}", i + 1, t_layer.elapsed());
        blocks.push(block);
    }
    Ok(blocks)
}

fn convert_layers_parallel(
    tensor_map: &mut RichTensorMap,
    num_layers: usize,
    config: &ConversionConfig,
) -> ConvResult<Vec<TransformerBlockRecord>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let mut layer_tensors = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let tensors = LayerTensors::extract_from_map(tensor_map, i)?;
        layer_tensors.push((i, tensors));
    }
    let counter = Arc::new(AtomicUsize::new(0));
    let results: ConvResult<Vec<_>> = layer_tensors
        .into_par_iter()
        .map(|(i, tensors)| {
            let t_layer = Instant::now();
            let block = pack_layer_optimized(tensors)?;
            println!("[CONVERT] Layer {} packed in {:.2?}", i + 1, t_layer.elapsed());
            if let Some(ref callback) = config.progress_callback {
                let count = counter.fetch_add(1, Ordering::SeqCst) + 1;
                callback(count, num_layers);
            }
            Ok((i, block))
        })
        .collect();
    let mut blocks: Vec<_> = results?;
    blocks.sort_by_key(|(i, _)| *i);
    Ok(blocks.into_iter().map(|(_, block)| block).collect())
}

/// Convenience function for backward compatibility: converts a tensor map to a BitNetModelRecord.
pub fn convert_model(
    tensor_map: RichTensorMap,
    num_layers: usize,
    use_parallel: bool,
) -> ConvResult<BitNetModelRecord> {
    let config = ConversionConfig {
        use_parallel,
        validate_tensors: true,
        progress_callback: Some(Box::new(|current, total| {
            println!("[CONVERT] Processing layer {}/{}", current, total);
        })),
    };
    
    convert_model_advanced(tensor_map, num_layers, config)
}

// ================================================================================================
// Enhanced Testing Suite
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::collections::HashMap;

    fn reverse_packing(packed_data: &[u32]) -> Vec<i8> {
        let mut i8_data = Vec::new();
        for &packed_u32 in packed_data {
            for i in 0..16 {
                let two_bits = ((packed_u32 >> (i * 2)) & 3) as u8;
                let val_i8 = match two_bits {
                    0 => 0,   // 00 -> 0
                    1 => 1,   // 01 -> 1
                    2 => -1,  // 10 -> -1
                    _ => 0,
                };
                i8_data.push(val_i8);
            }
        }
        i8_data
    }

    #[test]
    fn test_optimized_quantization_correctness() {
        let n = 32;
        let k = 64;
        let mut rng = rand::thread_rng();
        
        let tensor: Vec<f32> = (0..n * k)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();
        
        let (original_quant, _) = quantize_to_1_58_bit_optimized(&tensor, &[n, k]);
        let packed = pack_ternary_weights_optimized(&original_quant).unwrap();
        let mut recovered = reverse_packing(&packed);
        recovered.truncate(original_quant.len());
        
        assert_eq!(recovered, original_quant);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid weight count
        let invalid_weights = vec![0i8; 15]; // Not divisible by 16
        assert!(pack_ternary_weights_optimized(&invalid_weights).is_err());
        
        // Test invalid ternary weight
        let invalid_weights = vec![0i8, 1i8, -1i8, 2i8, 0i8, 0i8, 0i8, 0i8, 
                                   0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8, 0i8];
        assert!(pack_ternary_weights_optimized(&invalid_weights).is_err());
    }

    #[test]
    fn test_conversion_with_config() {
        let tensor_map = create_minimal_tensor_map();
        
        let config = ConversionConfig {
            use_parallel: false,
            validate_tensors: true,
            progress_callback: None,
        };
        
        let result = convert_model_advanced(tensor_map, 1, config);
        assert!(result.is_ok());
        
        let model = result.unwrap();
        assert_eq!(model.blocks.len(), 1);
        assert_eq!(model.metadata.num_layers, 1);
    }

    fn create_minimal_tensor_map() -> RichTensorMap {
        let mut tensor_map = HashMap::new();
        let shape = vec![2, 16];
        let data = vec![0.1f32; 32];
        let keys = [
            "model.embed_tokens.weight",
            "model.norm.weight", 
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight", 
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        for key in &keys {
            tensor_map.insert(key.to_string(), (TensorData::F32(data.clone()), shape.clone()));
        }
        tensor_map
    }

    #[test]
    fn test_simd_vs_fallback_consistency() {
        let row = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.9, -0.4];
        let scale = 0.5;
        
        let mut simd_result = vec![0i8; 8];
        let mut fallback_result = vec![0i8; 8];
        
        quantize_row_fallback(&row, scale, &mut fallback_result);
        quantize_row(&row, scale, &mut simd_result);
        
        assert_eq!(simd_result, fallback_result);
    }
}

/// Quantize and pack all tensors in a tensor map, preserving original keys for safetensors export.
pub fn quantize_tensor_map_preserve_keys(
    tensor_map: &crate::source::RichTensorMap,
) -> std::collections::BTreeMap<String, (Vec<u8>, Vec<usize>)> {
    let mut out = std::collections::BTreeMap::new();
    for (key, (data, shape)) in tensor_map.iter() {
        // For now, just convert f32 to bf16 bytes (no quantization for simplicity)
        // If you want to quantize, add logic here per tensor type
        if let Some(f32_data) = data.as_f32_vec() {
            let bf16_vec: Vec<half::bf16> = f32_data.iter().map(|&f| half::bf16::from_f32(f)).collect();
            let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
            let bytes = bytemuck::cast_slice(&u16_vec).to_vec();
            out.insert(key.clone(), (bytes, shape.clone()));
        } else if let TensorData::U32(u32_data) = data {
            let bytes = bytemuck::cast_slice(&u32_data).to_vec();
            out.insert(key.clone(), (bytes, shape.clone()));
        } else if let TensorData::I32(i32_data) = data {
            let bytes = bytemuck::cast_slice(&i32_data).to_vec();
            out.insert(key.clone(), (bytes, shape.clone()));
        } else if let TensorData::I8(i8_data) = data {
            let bytes = bytemuck::cast_slice(&i8_data).to_vec();
            out.insert(key.clone(), (bytes, shape.clone()));
        } else if let TensorData::U8(u8_data) = data {
            out.insert(key.clone(), (u8_data.clone(), shape.clone()));
        } else {
            log::warn!("quantize_tensor_map_preserve_keys: Skipping tensor '{}' with unsupported type", key);
        }
    }
    out
}

/// Export a quantized BitNet model to a safetensors file.
pub fn export_quantized_to_safetensors(
    model: &BitNetModelRecord,
    out_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use safetensors::tensor::Dtype;
    let mut tensor_specs: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    // Embedding
    tensor_specs.push((
        "tok_embeddings.weight".to_string(),
        Dtype::BF16,
        model.embedding.shape.clone(),
        {
            let bf16_vec: Vec<bf16> = model.embedding.weight.iter().map(|&f| bf16::from_f32(f)).collect();
            let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
            bytemuck::cast_slice(&u16_vec).to_vec()
        },
    ));
    // Norm
    tensor_specs.push((
        "norm.weight".to_string(),
        Dtype::BF16,
        model.norm.shape.clone(),
        {
            let bf16_vec: Vec<bf16> = model.norm.weight.iter().map(|&f| bf16::from_f32(f)).collect();
            let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
            bytemuck::cast_slice(&u16_vec).to_vec()
        },
    ));
    // LM Head
    tensor_specs.push((
        "output.weight".to_string(),
        Dtype::BF16,
        model.lm_head.shape.clone(),
        {
            let bf16_vec: Vec<bf16> = model.lm_head.weight.iter().map(|&f| bf16::from_f32(f)).collect();
            let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
            bytemuck::cast_slice(&u16_vec).to_vec()
        },
    ));

    for (i, block) in model.blocks.iter().enumerate() {
        // Attention wqkv
        tensor_specs.push((
            format!("layers.{}.attention.wqkv.weight", i),
            Dtype::U32,
            vec![block.attention.wqkv.out_features, block.attention.wqkv.in_features / 16],
            bytemuck::cast_slice(&block.attention.wqkv.packed_weights).to_vec(),
        ));
        tensor_specs.push((
            format!("layers.{}.attention.wqkv.weight_scale", i),
            Dtype::BF16,
            vec![block.attention.wqkv.weight_scales.len()],
            {
                let bf16_vec: Vec<bf16> = block.attention.wqkv.weight_scales.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
        // Attention o_proj
        tensor_specs.push((
            format!("layers.{}.attention.wo.weight", i),
            Dtype::U32,
            vec![block.attention.o_proj.out_features, block.attention.o_proj.in_features / 16],
            bytemuck::cast_slice(&block.attention.o_proj.packed_weights).to_vec(),
        ));
        tensor_specs.push((
            format!("layers.{}.attention.wo.weight_scale", i),
            Dtype::BF16,
            vec![block.attention.o_proj.weight_scales.len()],
            {
                let bf16_vec: Vec<bf16> = block.attention.o_proj.weight_scales.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
        // FeedForward w13
        tensor_specs.push((
            format!("layers.{}.feed_forward.w13.weight", i),
            Dtype::U32,
            vec![block.feed_forward.w13.out_features, block.feed_forward.w13.in_features / 16],
            bytemuck::cast_slice(&block.feed_forward.w13.packed_weights).to_vec(),
        ));
        tensor_specs.push((
            format!("layers.{}.feed_forward.w13.weight_scale", i),
            Dtype::BF16,
            vec![block.feed_forward.w13.weight_scales.len()],
            {
                let bf16_vec: Vec<bf16> = block.feed_forward.w13.weight_scales.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
        // FeedForward w2
        tensor_specs.push((
            format!("layers.{}.feed_forward.w2.weight", i),
            Dtype::U32,
            vec![block.feed_forward.w2.out_features, block.feed_forward.w2.in_features / 16],
            bytemuck::cast_slice(&block.feed_forward.w2.packed_weights).to_vec(),
        ));
        tensor_specs.push((
            format!("layers.{}.feed_forward.w2.weight_scale", i),
            Dtype::BF16,
            vec![block.feed_forward.w2.weight_scales.len()],
            {
                let bf16_vec: Vec<bf16> = block.feed_forward.w2.weight_scales.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
        // Norms
        tensor_specs.push((
            format!("layers.{}.attention_norm.weight", i),
            Dtype::BF16,
            block.attention_norm.shape.clone(),
            {
                let bf16_vec: Vec<bf16> = block.attention_norm.weight.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
        tensor_specs.push((
            format!("layers.{}.ffn_norm.weight", i),
            Dtype::BF16,
            block.ffn_norm.shape.clone(),
            {
                let bf16_vec: Vec<bf16> = block.ffn_norm.weight.iter().map(|&f| bf16::from_f32(f)).collect();
                let u16_vec: Vec<u16> = bf16_vec.iter().map(|b| b.to_bits()).collect();
                bytemuck::cast_slice(&u16_vec).to_vec()
            },
        ));
    }

    // Second pass: build tensor views and insert into map
    let mut tensors = BTreeMap::new();
    let mut backing: Vec<Vec<u8>> = Vec::new();
    let mut meta: Vec<(String, Dtype, Vec<usize>)> = Vec::new();
    for (name, dtype, shape, bytes) in tensor_specs {
        backing.push(bytes);
        meta.push((name, dtype, shape));
    }
    for (i, (name, dtype, shape)) in meta.into_iter().enumerate() {
        let buf_ref = &backing[i];
        tensors.insert(
            name,
            TensorView::new(dtype, shape, buf_ref).map_err(|e| Box::<dyn std::error::Error>::from(e))?,
        );
    }
    serialize_to_file(&tensors, &None, out_path).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    Ok(())
}

impl TensorData {
    /// Returns a reference to the underlying f32 vector, if present.
    pub fn as_f32_vec(&self) -> Option<&Vec<f32>> {
        match self {
            TensorData::F32(v) | TensorData::BF16(v) => Some(v),
            _ => None,
        }
    }
}