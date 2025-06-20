// Optimized Pure Rust packer for BitNet converter

use crate::source::TensorMap;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use thiserror::Error;
use std::time::Instant;
use rand::{self};

// ================================================================================================
// Error Handling
// ================================================================================================

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Missing required tensor: {tensor_name}")]
    MissingTensor { tensor_name: String },
    #[error("Invalid tensor shape for {tensor_name}: expected {expected:?}, got {actual:?}")]
    InvalidShape { tensor_name: String, expected: Vec<usize>, actual: Vec<usize> },
    #[error("Weight count {count} is not divisible by 16")]
    InvalidWeightCount { count: usize },
    #[error("Invalid ternary weight value: {value}")]
    InvalidTernaryWeight { value: i8 },
}

type Result<T> = std::result::Result<T, ConversionError>;

// ================================================================================================
// Core Data Structures (with better validation)
// ================================================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BitLinearRecord {
    pub packed_weights: Vec<u32>,
    pub weight_scales: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl BitLinearRecord {
    fn validate(&self) -> Result<()> {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RmsNormRecord {
    pub weight: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingRecord {
    pub weight: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AttentionRecord {
    pub wqkv: BitLinearRecord,
    pub o_proj: BitLinearRecord,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FeedForwardRecord {
    pub w13: BitLinearRecord,
    pub w2: BitLinearRecord,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransformerBlockRecord {
    pub attention: AttentionRecord,
    pub feed_forward: FeedForwardRecord,
    pub attention_norm: RmsNormRecord,
    pub ffn_norm: RmsNormRecord,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BitNetModelRecord {
    pub embedding: EmbeddingRecord,
    pub blocks: Vec<TransformerBlockRecord>,
    pub norm: RmsNormRecord,
    pub lm_head: EmbeddingRecord,
    pub metadata: ModelMetadata,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelMetadata {
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
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

// Optimized quantization with better memory management
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

pub fn pack_ternary_weights_optimized(weights: &[i8]) -> Result<Vec<u32>> {
    if weights.len() % 16 != 0 {
        return Err(ConversionError::InvalidWeightCount { count: weights.len() });
    }
    
    let mut packed = Vec::with_capacity(weights.len() / 16);
    
    for chunk in weights.chunks_exact(16) {
        let mut packed_val = 0u32;
        
        // Unroll loop for better performance
        for i in 0..16 {
            let encoded = match chunk[i] {
                -1 => 0u32,
                0 => 1u32,
                1 => 2u32,
                invalid => return Err(ConversionError::InvalidTernaryWeight { value: invalid }),
            };
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
    fn extract_from_map(tensor_map: &mut TensorMap, layer_idx: usize) -> Result<Self> {
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
            let tensor = tensor_map.remove(&key)
                .ok_or_else(|| ConversionError::MissingTensor { tensor_name: key })?;
            tensors.push(tensor);
        }
        
        Ok(LayerTensors {
            q: tensors[0].0.clone(), q_shape: tensors[0].1.clone(),
            k: tensors[1].0.clone(), k_shape: tensors[1].1.clone(),
            v: tensors[2].0.clone(), v_shape: tensors[2].1.clone(),
            wo: tensors[3].0.clone(), wo_shape: tensors[3].1.clone(),
            gate: tensors[4].0.clone(), gate_shape: tensors[4].1.clone(),
            up: tensors[5].0.clone(), up_shape: tensors[5].1.clone(),
            w2: tensors[6].0.clone(), w2_shape: tensors[6].1.clone(),
            attn_norm: tensors[7].0.clone(), attn_norm_shape: tensors[7].1.clone(),
            ffn_norm: tensors[8].0.clone(), ffn_norm_shape: tensors[8].1.clone(),
        })
    }
}

fn create_bit_linear_record(
    weights: Vec<f32>, 
    shape: &[usize]
) -> Result<BitLinearRecord> {
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

fn pack_layer_optimized(tensors: LayerTensors) -> Result<TransformerBlockRecord> {
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

pub struct ConversionConfig {
    pub use_parallel: bool,
    pub validate_tensors: bool,
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

pub fn convert_model_advanced(
    mut tensor_map: TensorMap,
    num_layers: usize,
    config: ConversionConfig,
) -> Result<BitNetModelRecord> {
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

fn extract_embedding(tensor_map: &mut TensorMap) -> Result<EmbeddingRecord> {
    let (weight, shape) = tensor_map.remove("model.embed_tokens.weight")
        .ok_or_else(|| ConversionError::MissingTensor { 
            tensor_name: "model.embed_tokens.weight".to_string() 
        })?;
    Ok(EmbeddingRecord { weight, shape })
}

fn extract_norm(tensor_map: &mut TensorMap) -> Result<RmsNormRecord> {
    let (weight, shape) = tensor_map.remove("model.norm.weight")
        .ok_or_else(|| ConversionError::MissingTensor { 
            tensor_name: "model.norm.weight".to_string() 
        })?;
    Ok(RmsNormRecord { weight, shape })
}

fn extract_lm_head(tensor_map: &mut TensorMap, embedding: &EmbeddingRecord) -> Result<EmbeddingRecord> {
    if let Some((weight, shape)) = tensor_map.remove("lm_head.weight") {
        Ok(EmbeddingRecord { weight, shape })
    } else if let Some((weight, shape)) = tensor_map.remove("output.weight") {
        Ok(EmbeddingRecord { weight, shape })
    } else {
        log::warn!("No lm_head.weight or output.weight found; using tied embeddings");
        Ok(embedding.clone())
    }
}

fn convert_layers_sequential(
    tensor_map: &mut TensorMap,
    num_layers: usize,
    config: &ConversionConfig,
) -> Result<Vec<TransformerBlockRecord>> {
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
    tensor_map: &mut TensorMap,
    num_layers: usize,
    config: &ConversionConfig,
) -> Result<Vec<TransformerBlockRecord>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let mut layer_tensors = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let tensors = LayerTensors::extract_from_map(tensor_map, i)?;
        layer_tensors.push((i, tensors));
    }
    let counter = Arc::new(AtomicUsize::new(0));
    let results: Result<Vec<_>> = layer_tensors
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

// Convenience function for backward compatibility
pub fn convert_model(
    tensor_map: TensorMap,
    num_layers: usize,
    use_parallel: bool,
) -> Result<BitNetModelRecord> {
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
                    0 => -1,
                    1 => 0,
                    2 => 1,
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

    fn create_minimal_tensor_map() -> TensorMap {
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
            tensor_map.insert(key.to_string(), (data.clone(), shape.clone()));
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