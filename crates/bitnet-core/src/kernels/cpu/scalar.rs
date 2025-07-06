//! Scalar (non-SIMD) reference implementation for BitNet CPU kernels.
//!
//! This implementation is used as a fallback for platforms without SIMD support
//! and as the "ground truth" for validating the correctness of the SIMD kernels.

use crate::error::BitNetError;
use rayon::prelude::*;

/// Scalar implementation of quantized matrix multiplication (QGEMM).
/// This is the reference implementation for all SIMD kernels.
pub fn qgemm_lut_scalar(
    q_activations: &[i8],
    packed_weights: &[u32],
    activation_scales: &[f32],
    weight_scales: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, BitNetError> {
    let mut output = vec![0.0; batch_size * out_features];
    let packed_per_row = (in_features + 15) / 16;

    output
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(batch_idx, output_chunk)| {
            let activation_scale = activation_scales[batch_idx];
            let activation_row = &q_activations[batch_idx * in_features..(batch_idx + 1) * in_features];
            for out_idx in 0..out_features {
                let weight_scale = weight_scales[out_idx];
                let weights_start_idx = out_idx * packed_per_row;
                let packed_weight_row = &packed_weights[weights_start_idx..weights_start_idx + packed_per_row];
                let mut sum: i32 = 0;
                for k_outer in 0..packed_per_row {
                    let packed_val = packed_weight_row[k_outer];
                    for k_inner in 0..16 {
                        let act_idx = k_outer * 16 + k_inner;
                        if act_idx >= in_features { break; }
                        let two_bits = (packed_val >> (k_inner * 2)) & 0b11;
                        let weight_val = match two_bits {
                            1 => 1,
                            2 => -1,
                            _ => 0,
                        };
                        sum += (activation_row[act_idx] as i32) * weight_val;
                    }
                }
                output_chunk[out_idx] = (sum as f32) * activation_scale * weight_scale;
            }
        });
    Ok(output)
} 