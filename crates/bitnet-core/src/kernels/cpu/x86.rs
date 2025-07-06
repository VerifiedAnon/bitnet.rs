//! AVX2-optimized SIMD kernels for BitNet on x86_64 CPUs.
//!
//! This module contains `unsafe` Rust code that directly uses AVX2 intrinsics
//! to implement the high-performance Look-Up Table (LUT) based quantized
//! matrix multiplication. It should only be compiled on `x86_64` targets.

#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::error::BitNetError;

/// AVX2-accelerated quantized matrix multiplication using a Look-Up Table strategy.
#[target_feature(enable = "avx2")]
pub unsafe fn qgemm_lut_avx2(
    q_activations: &[i8],
    packed_weights: &[u32],
    activation_scales: &[f32],
    weight_scales: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, BitNetError> {
    // Output buffer
    let mut output = vec![0.0; batch_size * out_features];
    let packed_per_row = (in_features + 15) / 16;

    // For each batch (row)
    for batch_idx in 0..batch_size {
        let activation_scale = activation_scales[batch_idx];
        let activation_row = &q_activations[batch_idx * in_features..(batch_idx + 1) * in_features];

        // For each output feature (column)
        for out_idx in 0..out_features {
            let weight_scale = weight_scales[out_idx];
            let weights_start_idx = out_idx * packed_per_row;
            let packed_weight_row = &packed_weights[weights_start_idx..weights_start_idx + packed_per_row];

            let mut sum = 0i32;
            let mut k = 0;
            // SIMD: process 32 activations at a time
            while k + 31 < in_features {
                // Unpack 32 ternary weights from 2 u32s (16 weights per u32, 2 bits per weight)
                let mut weights = [0i8; 32];
                for i in 0..2 {
                    let packed = packed_weight_row[(k / 16) + i];
                    for j in 0..16 {
                        let idx = i * 16 + j;
                        if k + idx >= in_features { break; }
                        let bits = ((packed >> (j * 2)) & 0b11) as u8;
                        weights[idx] = match bits {
                            1 => 1,
                            2 => -1,
                            _ => 0,
                        };
                    }
                }
                // Load activations and weights into __m256i
                let act_ptr = activation_row[k..k+32].as_ptr();
                let wgt_ptr = weights.as_ptr();
                let act_vec = _mm256_loadu_si256(act_ptr as *const __m256i);
                let wgt_vec = _mm256_loadu_si256(wgt_ptr as *const __m256i);
                // Correct signed multiply-accumulate: convert both to i16, multiply, and horizontally sum
                let act_i16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(act_vec, 0));
                let act_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(act_vec, 1));
                let wgt_i16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wgt_vec, 0));
                let wgt_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wgt_vec, 1));
                let prod_lo = _mm256_mullo_epi16(act_i16_lo, wgt_i16_lo);
                let prod_hi = _mm256_mullo_epi16(act_i16_hi, wgt_i16_hi);
                // Horizontally sum all 16 i16 lanes in each half
                let mut buf = [0i16; 16];
                _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, prod_lo);
                sum += buf.iter().map(|&x| x as i32).sum::<i32>();
                _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, prod_hi);
                sum += buf.iter().map(|&x| x as i32).sum::<i32>();
                k += 32;
            }
            // Scalar tail
            for kk in k..in_features {
                let packed_idx = kk / 16;
                let bit_idx = (kk % 16) * 2;
                let bits = ((packed_weight_row[packed_idx] >> bit_idx) & 0b11) as u8;
                let weight_val = match bits {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
                sum += (activation_row[kk] as i32) * (weight_val as i32);
            }
            output[batch_idx * out_features + out_idx] = (sum as f32) * activation_scale * weight_scale;
        }
    }
    Ok(output)
} 