//! NEON-optimized SIMD kernels for BitNet on AArch64 CPUs.
//!
//! This module contains `unsafe` Rust code that uses NEON intrinsics. It is the
//! ARM equivalent of the x86 AVX2 kernels and follows the same LUT strategy.

#![cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use crate::error::BitNetError;

/// NEON-accelerated quantized matrix multiplication using a Look-Up Table strategy.
#[target_feature(enable = "neon")]
pub unsafe fn qgemm_lut_neon(
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

    for batch_idx in 0..batch_size {
        let activation_scale = activation_scales[batch_idx];
        let activation_row = &q_activations[batch_idx * in_features..(batch_idx + 1) * in_features];
        for out_idx in 0..out_features {
            let weight_scale = weight_scales[out_idx];
            let weights_start_idx = out_idx * packed_per_row;
            let packed_weight_row = &packed_weights[weights_start_idx..weights_start_idx + packed_per_row];
            let mut sum = 0i32;
            let mut k = 0;
            // SIMD: process 16 activations at a time
            while k + 15 < in_features {
                // Unpack 16 ternary weights from 1 u32 (16 weights per u32, 2 bits per weight)
                let mut weights = [0i8; 16];
                let packed = packed_weight_row[k / 16];
                for j in 0..16 {
                    let bits = ((packed >> (j * 2)) & 0b11) as u8;
                    weights[j] = match bits {
                        1 => 1,
                        2 => -1,
                        _ => 0,
                    };
                }
                // Load activations and weights into int8x16_t
                let act_ptr = activation_row[k..k+16].as_ptr();
                let wgt_ptr = weights.as_ptr();
                let act_vec = vld1q_s8(act_ptr);
                let wgt_vec = vld1q_s8(wgt_ptr);
                // Multiply and accumulate
                let prod = vmull_s8(vget_low_s8(act_vec), vget_low_s8(wgt_vec));
                let mut buf = [0i16; 8];
                vst1q_s16(buf.as_mut_ptr(), prod);
                sum += buf.iter().map(|&x| x as i32).sum::<i32>();
                let prod2 = vmull_s8(vget_high_s8(act_vec), vget_high_s8(wgt_vec));
                vst1q_s16(buf.as_mut_ptr(), prod2);
                sum += buf.iter().map(|&x| x as i32).sum::<i32>();
                k += 16;
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