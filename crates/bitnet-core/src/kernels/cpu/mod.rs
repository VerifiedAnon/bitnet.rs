//! CPU kernel dispatcher for BitNet operations.
//!
//! This module contains the main entry point for CPU-based computation. It uses
//! runtime feature detection to select the most optimal SIMD implementation
//! available on the target machine (AVX2, NEON) and falls back to a scalar
//! implementation if no SIMD features are available.

// Platform-specific modules containing unsafe SIMD code.
#[cfg(target_arch = "x86_64")]
pub mod x86;
#[cfg(target_arch = "aarch64")]
pub mod arm;

// A pure Rust, safe, scalar implementation for fallback and validation.
pub mod scalar;

use super::BitNetError;

/// Dispatches the quantized matrix multiplication to the best available CPU kernel.
///
/// This function is the primary entry point for the CPU backend. It checks for CPU
/// features at runtime and calls the appropriate SIMD or scalar kernel.
///
/// # Arguments
/// * `q_activations`: A slice of `i8` quantized activations.
/// * `packed_weights`: A slice of `u32` packed ternary weights.
/// * `activation_scales`: A slice of `f32` scaling factors for activations.
/// * `weight_scales`: A slice of `f32` scaling factors for weights.
/// * `batch_size`: The number of sequences being processed.
/// * `in_features`: The input dimension of the linear layer.
/// * `out_features`: The output dimension of the linear layer.
///
/// # Returns
/// A `Vec<f32>` containing the computed output, or an error if dimensions are invalid.
pub fn execute(
    q_activations: &[i8],
    packed_weights: &[u32],
    activation_scales: &[f32],
    weight_scales: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, BitNetError> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                x86::qgemm_lut_avx2(
                    q_activations,
                    packed_weights,
                    activation_scales,
                    weight_scales,
                    batch_size,
                    in_features,
                    out_features,
                )
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe {
                arm::qgemm_lut_neon(
                    q_activations,
                    packed_weights,
                    activation_scales,
                    weight_scales,
                    batch_size,
                    in_features,
                    out_features,
                )
            };
        }
    }
    // Fallback to the scalar implementation if no SIMD features are available.
    scalar::qgemm_lut_scalar(
        q_activations,
        packed_weights,
        activation_scales,
        weight_scales,
        batch_size,
        in_features,
        out_features,
    )
} 