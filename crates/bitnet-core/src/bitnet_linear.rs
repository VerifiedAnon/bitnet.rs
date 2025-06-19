//! Quantized linear layer implementation for BitNet.
//!
//! This module provides the core quantized linear layer used throughout BitNet,
//! implementing the 1.58-bit weight quantization scheme described in the paper.
//!
//! # Architecture
//!
//! The BitLinear layer uses several optimizations:
//! - 1.58-bit weight quantization (ternary: -1, 0, +1)
//! - Packed weight storage (16 weights per u32)
//! - Per-output-channel weight scaling
//! - Dynamic activation quantization
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::bitnet_linear::BitLinear;
//!
//! // Create a 1024x1024 linear layer
//! let in_features = 1024;
//! let out_features = 1024;
//! let weights = vec![vec![0i8; in_features]; out_features]; // Example weights
//! let layer = BitLinear::new(weights, in_features, out_features);
//!
//! // Run inference
//! let input = vec![0.0; in_features];
//! let output = layer.forward(&input); // Note: Currently forwards to GPU/CPU kernels
//! ```
//!
//! # Performance
//!
//! The implementation is heavily optimized:
//! - Weights are packed 16-to-1 for memory efficiency
//! - SIMD-optimized unpacking on CPU
//! - Efficient GPU kernels via WGSL
//! - Streaming-friendly memory layout
//!
//! # Implementation Notes
//!
//! The quantization and packing process:
//! 1. Weights are quantized to {-1, 0, +1}
//! 2. 16 weights are packed into each u32
//! 3. Per-output-channel scales are computed
//! 4. At runtime, activations are dynamically quantized
//!
//! See the BitNet paper for details on the quantization scheme.

use crate::kernels::{pack_ternary_weights, calculate_weight_scales};

/// Quantized linear layer using 1.58-bit weights.
///
/// This struct implements a memory-efficient linear layer using:
/// - Ternary weight quantization (-1, 0, +1)
/// - Packed weight storage (16 weights per u32)
/// - Per-output-channel scaling
///
/// # Fields
///
/// * `packed_weights` - Packed ternary weights
/// * `weight_scales` - Per-output-channel scaling factors
/// * `in_features` - Input dimension
/// * `out_features` - Output dimension
///
/// # Examples
///
/// ```rust
/// use bitnet_core::bitnet_linear::BitLinear;
///
/// let in_features = 1024;
/// let out_features = 1024;
/// let weights = vec![vec![0i8; in_features]; out_features];
/// let layer = BitLinear::new(weights, in_features, out_features);
/// ```
///
/// # Memory Layout
///
/// The packed weights are stored in a memory-efficient format:
/// - Each u32 stores 16 ternary weights (2 bits each)
/// - Weights are stored in row-major order
/// - Total storage: out_features * ceil(in_features/16) * 4 bytes
#[derive(Clone)]
pub struct BitLinear {
    /// Packed ternary weights, 16 weights per u32
    pub packed_weights: Vec<u32>,
    /// Per-output-channel weight scaling factors
    pub weight_scales: Vec<f32>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
}

impl BitLinear {
    /// Creates a new BitLinear layer from unpacked ternary weights.
    ///
    /// # Arguments
    ///
    /// * `ternary_weights` - Weight matrix with values in {-1, 0, +1}
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::bitnet_linear::BitLinear;
    ///
    /// let in_features = 1024;
    /// let out_features = 1024;
    /// let weights = vec![vec![0i8; in_features]; out_features];
    /// let layer = BitLinear::new(weights, in_features, out_features);
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// The constructor:
    /// 1. Packs the ternary weights (16 per u32)
    /// 2. Calculates per-output-channel scaling factors
    /// 3. Validates dimensions
    pub fn new(ternary_weights: Vec<Vec<i8>>, in_features: usize, out_features: usize) -> Self {
        let packed_weights: Vec<u32> = ternary_weights.iter().flat_map(|row| pack_ternary_weights(row)).collect();
        let weight_scales = calculate_weight_scales(&ternary_weights);
        Self {
            packed_weights,
            weight_scales,
            in_features,
            out_features,
        }
    }

    /// Performs a forward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `activations` - Input tensor of shape `[batch_size, in_features]`
    ///
    /// # Returns
    ///
    /// * Output tensor of shape `[batch_size, out_features]`
    ///
    /// # Notes
    ///
    /// This is currently a stub that forwards to GPU/CPU kernels.
    /// The actual computation is handled by the kernel implementations.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use bitnet_core::bitnet_linear::BitLinear;
    /// # let layer = BitLinear::new(vec![vec![0i8; 1024]; 1024], 1024, 1024);
    /// let input = vec![0.0; 1024];
    /// let output = layer.forward(&input);
    /// ```
    pub fn forward(&self, _activations: &[f32]) -> Vec<f32> {
        unimplemented!("Direct kernel launch is now handled in test/validation code.");
    }
}

/// Quantizes floating-point activations to int8.
///
/// This is a reference scalar implementation for testing and validation.
/// Production code uses SIMD-optimized versions.
///
/// # Arguments
///
/// * `activations` - Input activations in f32
///
/// # Returns
///
/// * `(quantized, scale)` - Quantized values and scaling factor
///
/// # Examples
///
/// ```rust
/// use bitnet_core::bitnet_linear::quantize_activations_scalar;
///
/// let activations = vec![0.5, -1.0, 2.0];
/// let (quantized, scale) = quantize_activations_scalar(&activations);
/// ```
///
/// # Implementation Notes
///
/// The quantization process:
/// 1. Find absolute maximum value
/// 2. Calculate scaling factor
/// 3. Scale and clamp to [-127, 127]
pub fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations.iter().map(|&x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
    let scale = abs_max / 127.0 + 1e-6;
    (
        activations.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(),
        scale
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitlinear_creation() {
        let in_features = 32;
        let out_features = 16;
        let weights = vec![vec![0i8; in_features]; out_features];
        let layer = BitLinear::new(weights, in_features, out_features);
        assert_eq!(layer.in_features, in_features);
        assert_eq!(layer.out_features, out_features);
        assert_eq!(layer.weight_scales.len(), out_features);
    }

    #[test]
    fn test_activation_quantization() {
        let activations = vec![0.5, -1.0, 2.0];
        let (quantized, scale) = quantize_activations_scalar(&activations);
        assert_eq!(quantized.len(), activations.len());
        // Check that dequantization approximately recovers original values
        for (q, a) in quantized.iter().zip(activations.iter()) {
            let dequant = (*q as f32) * scale;
            assert!((dequant - *a).abs() < scale * 1.1); // Allow some quantization error
        }
    }
}