//! GPU and CPU kernel implementations for BitNet operations.
//!
//! This module provides the core computational kernels used in BitNet,
//! including weight packing, quantization, and matrix multiplication.
//!
//! # Architecture
//!
//! The kernels are implemented in both WGSL (for GPU) and Rust (for CPU):
//! - GPU kernels use packed weights and efficient WGSL compute shaders
//! - CPU kernels use SIMD intrinsics for optimal performance
//! - Both share common data structures and memory layouts
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::kernels::{pack_ternary_weights, calculate_weight_scales};
//!
//! // Pack ternary weights into u32s
//! let weights = vec![-1i8, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0];
//! let packed = pack_ternary_weights(&weights);
//!
//! // Calculate per-channel scaling factors
//! let channels = vec![
//!     vec![-1i8, 0, 1],
//!     vec![0, 1, -1],
//! ];
//! let scales = calculate_weight_scales(&channels);
//! ```
//!
//! # Memory Layout
//!
//! The kernels use carefully designed memory layouts for efficiency:
//! - Weights are packed 16-to-1 for memory efficiency
//! - Uniform buffers match WGSL struct layouts exactly
//! - Activations use cache-friendly row-major ordering
//!
//! # Performance
//!
//! Several optimizations are employed:
//! - Efficient bit manipulation for weight packing
//! - SIMD vectorization for CPU kernels
//! - Coalesced memory access patterns
//! - Workgroup-level parallelism on GPU

use bytemuck::{Pod, Zeroable};

/// Metadata for BitNet kernel execution.
///
/// This struct is used to pass configuration data to both CPU and GPU kernels.
/// Its memory layout MUST match the WGSL shader's `BitnetMetadata` struct exactly.
///
/// # Memory Layout
///
/// The struct is marked with `repr(C)` to ensure consistent memory layout:
/// ```text
/// | Offset | Field     | Type | Description                    |
/// |--------|-----------|------|--------------------------------|
/// | 0      | m         | u32  | Batch size                     |
/// | 4      | n         | u32  | Output features                |
/// | 8      | k         | u32  | Input features                 |
/// | 12     | k_packed  | u32  | Packed input features (k/16)   |
/// ```
///
/// # Examples
///
/// ```rust
/// use bitnet_core::kernels::BitnetMetadata;
///
/// let metadata = BitnetMetadata {
///     m: 32,         // Batch size
///     n: 1024,       // Output features
///     k: 1024,       // Input features
///     k_packed: 64,  // 1024/16 packed features
/// };
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BitnetMetadata {
    /// Batch size (number of rows in the activation matrix)
    pub m: u32,
    /// Output features (number of rows in the weight matrix)
    pub n: u32,
    /// Input features (number of columns in the weight matrix)
    pub k: u32,
    /// K / 16 (since 16 2-bit weights are packed into one u32)
    pub k_packed: u32,
}

/// Packs ternary weights into a compact bit representation.
///
/// This function converts an array of ternary weights (-1, 0, +1) into
/// a packed format where each weight uses 2 bits:
/// - -1 => 00 (binary 0)
/// - 0  => 01 (binary 1)
/// - +1 => 10 (binary 2)
///
/// # Arguments
///
/// * `weights` - Array of ternary weights (-1, 0, or +1)
///
/// # Returns
///
/// * Vector of u32s, each containing 16 packed weights
///
/// # Examples
///
/// ```rust
/// use bitnet_core::kernels::pack_ternary_weights;
///
/// let weights = vec![-1i8, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0];
/// let packed = pack_ternary_weights(&weights);
/// ```
///
/// # Panics
///
/// * If the number of weights is not divisible by 16
/// * If any weight is not -1, 0, or +1
///
/// # Implementation Notes
///
/// The packing process:
/// 1. Groups weights into chunks of 16
/// 2. Encodes each weight using 2 bits
/// 3. Combines encoded weights into u32s
pub fn pack_ternary_weights(weights: &[i8]) -> Vec<u32> {
    assert_eq!(
        weights.len() % 16,
        0,
        "Weight count must be divisible by 16 for packing."
    );
    let mut packed = Vec::with_capacity(weights.len() / 16);
    for chunk in weights.chunks(16) {
        let mut packed_val = 0u32;
        for (i, &weight) in chunk.iter().enumerate() {
            let encoded = match weight {
                -1 => 0b00, // Binary 0
                0 => 0b01,  // Binary 1
                1 => 0b10,  // Binary 2
                _ => panic!("Invalid ternary weight provided: {}", weight),
            };
            // Shift the 2-bit encoded value to its position in the u32
            packed_val |= encoded << (i * 2);
        }
        packed.push(packed_val);
    }
    packed
}

/// Calculates per-channel weight scaling factors.
///
/// This function computes the β scaling factor from the BitNet paper
/// for each output channel. The scale is the average magnitude of
/// non-zero weights in the channel.
///
/// # Arguments
///
/// * `weights` - 2D array of weights [out_channels][in_features]
///
/// # Returns
///
/// * Vector of scaling factors, one per output channel
///
/// # Examples
///
/// ```rust
/// use bitnet_core::kernels::calculate_weight_scales;
///
/// let channels = vec![
///     vec![-1i8, 0, 1],  // Channel 1: scale = 1.0
///     vec![0, 1, -1],    // Channel 2: scale = 1.0
/// ];
/// let scales = calculate_weight_scales(&channels);
/// ```
///
/// # Implementation Notes
///
/// For each channel:
/// 1. Sum the absolute values of all weights
/// 2. Count the number of non-zero weights
/// 3. Compute scale as sum/count (or 1.0 if all zeros)
pub fn calculate_weight_scales(weights: &[Vec<i8>]) -> Vec<f32> {
    weights
        .iter()
        .map(|channel| {
            let sum_abs: f32 = channel.iter().map(|&w| w.abs() as f32).sum();
            let non_zero_count = channel.iter().filter(|&&w| w != 0).count() as f32;
            if non_zero_count > 0.0 {
                sum_abs / non_zero_count
            } else {
                // If a channel is all zeros, its scale is 1.0 to avoid division by zero.
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_packing() {
        let weights = vec![-1i8, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0];
        let packed = pack_ternary_weights(&weights);
        assert_eq!(packed.len(), 1); // 16 weights -> 1 u32

        // Verify each 2-bit segment
        let expected = 0b10_00_10_01_10_00_01_01_10_00_10_01_10_01_00_01u32;
        assert_eq!(packed[0], expected);
    }

    #[test]
    fn test_weight_scales() {
        let channels = vec![
            vec![-1i8, 0, 1],  // Average magnitude = 1.0
            vec![0, 0, 0],     // All zeros -> scale = 1.0
            vec![1, 1, -1],    // Average magnitude = 1.0
        ];
        let scales = calculate_weight_scales(&channels);
        assert_eq!(scales.len(), 3);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 1.0).abs() < 1e-6);
        assert!((scales[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Weight count must be divisible by 16")]
    fn test_invalid_weight_count() {
        let weights = vec![-1i8, 0, 1]; // Not divisible by 16
        pack_ternary_weights(&weights);
    }

    #[test]
    #[should_panic(expected = "Invalid ternary weight")]
    fn test_invalid_weight_value() {
        let weights = vec![2i8; 16]; // Invalid weight value
        pack_ternary_weights(&weights);
    }
}