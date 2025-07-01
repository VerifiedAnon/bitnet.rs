// --- File: crates/bitnet-core/src/kernels.rs ---
// --- FULL REPLACEMENT ---

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
//! let weights = vec![vec![-1i8, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0]];
//! let (packed, _) = pack_ternary_weights(&weights).unwrap();
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
use crate::error::BitNetError;

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
    /// Batch size (M)
    pub m: u32,
    /// Output features (N)
    pub n: u32,
    /// Input features (K)
    pub k: u32,
    /// Input features / 16 (packed)
    pub k_packed: u32,
}

/// Pack ternary weights into u32 values (16 weights per u32) and calculate scales.
///
/// This function converts a 2D array of ternary weights (-1, 0, +1) into:
/// 1. A packed format where each weight uses 2 bits:
///    - +1 => 01 (binary 1)
///    - -1 => 10 (binary 2)
///    -  0 => 00 (binary 0)
/// 2. Per-output-channel scaling factors
///
/// # Packing Order (LSB-first)
///
/// The first weight goes into the lowest bits (bits 0-1), the next into bits 2-3, ..., the last (16th) into bits 30-31.
/// This matches the GPU shader and converter logic.
///
/// # Arguments
///
/// * `weights` - 2D array of ternary weights [out_features][in_features]
///
/// # Returns
///
/// * `(Vec<u32>, Vec<f32>)` - (packed_weights, weight_scales)
///
/// # Examples
///
/// ```rust
/// use bitnet_core::kernels::pack_ternary_weights;
///
/// let weights = vec![
///     vec![-1i8, 0, 1],  // First output channel
///     vec![0, 1, -1],     // Second output channel
/// ];
/// let (packed, scales) = pack_ternary_weights(&weights).unwrap();
/// ```
pub fn pack_ternary_weights(weights: &[Vec<i8>]) -> Result<(Vec<u32>, Vec<f32>), BitNetError> {
    let out_features = weights.len();
    if out_features == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let in_features = weights[0].len();
    let packed_size = (in_features + 15) / 16;
    
    let mut packed_weights = vec![0u32; out_features * packed_size];
    let mut weight_scales = vec![0.0f32; out_features];
    
    for (out_idx, row) in weights.iter().enumerate() {
        // Calculate scale for this output channel
        let mut sum_abs = 0.0f32;
        let mut count = 0;
        for &w in row.iter() {
            if w != 0 {
                sum_abs += w.abs() as f32;
                count += 1;
            }
        }
        weight_scales[out_idx] = if count > 0 { sum_abs / count as f32 } else { 1.0 };
        
        // Pack weights
        for (in_idx, &w) in row.iter().enumerate() {
            let pack_idx = in_idx / 16;
            let bit_idx = (in_idx % 16) * 2; // LSB-first: first weight in bits 0-1
            
            // CORRECTED: This mapping now matches the WGSL kernel's decoding logic.
            // WGSL decode: 1 -> +1, 2 -> -1, 0/3 -> 0
            let bits = match w {
                 1 => 1u32, // 01
                -1 => 2u32, // 10
                 0 => 0u32, // 00 (or 3, but 0 is simpler and matches the kernel's default case)
                _ => return Err(BitNetError::InvalidWeightValue(w)),
            };
            
            packed_weights[out_idx * packed_size + pack_idx] |= bits << bit_idx;
        }
    }
    
    Ok((packed_weights, weight_scales))
}


/// Calculate per-output-channel weight scaling factors.
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
pub fn calculate_weight_scales(weights: &[Vec<i8>]) -> Vec<f32> {
    weights.iter().map(|row| {
        let mut sum_abs = 0.0f32;
        let mut count = 0;
        for &w in row {
            if w != 0 {
                sum_abs += w.abs() as f32;
                count += 1;
            }
        }
        if count > 0 { sum_abs / count as f32 } else { 1.0 }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_weight_packing() {
        let t0 = Instant::now();
        let weights = vec![
            vec![-1i8, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0], // Row 0
            vec![1i8, -1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1], // Row 1
        ];
        let (packed, scales) = pack_ternary_weights(&weights).unwrap();
        
        assert_eq!(packed.len(), 2);
        assert_eq!(scales.len(), 2);
        assert!(scales.iter().all(|&s| s > 0.0));
        
        // --- Verify first row with CORRECTED packing logic ---
        // Input:  -1,  0,  1,  0, -1,  1,  0,  0,  1, -1,  0,  1,  0, -1,  1,  0
        // Bits:   10, 00, 01, 00, 10, 01, 00, 00, 01, 10, 00, 01, 00, 10, 01, 00
        // LSB-first: first weight in bits 0-1, last in 30-31
        let expected_row0 = 0b00_01_10_00_01_00_10_01_00_00_01_10_00_01_00_10u32;
        assert_eq!(packed[0], expected_row0, "Packed weights for row 0 don't match corrected pattern.\nExpected: {:032b}\nGot:      {:032b}", expected_row0, packed[0]);
        
        // --- Verify second row with CORRECTED packing logic ---
        // Input:   1, -1,  0,  1,  0, -1,  1,  0,  0,  1, -1,  0,  1,  0, -1,  1
        // Bits:   01, 10, 00, 01, 00, 10, 01, 00, 00, 01, 10, 00, 01, 00, 10, 01
        let expected_row1 = 0b01100001001001000001100001001001u32;
        assert_eq!(packed[1], expected_row1, "Packed weights for row 1 don't match corrected pattern.\nExpected: {:032b}\nGot:      {:032b}", expected_row1, packed[1]);
        
        println!("[TEST] test_weight_packing (took {:.2?})", t0.elapsed());
    }

    #[test]
    fn test_weight_scales() {
        let t0 = Instant::now();
        let weights = vec![
            vec![-1i8, 0, 1],
            vec![0, 0, 0],
            vec![1, 1, -1],
            vec![-1, -1, 0],
        ];
        let scales = calculate_weight_scales(&weights);
        
        assert_eq!(scales.len(), 4);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 1.0).abs() < 1e-6);
        assert!((scales[2] - 1.0).abs() < 1e-6);
        assert!((scales[3] - 1.0).abs() < 1e-6);
        
        println!("[TEST] test_weight_scales (took {:.2?})", t0.elapsed());
    }

    #[test]
    fn test_invalid_weight_value() {
        let t0 = Instant::now();
        let weights = vec![vec![2i8; 16]]; // Invalid weight value
        let result = pack_ternary_weights(&weights);
        assert!(matches!(result, Err(BitNetError::InvalidWeightValue(2))), "Expected InvalidWeightValue error, but got {:?}", result);
        println!("[TEST] test_invalid_weight_value (took {:.2?})", t0.elapsed());
    }

    #[test]
    fn test_packing_unpacking_symmetry() {
        // This test ensures that packing and then unpacking using the shader/scalar logic is symmetric.
        let original: Vec<i8> = vec![-1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0];
        let weights = vec![original.clone()];
        let (packed, _) = pack_ternary_weights(&weights).unwrap();
        let packed_val = packed[0];

        // Unpack using the CORRECTED logic (LSB-first) that matches the GPU kernel.
        let mut unpacked = Vec::with_capacity(16);
        for i in 0..16 {
            let bits = (packed_val >> (i * 2)) & 0b11;
            let w = match bits {
                1 => 1i8,
                2 => -1i8,
                _ => 0i8, // Handles 0b00 and 0b11
            };
            unpacked.push(w);
        }
        assert_eq!(original, unpacked, "Packing and unpacking are not symmetric!\nOriginal: {:?}\nUnpacked: {:?}", original, unpacked);
    }
}