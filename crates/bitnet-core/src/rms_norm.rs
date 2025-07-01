// --- File: crates/bitnet-core/src/rms_norm.rs ---
// --- FULL REPLACEMENT ---

//! Root Mean Square (RMS) normalization for BitNet.
//!
//! This implementation is adapted from the `burn` framework's `RmsNorm` layer
//! to correctly handle the learnable weight parameter (`gamma`) for scaling.
//! The formula is: Y = (X / sqrt(mean(X^2) + eps)) * gamma

use bitnet_converter::packer::RmsNormRecord;
use serde::{Deserialize, Serialize};

/// RMSNorm layer implementation with a learnable weight.
///
/// This struct implements the Root Mean Square normalization layer,
/// which normalizes inputs and then applies a learned scaling factor (gamma).
///
/// # Fields
/// * `weight` - Learnable scaling parameter (gamma), one per feature.
/// * `epsilon` - Small constant for numerical stability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BitnetRmsNorm {
    /// Learnable scaling parameter (gamma), one per feature.
    pub weight: Vec<f32>,
    /// Small constant for numerical stability.
    pub epsilon: f32,
}

impl BitnetRmsNorm {
    /// Creates a new RMSNorm layer with a given weight vector.
    pub fn new(weight: Vec<f32>, epsilon: f32) -> Self {
        Self { weight, epsilon }
    }

    /// Creates a new RMSNorm layer from a loaded record.
    pub fn from_record(record: RmsNormRecord) -> Self {
        // The record from the converter contains the weights (gamma).
        // The epsilon is a standard, small value not included in the record.
        Self::new(record.weight, 1e-6)
    }

    /// Performs RMS normalization on the input tensor.
    /// The input `x` is expected to be a flattened tensor of shape `[batch_size, n_features]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n_features = self.weight.len();
        if n_features == 0 || x.is_empty() {
            return x.to_vec();
        }

        let mut result = vec![0.0; x.len()];

        // Process each vector in the batch
        for (i, chunk) in x.chunks_exact(n_features).enumerate() {
            // 1. Calculate the mean of the squares
            let sum_sq: f32 = chunk.iter().map(|&v| v * v).sum();
            let mean_sq = sum_sq / (n_features as f32);
            
            // 2. Calculate the reciprocal square root (rrms)
            let rrms = 1.0 / (mean_sq + self.epsilon).sqrt();

            let output_chunk = &mut result[i * n_features..(i + 1) * n_features];

            // 3. Normalize and scale by the learned weight (gamma)
            for j in 0..n_features {
                output_chunk[j] = self.weight[j] * (chunk[j] * rrms);
            }
        }
        result
    }
}

// Default impl for testing
impl Default for BitnetRmsNorm {
    fn default() -> Self {
        Self {
            weight: vec![1.0; 128], // Dummy size
            epsilon: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_from_record() {
        let record = RmsNormRecord {
            weight: vec![0.1, 0.2, 0.3],
            shape: vec![3],
        };
        let norm = BitnetRmsNorm::from_record(record);
        assert_eq!(norm.weight, vec![0.1, 0.2, 0.3]);
        assert_eq!(norm.epsilon, 1e-6);
    }

    #[test]
    fn test_rms_norm_forward_with_weights() {
        let weights = vec![0.5, 1.0, 1.5, 2.0];
        let norm = BitnetRmsNorm::new(weights, 1e-6);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = norm.forward(&input);

        // Expected calculation
        let mean_sq = (1.0f32.powi(2) + 2.0f32.powi(2) + 3.0f32.powi(2) + 4.0f32.powi(2)) / 4.0; // (1+4+9+16)/4 = 7.5
        let rrms = 1.0 / (mean_sq + 1e-6).sqrt(); // approx 1 / 2.7386

        // Check each element
        assert!((normalized[0] - (0.5 * 1.0 * rrms)).abs() < 1e-5);
        assert!((normalized[1] - (1.0 * 2.0 * rrms)).abs() < 1e-5);
        assert!((normalized[2] - (1.5 * 3.0 * rrms)).abs() < 1e-5);
        assert!((normalized[3] - (2.0 * 4.0 * rrms)).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_batched_input() {
        let weights = vec![0.5, 2.0];
        let norm = BitnetRmsNorm::new(weights, 1e-6);
        // Batch of two items: [1.0, 3.0] and [4.0, 2.0]
        let input = vec![1.0, 3.0, 4.0, 2.0];
        let normalized = norm.forward(&input);

        assert_eq!(normalized.len(), 4);

        // --- First item in batch ---
        let mean_sq1 = (1.0f32 * 1.0f32 + 3.0f32 * 3.0f32) / 2.0f32; // (1+9)/2 = 5.0
        let rrms1 = 1.0f32 / (mean_sq1 + 1e-6f32).sqrt();
        assert!((normalized[0] - (0.5 * 1.0 * rrms1)).abs() < 1e-5);
        assert!((normalized[1] - (2.0 * 3.0 * rrms1)).abs() < 1e-5);
        
        // --- Second item in batch ---
        let mean_sq2 = (4.0f32 * 4.0f32 + 2.0f32 * 2.0f32) / 2.0f32; // (16+4)/2 = 10.0
        let rrms2 = 1.0f32 / (mean_sq2 + 1e-6f32).sqrt();
        assert!((normalized[2] - (0.5 * 4.0 * rrms2)).abs() < 1e-5);
        assert!((normalized[3] - (2.0 * 2.0 * rrms2)).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_zero_input() {
        let norm = BitnetRmsNorm::new(vec![1.0; 4], 1e-6);
        let input = vec![0.0; 4];
        let normalized = norm.forward(&input);

        // Zero input should remain zero, even with weights
        for &x in &normalized {
            assert!((x - 0.0).abs() < 1e-6);
        }
    }
}