//! Root Mean Square (RMS) normalization for BitNet.
//!
//! This module provides the RMSNorm layer used in BitNet's transformer blocks.
//! RMSNorm is a simpler and more efficient alternative to LayerNorm, as described
//! in "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019).
//!
//! # Architecture
//!
//! RMSNorm normalizes inputs using only the root mean square statistic:
//! ```text
//! y = x / sqrt(mean(x^2) + ε)
//! ```
//!
//! Compared to LayerNorm, RMSNorm:
//! - Removes mean centering (μ = 0)
//! - Eliminates learned scale/bias (γ, β)
//! - Reduces computation and memory
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::rms_norm::BitnetRmsNorm;
//!
//! let norm = BitnetRmsNorm::new();
//! let input = vec![1.0, 2.0, 3.0, 4.0];
//! let normalized = norm.forward(&input);
//! ```
//!
//! # Performance
//!
//! The implementation is optimized for efficiency:
//! - Single pass for mean square calculation
//! - Vectorizable operations
//! - Minimal memory overhead
//! - No learned parameters

/// RMSNorm layer implementation.
///
/// This struct implements the Root Mean Square normalization layer,
/// which normalizes inputs using only their RMS statistic.
///
/// # Fields
///
/// * `epsilon` - Small constant for numerical stability
///
/// # Examples
///
/// ```rust
/// use bitnet_core::rms_norm::BitnetRmsNorm;
///
/// let norm = BitnetRmsNorm::new();
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let normalized = norm.forward(&input);
/// ```
///
/// # Implementation Notes
///
/// The normalization process:
/// 1. Calculate mean of squared values
/// 2. Add epsilon for stability
/// 3. Take square root for scaling
/// 4. Divide input by scale factor
#[derive(Clone)]
pub struct BitnetRmsNorm {
    /// Small constant for numerical stability
    pub epsilon: f32,
}

impl BitnetRmsNorm {
    /// Creates a new RMSNorm layer with default epsilon.
    ///
    /// # Returns
    ///
    /// * A new RMSNorm instance with ε = 1e-6
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::rms_norm::BitnetRmsNorm;
    ///
    /// let norm = BitnetRmsNorm::new();
    /// assert_eq!(norm.epsilon, 1e-6);
    /// ```
    pub fn new() -> Self {
        Self { epsilon: 1e-6 }
    }

    /// Performs RMS normalization on the input.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor to normalize
    ///
    /// # Returns
    ///
    /// * Normalized tensor of same shape as input
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::rms_norm::BitnetRmsNorm;
    ///
    /// let norm = BitnetRmsNorm::new();
    /// let input = vec![1.0, 2.0, 3.0, 4.0];
    /// let normalized = norm.forward(&input);
    ///
    /// // Verify RMS ≈ 1.0
    /// let rms: f32 = normalized.iter()
    ///     .map(|&x| x * x)
    ///     .sum::<f32>()
    ///     .sqrt() / (normalized.len() as f32).sqrt();
    /// assert!((rms - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// The forward pass:
    /// 1. Compute mean(x^2) in single pass
    /// 2. Add epsilon for stability
    /// 3. Take sqrt for scaling factor
    /// 4. Divide input by scale
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Simple RMSNorm: y = x / sqrt(mean(x^2) + eps)
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / (x.len() as f32);
        let denom = (mean_sq + self.epsilon).sqrt();
        x.iter().map(|v| v / denom).collect()
    }
}

impl Default for BitnetRmsNorm {
    /// Creates a new RMSNorm layer with default epsilon (1e-6).
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_creation() {
        let norm = BitnetRmsNorm::new();
        assert_eq!(norm.epsilon, 1e-6);
    }

    #[test]
    fn test_rms_norm_forward() {
        let norm = BitnetRmsNorm::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = norm.forward(&input);

        // Check output length
        assert_eq!(normalized.len(), input.len());

        // Verify RMS ≈ 1.0
        let rms: f32 = normalized.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt() / (normalized.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_zero_input() {
        let norm = BitnetRmsNorm::new();
        let input = vec![0.0; 4];
        let normalized = norm.forward(&input);

        // Zero input should remain zero
        for x in normalized {
            assert!((x - 0.0).abs() < 1e-6);
        }
    }
}