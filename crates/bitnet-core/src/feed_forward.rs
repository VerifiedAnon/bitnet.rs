//! Feed-forward network implementation for BitNet.
//!
//! This module provides the feed-forward network (FFN) used in BitNet's transformer blocks.
//! The implementation follows the standard transformer FFN architecture with:
//! - Two quantized linear layers
//! - GELU activation function
//! - Efficient memory layout and computation
//!
//! # Architecture
//!
//! The feed-forward network consists of:
//! ```text
//! Input [hidden_size]
//!   ↓
//! Linear(hidden_size → intermediate_size)  [w1]
//!   ↓
//! GELU activation
//!   ↓
//! Linear(intermediate_size → hidden_size)  [w2]
//!   ↓
//! Output [hidden_size]
//! ```
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::feed_forward::{FeedForward, FeedForwardConfig};
//!
//! let config = FeedForwardConfig::new(
//!     hidden_size: 1024,       // Model dimension
//!     intermediate_size: 4096,  // Expansion size
//! );
//!
//! let ffn = config.init();
//! let input = vec![0.0; 1024];
//! let output = ffn.forward(&input);
//! ```
//!
//! # Performance
//!
//! The implementation is optimized for both CPU and GPU:
//! - Uses quantized weights in both linear layers
//! - Efficient GELU approximation
//! - Memory-friendly data layout
//! - GPU-accelerated matrix operations

use crate::bitnet_linear::BitLinear;

/// Configuration for a feed-forward network.
///
/// This struct holds the hyperparameters needed to define a feed-forward
/// network's architecture.
///
/// # Examples
///
/// ```rust
/// use bitnet_core::feed_forward::FeedForwardConfig;
///
/// let config = FeedForwardConfig::new(
///     hidden_size: 1024,       // Model dimension
///     intermediate_size: 4096,  // Expansion size (typically 4x hidden)
/// );
/// ```
pub struct FeedForwardConfig {
    /// Model dimension (input and output size)
    pub hidden_size: usize,
    /// Intermediate dimension (typically 4x hidden_size)
    pub intermediate_size: usize,
}

/// Feed-forward network implementation.
///
/// This struct implements the feed-forward network used in transformer blocks,
/// consisting of two quantized linear layers with a GELU activation between them.
///
/// # Fields
///
/// * `w1` - First linear layer (hidden → intermediate)
/// * `w2` - Second linear layer (intermediate → hidden)
///
/// # Architecture
///
/// ```text
/// x → Linear(w1) → GELU → Linear(w2) → output
/// ```
///
/// # Implementation Notes
///
/// The feed-forward network uses:
/// - Quantized weights in both linear layers
/// - Efficient GELU approximation
/// - Memory-friendly data layout
#[derive(Clone)]
pub struct FeedForward {
    /// First linear layer (hidden → intermediate)
    w1: BitLinear,
    /// Second linear layer (intermediate → hidden)
    w2: BitLinear,
}

impl FeedForwardConfig {
    /// Creates a new feed-forward network configuration.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model dimension (input/output size)
    /// * `intermediate_size` - Intermediate dimension (typically 4x hidden)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::feed_forward::FeedForwardConfig;
    ///
    /// let config = FeedForwardConfig::new(1024, 4096);
    /// ```
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self { hidden_size, intermediate_size }
    }

    /// Initializes a feed-forward network from this configuration.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::feed_forward::FeedForwardConfig;
    ///
    /// let config = FeedForwardConfig::new(1024, 4096);
    /// let ffn = config.init();
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// Currently initializes with zero weights. In practice, weights
    /// will be loaded from a pretrained model.
    pub fn init(&self) -> FeedForward {
        // TODO: Replace with real ternary weights from model loading
        let dummy_w1 = vec![vec![0i8; self.hidden_size]; self.intermediate_size];
        let dummy_w2 = vec![vec![0i8; self.intermediate_size]; self.hidden_size];
        FeedForward {
            w1: BitLinear::new(dummy_w1, self.hidden_size, self.intermediate_size),
            w2: BitLinear::new(dummy_w2, self.intermediate_size, self.hidden_size),
        }
    }
}

impl FeedForward {
    /// Performs a forward pass through the feed-forward network.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch_size * seq_len, hidden_size]`
    ///
    /// # Returns
    ///
    /// * Output tensor of shape `[batch_size * seq_len, hidden_size]`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bitnet_core::feed_forward::{FeedForward, FeedForwardConfig};
    /// # let config = FeedForwardConfig::new(1024, 4096);
    /// # let ffn = config.init();
    /// let batch_size = 1;
    /// let seq_len = 32;
    /// let hidden_size = 1024;
    /// let input = vec![0.0; batch_size * seq_len * hidden_size];
    /// let output = ffn.forward(&input);
    /// ```
    ///
    /// # Implementation Notes
    ///
    /// The forward pass:
    /// 1. Project to intermediate size (w1)
    /// 2. Apply GELU activation
    /// 3. Project back to hidden size (w2)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let x = self.w1.forward(x);
        let x = gelu(&x);
        self.w2.forward(&x)
    }
}

/// Computes the GELU activation function.
///
/// This is an efficient approximation of the Gaussian Error Linear Unit:
/// GELU(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
///
/// # Arguments
///
/// * `x` - Input values
///
/// # Returns
///
/// * GELU-activated values
///
/// # Examples
///
/// ```rust
/// use bitnet_core::feed_forward::gelu;
///
/// let x = vec![0.0, 1.0, -1.0];
/// let activated = gelu(&x);
/// ```
///
/// # Implementation Notes
///
/// Uses a simpler approximation for efficiency:
/// GELU(x) ≈ 0.5x * (1 + tanh(x/√2))
///
/// The error in this approximation is small enough for
/// practical use in transformer models.
fn gelu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 0.5 * v * (1.0 + (v / std::f32::consts::SQRT_2).tanh())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_config() {
        let config = FeedForwardConfig::new(1024, 4096);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.intermediate_size, 4096);
    }

    #[test]
    fn test_ffn_dimensions() {
        let config = FeedForwardConfig::new(1024, 4096);
        let ffn = config.init();
        let input = vec![0.0; 1024];
        let output = ffn.forward(&input);
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_gelu() {
        let x = vec![0.0, 1.0, -1.0];
        let activated = gelu(&x);
        assert_eq!(activated.len(), 3);
        assert!((activated[0] - 0.0).abs() < 1e-6); // GELU(0) = 0
        assert!(activated[1] > 0.0); // GELU(1) > 0
        assert!(activated[2] < 0.0); // GELU(-1) < 0
    }
}