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
use crate::wgpu_context::WgpuContext;

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
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    /// Model dimension (input and output size)
    pub hidden_size: usize,
    /// Intermediate dimension (typically 4x hidden_size)
    pub intermediate_size: usize,
}

/// A feed-forward network layer.
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
        let w1_weights = vec![vec![0i8; self.intermediate_size]; self.hidden_size];
        let w2_weights = vec![vec![0i8; self.hidden_size]; self.intermediate_size];

        FeedForward {
            w1: BitLinear::new(w1_weights, self.hidden_size, self.intermediate_size),
            w2: BitLinear::new(w2_weights, self.intermediate_size, self.hidden_size),
        }
    }
}

impl FeedForward {
    /// Creates a new feed-forward network from this configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Feed-forward network configuration
    ///
    /// # Returns
    ///
    /// * New feed-forward network
    pub fn new(config: FeedForwardConfig) -> Self {
        let w1_weights = vec![vec![0i8; config.intermediate_size]; config.hidden_size];
        let w2_weights = vec![vec![0i8; config.hidden_size]; config.intermediate_size];

        Self {
            w1: BitLinear::new(w1_weights, config.hidden_size, config.intermediate_size),
            w2: BitLinear::new(w2_weights, config.intermediate_size, config.hidden_size),
        }
    }

    /// Performs a forward pass through the feed-forward network.
    ///
    /// # Arguments
    ///
    /// * `context` - GPU context for matrix operations
    /// * `x` - Input tensor of shape `[batch_size * seq_len, hidden_size]`
    /// * `batch_size` - Number of sequences in the batch
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
    pub async fn forward(&self, context: &WgpuContext, x: &[f32], batch_size: usize) -> Vec<f32> {
        let x1 = self.w1.forward(context, x, batch_size).await;
        let x2 = gelu(&x1);
        self.w2.forward(context, &x2, batch_size).await
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
    x.iter()
        .map(|&x| {
            let sqrt_2_over_pi = 0.7978845608028654;
            let coef = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + f32::tanh(coef))
        })
        .collect()
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

    #[tokio::test]
    async fn test_ffn_dimensions() {
        let context = WgpuContext::new().await.unwrap();
        let config = FeedForwardConfig::new(1024, 4096);
        let ffn = config.init();
        let input = vec![0.0; 1024];
        let output = ffn.forward(&context, &input, 1).await;
        assert_eq!(output.len(), 1024);
    }

    #[test]
    fn test_gelu() {
        let input = vec![-1.0, 0.0, 1.0];
        let activated = gelu(&input);
        assert_eq!(activated.len(), 3);
        assert!(activated[0] < 0.0); // GELU(-1) < 0
        assert_eq!(activated[1], 0.0); // GELU(0) = 0
        assert!(activated[2] > 0.0); // GELU(1) > 0
    }

    #[tokio::test]
    async fn test_feed_forward() {
        let ffn = FeedForward::new(FeedForwardConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
        });

        let context = WgpuContext::new().await.unwrap();
        let input = vec![1.0; 1024];
        let output = ffn.forward(&context, &input, 1).await;
        assert_eq!(output.len(), 1024);
    }
}