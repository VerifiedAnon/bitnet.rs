//! Feed-forward network implementation for BitNet.
//!
//! This module provides the feed-forward network (FFN) used in BitNet's transformer blocks.
//! As per the "BitNet b1.58" paper, this FFN uses two linear layers and
//! a Squared ReLU activation function.

use crate::bitnet_linear::BitLinear;
use crate::wgpu_context::WgpuContext;
use bitnet_converter::packer::BitLinearRecord;

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
///     intermediate_size: 4096,  // Expansion size
/// );
/// ```
#[derive(Debug, Clone)]
pub struct FeedForwardConfig {
    /// Model dimension (input and output size)
    pub hidden_size: usize,
    /// Intermediate dimension (typically 4x hidden_size)
    pub intermediate_size: usize,
}

/// A feed-forward network layer using Squared ReLU.
///
/// This layer consists of two `BitLinear` projections with a
/// `ReLU^2` activation in between.
#[derive(Clone)]
pub struct FeedForward {
    /// First linear layer (hidden -> intermediate)
    w1: BitLinear,
    /// Second linear layer (intermediate -> hidden)
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
    /// Currently initializes with default BitLinear records. In practice, weights
    /// will be loaded from a pretrained model.
    pub fn init(&self) -> FeedForward {
        let w1_packed_len = (self.intermediate_size * self.hidden_size + 15) / 16;
        let w2_packed_len = (self.hidden_size * self.intermediate_size + 15) / 16;
        let w1 = BitLinear::from_record(BitLinearRecord {
            packed_weights: vec![0; w1_packed_len],
            weight_scales: vec![1.0; self.intermediate_size],
            in_features: self.hidden_size,
            out_features: self.intermediate_size,
        });
        let w2 = BitLinear::from_record(BitLinearRecord {
            packed_weights: vec![0; w2_packed_len],
            weight_scales: vec![1.0; self.hidden_size],
            in_features: self.intermediate_size,
            out_features: self.hidden_size,
        });
        FeedForward { w1, w2 }
    }
}

impl FeedForward {
    /// Creates a new FeedForward layer from pre-processed records.
    /// NOTE: The BitNet b1.58 paper mentions using two linear layers for the FFN,
    /// not the three-layer SwiGLU structure. The converter packs a `w13` for SwiGLU,
    /// which we will treat as just `w1` for this architecture.
    pub fn from_records(w1_record: BitLinearRecord, w2_record: BitLinearRecord) -> Self {
        Self {
            w1: BitLinear::from_record(w1_record),
            w2: BitLinear::from_record(w2_record),
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
    /// 2. Apply ReLU^2 activation
    /// 3. Project back to hidden size (w2)
    pub async fn forward(&self, context: &WgpuContext, x: &[f32], batch_size: usize) -> Vec<f32> {
        let x1 = self.w1.forward(context, x, batch_size).await;
        let x2 = relu_squared(&x1);
        self.w2.forward(context, &x2, batch_size).await
    }
}

/// Computes the Squared ReLU activation function: `max(0, x)^2`.
///
/// This activation is used in the BitNet b1.58 architecture's FFN layers.
///
/// # Arguments
/// * `x` - Input values
///
/// # Returns
/// * Activated values
fn relu_squared(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&val| {
            let relu_val = val.max(0.0);
            relu_val * relu_val
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn test_ffn_config() {
        let config = FeedForwardConfig::new(1024, 4096);
        let ffn = config.init();
        assert_eq!(ffn.w1.in_features, 1024);
        assert_eq!(ffn.w1.out_features, 4096);
        assert_eq!(ffn.w2.in_features, 4096);
        assert_eq!(ffn.w2.out_features, 1024);
    }

    #[test]
    fn test_relu_squared() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let activated = relu_squared(&input);
        // Expected: 0, 0, 0, 1*1, 2*2, 3*3
        let expected = vec![0.0, 0.0, 0.0, 1.0, 4.0, 9.0];
        assert_eq!(activated, expected);
    }
    
    #[test]
    fn test_feed_forward_pass_dimensions() {
        let context = block_on(crate::wgpu_context::WgpuContext::new()).unwrap();
        
        let hidden_size = 128;
        let intermediate_size = 256;
        let batch_size = 2;
        let seq_len = 10;
        
        let config = FeedForwardConfig::new(hidden_size, intermediate_size);
        let ffn = config.init();

        let input_len = batch_size * seq_len * hidden_size;
        let input = vec![0.1; input_len];

        // The input to the FFN is typically processed per-token, so the effective batch size
        // for the linear layers is `batch_size * seq_len`.
        let effective_batch_size = batch_size * seq_len;

        let output = block_on(ffn.forward(&context, &input, effective_batch_size));

        // The output should have the same dimensions as the input.
        assert_eq!(output.len(), input_len);
    }
}