//! Rotary Position Embedding (RoPE) implementation.
//!
//! This module is adapted from the `burn` crate's `RotaryEncoding` to provide
//! positional information to the attention mechanism by rotating the query and key vectors.

/// Rotary Position Embedding (RoPE) structure for applying rotary positional encodings.
#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    sin: Vec<f32>,
    cos: Vec<f32>,
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Creates a new `RotaryEmbedding` layer.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        // Pre-compute the inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (10000.0f32.powf(i as f32 / head_dim as f32)))
            .collect();

        // Pre-compute sin and cos values for all positions up to max_seq_len
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let freqs: Vec<f32> = t
            .iter()
            .flat_map(|&pos| inv_freq.iter().map(move |&freq| pos * freq))
            .collect();

        let cos: Vec<f32> = freqs.iter().map(|f| f.cos()).collect();
        let sin: Vec<f32> = freqs.iter().map(|f| f.sin()).collect();

        Self { sin, cos, head_dim }
    }

    /// Dynamically grows the sin/cos tables if needed for longer context.
    pub fn ensure_capacity(&mut self, new_max_seq_len: usize) {
        let half_head_dim = self.head_dim / 2;
        let current_seq_len = self.sin.len() / half_head_dim;
        if new_max_seq_len <= current_seq_len {
            return;
        }
        let inv_freq: Vec<f32> = (0..self.head_dim)
            .step_by(2)
            .map(|i| 1.0 / (10000.0f32.powf(i as f32 / self.head_dim as f32)))
            .collect();
        for pos in current_seq_len..new_max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                self.cos.push(angle.cos());
                self.sin.push(angle.sin());
            }
        }
    }

    /// Applies rotary embeddings to a single query or key tensor.
    /// The tensor is assumed to have shape [num_heads, seq_len, head_dim].
    pub fn forward(
        &mut self,
        x: &mut [f32],
        num_heads: usize,
        seq_len: usize,
        pos_offset: usize,
    ) {
        // Ensure tables are large enough for this context
        self.ensure_capacity(pos_offset + seq_len);
        let table_len = self.sin.len();
        let head_dim = self.head_dim;
        let half_head_dim = head_dim / 2;

        for h in 0..num_heads {
            for s in 0..seq_len {
                let pos = pos_offset + s;
                let table_offset = pos * half_head_dim;
                if table_offset + half_head_dim > table_len {
                    continue;
                }
                let x_offset = (h * seq_len + s) * head_dim;
                for i in 0..half_head_dim {
                    let idx0 = x_offset + 2 * i;
                    let idx1 = x_offset + 2 * i + 1;
                    let cos_val = self.cos[table_offset + i];
                    let sin_val = self.sin[table_offset + i];
                    let x0 = x[idx0];
                    let x1 = x[idx1];
                    x[idx0] = x0 * cos_val - x1 * sin_val;
                    x[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding_basic() {
        let head_dim = 4;
        let max_seq_len = 8;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        // Input: 1 head, 2 tokens, head_dim=4
        // Each token: [x0, x1, x2, x3] (2 pairs)
        let mut x = vec![1.0, 0.0, 0.0, 1.0,  // token 0
                         0.0, 1.0, 1.0, 0.0]; // token 1
        let orig_x = x.clone();
        rope.forward(&mut x, 1, 2, 0);
        // Should not panic and should change x
        assert_ne!(x, orig_x);
    }

    #[test]
    fn test_rotary_embedding_dynamic_expand() {
        let head_dim = 4;
        let max_seq_len = 2;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        // Use a position beyond initial max_seq_len
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        rope.forward(&mut x, 1, 1, 10); // pos_offset=10
        // Should not panic and should expand tables
        assert!(rope.sin.len() >= (10+1)*(head_dim/2));
    }

    #[test]
    fn test_rotary_embedding_rotation_correctness() {
        let head_dim = 2;
        let max_seq_len = 1;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        // For pos=0, angle=0, so cos=1, sin=0, rotation should be identity
        let mut x = vec![3.0, 4.0];
        rope.forward(&mut x, 1, 1, 0);
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 4.0).abs() < 1e-6);
    }
} 