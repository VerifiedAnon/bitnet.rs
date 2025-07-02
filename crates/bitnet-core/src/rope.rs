//! Rotary Position Embedding (RoPE) implementation.
//!
//! This module is adapted from the `burn` crate's `RotaryEncoding` to provide
//! positional information to the attention mechanism by rotating the query and key vectors.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

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
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        
        // Pre-compute the inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (10000.0f32.powf(i as f32 / head_dim as f32)))
            .collect();

        let half_head_dim = head_dim / 2;
        let mut sin = Vec::with_capacity(max_seq_len * half_head_dim);
        let mut cos = Vec::with_capacity(max_seq_len * half_head_dim);

        // Pre-compute sin and cos values for all positions up to max_seq_len
        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        Self { sin, cos, head_dim }
    }

    /// Dynamically grows the sin/cos tables if needed for longer context.
    pub fn ensure_capacity(&mut self, new_max_seq_len: usize) {
        let half_head_dim = self.head_dim / 2;
        let current_seq_len = self.sin.len() / half_head_dim;
        
        if new_max_seq_len <= current_seq_len {
            return;
        }
        
        println!("[BitNet][RoPE] Growing tables: current_seq_len={}, new_max_seq_len={}, half_head_dim={}", 
                 current_seq_len, new_max_seq_len, half_head_dim);
        
        // Pre-compute the inverse frequencies
        let inv_freq: Vec<f32> = (0..self.head_dim)
            .step_by(2)
            .map(|i| 1.0 / (10000.0f32.powf(i as f32 / self.head_dim as f32)))
            .collect();
        
        // Reserve space to avoid multiple reallocations
        let additional_elements = (new_max_seq_len - current_seq_len) * half_head_dim;
        self.cos.reserve(additional_elements);
        self.sin.reserve(additional_elements);
        
        // Add new entries for positions from current_seq_len to new_max_seq_len
        for pos in current_seq_len..new_max_seq_len {
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                self.cos.push(angle.cos());
                self.sin.push(angle.sin());
            }
        }
        
        println!("[BitNet][RoPE] After grow: sin.len()={}, cos.len()={}", 
                 self.sin.len(), self.cos.len());
        
        // Verify the tables are correctly sized
        debug_assert_eq!(self.sin.len(), new_max_seq_len * half_head_dim);
        debug_assert_eq!(self.cos.len(), new_max_seq_len * half_head_dim);
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
        assert_eq!(self.head_dim % 2, 0, "head_dim must be even");
        assert_eq!(x.len(), num_heads * seq_len * self.head_dim, 
                   "Input tensor size mismatch: expected {}, got {}", 
                   num_heads * seq_len * self.head_dim, x.len());
        
        let half_head_dim = self.head_dim / 2;
        let max_pos = pos_offset + seq_len;
        
        // Ensure we have enough capacity for the maximum position we'll access
        self.ensure_capacity(max_pos);
        
        let call_count = ROPE_FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
        if call_count < 2 {
            println!("[BitNet][RoPE] forward: pos_offset={}, seq_len={}, half_head_dim={}, sin.len()={}, cos.len()={}",
                pos_offset, seq_len, half_head_dim, self.sin.len(), self.cos.len());
        }
        
        let table_len = self.sin.len();
        let head_dim = self.head_dim;

        for h in 0..num_heads {
            for s in 0..seq_len {
                let pos = pos_offset + s;
                let table_offset = pos * half_head_dim;
                
                // Additional safety check - this should never trigger after ensure_capacity
                if table_offset + half_head_dim > table_len {
                    eprintln!("[BitNet][RoPE][ERROR] Insufficient table capacity: need {}, have {}", 
                             table_offset + half_head_dim, table_len);
                    continue;
                }
                
                let x_offset = (h * seq_len + s) * head_dim;
                
                // Apply rotation to each pair of dimensions
                for i in 0..half_head_dim {
                    let table_idx = table_offset + i;
                    debug_assert!(table_idx < table_len, 
                                 "Table index out of bounds: {} >= {}", table_idx, table_len);
                    
                    let idx0 = x_offset + 2 * i;
                    let idx1 = x_offset + 2 * i + 1;
                    
                    // Bounds check for input tensor
                    if idx1 >= x.len() {
                        eprintln!("[BitNet][RoPE][ERROR] Input tensor index out of bounds: {} >= {}", 
                                 idx1, x.len());
                        break;
                    }
                    
                    let cos_val = self.cos[table_idx];
                    let sin_val = self.sin[table_idx];
                    let x0 = x[idx0];
                    let x1 = x[idx1];
                    
                    // Apply 2D rotation
                    x[idx0] = x0 * cos_val - x1 * sin_val;
                    x[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
    
    /// Get the current maximum sequence length supported without reallocation
    pub fn max_seq_len(&self) -> usize {
        let half_head_dim = self.head_dim / 2;
        if half_head_dim == 0 {
            0
        } else {
            self.sin.len() / half_head_dim
        }
    }
    
    /// Pre-allocate tables for a specific maximum sequence length to avoid runtime growth
    pub fn prealloc(&mut self, max_seq_len: usize) {
        self.ensure_capacity(max_seq_len);
    }
}

static ROPE_FORWARD_CALLS: AtomicUsize = AtomicUsize::new(0);

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
        assert_eq!(rope.max_seq_len(), 11);
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

    #[test]
    fn test_rope_index_out_of_bounds_regression() {
        // Simulate the bug: small head_dim, small max_seq_len, large pos_offset
        let head_dim = 16;
        let max_seq_len = 1;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        // Create a buffer for 1 head, 1 token, head_dim=16
        let mut x = vec![0.0; head_dim];
        // Use a pos_offset that will force table growth and edge-case indexing
        let pos_offset = 40; // 40 * 8 = 320, so table will need to grow
        // Should not panic
        rope.forward(&mut x, 1, 1, pos_offset);
        // If we reach here, the bug is not present for this case
        assert_eq!(rope.max_seq_len(), 41); // Should support up to position 40
    }
    
    #[test]
    fn test_rope_large_context() {
        let head_dim = 128;
        let max_seq_len = 512;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        
        // Test with large context that requires growth
        let num_heads = 12;
        let seq_len = 1024;
        let pos_offset = 2048;
        let mut x = vec![1.0; num_heads * seq_len * head_dim];
        
        rope.forward(&mut x, num_heads, seq_len, pos_offset);
        assert_eq!(rope.max_seq_len(), pos_offset + seq_len);
    }
    
    #[test]
    fn test_rope_edge_cases() {
        let head_dim = 64;
        let max_seq_len = 1;
        let mut rope = RotaryEmbedding::new(head_dim, max_seq_len);
        
        // Test edge case: exactly at boundary
        let mut x = vec![1.0; head_dim];
        rope.forward(&mut x, 1, 1, rope.max_seq_len() - 1);
        
        // Test edge case: one past boundary (should trigger growth)
        let mut x = vec![1.0; head_dim];
        let old_max = rope.max_seq_len();
        rope.forward(&mut x, 1, 1, old_max);
        assert!(rope.max_seq_len() > old_max);
    }
    
    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn test_odd_head_dim_panics() {
        RotaryEmbedding::new(5, 10); // Should panic
    }
}