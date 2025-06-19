//! Embedding layer for BitNet, implemented in pure Rust.

/// Configuration for the Embedding layer.
pub struct EmbeddingConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The size of the model's hidden state (embedding dimension).
    pub hidden_size: usize,
}

/// The Embedding layer: holds the embedding matrix and provides lookup functionality.
pub struct Embedding {
    /// The embedding matrix, organized as [vocab_size][hidden_size].
    /// Each row represents the embedding vector for a token in the vocabulary.
    pub embedding: Vec<Vec<f32>>,
}

impl EmbeddingConfig {
    /// Initialize a new Embedding layer with all zeros (for testing or as a placeholder).
    pub fn init(&self) -> Embedding {
        Embedding {
            embedding: vec![vec![0.0; self.hidden_size]; self.vocab_size],
        }
    }
}

impl Embedding {
    /// Forward pass for the embedding layer.
    /// Takes a slice of token IDs and returns a flat Vec<f32> of their embeddings concatenated.
    pub fn forward(&self, token_ids: &[usize]) -> Vec<f32> {
        token_ids.iter()
            .flat_map(|&id| self.embedding[id].clone())
            .collect::<Vec<_>>()
    }
}