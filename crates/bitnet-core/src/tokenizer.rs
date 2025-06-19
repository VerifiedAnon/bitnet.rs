//! Tokenizer wrapper and chat format logic for BitNet.

use crate::error::BitNetError;
use tokenizers::Tokenizer as HfTokenizer;

/// A message in a conversation, containing a role and content.
pub struct ChatMessage {
    /// The role of the message sender (System, User, or Assistant).
    pub role: Role,
    /// The actual content/text of the message.
    pub content: String,
}

/// Defines the possible roles in a conversation.
pub enum Role {
    /// System messages provide context or instructions to the model.
    System,
    /// User messages contain the input/query from the user.
    User,
    /// Assistant messages contain the model's responses.
    Assistant,
}

/// A wrapper around the Hugging Face tokenizer providing BitNet-specific functionality.
pub struct Tokenizer {
    /// The underlying Hugging Face tokenizer instance.
    pub inner: HfTokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a file path (e.g., ".../tokenizer.json").
    pub fn from_file(path: &str) -> Result<Self, BitNetError> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| BitNetError::Config(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { inner })
    }

    /// Decode a slice of token IDs back into a string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, BitNetError> {
        self.inner.decode(ids, true)
            .map_err(|e| BitNetError::Config(format!("Failed to decode tokens: {}", e)))
    }

    /// Encode a single string of text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, BitNetError> {
        let encoding = self.inner.encode(text, true)
            .map_err(|e| BitNetError::Config(format!("Failed to encode text: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// A simple chat formatter.
    /// In a real application, this would use a more complex chat template.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>, BitNetError> {
        let mut prompt = String::new();
        for message in messages {
            let role_str = match message.role {
                Role::System => "[SYSTEM]",
                Role::User => "[USER]",
                Role::Assistant => "[ASSISTANT]",
            };
            prompt.push_str(&format!("{} {}\n", role_str, message.content));
        }
        self.encode(&prompt)
    }
}