//! Inference and generation settings for BitNet models.
//!
//! This module provides configuration options for controlling model inference
//! and text generation. It includes settings for:
//! - Sampling strategies (temperature, top-p, top-k)
//! - Generation constraints (length, time, tokens)
//! - Beam search parameters
//! - Special token handling
//! - Performance options
//!
//! # Examples
//!
//! ```rust
//! use bitnet_core::settings::InferenceSettings;
//!
//! let settings = InferenceSettings::default()
//!     .with_temperature(0.7)
//!     .with_top_p(0.9)
//!     .with_max_new_tokens(512);
//! ```
//!
//! # Common Settings
//!
//! The most commonly adjusted settings are:
//!
//! - `temperature` - Controls randomness (0.0 = deterministic, 1.0 = creative)
//! - `top_p` - Nucleus sampling threshold (0.9 = use top 90% of probability mass)
//! - `top_k` - Limits vocabulary to top K tokens
//! - `max_new_tokens` - Maximum number of tokens to generate
//! - `repetition_penalty` - Penalizes repeated tokens
//!
//! # Advanced Features
//!
//! For more control, you can configure:
//!
//! - Beam search parameters
//! - Token constraints (bad words, forced tokens)
//! - Generation timeouts
//! - System prompts
//! - Special token handling

/// Settings for controlling model inference and text generation.
///
/// This struct provides comprehensive configuration options for:
/// - Sampling strategies
/// - Generation constraints
/// - Beam search
/// - Special token handling
/// - Performance options
///
/// # Examples
///
/// ```rust
/// use bitnet_core::settings::InferenceSettings;
///
/// let settings = InferenceSettings::default()
///     .with_temperature(0.7)
///     .with_top_p(0.9)
///     .with_max_new_tokens(512);
/// ```
///
/// # Implementation Notes
///
/// The settings are organized into categories:
/// - Core sampling parameters
/// - Generation constraints
/// - Token handling
/// - Performance options
/// - Output control
#[derive(Debug, Clone)]
pub struct InferenceSettings {
    // Core Sampling
    /// Temperature for logit sampling (0.0 = greedy, 1.0 = more random)
    pub temperature: f64,
    /// Nucleus sampling threshold (0.0 to 1.0)
    pub top_p: f64,
    /// Limit vocabulary to top K tokens
    pub top_k: usize,
    /// Whether to use sampling (false = greedy)
    pub do_sample: bool,

    // Generation Constraints
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Global maximum sequence length (including input)
    pub max_length: usize,
    /// Minimum sequence length required
    pub min_length: usize,
    /// Minimum number of new tokens required
    pub min_new_tokens: usize,
    /// Batch size for parallel generation
    pub batch_size: usize,
    /// Number of sequences to return
    pub num_return_sequences: usize,
    /// Number of beams for beam search
    pub num_beams: usize,
    /// Number of beam groups for diverse beam search
    pub num_beam_groups: usize,
    /// Whether to stop early in beam search
    pub early_stopping: bool,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Diversity penalty for beam groups
    pub diversity_penalty: f32,
    /// Size of n-grams to prevent repetition
    pub no_repeat_ngram_size: usize,
    /// Penalty factor for repeated tokens
    pub repetition_penalty: f32,
    /// Alpha parameter for contrastive search
    pub penalty_alpha: f32,
    /// Number of CPU threads to use
    pub threads: usize,
    /// Whether to use KV cache
    pub use_cache: bool,
    /// Whether to use attention masking
    pub attention_mask: bool,
    /// Whether to output attention weights
    pub output_attentions: bool,
    /// Whether to output hidden states
    pub output_hidden_states: bool,
    /// Whether to output token scores
    pub output_scores: bool,
    /// Whether to remove invalid values
    pub remove_invalid_values: bool,
    /// Whether to return dictionary in generate
    pub return_dict_in_generate: bool,
    /// Maximum generation time in seconds
    pub max_time: Option<f32>,
    /// Optional prefix to prepend
    pub prefix: Option<String>,
    /// System prompt for chat models
    pub system_prompt: String,
    /// Random seed for reproducibility
    pub seed: u64,

    // Token control
    /// End of sequence token ID
    pub eos_token_id: Option<u32>,
    /// Beginning of sequence token ID
    pub bos_token_id: Option<u32>,
    /// Padding token ID
    pub pad_token_id: Option<u32>,
    /// Decoder start token ID
    pub decoder_start_token_id: Option<u32>,
    /// Forced beginning of sequence token ID
    pub forced_bos_token_id: Option<u32>,
    /// Forced end of sequence token ID
    pub forced_eos_token_id: Option<u32>,
    /// Token sequences to prevent
    pub bad_words_ids: Option<Vec<Vec<u32>>>,
    /// Individual tokens to suppress
    pub suppress_tokens: Option<Vec<u32>>,
}

impl Default for InferenceSettings {
    /// Creates default inference settings optimized for chat models.
    ///
    /// # Default Values
    ///
    /// Core sampling:
    /// - temperature: 0.7 (balanced creativity)
    /// - top_p: 0.9 (diverse but focused)
    /// - top_k: 50 (reasonable vocabulary limit)
    /// - do_sample: true (enable sampling)
    ///
    /// Generation constraints:
    /// - max_new_tokens: 512 (reasonable length)
    /// - max_length: 4096 (model context window)
    /// - repetition_penalty: 1.1 (mild repetition control)
    ///
    /// Performance:
    /// - threads: 2 (balanced CPU usage)
    /// - use_cache: true (enable KV cache)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::default();
    /// assert_eq!(settings.temperature, 0.7);
    /// assert_eq!(settings.top_p, 0.9);
    /// assert_eq!(settings.max_new_tokens, 512);
    /// ```
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_length: 4096,
            max_new_tokens: 512,
            min_length: 0,
            min_new_tokens: 0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            attention_mask: true,
            batch_size: 1,
            do_sample: true,
            eos_token_id: None,
            num_beams: 1,
            num_return_sequences: 1,
            pad_token_id: None,
            diversity_penalty: 0.0,
            early_stopping: false,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_beam_groups: 1,
            threads: 2,
            bad_words_ids: None,
            bos_token_id: None,
            decoder_start_token_id: None,
            forced_bos_token_id: None,
            forced_eos_token_id: None,
            max_time: None,
            output_attentions: false,
            output_hidden_states: false,
            output_scores: false,
            penalty_alpha: 0.0,
            prefix: None,
            remove_invalid_values: false,
            return_dict_in_generate: false,
            suppress_tokens: None,
            use_cache: true,
            system_prompt: "You are a helpful AI assistant.\nAlways provide clear, concise, and accurate answers.\nIf you are unsure, say so honestly.\nBe friendly, professional, and supportive.\nFormat lists and steps with bullet points when helpful.\nIf the user asks for code, provide well-commented examples.\nIf the user asks for advice, consider pros and cons.\nNever include harmful, unethical, or illegal content.\nIf the user asks for a summary, keep it brief and focused.\nIf the user asks for a translation, be accurate and note the language.\nIf the user asks for a joke, keep it light and appropriate.\n".to_string(),
            seed: 42,
        }
    }
}

impl InferenceSettings {
    /// Creates a new settings instance with default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the sampling temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Value between 0.0 and infinity (typically 0.0 to 2.0)
    ///   - 0.0: Greedy sampling (always pick highest probability)
    ///   - 1.0: Standard sampling
    ///   - >1.0: More random sampling
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::default()
    ///     .with_temperature(0.8);
    /// ```
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the nucleus sampling threshold.
    ///
    /// # Arguments
    ///
    /// * `top_p` - Value between 0.0 and 1.0
    ///   - 1.0: Use full vocabulary
    ///   - 0.9: Use tokens comprising top 90% of probability mass
    ///   - 0.1: Very focused sampling
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::default()
    ///     .with_top_p(0.95);
    /// ```
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    /// Sets the maximum number of new tokens to generate.
    ///
    /// # Arguments
    ///
    /// * `max_new_tokens` - Maximum number of tokens
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::default()
    ///     .with_max_new_tokens(1024);
    /// ```
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Sets the system prompt for chat models.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - Instructions for the model's behavior
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bitnet_core::settings::InferenceSettings;
    ///
    /// let settings = InferenceSettings::default()
    ///     .with_system_prompt("You are a helpful assistant.");
    /// ```
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = InferenceSettings::default();
        assert_eq!(settings.temperature, 0.7);
        assert_eq!(settings.top_p, 0.9);
        assert_eq!(settings.max_new_tokens, 512);
        assert_eq!(settings.max_length, 4096);
        assert!(settings.do_sample);
        assert!(settings.use_cache);
    }

    #[test]
    fn test_with_temperature() {
        let settings = InferenceSettings::default()
            .with_temperature(0.8);
        assert_eq!(settings.temperature, 0.8);
    }

    #[test]
    fn test_with_top_p() {
        let settings = InferenceSettings::default()
            .with_top_p(0.95);
        assert_eq!(settings.top_p, 0.95);
    }

    #[test]
    fn test_with_max_new_tokens() {
        let settings = InferenceSettings::default()
            .with_max_new_tokens(1024);
        assert_eq!(settings.max_new_tokens, 1024);
    }

    #[test]
    fn test_with_system_prompt() {
        let prompt = "You are a helpful assistant.";
        let settings = InferenceSettings::default()
            .with_system_prompt(prompt);
        assert_eq!(settings.system_prompt, prompt);
    }
} 