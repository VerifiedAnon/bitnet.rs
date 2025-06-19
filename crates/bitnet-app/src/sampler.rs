//! Logits processing for sampling (temperature, top_p, penalties, etc).

use rand::prelude::*;
use rand::distr::Uniform;

/// Settings for logits sampling.
#[derive(Debug, Clone)]
pub struct InferenceSettings {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub do_sample: bool,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            do_sample: true,
        }
    }
}

/// Stateless logits processor for sampling next token.
pub struct LogitsProcessor;

impl LogitsProcessor {
    /// Sample a token index from logits, given the generated token history and settings.
    pub fn sample(
        logits: &[f32],
        generated_ids: &[usize],
        settings: &InferenceSettings,
    ) -> usize {
        let mut processed_logits = logits.to_vec();

        // 1. Repetition penalty
        if (settings.repetition_penalty - 1.0).abs() > f32::EPSILON {
            let penalized: std::collections::HashSet<_> = generated_ids.iter().copied().collect();
            for &token_id in &penalized {
                if token_id < processed_logits.len() {
                    if processed_logits[token_id] > 0.0 {
                        processed_logits[token_id] /= settings.repetition_penalty;
                    } else {
                        processed_logits[token_id] *= settings.repetition_penalty;
                    }
                }
            }
        }

        // 2. No-repeat n-gram
        if settings.no_repeat_ngram_size > 0 && generated_ids.len() >= settings.no_repeat_ngram_size - 1 {
            let n = settings.no_repeat_ngram_size;
            let last_tokens = &generated_ids[generated_ids.len() + 1 - n..];
            let mut banned = std::collections::HashSet::new();
            for i in 0..=generated_ids.len().saturating_sub(n) {
                if &generated_ids[i..i + n - 1] == &last_tokens[..n - 1] {
                    banned.insert(generated_ids[i + n - 1]);
                }
            }
            for &token in &banned {
                if token < processed_logits.len() {
                    processed_logits[token] = f32::NEG_INFINITY;
                }
            }
        }

        // 3. Temperature
        if (settings.temperature - 1.0).abs() > f32::EPSILON && settings.temperature > 0.0 {
            for l in &mut processed_logits {
                *l /= settings.temperature;
            }
        }

        // 4. Top-k and top-p filtering
        // Build a vector of (logit, index) and filter
        let mut logits_with_indices: Vec<(f32, usize)> = processed_logits
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_i, v)| v != f32::NEG_INFINITY)
            .map(|(i, v)| (v, i))
            .collect();
        logits_with_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Top-k
        if settings.top_k > 0 && settings.top_k < logits_with_indices.len() {
            logits_with_indices.truncate(settings.top_k);
        }

        // Top-p (nucleus)
        if settings.top_p > 0.0 && settings.top_p < 1.0 {
            let sorted_logits: Vec<f32> = logits_with_indices.iter().map(|x| x.0).collect();
            let probabilities = softmax(&sorted_logits);
            let mut cumulative = 0.0;
            let mut nucleus = logits_with_indices.len();
            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if cumulative > settings.top_p {
                    nucleus = i + 1;
                    break;
                }
            }
            logits_with_indices.truncate(nucleus);
        }

        // 5. Sampling or argmax
        if settings.do_sample && logits_with_indices.len() > 1 {
            let logits_vec: Vec<f32> = logits_with_indices.iter().map(|x| x.0).collect();
            let probabilities = softmax(&logits_vec);
            let mut rng = rand::rng();
            let between = Uniform::new(0.0, 1.0).unwrap();
            let rand_val: f32 = between.sample(&mut rng);
            let mut cumulative = 0.0;
            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if rand_val <= cumulative {
                    return logits_with_indices[i].1;
                }
            }
            // Fallback: return last
            logits_with_indices.last().unwrap().1
        } else {
            // Argmax
            logits_with_indices.first().map(|x| x.1).unwrap_or(0)
        }
    }
}

/// Compute softmax over a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
} 