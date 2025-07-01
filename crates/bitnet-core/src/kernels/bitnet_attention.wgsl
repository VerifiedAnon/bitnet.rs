// bitnet_attention.wgsl
// BitNet-style multi-head, batched, causal-masked attention kernel (f32, reference)
//
// --- DESIGN NOTE ---
// This kernel implements the full-precision (f32) attention math for BitNet: softmax(QK^T / sqrt(d_k))V with causal masking.
// Ternary quantization (with {-1, 0, +1} weights) is used ONLY for the core matrix multiplications (Q/K/V projections, output projection, feed-forward layers),
// as described in the BitNet paper. The attention softmax block is always computed in f32 for accuracy, as quantization here would degrade model quality.
// See README and BitNet paper for rationale.
//
// Inputs: Q, K, V (f32, already projected by quantized matmuls)
// Output: Attended values (f32)
//
// For more details, see the project README and tests/attention_gpu_test.rs
//

struct AttentionMetadata {
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> metadata: AttentionMetadata;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Helper to get index in flattened buffer
fn idx(batch: u32, pos: u32, head: u32, d: u32, seq_len: u32, num_heads: u32, head_dim: u32) -> u32 {
    // [batch, seq_len, num_heads, head_dim]
    return batch * seq_len * num_heads * head_dim + pos * num_heads * head_dim + head * head_dim + d;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    let q_pos = gid.y;
    let h = gid.z;
    let batch = metadata.batch;
    let seq_len = metadata.seq_len;
    let num_heads = metadata.num_heads;
    let head_dim = metadata.head_dim;
    if b >= batch || q_pos >= seq_len || h >= num_heads {
        return;
    }
    // 1. Load Q for this (b, q_pos, h)
    var q_vec: array<f32, 128>;
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        q_vec[d] = q[idx(b, q_pos, h, d, seq_len, num_heads, head_dim)];
    }
    // 2. Compute attention scores for all k_pos
    var scores: array<f32, 2048>;
    var max_score = -1e30;
    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q_vec[d] * k[idx(b, k_pos, h, d, seq_len, num_heads, head_dim)];
        }
        // Scale
        let scale = 1.0 / sqrt(f32(head_dim));
        var score = dot * scale;
        // Causal mask: if k_pos > q_pos, set to -inf
        if (k_pos > q_pos) {
            score = -1e30;
        }
        scores[k_pos] = score;
        if (score > max_score) {
            max_score = score;
        }
    }
    // 3. Softmax
    var sum_exp: f32 = 0.0;
    for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum_exp = sum_exp + scores[k_pos];
    }
    // 4. Compute output = sum_k softmax * V
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        var out: f32 = 0.0;
        for (var k_pos: u32 = 0u; k_pos < seq_len; k_pos = k_pos + 1u) {
            let weight = scores[k_pos] / sum_exp;
            out = out + weight * v[idx(b, k_pos, h, d, seq_len, num_heads, head_dim)];
        }
        output[idx(b, q_pos, h, d, seq_len, num_heads, head_dim)] = out;
    }
} 