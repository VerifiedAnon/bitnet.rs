//! Multi-head attention implementation for BitNet.

use crate::bitnet_linear::BitLinear;
use crate::rope::RotaryEmbedding;
use crate::wgpu_context::WgpuContext;
use bitnet_converter::packer::BitLinearRecord;
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use std::sync::Arc;

/// Attention struct for transformer models.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Hidden size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads.
    pub num_kv_heads: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Dropout rate.
    pub dropout: f32,
}

impl AttentionConfig {
    /// Create a new AttentionConfig.
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, max_seq_len: usize) -> Self {
        assert!(num_heads > 0 && num_kv_heads > 0 && max_seq_len > 0);
        assert!(hidden_size % num_heads == 0 && num_heads % num_kv_heads == 0);
        Self { hidden_size, num_heads, num_kv_heads, max_seq_len, dropout: 0.0 }
    }
    
    /// Initialize an Attention struct.
    pub fn init(&self) -> Attention {
        let head_dim = self.hidden_size / self.num_heads;
        let q_out_dim = self.num_heads * head_dim;
        let kv_out_dim = self.num_kv_heads * head_dim;

        let create_dummy_record = |in_features, out_features| BitLinearRecord {
            packed_weights: vec![0; (out_features * in_features + 15) / 16],
            weight_scales: vec![1.0; out_features],
            in_features,
            out_features,
        };

        Attention {
            q_proj: BitLinear::from_record(create_dummy_record(self.hidden_size, q_out_dim)),
            k_proj: BitLinear::from_record(create_dummy_record(self.hidden_size, kv_out_dim)),
            v_proj: BitLinear::from_record(create_dummy_record(self.hidden_size, kv_out_dim)),
            o_proj: BitLinear::from_record(create_dummy_record(q_out_dim, self.hidden_size)),
            rotary_emb: RotaryEmbedding::new(head_dim, self.max_seq_len),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
            kv_cache: KVCache::default(),
        }
    }
}

/// Key/Value cache for attention.
#[derive(Debug, Clone, Default)]
pub struct KVCache {
    /// Key cache.
    pub key: Vec<f32>,
    /// Value cache.
    pub value: Vec<f32>,
}

/// Attention layer for transformer.
#[derive(Clone)]
pub struct Attention {
    /// Query projection.
    pub q_proj: BitLinear,
    /// Key projection.
    pub k_proj: BitLinear,
    /// Value projection.
    pub v_proj: BitLinear,
    /// Output projection.
    pub o_proj: BitLinear,
    /// Rotary embedding.
    pub rotary_emb: RotaryEmbedding,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Key/Value cache.
    pub kv_cache: KVCache,
}

impl Attention {
    /// Construct Attention from records.
    pub fn from_records(
        q_proj: BitLinearRecord,
        k_proj: BitLinearRecord,
        v_proj: BitLinearRecord,
        o_proj: BitLinearRecord,
        config: &AttentionConfig,
    ) -> Self {
        let head_dim = config.hidden_size / config.num_heads;
        Self {
            q_proj: BitLinear::from_record(q_proj),
            k_proj: BitLinear::from_record(k_proj),
            v_proj: BitLinear::from_record(v_proj),
            o_proj: BitLinear::from_record(o_proj),
            rotary_emb: RotaryEmbedding::new(head_dim, config.max_seq_len),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            kv_cache: KVCache::default(),
        }
    }
    
    /// The new, corrected, and performance-optimized CPU forward pass.
    pub fn cpu_forward(&mut self, x: &[f32], pos_offset: usize) -> Vec<f32> {
        let seq_len = x.len() / self.q_proj.in_features;

        // 1. Projections
        let mut q = self.q_proj.forward_cpu(x, seq_len);
        let mut k = self.k_proj.forward_cpu(x, seq_len);
        let v = self.v_proj.forward_cpu(x, seq_len);

        // 2. Apply RoPE to the new Q and K vectors
        self.rotary_emb.forward(&mut q, self.num_heads, seq_len, pos_offset);
        self.rotary_emb.forward(&mut k, self.num_kv_heads, seq_len, pos_offset);

        // 3. Update KV Cache
        self.kv_cache.key.extend_from_slice(&k);
        self.kv_cache.value.extend_from_slice(&v);
        let k_cache = &self.kv_cache.key;
        let v_cache = &self.kv_cache.value;
        let cache_len = k_cache.len() / (self.num_kv_heads * self.head_dim);

        let n_rep = self.num_heads / self.num_kv_heads;
        let mut attn_output = vec![0.0; seq_len * self.num_heads * self.head_dim];

        // 4. Parallelize computation over heads
        attn_output
            .par_chunks_mut(seq_len * self.head_dim)
            .enumerate()
            .for_each(|(h, out_head_slice)| {
                let kv_head_idx = h / n_rep;

                // For each token in the current sequence (batch)
            for s in 0..seq_len {
                    let q_token_start = s * (self.num_heads * self.head_dim) + h * self.head_dim;
                    let q_vec = &q[q_token_start..q_token_start + self.head_dim];

                    // --- Attention Scores (q * k^T) ---
                    let mut scores = vec![0.0; cache_len];
                    for k_pos in 0..cache_len {
                        let k_token_start = k_pos * (self.num_kv_heads * self.head_dim) + (kv_head_idx * self.head_dim);
                        let k_vec = &k_cache[k_token_start..k_token_start + self.head_dim];
                        
                        scores[k_pos] = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                    }

                    // --- Scale and Mask ---
                    let scale = 1.0 / (self.head_dim as f32).sqrt();
                    let current_pos = pos_offset + s;
                    for k_pos in 0..cache_len {
                        scores[k_pos] *= scale;
                        if k_pos > current_pos {
                            scores[k_pos] = f32::NEG_INFINITY;
                    }
                }

                    // --- Softmax ---
                    softmax_inplace(&mut scores);

                    // --- Weighted Value Sum (scores * V) ---
                    let out_token_start = s * self.head_dim;
                    for k_pos in 0..cache_len {
                        let v_token_start = k_pos * (self.num_kv_heads * self.head_dim) + (kv_head_idx * self.head_dim);
                        let v_vec = &v_cache[v_token_start..v_token_start + self.head_dim];
                        let weight = scores[k_pos];

                        for d in 0..self.head_dim {
                            out_head_slice[out_token_start + d] += weight * v_vec[d];
                }
            }
        }
            });
        
        // 5. Reshape and final projection
        let reshaped_output = reshape_heads_to_hidden(&attn_output, seq_len, self.num_heads, self.head_dim);
        self.o_proj.forward_cpu(&reshaped_output, seq_len)
    }

    /// Async GPU forward pass (now uses WGSL attention kernel for softmax/weighted sum)
    pub async fn gpu_forward(&mut self, context: &WgpuContext, x: &[f32], pos_offset: usize) -> Vec<f32> {
        let seq_len = x.len() / self.q_proj.in_features;
        let batch = 1; // Only batch size 1 supported for now (single prompt)
        // 1. Projections (GPU)
        let mut q = self.q_proj.forward(context, x, seq_len).await;
        let mut k = self.k_proj.forward(context, x, seq_len).await;
        let v = self.v_proj.forward(context, x, seq_len).await;
        // 2. Apply RoPE (CPU, fast)
        self.rotary_emb.forward(&mut q, self.num_heads, seq_len, pos_offset);
        self.rotary_emb.forward(&mut k, self.num_kv_heads, seq_len, pos_offset);
        // 3. Update KV Cache (CPU, for now)
        self.kv_cache.key.extend_from_slice(&k);
        self.kv_cache.value.extend_from_slice(&v);
        let k_cache = &self.kv_cache.key;
        let v_cache = &self.kv_cache.value;
        let cache_len = k_cache.len() / (self.num_kv_heads * self.head_dim);
        let n_rep = self.num_heads / self.num_kv_heads;
        // 4. Attention math (GPU kernel)
        let attended = launch_attention_gpu_kernel(
            context,
            &q,
            &k,
            &v,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).await;
        // 5. Reshape and final projection (GPU)
        // attended is already [batch, seq_len, num_heads, head_dim] flattened
        self.o_proj.forward(context, &attended, seq_len).await
    }

    /// Kept for API compatibility; delegates to the CPU path.
    pub async fn forward(&mut self, _context: &WgpuContext, x: &[f32], pos_offset: usize, _cache: Option<&mut KVCache>) -> Vec<f32> {
        self.cpu_forward(x, pos_offset)
    }
}

/// Launches the WGSL attention kernel on the GPU.
pub async fn launch_attention_gpu_kernel(
    context: &WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let device = &context.device;
    let queue = &context.queue;
    let total = batch * seq_len * num_heads * head_dim;
    // 1. Create buffers
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttentionMetadata {
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    }
    let metadata = AttentionMetadata {
        batch: batch as u32,
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        head_dim: head_dim as u32,
    };
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Attention Metadata Buffer"),
        contents: bytemuck::bytes_of(&metadata),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let q_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Q Buffer"),
        contents: bytemuck::cast_slice(q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let k_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K Buffer"),
        contents: bytemuck::cast_slice(k),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V Buffer"),
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Attention Output Buffer"),
        size: (total * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Attention Staging Buffer"),
        size: (total * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // 2. Load shader
    let shader_src = include_str!("kernels/bitnet_attention.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BitNet Attention Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    // 3. Create bind group layout and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Attention Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Attention Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Attention Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Attention Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
        ],
    });
    // 4. Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Attention Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Attention Compute Pass"), timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(batch as u32, seq_len as u32, num_heads as u32);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (total * std::mem::size_of::<f32>()) as u64);
    queue.submit(Some(encoder.finish()));
    // 5. Read back
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    if let Err(e) = device.poll(wgpu::MaintainBase::Wait) {
        eprintln!("[wgpu::Device::poll] error: {:?}", e);
    }
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    result
}

// --- Helper Functions ---

fn softmax_inplace(data: &mut [f32]) {
    if data.is_empty() { return; }
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;
    for val in data.iter_mut() {
        *val = (*val - max_val).exp();
        sum_exp += *val;
    }
    if sum_exp > 0.0 {
        for val in data.iter_mut() {
            *val /= sum_exp;
        }
    }
}

fn reshape_heads_to_hidden(data: &[f32], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let hidden_dim = num_heads * head_dim;
    let mut reshaped = vec![0.0; seq_len * hidden_dim];
    for s in 0..seq_len {
        for h in 0..num_heads {
            let src_start = h * (seq_len * head_dim) + s * head_dim;
            let dst_start = s * hidden_dim + h * head_dim;
            reshaped[dst_start..dst_start+head_dim].copy_from_slice(&data[src_start..src_start+head_dim]);
    }
}
    reshaped
}