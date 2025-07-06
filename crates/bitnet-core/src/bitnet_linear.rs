// --- File: crates/bitnet-core/src/bitnet_linear.rs ---
// --- FULL REPLACEMENT ---

//! Quantized linear layer implementation for BitNet.
//!
//! This module provides the core quantized linear layer used throughout BitNet,
//! implementing the 1.58-bit weight quantization scheme described in the paper.
//!
//! # Architecture
//!
//! The BitLinear layer uses several optimizations:
//! - 1.58-bit weight quantization (ternary: -1, 0, +1)
//! - Packed weight storage (16 weights per u32)
//! - Per-output-channel weight scaling
//! - Dynamic activation quantization
//!
//! # Examples
//!
//! ```rust,no_run
//! # use bitnet_core::bitnet_linear::BitLinear;
//! # use bitnet_core::wgpu_context::WgpuContext;
//! # use bitnet_converter::packer::BitLinearRecord;
//! # use futures::executor::block_on;
//! #
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // This is a conceptual example. In practice, you'd load a record.
//! let in_features = 1024;
//! let out_features = 1024;
//! let record = BitLinearRecord {
//!     packed_weights: vec![0; out_features * (in_features / 16)],
//!     weight_scales: vec![1.0; out_features],
//!     in_features,
//!     out_features,
//! };
//! let layer = BitLinear::from_record(record);
//! let context = block_on(WgpuContext::new())?;
//!
//! // Run inference
//! let input = vec![0.0; in_features]; // Batch size of 1
//! let output = block_on(layer.forward(&context, &input, 1));
//! # Ok(())
//! # }
//! ```
//!
//! # Performance
//!
//! The implementation is heavily optimized:
//! - Weights are packed 16-to-1 for memory efficiency
//! - SIMD-optimized unpacking on CPU
//! - Efficient GPU kernels via WGSL
//! - Streaming-friendly memory layout
//!
//! # Implementation Notes
//!
//! The quantization and packing process:
//! 1. Weights are quantized to {-1, 0, +1}
//! 2. 16 weights are packed into each u32
//! 3. Per-output-channel scales are computed
//! 4. At runtime, activations are dynamically quantized
//!

use crate::kernels;
use crate::wgpu_context::WgpuContext;
use bitnet_converter::packer::BitLinearRecord;
use wgpu::util::DeviceExt;
use futures_intrusive::channel::shared as channel;
use rayon::prelude::*;

/// Quantized linear layer using 1.58-bit weights.
///
/// This struct implements a memory-efficient linear layer using:
/// - Ternary weight quantization (-1, 0, +1)
/// - Packed weight storage (16 weights per u32)
/// - Per-output-channel scaling
///
/// # Fields
///
/// * `packed_weights` - Packed ternary weights
/// * `weight_scales` - Per-output-channel scaling factors
/// * `in_features` - Input dimension
/// * `out_features` - Output dimension
#[derive(Clone, Debug)]
pub struct BitLinear {
    /// Packed ternary weights, 16 weights per u32
    pub packed_weights: Vec<u32>,
    /// Per-output-channel weight scaling factors
    pub weight_scales: Vec<f32>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
}

impl BitLinear {
    /// Creates a new BitLinear layer from raw weights.
    /// This function is primarily for testing and demonstration.
    pub fn new(weights: Vec<Vec<i8>>, in_features: usize, out_features: usize) -> Self {
        // Pack weights and calculate scales
        let (packed_weights, weight_scales) = kernels::pack_ternary_weights(&weights).unwrap();
        
        Self {
            packed_weights,
            weight_scales,
            in_features,
            out_features,
        }
    }

    /// Creates a new BitLinear layer from a `BitLinearRecord` (produced by the converter).
    /// This is the standard way to initialize a layer in production.
    pub fn from_record(record: BitLinearRecord) -> Self {
        Self {
            packed_weights: record.packed_weights,
            weight_scales: record.weight_scales,
            in_features: record.in_features,
            out_features: record.out_features,
        }
    }

    /// Performs a forward pass through the layer using a WGPU compute kernel.
    /// This function now handles the activation quantization internally.
    pub async fn forward(
        &self,
        context: &WgpuContext,
        activations: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        let device = &context.device;
        let queue = &context.queue;

        // --- NEW: Dynamic Activation Quantization ---
        // We now perform the quantization inside the forward pass.
        let mut all_quantized_activations_i8: Vec<i8> = Vec::with_capacity(activations.len());
        let mut all_activation_scales: Vec<f32> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = i * self.in_features;
            let end = (i + 1) * self.in_features;
            let activation_slice = &activations[start..end];
            
            let (quantized_slice, scale) = self::quantize_activations_scalar(activation_slice);
            all_quantized_activations_i8.extend(quantized_slice);
            all_activation_scales.push(scale);
        }

        // The GPU kernel expects i32, so we must cast our i8 values.
        let all_quantized_activations_i32: Vec<i32> = all_quantized_activations_i8
            .into_iter()
            .map(|x| x as i32)
            .collect();
        // --- End of New Logic ---

        // Step 1: Create buffers
        let metadata = kernels::BitnetMetadata {
            m: batch_size as u32,
            n: self.out_features as u32,
            k: self.in_features as u32,
            k_packed: (self.in_features / 16) as u32,
        };

        let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Metadata Buffer"),
            contents: bytemuck::bytes_of(&metadata),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Use the newly quantized activations and scales for the buffers
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quantized Activation Buffer"),
            contents: bytemuck::cast_slice(&all_quantized_activations_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let activation_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation Scales Buffer"),
            contents: bytemuck::cast_slice(&all_activation_scales),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weights Buffer"),
            contents: bytemuck::cast_slice(&self.packed_weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weight Scales Buffer"),
            contents: bytemuck::cast_slice(&self.weight_scales),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_size = batch_size * self.out_features * std::mem::size_of::<f32>();
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Step 2: Create bind group layout and bind group (no changes here)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BitLinear Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BitLinear Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: activation_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: weights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scales_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: activation_scales_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
            ],
        });

        // Step 3: Create compute pipeline (no changes here)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bitnet Kernel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels/bitnet_kernel.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bitnet Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bitnet Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Step 4: Create and submit command buffer (no changes here)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size_x = 16u32;
            let workgroup_size_y = 16u32;
            let num_workgroups_x = (self.out_features as u32 + workgroup_size_x - 1) / workgroup_size_x;
            let num_workgroups_y = (batch_size as u32 + workgroup_size_y - 1) / workgroup_size_y;
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &output_buffer, 0, &staging_buffer, 0, output_size as u64,
        );

        queue.submit(Some(encoder.finish()));
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = channel::oneshot_channel();
        
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        if let Err(e) = device.poll(wgpu::MaintainBase::Wait) {
            log::error!("[wgpu::Device::poll] error: {:?}", e);
        }

        if let Some(Ok(())) = receiver.receive().await {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            // In a real app, this should return a proper error.
            panic!("Failed to map staging buffer");
        }
    }

    /// Performs a forward pass through the layer using pure Rust (CPU, multi-threaded, SIMD-friendly).
    /// This is the reference implementation, matching the GPU kernel logic.
    pub fn forward_cpu(
        &self,
        activations: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        let in_features = self.in_features;
        let out_features = self.out_features;
        let packed_per_row = (in_features + 15) / 16;
        // Defensive: check shape validity
        if self.packed_weights.len() < out_features * packed_per_row || self.weight_scales.len() < out_features {
            log::error!("[BitLinear::forward_cpu] Shape mismatch: packed_weights={}, expected={}, weight_scales={}, expected={}",
                self.packed_weights.len(), out_features * packed_per_row, self.weight_scales.len(), out_features);
            return vec![0.0; batch_size * out_features];
        }
        debug_assert_eq!(self.weight_scales.len(), out_features, "weight_scales shape mismatch");
        debug_assert!(self.packed_weights.len() >= out_features * packed_per_row, "packed_weights shape mismatch");

        // Quantize activations (batch)
        let mut all_quantized_activations_i8: Vec<i8> = Vec::with_capacity(activations.len());
        let mut all_activation_scales: Vec<f32> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * in_features;
            let end = (i + 1) * in_features;
            let activation_slice = &activations[start..end];
            let (quantized_slice, scale) = self::quantize_activations_scalar(activation_slice);
            all_quantized_activations_i8.extend(quantized_slice);
            all_activation_scales.push(scale);
        }

        // Always use the real CPU kernel dispatcher, regardless of features
        match kernels::cpu::execute(
            &all_quantized_activations_i8,
            &self.packed_weights,
            &all_activation_scales,
            &self.weight_scales,
            batch_size,
            self.in_features,
            self.out_features,
        ) {
            Ok(output) => output,
            Err(e) => {
                log::error!("[BitLinear::forward_cpu] CPU kernel error: {:?}", e);
                vec![0.0; batch_size * out_features]
            }
        }
    }
}

/// Quantizes floating-point activations to i8.
///
/// This is a scalar implementation that follows the BitNet paper's methodology:
/// 1. Find the absolute maximum value in the input tensor.
/// 2. Calculate a scaling factor: `scale = abs_max / Q_b`, where `Q_b` is 127 for 8-bit.
/// 3. Quantize each activation: `quantized = round(activation / scale)`.
/// 4. Clamp the result to the `[-127, 127]` range.
///
/// # Arguments
/// * `activations` - A slice of `f32` activations for a single item in a batch.
///
/// # Returns
/// * A tuple containing the `Vec<i8>` of quantized activations and the `f32` scaling factor.
pub fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations.iter().map(|&x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
    let scale = abs_max / 127.0 + 1e-6; // Epsilon to avoid division by zero
    (
        activations.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(),
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use crate::wgpu_context::WgpuContext;

    #[test]
    fn test_bitlinear_creation() {
        let in_features = 32;
        let out_features = 16;
        let weights = vec![vec![0i8; in_features]; out_features];
        let layer = BitLinear::new(weights, in_features, out_features);
        assert_eq!(layer.in_features, in_features);
        assert_eq!(layer.out_features, out_features);
        assert_eq!(layer.weight_scales.len(), out_features);
    }

    #[test]
    fn test_activation_quantization() {
        let activations = vec![0.5, -1.0, 2.0];
        let (quantized, scale) = quantize_activations_scalar(&activations);
        assert_eq!(quantized.len(), activations.len());
        // Check that dequantization approximately recovers original values
        for (q, a) in quantized.iter().zip(activations.iter()) {
            let dequant = (*q as f32) * scale;
            assert!((dequant - *a).abs() < scale * 1.1); // Allow some quantization error
        }
    }

    #[test]
    fn test_forward_pass_with_quantization() {
        // This test ensures the entire forward pass, including the new quantization logic, works.
        let context = block_on(WgpuContext::new()).expect("Failed to get WGPU context");

        let in_features = 16;
        let out_features = 4;
        let batch_size = 2;

        let weights = vec![vec![1i8; in_features]; out_features];
        let layer = BitLinear::new(weights, in_features, out_features);
        
        // Input activations for two items in a batch
        let activations: Vec<f32> = (0..batch_size * in_features).map(|i| (i % 5) as f32 - 2.0).collect();

        // Run the forward pass
        let result = block_on(layer.forward(&context, &activations, batch_size));

        // Basic sanity checks
        assert_eq!(result.len(), batch_size * out_features);
        assert!(result.iter().all(|&x| x.is_finite()), "Output contains NaN or Inf");

        // Optional: Compare against a manual scalar calculation for one element
        let (q_acts_0, scale_0) = quantize_activations_scalar(&activations[0..in_features]);
        let expected_sum_0: i32 = q_acts_0.iter().map(|&x| x as i32 * 1).sum(); // Weights are all 1
        let expected_output_0 = (expected_sum_0 as f32) * scale_0 * layer.weight_scales[0];
        
        // We can't know the exact value due to GPU float precision, but it should be close.
        assert!((result[0] - expected_output_0).abs() < 1e-3, "First element mismatch");
    }
}