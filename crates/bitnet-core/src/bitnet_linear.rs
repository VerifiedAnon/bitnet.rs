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
//! ```rust
//! use bitnet_core::bitnet_linear::BitLinear;
//!
//! // Create a 1024x1024 linear layer
//! let in_features = 1024;
//! let out_features = 1024;
//! let weights = vec![vec![0i8; in_features]; out_features]; // Example weights
//! let layer = BitLinear::new(weights, in_features, out_features);
//!
//! // Run inference
//! let input = vec![0.0; in_features];
//! let output = layer.forward(&input); // Note: Currently forwards to GPU/CPU kernels
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
//! See the BitNet paper for details on the quantization scheme.

use crate::kernels::{BitnetMetadata, pack_ternary_weights};
use crate::wgpu_context::WgpuContext;
use bitnet_converter::packer::BitLinearRecord;
use wgpu::util::DeviceExt;
use futures_intrusive::channel::shared as channel;

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
///
/// # Examples
///
/// ```rust
/// use bitnet_core::bitnet_linear::BitLinear;
///
/// let in_features = 1024;
/// let out_features = 1024;
/// let weights = vec![vec![0i8; in_features]; out_features];
/// let layer = BitLinear::new(weights, in_features, out_features);
/// ```
///
/// # Memory Layout
///
/// The packed weights are stored in a memory-efficient format:
/// - Each u32 stores 16 ternary weights (2 bits each)
/// - Weights are stored in row-major order
/// - Total storage: out_features * ceil(in_features/16) * 4 bytes
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
    pub fn new(weights: Vec<Vec<i8>>, in_features: usize, out_features: usize) -> Self {
        // Pack weights and calculate scales
        let (packed_weights, weight_scales) = pack_ternary_weights(&weights).unwrap();
        
        Self {
            packed_weights,
            weight_scales,
            in_features,
            out_features,
        }
    }

    /// Creates a new BitLinear layer from a `BitLinearRecord` (produced by the converter).
    pub fn from_record(record: BitLinearRecord) -> Self {
        Self {
            packed_weights: record.packed_weights,
            weight_scales: record.weight_scales,
            in_features: record.in_features,
            out_features: record.out_features,
        }
    }

    /// Performs a forward pass through the layer using a WGPU compute kernel.
    pub async fn forward(
        &self,
        context: &WgpuContext,
        activations: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        let device = &context.device;
        let queue = &context.queue;

        // Step 1: Create buffers
        let metadata = BitnetMetadata {
            m: batch_size as u32,
            n: self.out_features as u32,
            k: self.in_features as u32,
            k_packed: (self.in_features / 16) as u32,
        };

        let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&metadata),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(activations),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&self.packed_weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&self.weight_scales),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let activation_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![1.0f32; batch_size]), // TODO: Implement proper activation scaling
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_size = batch_size * self.out_features * std::mem::size_of::<f32>();
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Step 2: Create bind group layout and bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: activation_scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Step 3: Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels/bitnet_kernel.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bitnet Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Step 4: Create and submit command buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroup counts based on batch size and output features
            let workgroup_size_x = 16u32; // Must match WORKGROUP_SIZE_X in WGSL
            let workgroup_size_y = 16u32; // Must match WORKGROUP_SIZE_Y in WGSL
            
            let num_workgroups_x = (self.out_features as u32 + workgroup_size_x - 1) / workgroup_size_x;
            let num_workgroups_y = (batch_size as u32 + workgroup_size_y - 1) / workgroup_size_y;
            
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            output_size as u64,
        );

        // Submit command buffer and wait for completion
        queue.submit(Some(encoder.finish()));

        // Map staging buffer and retrieve results
        let slice = staging_buffer.slice(..);
        let (sender, receiver) = channel::oneshot_channel();
        
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::MaintainBase::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to map staging buffer");
        }
    }
}

/// Quantizes floating-point activations to i8.
/// This is a scalar implementation used here for simplicity.
/// NOTE: The WGSL kernel expects i32 activations, not i8.
/// This function should ideally produce i32s directly, or we cast.
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
}