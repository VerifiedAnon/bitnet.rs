//! Kernel tests for BitNet: pure Rust + direct wgpu (no burn)

use std::time::Instant;
use wgpu::util::DeviceExt;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use bitnet_core::kernels::{pack_ternary_weights, calculate_weight_scales, BitnetMetadata};
use bitnet_core::bitnet_linear::BitLinear;
use bitnet_core::wgpu_context::WgpuContext;
use bitnet_converter::packer::BitLinearRecord;
use bitnet_tools::test_utils::TestReporter;
use lazy_static::lazy_static;
use serial_test::serial;

// Initialize test reporter
lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("kernel_tests")
        .expect("Failed to create test reporter");
}

// --- Scalar reference logic ---
fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations.iter().map(|&x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
    let scale = abs_max / 127.0 + 1e-6;
    (activations.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(), scale)
}

fn matmul_quantized_scalar(
    q_activations: &[i8],
    packed_weights: &[u32],
    activation_scales: &[f32],
    weight_scales: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * out_features];
    let k_packed = in_features / 16;

    for batch_idx in 0..batch_size {
        let activation_scale = activation_scales[batch_idx];
        let activations_start = batch_idx * in_features;

        for out_idx in 0..out_features {
            let mut sum = 0i32;
            let weight_scale = weight_scales[out_idx];
            let weights_start = out_idx * k_packed;

            for k_idx in 0..k_packed {
                let packed_weight = packed_weights[weights_start + k_idx];
                let act_base = activations_start + k_idx * 16;

                for bit_idx in 0..16 {
                    let weight_val = match (packed_weight >> (bit_idx * 2)) & 0b11 {
                        0b01 => 1i8,
                        0b10 => -1i8,
                        _ => 0i8,
                    };
                    sum += (q_activations[act_base + bit_idx] as i32) * (weight_val as i32);
                }
            }

            output[batch_idx * out_features + out_idx] = 
                (sum as f32) * activation_scale * weight_scale;
        }
    }
    output
}

fn assert_vec_eq(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vector lengths don't match");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tolerance,
            "Vectors differ at index {}: {} != {} (diff = {})",
            i, x, y, (x - y).abs()
        );
    }
}

// --- Direct wgpu kernel launcher ---
async fn launch_gpu_kernel(
    q_acts_vec: &[i8],
    packed_weights_u32: &[u32],
    weight_scales_f32: &[f32],
    act_scales: &[f32],
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    shader_source: &str,
    context: &WgpuContext,
) -> Vec<f32> {
    // Convert i8 activations to i32 for GPU
    let q_acts_i32: Vec<i32> = q_acts_vec.iter().map(|&x| x as i32).collect();

    // Create metadata buffer
    let metadata = BitnetMetadata {
        m: batch_size as u32,
        n: out_features as u32,
        k: in_features as u32,
        k_packed: (in_features / 16) as u32,
    };
    let metadata_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("metadata_buffer"),
        contents: bytemuck::cast_slice(&[metadata]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create input/output buffers
    let q_acts_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q_acts_buffer"),
        contents: bytemuck::cast_slice(&q_acts_i32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let packed_weights_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("packed_weights_buffer"),
        contents: bytemuck::cast_slice(packed_weights_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let weight_scales_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weight_scales_buffer"),
        contents: bytemuck::cast_slice(weight_scales_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let act_scales_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("act_scales_buffer"),
        contents: bytemuck::cast_slice(act_scales),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = batch_size * out_features;
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: (output_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create shader module
    let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bitnet Kernel"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout and pipeline
    let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bitnet Bind Group Layout"),
        entries: &[
            // Metadata (uniform)
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
            // Activations (storage, read)
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
            // Packed weights (storage, read)
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
            // Weight scales (storage, read)
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
            // Activation scales (storage, read)
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
            // Output (storage, read_write)
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

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bitnet Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bitnet Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bitnet Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: metadata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q_acts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: packed_weights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: weight_scales_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: act_scales_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // Create command encoder and dispatch compute pass
    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Bitnet Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bitnet Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        // Use workgroup size from shader
        let workgroup_size_x = 16;  // WORKGROUP_SIZE_X from shader
        let workgroup_size_y = 16;  // WORKGROUP_SIZE_Y from shader
        
        compute_pass.dispatch_workgroups(
            (batch_size as u32 + workgroup_size_x - 1) / workgroup_size_x,
            (out_features as u32 + workgroup_size_y - 1) / workgroup_size_y,
            1,
        );
    }

    // Copy output to staging buffer
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (output_size * std::mem::size_of::<f32>()) as u64,
    );

    // Submit command buffer and wait for completion
    context.queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    let _ = context.device.poll(wgpu::MaintainBase::Wait);
    rx.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

// --- TESTS ---

#[test]
fn unit_test_pack_ternary_weights() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(1, "Running unit_test_pack_ternary_weights...");
    let weights = vec![vec![-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]];
    let (packed, _scales) = pack_ternary_weights(&weights);
    assert_eq!(packed.len(), 1);
    // Encoding: -1 -> 00 (0), 0 -> 01 (1), 1 -> 10 (2)
    // Input: -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1
    // Bits:   00, 01, 10, 00, 01, 10, 00, 01, 10, 00, 01, 10, 00, 01, 10, 00
    let expected = 0b00_01_10_00_01_10_00_01_10_00_01_10_00_01_10_00u32;
    assert_eq!(packed[0], expected, "Packing logic is incorrect");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("unit_test_pack_ternary_weights", duration);
    TEST_REPORTER.log_message(1, "unit_test_pack_ternary_weights passed.");
}

#[test]
fn unit_test_calculate_weight_scales() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(2, "Running unit_test_calculate_weight_scales...");
    let weights = vec![vec![-1i8, 0, 1, 1, 0, -1, 1, 0], vec![1, 1, 1, 1], vec![0, 0]];
    let scales = calculate_weight_scales(&weights);
    assert_eq!(scales, vec![1.0, 1.0, 1.0], "Scale calculation is incorrect");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("unit_test_calculate_weight_scales", duration);
    TEST_REPORTER.log_message(2, "unit_test_calculate_weight_scales passed.");
}

#[test]
fn test_matmul_quantized_scalar() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(3, "Starting test_matmul_quantized_scalar...");
    
    // Test case with proper BitNet dimensions
    let batch_size = 1;
    let in_features = 16;
    let out_features = 2;
    
    // Simple test inputs
    let q_activations = vec![1i8, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2];
    let packed_weights = vec![
        0b01001001_01001001_01001001_01001001u32,
        0b01001010_01001010_01001010_01001010u32
    ];
    let activation_scales = vec![0.5];
    let weight_scales = vec![1.0, 1.0];
    
    let output = matmul_quantized_scalar(
        &q_activations,
        &packed_weights,
        &activation_scales,
        &weight_scales,
        batch_size,
        in_features,
        out_features
    );
    
    assert!((output[0] - 8.0).abs() < 1e-5, "First output mismatch");
    assert!((output[1] - 4.0).abs() < 1e-5, "Second output mismatch");
    
    TEST_REPORTER.log_message(3, &format!("Output values: {:?}", output));
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_matmul_quantized_scalar", duration);
}

#[tokio::test]
async fn test_basic_gpu_buffer_operations() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(4, "Testing basic GPU operations...");
    
    let context = WgpuContext::new().await.expect("Failed to create WgpuContext");
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let test_size = (test_data.len() * std::mem::size_of::<f32>()) as u64;
    
    let buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_buffer"),
        contents: bytemuck::cast_slice(&test_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: test_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, test_size);
    context.queue.submit(Some(encoder.finish()));
    
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    let _ = context.device.poll(wgpu::MaintainBase::Wait);
    rx.receive().await.unwrap().unwrap();
    
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    
    assert_eq!(test_data, result, "Buffer readback data doesn't match original");
    TEST_REPORTER.log_message(4, "Basic GPU operations test passed!");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_basic_gpu_buffer_operations", duration);
}

#[tokio::test]
async fn low_level_kernel_correctness_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(5, "Running low_level_kernel_correctness_test...");
    
    let batch_size = 1;
    let in_features = 32;
    let out_features = 16;
    
    let mut rng = StdRng::seed_from_u64(43);
    let activations_f32: Vec<f32> = (0..batch_size * in_features)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| (0..in_features)
            .map(|_| (rng.random_range(0..3) - 1) as i8)
            .collect())
        .collect();
    
    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    
    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &[act_scale],
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );
    
    let context = WgpuContext::new().await.expect("Failed to create WgpuContext");
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &[act_scale],
        batch_size,
        in_features,
        out_features,
        shader_source,
        &context,
    ).await;
    
    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(5, "low_level_kernel_correctness_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("low_level_kernel_correctness_test", duration);
}

#[tokio::test]
async fn test_gpu_kernel_dimensions() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(6, "Running test_gpu_kernel_dimensions...");
    
    let batch_size = 1;
    let in_features = 16;
    let out_features = 2;
    
    let q_acts_vec = vec![1i8, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2];
    let act_scales = vec![0.5];
    
    let packed_weights_u32 = vec![
        0b01001001_01001001_01001001_01001001u32,
        0b01001010_01001010_01001010_01001010u32
    ];
    let weight_scales_f32 = vec![1.0, 1.0];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let context = WgpuContext::new().await.expect("Failed to create WgpuContext");
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &context,
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(6, "test_gpu_kernel_dimensions passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_gpu_kernel_dimensions", duration);
}

#[tokio::test]
async fn kernel_large_batch_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(7, "Running kernel_large_batch_test...");
    let batch_size = 64;
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(42);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| (0..in_features).map(|_| (rng.random_range(0..3) - 1) as i8).collect())
        .collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    let act_scales = vec![act_scale; batch_size];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &WgpuContext::new().await.unwrap(),
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(7, "kernel_large_batch_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("kernel_large_batch_test", duration);
}

#[tokio::test]
async fn kernel_all_zero_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(8, "Running kernel_all_zero_test...");
    let batch_size = 32;
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(44);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| vec![0i8; in_features])
        .collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    let act_scales = vec![act_scale; batch_size];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &WgpuContext::new().await.unwrap(),
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(8, "kernel_all_zero_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("kernel_all_zero_test", duration);
}

#[tokio::test]
async fn kernel_all_plus_one_weights_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(9, "Running kernel_all_plus_one_weights_test...");
    let batch_size = 32;
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(45);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| vec![1i8; in_features])
        .collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    let act_scales = vec![act_scale; batch_size];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &WgpuContext::new().await.unwrap(),
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(9, "kernel_all_plus_one_weights_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("kernel_all_plus_one_weights_test", duration);
}

#[tokio::test]
async fn kernel_all_minus_one_weights_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(10, "Running kernel_all_minus_one_weights_test...");
    let batch_size = 32;
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(46);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| vec![-1i8; in_features])
        .collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    let act_scales = vec![act_scale; batch_size];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &WgpuContext::new().await.unwrap(),
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(10, "kernel_all_minus_one_weights_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("kernel_all_minus_one_weights_test", duration);
}

#[tokio::test]
async fn kernel_non_divisible_batch_test() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(11, "Running kernel_non_divisible_batch_test...");
    let batch_size = 33;  // Not divisible by 32
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(47);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| (0..in_features).map(|_| (rng.random_range(0..3) - 1) as i8).collect())
        .collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let (packed_weights_u32, weight_scales_f32) = pack_ternary_weights(&weights_i8);
    let act_scales = vec![act_scale; batch_size];

    let expected_output = matmul_quantized_scalar(
        &q_acts_vec,
        &packed_weights_u32,
        &act_scales,
        &weight_scales_f32,
        batch_size,
        in_features,
        out_features
    );

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_gpu_kernel(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
        &WgpuContext::new().await.unwrap(),
    ).await;

    assert_vec_eq(&expected_output, &gpu_output, 1e-5);
    TEST_REPORTER.log_message(11, "kernel_non_divisible_batch_test passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("kernel_non_divisible_batch_test", duration);
}

#[tokio::test]
async fn test_bitlinear_layer_forward_pass() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(12, "Running test_bitlinear_layer_forward_pass...");
    let batch_size = 32;
    let in_features = 1024;
    let out_features = 1024;
    let mut rng = StdRng::seed_from_u64(48);

    // Generate random input
    let input: Vec<f32> = (0..batch_size * in_features)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect();

    // Generate random weights
    let weights_i8: Vec<Vec<i8>> = (0..out_features)
        .map(|_| (0..in_features).map(|_| (rng.random_range(0..3) - 1) as i8).collect())
        .collect();

    let record = BitLinearRecord {
        packed_weights: pack_ternary_weights(&weights_i8).0,
        weight_scales: calculate_weight_scales(&weights_i8),
        in_features,
        out_features,
    };

    let context = WgpuContext::new().await.unwrap();
    let layer = BitLinear::from_record(record);
    let output = layer.forward(&context, &input, batch_size).await;

    assert_eq!(output.len(), batch_size * out_features);
    TEST_REPORTER.log_message(12, "test_bitlinear_layer_forward_pass passed.");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_bitlinear_layer_forward_pass", duration);
}

#[test]
#[serial]
fn zzz_final_report() {
    // This test runs last and generates the final report.
    // Add a small delay to ensure all async tests complete.
    std::thread::sleep(std::time::Duration::from_secs(2));
    TEST_REPORTER.generate_report();
}