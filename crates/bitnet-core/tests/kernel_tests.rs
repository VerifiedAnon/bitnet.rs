//! Kernel tests for BitNet: pure Rust + direct wgpu (no burn)

use std::sync::Arc;
use wgpu::util::DeviceExt;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use bytemuck::{Pod, Zeroable};
use bitnet_core::kernels::{pack_ternary_weights, calculate_weight_scales, BitnetMetadata};

// --- Scalar reference logic ---
fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations.iter().map(|&x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
    let scale = abs_max / 127.0 + 1e-6;
    (activations.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect(), scale)
}

fn matmul_quantized_scalar(q_activations: &[i8], packed_weights: &[u32], activation_scales: &[f32], weight_scales: &[f32], batch_size: usize, in_features: usize, out_features: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch_size * out_features];
    for b in 0..batch_size {
        let act_offset = b * in_features;
        let current_activations = &q_activations[act_offset..act_offset + in_features];
        let current_act_scale = activation_scales[b];
        for n in 0..out_features {
            let mut acc: i32 = 0;
            for k_chunk in 0..(in_features / 16) {
                let weight_idx = n * (in_features / 16) + k_chunk;
                let packed_val = packed_weights[weight_idx];
                for i in 0..16 {
                    let encoded = (packed_val >> (i * 2)) & 0b11;
                    let weight = match encoded { 0b00 => -1, 0b01 => 0, 0b10 => 1, _ => 0 };
                    let act_idx = k_chunk * 16 + i;
                    acc += (current_activations[act_idx] as i32) * (weight as i32);
                }
            }
            let out_idx = b * out_features + n;
            output[out_idx] = (acc as f32) * current_act_scale * weight_scales[n];
        }
    }
    output
}

fn assert_vec_eq(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vectors have different lengths");
    a.iter().zip(b.iter()).enumerate().for_each(|(i, (x, y))| {
        assert!((x - y).abs() < tolerance, "Mismatch at index {}: left={}, right={} (diff={})", i, x, y, (x-y).abs());
    });
}

// --- Direct wgpu kernel launcher ---
async fn launch_bitnet_kernel_wgpu(
    quantized_input: &[i8], // [batch_size, in_features]
    packed_weights: &[u32], // [out_features, in_features/16]
    weight_scales: &[f32],  // [out_features]
    activation_scales: &[f32], // [batch_size]
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    shader_source: &str,
) -> Vec<f32> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await.expect("No suitable GPU adapters found");
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        label: None,
        memory_hints: Default::default(),
        trace: wgpu::Trace::default(),
    }).await.expect("Failed to create device");

    let input_bytes = bytemuck::cast_slice(quantized_input);
    let weights_bytes = bytemuck::cast_slice(packed_weights);
    let wscales_bytes = bytemuck::cast_slice(weight_scales);
    let ascales_bytes = bytemuck::cast_slice(activation_scales);
    let output_size = batch_size * out_features;
    let output_bytes = output_size * std::mem::size_of::<f32>();
    let metadata = BitnetMetadata {
        m: batch_size as u32,
        n: out_features as u32,
        k: in_features as u32,
        k_packed: (in_features / 16) as u32,
    };
    let metadata_bytes = bytemuck::bytes_of(&metadata);

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"), contents: input_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("weights"), contents: weights_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let wscales_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("wscales"), contents: wscales_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let ascales_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ascales"), contents: ascales_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"), size: output_bytes as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let metadata_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("metadata"), contents: metadata_bytes, usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bitnet_kernel"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BitNet kernel pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weights_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wscales_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: ascales_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cpass"), timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroup_x = (out_features as u32 + 63) / 64;
        let workgroup_y = (batch_size as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }
    queue.submit(Some(encoder.finish()));

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_bytes as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buf.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    // device.poll(wgpu::Maintain::Wait); // No longer needed in wgpu 25.x
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();
    result
}

// --- TESTS ---

#[test]
fn unit_test_pack_ternary_weights() {
    let weights = vec![-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1];
    let packed = pack_ternary_weights(&weights);
    assert_eq!(packed.len(), 1);
    assert_eq!(packed[0], 0b10010010010010010010010010010010, "Packing logic is incorrect");
}

#[test]
fn unit_test_calculate_weight_scales() {
    let weights = vec![vec![-1, 0, 1, 1, 0, -1, 1, 0], vec![1, 1, 1, 1], vec![0, 0]];
    let scales = calculate_weight_scales(&weights);
    assert_eq!(scales, vec![1.0, 1.0, 1.0], "Scale calculation is incorrect");
}

#[tokio::test]
async fn low_level_kernel_correctness_test() {
    let batch_size = 1;
    let in_features = 32;
    let out_features = 16;
    let mut rng = StdRng::seed_from_u64(43);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.gen::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features).map(|_| (0..in_features).map(|_| (rng.gen_range(0..3) - 1) as i8).collect()).collect();

    let (q_acts_vec, act_scale) = quantize_activations_scalar(&activations_f32);
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);

    let expected_output = matmul_quantized_scalar(&q_acts_vec, &packed_weights_u32, &[act_scale], &weight_scales_f32, batch_size, in_features, out_features);

    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &q_acts_vec,
        &packed_weights_u32,
        &weight_scales_f32,
        &[act_scale],
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;

    println!("LEVEL 2: Comparing direct kernel launch output with scalar reference...");
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("SUCCESS! Direct kernel launch is correct.");
}

#[tokio::test]
async fn kernel_large_batch_test() {
    let batch_size = 4;
    let in_features = 64;
    let out_features = 32;
    let mut rng = StdRng::seed_from_u64(123);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.gen::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features).map(|_| (0..in_features).map(|_| (rng.gen_range(0..3) - 1) as i8).collect()).collect();
    let mut all_q_activations = Vec::new();
    let mut all_act_scales = Vec::new();
    for b in 0..batch_size {
        let (q, s) = quantize_activations_scalar(&activations_f32[b*in_features..(b+1)*in_features]);
        all_q_activations.extend(q);
        all_act_scales.push(s);
    }
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);
    let expected_output = matmul_quantized_scalar(&all_q_activations, &packed_weights_u32, &all_act_scales, &weight_scales_f32, batch_size, in_features, out_features);
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &all_q_activations,
        &packed_weights_u32,
        &weight_scales_f32,
        &all_act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("kernel_large_batch_test passed!");
}

#[tokio::test]
async fn kernel_all_zero_test() {
    let batch_size = 2;
    let in_features = 16;
    let out_features = 8;
    let activations_f32 = vec![0.0; batch_size * in_features];
    let weights_i8 = vec![vec![0; in_features]; out_features];
    let all_q_activations = vec![0i8; batch_size * in_features];
    let all_act_scales = vec![1.0; batch_size];
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);
    let expected_output = matmul_quantized_scalar(&all_q_activations, &packed_weights_u32, &all_act_scales, &weight_scales_f32, batch_size, in_features, out_features);
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &all_q_activations,
        &packed_weights_u32,
        &weight_scales_f32,
        &all_act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("kernel_all_zero_test passed!");
}

#[tokio::test]
async fn kernel_all_plus_one_weights_test() {
    let batch_size = 2;
    let in_features = 16;
    let out_features = 8;
    let mut rng = StdRng::seed_from_u64(456);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.gen::<f32>() - 0.5).collect();
    let weights_i8 = vec![vec![1; in_features]; out_features];
    let mut all_q_activations = Vec::new();
    let mut all_act_scales = Vec::new();
    for b in 0..batch_size {
        let (q, s) = quantize_activations_scalar(&activations_f32[b*in_features..(b+1)*in_features]);
        all_q_activations.extend(q);
        all_act_scales.push(s);
    }
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);
    let expected_output = matmul_quantized_scalar(&all_q_activations, &packed_weights_u32, &all_act_scales, &weight_scales_f32, batch_size, in_features, out_features);
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &all_q_activations,
        &packed_weights_u32,
        &weight_scales_f32,
        &all_act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("kernel_all_plus_one_weights_test passed!");
}

#[tokio::test]
async fn kernel_all_minus_one_weights_test() {
    let batch_size = 2;
    let in_features = 16;
    let out_features = 8;
    let mut rng = StdRng::seed_from_u64(789);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.gen::<f32>() - 0.5).collect();
    let weights_i8 = vec![vec![-1; in_features]; out_features];
    let mut all_q_activations = Vec::new();
    let mut all_act_scales = Vec::new();
    for b in 0..batch_size {
        let (q, s) = quantize_activations_scalar(&activations_f32[b*in_features..(b+1)*in_features]);
        all_q_activations.extend(q);
        all_act_scales.push(s);
    }
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);
    let expected_output = matmul_quantized_scalar(&all_q_activations, &packed_weights_u32, &all_act_scales, &weight_scales_f32, batch_size, in_features, out_features);
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &all_q_activations,
        &packed_weights_u32,
        &weight_scales_f32,
        &all_act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("kernel_all_minus_one_weights_test passed!");
}

#[tokio::test]
async fn kernel_non_divisible_batch_test() {
    let batch_size = 3;
    let in_features = 32;
    let out_features = 7;
    let mut rng = StdRng::seed_from_u64(321);
    let activations_f32: Vec<f32> = (0..batch_size * in_features).map(|_| rng.gen::<f32>() - 0.5).collect();
    let weights_i8: Vec<Vec<i8>> = (0..out_features).map(|_| (0..in_features).map(|_| (rng.gen_range(0..3) - 1) as i8).collect()).collect();
    let mut all_q_activations = Vec::new();
    let mut all_act_scales = Vec::new();
    for b in 0..batch_size {
        let (q, s) = quantize_activations_scalar(&activations_f32[b*in_features..(b+1)*in_features]);
        all_q_activations.extend(q);
        all_act_scales.push(s);
    }
    let packed_weights_u32: Vec<u32> = weights_i8.iter().flat_map(|r| pack_ternary_weights(r)).collect();
    let weight_scales_f32 = calculate_weight_scales(&weights_i8);
    let expected_output = matmul_quantized_scalar(&all_q_activations, &packed_weights_u32, &all_act_scales, &weight_scales_f32, batch_size, in_features, out_features);
    let shader_source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let gpu_output = launch_bitnet_kernel_wgpu(
        &all_q_activations,
        &packed_weights_u32,
        &weight_scales_f32,
        &all_act_scales,
        batch_size,
        in_features,
        out_features,
        shader_source,
    ).await;
    assert_vec_eq(&gpu_output, &expected_output, 1e-3);
    println!("kernel_non_divisible_batch_test passed!");
}