use wgpu::util::DeviceExt;
use pollster::block_on;
use bytemuck;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use std::fs;
mod consts;
use consts::ADD_SCALAR_WGSL_PATH;
use consts::BITNET_KERNEL_WGSL_PATH;
use bytemuck::{Pod, Zeroable};

#[test]
fn test_wgpu_custom_kernel_add_scalar() {
    block_on(run());
}

#[test]
fn test_bitnet_kernel() {
    block_on(run_bitnet());
}

#[test]
fn test_bitnet_kernel_minimal() {
    block_on(run_bitnet_minimal());
}

async fn run() {
    // 1. Initialize wgpu
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

    // 2. Prepare data
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output_data: Vec<f32> = vec![0.0; input_data.len()];
    let scalar_data: Vec<f32> = vec![10.0];

    // 3. Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scalar Buffer"),
        contents: bytemuck::cast_slice(&scalar_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (output_data.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 4. Load WGSL kernel from path constant
    let shader_src = fs::read_to_string(ADD_SCALAR_WGSL_PATH).expect("Failed to read WGSL kernel file");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("AddScalarShader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // 5. Create pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scalar_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // 6. Encode and submit commands
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(((input_data.len() as u32) + 63) / 64, 1, 1);
    }

    // Copy output buffer to a mapped buffer so we can read it
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (output_data.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (output_data.len() * std::mem::size_of::<f32>()) as u64);

    queue.submit(Some(encoder.finish()));

    // 7. Read back the result
    let buffer_slice = staging_buffer.slice(..);
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
        done_clone.store(true, Ordering::SeqCst);
    });
    device.poll(wgpu::Maintain::Wait);
    while !done.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }
    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    let expected = vec![11.0, 12.0, 13.0, 14.0, 15.0];
    assert_eq!(result, expected);
    println!("WGPU custom kernel test passed! Result: {:?}", result);
}

async fn run_bitnet() {
    // 1. Problem size: 2x2x16 (M=2, N=2, K=16)
    let m = 2u32;
    let n = 2u32;
    let k = 16u32;
    let k_packed = 1u32; // 16/16

    // 2. Data setup
    // Activations: [[1..16], [16..1]]
    let activations_f32: Vec<f32> = (1..=16).map(|x| x as f32).chain((1..=16).rev().map(|x| x as f32)).collect();
    let (q_acts, act_scale) = quantize_activations_scalar(&activations_f32[0..16]);
    let (q_acts2, act_scale2) = quantize_activations_scalar(&activations_f32[16..32]);
    let q_acts_all: Vec<i32> = q_acts.iter().chain(q_acts2.iter()).copied().collect();
    let activation_scales = vec![act_scale, act_scale2];

    // Weights: 2 output features, each with 16 weights
    // Row 0: all +1, Row 1: [-1, 0, 1, -1, 0, 1, ...]
    let weights0: Vec<i8> = vec![1; 16];
    let weights1: Vec<i8> = (0..16).map(|i| match i % 3 {
        0 => -1,
        1 => 0,
        _ => 1,
    }).collect();
    let packed_weights = vec![pack_ternary_weights(&weights0), pack_ternary_weights(&weights1)];
    let weight_scales = vec![1.0f32, 1.0f32];

    // Metadata struct (must match WGSL layout)
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct BitnetMetadata {
        m: u32,
        n: u32,
        k: u32,
        k_packed: u32,
    }
    let metadata = BitnetMetadata { m, n, k, k_packed };

    // 3. WGPU setup
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

    // 4. Buffers
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Metadata"),
        contents: bytemuck::bytes_of(&metadata),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let activations_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Activations"),
        contents: bytemuck::cast_slice(&q_acts_all),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let packed_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PackedWeights"),
        contents: bytemuck::cast_slice(&packed_weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weight_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("WeightScales"),
        contents: bytemuck::cast_slice(&weight_scales),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let activation_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ActivationScales"),
        contents: bytemuck::cast_slice(&activation_scales),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (m * n * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 5. Load kernel
    let shader_src = std::fs::read_to_string(BITNET_KERNEL_WGSL_PATH).expect("Failed to read kernel");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BitNetKernel"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // 6. Bind group and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BindGroupLayout"),
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
        label: Some("BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: activations_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: packed_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weight_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: activation_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PipelineLayout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ComputePipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // 7. Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ComputePass"), timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1); // 2x2 fits in 1 workgroup
    }
    // Copy output to staging buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("StagingBuffer"),
        size: (m * n * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (m * n * std::mem::size_of::<f32>() as u32) as u64);
    queue.submit(Some(encoder.finish()));

    // 8. Read back
    let buffer_slice = staging_buffer.slice(..);
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
        done_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    device.poll(wgpu::Maintain::Wait);
    while !done.load(std::sync::atomic::Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);

    // 9. CPU reference
    let cpu_outputs = [
        // Row 0 x Col 0
        q_acts.iter().zip(weights0.iter()).map(|(&a, &w)| a as f32 * w as f32).sum::<f32>() * act_scale * 1.0,
        // Row 0 x Col 1
        q_acts.iter().zip(weights1.iter()).map(|(&a, &w)| a as f32 * w as f32).sum::<f32>() * act_scale * 1.0,
        // Row 1 x Col 0
        q_acts2.iter().zip(weights0.iter()).map(|(&a, &w)| a as f32 * w as f32).sum::<f32>() * act_scale2 * 1.0,
        // Row 1 x Col 1
        q_acts2.iter().zip(weights1.iter()).map(|(&a, &w)| a as f32 * w as f32).sum::<f32>() * act_scale2 * 1.0,
    ];
    for i in 0..4 {
        assert!((result[i] - cpu_outputs[i]).abs() < 1e-3, "Mismatch at {}: GPU {} vs CPU {}", i, result[i], cpu_outputs[i]);
    }
    println!("BitNet 2x2x16 kernel test passed! GPU: {:?}, CPU: {:?}", result, cpu_outputs);
}

fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i32>, f32) {
    let abs_max = activations.iter().map(|&x| x.abs()).fold(f32::NEG_INFINITY, f32::max);
    let scale = abs_max / 127.0 + 1e-6;
    (
        activations.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i32).collect(),
        scale
    )
}

fn pack_ternary_weights(weights: &[i8]) -> u32 {
    let mut packed = 0u32;
    for (i, &w) in weights.iter().enumerate() {
        let bits = match w {
            -1 => 0b00,
            0  => 0b01,
            1  => 0b10,
            _  => 0b01,
        };
        packed |= (bits as u32) << (i * 2);
    }
    packed
}

async fn run_bitnet_minimal() {
    // 1. Data setup
    let m = 1u32; // batch
    let n = 1u32; // output features
    let k = 16u32; // input features
    let k_packed = 1u32; // 16/16

    // Activations: [1, 2, ..., 16]
    let activations_f32: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let (q_acts, act_scale) = quantize_activations_scalar(&activations_f32);

    // Weights: all +1
    let weights_i8: Vec<i8> = vec![1; 16];
    let packed_weights = vec![pack_ternary_weights(&weights_i8)];

    // Scales
    let weight_scales = vec![1.0f32];
    let activation_scales = vec![act_scale];

    // Metadata struct (must match WGSL layout)
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct BitnetMetadata {
        m: u32,
        n: u32,
        k: u32,
        k_packed: u32,
    }
    let metadata = BitnetMetadata { m, n, k, k_packed };

    // 2. WGPU setup
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

    // 3. Buffers
    let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Metadata"),
        contents: bytemuck::bytes_of(&metadata),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let activations_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Activations"),
        contents: bytemuck::cast_slice(&q_acts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let packed_weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("PackedWeights"),
        contents: bytemuck::cast_slice(&packed_weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weight_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("WeightScales"),
        contents: bytemuck::cast_slice(&weight_scales),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let activation_scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ActivationScales"),
        contents: bytemuck::cast_slice(&activation_scales),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (m * n * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 4. Load kernel
    let shader_src = std::fs::read_to_string(BITNET_KERNEL_WGSL_PATH).expect("Failed to read kernel");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BitNetKernel"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // 5. Bind group and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BindGroupLayout"),
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
        label: Some("BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: activations_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: packed_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weight_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: activation_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PipelineLayout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ComputePipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    // 6. Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ComputePass"), timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1); // minimal: 1 workgroup
    }
    // Copy output to staging buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("StagingBuffer"),
        size: (m * n * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (m * n * std::mem::size_of::<f32>() as u32) as u64);
    queue.submit(Some(encoder.finish()));

    // 7. Read back
    let buffer_slice = staging_buffer.slice(..);
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
        done_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    device.poll(wgpu::Maintain::Wait);
    while !done.load(std::sync::atomic::Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);

    // 8. CPU reference
    let expected: f32 = q_acts.iter().map(|&x| x as f32).sum::<f32>() * act_scale * 1.0;

    assert!((result[0] - expected).abs() < 1e-3, "GPU: {}, CPU: {}", result[0], expected);
    println!("BitNet minimal kernel test passed! GPU: {}, CPU: {}", result[0], expected);
}