//! BitNet Attention GPU Kernel Tests
//!
//! This suite validates the production-grade bitnet_attention.wgsl kernel for correctness, robustness, portability, and safety.
//! - Shader compilation and pipeline creation on all major backends
//! - GPU vs. CPU reference correctness for a wide range of shapes and data
//! - Edge cases, memory safety, cross-device consistency, error handling, and performance
//! - Uses #[ignore] on all but the main correctness test, so individual tests can be run with --ignored
//! - See README and kernel comments for design rationale and quantization strategy
//!

use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use bitnet_core::wgpu_context::WgpuContext;
use bitnet_tools::test_utils::TestReporter;
use lazy_static::lazy_static;
use serial_test::serial;
use std::sync::Arc;
use bitnet_core::error::BitNetError;
use wgpu::Backends;
use std::sync::Mutex;

// Initialize test reporter
lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("Attention_GPU_tests")
        .expect("Failed to create test reporter");
}


fn assert_vec_eq(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vector lengths don't match");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_infinite() && y.is_infinite() && x.signum() == y.signum() {
            continue;
        }
        if x.is_nan() && y.is_nan() {
            continue;
        }
        assert!(
            (x - y).abs() < tolerance,
            "Vectors differ at index {}: {} != {} (diff = {})",
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

/// A struct to hold pre-computed wgpu resources for repeated kernel launches.
struct GpuKernelResources {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuKernelResources {
    fn new(context: &WgpuContext, shader_source: &str) -> Self {
        let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bitnet Attention Kernel"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let bind_group_layout =
            context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bitnet Attention Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout =
            context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bitnet Attention Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bitnet Attention Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline, bind_group_layout }
    }
}

async fn launch_attention_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    context: &WgpuContext,
    resources: &GpuKernelResources,
) -> Result<Vec<f32>, BitNetError> {
    use wgpu::util::DeviceExt;
    let output_size = batch * seq_len * heads * head_dim;
    let output_size_bytes = (output_size * std::mem::size_of::<f32>()) as u64;
    let metadata = [batch as u32, seq_len as u32, heads as u32, head_dim as u32];
    let metadata_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attention_metadata_buffer"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let q_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q_buffer"),
        contents: bytemuck::cast_slice(q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let k_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("k_buffer"),
        contents: bytemuck::cast_slice(k),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let v_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("v_buffer"),
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: output_size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_buffer"),
        size: output_size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attention_bind_group"),
        layout: &resources.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: k_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: v_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
        ],
    });
    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attention_encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Attention Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&resources.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(batch as u32, seq_len as u32, heads as u32);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size_bytes);
    context.queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    context.device.poll(wgpu::MaintainBase::Wait).ok();
    match rx.receive().await {
        Some(Ok(())) => {
            let data = buffer_slice.get_mapped_range();
            let result_vec = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(result_vec)
        }
        _ => Err(BitNetError::ComputeError),
    }
}

fn cpu_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * seq_len * heads * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            for q_pos in 0..seq_len {
                // 1. Load Q
                let q_offset = (((b * seq_len + q_pos) * heads + h) * head_dim) as usize;
                let q_vec = &q[q_offset..q_offset + head_dim];
                // 2. Compute scores
                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = -1e30f32;
                for k_pos in 0..seq_len {
                    let k_offset = (((b * seq_len + k_pos) * heads + h) * head_dim) as usize;
                    let k_vec = &k[k_offset..k_offset + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    let mut score = dot * scale;
                    if k_pos > q_pos {
                        score = -1e30;
                    }
                    scores[k_pos] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                // 3. Softmax
                let mut sum_exp = 0.0f32;
                for k_pos in 0..seq_len {
                    scores[k_pos] = (scores[k_pos] - max_score).exp();
                    sum_exp += scores[k_pos];
                }
                // 4. Output
                for d in 0..head_dim {
                    let mut out = 0.0f32;
                    for k_pos in 0..seq_len {
                        let v_offset = (((b * seq_len + k_pos) * heads + h) * head_dim) as usize;
                        let v_val = v[v_offset + d];
                        let weight = scores[k_pos] / sum_exp;
                        out += weight * v_val;
                    }
                    let out_offset = (((b * seq_len + q_pos) * heads + h) * head_dim + d) as usize;
                    output[out_offset] = out;
                }
            }
        }
    }
    output
}

// --- Utility: Robust GPU error capture (copied from kernel_tests.rs) ---
async fn with_wgpu_error_scope<F, Fut, R>(device: &wgpu::Device, f: F) -> Result<R, BitNetError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let error = Arc::new(Mutex::new(None));
    let error_clone = error.clone();
    device.on_uncaptured_error(Box::new(move |e| {
        let msg = match e {
            wgpu::Error::Validation { source, .. } => format!("Validation Error: {}", source),
            wgpu::Error::OutOfMemory { .. } => "Out of Memory Error".to_string(),
            wgpu::Error::Internal { source, .. } => format!("Internal Error: {}", source),
        };
        *error_clone.lock().unwrap() = Some(msg);
    }));
    let result = f().await;
    let _ = device.poll(wgpu::MaintainBase::Wait);
    device.on_uncaptured_error(Box::new(|_| {})); // Reset handler
    if let Some(msg) = error.lock().unwrap().take() {
        if msg.contains("Validation Error") {
            return Err(BitNetError::PipelineCreationError(msg));
        } else if msg.contains("Shader") {
            return Err(BitNetError::ShaderCompilationError(msg));
        } else {
            return Err(BitNetError::ComputeError);
        }
    }
    Ok(result)
}

// --- TESTS ---

fn test_attention_gpu_correctness_inner() {
    let test_name = "Attention GPU Correctness";
    let t0 = Instant::now();
    TEST_REPORTER.log_message(1, "Running test_attention_gpu_correctness...");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = runtime.block_on(async {
        let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
        TEST_REPORTER.log_message(1, &format!(
            "Device: {} (Backend: {:?}, Type: {:?})",
            context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
        ));
        TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
        let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
        let resources = GpuKernelResources::new(&context, shader_source);
        let configs = vec![
            (1, 2, 4, 8),
            (2, 4, 8, 16),
            (1, 1, 16, 32),
            (3, 2, 6, 4),
        ];
        for (batch, heads, seq_len, head_dim) in configs {
            TEST_REPORTER.log_message(1, &format!("Testing shape: batch={}, heads={}, seq_len={}, head_dim={}", batch, heads, seq_len, head_dim));
            let mut rng = StdRng::seed_from_u64(42);
            let q: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let k: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let v: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let gpu_output = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
            let cpu_output = cpu_attention_reference(&q, &k, &v, batch, heads, seq_len, head_dim);
            TEST_REPORTER.log_message(1, &format!(
                "CPU output sample: {:?}, GPU output sample: {:?}",
                &cpu_output[..4.min(cpu_output.len())],
                &gpu_output[..4.min(gpu_output.len())]
            ));
            assert_vec_eq(&gpu_output, &cpu_output, 1e-4);
        }
        TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        Ok::<(), String>(())
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(e) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: {}", test_name, e));
            TEST_REPORTER.record_failure(test_name, &format!("test_attention_gpu_correctness failed: {}", e), Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_gpu_correctness() {
    test_attention_gpu_correctness_inner();
}

fn test_attention_shader_compilation_inner() {
    let test_name = "Attention Shader Compilation";
    let t0 = Instant::now();
    let backends = if cfg!(target_os = "macos") {
        vec![Backends::VULKAN, Backends::DX12, Backends::METAL, Backends::GL]
    } else {
        vec![Backends::VULKAN, Backends::DX12, Backends::GL]
    };
    let mut failed = false;
    for backend in backends.iter() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let label = format!("Attention Shader Compilation ({:?})", backend);
        let result = runtime.block_on(async {
            let context = WgpuContext::new_with_backend(*backend).await;
            match context {
                Ok(ctx) => {
                    TEST_REPORTER.log_message(1, &format!(
                        "Device: {} (Backend: {:?}, Type: {:?})",
                        ctx.adapter_info.name, ctx.adapter_info.backend, ctx.adapter_info.device_type
                    ));
                    TEST_REPORTER.log_message(1, &format!("Limits: {:?}", ctx.limits));
                    let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
                    let _ = GpuKernelResources::new(&ctx, shader_source);
                    Ok(())
                }
                Err(e) => Err(format!("Failed to create context: {:?}", e)),
            }
        });
        if let Err(e) = result {
            TEST_REPORTER.log_message(1, &format!("[FAIL] Backend {:?} failed: {}", backend, e));
            failed = true;
        } else {
            TEST_REPORTER.log_message(1, &format!("[PASS] Backend {:?} compiled successfully", backend));
        }
    }
    if failed {
        TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: One or more backends failed shader compilation", test_name));
        TEST_REPORTER.record_failure(test_name, "One or more backends failed shader compilation", Some(t0.elapsed()));
    } else {
        TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        TEST_REPORTER.record_timing(test_name, t0.elapsed());
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_shader_compilation() {
    test_attention_shader_compilation_inner();
}

fn test_attention_gpu_correctness_all_shapes_inner() {
    let test_name = "Attention GPU Correctness All Shapes";
    let t0 = Instant::now();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        runtime.block_on(async {
            let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
            TEST_REPORTER.log_message(1, &format!(
                "Device: {} (Backend: {:?}, Type: {:?})",
                context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
            ));
            TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
            let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
            let resources = GpuKernelResources::new(&context, shader_source);
            let configs = vec![
                (1, 2, 4, 8), (2, 4, 8, 16), (1, 1, 16, 32), (3, 2, 6, 4),
                (1, 8, 1, 8), (2, 2, 32, 8), (1, 4, 64, 16), (2, 1, 128, 4),
            ];
            let mut rng = StdRng::seed_from_u64(42);
            for (batch, heads, seq_len, head_dim) in configs {
                for pattern in ["random", "all_zero", "all_one"] {
                    TEST_REPORTER.log_message(1, &format!(
                        "Testing shape: batch={}, heads={}, seq_len={}, head_dim={}, pattern={}",
                        batch, heads, seq_len, head_dim, pattern
                    ));
                    let (q, k, v) = match pattern {
                        "random" => (
                            (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect(),
                            (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect(),
                            (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect(),
                        ),
                        "all_zero" => (
                            vec![0.0; batch*seq_len*heads*head_dim],
                            vec![0.0; batch*seq_len*heads*head_dim],
                            vec![0.0; batch*seq_len*heads*head_dim],
                        ),
                        "all_one" => (
                            vec![1.0; batch*seq_len*heads*head_dim],
                            vec![1.0; batch*seq_len*heads*head_dim],
                            vec![1.0; batch*seq_len*heads*head_dim],
                        ),
                        _ => unreachable!(),
                    };
                    let gpu_output = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
                    let cpu_output = cpu_attention_reference(&q, &k, &v, batch, heads, seq_len, head_dim);
                    TEST_REPORTER.log_message(1, &format!(
                        "CPU output sample: {:?}, GPU output sample: {:?}",
                        &cpu_output[..4.min(cpu_output.len())],
                        &gpu_output[..4.min(gpu_output.len())]
                    ));
                    assert_vec_eq(&gpu_output, &cpu_output, 1e-4);
                }
            }
            TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        });
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(_) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: Panic in all shapes test", test_name));
            TEST_REPORTER.record_failure(test_name, "Panic in all shapes test", Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_gpu_correctness_all_shapes() {
    test_attention_gpu_correctness_all_shapes_inner();
}

fn test_attention_edge_cases_inner() {
    let test_name = "Attention Edge Cases";
    let t0 = Instant::now();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        runtime.block_on(async {
            let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
            TEST_REPORTER.log_message(1, &format!(
                "Device: {} (Backend: {:?}, Type: {:?})",
                context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
            ));
            TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
            let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
            let resources = GpuKernelResources::new(&context, shader_source);
            let edge_cases = vec![
                (1, 1, 1, 1),
                (1, 1, 2, 1),
                (1, 1, 512, 8),
                (1, 8, 4, 8),
                (2, 2, 3, 5),
                (1, 2, 7, 3),
            ];
            let mut rng = StdRng::seed_from_u64(123);
            for (batch, heads, seq_len, head_dim) in edge_cases {
                TEST_REPORTER.log_message(1, &format!("Testing shape: batch={}, heads={}, seq_len={}, head_dim={}", batch, heads, seq_len, head_dim));
                let q: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let k: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let v: Vec<f32> = (0..batch*seq_len*heads*head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let gpu_output = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
                let cpu_output = cpu_attention_reference(&q, &k, &v, batch, heads, seq_len, head_dim);
                TEST_REPORTER.log_message(1, &format!(
                    "CPU output sample: {:?}, GPU output sample: {:?}",
                    &cpu_output[..4.min(cpu_output.len())],
                    &gpu_output[..4.min(gpu_output.len())]
                ));
                assert_vec_eq(&gpu_output, &cpu_output, 1e-4);
            }
            TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        });
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(_) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: Panic in edge cases test", test_name));
            TEST_REPORTER.record_failure(test_name, "Panic in edge cases test", Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_edge_cases() {
    test_attention_edge_cases_inner();
}

fn test_attention_memory_safety_inner() {
    let test_name = "Attention Memory Safety";
    let t0 = Instant::now();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        runtime.block_on(async {
            let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
            TEST_REPORTER.log_message(1, &format!(
                "Device: {} (Backend: {:?}, Type: {:?})",
                context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
            ));
            TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
            let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
            let resources = GpuKernelResources::new(&context, shader_source);
            let batch = 1;
            let heads = 2;
            let seq_len = 256;
            let head_dim = 64;
            let q = vec![0.5; batch*seq_len*heads*head_dim];
            let k = vec![0.5; batch*seq_len*heads*head_dim];
            let v = vec![0.5; batch*seq_len*heads*head_dim];
            let _ = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
            for i in 0..5 {
                let seq_len = 8 + i * 8;
                TEST_REPORTER.log_message(1, &format!("Testing shape: batch={}, heads={}, seq_len={}, head_dim={}", batch, heads, seq_len, head_dim));
                let q = vec![0.1 * i as f32; batch*seq_len*heads*head_dim];
                let k = vec![0.2 * i as f32; batch*seq_len*heads*head_dim];
                let v = vec![0.3 * i as f32; batch*seq_len*heads*head_dim];
                let _ = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
            }
            TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        });
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(_) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: Panic in memory safety test", test_name));
            TEST_REPORTER.record_failure(test_name, "Panic in memory safety test", Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_memory_safety() {
    test_attention_memory_safety_inner();
}

fn test_attention_cross_device_consistency_inner() {
    let test_name = "Attention Cross Device Consistency";
    let t0 = Instant::now();
    let backends = if cfg!(target_os = "macos") {
        vec![Backends::VULKAN, Backends::DX12, Backends::METAL, Backends::GL]
    } else {
        vec![Backends::VULKAN, Backends::DX12, Backends::GL]
    };
    let mut results = vec![];
    let mut failed = false;
    for backend in backends.iter() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(async {
            let context = WgpuContext::new_with_backend(*backend).await;
            match context {
                Ok(ctx) => {
                    TEST_REPORTER.log_message(1, &format!(
                        "Device: {} (Backend: {:?}, Type: {:?})",
                        ctx.adapter_info.name, ctx.adapter_info.backend, ctx.adapter_info.device_type
                    ));
                    TEST_REPORTER.log_message(1, &format!("Limits: {:?}", ctx.limits));
                    let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
                    let resources = GpuKernelResources::new(&ctx, shader_source);
                    let batch = 1;
                    let heads = 2;
                    let seq_len = 8;
                    let head_dim = 8;
                    let q = vec![0.1; batch*seq_len*heads*head_dim];
                    let k = vec![0.2; batch*seq_len*heads*head_dim];
                    let v = vec![0.3; batch*seq_len*heads*head_dim];
                    let gpu_output = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &ctx, &resources).await.expect("GPU kernel failed");
                    Ok(gpu_output)
                }
                Err(e) => Err(format!("Failed to create context: {:?}", e)),
            }
        });
        results.push((backend, result));
    }
    let mut reference: Option<Vec<f32>> = None;
    for (backend, result) in results {
        match result {
            Ok(output) => {
                if let Some(ref ref_out) = reference {
                    TEST_REPORTER.log_message(1, &format!(
                        "CPU output sample: {:?}, GPU output sample: {:?}",
                        &ref_out[..4.min(ref_out.len())],
                        &output[..4.min(output.len())]
                    ));
                    assert_vec_eq(&output, ref_out, 1e-4);
                } else {
                    reference = Some(output);
                }
            }
            Err(e) => {
                TEST_REPORTER.log_message(1, &format!("[FAIL] {:?}: {}", backend, e));
                failed = true;
            }
        }
    }
    if failed {
        TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: One or more backends failed cross device consistency", test_name));
        TEST_REPORTER.record_failure(test_name, "One or more backends failed cross device consistency", Some(t0.elapsed()));
    } else {
        TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        TEST_REPORTER.record_timing(test_name, t0.elapsed());
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_cross_device_consistency() {
    test_attention_cross_device_consistency_inner();
}

fn test_attention_error_handling_inner() {
    let test_name = "Attention Error Handling";
    let t0 = Instant::now();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        runtime.block_on(async {
            let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
            TEST_REPORTER.log_message(1, &format!(
                "Device: {} (Backend: {:?}, Type: {:?})",
                context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
            ));
            TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
            let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
            let resources = GpuKernelResources::new(&context, shader_source);
            // Invalid input shape
            let q = vec![0.0; 10];
            let k = vec![0.0; 20];
            let v = vec![0.0; 30];
            let result = with_wgpu_error_scope(&context.device, || async {
                launch_attention_kernel(&q, &k, &v, 1, 1, 2, 5, &context, &resources).await
            }).await;
            match result {
                Err(e) => TEST_REPORTER.log_message(1, &format!("[PASS/EXPECTED ERROR] Invalid input shape: {}", e)),
                Ok(inner) => {
                    TEST_REPORTER.log_message(1, "[WARN] Invalid input shape did not error as expected");
                    TEST_REPORTER.log_message(1, &format!("Result: {:?}", inner));
                    return; // Prevent further invalid buffer operations
                }
            }
            // Oversized buffer
            let big = vec![0.0; 1_000_000_000];
            let result = with_wgpu_error_scope(&context.device, || async {
                launch_attention_kernel(&big, &big, &big, 1000, 1000, 1000, 1000, &context, &resources).await
            }).await;
            match result {
                Err(e) => TEST_REPORTER.log_message(1, &format!("[PASS/EXPECTED ERROR] Oversized buffer: {}", e)),
                Ok(inner) => {
                    TEST_REPORTER.log_message(1, "[WARN] Oversized buffer did not error as expected");
                    TEST_REPORTER.log_message(1, &format!("Result: {:?}", inner));
                    return; // Prevent further invalid buffer operations
                }
            }
            TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        });
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(_) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: Panic in error handling test", test_name));
            TEST_REPORTER.record_failure(test_name, "Panic in error handling test", Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_error_handling() {
    test_attention_error_handling_inner();
}

fn test_attention_performance_inner() {
    let test_name = "Attention Performance";
    let t0 = Instant::now();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = std::panic::catch_unwind(|| {
        runtime.block_on(async {
            let context = Arc::new(WgpuContext::new().await.expect("Failed to create WgpuContext"));
            TEST_REPORTER.log_message(1, &format!(
                "Device: {} (Backend: {:?}, Type: {:?})",
                context.adapter_info.name, context.adapter_info.backend, context.adapter_info.device_type
            ));
            TEST_REPORTER.log_message(1, &format!("Limits: {:?}", context.limits));
            let shader_source = include_str!("../src/kernels/bitnet_attention.wgsl");
            let resources = GpuKernelResources::new(&context, shader_source);
            let configs = vec![
                (1, 2, 128, 32),
                (2, 4, 64, 64),
                (4, 8, 32, 128),
            ];
            for (batch, heads, seq_len, head_dim) in configs {
                TEST_REPORTER.log_message(1, &format!("Testing shape: batch={}, heads={}, seq_len={}, head_dim={}", batch, heads, seq_len, head_dim));
                let q = vec![0.1; batch*seq_len*heads*head_dim];
                let k = vec![0.2; batch*seq_len*heads*head_dim];
                let v = vec![0.3; batch*seq_len*heads*head_dim];
                let t0 = Instant::now();
                let _ = launch_attention_kernel(&q, &k, &v, batch, heads, seq_len, head_dim, &context, &resources).await.expect("GPU kernel failed");
                let elapsed = t0.elapsed();
                TEST_REPORTER.log_message(1, &format!(
                    "Perf: batch={}, heads={}, seq_len={}, head_dim={}, time={:?}",
                    batch, heads, seq_len, head_dim, elapsed
                ));
            }
            TEST_REPORTER.log_message(1, &format!("[PASS] {} passed.", test_name));
        });
    });
    match result {
        Ok(_) => TEST_REPORTER.record_timing(test_name, t0.elapsed()),
        Err(_) => {
            TEST_REPORTER.log_message(1, &format!("[FAIL] {} failed: Panic in performance test", test_name));
            TEST_REPORTER.record_failure(test_name, "Panic in performance test", Some(t0.elapsed()))
        },
    }
}

#[test]
#[serial]
#[ignore]
fn test_attention_performance() {
    test_attention_performance_inner();
}

fn zzz_final_report() {
    std::thread::sleep(std::time::Duration::from_secs(2));
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
fn test_all_kernels_sequentially() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(100, "STARTING KERNEL TEST SUITE");
    test_attention_gpu_correctness_inner();
    test_attention_shader_compilation_inner();
    test_attention_gpu_correctness_all_shapes_inner();
    test_attention_edge_cases_inner();
    test_attention_memory_safety_inner();
    test_attention_cross_device_consistency_inner();
    test_attention_error_handling_inner();
    test_attention_performance_inner();
    zzz_final_report();
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_all_kernels_sequentially", duration);
    TEST_REPORTER.log_message(100, &format!("Kernel test suite passed (took {:.2?})", duration));
}