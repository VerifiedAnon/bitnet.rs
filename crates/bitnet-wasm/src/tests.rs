use std::str::FromStr;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

#[derive(Debug)]
pub enum GpuTestError {
    PipelineCreationError(String),
    ShaderCompilationError(String),
    ComputeError(String),
    DeviceError(String),
    AdapterError(String),
    Other(String),
}

#[derive(Debug, Clone, Copy)]
pub enum WasmKernel {
    Optimal,
    Wasm,
    Default,
}

impl WasmKernel {
    pub fn source(&self) -> &'static str {
        match self {
            WasmKernel::Optimal => include_str!("../../bitnet-core/src/kernels/bitnet_kernel_optimal.wgsl"),
            WasmKernel::Wasm => include_str!("../../bitnet-core/src/kernels/bitnet_kernel_wasm.wgsl"),
            WasmKernel::Default => include_str!("../../bitnet-core/src/kernels/bitnet_kernel.wgsl"),
        }
    }
}

impl FromStr for WasmKernel {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "optimal" => Ok(WasmKernel::Optimal),
            "wasm" => Ok(WasmKernel::Wasm),
            "default" => Ok(WasmKernel::Default),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for WasmKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmKernel::Optimal => write!(f, "optimal"),
            WasmKernel::Wasm => write!(f, "wasm"),
            WasmKernel::Default => write!(f, "default"),
        }
    }
}

// Simplified error handling for WASM
pub async fn create_wgpu_device() -> Result<(wgpu::Device, wgpu::Queue, wgpu::Adapter), JsValue> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| JsValue::from_str("Failed to find suitable GPU adapter"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WASM Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            },
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to create device: {:?}", e)))?;

    Ok((device, queue, adapter))
}

// More robust shader compilation with better error handling
pub async fn compile_shader_safe(
    device: &wgpu::Device,
    shader_source: &str,
    label: &str,
) -> Result<wgpu::ShaderModule, JsValue> {
    // Set up error callback before creating shader
    use std::sync::{Arc, Mutex};
    let error_occurred = Arc::new(Mutex::new(None::<String>));
    let error_clone = error_occurred.clone();

    device.on_uncaptured_error(Box::new(move |error| {
        let error_message = match error {
            wgpu::Error::Validation { source, .. } => {
                format!("Validation error: {}", source)
            }
            wgpu::Error::OutOfMemory { .. } => {
                "Out of memory error".to_string()
            }
            wgpu::Error::Internal { source, .. } => {
                format!("Internal error: {}", source)
            }
        };
        *error_clone.lock().unwrap() = Some(error_message);
    }));

    // Push error scope for additional error catching
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Poll the device to process any pending operations
    let _ = device.poll(wgpu::MaintainBase::Wait);

    // Check for scoped errors
    let scoped_error = device.pop_error_scope().await;
    
    // Check for uncaptured errors
    if let Some(error_msg) = error_occurred.lock().unwrap().take() {
        return Err(JsValue::from_str(&format!("Uncaptured error: {}", error_msg)));
    }

    // Check for scoped errors
    if let Some(error) = scoped_error {
        return Err(JsValue::from_str(&format!("Scoped error: {:?}", error)));
    }

    // Reset error callback
    device.on_uncaptured_error(Box::new(|_| {}));

    Ok(shader_module)
}

pub async fn test_minimal_shader_compile() -> Result<JsValue, JsValue> {
    // Ensure Rust panics are logged to the browser console
    console_error_panic_hook::set_once();
    // Very simple compute shader
    let shader_src = r#"
@compute @workgroup_size(1)
fn main() {
    // Minimal compute shader that does nothing
}
"#;

    // Use browser performance timer for WASM compatibility
    let perf = web_sys::window().unwrap().performance().unwrap();
    let start = perf.now();
    // Log: starting device creation
    web_sys::console::log_1(&"[test_minimal_shader_compile] Creating wgpu device...".into());
    let (device, _queue, adapter) = match create_wgpu_device().await {
        Ok(res) => res,
        Err(e) => {
            web_sys::console::error_1(&format!("[test_minimal_shader_compile] Device creation failed: {:?}", e).into());
            return Err(e);
        }
    };
    // Log: device created
    web_sys::console::log_1(&"[test_minimal_shader_compile] Device created.".into());
    let info = adapter.get_info();
    let device_info = format!(
        "GPU: {} | Backend: {:?} | Type: {:?}",
        info.name, info.backend, info.device_type
    );
    // Log: compiling shader
    web_sys::console::log_1(&"[test_minimal_shader_compile] Compiling shader...".into());
    match compile_shader_safe(&device, shader_src, "minimal_test").await {
        Ok(_shader_module) => {
            let elapsed = perf.now() - start;
            web_sys::console::log_1(&"[test_minimal_shader_compile] Shader compiled successfully.".into());
            Ok(JsValue::from_str(&format!(
                "[test_minimal_shader_compile] SUCCESS\n{}\nTime: {:.2} ms\nShader source:\n{}",
                device_info, elapsed, shader_src
            )))
        }
        Err(e) => {
            let elapsed = perf.now() - start;
            web_sys::console::error_1(&format!("[test_minimal_shader_compile] Shader compile failed: {:?}", e).into());
            Err(JsValue::from_str(&format!(
                "[test_minimal_shader_compile] FAILED\n{}\nTime: {:.2} ms\nError: {}\nShader source:\n{}",
                device_info, elapsed, e.as_string().unwrap_or_else(|| "Unknown error".to_string()), shader_src
            )))
        }
    }
}

pub async fn test_inline_add_scalar_shader_compile() -> Result<JsValue, JsValue> {
    // Use browser performance timer for WASM compatibility
    let perf = web_sys::window().unwrap().performance().unwrap();
    let start = perf.now();
    let shader_src = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scalar: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input[index] + scalar[0];
}
"#;

    let (device, _queue, adapter) = create_wgpu_device().await?;
    let info = adapter.get_info();
    let device_info = format!(
        "GPU: {} | Backend: {:?} | Type: {:?}",
        info.name, info.backend, info.device_type
    );

    match compile_shader_safe(&device, shader_src, "add_scalar_test").await {
        Ok(_shader_module) => {
            let elapsed = perf.now() - start;
            Ok(JsValue::from_str(&format!(
                "[test_inline_add_scalar_shader_compile] SUCCESS\n{}\nTime: {:.2} ms\nShader compiled successfully!",
                device_info, elapsed
            )))
        }
        Err(e) => {
            let elapsed = perf.now() - start;
            Err(JsValue::from_str(&format!(
                "[test_inline_add_scalar_shader_compile] FAILED\n{}\nTime: {:.2} ms\nError: {}\nShader source (first 200 chars):\n{}",
                device_info, elapsed, 
                e.as_string().unwrap_or_else(|| "Unknown error".to_string()),
                &shader_src[..shader_src.len().min(200)]
            )))
        }
    }
}

pub async fn try_compile_kernel(kernel_name: &str) -> Result<JsValue, JsValue> {
    // Use browser performance timer for WASM compatibility
    let perf = web_sys::window().unwrap().performance().unwrap();
    let start = perf.now();
    let kernel = kernel_name.parse::<WasmKernel>().unwrap_or(WasmKernel::Default);
    let kernel_src = kernel.source();

    let (device, _queue, adapter) = create_wgpu_device().await?;
    let info = adapter.get_info();
    let device_info = format!(
        "GPU: {} | Backend: {:?} | Type: {:?}",
        info.name, info.backend, info.device_type
    );

    match compile_shader_safe(&device, kernel_src, kernel_name).await {
        Ok(_shader_module) => {
            let elapsed = perf.now() - start;
            Ok(JsValue::from_str(&format!(
                "[try_compile_kernel] Kernel: {} SUCCESS\n{}\nTime: {:.2} ms\nWGSL source (first 100 chars):\n{}",
                kernel_name, device_info, elapsed,
                &kernel_src[..kernel_src.len().min(100)]
            )))
        }
        Err(e) => {
            let elapsed = perf.now() - start;
            Err(JsValue::from_str(&format!(
                "[try_compile_kernel] Kernel: {} FAILED\n{}\nTime: {:.2} ms\nError: {}\nWGSL source (first 100 chars):\n{}",
                kernel_name, device_info, elapsed,
                e.as_string().unwrap_or_else(|| "Unknown error".to_string()),
                &kernel_src[..kernel_src.len().min(100)]
            )))
        }
    }
}

// Public wrappers for macro-generated test runners
pub async fn run_test_kernel_compile_optimal() -> Result<JsValue, JsValue> {
    try_compile_kernel(&WasmKernel::Optimal.to_string()).await
}

pub async fn run_test_kernel_compile_wasm() -> Result<JsValue, JsValue> {
    try_compile_kernel(&WasmKernel::Wasm.to_string()).await
}

pub async fn run_test_kernel_compile_default() -> Result<JsValue, JsValue> {
    try_compile_kernel(&WasmKernel::Default.to_string()).await
}

pub async fn run_test_inline_add_scalar_shader_compile() -> Result<JsValue, JsValue> {
    test_inline_add_scalar_shader_compile().await
}

pub async fn run_test_minimal_shader_compile() -> Result<JsValue, JsValue> {
    test_minimal_shader_compile().await
}

// --- Test Ping: GPU Info ---
pub async fn test_ping_gpu_info() -> Result<JsValue, JsValue> {
    let (device, _queue, adapter) = create_wgpu_device().await?;
    
    let info = adapter.get_info();
    let limits = device.limits();
    
    Ok(JsValue::from_str(&format!(
        "=== GPU Information ===\nName: {}\nBackend: {:?}\nDevice Type: {:?}\nDriver: {}\nDriver Info: {}\nDevice ID: 0x{:04X}\nVendor ID: 0x{:04X}\n\n=== Device Limits ===\nMax Compute Workgroups X: {}\nMax Compute Workgroups Y: {}\nMax Compute Workgroups Z: {}\nMax Compute Workgroup Size X: {}\nMax Compute Workgroup Size Y: {}\nMax Compute Workgroup Size Z: {}\nMax Buffer Size: {}",
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        info.device,
        info.vendor,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroups_per_dimension,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z,
        limits.max_buffer_size,
    )))
}

// --- Scalar reference logic for WASM tests (single-threaded, WASM-safe) ---
fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations
        .iter()
        .filter(|v| v.is_finite())
        .map(|&x| x.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    let abs_max = if abs_max == f32::NEG_INFINITY { 0.0 } else { abs_max };
    let scale = abs_max / 127.0 + 1e-6;
    (
        activations
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect(),
        scale,
    )
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
                    let packed_bits = (packed_weight >> (bit_idx * 2)) & 0b11;
                    let weight_val = match packed_bits {
                        1 => 1i8,   // 01
                        2 => -1i8,  // 10
                        _ => 0i8,   // 00 or 11
                    };
                    sum += (q_activations[act_base + bit_idx] as i32) * (weight_val as i32);
                }
            }
            output[batch_idx * out_features + out_idx] = (sum as f32) * activation_scale * weight_scale;
        }
    }
    output
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

// === WASM Kernel Test Suite: Cold Tests ===

pub async fn unit_test_pack_ternary_weights() -> Result<JsValue, JsValue> {
    use bitnet_core::kernels::pack_ternary_weights;
    let weights = vec![vec![-1i8, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]];
    let (packed, _scales) = pack_ternary_weights(&weights)
        .map_err(|e| JsValue::from_str(&format!("Packing error: {:?}", e)))?;
    let expected = 0b00011000011000011000011000011000u32;
    if packed[0] != expected {
        let msg = format!(
            "Packing logic is incorrect: got {:032b}, expected {:032b}",
            packed[0], expected
        );
        web_sys::console::error_1(&msg.clone().into());
        return Err(JsValue::from_str(&msg));
    }
    web_sys::console::log_1(&"unit_test_pack_ternary_weights passed".into());
    Ok(JsValue::from_str("unit_test_pack_ternary_weights passed"))
}

pub async fn unit_test_calculate_weight_scales() -> Result<JsValue, JsValue> {
    use bitnet_core::kernels::calculate_weight_scales;
    let weights = vec![
        vec![-1i8, 0, 1],     // Average magnitude = 1.0 (sum=2, count=2)
        vec![0, 0, 0],        // All zeros -> scale = 1.0
        vec![1, 1, -1],       // Average magnitude = 1.0 (sum=3, count=3)
        vec![-1, -1, 0],      // Average magnitude = 1.0 (sum=2, count=2)
    ];
    let scales = calculate_weight_scales(&weights);
    let expected = vec![1.0, 1.0, 1.0, 1.0];
    for (i, (s, e)) in scales.iter().zip(expected.iter()).enumerate() {
        if (s - e).abs() > 1e-6 {
            let msg = format!("Scale mismatch at {}: got {}, expected {}", i, s, e);
            web_sys::console::error_1(&msg.clone().into());
            return Err(JsValue::from_str(&msg));
        }
    }
    web_sys::console::log_1(&"unit_test_calculate_weight_scales passed".into());
    Ok(JsValue::from_str("unit_test_calculate_weight_scales passed"))
}

pub async fn test_scalar_packing_decoding_symmetry() -> Result<JsValue, JsValue> {
    // 1. Define original weights
    let original_weights: Vec<i8> = vec![-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1];
    let weights_2d = vec![original_weights.clone()];
    use bitnet_core::kernels::pack_ternary_weights;
    let (packed, _scales) = pack_ternary_weights(&weights_2d)
        .map_err(|e| JsValue::from_str(&format!("Packing error: {:?}", e)))?;
    let packed_val = packed[0];
    // 2. Decode the packed value using the correct logic (MSB to LSB)
    let mut decoded_weights = Vec::with_capacity(16);
    for i in 0..16 {
        let bit_idx = 30 - (i * 2);
        let bits = (packed_val >> bit_idx) & 0b11;
        let weight = match bits {
            0b00 => -1i8,
            0b01 => 0i8,
            0b10 => 1i8,
            _ => return Err(JsValue::from_str("Invalid 2-bit value in decode")),
        };
        decoded_weights.push(weight);
    }
    if original_weights != decoded_weights {
        let msg = format!(
            "Packing/decoding not symmetrical!\nOriginal: {:?}\nDecoded:  {:?}",
            original_weights, decoded_weights
        );
        web_sys::console::error_1(&msg.clone().into());
        return Err(JsValue::from_str(&msg));
    }
    web_sys::console::log_1(&"test_scalar_packing_decoding_symmetry passed".into());
    Ok(JsValue::from_str("test_scalar_packing_decoding_symmetry passed"))
}

pub async fn test_matmul_quantized_scalar() -> Result<JsValue, JsValue> {
    // Test case with proper BitNet dimensions
    let batch_size = 1;
    let in_features = 16;
    let out_features = 2;
    // Simple test inputs
    let q_activations = vec![1i8, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2, 1, -1, 0, 2];
    let packed_weights = vec![
        0b01001001010010010100100101001001,
        0b01001001010010010100100101001001,
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
        out_features,
    );
    let expected_output = vec![8.0, 8.0];
    for (i, (o, e)) in output.iter().zip(expected_output.iter()).enumerate() {
        if (o - e).abs() > 1e-5 {
            let msg = format!("Output mismatch at {}: got {}, expected {}", i, o, e);
            web_sys::console::error_1(&msg.clone().into());
            return Err(JsValue::from_str(&msg));
        }
    }
    web_sys::console::log_1(&"test_matmul_quantized_scalar passed".into());
    Ok(JsValue::from_str("test_matmul_quantized_scalar passed"))
}