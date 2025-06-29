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