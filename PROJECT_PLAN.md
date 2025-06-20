# Project Plan: bitnet-rs

## How to Contribute / Onboard

Welcome to bitnet-rs! This document is your single source of truth for understanding, contributing to, and extending the project. **Before contributing code, please:**

- Read the full project plan below.
- Pay special attention to the file-by-file breakdown, validation strategies, and the Critique & Resolution section.
- For complex files (SIMD, GPU kernel, tokenizer), review the TODO/Validation checklists and ensure all tests pass before submitting a PR.
- If you are unsure about any part of the architecture or validation, ask in the project chat or open an issue.

---

## Full Project & Module Structure

Below is the up-to-date structure of the BitNet-rs repository. Each file and directory is annotated for purpose and onboarding clarity.

```text
bitnet-rs/
├── Cargo.toml                 # Workspace definition for all crates
├── .cargo/
│   └── config.toml            # Cargo alias for file combiner
├── .workspace_root            # Marker for workspace root
├── README.md                  # Project overview, build, usage, and structure
├── CHECKLIST.md               # Implementation status and TODOs
├── PROJECT_PLAN.md            # Detailed project plan and architecture
├── src/
│   └── main.rs                # Workspace-level entry point/test harness
├── custom-kernel-test/
│   ├── README.md              # Standalone kernel validation/prototyping
│   ├── Cargo.toml             # Minimal test workspace
│   ├── src/                   # Kernel test code
│   └── tests/                 # Kernel test cases
├── crates/
│   ├── bitnet-core/
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── model.rs
│   │   │   ├── attention.rs
│   │   │   ├── feed_forward.rs
│   │   │   ├── rms_norm.rs
│   │   │   ├── bitnet_linear.rs
│   │   │   ├── tokenizer.rs
│   │   │   ├── settings.rs
│   │   │   ├── embedding.rs
│   │   │   ├── training.rs
│   │   │   ├── visualization.rs
│   │   │   ├── kernels.rs
│   │   │   └── wgpu_context.rs
│   │   │       ├── bitnet_kernel.wgsl
│   │   │       └── README.md
│   │   ├── tests/
│   │   │   ├── pipeline_integration.rs
│   │   │   ├── pipeline_validation.rs
│   │   │   └── kernel_tests.rs
│   │   └── gui/
│   │       ├── mod.rs
│   │       ├── dashboard.rs
│   │       ├── weights_viewer.rs
│   │       ├── kernel_profiler.rs
│   │       ├── attention_map.rs
│   │       └── README.md
│   ├── bitnet-converter/
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── packer.rs
│   │   │   ├── lib.rs
│   │   │   └── source.rs
│   │   ├── tests/
│   │   │   ├── serialization_test.rs
│   │   │   └── official_parity.rs
│   ├── bitnet-app/
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   └── src/
│   │       ├── main.rs
│   │       ├── cli.rs
│   │       ├── generation.rs
│   │       ├── sampler.rs
│   │       └── gui/
│   │           ├── mod.rs
│   │           ├── app.rs
│   │           ├── state.rs
│   │           ├── backend.rs
│   │           └── README.md
│   └── bitnet-tools/
│       ├── Cargo.toml
│       ├── README.md
│       ├── src/
│       │   ├── lib.rs
│       │   ├── constants.rs
│       │   ├── error.rs
│       │   ├── combine.rs
│       │   ├── hf_loader.rs
│       │   └── test_utils.rs
│       └── gui_combiner/
│           ├── Cargo.toml
│           └── src/
│               └── main.rs
├── models/
│   ├── Original/              # Downloaded Hugging Face models
│   └── Converted/             # BitNet-optimized, quantized models
├── logs/                      # Conversion and run logs
├── References/                # Official reference code, assets, and docs
│   └── official/
│       ├── gpu/
│       ├── utils/
│       ├── src/
│       ├── preset_kernels/
│       ├── media/
│       ├── include/
│       ├── assets/
│       ├── docs/
│       ├── 3rdparty/
│       └── README.md
├── found_files.txt            # Utility/output files
├── safetensor_keys.txt        # Utility/output files
└── ...                        # (other utility or temporary files)
```

<!-- (The rest of the file-by-file breakdown and component descriptions should be updated to match this structure. Adjust comments and notes for accuracy. Remove references to non-existent directories like shaders/ if not present, and add new ones as needed.) -->

## File-by-File Purpose & Challenge Table

| File/Module | Purpose | Contents/Notes | Special Challenges/Validation |
|-------------|---------|---------------|-------------------------------|
| **Cargo.toml (root)** | Workspace definition | Lists all member crates | - |
| **bitnet-core/src/kernels/bitnet_kernel.wgsl** | GPU compute shader | Pure WGSL, replicates CUDA logic, includes decode, dp4a emulation, reductions | **Very difficult**: Porting CUDA to WGSL, must validate correctness and tune for performance. |
| **bitnet-core/src/kernels/README.md** | Shader/kernels documentation | Data layout, kernel design, usage notes | - |
| **bitnet-core/Cargo.toml** | Crate identity, features | Features: gpu flag; Deps: wgpu, half, thiserror, tokenizers, safetensors, hf-hub; GPU-only: wgpu | Feature gating must be correct for portability |
| **bitnet-core/src/lib.rs** | Library root | Exports main modules | - |
| **bitnet-core/src/model.rs** | Transformer architecture | Transformer, TransformerBlock, Attention, FeedForward structs; uses BitLinear op | Must match Python model.py logic exactly |
| **bitnet-core/src/attention.rs** | Attention block | Implements multi-head attention, RoPE, uses BitLinear | - |
| **bitnet-core/src/feed_forward.rs** | Feed-Forward block | Implements SwiGLU, uses BitLinear | - |
| **bitnet-core/src/rms_norm.rs** | RMSNorm | Wrapper for RMSNorm logic for API consistency and to potentially add tracing or other custom logic later. | - |
| **bitnet-core/src/bitnet_linear.rs** | BitLinear CustomOp | BitLinear struct, CustomOp impl, backend dispatch | Central hub for CPU/GPU, must be correct and fast |
| **bitnet-core/src/tokenizer.rs** | Text tokenizer | Tokenizer struct wrapping tokenizers crate, ChatFormat logic | Must match Python tokenizer, handle chat templates |
| **bitnet-core/src/kernels/mod.rs** | Kernel aggregator | Declares cpu, wgpu modules | - |
| **bitnet-core/src/kernels/cpu.rs** | CPU backend | Forward function, runtime SIMD detection, scalar fallback | **Fragile**: SIMD dispatch, must validate against scalar |
| **bitnet-core/src/kernels/cpu_x86.rs** | AVX2 SIMD kernel | Rust AVX2 intrinsics, lut_ctor, tbl_impl_* | **Very fragile**: Unsafe, must match scalar exactly |
| **bitnet-core/src/kernels/cpu_arm.rs** | NEON SIMD kernel | Rust NEON intrinsics, lut_ctor, tbl_impl_* | **Very fragile**: Unsafe, must match scalar exactly |
| **bitnet-core/src/kernels/wgpu.rs** | GPU backend | Manages wgpu, loads shader, dispatches compute | **Very difficult**: Performance tuning, correctness |
| **bitnet-core/tests/pipeline_validation.rs** | End-to-end tests | Placeholder for golden file test, prompt-to-token match | Must match Python output exactly. Currently ignored. |
| **bitnet-core/tests/pipeline_integration.rs**| Integration test | Tests full model pipeline ensuring all components work together. | Uses the project-wide `TestReporter`. |
| **bitnet-core/tests/kernel_tests.rs** | Kernel validation | Comprehensive, low-level validation of the wgpu kernel against a scalar CPU implementation. | **Non-negotiable**: Must pass for all kernels. Includes extensive tests for correctness, dimensions, and edge cases. Uses the project-wide `TestReporter` for detailed markdown reports. |
| **bitnet-converter/src/main.rs** | CLI entry | Uses clap, calls packer | - |
| **bitnet-converter/src/packer.rs** | Weight conversion | quantize -> permutate -> pack -> interleave | Must match Python scripts exactly |
| **bitnet-app/src/main.rs** | User app entry | Loads model/tokenizer, runs generation loop | - |
| **bitnet-app/src/generation.rs** | Generation engine | Generator struct, manages KV cache | - |
| **bitnet-app/src/sampler.rs** | Logits processor | LogitsProcessor struct, sampling logic | - |
| **bitnet-app/src/gui/backend.rs** | GUI backend | Threaded model execution, mpsc channels | - |
| **bitnet-tools/src/test_utils.rs** | Test Reporting Utility | Provides a robust, thread-safe test reporting utility (`TestReporter`) that generates detailed markdown logs, handling parallel tests gracefully. | - |

---

## Special Callouts: GPU Kernel & SIMD Code

- **GPU Kernel (kernels/wgpu.rs, crates/bitnet-core/src/kernels/bitnet_kernel.wgsl):**
  - Porting CUDA to WGSL is extremely challenging. WGSL lacks some low-level features of CUDA, so emulation (e.g., dp4a) and careful memory layout are required.
  - Performance tuning is iterative: start with correctness, then use wgpu timestamp queries, experiment with workgroup sizes, and optimize shared memory usage.
  - **Validation:** Must compare output to both scalar CPU and official CUDA outputs. All changes must be tested for both correctness and speed.

- **SIMD Kernels (cpu_x86.rs, cpu_arm.rs):**
  - Translating C++ intrinsics to Rust is error-prone. Unsafe code can cause silent data corruption.
  - **Validation:** Every SIMD function must have a scalar equivalent. Tests must assert bit-for-bit identical output for random data.

---

## Critique & Resolution

- **CPU SIMD code is complex and fragile.**
  - *Resolution:* Rigorous, mandatory validation. Every SIMD function is tested against a scalar version. No PR is merged unless all tests pass. A comprehensive test suite (`kernel_tests.rs`) and a robust reporting utility (`TestReporter`) have been developed to enforce this.
- **GPU implementation may not be fast initially.**
  - *Resolution:* Plan for iterative tuning. Use wgpu profiling, experiment with workgroup sizes, memory layout, and compare different kernel strategies.
- **Dependency on GGML logic is a risk.**
  - *Resolution:* Treat GGML as a spec, not gospel. Validate against our own scalar implementation first, then against official outputs.
- **Numeric precision differences could cause divergence.**
  - *Resolution:* End-to-end golden testing. CI must run golden prompt tests for both CPU and GPU, asserting output token IDs match the reference. The `pipeline_validation.rs` test is the placeholder for this.

---

## Training

- **Purpose:** Enable model training, fine-tuning, and optimizer/scheduler logic in Rust.
- **Stub files/modules:**
  - `crates/bitnet-core/src/training.rs` — Training loop, optimizer, scheduler, checkpointing (stub)
  - (Optionally, add `optimizer.rs`, `scheduler.rs`, or a `training/` subdirectory for modularity)

## Advanced Debugging & Visualization

- **Purpose:** Provide hooks and APIs for real-time and post-hoc inspection of model internals during training and inference.
- **Stub files/modules:**
  - `crates/bitnet-core/src/visualization.rs` — Logging, metrics, and visualization hooks (stub)
  - (Optionally, add `metrics.rs`, `dashboard.rs`, or a `visualization/` subdirectory for extensibility)
- **Implementation notes:**
  - All visualization/debug code will be gated behind a `visualization` feature flag.
  - APIs will be provided for external tools or dashboards to access metrics and logs.
  - Documentation/examples will be provided for extending or integrating with external tools.

## Not in Scope / Explicitly Rejected Approaches

- No one-size-fits-all kernels: CPU and GPU are specialized.
- No Candle, CUDA, or Metal dependencies: all GPU compute is via wgpu.
- No panics or unwraps in production code.
- No println! debugging in production; use structured logging/tracing only.

---

## Iterative Tuning & Validation for Kernels

- **GPU/CPU kernel development is staged:**
  1. **Correctness:** Implement a simple, correct version. Validate against scalar and reference outputs.
  2. **Performance:** Profile and tune (workgroup size, memory layout, SIMD width).
  3. **Validation:** All changes must pass kernel and golden tests in CI.

---

## Deep Dive: Critical Components & Data Flow

### Deep Dive 1: bitnet_linear.rs — The BitLinear CustomOp & Kernel Signatures

**File:** `crates/bitnet-core/src/bitnet_linear.rs`

This is the most important interface in the project. Its implementation must be precise.

**Why:** This is the most important interface in the project. It abstracts backend-specific logic and ensures the model code is backend-agnostic.

**Struct Definition:**

```rust
// In crates/bitnet-core/src/bitnet_linear.rs
pub struct BitLinear {
    // Stored in the specific format our kernels expect after conversion.
    // Shape: [out_features, in_features / 4]
    packed_weights: Tensor,

    // Weight scaling factor, one per output channel.
    // Shape: [out_features]
    weight_scales: Tensor,

    // Model dimensions, needed for kernel dispatch.
    in_features: usize,
    out_features: usize,
    
    // **CPU-ONLY**: Pre-computed Look-Up Table. Generated on model load.
    #[cfg(not(feature = "gpu"))]
    precomputed_lut: Tensor,
}

// Implementation of the core forward pass
impl CustomOp for BitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Online Activation Quantization
        let (x_quant, x_scales) = self.quantize_activations(x)?;

        // 2. Backend Dispatch (Compile-Time)
        #[cfg(feature = "gpu")]
        {
            // GPU path: call the wgpu kernel executor
            crate::kernels::wgpu::execute(
                &x_quant,           // i8 activations
                &x_scales,          // f32 scales
                &self.packed_weights, // i8 packed weights
                &self.weight_scales,  // f32 weight scales
                self.out_features,
            )
        }
        #[cfg(not(feature = "gpu"))]
        {
            // CPU path: call the SIMD dispatcher
            crate::kernels::cpu::execute(
                &x_quant,
                &x_scales,
                &self.precomputed_lut,
                &self.weight_scales,
            )
        }
    }
}
```

### Deep Dive 2: kernels/cpu_*.rs — The CPU LUT Kernel Logic

**Files:**

- `crates/bitnet-core/src/kernels/cpu_x86.rs`
- `crates/bitnet-core/src/kernels/cpu_arm.rs`

**Core SIMD Implementation (x86):**

```rust
// In crates/bitnet-core/src/kernels/cpu_x86.rs
#[cfg(target_arch = "x86_64")]
pub unsafe fn qgemm_lut(
    activations: &[i8],
    lut: &[i8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    
    let mut output = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum_vec = _mm256_setzero_si256();
            
            // Process 32 elements at a time using AVX2
            for k_idx in (0..k).step_by(32) {
                // Load 32 activations
                let act = _mm256_loadu_si256(
                    activations.as_ptr().add(i * k + k_idx) as *const __m256i
                );
                
                // Use activations as indices into LUT
                let lut_result = _mm256_shuffle_epi8(
                    _mm256_loadu_si256(
                        lut.as_ptr().add(j * k + k_idx) as *const __m256i
                    ),
                    act
                );
                
                // Accumulate results
                sum_vec = _mm256_add_epi32(sum_vec, lut_result);
            }
            
            // Horizontal sum and scaling
            let sum = _mm256_extract_epi32(sum_vec, 0) +
                     _mm256_extract_epi32(sum_vec, 1) +
                     _mm256_extract_epi32(sum_vec, 2) +
                     _mm256_extract_epi32(sum_vec, 3) +
                     _mm256_extract_epi32(sum_vec, 4) +
                     _mm256_extract_epi32(sum_vec, 5) +
                     _mm256_extract_epi32(sum_vec, 6) +
                     _mm256_extract_epi32(sum_vec, 7);
                     
            output[i * n + j] = sum as f32 * scales[j];
        }
    }
    
    output
}
```

### Deep Dive 3: bitnet-core/src/kernels/bitnet_kernel.wgsl — The GPU Kernel Data Flow

**File:** `crates/bitnet-core/src/kernels/bitnet_kernel.wgsl`

```wgsl
// bitnet_kernel.wgsl
// Optimized BitNet B1.58 Ternary Kernel for WGPU 
// Supports {-1, 0, +1} ternary weights with efficient packing and vectorization

struct BitnetMetadata {
    M: u32,           // Batch size
    N: u32,           // Output features  
    K: u32,           // Input features
    K_packed: u32,    // K / 16 (since we pack 16 weights per u32)
};

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>; // Per-batch activation scales
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Optimized tiling parameters for modern GPUs
const TILE_DIM_M: u32 = 64u;   // Reduced for better occupancy
const TILE_DIM_N: u32 = 64u;   
const TILE_DIM_K: u32 = 32u;   // Increased K tile for better data reuse

const THREAD_TILE_M: u32 = 4u; // Smaller thread tiles for better vectorization
const THREAD_TILE_N: u32 = 4u;

const WORKGROUP_SIZE_X: u32 = 16u; // TILE_DIM_N / THREAD_TILE_N
const WORKGROUP_SIZE_Y: u32 = 16u; // TILE_DIM_M / THREAD_TILE_M

// --- Explicit array sizes for WGSL compliance ---
const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u; // for vec4<i32>
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;         // for i32

// Shared memory with better alignment
var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

// Remove LUT and use direct decode function for ternary weights
fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 0u: { return -1; }
        case 1u: { return 0; }
        case 2u: { return 1; }
        default: { return 0; } // 0b11 is unused, map to 0
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let bits = (packed_val >> (i * 2u)) & 0x3u;
        decoded[i] = decode_2bit(bits);
    }
    return decoded;
}

// Vectorized dot product for better throughput
fn dot_product_4x4(a: vec4<i32>, b: vec4<i32>) -> i32 {
    return dot(a, b);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // Vectorized accumulators for better performance  
    var accumulators: array<vec4<i32>, THREAD_TILE_M>;
    for (var i = 0u; i < THREAD_TILE_M; i = i + 1u) {
        accumulators[i] = vec4<i32>(0);
    }
    
    // Main tiling loop with optimizations
    let num_k_tiles = (metadata.K + TILE_DIM_K - 1u) / TILE_DIM_K;
    
    for (var k_tile_idx = 0u; k_tile_idx < num_k_tiles; k_tile_idx = k_tile_idx + 1u) {
        let k_tile_start = k_tile_idx * TILE_DIM_K;
        
        // === Cooperative Loading with Coalescing ===
        // Load activations with vectorization
        let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
        let loads_per_thread_a = (total_a_elements + 255u) / 256u; // Ceiling division
        
        for (var i = 0u; i < loads_per_thread_a; i = i + 1u) {
            let load_idx = i * 256u + local_index;
            if (load_idx < total_a_elements) {
                let vec_idx = load_idx;
                let flat_idx = load_idx * 4u;
                let m = flat_idx / TILE_DIM_K;
                let k = flat_idx % TILE_DIM_K;
                
                let global_m = tile_start_m + m;
                let global_k = k_tile_start + k;
                
                if (global_m < metadata.M && global_k + 3u < metadata.K) {
                    // Load 4 activations at once
                    let base_addr = global_m * metadata.K + global_k;
                    tile_a[vec_idx] = vec4<i32>(
                        activations[base_addr],
                        activations[base_addr + 1u],
                        activations[base_addr + 2u], 
                        activations[base_addr + 3u]
                    );
                } else {
                    tile_a[vec_idx] = vec4<i32>(0);
                }
            }
        }
        
        // Load and decode weights
        let total_b_elements = TILE_DIM_N * TILE_DIM_K;
        let loads_per_thread_b = (total_b_elements + 255u) / 256u;
        
        for (var i = 0u; i < loads_per_thread_b; i = i + 1u) {
            let load_idx = i * 256u + local_index;
            if (load_idx < total_b_elements && (load_idx % 16u) == 0u) {
                let n = load_idx / TILE_DIM_K;
                let k = load_idx % TILE_DIM_K;
                
                let global_n = tile_start_n + n;  
                let global_k_packed_idx = (k_tile_start + k) / 16u;
                
                if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
                    let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
                    let packed_w = packed_weights[weight_idx];
                    let decoded = decode_16x2bit_ternary(packed_w);
                    // Store decoded weights (unrolled for WGSL compliance)
                    tile_b[n * TILE_DIM_K + k + 0u] = decoded[0u];
                    tile_b[n * TILE_DIM_K + k + 1u] = decoded[1u];
                    tile_b[n * TILE_DIM_K + k + 2u] = decoded[2u];
                    tile_b[n * TILE_DIM_K + k + 3u] = decoded[3u];
                    tile_b[n * TILE_DIM_K + k + 4u] = decoded[4u];
                    tile_b[n * TILE_DIM_K + k + 5u] = decoded[5u];
                    tile_b[n * TILE_DIM_K + k + 6u] = decoded[6u];
                    tile_b[n * TILE_DIM_K + k + 7u] = decoded[7u];
                    tile_b[n * TILE_DIM_K + k + 8u] = decoded[8u];
                    tile_b[n * TILE_DIM_K + k + 9u] = decoded[9u];
                    tile_b[n * TILE_DIM_K + k + 10u] = decoded[10u];
                    tile_b[n * TILE_DIM_K + k + 11u] = decoded[11u];
                    tile_b[n * TILE_DIM_K + k + 12u] = decoded[12u];
                    tile_b[n * TILE_DIM_K + k + 13u] = decoded[13u];
                    tile_b[n * TILE_DIM_K + k + 14u] = decoded[14u];
                    tile_b[n * TILE_DIM_K + k + 15u] = decoded[15u];
                } else {
                    // Pad with zeros
                    for (var j = 0u; j < 16u; j = j + 1u) {
                        tile_b[n * TILE_DIM_K + k + j] = 0;
                    }
                }
            }
        }
        
        workgroupBarrier();
        
        // === Vectorized Computation ===
        for (var k_inner = 0u; k_inner < TILE_DIM_K; k_inner = k_inner + 4u) {
            // Load vectorized activations
            var a_vecs: array<vec4<i32>, THREAD_TILE_M>;
            for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                let base_m = thread_idx_m * THREAD_TILE_M + m;
                let vec_idx = (base_m * TILE_DIM_K + k_inner) / 4u;
                let a_i32 = tile_a[vec_idx];
                a_vecs[m] = a_i32;
            }
            
            // Load vectorized weights and compute
            for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
                let base_n = thread_idx_n * THREAD_TILE_N + n;
                let b_vec = vec4<i32>(
                    tile_b[base_n * TILE_DIM_K + k_inner],
                    tile_b[base_n * TILE_DIM_K + k_inner + 1u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 2u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 3u]
                );
                
                // Vectorized multiply-accumulate
                for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                    let dot_result = dot_product_4x4(a_vecs[m], b_vec);
                    accumulators[m][n] += dot_result;
                }
            }
        }
        
        workgroupBarrier();
    }
    
    // === Write Results with Proper Scaling ===
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M + m;
            let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N + n;
            
            if (global_m < metadata.M && global_n < metadata.N) {
                // BitNet B1.58 scaling: result = activation_scale * weight_scale * dot_product
                let activation_scale = activation_scales[global_m];
                let weight_scale = weight_scales[global_n];
                let final_result = f32(accumulators[m][n]) * activation_scale * weight_scale;
                
                output[global_m * metadata.N + global_n] = final_result;
            }
        }
    }
} 
```

### Deep Dive 4: bitnet-converter/packer.rs — The Weight Conversion Pipeline

**File:** `crates/bitnet-converter/src/packer.rs`

```rust
// In crates/bitnet-converter/src/packer.rs

/// Converts a tensor of f32 weights into our packed ternary format
pub fn convert_weights_to_ternary(
    weights: &[f32],
    shape: &[usize],
) -> Result<(Vec<i8>, Vec<f32>), ConversionError> {
    // 1. Quantize to {-1, 0, 1}
    let (quantized, scales) = quantize_to_ternary(weights, shape)?;
    
    // 2. Permute for memory access patterns
    let permuted = permute_for_kernel_access(&quantized, shape)?;
    
    // 3. Pack 4 ternary values into each i8
    let packed = pack_ternary_values(&permuted)?;
    
    // 4. Final interleaving for kernel efficiency
    let interleaved = interleave_for_kernel(&packed)?;
    
    Ok((interleaved, scales))
}

/// Step 1: Quantize f32 weights to {-1, 0, 1} with scaling
fn quantize_to_ternary(
    weights: &[f32],
    shape: &[usize],
) -> Result<(Vec<i8>, Vec<f32>), ConversionError> {
    let mut quantized = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(shape[0]);  // One scale per output feature
    
    for row in weights.chunks(shape[1]) {
        // Calculate scale for this row
        let scale = row.iter()
            .map(|x| x.abs())
            .sum::<f32>() / row.len() as f32;
            
        scales.push(scale);
        
        // Quantize using calculated scale
        for &w in row {
            let scaled = w / scale;
            let q = match scaled {
                x if x < -0.5 => -1i8,
                x if x > 0.5 => 1i8,
                _ => 0i8,
            };
            quantized.push(q);
        }
    }
    
    Ok((quantized, scales))
}

// ... rest of the implementation ...
```

## Component Descriptions

### Crate: bitnet-core — The Engine

**Core Purpose:** This library is the heart of the project. It contains all the performance-critical logic, model definitions, and backend implementations. It is designed to be a dependency for other applications, providing the building blocks for BitNet inference without being tied to any specific UI or application logic.

- **File: crates/bitnet-core/Cargo.toml**
  - **Core Purpose:** Defines the crate's identity, dependencies, and the crucial features section that controls the backend compilation.
  - **Detailed Breakdown & Logic Flow:**
    - `[package]`: Defines the library name (`bitnet-core`), version, and authors.
    - `[dependencies]`: Includes non-optional dependencies required by all backends:
      - `wgpu`: Cross-platform GPU compute backend for BitNet kernels.
      - `tokenizers`: Hugging Face tokenizer support.
      - `safetensors`: Efficient tensor/model storage.
      - `hf-hub`: Model download and management from Hugging Face.
      - `log`, `tracing`: For structured logging and performance tracing hooks.
    - `[features]`: The control center for our dual-backend strategy.
      - `default = []` (CPU-only SIMD acceleration by default)
      - `gpu = ["dep:wgpu"]` (enables the wgpu backend)
    - **Rationale:** This feature-gating mechanism is the core of our "write once, compile for many targets" strategy. It ensures that a user who only wants the CPU version does not need to download or compile any GPU-related libraries.

- **Files: Model Architecture (model.rs, attention.rs, feed_forward.rs, rms_norm.rs)**
  - **Core Purpose:** To define the logical architecture of the BitNet Transformer model using custom Rust and wgpu-based building blocks.
  - **Informed By:** The architectural patterns (RoPE, SwiGLU, pre-normalization) are adopted from our analysis of the bitnet.rs project and are common in modern Llama-like models. We explicitly reject its Straight-Through-Estimator (STE) implementation in favor of a true inference approach.
  - **Detailed Breakdown & Logic Flow:**
    - `model.rs`: Defines the top-level struct `Transformer`. Contains an Embedding layer, a `Vec<TransformerBlock>`, a final `rms_norm::RmsNorm`, and a standard `candle_nn::Linear` layer for the final logits head (as this layer is typically not quantized). Its `forward` method orchestrates the data flow through the entire model, managing the main residual stream.
    - `attention.rs`: Defines struct `TransformerBlock` and struct `Attention`.
      - `TransformerBlock`: Contains an `Attention` and a `FeedForward` instance. Its `forward` method implements the critical residual connections: `x = x + attention(norm(x))` and `x = x + feed_forward(norm(x))`.
      - `Attention`: The workhorse. Holds four `op::BitLinear` layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Its `forward` method is stateful, taking an `index_pos` to manage the KV Cache. Contains logic for applying Rotary Position Embeddings (RoPE) to the query and key tensors before the attention calculation. The actual attention is performed by Candle's built-in FlashAttention-v2 equivalent.
    - `feed_forward.rs`: Defines struct `FeedForward`, holding the two `op::BitLinear` layers (`w1/gate_proj` and `w2/down_proj`) and the logic for the SwiGLU activation.
    - `rms_norm.rs`: A simple newtype wrapper for RMSNorm logic for API consistency and to potentially add tracing or other custom logic later.

- **File: crates/bitnet-core/src/op.rs — The BitLinear CustomOp**
  - **Core Purpose:** The polymorphic heart of the library. This file defines the BitLinear layer as a CustomOp, creating a seamless abstraction that separates the high-level model architecture from the low-level, high-performance computation backends, using wgpu for GPU compute.
  - **Informed By:** This design is a direct response to the need for separate, highly-specialized kernels for different hardware, a conclusion drawn from analyzing the difference between the Microsoft CUDA kernels and the GGML CPU kernels, and porting CUDA logic to WGSL for wgpu.
  - **Detailed Breakdown & Logic Flow:**
    - Defines `pub struct BitLinear`, holding the pre-processed (packed and permuted) weights and scaling factors as Tensors.
    - Implements `impl CustomOp for BitLinear`.
    - The `forward()` method is the crucial dispatcher. Its logic:
      - Step 1: Perform online activation quantization (derived from `model.py`'s `quant_input` method).
      - Step 2: Dispatch to the compile-time selected backend:
        - `#[cfg(feature = "gpu")]` — GPU path: call the wgpu kernel executor (`kernels::wgpu::execute(...)`)
        - `#[cfg(not(feature = "gpu"))]` — CPU path: call the SIMD dispatcher (`kernels::cpu::execute(...)`)
    - **Key Functions:**
      - `fn quantize_activations(&self, input: &Tensor) -> Result<(Tensor, Tensor)>` — helper for quantization logic.

- **The kernels Module: The Engine Room**
  - **kernels/wgpu.rs — GPU Backend**
    - **Purpose:** Implements the Decode-and-Multiply strategy using the wgpu API and WGSL shaders.
    - **Informed By:** A direct translation of the concepts in microsoft/BitNet/gpu/bitnet_kernels.cu, ported to WGSL for wgpu.
    - **Detailed Breakdown & Logic Flow:**
      - `pub fn execute(...)` receives the pre-quantized i8 activations and the packed i8 weights.
      - Loads the WGSL shader code from `crates/bitnet-core/src/kernels/bitnet_kernel.wgsl` via `include_str!`.
      - Creates and caches (e.g., in a OnceLock) the `wgpu::ComputePipeline` to avoid recompilation.
      - Creates a `wgpu::BindGroup` linking the tensor buffers (activations, weights, scales, output buffer) to the shader's storage buffer bindings.
      - Calculates the dispatch dimensions for the compute job based on the output tensor's shape.
      - Submits the compute pass to the GPU queue and awaits completion.
  - **kernels/cpu.rs & kernels/cpu_{x86,arm}.rs — CPU Backend**
    - **Purpose:** Implements the Look-Up Table (LUT) strategy using CPU-specific SIMD intrinsics.
    - **Informed By:** A direct translation of the concepts in the ggml bitnet-lut-kernels-*.h files.
    - **Detailed Breakdown & Logic Flow:**
      - `cpu.rs`: The main dispatcher. Its `execute` function uses runtime feature detection. It holds the pre-computed LUT (generated once when the model is loaded). Calls `cpu_x86::qgemm_lut(...)` or `cpu_arm::qgemm_lut(...)` based on `is_x86_feature_detected!`, etc.
      - `cpu_x86.rs` and `cpu_arm.rs`: Contain the unsafe Rust ports of the C++ SIMD logic.
        - `lut_ctor`: Takes the packed weights and generates the full look-up table, involving transpositions and pre-calculation of all possible outcomes, mirroring the C++ `three_lut_ctor` and `two_lut_ctor`.
        - `qgemm_lut`: The main GEMM function. Takes the quantized activations and uses them as indices into the LUT. The core operation is a series of SIMD shuffle/permute instructions (e.g., `_mm256_shuffle_epi8` on x86, `vqtbl1q_s8` on ARM) followed by additions.

- **The tests Directory**
  - **tests/kernel_tests.rs — Low-Level Correctness**
    - **Purpose:** To rigorously validate the unsafe SIMD code and the wgpu shader, which are the most error-prone parts of the project.
    - **Detailed Breakdown & Logic Flow:**
      - `fn scalar_lut_matmul(...)` in pure, safe Rust, performing the same LUT logic as the SIMD kernels, serving as the "ground truth". (Stub present)
      - Tests for each SIMD implementation generate random input and weight tensors, run them through both the unsafe SIMD function and the scalar ground truth, and assert bit-for-bit identical results. (Stub present)
      - A test for the wgpu kernel compares the GPU output against the scalar ground truth. (Stub present)
  - **tests/validation.rs — End-to-End Correctness**
    - **Purpose:** To ensure the entire system, when assembled, produces the exact same output as the original implementation, preventing subtle bugs from numerical precision or logic errors.
    - **Detailed Breakdown & Logic Flow:**
      - A "golden file" (e.g., `tests/data/golden_output.json`) contains a specific prompt and the first ~50 token IDs generated by the reference Python implementation.
      - A test function initializes the full Transformer model, loads converted weights, and runs the generation loop, asserting the generated token sequence matches the golden file for both CPU and GPU backends.

---

### Crate: bitnet-converter

**Purpose:** A critical command-line tool to convert standard model weights into the specific, pre-processed format our engine requires.

- **src/packer.rs**
  - **Informed By:** A direct reverse-engineering of `convert_checkpoint.py` and `pack_weight.py`.
  - **Detailed Breakdown & Logic Flow:**
    - The `full_conversion_pipeline` function executes the following four steps in strict order:
      1. **Quantize:** Implements the `quant_weight_int8` logic, scaling and clamping weights to {-1, 0, 1}.
      2. **Permutate:** Implements the `permutate_weight_fastest` logic, a complex reordering of weights for optimal memory access.
      3. **Pack:** Implements the `compress_int2_to_int8` logic, mapping values to 2-bit representations and packing four into a single i8 byte.
      4. **Interleave:** Implements the `interleave_weight_int8` logic, a final bit-level shuffle to match kernel expectations.

---

### Crate: bitnet-app

**Purpose:** The user-facing application, providing both a CLI and a simple, responsive GUI.

- **src/generation.rs**
  - **Purpose:** Acts as the stateful inference controller.
  - **Detailed Breakdown & Logic Flow:**
    - A `Generator` struct is initialized with a loaded model and tokenizer, holding the state (primarily the KV Cache tensors).
    - Its `generate_next_token()` method takes the last token, runs one step of the model's forward pass (updating the KV Cache), and returns the raw logits for the next token.

- **src/sampler.rs**
  - **Purpose:** Provides a stateless logits processor.
  - **Informed By:** `sample_utils.py` and standard Hugging Face samplers.
  - **Detailed Breakdown & Logic Flow:**
    - A `LogitsProcessor` struct is configured with the `InferenceSettings`.
    - Its `sample(logits: &Tensor) -> u32` method applies temperature, repetition penalties, top_k, and top_p filtering to the incoming logits tensor and returns a single sampled token ID.

- **src/gui/backend.rs**
  - **Purpose:** Enables a non-blocking UI by running the model on a separate thread.
  - **Detailed Breakdown & Logic Flow:**
    - On GUI start, this module spawns a new thread for the model.
    - Uses `std::sync::mpsc::channel` for message passing. The UI thread holds the Sender, and the model thread holds the Receiver.
    - When the user sends a message, the UI thread sends a `BackendCommand::Generate { prompt: String }` message to the model thread.
    - The model thread loops, receiving commands. On `Generate`, it enters a generation loop. For each token produced, it sends a `UICommand::AppendToken { token: String }` message back to the UI thread.
    - The UI thread's update loop uses `try_recv()` to check for new tokens without blocking, appending them to the chat display as they arrive. This creates the "streaming" effect and keeps the UI responsive.

## Guiding Principles & Philosophy

This project aims to create a high-performance, dual-backend inference engine for BitNet-style models, written entirely in pure Rust. Our philosophy is guided by three core principles:

- **Performance through Specialization:**
We recognize that peak performance on different hardware (GPU vs. CPU) requires fundamentally different algorithmic approaches. We will implement two highly specialized backends: a "Decode-and-Multiply" kernel for GPUs (informed by the Microsoft CUDA implementation) and a "Look-Up Table" (LUT) kernel for CPUs (informed by the GGML CPU kernels), leveraging the unique strengths of each architecture.

- **Ergonomics and Accessibility:**

The library must be easy to use. Rust's feature flags will allow users to compile a CPU-only version without needing GPU drivers or libraries, ensuring maximum portability. The final application will be available as both a powerful CLI and an intuitive, responsive GUI.

- **Correctness through Rigorous Testing:**

unsafe code for performance demands a non-negotiable commitment to correctness. Our plan includes multiple layers of testing: low-level kernel validation against scalar ground truths, and high-level, end-to-end "golden file" testing to ensure perfect fidelity with the original model's output.

## Rust Development: Traps, Pitfalls, & Best Practices

This section outlines key Rust-specific concepts to ensure the project is performant, robust, and idiomatic.

- **Memory and Performance: The Zero-Cost Abstraction is Not Free**
  - The .clone() Trap: clone() is often a deep, allocating copy. In hot loops like the generation cycle, cloning a Tensor or a large Vec is a performance disaster.
    - **Instruction:**
    Pass by reference (& or &mut) wherever possible. A clone() should be a deliberate design choice, not a default.
  - Heap Allocations in Loops: Avoid creating objects that allocate on the heap (e.g., Vec, String) inside performance-critical loops.
    - **Instruction:**
    Pre-allocate buffers, strings, or vectors outside the loop and reuse them. The generation.rs loop should pre-allocate its token history vector with a reasonable capacity.

- **Error Handling: Production Code Must Not Panic**
  - The .unwrap() and .expect() Pitfall: These methods are for prototypes, tests, and examples only. Using them in library or application code is a bug.
    - **Instruction:** All fallible operations must return a Result. Use the ? operator to propagate errors up the call stack. Define custom, descriptive error types in error.rs using thiserror.

- **Concurrency: Choose the Right Tool for the Job**
  - The async Misconception: async/await is primarily for I/O-bound tasks. It does not make CPU-bound code faster; it adds overhead.
    - **Instruction:** Our compute-heavy kernels are CPU-bound. For the GUI, where we need to offload this work to keep the UI responsive, we will use std::thread::spawn and communicate with mpsc channels.

- **Code Style & Idioms: Write Clear, Maintainable Rust**
  - Clarity Over Cleverness: Avoid overly complex, chained functional iterators if a simple, imperative for loop is more readable.
  - Clippy is Law: The Rust clippy linter is an essential tool.
    - **Instruction:** The project's CI pipeline must include a cargo clippy -- -D warnings step to fail the build on any linting errors.

- **Logging and Tracing:**
  - **Instruction:** Do not use println! for debugging. Use the tracing crate. It allows for structured, leveled logging that can be configured at runtime and routed to performance analysis tools. We will instrument key functions with #[instrument] to automatically get performance spans.

## Documentation Strategy & Docs.rs Integration

Our documentation strategy follows Rust's best practices for both internal development and public API documentation.

### Docs.rs Integration

- **Crate-Level Documentation:**
  - Each crate's `lib.rs` must have comprehensive root documentation.
  - Include examples, quick start guide, and feature flag explanations.
  - Use `#[cfg(doctest)]` modules to ensure examples in documentation are tested.
  - Add `#[doc(html_root_url = "...")]` to specify documentation root URL.

- **Module-Level Documentation:**
  - Every public module must have a module-level doc comment explaining its purpose.
  - Include examples of common use cases.
  - Document feature flag implications.
  - Use `#[doc(alias = "...")]` for searchable alternative names.

- **Type & Function Documentation:**
  - Every public type and function must have documentation.
  - Include `# Examples` sections with runnable code.
  - Document error cases and panics.
  - Use `# Safety` sections for unsafe functions.
  - Add `# Performance` notes for critical path code.

- **Documentation Tests:**
  - All example code must be tested via `cargo test --doc`.
  - Include both success and error cases.
  - Test with different feature flag combinations.

### Internal Documentation

- **Architecture Documentation:**
  - Maintain detailed README.md files in each directory.
  - Document architectural decisions and their rationales.
  - Keep diagrams and flow charts up to date.

- **Code Comments:**
  - Use `//!` for module documentation.
  - Use `///` for public item documentation.
  - Add `// Note:` comments for important implementation details.
  - Document UNSAFE blocks with clear safety requirements.

- **Performance Documentation:**
  - Document performance characteristics of critical functions.
  - Include benchmark results and optimization notes.
  - Document SIMD and GPU kernel implementation details.

### Documentation Workflow

1. **Write First:**
   - Documentation must be written before code review.
   - Examples must be included and tested.
   - Performance characteristics must be documented.

2. **Review Process:**
   - Documentation is reviewed as part of code review.
   - Examples must be verified to work.
   - Check for clarity and completeness.

3. **Maintenance:**
   - Update docs when API changes.
   - Keep performance notes current.
   - Review and update examples regularly.

4. **CI Integration:**
   - Run `cargo doc --no-deps` in CI.
   - Ensure all doc tests pass.
   - Check for broken links.
   - Verify documentation coverage.

### Documentation Standards

- **Style:**
  - Use active voice.
  - Be concise but complete.
  - Include working examples.
  - Document error cases.

- **Structure:**
  - Follow standard sections: Examples, Errors, Safety, Performance.
  - Use consistent formatting.
  - Include links to related items.

- **Coverage:**
  - All public items must be documented.
  - Critical private items should be documented.
  - Document feature flag implications.
  - Include version compatibility notes.

### Versioning & Stability

- **Version Policy:**
  - Follow SemVer strictly.
  - Document breaking changes.
  - Use stability attributes appropriately.

- **Compatibility:**
  - Document MSRV (Minimum Supported Rust Version).
  - Note platform-specific features.
  - Document feature flag combinations.

### Core GUI/Visualization Module (bitnet-core/src/gui/)

**Purpose:**

- Provide a developer-facing, core-level UI for visualizing and debugging model internals kernel performance, and training progress.
- Enable advanced users to inspect weights, activations, attention maps, and kernel timings directly from the core library, independent of the main application GUI.
- Facilitate rapid debugging and performance tuning during development and research.

**Planned Files/Modules:**

- `mod.rs`: Entry point for the core GUI/visualization module.
- `dashboard.rs`: Minimal dashboard for real-time metrics, kernel timings, and training progress.
- `weights_viewer.rs`: Tools for visualizing model weights, distributions, and quantization effects.
- `kernel_profiler.rs`: Interactive profiling and visualization of CPU/GPU kernel performance and correctness.
- `attention_map.rs`: Visualization of attention matrices and activations.
- `README.md`: Documentation and usage examples for core GUI features.

**Implementation Notes:**

- This is intended for advanced users, developers, and researchers.

- It may use `egui`, `plotters`, or other Rust-native visualization libraries.
- All core GUI features should be optional and gated behind a feature flag (e.g., `core-gui`).
- The main application GUI (in bitnet-app) remains the user-facing interface; this core GUI is for internal development, debugging, and research.
