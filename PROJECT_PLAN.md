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

```
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
├── burn-custom-kernel-test-0-17/
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
│   │   │   ├── op.rs
│   │   │   ├── tokenizer.rs
│   │   │   ├── hf.rs
│   │   │   ├── settings.rs
│   │   │   ├── embedding.rs
│   │   │   ├── training.rs
│   │   │   ├── visualization.rs
│   │   │   ├── kernels.rs
│   │   │   └── kernels/
│   │   │       ├── bitnet_kernel.wgsl
│   │   │       ├── wgpu.rs
│   │   │       ├── cpu.rs
│   │   │       ├── cpu_x86.rs
│   │   │       ├── cpu_arm.rs
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
│       │   └── hf_loader.rs
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
| **bitnet-core/src/op.rs** | BitLinear CustomOp | BitLinear struct, CustomOp impl, backend dispatch | Central hub for CPU/GPU, must be correct and fast |
| **bitnet-core/src/tokenizer.rs** | Text tokenizer | Tokenizer struct wrapping tokenizers crate, ChatFormat logic | Must match Python tokenizer, handle chat templates |
| **bitnet-core/src/kernels/mod.rs** | Kernel aggregator | Declares cpu, wgpu modules | - |
| **bitnet-core/src/kernels/cpu.rs** | CPU backend | Forward function, runtime SIMD detection, scalar fallback | **Fragile**: SIMD dispatch, must validate against scalar |
| **bitnet-core/src/kernels/cpu_x86.rs** | AVX2 SIMD kernel | Rust AVX2 intrinsics, lut_ctor, tbl_impl_* | **Very fragile**: Unsafe, must match scalar exactly |
| **bitnet-core/src/kernels/cpu_arm.rs** | NEON SIMD kernel | Rust NEON intrinsics, lut_ctor, tbl_impl_* | **Very fragile**: Unsafe, must match scalar exactly |
| **bitnet-core/src/kernels/wgpu.rs** | GPU backend | Manages wgpu, loads shader, dispatches compute | **Very difficult**: Performance tuning, correctness |
| **bitnet-core/tests/validation.rs** | End-to-end tests | Golden file test, prompt-to-token match | Must match Python output exactly |
| **bitnet-core/tests/kernel_tests.rs** | Kernel validation | Scalar vs SIMD/GPU output comparison | **Non-negotiable**: Must pass for all kernels. Now exists as a stub with scalar reference and test stubs. |
| **bitnet-converter/src/main.rs** | CLI entry | Uses clap, calls packer | - |
| **bitnet-converter/src/packer.rs** | Weight conversion | quantize -> permutate -> pack -> interleave | Must match Python scripts exactly |
| **bitnet-app/src/main.rs** | User app entry | Loads model/tokenizer, runs generation loop | - |
| **bitnet-app/src/generation.rs** | Generation engine | Generator struct, manages KV cache | - |
| **bitnet-app/src/sampler.rs** | Logits processor | LogitsProcessor struct, sampling logic | - |
| **bitnet-app/src/gui/backend.rs** | GUI backend | Threaded model execution, mpsc channels | - |

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
  - *Resolution:* Rigorous, mandatory validation. Every SIMD function is tested against a scalar version. No PR is merged unless all tests pass.
- **GPU implementation may not be fast initially.**
  - *Resolution:* Plan for iterative tuning. Use wgpu profiling, experiment with workgroup sizes, memory layout, and compare different kernel strategies.
- **Dependency on GGML logic is a risk.**
  - *Resolution:* Treat GGML as a spec, not gospel. Validate against our own scalar implementation first, then against official outputs.
- **Numeric precision differences could cause divergence.**
  - *Resolution:* End-to-end golden testing. CI must run golden prompt tests for both CPU and GPU, asserting output token IDs match the reference.

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

### Deep Dive 1: op.rs — The BitLinear CustomOp & Kernel Signatures

This is the most important interface in the project. Its implementation must be precise.

**Why:** This is the most important interface in the project. It abstracts backend-specific logic and ensures the model code is backend-agnostic.

**Struct Definition:**

```rust
// In crates/bitnet-core/src/op.rs
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
    // This is a prime example of backend-specific data.
    #[cfg(not(feature = "gpu"))]
    precomputed_lut: Tensor,
}
```

**forward() Logic Flow:**

```rust
// In impl CustomOp for BitLinear
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // 1. Online Activation Quantization
    // Replicates the `quant_input` logic from the Python model.py.
    let (x_quant, x_scales) = self.quantize_activations(x)?;

    // 2. Backend Dispatch (Compile-Time)
    #[cfg(feature = "gpu")]
    {
        // GPU path: call the wgpu kernel executor.
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
        // CPU path: call the SIMD dispatcher.
        crate::kernels::cpu::execute(
            &x_quant,
            &x_scales,
            &self.precomputed_lut, // This would be part of the BitLinear struct
            &self.weight_scales,
        )
    }
}
```

**Validation Strategy:** The op.rs itself doesn't need unit tests, as its logic is just dispatch. The real testing happens in the kernel tests, which validate the execute functions it calls.

---

### Deep Dive 2: kernels/cpu_*.rs — The CPU LUT Kernel Logic

**Why:** The LUT approach is the only way to achieve high performance for quantized inference on CPUs. It is also the most fragile and error-prone part of the codebase.

**Informed By:** The ggml/bitnet-lut-kernels-*.h files.

**Core Concept:** The matrix multiplication is transformed into a series of table lookups. This happens in two phases: a one-time lut_ctor (LUT constructor) and the actual inference qgemm_lut.

### Purpose

- Provide a developer-facing, core-level UI for visualizing and debugging model internals kernel performance, and training progress.
- Enable advanced users to inspect weights, activations, attention maps, and kernel timings directly from the core library, independent of the main application GUI.
- Facilitate rapid debugging and performance tuning during development and research.

### Implementation Notes

- This is intended for advanced users, developers, and researchers.
- It may use `egui`, `plotters`, or other Rust-native visualization libraries.
- All core GUI features should be optional and gated behind a feature flag (e.g., `core-gui`).

### Phase A: LUT Constructor (lut_ctor)

- Runs once when BitLinear is created.
- **Input:** The packed i8 weight tensor.
- **Output:** A new i8 tensor representing the Look-Up Table (QLUT).
- **Logic Flow:**
  - For each group of weights that a SIMD instruction will process, calculate all possible output sums for every combination of input values {-1, 0, 1}.
  - Store these pre-calculated sums in a small table.
  - Transpose & permute the LUT data for a SIMD-friendly memory layout (mirroring GGML's Transpose_8_8).
  - Concatenate these small tables into the large, final QLUT tensor.

### Phase B: Inference (qgemm_lut)

- **Input:** The i8 quantized activations and the pre-computed QLUT.
- **Logic Flow:**
  - Load a vector of quantized activations using a SIMD instruction.
  - Use these activation values as indices into the QLUT via SIMD shuffle/table instructions (_mm256_shuffle_epi8 on x86, vqtbl1q_s8 on ARM). This is the core "multiplication-free" step.
  - Accumulate the results from the lookups into a SIMD vector of sums.
  - After the inner loop, de-quantize: convert the accumulator vector to f32, multiply by activation and weight scales.

**Validation Strategy:** kernel_tests.rs must contain a fn scalar_qgemm_lut(...) that performs the same logic with simple for loops and array lookups. Tests for cpu_x86.rs and cpu_arm.rs will assert bit-for-bit identical output to this scalar version for a variety of inputs.

---

### Deep Dive 3: bitnet-core/src/kernels/bitnet_kernel.wgsl — The GPU Kernel Data Flow

**Why:** The GPU kernel is the most performance-critical and hardware-specific part of the project. It must be correct, fast, and match the CPU/official outputs.

**Informed By:** microsoft/BitNet/gpu/bitnet_kernels.cu.

**Core Concept:** Decode-and-Multiply. Each thread in a workgroup processes a portion of the matrix multiplication in parallel.

**WGSL Pseudo-Shader Structure:**

```wgsl
// Bindings for our input and output buffers
@group(0) @binding(0) var<storage, read> activations: array<i32>; // i8 packed into i32 for efficiency
@group(0) @binding(1) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<i32>; // e.g., 16x 2-bit weights per i32
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, write> output: array<f32>;

var<workgroup> partial_sums: array<atomic<i32>, 128>; // Size matches workgroup size

fn decode_weights(packed: u32) -> array<i32, 16> { /* bitwise shifts and masks */ }
fn dp4a_emulation(a: vec4<i32>, b: vec4<i32>) -> i32 { return dot(a, b); }

@compute @workgroup_size(8, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_idx = local_id.y * 8 + local_id.x;
    var accumulator: i32 = 0;
    // Main Loop: decode, multiply, accumulate
    // Reduction: atomicStore, workgroupBarrier, final sum, dequantize, write output
}
```

**Validation Strategy:** kernel_tests.rs will compare the GPU output against the scalar CPU implementation. The bitwise logic in decode_weights must be the exact inverse of the packing logic in bitnet-converter.

---

### Deep Dive 4: bitnet-converter/packer.rs — The Weight Conversion Pipeline

**Why:** The converter is the gatekeeper of correctness for the entire system. If the weights are packed incorrectly, all inference will be wrong.

**Informed By:** A direct reverse-engineering of convert_checkpoint.py and pack_weight.py. This utility is the gatekeeper of correctness for the entire system.

**Data Flow Diagram:**
Tensor<f32> -> [1. Quantize] -> (Tensor<i8>, Tensor<f32>) -> [2. Permutate] -> (Tensor<i8>, ...) -> [3. Pack] -> (Tensor<i8>, ...) -> [4. Interleave] -> (Tensor<i8>, ...) -> Final Saved Artifact

- **Step 1: Quantize**
  - Input: f32 weight tensor.
  - Logic: Implements quant_weight_int8. Calculates per-tensor or per-channel mean, scales values, rounds, and clamps to the set {-1, 0, 1}.
  - Output: i8 tensor containing only {-1, 0, 1} + f32 scales tensor.
- **Step 2: Permutate**
  - Input: i8 quantized tensor.
  - Logic: Implements permutate_weight_fastest. This is a complex, non-trivial reordering of weight elements to ensure that data accessed by consecutive threads in a kernel is also consecutive in memory, maximizing memory bandwidth.
  - Output: A new i8 tensor with the same data but a different layout.
- **Step 3: Pack**
  - Input: Permuted i8 tensor.
  - Logic: Maps {-1, 0, 1} to their 2-bit representations (e.g., 01, 00, 10). Uses bitwise shifts and OR operations to pack four of these 2-bit values into a single i8 byte.
  - Output: i8 tensor, with its size along the packed dimension reduced by a factor of 4.
- **Step 4: Interleave**
  - Input: Packed i8 tensor.
  - Logic: Implements interleave_weight_int8. Performs a final, bit-level shuffle on the packed data to exactly match the memory access pattern the kernel's decoding function expects.
  - Output: The final i8 tensor to be saved.

**Validation Strategy:** Each step will be a separate, public function in the packer module. tests within the crate will use small, hand-crafted input tensors and assert that the output of each step matches a pre-calculated, known-good result.

---

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
We recognize that peak performance on different hardware (GPU vs. CPU) requires fundamentally different algorithmic approaches. We will implement two highly specialized backends: a "Decode-and-Multiply" kernel for GPUs (informed by the Microsoft CUDA implementation) and a "Look-Up Table" (LUT) kernel for CPUs (informed by the GGML implementation), leveraging the unique strengths of each architecture.

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