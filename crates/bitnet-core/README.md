# BitNet Core (`bitnet-core`)

A pure Rust, streaming-friendly core engine for BitNet models, focused on high-performance inference, quantization, and kernel dispatch. Includes all performance-critical logic, model definitions, and backend implementations for both CPU and GPU (WGSL).

---

## Table of Contents

- [Purpose](#purpose)
- [Main Modules](#main-modules)
- [Architecture](#architecture)
- [How to Use](#how-to-use)
- [Features](#features)
- [Kernel & Quantization](#kernel--quantization)
- [Attention, Quantization, and Kernel Design Rationale](#attention-quantization-and-kernel-design-rationale)
- [Test Coverage](#test-coverage)
- [Implementation Notes](#implementation-notes)

---

## Purpose

- Serve as the backend engine for BitNet inference (and planned training)
- Provide modular, extensible components for model architecture, quantization, and kernel dispatch
- Support both CPU (SIMD) and GPU (WGSL) backends
- Enable streaming-friendly, per-block model loading and execution

## Main Modules

- `model.rs`: Pure Rust Transformer model architecture (no burn dependency)
- `attention.rs`, `feed_forward.rs`, `rms_norm.rs`: Core model submodules (pure Rust)
- `bitnet_linear.rs`: BitLinear quantized layer, packing, and quantization utilities
- `kernels/`: CPU/GPU kernel implementations (WGSL, SIMD)
- `settings.rs`: Inference and generation settings
- `embedding.rs`: Embedding layer
- `tokenizer.rs`: Tokenizer and chat template logic
- `error.rs`: Error types and handling
- `gui/`: (Optional) Core-level visualization and debugging UI for developers (feature-gated)
- `training.rs`, `visualization.rs`: (Planned) Training and logging/metrics hooks

## Architecture

- **Pure Rust, burn-free**: All core logic is implemented in Rust, with no dependency on the burn framework for inference
- **Streaming-friendly**: Model weights are loaded per-block, supporting large models and efficient memory usage
- **Quantized & packed**: Uses ternary quantization and efficient packing for weights and activations
- **GPU kernel integration**: Includes WGSL kernels for high-performance inference on modern GPUs

## How to Use

Add to your `Cargo.toml`:

```toml
bitnet-core = { path = "../bitnet-core" }
```

Then in your code:

```rust
use bitnet_core::model::Transformer;
// ...
```

## Features

- Modular, extensible design
- Optional GPU and core-gui features (feature flags)
- Designed for correctness, performance, and portability
- Streaming-friendly model loading and execution
- Robust error handling and test coverage

## Kernel & Quantization

- **WGSL GPU kernel**: See `src/kernels/bitnet_kernel.wgsl` for the main ternary matmul kernel
- **Packing utilities**: See `src/kernels.rs` for pure Rust packing and scale calculation
- **Quantization**: Scalar and SIMD quantization utilities for activations and weights
- **Tested against scalar reference**: All kernels are validated against pure Rust reference implementations

## Attention, Quantization, and Kernel Design Rationale

BitNet uses a hybrid approach for kernel and quantization design, following the original BitNet paper and best practices for efficient transformer inference:

### Summary Table

| Operation                | Quantized (Ternary)? | Kernel Used                  |
|--------------------------|----------------------|------------------------------|
| Q/K/V Projections        | Yes                  | `bitnet_kernel.wgsl`         |
| Output Projection        | Yes                  | `bitnet_kernel.wgsl`         |
| Feed-Forward Layers      | Yes                  | `bitnet_kernel.wgsl`         |
| Attention (softmax, etc) | No (f32)             | `bitnet_attention.wgsl`      |

### Rationale

- **Ternary Quantization for Linear Layers:**
  - The BitNet paper applies ternary quantization (`{-1, 0, +1}`) only to the core matrix multiplications (all linear layers: Q/K/V projections, output projection, feed-forward layers).
  - This is where most parameters and compute reside, so quantizing these gives the largest efficiency gain.
  - Our `bitnet_kernel.wgsl` and `bitnet_kernel_optimal.wgsl` implement these quantized matmuls, validated against scalar references and used throughout the model.

- **Full-Precision Attention Math:**
  - The actual attention computation—softmax(QKᵀ)V with causal masking—is performed in full precision (f32), as in the BitNet paper.
  - This is because the attention mechanism is sensitive to quantization errors, especially in the softmax and masking steps.
  - Our `bitnet_attention.wgsl` kernel implements this operation, taking Q, K, V (already projected by quantized matmuls) and producing the attended output in f32.

- **Testing Philosophy:**
  - All kernels are validated against pure Rust scalar reference implementations.
  - Dedicated tests exist for both quantized matmul and full-precision attention, ensuring correctness and cross-device consistency.

- **Why This Design?**
  - This hybrid approach maximizes efficiency (by quantizing the heavy matmuls) while preserving accuracy in the most sensitive part of the transformer (the attention softmax block).
  - It matches the BitNet paper and is standard in high-performance transformer inference.

For more details, see the code comments in each kernel and the test suite in `tests/`.

## Test Coverage

- Unit tests for packing, quantization, and kernel correctness
- Direct wgpu kernel launch tests (no burn dependency)
- End-to-end model pipeline validation (see `tests/pipeline_validation.rs`)
- Streaming and per-block model loading tests
- **Optional Stress Test**: A long-running stress test (`stress_test_maximum_dimension_support`) is available but ignored by default. To run it, set the `RUN_STRESS_TESTS` environment variable:
  - **PowerShell**:
  
    ```powershell

    $env:RUN_STRESS_TESTS="1"; cargo test --package bitnet-core --test kernel_tests -- --nocapture

    ```

  - **Linux/macOS**:

    ```bash
    RUN_STRESS_TESTS=1 cargo test --package bitnet-core --test kernel_tests -- --nocapture
    ```

### Test Results Summary

- All GPU kernels (both SAFE and OPTIMAL variants) are validated for correctness against the scalar CPU reference implementation.
- Large batch benchmarks show a **~15x speedup** for GPU inference over CPU for typical matrix sizes (e.g., 2048x1024x1024), with both SAFE and OPTIMAL kernels performing similarly on tested hardware.
- All tests (correctness, performance, memory safety, and cross-device consistency) pass on major backends (Vulkan, DX12, OpenGL).
- For detailed results and timing breakdowns, see [`logs/kernel_tests.md`](../../logs/kernel_tests.md).

## Implementation Notes

- See the project plan for architecture and validation strategies
- Use feature flags to enable GPU or core-gui modules
- For kernel and quantization details, see code comments in `src/kernels.rs` and `src/kernels/bitnet_kernel.wgsl`

---

**For questions or contributions, see the main project README or open an issue.** 