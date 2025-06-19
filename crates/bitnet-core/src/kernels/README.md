# BitNet Shaders

This directory contains GPU compute shaders for BitNet, written in WGSL (WebGPU Shading Language).

## Files
- `bitnet_kernel.wgsl`: The primary GPU kernel for the decode-and-multiply strategy used in BitNet inference.

## Purpose
- Accelerate quantized matrix multiplication and other core operations on GPU hardware
- Enable high-performance inference for large models

## How to Use
- The kernel is loaded and dispatched by the `bitnet-core` crate's `kernels/wgpu.rs` module
- To modify the kernel, edit `bitnet_kernel.wgsl` and rebuild the project

## Testing and Validation
- Kernel correctness is validated against CPU and scalar implementations in the test suite
- Performance tuning is iterative: start with correctness, then profile and optimize

## Implementation Notes
- WGSL is the shading language for WebGPU, supported on modern GPUs
- See the project plan for details on kernel design and validation

---

# Reference: Official BitNet Kernels (Must Read Before Implementing)

## GPU (CUDA) Kernels
- **Files:**
  - `References/gpu/bitnet_kernels/bitnet_kernels.cu`
  - `References/gpu/bitnet_kernels/bitnet_kernels.h`

### What They Do
- Implement highly-optimized CUDA kernels for quantized matrix multiplication (int8 × int2), with hardcoded shapes (M, N, K) for maximum performance.
- Use template metaprogramming and CUDA intrinsics for memory access, quantization, and reduction.
- Entry point: `bitlinear_int8xint2` dispatches to specialized kernels based on matrix shape.

### Key Concepts to Learn/Port
- **Data Layout:** Inputs are packed/quantized, often in int2 or int8, and require decoding on the device.
- **Shape Specialization:** Kernels are written for specific (M, N, K) shapes, not generic GEMM.
- **Quantization:** Input weights are quantized, and scaling factors are used to recover floating-point results.
- **Reduction:** Use of warp shuffles and shared memory for fast reductions.
- **Launch Parameters:** Grid/block sizes are tuned for each shape.

### If Implementing in Rust (with wgpu/wgsl):
- **Data Upload:** You must pack/quantize your data on the Rust side, matching the layout expected by the kernel.
- **Specialization:** Consider specializing your WGSL kernels for common shapes, or use dynamic branching for flexibility (at some performance cost).
- **Decoding:** Implement int2/int8 decoding in WGSL. See the device-side decode logic in `bitnet_kernels.h`.
- **Scaling:** Pass scaling factors as uniforms or buffers, and apply them after the dot product.
- **Thread Layout:** Map CUDA thread/block logic to WGSL workgroups and invocations. Be aware of differences in synchronization and memory model.
- **Testing:** Validate against the CPU reference (see below) for correctness.

---

## CPU (SIMD) Kernels
- **Files:**
  - `References/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h` (ARM/NEON)
  - `References/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h` (x86/AVX2)
  - (And similar files for other models in `preset_kernels/`)

### What They Do
- Provide highly-optimized CPU kernels for quantized matrix multiplication using SIMD intrinsics.
- Implement quantization, LUT (lookup table) construction, and specialized GEMM for specific shapes.
- Split by architecture (tl1 = ARM, tl2 = x86) and by model size.

### Key Concepts to Learn/Port
- **SIMD Quantization:** How to quantize weights and inputs efficiently using vector instructions.
- **LUT Construction:** How to build lookup tables for fast int2/int8 operations.
- **Shape Specialization:** As with GPU, kernels are specialized for certain shapes.
- **Preprocessing:** Quantization and LUT construction are often done as a preprocessing step.

### If Implementing in Rust:
- **SIMD:** Use `std::arch` or crates like `packed_simd`/`wide` for SIMD operations. For fallback, provide scalar versions.
- **FFI:** If you want to use the C/C++ kernels directly, use Rust's FFI (`bindgen`, `cc` crate) to call them.
- **Data Layout:** Match the memory layout and quantization format used in the reference kernels.
- **Testing:** Use the CPU kernels as a correctness reference for your GPU implementation.

---

## Concrete Steps for BitNet Kernel Development

### 1. Study the Reference Kernels
- Read the CUDA and CPU kernel files in detail. Take notes on:
  - Data layout (how are int2/int8 packed? How are scales stored?)
  - Kernel entry points and dispatch logic
  - Quantization and dequantization math
  - Thread/block/workgroup organization

### 2. Decide on Your Target Shapes
- Will you support only the common shapes (as in the reference), or make a general kernel?
- Specialization gives better performance, but less flexibility.

### 3. Plan Data Flow in Rust
- **Quantize weights/inputs** on the Rust side, matching the reference layout.
- **Upload data** to GPU buffers, ensuring alignment and packing match the kernel's expectations.
- **Pass scaling factors** as uniforms or buffers.
- **Dispatch kernels** with the correct workgroup sizes.

### 4. Implement Decoding and GEMM in WGSL
- Port the int2/int8 decode logic from CUDA/CPU to WGSL.
- Implement the dot product and reduction logic, using WGSL's workgroup/shared memory features.
- Apply scaling factors after accumulation.

### 5. Validate and Benchmark
- Compare GPU results to CPU reference for correctness.
- Profile and tune workgroup sizes, memory access patterns, and specialization.

### 6. Integration
- Expose the kernel via Rust (e.g., in `bitnet-core/src/kernels/wgpu.rs`).
- Provide a fallback to CPU if GPU is unavailable or for unsupported shapes.
- Document all data formats and kernel expectations.

---

## Example: Data Packing and Kernel Call in Rust

```rust
// Example: Packing int2 weights in Rust
fn pack_int2(weights: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((weights.len() + 3) / 4);
    for chunk in weights.chunks(4) {
        let mut byte = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            let val = (w as u8) & 0x3; // 2 bits
            byte |= val << (i * 2);
        }
        packed.push(byte);
    }
    packed
}

// Example: Dispatching a WGSL kernel with wgpu
let bind_group = ...; // Set up with packed weights, input, scales
let workgroup_count = ...; // Compute based on shape
encoder.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
```

---

## Further Reading & Next Steps
- **Official BitNet kernels** are the best source for understanding the math and performance tricks.
- **Rust GPU ecosystem:** See crates like `wgpu`, `naga`, and `bytemuck` for GPU programming and data layout.
- **SIMD in Rust:** See `std::arch`, `wide`, or `packed_simd` for CPU-side vectorization.
- **FFI:** If you want to use the C/C++ kernels directly, use `bindgen` and the `cc` crate.

---

## Summary Table

| Task                | Where to Look                                              | What to Learn/Do                |
|---------------------|-----------------------------------------------------------|---------------------------------|
| GPU kernel (CUDA)   | `bitnet_kernels.cu`, `bitnet_kernels.h`                   | Data layout, quantization, GEMM |
| CPU kernel (SIMD)   | `bitnet-lut-kernels-tl1.h`, `bitnet-lut-kernels-tl2.h`    | SIMD quantization, LUT, GEMM    |
| Model-specific info | `preset_kernels/`                                         | Shape specialization            |

---

## Final Advice
- **Don't reinvent the wheel:** The official BitNet kernels are highly optimized. Use them as your starting point.
- **Document as you go:** Keep notes on data layout, kernel logic, and integration points.
- **Prototype for correctness first, then optimize.**
- **Share findings with the team:** This is a complex area—collaboration is key.

---


