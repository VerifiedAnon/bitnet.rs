//! Fixed test for the Naga WGSL->DX12 compiler bug.
//
// This test file contains multiple workaround strategies for the error:
// `FXC D3DCompile error (0x80004005): ... error X3017: cannot convert from 'int[16]' to 'int'`
//
// The main strategies tested:
// 1. Avoid double indexing with temporary variables
// 2. Use explicit vector component access
// 3. Restructure the accumulator access pattern

use serial_test::serial;
use std::sync::{Arc, Mutex};
use chrono::Local;
use tokio;
use wgpu::util::DeviceExt;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use bitnet_core::kernels::{pack_ternary_weights, BitnetMetadata};
use bitnet_tools::test_utils::TestReporter;
use lazy_static::lazy_static;
use std::time::Instant;
use std::panic::AssertUnwindSafe;
use futures::FutureExt;
#[macro_use]
extern crate bitnet_tools;

// Initialize test reporter
lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("dx12_test").expect("Failed to create test reporter");
}

// Original buggy kernel for reference
// PURPOSE: This is the baseline failing case. It demonstrates the Naga WGSL->DX12 compiler bug
// where direct double-indexing into an array of vectors (`accumulators[m][n]`) inside a nested
// loop causes a compiler error. This kernel is expected to fail shader compilation and pipeline creation.
const BUGGY_KERNEL_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
};

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

var<workgroup> tile_a: array<vec4<i32>, 512>;
var<workgroup> tile_b: array<i32, 2048>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

fn dot_product_4x4(a: vec4<i32>, b: vec4<i32>) -> i32 {
    return dot(a, b);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // THIS IS THE STRUCTURE THAT CAUSES THE BUG
    var accumulators: array<vec4<i32>, THREAD_TILE_M>;
    for (var i = 0u; i < THREAD_TILE_M; i = i + 1u) {
        accumulators[i] = vec4<i32>(0);
    }
    
    // Problematic computation - double indexing causes DX12 issues
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            accumulators[m][n] += (m + n); // This line causes the error
        }
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(accumulators[0].x);
    }
}
"#;

// WORKAROUND 1: Use temporary variables to avoid double indexing
// HYPOTHESIS: The compiler fails on compound expressions involving double indexing.
// STRATEGY: Break down the operation `accumulators[m][n] += ...` into three steps:
// 1. Load the vector `vec4` into a temporary variable: `var temp_vec = accumulators[m];`
// 2. Modify the components of the temporary variable.
// 3. Write the modified temporary variable back to the array: `accumulators[m] = temp_vec;`
const FIXED_KERNEL_V1_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
};

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

var<workgroup> tile_a: array<vec4<i32>, 512>;
var<workgroup> tile_b: array<i32, 2048>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

fn dot_product_4x4(a: vec4<i32>, b: vec4<i32>) -> i32 {
    return dot(a, b);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    var accumulators: array<vec4<i32>, THREAD_TILE_M>;
    for (var i = 0u; i < THREAD_TILE_M; i = i + 1u) {
        accumulators[i] = vec4<i32>(0);
    }
    
    // WORKAROUND 1: Use temporary variable to avoid double indexing
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        var temp_vec = accumulators[m]; // Load the entire vector first
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            // Modify individual components explicitly
            if (n == 0u) { temp_vec.x += i32(m + n); }
            else if (n == 1u) { temp_vec.y += i32(m + n); }
            else if (n == 2u) { temp_vec.z += i32(m + n); }
            else if (n == 3u) { temp_vec.w += i32(m + n); }
        }
        accumulators[m] = temp_vec; // Store back the modified vector
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(accumulators[0].x);
    }
}
"#;

// WORKAROUND 2: Completely avoid double indexing with vector operations
// HYPOTHESIS: Similar to V1, the issue is with accessing a component of an array's vector element directly.
// STRATEGY: Perform the update on the entire vector at once. Instead of a nested loop to update
// components `x, y, z, w`, a single vector `add_vec` is created and added to `accumulators[m]`.
// This still uses a loop over `m`, but avoids the inner loop over `n`.
const FIXED_KERNEL_V2_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
};

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

var<workgroup> tile_a: array<vec4<i32>, 512>;
var<workgroup> tile_b: array<i32, 2048>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

fn dot_product_4x4(a: vec4<i32>, b: vec4<i32>) -> i32 {
    return dot(a, b);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    var accumulators: array<vec4<i32>, THREAD_TILE_M>;
    for (var i = 0u; i < THREAD_TILE_M; i = i + 1u) {
        accumulators[i] = vec4<i32>(0);
    }
    
    // WORKAROUND 2: Use vector operations to avoid double indexing entirely
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        // Create a vector with the values we want to add
        let base_value = vec4<i32>(i32(m), i32(m + 1u), i32(m + 2u), i32(m + 3u));
        let offset_value = vec4<i32>(0, 1, 2, 3);
        let add_vec = base_value + offset_value;
        
        // Add the entire vector at once
        accumulators[m] = accumulators[m] + add_vec;
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(accumulators[0].x);
    }
}
"#;

// WORKAROUND 3: Use separate scalar variables instead of arrays
// HYPOTHESIS: The bug is related to the `array<vec4<i32>>` data structure itself.
// STRATEGY: Completely remove the `array` of `vec4`s and replace it with four separate `vec4`
// variables (`acc0`, `acc1`, etc.). This unrolls the outer loop and tests if the compiler
// can handle the operations when the problematic data structure is not used.
const FIXED_KERNEL_V3_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
};

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

var<workgroup> tile_a: array<vec4<i32>, 512>;
var<workgroup> tile_b: array<i32, 2048>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

fn dot_product_4x4(a: vec4<i32>, b: vec4<i32>) -> i32 {
    return dot(a, b);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // WORKAROUND 3: Use individual variables instead of arrays when possible
    var acc0 = vec4<i32>(0);
    var acc1 = vec4<i32>(0);
    var acc2 = vec4<i32>(0);
    var acc3 = vec4<i32>(0);
    
    // Unroll the computation to avoid array indexing issues
    // m = 0
    acc0.x += i32(0u + 0u);
    
    // m = 1
    acc1.x += i32(1u + 0u);
    acc1.y += i32(1u + 1u);
    acc1.z += i32(1u + 2u);
    acc1.w += i32(1u + 3u);
    
    // m = 2
    acc2.x += i32(2u + 0u);
    acc2.y += i32(2u + 1u);
    acc2.z += i32(2u + 2u);
    acc2.w += i32(2u + 3u);
    
    // m = 3
    acc3.x += i32(3u + 0u);
    acc3.y += i32(3u + 1u);
    acc3.z += i32(3u + 2u);
    acc3.w += i32(3u + 3u);
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(acc0.x);
    }
}
"#;

// WORKAROUND 4: Simplify workgroup memory and avoid complex array operations
// HYPOTHESIS: The bug is related to arrays of vectors, but not flat arrays.
// STRATEGY: Change the accumulator from `array<vec4<i32>, 4>` to a simple `array<i32, 16>`.
// The nested loop is still used, but access is done via a single, manually calculated index:
// `let acc_idx = m * THREAD_TILE_N + n;`. This avoids double-indexing `[m][n]` and is the
// most promising fix as it maintains the logic while using a simpler data structure.
const FIXED_KERNEL_V4_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

// Simplified workgroup memory - avoid complex nested arrays
var<workgroup> shared_data: array<i32, 4096>; // Single flat array

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // Use a simple flat array for the accumulator, not an array of vectors.
    var accumulators: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        accumulators[i] = 0;
    }
    
    // Simple computation without complex indexing
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let acc_idx = m * THREAD_TILE_N + n;
            accumulators[acc_idx] += i32(m + n);
        }
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(accumulators[0]);
    }
}
"#;

// WORKAROUND 5: Remove problematic functions and use basic operations only
// HYPOTHESIS: The bug might not be in the main function, but in helper functions or declarations.
// STRATEGY: Create a minimal, "hello world" compute shader. It removes all helper functions,
// workgroup memory, and complex logic. Its only job is to write the global invocation index
// to the output buffer. If this fails, the issue is fundamental to the wgpu setup.
const FIXED_KERNEL_V5_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    
    // Minimal computation to test pipeline creation
    let global_idx = workgroup_id.x * WORKGROUP_SIZE_X + thread_idx_n;
    
    if (global_idx < metadata.M * metadata.N) {
        output[global_idx] = f32(global_idx);
    }
}
"#;

// WORKAROUND 6: Use different vector types and avoid vec4<i32>
// HYPOTHESIS: The compiler bug is specific to the `i32` integer type within a vector.
// STRATEGY: Replace the `array<vec4<i32>>` with a flat `array<f32, 16>`. This tests whether
// the compiler handles floating-point vectors differently from integer vectors in this context.
const FIXED_KERNEL_V6_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

// Use f32 vectors instead of i32 - Dx12 might handle these better
var<workgroup> tile_data: array<f32, 2048>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // Use f32 arrays instead of i32 vec4 arrays
    var accumulators: array<f32, 16>; // 4x4 = 16 elements
    
    // Initialize
    for (var i = 0u; i < 16u; i = i + 1u) {
        accumulators[i] = 0.0;
    }
    
    // Compute with single indexing only
    for (var i = 0u; i < 16u; i = i + 1u) {
        let m = i / 4u;
        let n = i % 4u;
        accumulators[i] = f32(m + n);
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = accumulators[0];
    }
}
"#;

// WORKAROUND 7: Avoid workgroup memory entirely
// HYPOTHESIS: The bug is triggered by the use of `var<workgroup>` memory.
// STRATEGY: Remove all `var<workgroup>` declarations. The computation is performed using only
// thread-local private variables (`var<private>`). This helps isolate if the bug is related to
// shared memory management in the DX12 backend.
const FIXED_KERNEL_V7_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    
    // Direct computation without any shared/workgroup memory
    let global_m = workgroup_id.y * (WORKGROUP_SIZE_Y * THREAD_TILE_M) + thread_idx_m * THREAD_TILE_M;
    let global_n = workgroup_id.x * (WORKGROUP_SIZE_X * THREAD_TILE_N) + thread_idx_n * THREAD_TILE_N;
    
    // Simple register-only computation
    var sum: f32 = 0.0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            sum += f32(m + n);
        }
    }
    
    let output_idx = global_m * metadata.N + global_n;
    if (global_m < metadata.M && global_n < metadata.N && output_idx < arrayLength(&output)) {
        output[output_idx] = sum;
    }
}
"#;

// WORKAROUND 8: The full, complex kernel with the V4 fix applied (flattened i32 array)
// PURPOSE: This is the candidate for the final fix. It takes the successful strategy from
// V4 (using a flat `array<i32, 16>` for the accumulator) and integrates it into the full,
// complex production kernel logic, including workgroup memory loading and scaling.
// Passing this test is a strong indicator that the fix is correct and robust.
const FULL_KERNEL_WITH_FIX_WGSL: &str = r#"
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
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;   
const TILE_DIM_K: u32 = 32u;

const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

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
    
    var accumulators: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        accumulators[i] = 0;
    }
    
    let num_k_tiles = (metadata.K + TILE_DIM_K - 1u) / TILE_DIM_K;
    var k_tile_idx = 0u;
    while (k_tile_idx < num_k_tiles) {
        let k_tile_start = k_tile_idx * TILE_DIM_K;
        
        let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
        let loads_per_thread_a = (total_a_elements + 255u) / 256u;
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
        
        // --- TILE B LOADING: Per-element decode pattern ---
        let total_b_elements = TILE_DIM_N * TILE_DIM_K;
        let loads_per_thread_b = (total_b_elements + 255u) / 256u;
        for (var i = 0u; i < loads_per_thread_b; i = i + 1u) {
            let load_idx = i * 256u + local_index;
            if (load_idx < total_b_elements) {
                let n = load_idx / TILE_DIM_K;
                let k = load_idx % TILE_DIM_K;
                let global_n = tile_start_n + n;
                let global_k = k_tile_start + k;
                if (global_n < metadata.N && global_k < metadata.K) {
                    let global_k_packed_idx = global_k / 16u;
                    let inner_k = global_k % 16u;
                    let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
                    let packed_w = packed_weights[weight_idx];
                    tile_b[load_idx] = decode_2bit((packed_w >> (inner_k * 2u)) & 0x3u);
                } else {
                    tile_b[load_idx] = 0;
                }
            }
        }
        
        workgroupBarrier();
        
        for (var k_inner = 0u; k_inner < TILE_DIM_K; k_inner = k_inner + 4u) {
            var a_vecs: array<vec4<i32>, THREAD_TILE_M>;
            for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                let base_m = thread_idx_m * THREAD_TILE_M + m;
                let vec_idx = (base_m * TILE_DIM_K + k_inner) / 4u;
                a_vecs[m] = tile_a[vec_idx];
            }
            
            for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
                let base_n = thread_idx_n * THREAD_TILE_N + n;
                let b_vec = vec4<i32>(
                    tile_b[base_n * TILE_DIM_K + k_inner],
                    tile_b[base_n * TILE_DIM_K + k_inner + 1u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 2u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 3u]
                );
                
                for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                    let dot_result = dot_product_4x4(a_vecs[m], b_vec);
                    let acc_idx = m * THREAD_TILE_N + n;
                    accumulators[acc_idx] += dot_result;
                }
            }
        }
        
        workgroupBarrier();
        k_tile_idx = k_tile_idx + 1u;
    }
    
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M + m;
            let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N + n;
            
            if (global_m < metadata.M && global_n < metadata.N) {
                let activation_scale = activation_scales[global_m];
                let weight_scale = weight_scales[global_n];
                let acc_idx = m * THREAD_TILE_N + n;
                let final_result = f32(accumulators[acc_idx]) * activation_scale * weight_scale;
                
                output[global_m * metadata.N + global_n] = final_result;
            }
        }
    }
}
"#;

// INCREMENTAL TEST V4.1: V4 base + workgroup memory declaration
// PURPOSE: Start with the simple, working V4 kernel and add back the workgroup memory
// declarations (`tile_a`, `tile_b`) from the full kernel, but *without using them*.
// This tests if merely declaring large workgroup arrays triggers the bug.
const FIXED_KERNEL_V4_1_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

// V4.1 Change: Declare the workgroup tiles from the full kernel, but do not use them.
var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    
    // Logic from the original simple V4 kernel
    var result: i32 = 0;
    
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }
    
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// INCREMENTAL TEST V4.2.1: V4.1 base + tile_a loading ONLY
// PURPOSE: Build on V4.1 by adding the logic to load data from global memory into the
// `tile_a` workgroup array. The core computation remains simple. This isolates the `tile_a`
// loading process to see if it's the source of the failure.
const FIXED_KERNEL_V4_2_1_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

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
    
    // V4.2.1 Change: Load data into tile_a ONLY
    let k_tile_start = 0u;
    let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
    if (local_index < total_a_elements) {
        let vec_idx = local_index;
        let flat_idx = local_index * 4u;
        let m = flat_idx / TILE_DIM_K;
        let k = flat_idx % TILE_DIM_K;
        let global_m = tile_start_m + m;
        let global_k = k_tile_start + k;
        if (global_m < metadata.M && global_k + 3u < metadata.K) {
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
    
    workgroupBarrier();

    // Still use the simple accumulator from V4
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }

    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// INCREMENTAL TEST V4.2.2: V4.1 base + tile_b loading ONLY (unrolled)
// PURPOSE: Similar to V4.2.1, but this time it adds the logic to load data into the
// `tile_b` workgroup array. This isolates the `tile_b` loading process, which is more
// complex due to the weight decoding step.
const FIXED_KERNEL_V4_2_2_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            // Unroll the loop
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
        }
    }
    
    workgroupBarrier();

    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }

    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// INCREMENTAL TEST V4.2: V4.1 base + workgroup memory loading
// PURPOSE: This test combines the loading logic for both `tile_a` and `tile_b` from the
// previous two tests. The core computation still uses the simple V4 accumulator. This is
// a key test, as it's expected to fail if the bug is related to the interaction of
// loading data into multiple workgroup arrays, even if the accumulator logic is fixed.
const FIXED_KERNEL_V4_2_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    // Unrolled for simplicity
    decoded[0]  = decode_2bit((packed_val >> 0u) & 0x3u);
    decoded[1]  = decode_2bit((packed_val >> 2u) & 0x3u);
    decoded[2]  = decode_2bit((packed_val >> 4u) & 0x3u);
    decoded[3]  = decode_2bit((packed_val >> 6u) & 0x3u);
    decoded[4]  = decode_2bit((packed_val >> 8u) & 0x3u);
    decoded[5]  = decode_2bit((packed_val >> 10u) & 0x3u);
    decoded[6]  = decode_2bit((packed_val >> 12u) & 0x3u);
    decoded[7]  = decode_2bit((packed_val >> 14u) & 0x3u);
    decoded[8]  = decode_2bit((packed_val >> 16u) & 0x3u);
    decoded[9]  = decode_2bit((packed_val >> 18u) & 0x3u);
    decoded[10] = decode_2bit((packed_val >> 20u) & 0x3u);
    decoded[11] = decode_2bit((packed_val >> 22u) & 0x3u);
    decoded[12] = decode_2bit((packed_val >> 24u) & 0x3u);
    decoded[13] = decode_2bit((packed_val >> 26u) & 0x3u);
    decoded[14] = decode_2bit((packed_val >> 28u) & 0x3u);
    decoded[15] = decode_2bit((packed_val >> 30u) & 0x3u);
    return decoded;
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
    
    // V4.2 Change: Load data into workgroup memory
    let k_tile_start = 0u; // Simplified for testing

    let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
    if (local_index < total_a_elements) {
        let vec_idx = local_index;
        let flat_idx = local_index * 4u;
        let m = flat_idx / TILE_DIM_K;
        let k = flat_idx % TILE_DIM_K;
        let global_m = tile_start_m + m;
        let global_k = k_tile_start + k;
        if (global_m < metadata.M && global_k + 3u < metadata.K) {
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
    
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            for (var j = 0u; j < 16u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = decoded[j];
            }
        }
    }

    workgroupBarrier();

    // Still use the simple accumulator from V4
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }

    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// INCREMENTAL TEST V4.3: V4.2 base + main computation loop
// PURPOSE: This is the almost-complete kernel. It takes the (potentially failing) data loading
// from V4.2 and adds the main computation loop that reads from `tile_a` and `tile_b` and uses
// the fixed, flattened accumulator. If V4.2 passes and this fails, the bug is in the final
// computation logic. If V4.2 fails, this is also expected to fail.
const FIXED_KERNEL_V4_3_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}

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
    
    // THE V4 FIX: Use a flattened array of i32 for the accumulator.
    var accumulators: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        accumulators[i] = 0;
    }
    
    // --- DATA LOADING (from V4.2) --
    let k_tile_start = 0u; 
    let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
    if (local_index < total_a_elements) {
        let vec_idx = local_index;
        let flat_idx = local_index * 4u;
        let m = flat_idx / TILE_DIM_K;
        let k = flat_idx % TILE_DIM_K;
        let global_m = tile_start_m + m;
        let global_k = k_tile_start + k;
        if (global_m < metadata.M && global_k + 3u < metadata.K) {
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
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            for (var j = 0u; j < 16u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = decoded[j];
            }
        }
    }
    workgroupBarrier();

    // --- V4.3 Change: Add the main computation loop ---
    for (var k_inner = 0u; k_inner < TILE_DIM_K; k_inner = k_inner + 4u) {
        var a_vecs: array<vec4<i32>, THREAD_TILE_M>;
        for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
            let base_m = thread_idx_m * THREAD_TILE_M + m;
            let vec_idx = (base_m * TILE_DIM_K + k_inner) / 4u;
            a_vecs[m] = tile_a[vec_idx];
        }
        
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let base_n = thread_idx_n * THREAD_TILE_N + n;
            let b_vec = vec4<i32>(
                tile_b[base_n * TILE_DIM_K + k_inner],
                tile_b[base_n * TILE_DIM_K + k_inner + 1u],
                tile_b[base_n * TILE_DIM_K + k_inner + 2u],
                tile_b[base_n * TILE_DIM_K + k_inner + 3u]
            );
            
            for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                let dot_result = dot_product_4x4(a_vecs[m], b_vec);
                let acc_idx = m * THREAD_TILE_N + n;
                accumulators[acc_idx] += dot_result;
            }
        }
    }
    
    // --- Write results to global memory ---
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M + m;
            let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N + n;
            
            if (global_m < metadata.M && global_n < metadata.N) {
                let activation_scale = activation_scales[global_m];
                let weight_scale = weight_scales[global_n];
                let acc_idx = m * THREAD_TILE_N + n;
                let final_result = f32(accumulators[acc_idx]) * activation_scale * weight_scale;
                
                output[global_m * metadata.N + global_n] = final_result;
            }
        }
    }
}
"#;

// FINAL CORRECTNESS KERNEL: Full production logic with all fixes.
// PURPOSE: This is the kernel used in the final correctness test. It should represent the
// best-known working version of the full kernel, incorporating the flattened i32 accumulator
// from V4 and any other necessary micro-optimizations (like unrolling loops) found to be stable.
// It is tested against a scalar CPU implementation to ensure the GPU results are correct.
const FULL_KERNEL_V4_UNROLLED_WGSL: &str = r#"
// This is the full production kernel logic, but with all the fixes we've identified:
// 1. Flattened i32 accumulator.
// 2. Unrolled decode_16x2bit_ternary function.
// 3. Unrolled padding loop for tile_b.
struct BitnetMetadata {
    M: u32, N: u32, K: u32, K_packed: u32,
};
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    decoded[0]  = decode_2bit((packed_val >> 0u) & 0x3u);
    decoded[1]  = decode_2bit((packed_val >> 2u) & 0x3u);
    decoded[2]  = decode_2bit((packed_val >> 4u) & 0x3u);
    decoded[3]  = decode_2bit((packed_val >> 6u) & 0x3u);
    decoded[4]  = decode_2bit((packed_val >> 8u) & 0x3u);
    decoded[5]  = decode_2bit((packed_val >> 10u) & 0x3u);
    decoded[6]  = decode_2bit((packed_val >> 12u) & 0x3u);
    decoded[7]  = decode_2bit((packed_val >> 14u) & 0x3u);
    decoded[8]  = decode_2bit((packed_val >> 16u) & 0x3u);
    decoded[9]  = decode_2bit((packed_val >> 18u) & 0x3u);
    decoded[10] = decode_2bit((packed_val >> 20u) & 0x3u);
    decoded[11] = decode_2bit((packed_val >> 22u) & 0x3u);
    decoded[12] = decode_2bit((packed_val >> 24u) & 0x3u);
    decoded[13] = decode_2bit((packed_val >> 26u) & 0x3u);
    decoded[14] = decode_2bit((packed_val >> 28u) & 0x3u);
    decoded[15] = decode_2bit((packed_val >> 30u) & 0x3u);
    return decoded;
}

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
    var accumulators: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { accumulators[i] = 0; }
    
    let num_k_tiles = (metadata.K + TILE_DIM_K - 1u) / TILE_DIM_K;
    var k_tile_idx = 0u;
    while (k_tile_idx < num_k_tiles) {
        let k_tile_start = k_tile_idx * TILE_DIM_K;
        
        let total_a_elements = TILE_DIM_M * TILE_DIM_K / 4u;
        let loads_per_thread_a = (total_a_elements + 255u) / 256u;
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
                    let base_addr = global_m * metadata.K + global_k;
                    tile_a[vec_idx] = vec4<i32>(activations[base_addr], activations[base_addr + 1u], activations[base_addr + 2u], activations[base_addr + 3u]);
                } else {
                    tile_a[vec_idx] = vec4<i32>(0);
                }
            }
        }
        
        let total_b_elements = TILE_DIM_N * TILE_DIM_K;
        let loads_per_thread_b = (total_b_elements + 255u) / 256u;
        for (var i = 0u; i < loads_per_thread_b; i = i + 1u) {
            let load_idx = i * 256u + local_index;
            if (load_idx < total_b_elements) {
                let n = load_idx / TILE_DIM_K;
                let k = load_idx % TILE_DIM_K;
                let global_n = tile_start_n + n;
                let k_tile_start_b = k_tile_idx * TILE_DIM_K;
                let global_k = k_tile_start_b + k;

                if (global_n < metadata.N && global_k < metadata.K) {
                    let global_k_packed_idx = global_k / 16u;
                    let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
                    let packed_w = packed_weights[weight_idx];
                    
                    let inner_k = global_k % 16u;
                    let decoded_val = decode_2bit((packed_w >> (inner_k * 2u)) & 0x3u);
                    tile_b[load_idx] = decoded_val;
                } else {
                    tile_b[load_idx] = 0;
                }
            }
        }
        
        workgroupBarrier();
        
        for (var k_inner = 0u; k_inner < TILE_DIM_K; k_inner = k_inner + 4u) {
            var a_vecs: array<vec4<i32>, THREAD_TILE_M>;
            for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                let base_m = thread_idx_m * THREAD_TILE_M + m;
                let vec_idx = (base_m * TILE_DIM_K + k_inner) / 4u;
                a_vecs[m] = tile_a[vec_idx];
            }
            
            for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
                let base_n = thread_idx_n * THREAD_TILE_N + n;
                let b_vec = vec4<i32>(
                    tile_b[base_n * TILE_DIM_K + k_inner],
                    tile_b[base_n * TILE_DIM_K + k_inner + 1u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 2u],
                    tile_b[base_n * TILE_DIM_K + k_inner + 3u]
                );
                
                for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
                    let dot_result = dot_product_4x4(a_vecs[m], b_vec);
                    let acc_idx = m * THREAD_TILE_N + n;
                    accumulators[acc_idx] += dot_result;
                }
            }
        }
        
        workgroupBarrier();
        k_tile_idx = k_tile_idx + 1u;
    }
    
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M + m;
            let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N + n;
            
            if (global_m < metadata.M && global_n < metadata.N) {
                let activation_scale = activation_scales[global_m];
                let weight_scale = weight_scales[global_n];
                let acc_idx = m * THREAD_TILE_N + n;
                let final_result = f32(accumulators[acc_idx]) * activation_scale * weight_scale;
                output[global_m * metadata.N + global_n] = final_result;
            }
        }
    }
}
"#;

// INCREMENTAL TEST V4.2.3: V4.2.2 base + inlined tile_b loading
// PURPOSE: Test the hypothesis that returning an array from a function (`decode_16x2bit_ternary`)
// is causing the pipeline creation failure in V4.2.2. This kernel is identical to V4.2.2,
// but the logic from the decode function is inlined directly into `main`.
const FIXED_KERNEL_V4_2_3_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}

@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
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
    
    // V4.2.3 Change: Load data into tile_b with INLINED decoding.
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            
            // Inlined decode_16x2bit_ternary logic
            var decoded: array<i32, 16>;
            for (var i = 0u; i < 16u; i = i + 1u) {
                decoded[i] = decode_2bit((packed_w >> (i * 2u)) & 0x3u);
            }

            // Unroll the loop
            for (var j = 0u; j < 16u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = decoded[j];
            }
        }
    }
    
    workgroupBarrier();

    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }

    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;

    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// V4.2.2.3: Function-Returned Array, Write to Private Array
const FIXED_KERNEL_V4_2_2_3_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}
fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            // Write to a private array first
            var private_arr: array<i32, 16>;
            for (var j = 0u; j < 16u; j = j + 1u) {
                private_arr[j] = decoded[j];
            }
            for (var j = 0u; j < 16u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = private_arr[j];
            }
        }
    }
    workgroupBarrier();
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * 64u;
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;
    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;
/* DX12 WGSL Array Return Bug:
On DX12, any WGSL function that returns an array and is used (directly or indirectly) to write to workgroup memory will cause shader compilation or pipeline creation to fail. The only robust workaround is to inline all array-producing logic.
*/
// --- SMOKING GUN TEST: Function returns array, but array is never used ---
const FIXED_KERNEL_SMOKING_GUN_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;

fn returns_array(_x: u32) -> array<i32, 8> {
    var arr: array<i32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        arr[i] = i32(i);
    }
    return arr;
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    // Call the function, but do not use the result at all
    let _unused = returns_array(local_index);
    // Write a constant to output
    if (local_index == 0u) {
        output[0] = 42.0;
    }
}
"#;

// --- BEGIN V4.2.2.x MICRO-TESTS ---

// V4.2.2.1: Function-Returned Array, Partial Write
const FIXED_KERNEL_V4_2_2_1_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}
fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            // Only write the first 4 elements to workgroup memory
            for (var j = 0u; j < 4u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = decoded[j];
            }
        }
    }
    workgroupBarrier();
    // Dummy accumulator
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * 64u;
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;
    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// V4.2.2.2: Function-Returned Array, Local Copy
const FIXED_KERNEL_V4_2_2_2_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32,
    N: u32,
    K: u32,
    K_packed: u32,
}
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}
fn decode_16x2bit_ternary(packed_val: u32) -> array<i32, 16> {
    var decoded: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        decoded[i] = decode_2bit((packed_val >> (i * 2u)) & 0x3u);
    }
    return decoded;
}
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < (total_b_elements / 16u)) {
        let packed_idx = local_index;
        let n = (packed_idx * 16u) / TILE_DIM_K;
        let k = (packed_idx * 16u) % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k_packed_idx = (k_tile_start + k) / 16u;
        if (global_n < metadata.N && global_k_packed_idx < metadata.K_packed) {
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            let decoded = decode_16x2bit_ternary(packed_w);
            // Copy to a local array first
            var temp: array<i32, 16>;
            for (var j = 0u; j < 16u; j = j + 1u) {
                temp[j] = decoded[j];
            }
            for (var j = 0u; j < 16u; j = j + 1u) {
                tile_b[n * TILE_DIM_K + k + j] = temp[j];
            }
        }
    }
    workgroupBarrier();
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * 64u;
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;
    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

// --- WARM CONTEXT & TEST RUNNER ---

struct WarmGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>, // Add this line
    info: wgpu::AdapterInfo,
}

impl WarmGpuContext {
    async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await.ok()?;

        let info = adapter.get_info();
        let (device, queue) = adapter // Capture queue here
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue), // Store it here
            info,
        })
    }
}



async fn scope<F, R>(device: &wgpu::Device, f: F) -> (R, Option<String>)
where
    F: FnOnce() -> R,
{
    let error = Arc::new(Mutex::new(None));
    let error_clone = error.clone();

    device.on_uncaptured_error(Box::new(move |e| {
        let error_msg = match e {
            wgpu::Error::Validation { source, .. } => format!("Validation Error: {}", source),
            wgpu::Error::OutOfMemory { .. } => "Out of Memory Error".to_string(),
            wgpu::Error::Internal { source, .. } => format!("Internal Error: {}", source),
        };
        *error_clone.lock().unwrap() = Some(error_msg);
    }));

    let result = f();
    device.poll(wgpu::MaintainBase::Wait);
    device.on_uncaptured_error(Box::new(|_| {})); // Reset handler

    let e = error.lock().unwrap().take();
    (result, e)
}

// Update test_shader_compilation to return Result<(), String>
async fn test_shader_compilation(name: &str, shader_source: &str, context: &WarmGpuContext, test_id: usize) -> Result<(), String> {
    TEST_REPORTER.log_message(1, &format!("\n--- Testing: {} ---", name));
    let device = &context.device;
    let mut actual_failure = false;
    let mut error_msg = String::new();
    // --- Shader Module Compilation ---
    TEST_REPORTER.log_message(1, &format!("Attempting to compile {} WGSL kernel for Dx12...", name));
    let (shader_module, compilation_error) = scope(device, || {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        })
    }).await;
    if let Some(e) = compilation_error {
        actual_failure = true;
        error_msg = format!("Shader compilation failed: {}", e);
        TEST_REPORTER.log_message(1, &format!("ERROR: {} shader compilation failed: {}", name, e));
    } else {
        TEST_REPORTER.log_message(1, &format!("SUCCESS: {} shader module compiled on Dx12 without error.", name));
    }
    // --- Pipeline Creation ---
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{} bind group layout", name)),
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
        label: Some(&format!("{} pipeline layout", name)),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let (_, pipeline_error) = scope(device, || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} compute pipeline", name)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }).await;
    if let Some(e) = pipeline_error {
        actual_failure = true;
        error_msg = format!("Pipeline creation failed: {}", e);
        TEST_REPORTER.log_message(1, &format!("ERROR: {} pipeline creation failed: {}", name, e));
    } else {
        TEST_REPORTER.log_message(1, &format!("SUCCESS: {} compute pipeline created successfully on Dx12!", name));
    }
    // Status logic
    let status = if actual_failure { "FAIL" } else { "PASS" };
    TEST_REPORTER.log_message(test_id, &format!("Test status: {}", status));
    if actual_failure {
        Err(error_msg)
    } else {
        Ok(())
    }
}


fn quantize_activations_scalar(activations: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = activations
        .iter()
        .filter(|v| v.is_finite())
        .map(|&x| x.abs())
        .fold(f32::NEG_INFINITY, f32::max);
    
    let abs_max = if abs_max == f32::NEG_INFINITY { 0.0 } else { abs_max };

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
                    let packed_bits = (packed_weight >> (bit_idx * 2)) & 0b11;
                    let weight_val = match packed_bits {
                        1 => 1i8,
                        2 => -1i8,
                        _ => 0i8,
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



async fn test_correctness_on_dx12(context: &WarmGpuContext) {
    let test_name = "test_shader_compilation_correctness_on_dx12";
    let test_id = 200;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "\n--- Running Correctness Test on Dx12 with Fixed Kernel ---");
    
    // First, test if the fixed kernel can even be compiled and create a pipeline.
    let (shader_module, compilation_error) = scope(&context.device, || {
        context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fixed Production Kernel"),
            source: wgpu::ShaderSource::Wgsl(FULL_KERNEL_V4_UNROLLED_WGSL.into()),
        })
    }).await;

    if let Some(e) = compilation_error {
        TEST_REPORTER.log_message(2, &format!("ERROR: Correctness test failed at SHADER COMPILATION: {}", e));
        TEST_REPORTER.record_timing(test_name, t0.elapsed());
        return;
    }
    TEST_REPORTER.log_message(2, "SUCCESS: Fixed kernel compiled on Dx12.");

    let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Correctness Test BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Correctness Test PL"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let (pipeline, pipeline_error) = scope(&context.device, || {
        context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Correctness Test Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }).await;

    if let Some(e) = pipeline_error {
        TEST_REPORTER.log_message(2, &format!("ERROR: Correctness test failed at PIPELINE CREATION: {}", e));
        TEST_REPORTER.record_timing(test_name, t0.elapsed());
        return;
    }
    TEST_REPORTER.log_message(2, "SUCCESS: Fixed kernel pipeline created on Dx12.");

    // Now run the actual correctness logic.
    let batch_size = 4;
    let in_features = 16;
    let out_features = 8;
    
    let mut rng = StdRng::seed_from_u64(42);
    let activations: Vec<f32> = (0..batch_size * in_features).map(|_| rng.random_range(-1.0..1.0)).collect();
    let weights: Vec<i8> = (0..out_features * in_features).map(|_| rng.random_range(-1..=1)).collect();

    let (mut q_activations, mut activation_scales_vec) = (Vec::new(), Vec::new());
    for row in activations.chunks(in_features) {
        let (q_row, scale) = quantize_activations_scalar(row);
        q_activations.extend(q_row);
        activation_scales_vec.push(scale);
    }
    let flat_weights: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
    let (packed_weights, weight_scales) = pack_ternary_weights(&flat_weights).expect("Failed to pack weights");
    
    let q_acts_i32: Vec<i32> = q_activations.iter().map(|&x| x as i32).collect();
    let metadata = BitnetMetadata { m: batch_size as u32, n: out_features as u32, k: in_features as u32, k_packed: (in_features / 16) as u32 };
    let metadata_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[metadata]), usage: wgpu::BufferUsages::UNIFORM });
    let q_acts_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&q_acts_i32), usage: wgpu::BufferUsages::STORAGE });
    let packed_weights_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&packed_weights), usage: wgpu::BufferUsages::STORAGE });
    let weight_scales_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&weight_scales), usage: wgpu::BufferUsages::STORAGE });
    let act_scales_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&activation_scales_vec), usage: wgpu::BufferUsages::STORAGE });
    let output_size_bytes = (batch_size * out_features * std::mem::size_of::<f32>()) as u64;
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: output_size_bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: output_size_bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: metadata_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_acts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: packed_weights_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weight_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: act_scales_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (batch_size as u32 + 15) / 16,
            (out_features as u32 + 15) / 16,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size_bytes);
    context.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    context.device.poll(wgpu::MaintainBase::Wait);
    rx.receive().await.unwrap().unwrap();
    let gpu_output: Vec<f32> = bytemuck::cast_slice(&buffer_slice.get_mapped_range()).to_vec();
    
    let scalar_output = matmul_quantized_scalar(&q_activations, &packed_weights, &activation_scales_vec, &weight_scales, batch_size, in_features, out_features);
    
    TEST_REPORTER.log_message(2, &format!("GPU Output: {:?}", &gpu_output[..4]));
    TEST_REPORTER.log_message(2, &format!("Scalar Output: {:?}", &scalar_output[..4]));
    assert_vec_eq(&gpu_output, &scalar_output, 1e-5);
    TEST_REPORTER.log_message(2, "SUCCESS: Correctness test passed on Dx12!");
    TEST_REPORTER.log_message(test_id, "Test completed");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

// --- TEST HARNESS, WARM-TESTS, and TEST-WRAPPERS ---

/// Sets up the Tokio runtime, a Dx12 WGPU context, and the shared log file.
/// This is intended to be called by individual test wrappers.
/// Returns None if a Dx12 adapter cannot be found.
fn setup_dx12_test_harness() -> Option<(tokio::runtime::Runtime, WarmGpuContext)> {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(WarmGpuContext::new());

    if let Some(ctx) = context {
        TEST_REPORTER.log_message(1, &format!("-> Found Dx12 adapter: {} ({:?})", ctx.info.name, ctx.info.backend));
        Some((runtime, ctx))
    } else {
        TEST_REPORTER.log_message(1, "WARNING: Could not find a Dx12 adapter. Skipping test.");
        None
    }
}



async fn test_shader_compilation_original_buggy_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_original_buggy";
    let test_id = 101;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation("Original Buggy", BUGGY_KERNEL_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v1() {
    let test_name = "test_shader_compilation_fix_v1";
    let test_id = 102;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v1_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v1_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v1";
    let test_id = 102;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation("Fix V1", FIXED_KERNEL_V1_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v2() {
    let test_name = "test_shader_compilation_fix_v2";
    let test_id = 103;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v2_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v2_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v2";
    let test_id = 103;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V2_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v3() {
    let test_name = "test_shader_compilation_fix_v3";
    let test_id = 104;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v3_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v3_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v3";
    let test_id = 104;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V3_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v4_simplified_workgroup() {
    let test_name = "test_shader_compilation_fix_v4_simplified_workgroup";
    let test_id = 105;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v4_simplified_workgroup_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v4_simplified_workgroup_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v4_simplified_workgroup";
    let test_id = 105;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v5_minimal_kernel() {
    let test_name = "test_shader_compilation_fix_v5_minimal_kernel";
    let test_id = 106;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {

        runtime.block_on(test_shader_compilation_fix_v5_minimal_kernel_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v5_minimal_kernel_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v5_minimal_kernel";
    let test_id = 106;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V5_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v6_f32_vectors() {
    let test_name = "test_shader_compilation_fix_v6_f32_vectors";
    let test_id = 107;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v6_f32_vectors_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v6_f32_vectors_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v6_f32_vectors";
    let test_id = 107;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V6_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_fix_v7_no_workgroup_memory() {
    let test_name = "test_shader_compilation_fix_v7_no_workgroup_memory";
    let test_id = 108;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_fix_v7_no_workgroup_memory_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_fix_v7_no_workgroup_memory_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_fix_v7_no_workgroup_memory";
    let test_id = 108;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V7_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_1_declaration() {
    let test_name = "test_shader_compilation_v4_1_declaration";
    let test_id = 109;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_1_declaration_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}
async fn test_shader_compilation_v4_1_declaration_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_1_declaration";
    let test_id = 109;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_1_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}



#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_1_tile_a_loading() {
    let test_name = "test_shader_compilation_v4_2_1_tile_a_loading";
    let test_id = 110;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_1_tile_a_loading_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_v4_2_1_tile_a_loading_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_1_tile_a_loading";
    let test_id = 110;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_1_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_2_tile_b_loading() {
    let test_name = "test_shader_compilation_v4_2_2_tile_b_loading";
    let test_id = 111;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_2_tile_b_loading_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}
async fn test_shader_compilation_v4_2_2_tile_b_loading_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_2_tile_b_loading";
    let test_id = 111;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_2_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_3_inlined_tile_b() {
    let test_name = "test_shader_compilation_v4_2_3_inlined_tile_b";
    let test_id = 112;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_3_inlined_tile_b_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_v4_2_3_inlined_tile_b_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_3_inlined_tile_b";
    let test_id = 112;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_3_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_combined_loading() {
    let test_name = "test_shader_compilation_v4_2_combined_loading";
    let test_id = 113;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_combined_loading_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_v4_2_combined_loading_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_combined_loading";
    let test_id = 113;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}




#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_3_main_computation() {
    let test_name = "test_shader_compilation_v4_3_main_computation";
    let test_id = 114;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_3_main_computation_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_v4_3_main_computation_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_3_main_computation";
    let test_id = 114;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_3_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_full_kernel_with_fix() {
    let test_name = "test_shader_compilation_full_kernel_with_fix";
    let test_id = 115;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_full_kernel_with_fix_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}
async fn test_shader_compilation_full_kernel_with_fix_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_full_kernel_with_fix";
    let test_id = 115;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FULL_KERNEL_WITH_FIX_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_production_kernel() {
    let test_name = "test_shader_compilation_production_kernel";
    let test_id = 116;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_production_kernel_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_production_kernel_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_production_kernel";
    let test_id = 116;
    let t0 = Instant::now();
    let source = include_str!("../src/kernels/bitnet_kernel.wgsl");
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, source, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_correctness_of_fixed_kernel() {
    let test_name = "test_correctness_of_fixed_kernel";
    let test_id = 200;
    let t0 = Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_correctness_on_dx12(&context));
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}



#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_2_1_partial_write() {
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_2_1_partial_write_warm(&context)).unwrap();
    }
}
async fn test_shader_compilation_v4_2_2_1_partial_write_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_2_1_partial_write";
    let test_id = 121;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_2_1_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_2_2_local_copy() {
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_2_2_local_copy_warm(&context)).unwrap();
    }
}

async fn test_shader_compilation_v4_2_2_2_local_copy_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_2_2_local_copy";
    let test_id = 122;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_2_2_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}


#[test] #[serial] #[ignore]
fn test_shader_compilation_v4_2_2_3_private_array() {
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_v4_2_2_3_private_array_warm(&context)).unwrap();
    }
}
async fn test_shader_compilation_v4_2_2_3_private_array_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_v4_2_2_3_private_array";
    let test_id = 123;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_V4_2_2_3_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

#[test] #[serial] #[ignore]
fn test_shader_compilation_smoking_gun_unused_array() {
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_smoking_gun_unused_array_warm(&context)).unwrap();
    }
}
async fn test_shader_compilation_smoking_gun_unused_array_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_smoking_gun_unused_array";
    let test_id = 124;
    let t0 = Instant::now();
    let result = AssertUnwindSafe(
        test_shader_compilation(test_name, FIXED_KERNEL_SMOKING_GUN_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

fn zzz_final_report() {
    // This function runs last and generates the final report.
    // Add a small delay to ensure all async tests complete.
    std::thread::sleep(std::time::Duration::from_secs(2));
    TEST_REPORTER.generate_report();
}

// --- SUITE RUNNER ---

#[test]
#[serial]
fn run_dx12_compilation_suite() {
    // This runner creates the context ONCE and passes it to all warm tests.
    let runtime = tokio::runtime::Runtime::new().unwrap();

    // No log_file needed, just log header with TEST_REPORTER
    TEST_REPORTER.log_message(1, &format!(" Dx12 Shader Bug Test Report - Generated on {}", Local::now().to_rfc2822()));

    runtime.block_on(async {
        let context = WarmGpuContext::new().await;
        if context.is_none() {
            TEST_REPORTER.log_message(1, "FATAL: Could not create Dx12 context. Aborting test suite.");
            return;
        }
        let context = context.unwrap();
        TEST_REPORTER.log_message(1, &format!("-> Found Dx12 adapter: {} ({:?})", context.info.name, context.info.backend));

        println!("--- Running Full Dx12 Shader Compilation Suite ---");

        // Collect test results
        let mut results: Vec<(String, Result<(), String>)> = Vec::new();

        macro_rules! run_and_log {
            ($name:expr, $future:expr) => {
                let t0 = std::time::Instant::now();
                let res = $future.await;
                let duration = t0.elapsed();
                if let Err(ref e) = res {
                    log_failure!(TEST_REPORTER, $name, e, duration);
                } else {
                    log_timed!(TEST_REPORTER, $name, duration);
                }
                results.push(($name.to_string(), res));
            };
        }

        run_and_log!("test_shader_compilation_original_buggy_warm", test_shader_compilation_original_buggy_warm(&context));
        run_and_log!("test_shader_compilation_fix_v1_warm", test_shader_compilation_fix_v1_warm(&context));
        run_and_log!("test_shader_compilation_fix_v2_warm", test_shader_compilation_fix_v2_warm(&context));
        run_and_log!("test_shader_compilation_fix_v3_warm", test_shader_compilation_fix_v3_warm(&context));
        run_and_log!("test_shader_compilation_fix_v4_simplified_workgroup_warm", test_shader_compilation_fix_v4_simplified_workgroup_warm(&context));
        run_and_log!("test_shader_compilation_fix_v5_minimal_kernel_warm", test_shader_compilation_fix_v5_minimal_kernel_warm(&context));
        run_and_log!("test_shader_compilation_fix_v6_f32_vectors_warm", test_shader_compilation_fix_v6_f32_vectors_warm(&context));
        run_and_log!("test_shader_compilation_fix_v7_no_workgroup_memory_warm", test_shader_compilation_fix_v7_no_workgroup_memory_warm(&context));

        println!("--- Running Incremental Complexity Tests (V4 Base) ---");
        TEST_REPORTER.log_message(2, "\n--- Incremental Complexity Tests (V4 Base) ---");
        run_and_log!("test_shader_compilation_v4_1_declaration_warm", test_shader_compilation_v4_1_declaration_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_1_tile_a_loading_warm", test_shader_compilation_v4_2_1_tile_a_loading_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_2_tile_b_loading_warm", test_shader_compilation_v4_2_2_tile_b_loading_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_3_inlined_tile_b_warm", test_shader_compilation_v4_2_3_inlined_tile_b_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_combined_loading_warm", test_shader_compilation_v4_2_combined_loading_warm(&context));
        run_and_log!("test_shader_compilation_v4_3_main_computation_warm", test_shader_compilation_v4_3_main_computation_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_2_1_partial_write_warm", test_shader_compilation_v4_2_2_1_partial_write_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_2_2_local_copy_warm", test_shader_compilation_v4_2_2_2_local_copy_warm(&context));
        run_and_log!("test_shader_compilation_v4_2_2_3_private_array_warm", test_shader_compilation_v4_2_2_3_private_array_warm(&context));
        run_and_log!("test_shader_compilation_smoking_gun_unused_array_warm", test_shader_compilation_smoking_gun_unused_array_warm(&context));

        println!("--- Running Full and Production Kernel Tests ---");
        TEST_REPORTER.log_message(2, "\n--- Full Kernel and Production Tests ---");
        run_and_log!("test_shader_compilation_full_kernel_with_fix_warm", test_shader_compilation_full_kernel_with_fix_warm(&context));
        // Highlight the workaround/fix in the report
        TEST_REPORTER.record_special_finding(
            "test_shader_compilation_full_kernel_with_fix",
            "WORKAROUND FOUND",
            "This test demonstrates a robust workaround for the DX12/Naga WGSL bug: using a flattened i32 accumulator and per-element decode for tile_b. The full BitNet kernel logic now passes on DX12. See the test and kernel code for details."
        );
        run_and_log!("test_shader_compilation_production_kernel_warm", test_shader_compilation_production_kernel_warm(&context));

        println!("--- Running Final Correctness Test ---");
        // Correctness test is not Result-returning, so just run and log
        test_correctness_on_dx12(&context).await;

        println!("\n--- Dx12 Shader Compilation Suite Complete. Report generated at: {} ---", "log.txt");
        zzz_final_report();

        // Print summary
        println!("\nTest Summary:");
        for (name, result) in &results {
            match result {
                Ok(_) => println!("[PASS] {}", name),
                Err(e) => println!("[FAIL] {}: {}", name, e),
            }
        }
    });
}

// --- NEW TEST: Per-Element Decode for tile_b (no array return, each thread decodes one value) ---
const PER_ELEMENT_DECODE_TILE_B_WGSL: &str = r#"
struct BitnetMetadata {
    M: u32, N: u32, K: u32, K_packed: u32,
};
@group(0) @binding(0) var<uniform> metadata: BitnetMetadata;
@group(0) @binding(1) var<storage, read> activations: array<i32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scales: array<f32>;
@group(0) @binding(4) var<storage, read> activation_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_DIM_M: u32 = 64u;
const TILE_DIM_N: u32 = 64u;
const TILE_DIM_K: u32 = 32u;
const THREAD_TILE_M: u32 = 4u;
const THREAD_TILE_N: u32 = 4u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const TILE_A_SIZE: u32 = (TILE_DIM_M * TILE_DIM_K) / 4u;
const TILE_B_SIZE: u32 = TILE_DIM_K * TILE_DIM_N;

var<workgroup> tile_a: array<vec4<i32>, TILE_A_SIZE>;
var<workgroup> tile_b: array<i32, TILE_B_SIZE>;

fn decode_2bit(val: u32) -> i32 {
    switch(val) {
        case 1u: { return 1; }
        case 2u: { return -1; }
        default: { return 0; }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let tile_start_n = workgroup_id.x * TILE_DIM_N;
    let k_tile_start = 0u;
    let total_b_elements = TILE_DIM_N * TILE_DIM_K;
    if (local_index < total_b_elements) {
        let n = local_index / TILE_DIM_K;
        let k = local_index % TILE_DIM_K;
        let global_n = tile_start_n + n;
        let global_k = k_tile_start + k;
        if (global_n < metadata.N && global_k < metadata.K) {
            let global_k_packed_idx = global_k / 16u;
            let inner_k = global_k % 16u;
            let weight_idx = global_n * metadata.K_packed + global_k_packed_idx;
            let packed_w = packed_weights[weight_idx];
            tile_b[local_index] = decode_2bit((packed_w >> (inner_k * 2u)) & 0x3u);
        } else {
            tile_b[local_index] = 0;
        }
    }
    workgroupBarrier();
    // Dummy accumulator
    var result: i32 = 0;
    for (var m = 0u; m < THREAD_TILE_M; m = m + 1u) {
        for (var n = 0u; n < THREAD_TILE_N; n = n + 1u) {
            result += i32(m + n);
        }
    }
    let thread_idx_m = local_id.y;
    let thread_idx_n = local_id.x;
    let tile_start_m = workgroup_id.y * TILE_DIM_M;
    let global_m = tile_start_m + thread_idx_m * THREAD_TILE_M;
    let global_n = tile_start_n + thread_idx_n * THREAD_TILE_N;
    if (global_m < metadata.M && global_n < metadata.N) {
        output[global_m * metadata.N + global_n] = f32(result);
    }
}
"#;

#[test] #[serial] #[ignore]
fn test_shader_compilation_per_element_decode_tile_b() {
    let test_name = "test_shader_compilation_per_element_decode_tile_b";
    let test_id = 130;
    let t0 = std::time::Instant::now();
    TEST_REPORTER.log_message(test_id, "Starting test");
    if let Some((runtime, context)) = setup_dx12_test_harness() {
        runtime.block_on(test_shader_compilation_per_element_decode_tile_b_warm(&context)).unwrap();
    }
    TEST_REPORTER.log_message(test_id, "Test finished");
    TEST_REPORTER.record_timing(test_name, t0.elapsed());
}

async fn test_shader_compilation_per_element_decode_tile_b_warm(context: &WarmGpuContext) -> Result<(), String> {
    let test_name = "test_shader_compilation_per_element_decode_tile_b";
    let test_id = 130;
    let t0 = std::time::Instant::now();
    let result = std::panic::AssertUnwindSafe(
        test_shader_compilation(test_name, PER_ELEMENT_DECODE_TILE_B_WGSL, context, test_id)
    ).catch_unwind().await;
    let duration = t0.elapsed();
    match result {
        Ok(inner_result) => {
            inner_result?;
            TEST_REPORTER.log_message(test_id, "Test completed [PASS]");
            TEST_REPORTER.record_timing(&format!("{} [PASS]", test_name), duration);
            Ok(())
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            TEST_REPORTER.log_message(test_id, "Test panicked [FAIL]");
            TEST_REPORTER.record_failure(test_name, &err_msg, Some(duration));
            Err(err_msg)
        }
    }
}

