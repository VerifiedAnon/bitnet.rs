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
        case 1u: { return 1; }   // 01
        case 2u: { return -1; }  // 10
        default: { return 0; }   // 00 or 11
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
    
    var k_tile_idx = 0u;
    while (k_tile_idx < num_k_tiles) {
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

        k_tile_idx = k_tile_idx + 1u;
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