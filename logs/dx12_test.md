# DX12_TEST Test Report

> Generated on: 2025-06-30 22:55:01

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Shader Compilation Correctness On Dx12             | ✅ Pass |  11.80 sec |             |
|  2 | Shader Compilation Fix V1 [PASS]                   | ✅ Pass |   24.00 ms |             |
|  3 | Shader Compilation Fix V1 Warm                     | ✅ Pass |   24.00 ms |             |
|  4 | Shader Compilation Fix V2 [PASS]                   | ✅ Pass |   16.00 ms |             |
|  5 | Shader Compilation Fix V2 Warm                     | ✅ Pass |   16.00 ms |             |
|  6 | Shader Compilation Fix V3 [PASS]                   | ✅ Pass |   11.00 ms |             |
|  7 | Shader Compilation Fix V3 Warm                     | ✅ Pass |   11.00 ms |             |
|  8 | Shader Compilation Fix V4 Simplified Workgroup [PASS] | ✅ Pass |   21.00 ms |             |
|  9 | Shader Compilation Fix V4 Simplified Workgroup Warm | ✅ Pass |   21.00 ms |             |
| 10 | Shader Compilation Fix V5 Minimal Kernel [PASS]    | ✅ Pass |    6.00 ms |             |
| 11 | Shader Compilation Fix V5 Minimal Kernel Warm      | ✅ Pass |    6.00 ms |             |
| 12 | Shader Compilation Fix V6 F32 Vectors [PASS]       | ✅ Pass |   15.00 ms |             |
| 13 | Shader Compilation Fix V6 F32 Vectors Warm         | ✅ Pass |   15.00 ms |             |
| 14 | Shader Compilation Fix V7 No Workgroup Memory [PASS] | ✅ Pass |   13.00 ms |             |
| 15 | Shader Compilation Fix V7 No Workgroup Memory Warm | ✅ Pass |   13.00 ms |             |
| 16 | Shader Compilation Full Kernel With Fix [PASS]     | ✅ Pass |  11.81 sec |             |
| 17 | Shader Compilation Full Kernel With Fix Warm       | ✅ Pass |  11.81 sec |             |
| 18 | Shader Compilation Original Buggy Warm             | ❌ Fail |    3.00 ms | Pipeline creation failed: Validation Error: In ... |
| 19 | Shader Compilation Production Kernel [PASS]        | ✅ Pass |  11.88 sec |             |
| 20 | Shader Compilation Production Kernel Warm          | ✅ Pass |  11.88 sec |             |
| 21 | Shader Compilation Smoking Gun Unused Array [PASS] | ✅ Pass |    7.00 ms |             |
| 22 | Shader Compilation Smoking Gun Unused Array Warm   | ✅ Pass |    7.00 ms |             |
| 23 | Shader Compilation V4 1 Declaration [PASS]         | ✅ Pass |   16.00 ms |             |
| 24 | Shader Compilation V4 1 Declaration Warm           | ✅ Pass |   16.00 ms |             |
| 25 | Shader Compilation V4 2 1 Tile A Loading [PASS]    | ✅ Pass |   20.00 ms |             |
| 26 | Shader Compilation V4 2 1 Tile A Loading Warm      | ✅ Pass |   20.00 ms |             |
| 27 | Shader Compilation V4 2 2 1 Partial Write [PASS]   | ✅ Pass |   26.00 ms |             |
| 28 | Shader Compilation V4 2 2 1 Partial Write Warm     | ✅ Pass |   26.00 ms |             |
| 29 | Shader Compilation V4 2 2 2 Local Copy [PASS]      | ✅ Pass |   29.00 ms |             |
| 30 | Shader Compilation V4 2 2 2 Local Copy Warm        | ✅ Pass |   29.00 ms |             |
| 31 | Shader Compilation V4 2 2 3 Private Array [PASS]   | ✅ Pass |   28.00 ms |             |
| 32 | Shader Compilation V4 2 2 3 Private Array Warm     | ✅ Pass |   28.00 ms |             |
| 33 | Shader Compilation V4 2 2 Tile B Loading Warm      | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 34 | Shader Compilation V4 2 3 Inlined Tile B [PASS]    | ✅ Pass |   28.00 ms |             |
| 35 | Shader Compilation V4 2 3 Inlined Tile B Warm      | ✅ Pass |   28.00 ms |             |
| 36 | Shader Compilation V4 2 Combined Loading [PASS]    | ✅ Pass |   32.00 ms |             |
| 37 | Shader Compilation V4 2 Combined Loading Warm      | ✅ Pass |   33.00 ms |             |
| 38 | Shader Compilation V4 3 Main Computation [PASS]    | ✅ Pass |  11.92 sec |             |
| 39 | Shader Compilation V4 3 Main Computation Warm      | ✅ Pass |  11.92 sec |             |

## ⭐ Special Finding

**[WORKAROUND FOUND]**: `test_shader_compilation_full_kernel_with_fix`  
This test demonstrates a robust workaround for the DX12/Naga WGSL bug: using a flattened i32 accumulator and per-element decode for tile_b. The full BitNet kernel logic now passes on DX12. See the test and kernel code for details.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-30, 22:54:12.257] -> [FAIL] test_shader_compilation_original_buggy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-06-30, 22:54:12.282] -> test_shader_compilation_fix_v1_warm      took     24.680 ms
[2025-06-30, 22:54:12.299] -> test_shader_compilation_fix_v2_warm      took     16.861 ms
[2025-06-30, 22:54:12.310] -> test_shader_compilation_fix_v3_warm      took     11.509 ms
[2025-06-30, 22:54:12.331] -> test_shader_compilation_fix_v4_simplified_workgroup_warm took     21.103 ms
[2025-06-30, 22:54:12.338] -> test_shader_compilation_fix_v5_minimal_kernel_warm took      6.293 ms
[2025-06-30, 22:54:12.353] -> test_shader_compilation_fix_v6_f32_vectors_warm took     15.155 ms
[2025-06-30, 22:54:12.366] -> test_shader_compilation_fix_v7_no_workgroup_memory_warm took     13.701 ms
[2025-06-30, 22:54:12.383] -> test_shader_compilation_v4_1_declaration_warm took     16.320 ms
[2025-06-30, 22:54:12.404] -> test_shader_compilation_v4_2_1_tile_a_loading_warm took     20.728 ms
[2025-06-30, 22:54:12.408] -> [FAIL] test_shader_compilation_v4_2_2_tile_b_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-06-30, 22:54:12.437] -> test_shader_compilation_v4_2_3_inlined_tile_b_warm took     28.732 ms
[2025-06-30, 22:54:12.470] -> test_shader_compilation_v4_2_combined_loading_warm took     33.013 ms
[2025-06-30, 22:54:24.386] -> test_shader_compilation_v4_3_main_computation_warm took  11916.783 ms
[2025-06-30, 22:54:24.413] -> test_shader_compilation_v4_2_2_1_partial_write_warm took     26.524 ms
[2025-06-30, 22:54:24.442] -> test_shader_compilation_v4_2_2_2_local_copy_warm took     29.417 ms
[2025-06-30, 22:54:24.471] -> test_shader_compilation_v4_2_2_3_private_array_warm took     28.849 ms
[2025-06-30, 22:54:24.479] -> test_shader_compilation_smoking_gun_unused_array_warm took      7.361 ms
[2025-06-30, 22:54:36.291] -> test_shader_compilation_full_kernel_with_fix_warm took  11812.510 ms
[2025-06-30, 22:54:48.174] -> test_shader_compilation_production_kernel_warm took  11882.677 ms
[2025-06-30, 22:54:11.788] ->  Dx12 Shader Bug Test Report - Generated on Mon, 30 Jun 2025 22:54:11 -0400
[2025-06-30, 22:54:12.254] -> -> Found Dx12 adapter: NVIDIA GeForce RTX 2070 SUPER (Dx12)
[2025-06-30, 22:54:12.254] -> 
--- Testing: Original Buggy ---
[2025-06-30, 22:54:12.254] -> Attempting to compile Original Buggy WGSL kernel for Dx12...
[2025-06-30, 22:54:12.256] -> ERROR: Original Buggy shader compilation failed: Validation Error: In Device::create_shader_module, label = 'Original Buggy'
[2025-06-30, 22:54:12.257] -> ERROR: Original Buggy pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-06-30, 22:54:12.257] -> 
--- Testing: Fix V1 ---
[2025-06-30, 22:54:12.257] -> Attempting to compile Fix V1 WGSL kernel for Dx12...
[2025-06-30, 22:54:12.260] -> SUCCESS: Fix V1 shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.281] -> SUCCESS: Fix V1 compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.282] -> 
--- Testing: test_shader_compilation_fix_v2 ---
[2025-06-30, 22:54:12.282] -> Attempting to compile test_shader_compilation_fix_v2 WGSL kernel for Dx12...
[2025-06-30, 22:54:12.284] -> SUCCESS: test_shader_compilation_fix_v2 shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.298] -> SUCCESS: test_shader_compilation_fix_v2 compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.299] -> 
--- Testing: test_shader_compilation_fix_v3 ---
[2025-06-30, 22:54:12.299] -> Attempting to compile test_shader_compilation_fix_v3 WGSL kernel for Dx12...
[2025-06-30, 22:54:12.301] -> SUCCESS: test_shader_compilation_fix_v3 shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.310] -> SUCCESS: test_shader_compilation_fix_v3 compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.310] -> 
--- Testing: test_shader_compilation_fix_v4_simplified_workgroup ---
[2025-06-30, 22:54:12.310] -> Attempting to compile test_shader_compilation_fix_v4_simplified_workgroup WGSL kernel for Dx12...
[2025-06-30, 22:54:12.312] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.331] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.331] -> 
--- Testing: test_shader_compilation_fix_v5_minimal_kernel ---
[2025-06-30, 22:54:12.331] -> Attempting to compile test_shader_compilation_fix_v5_minimal_kernel WGSL kernel for Dx12...
[2025-06-30, 22:54:12.332] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.337] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.338] -> 
--- Testing: test_shader_compilation_fix_v6_f32_vectors ---
[2025-06-30, 22:54:12.338] -> Attempting to compile test_shader_compilation_fix_v6_f32_vectors WGSL kernel for Dx12...
[2025-06-30, 22:54:12.339] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.352] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.353] -> 
--- Testing: test_shader_compilation_fix_v7_no_workgroup_memory ---
[2025-06-30, 22:54:12.353] -> Attempting to compile test_shader_compilation_fix_v7_no_workgroup_memory WGSL kernel for Dx12...
[2025-06-30, 22:54:12.354] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.366] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.366] -> 
--- Testing: test_shader_compilation_v4_1_declaration ---
[2025-06-30, 22:54:12.366] -> Attempting to compile test_shader_compilation_v4_1_declaration WGSL kernel for Dx12...
[2025-06-30, 22:54:12.368] -> SUCCESS: test_shader_compilation_v4_1_declaration shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.383] -> SUCCESS: test_shader_compilation_v4_1_declaration compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.383] -> 
--- Testing: test_shader_compilation_v4_2_1_tile_a_loading ---
[2025-06-30, 22:54:12.383] -> Attempting to compile test_shader_compilation_v4_2_1_tile_a_loading WGSL kernel for Dx12...
[2025-06-30, 22:54:12.386] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.403] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.404] -> 
--- Testing: test_shader_compilation_v4_2_2_tile_b_loading ---
[2025-06-30, 22:54:12.404] -> Attempting to compile test_shader_compilation_v4_2_2_tile_b_loading WGSL kernel for Dx12...
[2025-06-30, 22:54:12.408] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading shader compilation failed: Validation Error: In Device::create_shader_module, label = 'test_shader_compilation_v4_2_2_tile_b_loading'
[2025-06-30, 22:54:12.408] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-06-30, 22:54:12.408] -> 
--- Testing: test_shader_compilation_v4_2_3_inlined_tile_b ---
[2025-06-30, 22:54:12.408] -> Attempting to compile test_shader_compilation_v4_2_3_inlined_tile_b WGSL kernel for Dx12...
[2025-06-30, 22:54:12.411] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.436] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.437] -> 
--- Testing: test_shader_compilation_v4_2_combined_loading ---
[2025-06-30, 22:54:12.437] -> Attempting to compile test_shader_compilation_v4_2_combined_loading WGSL kernel for Dx12...
[2025-06-30, 22:54:12.442] -> SUCCESS: test_shader_compilation_v4_2_combined_loading shader module compiled on Dx12 without error.
[2025-06-30, 22:54:12.469] -> SUCCESS: test_shader_compilation_v4_2_combined_loading compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.470] -> 
--- Testing: test_shader_compilation_v4_3_main_computation ---
[2025-06-30, 22:54:12.470] -> Attempting to compile test_shader_compilation_v4_3_main_computation WGSL kernel for Dx12...
[2025-06-30, 22:54:12.476] -> SUCCESS: test_shader_compilation_v4_3_main_computation shader module compiled on Dx12 without error.
[2025-06-30, 22:54:24.386] -> SUCCESS: test_shader_compilation_v4_3_main_computation compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:24.386] -> 
--- Testing: test_shader_compilation_v4_2_2_1_partial_write ---
[2025-06-30, 22:54:24.387] -> Attempting to compile test_shader_compilation_v4_2_2_1_partial_write WGSL kernel for Dx12...
[2025-06-30, 22:54:24.390] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write shader module compiled on Dx12 without error.
[2025-06-30, 22:54:24.413] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:24.413] -> 
--- Testing: test_shader_compilation_v4_2_2_2_local_copy ---
[2025-06-30, 22:54:24.413] -> Attempting to compile test_shader_compilation_v4_2_2_2_local_copy WGSL kernel for Dx12...
[2025-06-30, 22:54:24.416] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy shader module compiled on Dx12 without error.
[2025-06-30, 22:54:24.442] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:24.442] -> 
--- Testing: test_shader_compilation_v4_2_2_3_private_array ---
[2025-06-30, 22:54:24.442] -> Attempting to compile test_shader_compilation_v4_2_2_3_private_array WGSL kernel for Dx12...
[2025-06-30, 22:54:24.446] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array shader module compiled on Dx12 without error.
[2025-06-30, 22:54:24.471] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:24.471] -> 
--- Testing: test_shader_compilation_smoking_gun_unused_array ---
[2025-06-30, 22:54:24.471] -> Attempting to compile test_shader_compilation_smoking_gun_unused_array WGSL kernel for Dx12...
[2025-06-30, 22:54:24.472] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array shader module compiled on Dx12 without error.
[2025-06-30, 22:54:24.478] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:24.479] -> 
--- Testing: test_shader_compilation_full_kernel_with_fix ---
[2025-06-30, 22:54:24.479] -> Attempting to compile test_shader_compilation_full_kernel_with_fix WGSL kernel for Dx12...
[2025-06-30, 22:54:24.485] -> SUCCESS: test_shader_compilation_full_kernel_with_fix shader module compiled on Dx12 without error.
[2025-06-30, 22:54:36.291] -> SUCCESS: test_shader_compilation_full_kernel_with_fix compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:36.291] -> 
--- Testing: test_shader_compilation_production_kernel ---
[2025-06-30, 22:54:36.291] -> Attempting to compile test_shader_compilation_production_kernel WGSL kernel for Dx12...
[2025-06-30, 22:54:36.301] -> SUCCESS: test_shader_compilation_production_kernel shader module compiled on Dx12 without error.
[2025-06-30, 22:54:48.174] -> SUCCESS: test_shader_compilation_production_kernel compute pipeline created successfully on Dx12!
[2025-06-30, 22:54:12.366] -> 
--- Incremental Complexity Tests (V4 Base) ---
[2025-06-30, 22:54:24.479] -> 
--- Full Kernel and Production Tests ---
[2025-06-30, 22:54:48.182] -> SUCCESS: Fixed kernel compiled on Dx12.
[2025-06-30, 22:54:59.958] -> SUCCESS: Fixed kernel pipeline created on Dx12.
[2025-06-30, 22:54:59.974] -> GPU Output: [0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 22:54:59.974] -> Scalar Output: [0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 22:54:59.974] -> SUCCESS: Correctness test passed on Dx12!
[2025-06-30, 22:54:12.257] -> Test status: FAIL
[2025-06-30, 22:54:12.281] -> Test status: PASS
[2025-06-30, 22:54:12.282] -> Test completed [PASS]
[2025-06-30, 22:54:12.298] -> Test status: PASS
[2025-06-30, 22:54:12.298] -> Test completed [PASS]
[2025-06-30, 22:54:12.310] -> Test status: PASS
[2025-06-30, 22:54:12.310] -> Test completed [PASS]
[2025-06-30, 22:54:12.331] -> Test status: PASS
[2025-06-30, 22:54:12.331] -> Test completed [PASS]
[2025-06-30, 22:54:12.337] -> Test status: PASS
[2025-06-30, 22:54:12.337] -> Test completed [PASS]
[2025-06-30, 22:54:12.352] -> Test status: PASS
[2025-06-30, 22:54:12.353] -> Test completed [PASS]
[2025-06-30, 22:54:12.366] -> Test status: PASS
[2025-06-30, 22:54:12.366] -> Test completed [PASS]
[2025-06-30, 22:54:12.383] -> Test status: PASS
[2025-06-30, 22:54:12.383] -> Test completed [PASS]
[2025-06-30, 22:54:12.403] -> Test status: PASS
[2025-06-30, 22:54:12.403] -> Test completed [PASS]
[2025-06-30, 22:54:12.408] -> Test status: FAIL
[2025-06-30, 22:54:12.436] -> Test status: PASS
[2025-06-30, 22:54:12.437] -> Test completed [PASS]
[2025-06-30, 22:54:12.469] -> Test status: PASS
[2025-06-30, 22:54:12.470] -> Test completed [PASS]
[2025-06-30, 22:54:24.386] -> Test status: PASS
[2025-06-30, 22:54:24.386] -> Test completed [PASS]
[2025-06-30, 22:54:36.291] -> Test status: PASS
[2025-06-30, 22:54:36.291] -> Test completed [PASS]
[2025-06-30, 22:54:48.174] -> Test status: PASS
[2025-06-30, 22:54:48.174] -> Test completed [PASS]
[2025-06-30, 22:54:24.413] -> Test status: PASS
[2025-06-30, 22:54:24.413] -> Test completed [PASS]
[2025-06-30, 22:54:24.442] -> Test status: PASS
[2025-06-30, 22:54:24.442] -> Test completed [PASS]
[2025-06-30, 22:54:24.471] -> Test status: PASS
[2025-06-30, 22:54:24.471] -> Test completed [PASS]
[2025-06-30, 22:54:24.479] -> Test status: PASS
[2025-06-30, 22:54:24.479] -> Test completed [PASS]
[2025-06-30, 22:54:48.174] -> 
--- Running Correctness Test on Dx12 with Fixed Kernel ---
[2025-06-30, 22:54:59.974] -> Test completed
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 39
- **Passed:** 37
- **Failed:** 2

### Timing Information

- **Total Time:** 83.63 sec
- **Average Time:** 2144.00 ms

### Status

❌ 2 test(s) failed. See above for details.

---

_Report generated by BitNet Test Framework_
