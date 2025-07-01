# DX12_TEST Test Report

> Generated on: 2025-07-01 16:20:01

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Shader Compilation Correctness On Dx12             | ✅ Pass |  11.94 sec |             |
|  2 | Shader Compilation Fix V1 [PASS]                   | ✅ Pass |   36.00 ms |             |
|  3 | Shader Compilation Fix V1 Warm                     | ✅ Pass |   36.00 ms |             |
|  4 | Shader Compilation Fix V2 [PASS]                   | ✅ Pass |   21.00 ms |             |
|  5 | Shader Compilation Fix V2 Warm                     | ✅ Pass |   21.00 ms |             |
|  6 | Shader Compilation Fix V3 [PASS]                   | ✅ Pass |   16.00 ms |             |
|  7 | Shader Compilation Fix V3 Warm                     | ✅ Pass |   16.00 ms |             |
|  8 | Shader Compilation Fix V4 Simplified Workgroup [PASS] | ✅ Pass |   28.00 ms |             |
|  9 | Shader Compilation Fix V4 Simplified Workgroup Warm | ✅ Pass |   28.00 ms |             |
| 10 | Shader Compilation Fix V5 Minimal Kernel [PASS]    | ✅ Pass |   11.00 ms |             |
| 11 | Shader Compilation Fix V5 Minimal Kernel Warm      | ✅ Pass |   11.00 ms |             |
| 12 | Shader Compilation Fix V6 F32 Vectors [PASS]       | ✅ Pass |   21.00 ms |             |
| 13 | Shader Compilation Fix V6 F32 Vectors Warm         | ✅ Pass |   21.00 ms |             |
| 14 | Shader Compilation Fix V7 No Workgroup Memory [PASS] | ✅ Pass |   20.00 ms |             |
| 15 | Shader Compilation Fix V7 No Workgroup Memory Warm | ✅ Pass |   20.00 ms |             |
| 16 | Shader Compilation Full Kernel With Fix [PASS]     | ✅ Pass |  11.94 sec |             |
| 17 | Shader Compilation Full Kernel With Fix Warm       | ✅ Pass |  11.94 sec |             |
| 18 | Shader Compilation Original Buggy Warm             | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 19 | Shader Compilation Production Kernel [PASS]        | ✅ Pass |  12.06 sec |             |
| 20 | Shader Compilation Production Kernel Warm          | ✅ Pass |  12.06 sec |             |
| 21 | Shader Compilation Smoking Gun Unused Array [PASS] | ✅ Pass |   11.00 ms |             |
| 22 | Shader Compilation Smoking Gun Unused Array Warm   | ✅ Pass |   11.00 ms |             |
| 23 | Shader Compilation V4 1 Declaration [PASS]         | ✅ Pass |   22.00 ms |             |
| 24 | Shader Compilation V4 1 Declaration Warm           | ✅ Pass |   22.00 ms |             |
| 25 | Shader Compilation V4 2 1 Tile A Loading [PASS]    | ✅ Pass |   22.00 ms |             |
| 26 | Shader Compilation V4 2 1 Tile A Loading Warm      | ✅ Pass |   22.00 ms |             |
| 27 | Shader Compilation V4 2 2 1 Partial Write [PASS]   | ✅ Pass |   32.00 ms |             |
| 28 | Shader Compilation V4 2 2 1 Partial Write Warm     | ✅ Pass |   32.00 ms |             |
| 29 | Shader Compilation V4 2 2 2 Local Copy [PASS]      | ✅ Pass |   30.00 ms |             |
| 30 | Shader Compilation V4 2 2 2 Local Copy Warm        | ✅ Pass |   30.00 ms |             |
| 31 | Shader Compilation V4 2 2 3 Private Array [PASS]   | ✅ Pass |   30.00 ms |             |
| 32 | Shader Compilation V4 2 2 3 Private Array Warm     | ✅ Pass |   31.00 ms |             |
| 33 | Shader Compilation V4 2 2 Tile B Loading Warm      | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
| 34 | Shader Compilation V4 2 3 Inlined Tile B [PASS]    | ✅ Pass |   35.00 ms |             |
| 35 | Shader Compilation V4 2 3 Inlined Tile B Warm      | ✅ Pass |   35.00 ms |             |
| 36 | Shader Compilation V4 2 Combined Loading [PASS]    | ✅ Pass |   39.00 ms |             |
| 37 | Shader Compilation V4 2 Combined Loading Warm      | ✅ Pass |   39.00 ms |             |
| 38 | Shader Compilation V4 3 Main Computation [PASS]    | ✅ Pass |  12.04 sec |             |
| 39 | Shader Compilation V4 3 Main Computation Warm      | ✅ Pass |  12.04 sec |             |

## ⭐ Special Finding

**[WORKAROUND FOUND]**: `test_shader_compilation_full_kernel_with_fix`  
This test demonstrates a robust workaround for the DX12/Naga WGSL bug: using a flattened i32 accumulator and per-element decode for tile_b. The full BitNet kernel logic now passes on DX12. See the test and kernel code for details.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-07-01, 16:19:11.323] -> [FAIL] test_shader_compilation_original_buggy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-07-01, 16:19:11.360] -> test_shader_compilation_fix_v1_warm      took     36.958 ms
[2025-07-01, 16:19:11.382] -> test_shader_compilation_fix_v2_warm      took     21.874 ms
[2025-07-01, 16:19:11.399] -> test_shader_compilation_fix_v3_warm      took     16.836 ms
[2025-07-01, 16:19:11.428] -> test_shader_compilation_fix_v4_simplified_workgroup_warm took     28.276 ms
[2025-07-01, 16:19:11.439] -> test_shader_compilation_fix_v5_minimal_kernel_warm took     11.552 ms
[2025-07-01, 16:19:11.461] -> test_shader_compilation_fix_v6_f32_vectors_warm took     21.506 ms
[2025-07-01, 16:19:11.481] -> test_shader_compilation_fix_v7_no_workgroup_memory_warm took     20.193 ms
[2025-07-01, 16:19:11.504] -> test_shader_compilation_v4_1_declaration_warm took     22.309 ms
[2025-07-01, 16:19:11.526] -> test_shader_compilation_v4_2_1_tile_a_loading_warm took     22.147 ms
[2025-07-01, 16:19:11.532] -> [FAIL] test_shader_compilation_v4_2_2_tile_b_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-07-01, 16:19:11.568] -> test_shader_compilation_v4_2_3_inlined_tile_b_warm took     35.924 ms
[2025-07-01, 16:19:11.608] -> test_shader_compilation_v4_2_combined_loading_warm took     39.752 ms
[2025-07-01, 16:19:23.645] -> test_shader_compilation_v4_3_main_computation_warm took  12037.085 ms
[2025-07-01, 16:19:23.678] -> test_shader_compilation_v4_2_2_1_partial_write_warm took     32.821 ms
[2025-07-01, 16:19:23.708] -> test_shader_compilation_v4_2_2_2_local_copy_warm took     30.346 ms
[2025-07-01, 16:19:23.739] -> test_shader_compilation_v4_2_2_3_private_array_warm took     31.107 ms
[2025-07-01, 16:19:23.751] -> test_shader_compilation_smoking_gun_unused_array_warm took     11.654 ms
[2025-07-01, 16:19:35.692] -> test_shader_compilation_full_kernel_with_fix_warm took  11940.199 ms
[2025-07-01, 16:19:47.757] -> test_shader_compilation_production_kernel_warm took  12064.962 ms
[2025-07-01, 16:19:10.861] ->  Dx12 Shader Bug Test Report - Generated on Tue, 1 Jul 2025 16:19:10 -0400
[2025-07-01, 16:19:11.319] -> -> Found Dx12 adapter: NVIDIA GeForce RTX 2070 SUPER (Dx12)
[2025-07-01, 16:19:11.319] -> 
--- Testing: Original Buggy ---
[2025-07-01, 16:19:11.319] -> Attempting to compile Original Buggy WGSL kernel for Dx12...
[2025-07-01, 16:19:11.322] -> ERROR: Original Buggy shader compilation failed: Validation Error: In Device::create_shader_module, label = 'Original Buggy'
[2025-07-01, 16:19:11.323] -> ERROR: Original Buggy pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-07-01, 16:19:11.323] -> 
--- Testing: Fix V1 ---
[2025-07-01, 16:19:11.323] -> Attempting to compile Fix V1 WGSL kernel for Dx12...
[2025-07-01, 16:19:11.326] -> SUCCESS: Fix V1 shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.359] -> SUCCESS: Fix V1 compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.360] -> 
--- Testing: test_shader_compilation_fix_v2 ---
[2025-07-01, 16:19:11.361] -> Attempting to compile test_shader_compilation_fix_v2 WGSL kernel for Dx12...
[2025-07-01, 16:19:11.364] -> SUCCESS: test_shader_compilation_fix_v2 shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.382] -> SUCCESS: test_shader_compilation_fix_v2 compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.382] -> 
--- Testing: test_shader_compilation_fix_v3 ---
[2025-07-01, 16:19:11.383] -> Attempting to compile test_shader_compilation_fix_v3 WGSL kernel for Dx12...
[2025-07-01, 16:19:11.386] -> SUCCESS: test_shader_compilation_fix_v3 shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.399] -> SUCCESS: test_shader_compilation_fix_v3 compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.399] -> 
--- Testing: test_shader_compilation_fix_v4_simplified_workgroup ---
[2025-07-01, 16:19:11.400] -> Attempting to compile test_shader_compilation_fix_v4_simplified_workgroup WGSL kernel for Dx12...
[2025-07-01, 16:19:11.402] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.427] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.428] -> 
--- Testing: test_shader_compilation_fix_v5_minimal_kernel ---
[2025-07-01, 16:19:11.428] -> Attempting to compile test_shader_compilation_fix_v5_minimal_kernel WGSL kernel for Dx12...
[2025-07-01, 16:19:11.429] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.439] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.440] -> 
--- Testing: test_shader_compilation_fix_v6_f32_vectors ---
[2025-07-01, 16:19:11.440] -> Attempting to compile test_shader_compilation_fix_v6_f32_vectors WGSL kernel for Dx12...
[2025-07-01, 16:19:11.442] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.460] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.461] -> 
--- Testing: test_shader_compilation_fix_v7_no_workgroup_memory ---
[2025-07-01, 16:19:11.461] -> Attempting to compile test_shader_compilation_fix_v7_no_workgroup_memory WGSL kernel for Dx12...
[2025-07-01, 16:19:11.463] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.481] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.482] -> 
--- Testing: test_shader_compilation_v4_1_declaration ---
[2025-07-01, 16:19:11.482] -> Attempting to compile test_shader_compilation_v4_1_declaration WGSL kernel for Dx12...
[2025-07-01, 16:19:11.484] -> SUCCESS: test_shader_compilation_v4_1_declaration shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.503] -> SUCCESS: test_shader_compilation_v4_1_declaration compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.504] -> 
--- Testing: test_shader_compilation_v4_2_1_tile_a_loading ---
[2025-07-01, 16:19:11.504] -> Attempting to compile test_shader_compilation_v4_2_1_tile_a_loading WGSL kernel for Dx12...
[2025-07-01, 16:19:11.507] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.526] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.527] -> 
--- Testing: test_shader_compilation_v4_2_2_tile_b_loading ---
[2025-07-01, 16:19:11.527] -> Attempting to compile test_shader_compilation_v4_2_2_tile_b_loading WGSL kernel for Dx12...
[2025-07-01, 16:19:11.531] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading shader compilation failed: Validation Error: In Device::create_shader_module, label = 'test_shader_compilation_v4_2_2_tile_b_loading'
[2025-07-01, 16:19:11.531] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-07-01, 16:19:11.532] -> 
--- Testing: test_shader_compilation_v4_2_3_inlined_tile_b ---
[2025-07-01, 16:19:11.532] -> Attempting to compile test_shader_compilation_v4_2_3_inlined_tile_b WGSL kernel for Dx12...
[2025-07-01, 16:19:11.535] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.567] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.568] -> 
--- Testing: test_shader_compilation_v4_2_combined_loading ---
[2025-07-01, 16:19:11.568] -> Attempting to compile test_shader_compilation_v4_2_combined_loading WGSL kernel for Dx12...
[2025-07-01, 16:19:11.574] -> SUCCESS: test_shader_compilation_v4_2_combined_loading shader module compiled on Dx12 without error.
[2025-07-01, 16:19:11.607] -> SUCCESS: test_shader_compilation_v4_2_combined_loading compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.608] -> 
--- Testing: test_shader_compilation_v4_3_main_computation ---
[2025-07-01, 16:19:11.608] -> Attempting to compile test_shader_compilation_v4_3_main_computation WGSL kernel for Dx12...
[2025-07-01, 16:19:11.614] -> SUCCESS: test_shader_compilation_v4_3_main_computation shader module compiled on Dx12 without error.
[2025-07-01, 16:19:23.644] -> SUCCESS: test_shader_compilation_v4_3_main_computation compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:23.645] -> 
--- Testing: test_shader_compilation_v4_2_2_1_partial_write ---
[2025-07-01, 16:19:23.645] -> Attempting to compile test_shader_compilation_v4_2_2_1_partial_write WGSL kernel for Dx12...
[2025-07-01, 16:19:23.648] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write shader module compiled on Dx12 without error.
[2025-07-01, 16:19:23.677] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:23.678] -> 
--- Testing: test_shader_compilation_v4_2_2_2_local_copy ---
[2025-07-01, 16:19:23.678] -> Attempting to compile test_shader_compilation_v4_2_2_2_local_copy WGSL kernel for Dx12...
[2025-07-01, 16:19:23.681] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy shader module compiled on Dx12 without error.
[2025-07-01, 16:19:23.707] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:23.708] -> 
--- Testing: test_shader_compilation_v4_2_2_3_private_array ---
[2025-07-01, 16:19:23.709] -> Attempting to compile test_shader_compilation_v4_2_2_3_private_array WGSL kernel for Dx12...
[2025-07-01, 16:19:23.712] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array shader module compiled on Dx12 without error.
[2025-07-01, 16:19:23.739] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:23.740] -> 
--- Testing: test_shader_compilation_smoking_gun_unused_array ---
[2025-07-01, 16:19:23.740] -> Attempting to compile test_shader_compilation_smoking_gun_unused_array WGSL kernel for Dx12...
[2025-07-01, 16:19:23.741] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array shader module compiled on Dx12 without error.
[2025-07-01, 16:19:23.751] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:23.752] -> 
--- Testing: test_shader_compilation_full_kernel_with_fix ---
[2025-07-01, 16:19:23.752] -> Attempting to compile test_shader_compilation_full_kernel_with_fix WGSL kernel for Dx12...
[2025-07-01, 16:19:23.759] -> SUCCESS: test_shader_compilation_full_kernel_with_fix shader module compiled on Dx12 without error.
[2025-07-01, 16:19:35.691] -> SUCCESS: test_shader_compilation_full_kernel_with_fix compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:35.692] -> 
--- Testing: test_shader_compilation_production_kernel ---
[2025-07-01, 16:19:35.692] -> Attempting to compile test_shader_compilation_production_kernel WGSL kernel for Dx12...
[2025-07-01, 16:19:35.702] -> SUCCESS: test_shader_compilation_production_kernel shader module compiled on Dx12 without error.
[2025-07-01, 16:19:47.756] -> SUCCESS: test_shader_compilation_production_kernel compute pipeline created successfully on Dx12!
[2025-07-01, 16:19:11.482] -> 
--- Incremental Complexity Tests (V4 Base) ---
[2025-07-01, 16:19:23.752] -> 
--- Full Kernel and Production Tests ---
[2025-07-01, 16:19:47.765] -> SUCCESS: Fixed kernel compiled on Dx12.
[2025-07-01, 16:19:59.680] -> SUCCESS: Fixed kernel pipeline created on Dx12.
[2025-07-01, 16:19:59.695] -> GPU Output: [0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-07-01, 16:19:59.695] -> Scalar Output: [0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-07-01, 16:19:59.695] -> SUCCESS: Correctness test passed on Dx12!
[2025-07-01, 16:19:11.323] -> Test status: FAIL
[2025-07-01, 16:19:11.360] -> Test status: PASS
[2025-07-01, 16:19:11.360] -> Test completed [PASS]
[2025-07-01, 16:19:11.382] -> Test status: PASS
[2025-07-01, 16:19:11.382] -> Test completed [PASS]
[2025-07-01, 16:19:11.399] -> Test status: PASS
[2025-07-01, 16:19:11.399] -> Test completed [PASS]
[2025-07-01, 16:19:11.427] -> Test status: PASS
[2025-07-01, 16:19:11.428] -> Test completed [PASS]
[2025-07-01, 16:19:11.439] -> Test status: PASS
[2025-07-01, 16:19:11.439] -> Test completed [PASS]
[2025-07-01, 16:19:11.460] -> Test status: PASS
[2025-07-01, 16:19:11.461] -> Test completed [PASS]
[2025-07-01, 16:19:11.481] -> Test status: PASS
[2025-07-01, 16:19:11.481] -> Test completed [PASS]
[2025-07-01, 16:19:11.504] -> Test status: PASS
[2025-07-01, 16:19:11.504] -> Test completed [PASS]
[2025-07-01, 16:19:11.526] -> Test status: PASS
[2025-07-01, 16:19:11.526] -> Test completed [PASS]
[2025-07-01, 16:19:11.531] -> Test status: FAIL
[2025-07-01, 16:19:11.567] -> Test status: PASS
[2025-07-01, 16:19:11.568] -> Test completed [PASS]
[2025-07-01, 16:19:11.607] -> Test status: PASS
[2025-07-01, 16:19:11.607] -> Test completed [PASS]
[2025-07-01, 16:19:23.644] -> Test status: PASS
[2025-07-01, 16:19:23.645] -> Test completed [PASS]
[2025-07-01, 16:19:35.691] -> Test status: PASS
[2025-07-01, 16:19:35.692] -> Test completed [PASS]
[2025-07-01, 16:19:47.757] -> Test status: PASS
[2025-07-01, 16:19:47.757] -> Test completed [PASS]
[2025-07-01, 16:19:23.677] -> Test status: PASS
[2025-07-01, 16:19:23.678] -> Test completed [PASS]
[2025-07-01, 16:19:23.708] -> Test status: PASS
[2025-07-01, 16:19:23.708] -> Test completed [PASS]
[2025-07-01, 16:19:23.739] -> Test status: PASS
[2025-07-01, 16:19:23.739] -> Test completed [PASS]
[2025-07-01, 16:19:23.751] -> Test status: PASS
[2025-07-01, 16:19:23.751] -> Test completed [PASS]
[2025-07-01, 16:19:47.757] -> 
--- Running Correctness Test on Dx12 with Fixed Kernel ---
[2025-07-01, 16:19:59.696] -> Test completed
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 39
- **Passed:** 37
- **Failed:** 2

### Timing Information

- **Total Time:** 84.80 sec
- **Average Time:** 2174.00 ms

### Status

❌ 2 test(s) failed. See above for details.

---

_Report generated by BitNet Test Framework_
