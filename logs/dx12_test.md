# DX12_TEST Test Report

> Generated on: 2025-06-29 18:19:47

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Shader Compilation Correctness On Dx12             | ✅ Pass |  11.64 sec |             |
|  2 | Shader Compilation Fix V1 [PASS]                   | ✅ Pass |   25.00 ms |             |
|  3 | Shader Compilation Fix V1 Warm                     | ✅ Pass |   26.00 ms |             |
|  4 | Shader Compilation Fix V2 [PASS]                   | ✅ Pass |   17.00 ms |             |
|  5 | Shader Compilation Fix V2 Warm                     | ✅ Pass |   17.00 ms |             |
|  6 | Shader Compilation Fix V3 [PASS]                   | ✅ Pass |   13.00 ms |             |
|  7 | Shader Compilation Fix V3 Warm                     | ✅ Pass |   13.00 ms |             |
|  8 | Shader Compilation Fix V4 Simplified Workgroup [PASS] | ✅ Pass |   21.00 ms |             |
|  9 | Shader Compilation Fix V4 Simplified Workgroup Warm | ✅ Pass |   21.00 ms |             |
| 10 | Shader Compilation Fix V5 Minimal Kernel [PASS]    | ✅ Pass |    7.00 ms |             |
| 11 | Shader Compilation Fix V5 Minimal Kernel Warm      | ✅ Pass |    7.00 ms |             |
| 12 | Shader Compilation Fix V6 F32 Vectors [PASS]       | ✅ Pass |   16.00 ms |             |
| 13 | Shader Compilation Fix V6 F32 Vectors Warm         | ✅ Pass |   16.00 ms |             |
| 14 | Shader Compilation Fix V7 No Workgroup Memory [PASS] | ✅ Pass |   14.00 ms |             |
| 15 | Shader Compilation Fix V7 No Workgroup Memory Warm | ✅ Pass |   14.00 ms |             |
| 16 | Shader Compilation Full Kernel With Fix [PASS]     | ✅ Pass |  11.61 sec |             |
| 17 | Shader Compilation Full Kernel With Fix Warm       | ✅ Pass |  11.61 sec |             |
| 18 | Shader Compilation Original Buggy Warm             | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 19 | Shader Compilation Production Kernel [PASS]        | ✅ Pass |  11.84 sec |             |
| 20 | Shader Compilation Production Kernel Warm          | ✅ Pass |  11.84 sec |             |
| 21 | Shader Compilation Smoking Gun Unused Array [PASS] | ✅ Pass |    8.00 ms |             |
| 22 | Shader Compilation Smoking Gun Unused Array Warm   | ✅ Pass |    8.00 ms |             |
| 23 | Shader Compilation V4 1 Declaration [PASS]         | ✅ Pass |   17.00 ms |             |
| 24 | Shader Compilation V4 1 Declaration Warm           | ✅ Pass |   17.00 ms |             |
| 25 | Shader Compilation V4 2 1 Tile A Loading [PASS]    | ✅ Pass |   21.00 ms |             |
| 26 | Shader Compilation V4 2 1 Tile A Loading Warm      | ✅ Pass |   21.00 ms |             |
| 27 | Shader Compilation V4 2 2 1 Partial Write [PASS]   | ✅ Pass |   27.00 ms |             |
| 28 | Shader Compilation V4 2 2 1 Partial Write Warm     | ✅ Pass |   27.00 ms |             |
| 29 | Shader Compilation V4 2 2 2 Local Copy [PASS]      | ✅ Pass |   31.00 ms |             |
| 30 | Shader Compilation V4 2 2 2 Local Copy Warm        | ✅ Pass |   31.00 ms |             |
| 31 | Shader Compilation V4 2 2 3 Private Array [PASS]   | ✅ Pass |   31.00 ms |             |
| 32 | Shader Compilation V4 2 2 3 Private Array Warm     | ✅ Pass |   31.00 ms |             |
| 33 | Shader Compilation V4 2 2 Tile B Loading Warm      | ❌ Fail |   11.00 ms | Pipeline creation failed: Validation Error: In ... |
| 34 | Shader Compilation V4 2 3 Inlined Tile B [PASS]    | ✅ Pass |   33.00 ms |             |
| 35 | Shader Compilation V4 2 3 Inlined Tile B Warm      | ✅ Pass |   33.00 ms |             |
| 36 | Shader Compilation V4 2 Combined Loading [PASS]    | ✅ Pass |   37.00 ms |             |
| 37 | Shader Compilation V4 2 Combined Loading Warm      | ✅ Pass |   37.00 ms |             |
| 38 | Shader Compilation V4 3 Main Computation [PASS]    | ✅ Pass |  11.67 sec |             |
| 39 | Shader Compilation V4 3 Main Computation Warm      | ✅ Pass |  11.67 sec |             |

## ⭐ Special Finding

**[WORKAROUND FOUND]**: `test_shader_compilation_full_kernel_with_fix`  
This test demonstrates a robust workaround for the DX12/Naga WGSL bug: using a flattened i32 accumulator and per-element decode for tile_b. The full BitNet kernel logic now passes on DX12. See the test and kernel code for details.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-29, 18:18:58.604] -> [FAIL] test_shader_compilation_original_buggy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-06-29, 18:18:58.631] -> test_shader_compilation_fix_v1_warm      took     26.125 ms
[2025-06-29, 18:18:58.648] -> test_shader_compilation_fix_v2_warm      took     17.362 ms
[2025-06-29, 18:18:58.662] -> test_shader_compilation_fix_v3_warm      took     13.424 ms
[2025-06-29, 18:18:58.683] -> test_shader_compilation_fix_v4_simplified_workgroup_warm took     21.459 ms
[2025-06-29, 18:18:58.691] -> test_shader_compilation_fix_v5_minimal_kernel_warm took      7.994 ms
[2025-06-29, 18:18:58.708] -> test_shader_compilation_fix_v6_f32_vectors_warm took     16.747 ms
[2025-06-29, 18:18:58.723] -> test_shader_compilation_fix_v7_no_workgroup_memory_warm took     14.813 ms
[2025-06-29, 18:18:58.741] -> test_shader_compilation_v4_1_declaration_warm took     17.560 ms
[2025-06-29, 18:18:58.763] -> test_shader_compilation_v4_2_1_tile_a_loading_warm took     21.650 ms
[2025-06-29, 18:18:58.775] -> [FAIL] test_shader_compilation_v4_2_2_tile_b_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-06-29, 18:18:58.809] -> test_shader_compilation_v4_2_3_inlined_tile_b_warm took     33.811 ms
[2025-06-29, 18:18:58.846] -> test_shader_compilation_v4_2_combined_loading_warm took     37.256 ms
[2025-06-29, 18:19:10.513] -> test_shader_compilation_v4_3_main_computation_warm took  11666.809 ms
[2025-06-29, 18:19:10.541] -> test_shader_compilation_v4_2_2_1_partial_write_warm took     27.659 ms
[2025-06-29, 18:19:10.573] -> test_shader_compilation_v4_2_2_2_local_copy_warm took     31.846 ms
[2025-06-29, 18:19:10.604] -> test_shader_compilation_v4_2_2_3_private_array_warm took     31.341 ms
[2025-06-29, 18:19:10.613] -> test_shader_compilation_smoking_gun_unused_array_warm took      8.414 ms
[2025-06-29, 18:19:22.219] -> test_shader_compilation_full_kernel_with_fix_warm took  11605.415 ms
[2025-06-29, 18:19:34.064] -> test_shader_compilation_production_kernel_warm took  11844.554 ms
[2025-06-29, 18:18:58.143] ->  Dx12 Shader Bug Test Report - Generated on Sun, 29 Jun 2025 18:18:58 -0400
[2025-06-29, 18:18:58.600] -> -> Found Dx12 adapter: NVIDIA GeForce RTX 2070 SUPER (Dx12)
[2025-06-29, 18:18:58.600] -> 
--- Testing: Original Buggy ---
[2025-06-29, 18:18:58.600] -> Attempting to compile Original Buggy WGSL kernel for Dx12...
[2025-06-29, 18:18:58.603] -> ERROR: Original Buggy shader compilation failed: Validation Error: In Device::create_shader_module, label = 'Original Buggy'
[2025-06-29, 18:18:58.604] -> ERROR: Original Buggy pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'Original Buggy compute pipeline'
[2025-06-29, 18:18:58.604] -> 
--- Testing: Fix V1 ---
[2025-06-29, 18:18:58.605] -> Attempting to compile Fix V1 WGSL kernel for Dx12...
[2025-06-29, 18:18:58.608] -> SUCCESS: Fix V1 shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.630] -> SUCCESS: Fix V1 compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.631] -> 
--- Testing: test_shader_compilation_fix_v2 ---
[2025-06-29, 18:18:58.631] -> Attempting to compile test_shader_compilation_fix_v2 WGSL kernel for Dx12...
[2025-06-29, 18:18:58.634] -> SUCCESS: test_shader_compilation_fix_v2 shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.647] -> SUCCESS: test_shader_compilation_fix_v2 compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.648] -> 
--- Testing: test_shader_compilation_fix_v3 ---
[2025-06-29, 18:18:58.648] -> Attempting to compile test_shader_compilation_fix_v3 WGSL kernel for Dx12...
[2025-06-29, 18:18:58.652] -> SUCCESS: test_shader_compilation_fix_v3 shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.661] -> SUCCESS: test_shader_compilation_fix_v3 compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.662] -> 
--- Testing: test_shader_compilation_fix_v4_simplified_workgroup ---
[2025-06-29, 18:18:58.662] -> Attempting to compile test_shader_compilation_fix_v4_simplified_workgroup WGSL kernel for Dx12...
[2025-06-29, 18:18:58.664] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.682] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.683] -> 
--- Testing: test_shader_compilation_fix_v5_minimal_kernel ---
[2025-06-29, 18:18:58.684] -> Attempting to compile test_shader_compilation_fix_v5_minimal_kernel WGSL kernel for Dx12...
[2025-06-29, 18:18:58.685] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.691] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.692] -> 
--- Testing: test_shader_compilation_fix_v6_f32_vectors ---
[2025-06-29, 18:18:58.692] -> Attempting to compile test_shader_compilation_fix_v6_f32_vectors WGSL kernel for Dx12...
[2025-06-29, 18:18:58.694] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.707] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.708] -> 
--- Testing: test_shader_compilation_fix_v7_no_workgroup_memory ---
[2025-06-29, 18:18:58.709] -> Attempting to compile test_shader_compilation_fix_v7_no_workgroup_memory WGSL kernel for Dx12...
[2025-06-29, 18:18:58.711] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.722] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.724] -> 
--- Testing: test_shader_compilation_v4_1_declaration ---
[2025-06-29, 18:18:58.724] -> Attempting to compile test_shader_compilation_v4_1_declaration WGSL kernel for Dx12...
[2025-06-29, 18:18:58.726] -> SUCCESS: test_shader_compilation_v4_1_declaration shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.741] -> SUCCESS: test_shader_compilation_v4_1_declaration compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.742] -> 
--- Testing: test_shader_compilation_v4_2_1_tile_a_loading ---
[2025-06-29, 18:18:58.742] -> Attempting to compile test_shader_compilation_v4_2_1_tile_a_loading WGSL kernel for Dx12...
[2025-06-29, 18:18:58.745] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.762] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.763] -> 
--- Testing: test_shader_compilation_v4_2_2_tile_b_loading ---
[2025-06-29, 18:18:58.764] -> Attempting to compile test_shader_compilation_v4_2_2_tile_b_loading WGSL kernel for Dx12...
[2025-06-29, 18:18:58.770] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading shader compilation failed: Validation Error: In Device::create_shader_module, label = 'test_shader_compilation_v4_2_2_tile_b_loading'
[2025-06-29, 18:18:58.773] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline, label = 'test_shader_compilation_v4_2_2_tile_b_loading compute pipeline'
[2025-06-29, 18:18:58.775] -> 
--- Testing: test_shader_compilation_v4_2_3_inlined_tile_b ---
[2025-06-29, 18:18:58.775] -> Attempting to compile test_shader_compilation_v4_2_3_inlined_tile_b WGSL kernel for Dx12...
[2025-06-29, 18:18:58.779] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.807] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.809] -> 
--- Testing: test_shader_compilation_v4_2_combined_loading ---
[2025-06-29, 18:18:58.809] -> Attempting to compile test_shader_compilation_v4_2_combined_loading WGSL kernel for Dx12...
[2025-06-29, 18:18:58.815] -> SUCCESS: test_shader_compilation_v4_2_combined_loading shader module compiled on Dx12 without error.
[2025-06-29, 18:18:58.845] -> SUCCESS: test_shader_compilation_v4_2_combined_loading compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.846] -> 
--- Testing: test_shader_compilation_v4_3_main_computation ---
[2025-06-29, 18:18:58.847] -> Attempting to compile test_shader_compilation_v4_3_main_computation WGSL kernel for Dx12...
[2025-06-29, 18:18:58.853] -> SUCCESS: test_shader_compilation_v4_3_main_computation shader module compiled on Dx12 without error.
[2025-06-29, 18:19:10.512] -> SUCCESS: test_shader_compilation_v4_3_main_computation compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:10.513] -> 
--- Testing: test_shader_compilation_v4_2_2_1_partial_write ---
[2025-06-29, 18:19:10.513] -> Attempting to compile test_shader_compilation_v4_2_2_1_partial_write WGSL kernel for Dx12...
[2025-06-29, 18:19:10.517] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write shader module compiled on Dx12 without error.
[2025-06-29, 18:19:10.540] -> SUCCESS: test_shader_compilation_v4_2_2_1_partial_write compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:10.541] -> 
--- Testing: test_shader_compilation_v4_2_2_2_local_copy ---
[2025-06-29, 18:19:10.541] -> Attempting to compile test_shader_compilation_v4_2_2_2_local_copy WGSL kernel for Dx12...
[2025-06-29, 18:19:10.544] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy shader module compiled on Dx12 without error.
[2025-06-29, 18:19:10.572] -> SUCCESS: test_shader_compilation_v4_2_2_2_local_copy compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:10.573] -> 
--- Testing: test_shader_compilation_v4_2_2_3_private_array ---
[2025-06-29, 18:19:10.573] -> Attempting to compile test_shader_compilation_v4_2_2_3_private_array WGSL kernel for Dx12...
[2025-06-29, 18:19:10.577] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array shader module compiled on Dx12 without error.
[2025-06-29, 18:19:10.603] -> SUCCESS: test_shader_compilation_v4_2_2_3_private_array compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:10.604] -> 
--- Testing: test_shader_compilation_smoking_gun_unused_array ---
[2025-06-29, 18:19:10.605] -> Attempting to compile test_shader_compilation_smoking_gun_unused_array WGSL kernel for Dx12...
[2025-06-29, 18:19:10.606] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array shader module compiled on Dx12 without error.
[2025-06-29, 18:19:10.612] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:10.613] -> 
--- Testing: test_shader_compilation_full_kernel_with_fix ---
[2025-06-29, 18:19:10.614] -> Attempting to compile test_shader_compilation_full_kernel_with_fix WGSL kernel for Dx12...
[2025-06-29, 18:19:10.620] -> SUCCESS: test_shader_compilation_full_kernel_with_fix shader module compiled on Dx12 without error.
[2025-06-29, 18:19:22.218] -> SUCCESS: test_shader_compilation_full_kernel_with_fix compute pipeline created successfully on Dx12!
[2025-06-29, 18:19:22.219] -> 
--- Testing: test_shader_compilation_production_kernel ---
[2025-06-29, 18:19:22.219] -> Attempting to compile test_shader_compilation_production_kernel WGSL kernel for Dx12...
[2025-06-29, 18:19:22.229] -> SUCCESS: test_shader_compilation_production_kernel shader module compiled on Dx12 without error.
[2025-06-29, 18:19:34.063] -> SUCCESS: test_shader_compilation_production_kernel compute pipeline created successfully on Dx12!
[2025-06-29, 18:18:58.724] -> 
--- Incremental Complexity Tests (V4 Base) ---
[2025-06-29, 18:19:10.613] -> 
--- Full Kernel and Production Tests ---
[2025-06-29, 18:19:34.073] -> SUCCESS: Fixed kernel compiled on Dx12.
[2025-06-29, 18:19:45.684] -> SUCCESS: Fixed kernel pipeline created on Dx12.
[2025-06-29, 18:19:45.700] -> GPU Output: [2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 18:19:45.700] -> Scalar Output: [2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 18:19:45.700] -> SUCCESS: Correctness test passed on Dx12!
[2025-06-29, 18:18:58.604] -> Test status: FAIL
[2025-06-29, 18:18:58.630] -> Test status: PASS
[2025-06-29, 18:18:58.630] -> Test completed [PASS]
[2025-06-29, 18:18:58.648] -> Test status: PASS
[2025-06-29, 18:18:58.648] -> Test completed [PASS]
[2025-06-29, 18:18:58.661] -> Test status: PASS
[2025-06-29, 18:18:58.661] -> Test completed [PASS]
[2025-06-29, 18:18:58.683] -> Test status: PASS
[2025-06-29, 18:18:58.683] -> Test completed [PASS]
[2025-06-29, 18:18:58.691] -> Test status: PASS
[2025-06-29, 18:18:58.691] -> Test completed [PASS]
[2025-06-29, 18:18:58.708] -> Test status: PASS
[2025-06-29, 18:18:58.708] -> Test completed [PASS]
[2025-06-29, 18:18:58.723] -> Test status: PASS
[2025-06-29, 18:18:58.723] -> Test completed [PASS]
[2025-06-29, 18:18:58.741] -> Test status: PASS
[2025-06-29, 18:18:58.741] -> Test completed [PASS]
[2025-06-29, 18:18:58.762] -> Test status: PASS
[2025-06-29, 18:18:58.763] -> Test completed [PASS]
[2025-06-29, 18:18:58.774] -> Test status: FAIL
[2025-06-29, 18:18:58.808] -> Test status: PASS
[2025-06-29, 18:18:58.808] -> Test completed [PASS]
[2025-06-29, 18:18:58.845] -> Test status: PASS
[2025-06-29, 18:18:58.846] -> Test completed [PASS]
[2025-06-29, 18:19:10.512] -> Test status: PASS
[2025-06-29, 18:19:10.513] -> Test completed [PASS]
[2025-06-29, 18:19:22.218] -> Test status: PASS
[2025-06-29, 18:19:22.219] -> Test completed [PASS]
[2025-06-29, 18:19:34.063] -> Test status: PASS
[2025-06-29, 18:19:34.063] -> Test completed [PASS]
[2025-06-29, 18:19:10.540] -> Test status: PASS
[2025-06-29, 18:19:10.541] -> Test completed [PASS]
[2025-06-29, 18:19:10.572] -> Test status: PASS
[2025-06-29, 18:19:10.573] -> Test completed [PASS]
[2025-06-29, 18:19:10.604] -> Test status: PASS
[2025-06-29, 18:19:10.604] -> Test completed [PASS]
[2025-06-29, 18:19:10.612] -> Test status: PASS
[2025-06-29, 18:19:10.613] -> Test completed [PASS]
[2025-06-29, 18:19:34.064] -> 
--- Running Correctness Test on Dx12 with Fixed Kernel ---
[2025-06-29, 18:19:45.700] -> Test completed
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 39
- **Passed:** 37
- **Failed:** 2

### Timing Information

- **Total Time:** 82.54 sec
- **Average Time:** 2116.00 ms

### Status

❌ 2 test(s) failed. See above for details.

---

_Report generated by BitNet Test Framework_
