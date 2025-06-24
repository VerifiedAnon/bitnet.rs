# DX12_TEST Test Report

> Generated on: 2025-06-23 21:46:39

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Shader Compilation Correctness On Dx12             | ✅ Pass |   11.00 ms |             |
|  2 | Shader Compilation Fix V1 Warm                     | ❌ Fail |    6.00 ms | Pipeline creation failed: Validation Error: In ... |
|  3 | Shader Compilation Fix V2 Warm                     | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
|  4 | Shader Compilation Fix V3 Warm                     | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
|  5 | Shader Compilation Fix V4 Simplified Workgroup [PASS] | ✅ Pass |   20.00 ms |             |
|  6 | Shader Compilation Fix V4 Simplified Workgroup Warm | ✅ Pass |   21.00 ms |             |
|  7 | Shader Compilation Fix V5 Minimal Kernel [PASS]    | ✅ Pass |    8.00 ms |             |
|  8 | Shader Compilation Fix V5 Minimal Kernel Warm      | ✅ Pass |    8.00 ms |             |
|  9 | Shader Compilation Fix V6 F32 Vectors [PASS]       | ✅ Pass |   14.00 ms |             |
| 10 | Shader Compilation Fix V6 F32 Vectors Warm         | ✅ Pass |   14.00 ms |             |
| 11 | Shader Compilation Fix V7 No Workgroup Memory [PASS] | ✅ Pass |   13.00 ms |             |
| 12 | Shader Compilation Fix V7 No Workgroup Memory Warm | ✅ Pass |   13.00 ms |             |
| 13 | Shader Compilation Full Kernel With Fix [PASS]     | ✅ Pass |  11.78 sec |             |
| 14 | Shader Compilation Full Kernel With Fix Warm       | ✅ Pass |  11.78 sec |             |
| 15 | Shader Compilation Original Buggy Warm             | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
| 16 | Shader Compilation Production Kernel Warm          | ❌ Fail |   16.00 ms | Pipeline creation failed: Validation Error: In ... |
| 17 | Shader Compilation Smoking Gun Unused Array Warm   | ❌ Fail |    3.00 ms | Pipeline creation failed: Validation Error: In ... |
| 18 | Shader Compilation V4 1 Declaration [PASS]         | ✅ Pass |   15.00 ms |             |
| 19 | Shader Compilation V4 1 Declaration Warm           | ✅ Pass |   15.00 ms |             |
| 20 | Shader Compilation V4 2 1 Tile A Loading [PASS]    | ✅ Pass |   20.00 ms |             |
| 21 | Shader Compilation V4 2 1 Tile A Loading Warm      | ✅ Pass |   20.00 ms |             |
| 22 | Shader Compilation V4 2 2 1 Partial Write Warm     | ❌ Fail |    3.00 ms | Pipeline creation failed: Validation Error: In ... |
| 23 | Shader Compilation V4 2 2 2 Local Copy Warm        | ❌ Fail |    3.00 ms | Pipeline creation failed: Validation Error: In ... |
| 24 | Shader Compilation V4 2 2 3 Private Array Warm     | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 25 | Shader Compilation V4 2 2 Tile B Loading Warm      | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
| 26 | Shader Compilation V4 2 3 Inlined Tile B [PASS]    | ✅ Pass |   26.00 ms |             |
| 27 | Shader Compilation V4 2 3 Inlined Tile B Warm      | ✅ Pass |   26.00 ms |             |
| 28 | Shader Compilation V4 2 Combined Loading Warm      | ❌ Fail |    6.00 ms | Pipeline creation failed: Validation Error: In ... |
| 29 | Shader Compilation V4 3 Main Computation Warm      | ❌ Fail |    7.00 ms | Pipeline creation failed: Validation Error: In ... |

## ⭐ Special Finding

**[WORKAROUND FOUND]**: `test_shader_compilation_full_kernel_with_fix`  
This test demonstrates a robust workaround for the DX12/Naga WGSL bug: using a flattened i32 accumulator and per-element decode for tile_b. The full BitNet kernel logic now passes on DX12. See the test and kernel code for details.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-23, 21:46:25.620] -> [FAIL] test_shader_compilation_original_buggy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.627] -> [FAIL] test_shader_compilation_fix_v1_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.633] -> [FAIL] test_shader_compilation_fix_v2_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.639] -> [FAIL] test_shader_compilation_fix_v3_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.660] -> test_shader_compilation_fix_v4_simplified_workgroup_warm took     21.082 ms
[2025-06-23, 21:46:25.669] -> test_shader_compilation_fix_v5_minimal_kernel_warm took      8.593 ms
[2025-06-23, 21:46:25.684] -> test_shader_compilation_fix_v6_f32_vectors_warm took     14.478 ms
[2025-06-23, 21:46:25.698] -> test_shader_compilation_fix_v7_no_workgroup_memory_warm took     13.795 ms
[2025-06-23, 21:46:25.714] -> test_shader_compilation_v4_1_declaration_warm took     15.893 ms
[2025-06-23, 21:46:25.735] -> test_shader_compilation_v4_2_1_tile_a_loading_warm took     20.296 ms
[2025-06-23, 21:46:25.740] -> [FAIL] test_shader_compilation_v4_2_2_tile_b_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.767] -> test_shader_compilation_v4_2_3_inlined_tile_b_warm took     26.600 ms
[2025-06-23, 21:46:25.774] -> [FAIL] test_shader_compilation_v4_2_combined_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.781] -> [FAIL] test_shader_compilation_v4_3_main_computation_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.785] -> [FAIL] test_shader_compilation_v4_2_2_1_partial_write_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.789] -> [FAIL] test_shader_compilation_v4_2_2_2_local_copy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.794] -> [FAIL] test_shader_compilation_v4_2_2_3_private_array_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.797] -> [FAIL] test_shader_compilation_smoking_gun_unused_array_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:37.575] -> test_shader_compilation_full_kernel_with_fix_warm took  11777.256 ms
[2025-06-23, 21:46:37.592] -> [FAIL] test_shader_compilation_production_kernel_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.165] ->  Dx12 Shader Bug Test Report - Generated on Mon, 23 Jun 2025 21:46:25 -0400
[2025-06-23, 21:46:25.614] -> -> Found Dx12 adapter: NVIDIA GeForce RTX 2070 SUPER (Dx12)
[2025-06-23, 21:46:25.615] -> 
--- Testing: Original Buggy ---
[2025-06-23, 21:46:25.615] -> Attempting to compile Original Buggy WGSL kernel for Dx12...
[2025-06-23, 21:46:25.619] -> ERROR: Original Buggy shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.620] -> ERROR: Original Buggy pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.620] -> 
--- Testing: Fix V1 ---
[2025-06-23, 21:46:25.621] -> Attempting to compile Fix V1 WGSL kernel for Dx12...
[2025-06-23, 21:46:25.624] -> SUCCESS: Fix V1 shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.627] -> ERROR: Fix V1 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.627] -> 
--- Testing: test_shader_compilation_fix_v2 ---
[2025-06-23, 21:46:25.628] -> Attempting to compile test_shader_compilation_fix_v2 WGSL kernel for Dx12...
[2025-06-23, 21:46:25.630] -> SUCCESS: test_shader_compilation_fix_v2 shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.633] -> ERROR: test_shader_compilation_fix_v2 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.633] -> 
--- Testing: test_shader_compilation_fix_v3 ---
[2025-06-23, 21:46:25.634] -> Attempting to compile test_shader_compilation_fix_v3 WGSL kernel for Dx12...
[2025-06-23, 21:46:25.637] -> SUCCESS: test_shader_compilation_fix_v3 shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.639] -> ERROR: test_shader_compilation_fix_v3 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.639] -> 
--- Testing: test_shader_compilation_fix_v4_simplified_workgroup ---
[2025-06-23, 21:46:25.639] -> Attempting to compile test_shader_compilation_fix_v4_simplified_workgroup WGSL kernel for Dx12...
[2025-06-23, 21:46:25.642] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.659] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.661] -> 
--- Testing: test_shader_compilation_fix_v5_minimal_kernel ---
[2025-06-23, 21:46:25.661] -> Attempting to compile test_shader_compilation_fix_v5_minimal_kernel WGSL kernel for Dx12...
[2025-06-23, 21:46:25.662] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.669] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.669] -> 
--- Testing: test_shader_compilation_fix_v6_f32_vectors ---
[2025-06-23, 21:46:25.670] -> Attempting to compile test_shader_compilation_fix_v6_f32_vectors WGSL kernel for Dx12...
[2025-06-23, 21:46:25.672] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.683] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.684] -> 
--- Testing: test_shader_compilation_fix_v7_no_workgroup_memory ---
[2025-06-23, 21:46:25.684] -> Attempting to compile test_shader_compilation_fix_v7_no_workgroup_memory WGSL kernel for Dx12...
[2025-06-23, 21:46:25.686] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.697] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.698] -> 
--- Testing: test_shader_compilation_v4_1_declaration ---
[2025-06-23, 21:46:25.699] -> Attempting to compile test_shader_compilation_v4_1_declaration WGSL kernel for Dx12...
[2025-06-23, 21:46:25.701] -> SUCCESS: test_shader_compilation_v4_1_declaration shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.714] -> SUCCESS: test_shader_compilation_v4_1_declaration compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.714] -> 
--- Testing: test_shader_compilation_v4_2_1_tile_a_loading ---
[2025-06-23, 21:46:25.715] -> Attempting to compile test_shader_compilation_v4_2_1_tile_a_loading WGSL kernel for Dx12...
[2025-06-23, 21:46:25.718] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.734] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.735] -> 
--- Testing: test_shader_compilation_v4_2_2_tile_b_loading ---
[2025-06-23, 21:46:25.735] -> Attempting to compile test_shader_compilation_v4_2_2_tile_b_loading WGSL kernel for Dx12...
[2025-06-23, 21:46:25.739] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.740] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.740] -> 
--- Testing: test_shader_compilation_v4_2_3_inlined_tile_b ---
[2025-06-23, 21:46:25.741] -> Attempting to compile test_shader_compilation_v4_2_3_inlined_tile_b WGSL kernel for Dx12...
[2025-06-23, 21:46:25.744] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.766] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:25.767] -> 
--- Testing: test_shader_compilation_v4_2_combined_loading ---
[2025-06-23, 21:46:25.767] -> Attempting to compile test_shader_compilation_v4_2_combined_loading WGSL kernel for Dx12...
[2025-06-23, 21:46:25.773] -> ERROR: test_shader_compilation_v4_2_combined_loading shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.773] -> ERROR: test_shader_compilation_v4_2_combined_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.774] -> 
--- Testing: test_shader_compilation_v4_3_main_computation ---
[2025-06-23, 21:46:25.774] -> Attempting to compile test_shader_compilation_v4_3_main_computation WGSL kernel for Dx12...
[2025-06-23, 21:46:25.780] -> ERROR: test_shader_compilation_v4_3_main_computation shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.781] -> ERROR: test_shader_compilation_v4_3_main_computation pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.781] -> 
--- Testing: test_shader_compilation_v4_2_2_1_partial_write ---
[2025-06-23, 21:46:25.781] -> Attempting to compile test_shader_compilation_v4_2_2_1_partial_write WGSL kernel for Dx12...
[2025-06-23, 21:46:25.784] -> ERROR: test_shader_compilation_v4_2_2_1_partial_write shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.785] -> ERROR: test_shader_compilation_v4_2_2_1_partial_write pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.785] -> 
--- Testing: test_shader_compilation_v4_2_2_2_local_copy ---
[2025-06-23, 21:46:25.785] -> Attempting to compile test_shader_compilation_v4_2_2_2_local_copy WGSL kernel for Dx12...
[2025-06-23, 21:46:25.788] -> ERROR: test_shader_compilation_v4_2_2_2_local_copy shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.789] -> ERROR: test_shader_compilation_v4_2_2_2_local_copy pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.789] -> 
--- Testing: test_shader_compilation_v4_2_2_3_private_array ---
[2025-06-23, 21:46:25.790] -> Attempting to compile test_shader_compilation_v4_2_2_3_private_array WGSL kernel for Dx12...
[2025-06-23, 21:46:25.793] -> ERROR: test_shader_compilation_v4_2_2_3_private_array shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 21:46:25.793] -> ERROR: test_shader_compilation_v4_2_2_3_private_array pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.794] -> 
--- Testing: test_shader_compilation_smoking_gun_unused_array ---
[2025-06-23, 21:46:25.794] -> Attempting to compile test_shader_compilation_smoking_gun_unused_array WGSL kernel for Dx12...
[2025-06-23, 21:46:25.795] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array shader module compiled on Dx12 without error.
[2025-06-23, 21:46:25.797] -> ERROR: test_shader_compilation_smoking_gun_unused_array pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.798] -> 
--- Testing: test_shader_compilation_full_kernel_with_fix ---
[2025-06-23, 21:46:25.798] -> Attempting to compile test_shader_compilation_full_kernel_with_fix WGSL kernel for Dx12...
[2025-06-23, 21:46:25.805] -> SUCCESS: test_shader_compilation_full_kernel_with_fix shader module compiled on Dx12 without error.
[2025-06-23, 21:46:37.575] -> SUCCESS: test_shader_compilation_full_kernel_with_fix compute pipeline created successfully on Dx12!
[2025-06-23, 21:46:37.575] -> 
--- Testing: test_shader_compilation_production_kernel ---
[2025-06-23, 21:46:37.575] -> Attempting to compile test_shader_compilation_production_kernel WGSL kernel for Dx12...
[2025-06-23, 21:46:37.588] -> SUCCESS: test_shader_compilation_production_kernel shader module compiled on Dx12 without error.
[2025-06-23, 21:46:37.591] -> ERROR: test_shader_compilation_production_kernel pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.698] -> 
--- Incremental Complexity Tests (V4 Base) ---
[2025-06-23, 21:46:25.798] -> 
--- Full Kernel and Production Tests ---
[2025-06-23, 21:46:37.599] -> SUCCESS: Fixed kernel compiled on Dx12.
[2025-06-23, 21:46:37.603] -> ERROR: Correctness test failed at PIPELINE CREATION: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 21:46:25.620] -> Test status: FAIL
[2025-06-23, 21:46:25.627] -> Test status: FAIL
[2025-06-23, 21:46:25.633] -> Test status: FAIL
[2025-06-23, 21:46:25.639] -> Test status: FAIL
[2025-06-23, 21:46:25.660] -> Test status: PASS
[2025-06-23, 21:46:25.660] -> Test completed [PASS]
[2025-06-23, 21:46:25.669] -> Test status: PASS
[2025-06-23, 21:46:25.669] -> Test completed [PASS]
[2025-06-23, 21:46:25.683] -> Test status: PASS
[2025-06-23, 21:46:25.684] -> Test completed [PASS]
[2025-06-23, 21:46:25.697] -> Test status: PASS
[2025-06-23, 21:46:25.698] -> Test completed [PASS]
[2025-06-23, 21:46:25.714] -> Test status: PASS
[2025-06-23, 21:46:25.714] -> Test completed [PASS]
[2025-06-23, 21:46:25.734] -> Test status: PASS
[2025-06-23, 21:46:25.735] -> Test completed [PASS]
[2025-06-23, 21:46:25.740] -> Test status: FAIL
[2025-06-23, 21:46:25.766] -> Test status: PASS
[2025-06-23, 21:46:25.767] -> Test completed [PASS]
[2025-06-23, 21:46:25.773] -> Test status: FAIL
[2025-06-23, 21:46:25.781] -> Test status: FAIL
[2025-06-23, 21:46:37.575] -> Test status: PASS
[2025-06-23, 21:46:37.575] -> Test completed [PASS]
[2025-06-23, 21:46:37.591] -> Test status: FAIL
[2025-06-23, 21:46:25.785] -> Test status: FAIL
[2025-06-23, 21:46:25.789] -> Test status: FAIL
[2025-06-23, 21:46:25.793] -> Test status: FAIL
[2025-06-23, 21:46:25.797] -> Test status: FAIL
[2025-06-23, 21:46:37.592] -> 
--- Running Correctness Test on Dx12 with Fixed Kernel ---
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 29
- **Passed:** 17
- **Failed:** 12

### Timing Information

- **Total Time:** 23.88 sec
- **Average Time:** 823.00 ms

### Status

❌ 12 test(s) failed. See above for details.

---

_Report generated by BitNet Test Framework_
