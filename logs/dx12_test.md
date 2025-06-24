# DX12_TEST Test Report

> Generated on: 2025-06-23 20:43:49

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Shader Compilation Correctness On Dx12             | ✅ Pass |   11.00 ms |             |
|  2 | Shader Compilation Fix V1 Warm                     | ❌ Fail |    6.00 ms | Pipeline creation failed: Validation Error: In ... |
|  3 | Shader Compilation Fix V2 Warm                     | ❌ Fail |    6.00 ms | Pipeline creation failed: Validation Error: In ... |
|  4 | Shader Compilation Fix V3 Warm                     | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
|  5 | Shader Compilation Fix V4 Simplified Workgroup [PASS] | ✅ Pass |   22.00 ms |             |
|  6 | Shader Compilation Fix V4 Simplified Workgroup Warm | ✅ Pass |   22.00 ms |             |
|  7 | Shader Compilation Fix V5 Minimal Kernel [PASS]    | ✅ Pass |    8.00 ms |             |
|  8 | Shader Compilation Fix V5 Minimal Kernel Warm      | ✅ Pass |    8.00 ms |             |
|  9 | Shader Compilation Fix V6 F32 Vectors [PASS]       | ✅ Pass |   15.00 ms |             |
| 10 | Shader Compilation Fix V6 F32 Vectors Warm         | ✅ Pass |   15.00 ms |             |
| 11 | Shader Compilation Fix V7 No Workgroup Memory [PASS] | ✅ Pass |   14.00 ms |             |
| 12 | Shader Compilation Fix V7 No Workgroup Memory Warm | ✅ Pass |   14.00 ms |             |
| 13 | Shader Compilation Full Kernel With Fix Warm       | ❌ Fail |   11.00 ms | Pipeline creation failed: Validation Error: In ... |
| 14 | Shader Compilation Original Buggy Warm             | ❌ Fail |    6.00 ms | Pipeline creation failed: Validation Error: In ... |
| 15 | Shader Compilation Production Kernel Warm          | ❌ Fail |   18.00 ms | Pipeline creation failed: Validation Error: In ... |
| 16 | Shader Compilation Smoking Gun Unused Array Warm   | ❌ Fail |    3.00 ms | Pipeline creation failed: Validation Error: In ... |
| 17 | Shader Compilation V4 1 Declaration [PASS]         | ✅ Pass |   16.00 ms |             |
| 18 | Shader Compilation V4 1 Declaration Warm           | ✅ Pass |   16.00 ms |             |
| 19 | Shader Compilation V4 2 1 Tile A Loading [PASS]    | ✅ Pass |   21.00 ms |             |
| 20 | Shader Compilation V4 2 1 Tile A Loading Warm      | ✅ Pass |   21.00 ms |             |
| 21 | Shader Compilation V4 2 2 1 Partial Write Warm     | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 22 | Shader Compilation V4 2 2 2 Local Copy Warm        | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 23 | Shader Compilation V4 2 2 3 Private Array Warm     | ❌ Fail |    4.00 ms | Pipeline creation failed: Validation Error: In ... |
| 24 | Shader Compilation V4 2 2 Tile B Loading Warm      | ❌ Fail |    5.00 ms | Pipeline creation failed: Validation Error: In ... |
| 25 | Shader Compilation V4 2 3 Inlined Tile B [PASS]    | ✅ Pass |   26.00 ms |             |
| 26 | Shader Compilation V4 2 3 Inlined Tile B Warm      | ✅ Pass |   26.00 ms |             |
| 27 | Shader Compilation V4 2 Combined Loading Warm      | ❌ Fail |    7.00 ms | Pipeline creation failed: Validation Error: In ... |
| 28 | Shader Compilation V4 3 Main Computation Warm      | ❌ Fail |    7.00 ms | Pipeline creation failed: Validation Error: In ... |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-23, 20:43:46.992] -> [FAIL] test_shader_compilation_original_buggy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:46.999] -> [FAIL] test_shader_compilation_fix_v1_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.005] -> [FAIL] test_shader_compilation_fix_v2_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.011] -> [FAIL] test_shader_compilation_fix_v3_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.034] -> test_shader_compilation_fix_v4_simplified_workgroup_warm took     22.406 ms
[2025-06-23, 20:43:47.043] -> test_shader_compilation_fix_v5_minimal_kernel_warm took      8.605 ms
[2025-06-23, 20:43:47.058] -> test_shader_compilation_fix_v6_f32_vectors_warm took     15.208 ms
[2025-06-23, 20:43:47.073] -> test_shader_compilation_fix_v7_no_workgroup_memory_warm took     14.332 ms
[2025-06-23, 20:43:47.090] -> test_shader_compilation_v4_1_declaration_warm took     16.516 ms
[2025-06-23, 20:43:47.112] -> test_shader_compilation_v4_2_1_tile_a_loading_warm took     21.823 ms
[2025-06-23, 20:43:47.117] -> [FAIL] test_shader_compilation_v4_2_2_tile_b_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.143] -> test_shader_compilation_v4_2_3_inlined_tile_b_warm took     26.485 ms
[2025-06-23, 20:43:47.151] -> [FAIL] test_shader_compilation_v4_2_combined_loading_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.158] -> [FAIL] test_shader_compilation_v4_3_main_computation_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.163] -> [FAIL] test_shader_compilation_v4_2_2_1_partial_write_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.167] -> [FAIL] test_shader_compilation_v4_2_2_2_local_copy_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.173] -> [FAIL] test_shader_compilation_v4_2_2_3_private_array_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.177] -> [FAIL] test_shader_compilation_smoking_gun_unused_array_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.189] -> [FAIL] test_shader_compilation_full_kernel_with_fix_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.208] -> [FAIL] test_shader_compilation_production_kernel_warm: Pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:46.534] ->  Dx12 Shader Bug Test Report - Generated on Mon, 23 Jun 2025 20:43:46 -0400
[2025-06-23, 20:43:46.985] -> -> Found Dx12 adapter: NVIDIA GeForce RTX 2070 SUPER (Dx12)
[2025-06-23, 20:43:46.986] -> 
--- Testing: Original Buggy ---
[2025-06-23, 20:43:46.986] -> Attempting to compile Original Buggy WGSL kernel for Dx12...
[2025-06-23, 20:43:46.991] -> ERROR: Original Buggy shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:46.992] -> ERROR: Original Buggy pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:46.992] -> 
--- Testing: Fix V1 ---
[2025-06-23, 20:43:46.993] -> Attempting to compile Fix V1 WGSL kernel for Dx12...
[2025-06-23, 20:43:46.996] -> SUCCESS: Fix V1 shader module compiled on Dx12 without error.
[2025-06-23, 20:43:46.999] -> ERROR: Fix V1 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:46.999] -> 
--- Testing: test_shader_compilation_fix_v2 ---
[2025-06-23, 20:43:47.000] -> Attempting to compile test_shader_compilation_fix_v2 WGSL kernel for Dx12...
[2025-06-23, 20:43:47.002] -> SUCCESS: test_shader_compilation_fix_v2 shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.005] -> ERROR: test_shader_compilation_fix_v2 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.006] -> 
--- Testing: test_shader_compilation_fix_v3 ---
[2025-06-23, 20:43:47.006] -> Attempting to compile test_shader_compilation_fix_v3 WGSL kernel for Dx12...
[2025-06-23, 20:43:47.009] -> SUCCESS: test_shader_compilation_fix_v3 shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.011] -> ERROR: test_shader_compilation_fix_v3 pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.011] -> 
--- Testing: test_shader_compilation_fix_v4_simplified_workgroup ---
[2025-06-23, 20:43:47.012] -> Attempting to compile test_shader_compilation_fix_v4_simplified_workgroup WGSL kernel for Dx12...
[2025-06-23, 20:43:47.014] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.033] -> SUCCESS: test_shader_compilation_fix_v4_simplified_workgroup compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.034] -> 
--- Testing: test_shader_compilation_fix_v5_minimal_kernel ---
[2025-06-23, 20:43:47.034] -> Attempting to compile test_shader_compilation_fix_v5_minimal_kernel WGSL kernel for Dx12...
[2025-06-23, 20:43:47.036] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.042] -> SUCCESS: test_shader_compilation_fix_v5_minimal_kernel compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.043] -> 
--- Testing: test_shader_compilation_fix_v6_f32_vectors ---
[2025-06-23, 20:43:47.043] -> Attempting to compile test_shader_compilation_fix_v6_f32_vectors WGSL kernel for Dx12...
[2025-06-23, 20:43:47.045] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.057] -> SUCCESS: test_shader_compilation_fix_v6_f32_vectors compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.058] -> 
--- Testing: test_shader_compilation_fix_v7_no_workgroup_memory ---
[2025-06-23, 20:43:47.058] -> Attempting to compile test_shader_compilation_fix_v7_no_workgroup_memory WGSL kernel for Dx12...
[2025-06-23, 20:43:47.061] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.072] -> SUCCESS: test_shader_compilation_fix_v7_no_workgroup_memory compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.073] -> 
--- Testing: test_shader_compilation_v4_1_declaration ---
[2025-06-23, 20:43:47.073] -> Attempting to compile test_shader_compilation_v4_1_declaration WGSL kernel for Dx12...
[2025-06-23, 20:43:47.076] -> SUCCESS: test_shader_compilation_v4_1_declaration shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.089] -> SUCCESS: test_shader_compilation_v4_1_declaration compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.090] -> 
--- Testing: test_shader_compilation_v4_2_1_tile_a_loading ---
[2025-06-23, 20:43:47.090] -> Attempting to compile test_shader_compilation_v4_2_1_tile_a_loading WGSL kernel for Dx12...
[2025-06-23, 20:43:47.093] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.111] -> SUCCESS: test_shader_compilation_v4_2_1_tile_a_loading compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.112] -> 
--- Testing: test_shader_compilation_v4_2_2_tile_b_loading ---
[2025-06-23, 20:43:47.112] -> Attempting to compile test_shader_compilation_v4_2_2_tile_b_loading WGSL kernel for Dx12...
[2025-06-23, 20:43:47.116] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.116] -> ERROR: test_shader_compilation_v4_2_2_tile_b_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.117] -> 
--- Testing: test_shader_compilation_v4_2_3_inlined_tile_b ---
[2025-06-23, 20:43:47.117] -> Attempting to compile test_shader_compilation_v4_2_3_inlined_tile_b WGSL kernel for Dx12...
[2025-06-23, 20:43:47.120] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.143] -> SUCCESS: test_shader_compilation_v4_2_3_inlined_tile_b compute pipeline created successfully on Dx12!
[2025-06-23, 20:43:47.144] -> 
--- Testing: test_shader_compilation_v4_2_combined_loading ---
[2025-06-23, 20:43:47.144] -> Attempting to compile test_shader_compilation_v4_2_combined_loading WGSL kernel for Dx12...
[2025-06-23, 20:43:47.150] -> ERROR: test_shader_compilation_v4_2_combined_loading shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.150] -> ERROR: test_shader_compilation_v4_2_combined_loading pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.151] -> 
--- Testing: test_shader_compilation_v4_3_main_computation ---
[2025-06-23, 20:43:47.151] -> Attempting to compile test_shader_compilation_v4_3_main_computation WGSL kernel for Dx12...
[2025-06-23, 20:43:47.157] -> ERROR: test_shader_compilation_v4_3_main_computation shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.158] -> ERROR: test_shader_compilation_v4_3_main_computation pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.158] -> 
--- Testing: test_shader_compilation_v4_2_2_1_partial_write ---
[2025-06-23, 20:43:47.159] -> Attempting to compile test_shader_compilation_v4_2_2_1_partial_write WGSL kernel for Dx12...
[2025-06-23, 20:43:47.162] -> ERROR: test_shader_compilation_v4_2_2_1_partial_write shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.162] -> ERROR: test_shader_compilation_v4_2_2_1_partial_write pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.163] -> 
--- Testing: test_shader_compilation_v4_2_2_2_local_copy ---
[2025-06-23, 20:43:47.163] -> Attempting to compile test_shader_compilation_v4_2_2_2_local_copy WGSL kernel for Dx12...
[2025-06-23, 20:43:47.167] -> ERROR: test_shader_compilation_v4_2_2_2_local_copy shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.167] -> ERROR: test_shader_compilation_v4_2_2_2_local_copy pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.168] -> 
--- Testing: test_shader_compilation_v4_2_2_3_private_array ---
[2025-06-23, 20:43:47.168] -> Attempting to compile test_shader_compilation_v4_2_2_3_private_array WGSL kernel for Dx12...
[2025-06-23, 20:43:47.172] -> ERROR: test_shader_compilation_v4_2_2_3_private_array shader compilation failed: Validation Error: In Device::create_shader_module
[2025-06-23, 20:43:47.172] -> ERROR: test_shader_compilation_v4_2_2_3_private_array pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.173] -> 
--- Testing: test_shader_compilation_smoking_gun_unused_array ---
[2025-06-23, 20:43:47.173] -> Attempting to compile test_shader_compilation_smoking_gun_unused_array WGSL kernel for Dx12...
[2025-06-23, 20:43:47.174] -> SUCCESS: test_shader_compilation_smoking_gun_unused_array shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.176] -> ERROR: test_shader_compilation_smoking_gun_unused_array pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.177] -> 
--- Testing: test_shader_compilation_full_kernel_with_fix ---
[2025-06-23, 20:43:47.178] -> Attempting to compile test_shader_compilation_full_kernel_with_fix WGSL kernel for Dx12...
[2025-06-23, 20:43:47.185] -> SUCCESS: test_shader_compilation_full_kernel_with_fix shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.189] -> ERROR: test_shader_compilation_full_kernel_with_fix pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.189] -> 
--- Testing: test_shader_compilation_production_kernel ---
[2025-06-23, 20:43:47.190] -> Attempting to compile test_shader_compilation_production_kernel WGSL kernel for Dx12...
[2025-06-23, 20:43:47.203] -> SUCCESS: test_shader_compilation_production_kernel shader module compiled on Dx12 without error.
[2025-06-23, 20:43:47.207] -> ERROR: test_shader_compilation_production_kernel pipeline creation failed: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:47.073] -> 
--- Incremental Complexity Tests (V4 Base) ---
[2025-06-23, 20:43:47.177] -> 
--- Full Kernel and Production Tests ---
[2025-06-23, 20:43:47.216] -> SUCCESS: Fixed kernel compiled on Dx12.
[2025-06-23, 20:43:47.220] -> ERROR: Correctness test failed at PIPELINE CREATION: Validation Error: In Device::create_compute_pipeline
[2025-06-23, 20:43:46.992] -> Test status: FAIL
[2025-06-23, 20:43:46.999] -> Test status: FAIL
[2025-06-23, 20:43:47.005] -> Test status: FAIL
[2025-06-23, 20:43:47.011] -> Test status: FAIL
[2025-06-23, 20:43:47.033] -> Test status: PASS
[2025-06-23, 20:43:47.034] -> Test completed [PASS]
[2025-06-23, 20:43:47.042] -> Test status: PASS
[2025-06-23, 20:43:47.042] -> Test completed [PASS]
[2025-06-23, 20:43:47.058] -> Test status: PASS
[2025-06-23, 20:43:47.058] -> Test completed [PASS]
[2025-06-23, 20:43:47.072] -> Test status: PASS
[2025-06-23, 20:43:47.072] -> Test completed [PASS]
[2025-06-23, 20:43:47.089] -> Test status: PASS
[2025-06-23, 20:43:47.089] -> Test completed [PASS]
[2025-06-23, 20:43:47.111] -> Test status: PASS
[2025-06-23, 20:43:47.111] -> Test completed [PASS]
[2025-06-23, 20:43:47.117] -> Test status: FAIL
[2025-06-23, 20:43:47.143] -> Test status: PASS
[2025-06-23, 20:43:47.143] -> Test completed [PASS]
[2025-06-23, 20:43:47.150] -> Test status: FAIL
[2025-06-23, 20:43:47.158] -> Test status: FAIL
[2025-06-23, 20:43:47.189] -> Test status: FAIL
[2025-06-23, 20:43:47.208] -> Test status: FAIL
[2025-06-23, 20:43:47.163] -> Test status: FAIL
[2025-06-23, 20:43:47.167] -> Test status: FAIL
[2025-06-23, 20:43:47.172] -> Test status: FAIL
[2025-06-23, 20:43:47.177] -> Test status: FAIL
[2025-06-23, 20:43:47.208] -> 
--- Running Correctness Test on Dx12 with Fixed Kernel ---
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 28
- **Passed:** 15
- **Failed:** 13

### Timing Information

- **Total Time:** 0.35 sec
- **Average Time:** 12.00 ms

### Status

❌ 13 test(s) failed. See above for details.

---

_Report generated by BitNet Test Framework_
