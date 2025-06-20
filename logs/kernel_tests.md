# KERNEL_TESTS Test Report

> Generated on: 2025-06-20 12:41:40

## Test Results

| No. | Test Name | Status | Time Taken |
|:---:|:----------|:------:|:----------:|
|  1 | Edge Case Invalid Input Weights                    | ✅ Pass |    0.00 ms |
|  2 | Error Handling GPU Unavailable                     | ✅ Pass |  395.00 ms |
|  3 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  395.00 ms |
|  4 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  377.00 ms |
|  5 | Cross Device Consistency Test                      | ✅ Pass |  421.00 ms |
|  6 | Kernel All Minus One Weights Test                  | ✅ Pass |  396.00 ms |
|  7 | Kernel All Plus One Weights Test                   | ✅ Pass |  402.00 ms |
|  8 | Kernel All Zero Test                               | ✅ Pass |  410.00 ms |
|  9 | Kernel Large Batch Test                            | ✅ Pass |  403.00 ms |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  402.00 ms |
| 11 | Low Level Kernel Correctness Test                  | ✅ Pass |  395.00 ms |
| 12 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   35.00 ms |
| 13 | Precision Test Fp Edge Cases                       | ✅ Pass |  379.00 ms |
| 14 | Streaming Load Test                                | ✅ Pass |  578.00 ms |
| 15 | Stress Test Maximum Dimension Support              | ✅ Pass |   2.29 sec |
| 16 | Basic GPU Buffer Operations                        | ✅ Pass |  649.00 ms |
| 17 | Bitlinear Layer Forward Pass                       | ✅ Pass |  705.00 ms |
| 18 | GPU Kernel Dimensions                              | ✅ Pass |  371.00 ms |
| 19 | Matmul Quantized Scalar                            | ✅ Pass |    1.00 ms |
| 20 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |
| 21 | Unit Test Pack Ternary Weights                     | ✅ Pass |    0.00 ms |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-20, 12:41:29.178] -> Running unit_test_pack_ternary_weights...
[2025-06-20, 12:41:29.178] -> Packed value check: Expected=0b00011000011000011000011000011000, Got=0b00011000011000011000011000011000
[2025-06-20, 12:41:29.178] -> unit_test_pack_ternary_weights passed.
[2025-06-20, 12:41:29.178] -> Running unit_test_calculate_weight_scales...
[2025-06-20, 12:41:29.179] -> Scales check: Expected=[1.0, 1.0, 1.0], Got=[1.0, 1.0, 1.0]
[2025-06-20, 12:41:29.179] -> unit_test_calculate_weight_scales passed.
[2025-06-20, 12:41:29.179] -> Starting test_matmul_quantized_scalar...
[2025-06-20, 12:41:29.180] -> Output values: [8.0, 4.0]
[2025-06-20, 12:41:29.181] -> Testing basic GPU operations...
[2025-06-20, 12:41:29.827] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-20, 12:41:29.831] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-20, 12:41:29.831] -> Basic GPU operations test passed!
[2025-06-20, 12:41:30.239] -> Running low_level_kernel_correctness_test...
[2025-06-20, 12:41:30.239] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 12:41:30.259] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 12:41:30.260] -> low_level_kernel_correctness_test passed.
[2025-06-20, 12:41:30.291] -> Running test_gpu_kernel_dimensions...
[2025-06-20, 12:41:30.291] -> Test dims: batch=1, in=16, out=2
[2025-06-20, 12:41:30.662] -> GPU dimension test comparison: Expected[..2]=[8.0, 4.0], Got[..2]=[8.0, 4.0]
[2025-06-20, 12:41:30.662] -> test_gpu_kernel_dimensions passed.
[2025-06-20, 12:41:30.694] -> Running kernel_large_batch_test...
[2025-06-20, 12:41:30.694] -> Test dims: batch=64, in=32, out=16
[2025-06-20, 12:41:31.096] -> Large batch test comparison: Expected[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784], Got[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784]
[2025-06-20, 12:41:31.097] -> kernel_large_batch_test passed.
[2025-06-20, 12:41:31.101] -> Running kernel_all_zero_test...
[2025-06-20, 12:41:31.101] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 12:41:31.512] -> All-zero test comparison: All outputs should be zero. Got[..4]=[1.601707, 1.601707, 1.601707, 1.601707]
[2025-06-20, 12:41:31.512] -> kernel_all_zero_test passed.
[2025-06-20, 12:41:31.516] -> Running kernel_all_plus_one_weights_test...
[2025-06-20, 12:41:31.517] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 12:41:31.919] -> All-plus-one test comparison: Expected[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514], Got[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514]
[2025-06-20, 12:41:31.919] -> kernel_all_plus_one_weights_test passed.
[2025-06-20, 12:41:31.923] -> Running kernel_all_minus_one_weights_test...
[2025-06-20, 12:41:31.923] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 12:41:32.318] -> All-minus-one test comparison: Expected[..4]=[0.0, 0.0, 0.0, 0.0], Got[..4]=[0.0, 0.0, 0.0, 0.0]
[2025-06-20, 12:41:32.319] -> kernel_all_minus_one_weights_test passed.
[2025-06-20, 12:41:32.323] -> Running kernel_non_divisible_batch_test...
[2025-06-20, 12:41:32.323] -> Test dims: batch=33, in=32, out=16
[2025-06-20, 12:41:32.726] -> Non-divisible batch test comparison: Expected[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603], Got[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603]
[2025-06-20, 12:41:32.726] -> kernel_non_divisible_batch_test passed.
[2025-06-20, 12:41:32.730] -> Running test_bitlinear_layer_forward_pass...
[2025-06-20, 12:41:33.435] -> BitLinear forward pass output length: 32768
[2025-06-20, 12:41:33.435] -> test_bitlinear_layer_forward_pass passed.
[2025-06-20, 12:41:33.841] -> Timestamp query enabled with period: 1 ns/tick
[2025-06-20, 12:41:33.889] -> Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 351.917µs  | Total: 35.192ms  
  GPU (Kernel Time):  Avg: 21.340µs   | Total: 2.134ms   
  Scalar (CPU Time):  Avg: 118.371µs  | Total: 11.837ms  
Speedup (Wall vs Scalar):   0.34x
Speedup (Kernel vs Scalar): 5.55x
[2025-06-20, 12:41:34.567] -> Found 5 adapters. Running consistency test.
[2025-06-20, 12:41:34.567] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-20, 12:41:34.621] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 12:41:34.638] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 12:41:34.665] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 12:41:34.665] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 12:41:34.699] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-20, 12:41:34.699] -> WARNING: Skipping test on "Microsoft Basic Render Driver" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 12:41:34.701] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 12:41:34.701] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 12:41:34.735] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-20, 12:41:34.736] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 12:41:34.753] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 12:41:34.754] -> Kernel correctness passed on all available devices.
[2025-06-20, 12:41:35.339] -> Streaming Load Test (10 streams): Avg Latency: 13.639ms
[2025-06-20, 12:41:34.283] -> Original Activations for FP Edge Case Test: [1.0, -1.0, 0.0, 127.0, -127.0, 1e-6, 1000000.0, -1000000.0, 3.4028235e38, -3.4028235e38, 1.1920929e-7, NaN, inf, -inf, -10.0, 0.1]
[2025-06-20, 12:41:34.283] -> Quantized Activations from FP Edge Case Test: [0, 0, 0, 0, 0, 0, 0, 0, 127, -127, 0, 0, 127, -127, 0, 0]
[2025-06-20, 12:41:34.283] -> Activation Scale from FP Edge Case Test: 2679388700000000000000000000000000000
[2025-06-20, 12:41:34.299] -> Precision test with FP edge cases (NaN, Infinity) passed.
[2025-06-20, 12:41:35.767] -> Running edge_case_invalid_input_weights...
[2025-06-20, 12:41:35.767] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-20, 12:41:35.767] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-20, 12:41:35.767] -> Running memory_safety_buffer_overflow_test...
[2025-06-20, 12:41:36.133] -> Memory safety test: Device max_buffer_size = 268435456. Calculated oversized batch size = 4194305.
[2025-06-20, 12:41:36.133] -> Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-20, 12:41:35.371] -> Running error_handling_gpu_unavailable...
[2025-06-20, 12:41:35.736] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-20, 12:41:35.737] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-20, 12:41:35.737] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-20, 12:41:35.766] -> error_handling_gpu_unavailable passed.
[2025-06-20, 12:41:36.540] -> Running stress_test_maximum_dimension_support...
[2025-06-20, 12:41:36.541] -> Stress Test: Initializing with large dimensions (1024x1024x1024)...
[2025-06-20, 12:41:36.890] -> Stress Test: Generating random data (this may take a moment)...
[2025-06-20, 12:41:37.535] -> Stress Test: Data generation complete. Time: 994.02ms
[2025-06-20, 12:41:37.535] -> Stress Test: Starting scalar pre-computation...
[2025-06-20, 12:41:37.622] -> Stress Test: Scalar pre-computation complete. Time: 86.32ms
[2025-06-20, 12:41:37.622] -> Stress Test: Starting GPU execution...
[2025-06-20, 12:41:37.729] -> Stress Test: GPU execution complete. Time: 106.91ms
[2025-06-20, 12:41:37.730] -> Stress Test: Starting scalar reference execution (this will be slow)...
[2025-06-20, 12:41:38.793] -> Stress Test: Scalar reference execution complete. Time: 1.06s
[2025-06-20, 12:41:38.794] -> Stress Test: Comparing results...
[2025-06-20, 12:41:38.827] -> Stress Test: Comparison complete. Test passed! Total time: 2.29s
[2025-06-20, 12:41:38.859] -> stress_test_maximum_dimension_support passed.
[2025-06-20, 12:41:36.162] -> Running memory_safety_hardcoded_large_allocation_test...
[2025-06-20, 12:41:36.510] -> Memory safety test (10GB): Attempting to allocate 10737418240 bytes.
[2025-06-20, 12:41:36.511] -> Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-20, 12:41:29.177] -> STARTING KERNEL TEST SUITE
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 21
- **Passed:** 21
- **Failed:** 0

### Timing Information

- **Total Time:** 9.01 sec
- **Average Time:** 428.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
