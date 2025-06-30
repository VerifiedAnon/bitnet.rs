# KERNEL_TESTS_FASTEST Test Report

> Generated on: 2025-06-30 13:14:26

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Edge Case Invalid Input Weights                    | ✅ Pass |    0.00 ms |             |
|  2 | Error Handling GPU Unavailable                     | ✅ Pass |  377.00 ms |             |
|  3 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  386.00 ms |             |
|  4 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  361.00 ms |             |
|  5 | Cross Device Consistency Test                      | ✅ Pass |  480.00 ms |             |
|  6 | Kernel All Minus One Weights Test                  | ✅ Pass |  361.00 ms |             |
|  7 | Kernel All Plus One Weights Test                   | ✅ Pass |  378.00 ms |             |
|  8 | Kernel All Zero Test                               | ✅ Pass |  367.00 ms |             |
|  9 | Kernel Large Batch Test                            | ✅ Pass |  357.00 ms |             |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  366.00 ms |             |
| 11 | Low Level Kernel Correctness Test                  | ✅ Pass |  399.00 ms |             |
| 12 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   34.00 ms |             |
| 13 | Precision Test Fp Edge Cases                       | ✅ Pass |  393.00 ms |             |
| 14 | Streaming Load Test                                | ✅ Pass |  616.00 ms |             |
| 15 | Basic GPU Buffer Operations                        | ✅ Pass |  536.00 ms |             |
| 16 | Bitlinear Layer Forward Pass                       | ✅ Pass |  700.00 ms |             |
| 17 | GPU Kernel Dimensions                              | ✅ Pass |  381.00 ms |             |
| 18 | Matmul Quantized Scalar                            | ✅ Pass |    1.00 ms |             |
| 19 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    5.00 ms |             |
| 20 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |             |
| 21 | Unit Test Pack Ternary Weights                     | ✅ Pass |    3.00 ms |             |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-30, 13:14:16.838] -> Running unit_test_pack_ternary_weights...
[2025-06-30, 13:14:16.840] -> Packed value check: Expected=0b00100100100100100100100100100100, Got=0b00100100100100100100100100100100
[2025-06-30, 13:14:16.842] -> unit_test_pack_ternary_weights passed.
[2025-06-30, 13:14:16.842] -> Running unit_test_calculate_weight_scales...
[2025-06-30, 13:14:16.842] -> Scales check: Expected=[1.0, 1.0, 1.0], Got=[1.0, 1.0, 1.0]
[2025-06-30, 13:14:16.842] -> unit_test_calculate_weight_scales passed.
[2025-06-30, 13:14:16.843] -> Starting test_matmul_quantized_scalar...
[2025-06-30, 13:14:16.844] -> Scalar matmul check: Expected=[-0.5, -0.5], Got=[-0.5, -0.5]
[2025-06-30, 13:14:16.845] -> Testing basic GPU operations...
[2025-06-30, 13:14:17.371] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 13:14:17.381] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 13:14:17.381] -> Basic GPU operations test passed!
[2025-06-30, 13:14:17.774] -> Running low_level_kernel_correctness_test...
[2025-06-30, 13:14:17.774] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 13:14:17.808] -> [Profile] Buffer Setup: 18.76ms
[2025-06-30, 13:14:17.809] -> [Profile] Bind Group Setup: 264.50µs
[2025-06-30, 13:14:17.809] -> [Profile] Dispatch & Submit: 597.90µs
[2025-06-30, 13:14:17.810] -> [Profile] Readback (map/poll/copy): 179.70µs
[2025-06-30, 13:14:17.810] -> [Profile] Total launch_gpu_kernel Time: 20.49ms
[2025-06-30, 13:14:17.810] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 13:14:17.811] -> low_level_kernel_correctness_test passed.
[2025-06-30, 13:14:17.841] -> Running test_gpu_kernel_dimensions...
[2025-06-30, 13:14:17.842] -> Test dims: batch=1, in=16, out=2
[2025-06-30, 13:14:18.220] -> [Profile] Buffer Setup: 21.64ms
[2025-06-30, 13:14:18.220] -> [Profile] Bind Group Setup: 244.50µs
[2025-06-30, 13:14:18.221] -> [Profile] Dispatch & Submit: 560.20µs
[2025-06-30, 13:14:18.221] -> [Profile] Readback (map/poll/copy): 253.10µs
[2025-06-30, 13:14:18.222] -> [Profile] Total launch_gpu_kernel Time: 23.40ms
[2025-06-30, 13:14:18.222] -> GPU dimension test comparison: Expected[..2]=[8.0, 4.0], Got[..2]=[8.0, 4.0]
[2025-06-30, 13:14:18.222] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 13:14:18.274] -> Running kernel_large_batch_test...
[2025-06-30, 13:14:18.275] -> Test dims: batch=64, in=32, out=16
[2025-06-30, 13:14:18.629] -> [Profile] Buffer Setup: 8.44ms
[2025-06-30, 13:14:18.630] -> [Profile] Bind Group Setup: 233.40µs
[2025-06-30, 13:14:18.631] -> [Profile] Dispatch & Submit: 556.90µs
[2025-06-30, 13:14:18.631] -> [Profile] Readback (map/poll/copy): 229.70µs
[2025-06-30, 13:14:18.631] -> [Profile] Total launch_gpu_kernel Time: 10.19ms
[2025-06-30, 13:14:18.632] -> Large batch test comparison: Expected[..4]=[0.350263, 0.26761666, 1.6883463, 0.33452082], Got[..4]=[0.350263, 0.26761666, 1.6883463, 0.33452082]
[2025-06-30, 13:14:18.632] -> kernel_large_batch_test passed.
[2025-06-30, 13:14:18.662] -> Running kernel_all_zero_test...
[2025-06-30, 13:14:18.662] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 13:14:19.027] -> [Profile] Buffer Setup: 9.05ms
[2025-06-30, 13:14:19.027] -> [Profile] Bind Group Setup: 223.10µs
[2025-06-30, 13:14:19.028] -> [Profile] Dispatch & Submit: 573.50µs
[2025-06-30, 13:14:19.028] -> [Profile] Readback (map/poll/copy): 229.00µs
[2025-06-30, 13:14:19.028] -> [Profile] Total launch_gpu_kernel Time: 10.73ms
[2025-06-30, 13:14:19.029] -> All-zero test comparison: All outputs should be zero. Got[..4]=[1.601707, 1.601707, 1.601707, 1.601707]
[2025-06-30, 13:14:19.029] -> kernel_all_zero_test passed.
[2025-06-30, 13:14:19.060] -> Running kernel_all_plus_one_weights_test...
[2025-06-30, 13:14:19.060] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 13:14:19.436] -> [Profile] Buffer Setup: 10.40ms
[2025-06-30, 13:14:19.436] -> [Profile] Bind Group Setup: 252.00µs
[2025-06-30, 13:14:19.437] -> [Profile] Dispatch & Submit: 509.60µs
[2025-06-30, 13:14:19.437] -> [Profile] Readback (map/poll/copy): 214.90µs
[2025-06-30, 13:14:19.437] -> [Profile] Total launch_gpu_kernel Time: 12.10ms
[2025-06-30, 13:14:19.438] -> All-plus-one test comparison: Expected[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514], Got[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514]
[2025-06-30, 13:14:19.438] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 13:14:19.466] -> Running kernel_all_minus_one_weights_test...
[2025-06-30, 13:14:19.467] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 13:14:19.825] -> [Profile] Buffer Setup: 10.07ms
[2025-06-30, 13:14:19.826] -> [Profile] Bind Group Setup: 238.30µs
[2025-06-30, 13:14:19.827] -> [Profile] Dispatch & Submit: 520.00µs
[2025-06-30, 13:14:19.827] -> [Profile] Readback (map/poll/copy): 188.10µs
[2025-06-30, 13:14:19.827] -> [Profile] Total launch_gpu_kernel Time: 11.70ms
[2025-06-30, 13:14:19.827] -> All-minus-one test comparison: Expected[..4]=[0.0, 0.0, 0.0, 0.0], Got[..4]=[0.0, 0.0, 0.0, 0.0]
[2025-06-30, 13:14:19.828] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 13:14:19.859] -> Running kernel_non_divisible_batch_test...
[2025-06-30, 13:14:19.859] -> Test dims: batch=33, in=32, out=16
[2025-06-30, 13:14:20.223] -> [Profile] Buffer Setup: 10.77ms
[2025-06-30, 13:14:20.224] -> [Profile] Bind Group Setup: 253.50µs
[2025-06-30, 13:14:20.225] -> [Profile] Dispatch & Submit: 538.20µs
[2025-06-30, 13:14:20.225] -> [Profile] Readback (map/poll/copy): 202.60µs
[2025-06-30, 13:14:20.225] -> [Profile] Total launch_gpu_kernel Time: 12.45ms
[2025-06-30, 13:14:20.225] -> Non-divisible batch test comparison: Expected[..4]=[1.0184668, 0.90836227, 0.5308611, 0.20054752], Got[..4]=[1.0184668, 0.90836227, 0.5308611, 0.20054752]
[2025-06-30, 13:14:20.226] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 13:14:20.255] -> Running test_bitlinear_layer_forward_pass...
[2025-06-30, 13:14:20.955] -> BitLinear forward pass output length: 32768
[2025-06-30, 13:14:20.956] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 13:14:21.378] -> Timestamp query enabled with period: 1 ns/tick
[2025-06-30, 13:14:21.425] -> Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 343.788µs  | Total: 34.379ms  
  GPU (Kernel Time):  Avg: 46.418µs   | Total: 4.642ms   
  Scalar (CPU Time):  Avg: 120.480µs  | Total: 12.048ms  
Speedup (Wall vs Scalar):   0.35x
Speedup (Kernel vs Scalar): 2.60x
[2025-06-30, 13:14:22.121] -> Found 5 adapters. Running consistency test.
[2025-06-30, 13:14:22.121] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-30, 13:14:22.188] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 13:14:22.227] -> [Profile] Buffer Setup: 23.80ms
[2025-06-30, 13:14:22.227] -> [Profile] Bind Group Setup: 209.70µs
[2025-06-30, 13:14:22.228] -> [Profile] Dispatch & Submit: 613.80µs
[2025-06-30, 13:14:22.228] -> [Profile] Readback (map/poll/copy): 202.10µs
[2025-06-30, 13:14:22.228] -> [Profile] Total launch_gpu_kernel Time: 25.45ms
[2025-06-30, 13:14:22.229] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 13:14:22.277] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-30, 13:14:22.278] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 13:14:22.296] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-30, 13:14:22.297] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 13:14:22.331] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-30, 13:14:22.331] -> WARNING: Skipping test on "Microsoft Basic Render Driver" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 13:14:22.335] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-30, 13:14:22.343] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 13:14:22.359] -> [Profile] Buffer Setup: 1.77ms
[2025-06-30, 13:14:22.359] -> [Profile] Bind Group Setup: 77.00µs
[2025-06-30, 13:14:22.361] -> [Profile] Dispatch & Submit: 1.54ms
[2025-06-30, 13:14:22.362] -> [Profile] Readback (map/poll/copy): 1.25ms
[2025-06-30, 13:14:22.362] -> [Profile] Total launch_gpu_kernel Time: 5.24ms
[2025-06-30, 13:14:22.363] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 13:14:22.364] -> Kernel correctness passed on all available devices.
[2025-06-30, 13:14:22.986] -> Streaming Load Test (10 streams): Avg Latency: 2.956ms
[2025-06-30, 13:14:21.824] -> Original Activations for FP Edge Case Test: [1.0, -1.0, 0.0, 127.0, -127.0, 1e-6, 1000000.0, -1000000.0, 3.4028235e38, -3.4028235e38, 1.1920929e-7, NaN, inf, -inf, -10.0, 0.1]
[2025-06-30, 13:14:21.824] -> Quantized Activations from FP Edge Case Test: [0, 0, 0, 0, 0, 0, 0, 0, 127, -127, 0, 0, 127, -127, 0, 0]
[2025-06-30, 13:14:21.824] -> Activation Scale from FP Edge Case Test: 2679388700000000000000000000000000000
[2025-06-30, 13:14:21.849] -> [Profile] Buffer Setup: 10.15ms
[2025-06-30, 13:14:21.850] -> [Profile] Bind Group Setup: 278.20µs
[2025-06-30, 13:14:21.851] -> [Profile] Dispatch & Submit: 622.60µs
[2025-06-30, 13:14:21.851] -> [Profile] Readback (map/poll/copy): 182.20µs
[2025-06-30, 13:14:21.851] -> [Profile] Total launch_gpu_kernel Time: 12.01ms
[2025-06-30, 13:14:21.852] -> Precision test with FP edge cases (NaN, Infinity) passed.
[2025-06-30, 13:14:23.392] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-30, 13:14:23.393] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-30, 13:14:23.740] -> Memory safety test: Device max_buffer_size = 268435456. Calculated oversized batch size = 4194305.
[2025-06-30, 13:14:23.754] -> Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 13:14:23.363] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-30, 13:14:23.363] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 13:14:23.364] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 13:14:24.141] -> Skipping stress_test_maximum_dimension_support. (Set RUN_STRESS_TESTS=1 to run)
[2025-06-30, 13:14:24.099] -> Memory safety test (10GB): Attempting to allocate 10737418240 bytes.
[2025-06-30, 13:14:24.113] -> Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 13:14:16.836] -> Testing scalar packing-decoding symmetry...
[2025-06-30, 13:14:16.838] -> Original weights:  [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-30, 13:14:16.840] -> Decoded weights:   [0, 1, -1, 1, -1, -1, 1, 0, 0, 0, 1, 1, -1, -1, 1, -1]
[2025-06-30, 13:14:16.842] -> Scalar packing-decoding symmetry test passed.
[2025-06-30, 13:14:16.836] -> STARTING KERNEL TEST SUITE
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 21
- **Passed:** 21
- **Failed:** 0

### Timing Information

- **Total Time:** 6.51 sec
- **Average Time:** 310.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
