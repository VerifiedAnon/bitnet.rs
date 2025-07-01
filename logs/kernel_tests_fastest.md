# KERNEL_TESTS_FASTEST Test Report

> Generated on: 2025-06-30 20:23:32

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Edge Case Invalid Input Weights                    | ✅ Pass |    0.00 ms |             |
|  2 | Error Handling GPU Unavailable                     | ✅ Pass |  382.00 ms |             |
|  3 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  409.00 ms |             |
|  4 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  409.00 ms |             |
|  5 | Cross Device Consistency Test                      | ✅ Pass |  480.00 ms |             |
|  6 | Kernel All Minus One Weights Test                  | ✅ Pass |  417.00 ms |             |
|  7 | Kernel All Plus One Weights Test                   | ✅ Pass |  393.00 ms |             |
|  8 | Kernel All Zero Test                               | ✅ Pass |  417.00 ms |             |
|  9 | Kernel Large Batch Test                            | ✅ Pass |  399.00 ms |             |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  408.00 ms |             |
| 11 | Low Level Kernel Correctness Test                  | ✅ Pass |  458.00 ms |             |
| 12 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   33.00 ms |             |
| 13 | Precision Test Fp Edge Cases                       | ✅ Pass |  408.00 ms |             |
| 14 | Streaming Load Test                                | ✅ Pass |  666.00 ms |             |
| 15 | Basic GPU Buffer Operations                        | ✅ Pass |  637.00 ms |             |
| 16 | Bitlinear Layer Forward Pass                       | ✅ Pass |  725.00 ms |             |
| 17 | GPU Kernel Dimensions                              | ✅ Pass |  407.00 ms |             |
| 18 | Matmul Quantized Scalar                            | ✅ Pass |    1.00 ms |             |
| 19 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    6.00 ms |             |
| 20 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |             |
| 21 | Unit Test Pack Ternary Weights                     | ✅ Pass |    2.00 ms |             |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-30, 20:23:22.337] -> Running unit_test_pack_ternary_weights...
[2025-06-30, 20:23:22.338] -> Packed value check: Expected=0b10010010010010010010010010010010, Got=0b10010010010010010010010010010010
[2025-06-30, 20:23:22.339] -> unit_test_pack_ternary_weights passed.
[2025-06-30, 20:23:22.340] -> Running unit_test_calculate_weight_scales...
[2025-06-30, 20:23:22.340] -> Scales check: Expected=[1.0, 1.0, 1.0], Got=[1.0, 1.0, 1.0]
[2025-06-30, 20:23:22.340] -> unit_test_calculate_weight_scales passed.
[2025-06-30, 20:23:22.340] -> Starting test_matmul_quantized_scalar...
[2025-06-30, 20:23:22.341] -> Scalar matmul check: Expected=[-1.5, -1.5], Got=[-1.5, -1.5]
[2025-06-30, 20:23:22.343] -> Testing basic GPU operations...
[2025-06-30, 20:23:22.958] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 20:23:22.981] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 20:23:22.981] -> Basic GPU operations test passed!
[2025-06-30, 20:23:23.434] -> Running low_level_kernel_correctness_test...
[2025-06-30, 20:23:23.434] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:23:23.468] -> [Profile] Buffer Setup: 18.32ms
[2025-06-30, 20:23:23.469] -> [Profile] Bind Group Setup: 253.10µs
[2025-06-30, 20:23:23.470] -> [Profile] Dispatch & Submit: 631.00µs
[2025-06-30, 20:23:23.470] -> [Profile] Readback (map/poll/copy): 192.60µs
[2025-06-30, 20:23:23.470] -> [Profile] Total launch_gpu_kernel Time: 20.05ms
[2025-06-30, 20:23:23.471] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:23:23.471] -> low_level_kernel_correctness_test passed.
[2025-06-30, 20:23:23.505] -> Running test_gpu_kernel_dimensions...
[2025-06-30, 20:23:23.506] -> Test dims: batch=1, in=16, out=2
[2025-06-30, 20:23:23.910] -> [Profile] Buffer Setup: 10.31ms
[2025-06-30, 20:23:23.911] -> [Profile] Bind Group Setup: 249.50µs
[2025-06-30, 20:23:23.912] -> [Profile] Dispatch & Submit: 667.10µs
[2025-06-30, 20:23:23.912] -> [Profile] Readback (map/poll/copy): 174.70µs
[2025-06-30, 20:23:23.912] -> [Profile] Total launch_gpu_kernel Time: 12.16ms
[2025-06-30, 20:23:23.913] -> GPU dimension test comparison: Expected[..2]=[8.0, 4.0], Got[..2]=[8.0, 4.0]
[2025-06-30, 20:23:23.913] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 20:23:23.945] -> Running kernel_large_batch_test...
[2025-06-30, 20:23:23.945] -> Test dims: batch=64, in=32, out=16
[2025-06-30, 20:23:24.342] -> [Profile] Buffer Setup: 10.50ms
[2025-06-30, 20:23:24.342] -> [Profile] Bind Group Setup: 145.90µs
[2025-06-30, 20:23:24.343] -> [Profile] Dispatch & Submit: 621.90µs
[2025-06-30, 20:23:24.344] -> [Profile] Readback (map/poll/copy): 267.10µs
[2025-06-30, 20:23:24.344] -> [Profile] Total launch_gpu_kernel Time: 12.23ms
[2025-06-30, 20:23:24.344] -> Large batch test comparison: Expected[..4]=[-1.4482784, 0.75955904, -1.101951, -2.3849368], Got[..4]=[-1.4482784, 0.75955904, -1.101951, -2.3849368]
[2025-06-30, 20:23:24.344] -> kernel_large_batch_test passed.
[2025-06-30, 20:23:24.375] -> Running kernel_all_zero_test...
[2025-06-30, 20:23:24.375] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 20:23:24.789] -> [Profile] Buffer Setup: 21.65ms
[2025-06-30, 20:23:24.789] -> [Profile] Bind Group Setup: 227.30µs
[2025-06-30, 20:23:24.790] -> [Profile] Dispatch & Submit: 675.40µs
[2025-06-30, 20:23:24.791] -> [Profile] Readback (map/poll/copy): 298.50µs
[2025-06-30, 20:23:24.791] -> [Profile] Total launch_gpu_kernel Time: 23.67ms
[2025-06-30, 20:23:24.792] -> All-zero test comparison: All outputs should be zero. Got[..4]=[0.0, 0.0, 0.0, 0.0]
[2025-06-30, 20:23:24.792] -> kernel_all_zero_test passed.
[2025-06-30, 20:23:24.846] -> Running kernel_all_plus_one_weights_test...
[2025-06-30, 20:23:24.847] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 20:23:25.237] -> [Profile] Buffer Setup: 9.33ms
[2025-06-30, 20:23:25.238] -> [Profile] Bind Group Setup: 323.80µs
[2025-06-30, 20:23:25.238] -> [Profile] Dispatch & Submit: 637.90µs
[2025-06-30, 20:23:25.239] -> [Profile] Readback (map/poll/copy): 264.10µs
[2025-06-30, 20:23:25.239] -> [Profile] Total launch_gpu_kernel Time: 11.50ms
[2025-06-30, 20:23:25.240] -> All-plus-one test comparison: Expected[..4]=[-2.4678514, -2.4678514, -2.4678514, -2.4678514], Got[..4]=[-2.4678514, -2.4678514, -2.4678514, -2.4678514]
[2025-06-30, 20:23:25.240] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 20:23:25.278] -> Running kernel_all_minus_one_weights_test...
[2025-06-30, 20:23:25.278] -> Test dims: batch=32, in=32, out=16
[2025-06-30, 20:23:25.693] -> [Profile] Buffer Setup: 9.75ms
[2025-06-30, 20:23:25.694] -> [Profile] Bind Group Setup: 256.20µs
[2025-06-30, 20:23:25.694] -> [Profile] Dispatch & Submit: 564.00µs
[2025-06-30, 20:23:25.695] -> [Profile] Readback (map/poll/copy): 199.70µs
[2025-06-30, 20:23:25.695] -> [Profile] Total launch_gpu_kernel Time: 11.43ms
[2025-06-30, 20:23:25.695] -> All-minus-one test comparison: Expected[..4]=[0.68914646, 0.68914646, 0.68914646, 0.68914646], Got[..4]=[0.68914646, 0.68914646, 0.68914646, 0.68914646]
[2025-06-30, 20:23:25.695] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 20:23:25.727] -> Running kernel_non_divisible_batch_test...
[2025-06-30, 20:23:25.727] -> Test dims: batch=33, in=32, out=16
[2025-06-30, 20:23:26.133] -> [Profile] Buffer Setup: 10.19ms
[2025-06-30, 20:23:26.133] -> [Profile] Bind Group Setup: 245.30µs
[2025-06-30, 20:23:26.134] -> [Profile] Dispatch & Submit: 522.30µs
[2025-06-30, 20:23:26.134] -> [Profile] Readback (map/poll/copy): 207.00µs
[2025-06-30, 20:23:26.134] -> [Profile] Total launch_gpu_kernel Time: 11.82ms
[2025-06-30, 20:23:26.135] -> Non-divisible batch test comparison: Expected[..4]=[0.44435036, 0.33424586, 0.49940264, 2.870582], Got[..4]=[0.44435036, 0.33424586, 0.49940264, 2.870582]
[2025-06-30, 20:23:26.135] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 20:23:26.167] -> Running test_bitlinear_layer_forward_pass...
[2025-06-30, 20:23:26.891] -> BitLinear forward pass output length: 32768
[2025-06-30, 20:23:26.892] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 20:23:27.349] -> Timestamp query enabled with period: 1 ns/tick
[2025-06-30, 20:23:27.395] -> Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 332.164µs  | Total: 33.216ms  
  GPU (Kernel Time):  Avg: 46.280µs   | Total: 4.628ms   
  Scalar (CPU Time):  Avg: 118.410µs  | Total: 11.841ms  
Speedup (Wall vs Scalar):   0.36x
Speedup (Kernel vs Scalar): 2.56x
[2025-06-30, 20:23:28.120] -> Found 5 adapters. Running consistency test.
[2025-06-30, 20:23:28.120] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-30, 20:23:28.203] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:23:28.227] -> [Profile] Buffer Setup: 9.66ms
[2025-06-30, 20:23:28.228] -> [Profile] Bind Group Setup: 206.00µs
[2025-06-30, 20:23:28.229] -> [Profile] Dispatch & Submit: 625.50µs
[2025-06-30, 20:23:28.229] -> [Profile] Readback (map/poll/copy): 230.00µs
[2025-06-30, 20:23:28.229] -> [Profile] Total launch_gpu_kernel Time: 11.37ms
[2025-06-30, 20:23:28.230] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:23:28.255] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-30, 20:23:28.256] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 20:23:28.277] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-30, 20:23:28.277] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 20:23:28.312] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-30, 20:23:28.313] -> WARNING: Skipping test on "Microsoft Basic Render Driver" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-30, 20:23:28.316] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-30, 20:23:28.323] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:23:28.339] -> [Profile] Buffer Setup: 1.75ms
[2025-06-30, 20:23:28.340] -> [Profile] Bind Group Setup: 164.00µs
[2025-06-30, 20:23:28.341] -> [Profile] Dispatch & Submit: 1.66ms
[2025-06-30, 20:23:28.345] -> [Profile] Readback (map/poll/copy): 3.23ms
[2025-06-30, 20:23:28.345] -> [Profile] Total launch_gpu_kernel Time: 7.69ms
[2025-06-30, 20:23:28.346] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:23:28.347] -> Kernel correctness passed on all available devices.
[2025-06-30, 20:23:29.019] -> Streaming Load Test (10 streams): Avg Latency: 3.932ms
[2025-06-30, 20:23:27.805] -> Original Activations for FP Edge Case Test: [1.0, -1.0, 0.0, 127.0, -127.0, 1e-6, 1000000.0, -1000000.0, 3.4028235e38, -3.4028235e38, 1.1920929e-7, NaN, inf, -inf, -10.0, 0.1]
[2025-06-30, 20:23:27.806] -> Quantized Activations from FP Edge Case Test: [0, 0, 0, 0, 0, 0, 0, 0, 127, -127, 0, 0, 127, -127, 0, 0]
[2025-06-30, 20:23:27.806] -> Activation Scale from FP Edge Case Test: 2679388700000000000000000000000000000
[2025-06-30, 20:23:27.833] -> [Profile] Buffer Setup: 12.81ms
[2025-06-30, 20:23:27.833] -> [Profile] Bind Group Setup: 172.90µs
[2025-06-30, 20:23:27.834] -> [Profile] Dispatch & Submit: 645.60µs
[2025-06-30, 20:23:27.834] -> [Profile] Readback (map/poll/copy): 189.20µs
[2025-06-30, 20:23:27.835] -> [Profile] Total launch_gpu_kernel Time: 14.47ms
[2025-06-30, 20:23:27.835] -> Precision test with FP edge cases (NaN, Infinity) passed.
[2025-06-30, 20:23:29.459] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-30, 20:23:29.460] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-30, 20:23:29.826] -> Memory safety test: Device max_buffer_size = 268435456. Calculated oversized batch size = 4194305.
[2025-06-30, 20:23:29.840] -> Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 20:23:29.431] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-30, 20:23:29.431] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 20:23:29.432] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 20:23:30.279] -> Skipping stress_test_maximum_dimension_support. (Set RUN_STRESS_TESTS=1 to run)
[2025-06-30, 20:23:30.236] -> Memory safety test (10GB): Attempting to allocate 10737418240 bytes.
[2025-06-30, 20:23:30.251] -> Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 20:23:22.333] -> Testing scalar packing-decoding symmetry...
[2025-06-30, 20:23:22.337] -> Original weights:  [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-30, 20:23:22.338] -> Decoded weights:   [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-30, 20:23:22.339] -> Scalar packing-decoding symmetry test passed.
[2025-06-30, 20:23:22.333] -> STARTING KERNEL TEST SUITE
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 21
- **Passed:** 21
- **Failed:** 0

### Timing Information

- **Total Time:** 7.07 sec
- **Average Time:** 336.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
