# KERNEL_TESTS Test Report

> Generated on: 2025-06-24 13:38:14

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Cross Device Consistency Test                      | ✅ Pass |  24.69 sec |             |
|  2 | Kernel All Minus One Weights Test                  | ✅ Pass |  349.00 ms |             |
|  3 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    3.00 ms |             |
|  4 | Kernel All Plus One Weights Test                   | ✅ Pass |  358.00 ms |             |
|  5 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    3.00 ms |             |
|  6 | Kernel All Zero Test                               | ✅ Pass |  343.00 ms |             |
|  7 | Kernel All Zero Test Warm                          | ✅ Pass |    3.00 ms |             |
|  8 | Kernel Large Batch Test                            | ✅ Pass |  351.00 ms |             |
|  9 | Kernel Large Batch Test Warm                       | ✅ Pass |    3.00 ms |             |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  356.00 ms |             |
| 11 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    3.00 ms |             |
| 12 | Low Level Kernel Correctness Test                  | ✅ Pass |  370.00 ms |             |
| 13 | Low Level Kernel Correctness Test Warm             | ✅ Pass |    5.00 ms |             |
| 14 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  368.00 ms |             |
| 15 | Memory Safety Buffer Overflow Test Warm            | ✅ Pass |    0.00 ms |             |
| 16 | Memory Safety Hardcoded Large Allocation Test Warm | ✅ Pass |    0.00 ms |             |
| 17 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |  521.00 ms |             |
| 18 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |  173.00 ms |             |
| 19 | Precision Test Fp Edge Cases                       | ✅ Pass |  363.00 ms |             |
| 20 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    2.00 ms |             |
| 21 | Streaming Load Test                                | ✅ Pass |  417.00 ms |             |
| 22 | Streaming Load Test Warm                           | ✅ Pass |   16.00 ms |             |
| 23 | Stress Test Maximum Dimension Support Warm         | ✅ Pass |   1.92 sec |             |
| 24 | Basic GPU Buffer Operations                        | ✅ Pass |  568.00 ms |             |
| 25 | Bitlinear Layer Forward Pass                       | ✅ Pass |  698.00 ms |             |
| 26 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  344.00 ms |             |
| 27 | GPU Kernel Dimensions                              | ✅ Pass |  355.00 ms |             |
| 28 | GPU Kernel Dimensions Warm                         | ✅ Pass |    1.00 ms |             |
| 29 | Matmul Quantized Scalar                            | ✅ Pass |    2.00 ms |             |
| 30 | Matmul Quantized Scalar Warm                       | ✅ Pass |    0.00 ms |             |
| 31 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    0.00 ms |             |
| 32 | Unit Test Calculate Weight Scales                  | ✅ Pass |    1.00 ms |             |
| 33 | Unit Test Calculate Weight Scales Warm             | ✅ Pass |    0.00 ms |             |
| 34 | Unit Test Pack Ternary Weights                     | ✅ Pass |    2.00 ms |             |
| 35 | Unit Test Pack Ternary Weights Warm                | ✅ Pass |    0.00 ms |             |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-24, 13:37:38.618] -> Running unit_test_pack_ternary_weights...
[2025-06-24, 13:37:38.620] -> unit_test_pack_ternary_weights passed.
[2025-06-24, 13:37:38.622] -> Running unit_test_calculate_weight_scales...
[2025-06-24, 13:37:38.624] -> unit_test_calculate_weight_scales passed.
[2025-06-24, 13:37:38.625] -> Starting test_matmul_quantized_scalar...
[2025-06-24, 13:37:38.627] -> test_matmul_quantized_scalar passed.
[2025-06-24, 13:37:38.630] -> Testing basic GPU operations...
[2025-06-24, 13:37:39.196] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-24, 13:37:39.198] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-24, 13:37:39.198] -> Basic GPU operations test passed!
[2025-06-24, 13:37:39.593] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-24, 13:37:39.595] -> [Profile] Buffer Setup: 2.09ms
[2025-06-24, 13:37:39.596] -> [Profile] Bind Group Setup: 622.00µs
[2025-06-24, 13:37:39.597] -> [Profile] Dispatch & Submit: 628.70µs
[2025-06-24, 13:37:39.597] -> [Profile] Readback (map/poll/copy): 175.70µs
[2025-06-24, 13:37:39.597] -> [Profile] Total launch_gpu_kernel Time: 4.13ms
[2025-06-24, 13:37:39.598] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-24, 13:37:39.598] -> low_level_kernel_correctness_test passed.
[2025-06-24, 13:38:09.617] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-24, 13:38:09.620] -> [Profile] Buffer Setup: 2.88ms
[2025-06-24, 13:38:09.620] -> [Profile] Bind Group Setup: 168.80µs
[2025-06-24, 13:38:09.621] -> [Profile] Dispatch & Submit: 1.04ms
[2025-06-24, 13:38:09.622] -> [Profile] Readback (map/poll/copy): 100.40µs
[2025-06-24, 13:38:09.622] -> [Profile] Total launch_gpu_kernel Time: 4.93ms
[2025-06-24, 13:38:09.622] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-24, 13:37:39.601] -> Running test_gpu_kernel_dimensions...
[2025-06-24, 13:37:39.954] -> [Profile] Buffer Setup: 1.39ms
[2025-06-24, 13:37:39.954] -> [Profile] Bind Group Setup: 225.00µs
[2025-06-24, 13:37:39.955] -> [Profile] Dispatch & Submit: 763.60µs
[2025-06-24, 13:37:39.955] -> [Profile] Readback (map/poll/copy): 209.70µs
[2025-06-24, 13:37:39.956] -> [Profile] Total launch_gpu_kernel Time: 3.34ms
[2025-06-24, 13:37:39.956] -> test_gpu_kernel_dimensions passed.
[2025-06-24, 13:38:09.623] -> [Profile] Buffer Setup: 184.00µs
[2025-06-24, 13:38:09.623] -> [Profile] Bind Group Setup: 85.80µs
[2025-06-24, 13:38:09.623] -> [Profile] Dispatch & Submit: 285.40µs
[2025-06-24, 13:38:09.624] -> [Profile] Readback (map/poll/copy): 167.80µs
[2025-06-24, 13:38:09.624] -> [Profile] Total launch_gpu_kernel Time: 1.25ms
[2025-06-24, 13:37:39.959] -> Running kernel_large_batch_test...
[2025-06-24, 13:37:40.308] -> [Profile] Buffer Setup: 1.38ms
[2025-06-24, 13:37:40.308] -> [Profile] Bind Group Setup: 157.00µs
[2025-06-24, 13:37:40.309] -> [Profile] Dispatch & Submit: 852.20µs
[2025-06-24, 13:37:40.309] -> [Profile] Readback (map/poll/copy): 194.40µs
[2025-06-24, 13:37:40.310] -> [Profile] Total launch_gpu_kernel Time: 3.27ms
[2025-06-24, 13:37:40.310] -> kernel_large_batch_test passed.
[2025-06-24, 13:38:09.626] -> [Profile] Buffer Setup: 845.90µs
[2025-06-24, 13:38:09.627] -> [Profile] Bind Group Setup: 58.40µs
[2025-06-24, 13:38:09.627] -> [Profile] Dispatch & Submit: 247.00µs
[2025-06-24, 13:38:09.627] -> [Profile] Readback (map/poll/copy): 73.50µs
[2025-06-24, 13:38:09.627] -> [Profile] Total launch_gpu_kernel Time: 1.83ms
[2025-06-24, 13:37:40.313] -> Running kernel_all_zero_test...
[2025-06-24, 13:37:40.654] -> [Profile] Buffer Setup: 1.45ms
[2025-06-24, 13:37:40.654] -> [Profile] Bind Group Setup: 172.50µs
[2025-06-24, 13:37:40.655] -> [Profile] Dispatch & Submit: 665.40µs
[2025-06-24, 13:37:40.656] -> [Profile] Readback (map/poll/copy): 108.00µs
[2025-06-24, 13:37:40.656] -> [Profile] Total launch_gpu_kernel Time: 3.23ms
[2025-06-24, 13:37:40.657] -> kernel_all_zero_test passed.
[2025-06-24, 13:38:09.630] -> [Profile] Buffer Setup: 991.30µs
[2025-06-24, 13:38:09.630] -> [Profile] Bind Group Setup: 63.50µs
[2025-06-24, 13:38:09.631] -> [Profile] Dispatch & Submit: 416.60µs
[2025-06-24, 13:38:09.631] -> [Profile] Readback (map/poll/copy): 136.90µs
[2025-06-24, 13:38:09.631] -> [Profile] Total launch_gpu_kernel Time: 2.15ms
[2025-06-24, 13:37:40.660] -> Running kernel_all_plus_one_weights_test...
[2025-06-24, 13:37:41.015] -> [Profile] Buffer Setup: 1.40ms
[2025-06-24, 13:37:41.016] -> [Profile] Bind Group Setup: 156.40µs
[2025-06-24, 13:37:41.016] -> [Profile] Dispatch & Submit: 579.50µs
[2025-06-24, 13:37:41.017] -> [Profile] Readback (map/poll/copy): 219.50µs
[2025-06-24, 13:37:41.017] -> [Profile] Total launch_gpu_kernel Time: 3.07ms
[2025-06-24, 13:37:41.017] -> kernel_all_plus_one_weights_test passed.
[2025-06-24, 13:38:09.633] -> [Profile] Buffer Setup: 866.60µs
[2025-06-24, 13:38:09.633] -> [Profile] Bind Group Setup: 111.70µs
[2025-06-24, 13:38:09.634] -> [Profile] Dispatch & Submit: 316.60µs
[2025-06-24, 13:38:09.634] -> [Profile] Readback (map/poll/copy): 84.20µs
[2025-06-24, 13:38:09.634] -> [Profile] Total launch_gpu_kernel Time: 2.01ms
[2025-06-24, 13:37:41.020] -> Running kernel_all_minus_one_weights_test...
[2025-06-24, 13:37:41.367] -> [Profile] Buffer Setup: 1.49ms
[2025-06-24, 13:37:41.368] -> [Profile] Bind Group Setup: 155.90µs
[2025-06-24, 13:37:41.369] -> [Profile] Dispatch & Submit: 711.80µs
[2025-06-24, 13:37:41.369] -> [Profile] Readback (map/poll/copy): 138.20µs
[2025-06-24, 13:37:41.369] -> [Profile] Total launch_gpu_kernel Time: 3.17ms
[2025-06-24, 13:37:41.370] -> kernel_all_minus_one_weights_test passed.
[2025-06-24, 13:38:09.636] -> [Profile] Buffer Setup: 806.90µs
[2025-06-24, 13:38:09.637] -> [Profile] Bind Group Setup: 178.40µs
[2025-06-24, 13:38:09.637] -> [Profile] Dispatch & Submit: 366.30µs
[2025-06-24, 13:38:09.638] -> [Profile] Readback (map/poll/copy): 280.30µs
[2025-06-24, 13:38:09.638] -> [Profile] Total launch_gpu_kernel Time: 2.26ms
[2025-06-24, 13:37:41.373] -> Running kernel_non_divisible_batch_test...
[2025-06-24, 13:37:41.727] -> [Profile] Buffer Setup: 1.32ms
[2025-06-24, 13:37:41.727] -> [Profile] Bind Group Setup: 151.60µs
[2025-06-24, 13:37:41.728] -> [Profile] Dispatch & Submit: 607.80µs
[2025-06-24, 13:37:41.728] -> [Profile] Readback (map/poll/copy): 156.40µs
[2025-06-24, 13:37:41.728] -> [Profile] Total launch_gpu_kernel Time: 2.95ms
[2025-06-24, 13:37:41.729] -> kernel_non_divisible_batch_test passed.
[2025-06-24, 13:38:09.640] -> [Profile] Buffer Setup: 904.10µs
[2025-06-24, 13:38:09.641] -> [Profile] Bind Group Setup: 78.00µs
[2025-06-24, 13:38:09.641] -> [Profile] Dispatch & Submit: 265.80µs
[2025-06-24, 13:38:09.641] -> [Profile] Readback (map/poll/copy): 77.10µs
[2025-06-24, 13:38:09.641] -> [Profile] Total launch_gpu_kernel Time: 1.91ms
[2025-06-24, 13:37:41.732] -> Running test_bitlinear_layer_forward_pass...
[2025-06-24, 13:37:42.430] -> test_bitlinear_layer_forward_pass passed.
[2025-06-24, 13:37:42.434] -> Running performance_benchmark_gpu_vs_scalar...
[2025-06-24, 13:37:42.955] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-24, 13:37:43.326] -> Starting cross-device consistency test...
[2025-06-24, 13:37:43.327] -> Calculating scalar reference result...
[2025-06-24, 13:37:43.327] -> Scalar reference calculation complete.
[2025-06-24, 13:37:43.569] -> Found 5 adapters. Running per-device subtests.
[2025-06-24, 13:37:43.569] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-24, 13:37:43.641] -> [Profile] Buffer Setup: 1.27ms
[2025-06-24, 13:37:43.642] -> [Profile] Bind Group Setup: 245.20µs
[2025-06-24, 13:37:43.643] -> [Profile] Dispatch & Submit: 665.40µs
[2025-06-24, 13:37:43.643] -> [Profile] Readback (map/poll/copy): 158.70µs
[2025-06-24, 13:37:43.643] -> [Profile] Total launch_gpu_kernel Time: 3.08ms
[2025-06-24, 13:37:43.669] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-24, 13:37:43.670] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-24, 13:37:55.731] -> [Profile] Buffer Setup: 17.42ms
[2025-06-24, 13:37:55.731] -> [Profile] Bind Group Setup: 175.40µs
[2025-06-24, 13:37:55.735] -> [Profile] Dispatch & Submit: 3.31ms
[2025-06-24, 13:37:55.737] -> [Profile] Readback (map/poll/copy): 1.76ms
[2025-06-24, 13:37:55.737] -> [Profile] Total launch_gpu_kernel Time: 23.36ms
[2025-06-24, 13:37:55.770] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-24, 13:37:55.770] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-24, 13:37:55.771] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-24, 13:37:55.771] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-24, 13:38:07.945] -> [Profile] Buffer Setup: 10.57ms
[2025-06-24, 13:38:07.945] -> [Profile] Bind Group Setup: 198.60µs
[2025-06-24, 13:38:07.949] -> [Profile] Dispatch & Submit: 3.25ms
[2025-06-24, 13:38:07.950] -> [Profile] Readback (map/poll/copy): 1.56ms
[2025-06-24, 13:38:07.951] -> [Profile] Total launch_gpu_kernel Time: 16.26ms
[2025-06-24, 13:38:07.987] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-24, 13:38:07.987] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-24, 13:38:08.004] -> [Profile] Buffer Setup: 1.82ms
[2025-06-24, 13:38:08.005] -> [Profile] Bind Group Setup: 117.20µs
[2025-06-24, 13:38:08.007] -> [Profile] Dispatch & Submit: 2.21ms
[2025-06-24, 13:38:08.013] -> [Profile] Readback (map/poll/copy): 5.64ms
[2025-06-24, 13:38:08.013] -> [Profile] Total launch_gpu_kernel Time: 10.53ms
[2025-06-24, 13:38:08.015] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-24, 13:38:08.015] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-24, 13:38:08.084] -> Running streaming_load_test...
[2025-06-24, 13:38:08.502] -> streaming_load_test passed.
[2025-06-24, 13:37:42.959] -> Running precision_test_fp_edge_cases...
[2025-06-24, 13:37:43.320] -> [Profile] Buffer Setup: 1.22ms
[2025-06-24, 13:37:43.320] -> [Profile] Bind Group Setup: 251.80µs
[2025-06-24, 13:37:43.321] -> [Profile] Dispatch & Submit: 810.50µs
[2025-06-24, 13:37:43.322] -> [Profile] Readback (map/poll/copy): 242.20µs
[2025-06-24, 13:37:43.322] -> [Profile] Total launch_gpu_kernel Time: 3.45ms
[2025-06-24, 13:37:43.322] -> precision_test_fp_edge_cases passed.
[2025-06-24, 13:38:10.161] -> [Profile] Buffer Setup: 771.60µs
[2025-06-24, 13:38:10.161] -> [Profile] Bind Group Setup: 96.90µs
[2025-06-24, 13:38:10.162] -> [Profile] Dispatch & Submit: 321.20µs
[2025-06-24, 13:38:10.162] -> [Profile] Readback (map/poll/copy): 144.40µs
[2025-06-24, 13:38:10.162] -> [Profile] Total launch_gpu_kernel Time: 1.85ms
[2025-06-24, 13:38:08.880] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-24, 13:38:08.881] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-24, 13:38:08.881] -> Running memory_safety_buffer_overflow_test...
[2025-06-24, 13:38:09.250] -> memory_safety_buffer_overflow_test passed.
[2025-06-24, 13:38:10.180] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-24, 13:38:08.850] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-24, 13:38:08.850] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-24, 13:38:08.851] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-24, 13:38:10.923] -> [Profile] Buffer Setup: 10.36ms
[2025-06-24, 13:38:10.924] -> [Profile] Bind Group Setup: 189.10µs
[2025-06-24, 13:38:10.925] -> [Profile] Dispatch & Submit: 549.20µs
[2025-06-24, 13:38:11.007] -> [Profile] Readback (map/poll/copy): 82.13ms
[2025-06-24, 13:38:11.007] -> [Profile] Total launch_gpu_kernel Time: 94.23ms
[2025-06-24, 13:38:10.180] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-24, 13:38:09.252] -> Testing scalar packing-decoding symmetry...
[2025-06-24, 13:38:09.252] -> Original weights:  [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-24, 13:38:09.253] -> Decoded weights:   [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-24, 13:38:09.253] -> Scalar packing-decoding symmetry test passed.
[2025-06-24, 13:37:38.617] -> STARTING KERNEL TEST SUITE
[2025-06-24, 13:37:38.618] -> --- STARTING COLD RUN (INDIVIDUAL TESTS) ---
[2025-06-24, 13:38:09.253] -> --- STARTING WARM RUN (SHARED CONTEXT) ---
[2025-06-24, 13:38:09.616] -> [WARM] unit_test_pack_ternary_weights passed.
[2025-06-24, 13:38:09.622] -> [WARM] low_level_kernel_correctness_test passed.
[2025-06-24, 13:38:09.624] -> [WARM] test_gpu_kernel_dimensions passed.
[2025-06-24, 13:38:09.628] -> [WARM] kernel_large_batch_test passed.
[2025-06-24, 13:37:40.656] -> kernel_all_zero_test passed.
[2025-06-24, 13:38:09.632] -> [WARM] kernel_all_zero_test passed.
[2025-06-24, 13:38:09.616] -> [WARM] unit_test_calculate_weight_scales passed.
[2025-06-24, 13:38:09.617] -> [WARM] test_matmul_quantized_scalar passed.
[2025-06-24, 13:37:41.017] -> kernel_all_plus_one_weights_test passed.
[2025-06-24, 13:38:09.635] -> [WARM] kernel_all_plus_one_weights_test passed.
[2025-06-24, 13:37:41.370] -> kernel_all_minus_one_weights_test passed.
[2025-06-24, 13:38:09.638] -> [WARM] kernel_all_minus_one_weights_test passed.
[2025-06-24, 13:37:41.729] -> kernel_non_divisible_batch_test passed.
[2025-06-24, 13:38:09.642] -> [WARM] kernel_non_divisible_batch_test passed.
[2025-06-24, 13:38:09.987] -> [WARM] test_bitlinear_layer_forward_pass passed.
[2025-06-24, 13:38:10.160] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.602ms    | Total: 160.235ms 
  Scalar (CPU Time):  Avg: 112.948µs  | Total: 11.295ms  
Speedup (Wall vs Scalar):   0.07x
[2025-06-24, 13:38:10.160] -> [WARM] performance_benchmark_gpu_vs_scalar passed.
[2025-06-24, 13:38:10.163] -> [WARM] precision_test_fp_edge_cases passed.
[2025-06-24, 13:38:10.179] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.558ms
[2025-06-24, 13:38:12.102] -> [WARM] stress_test_maximum_dimension_support passed.
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 35
- **Passed:** 35
- **Failed:** 0

### Timing Information

- **Total Time:** 32.60 sec
- **Average Time:** 931.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
