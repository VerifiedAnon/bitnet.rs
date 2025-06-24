# KERNEL_TESTS Test Report

> Generated on: 2025-06-23 13:45:29

## Test Results

| No. | Test Name | Status | Time Taken |
|:---:|:----------|:------:|:----------:|
|  1 | Cross Device Consistency Test                      | ✅ Pass |  427.00 ms |
|  2 | Kernel All Minus One Weights Test                  | ✅ Pass |  365.00 ms |
|  3 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    2.00 ms |
|  4 | Kernel All Plus One Weights Test                   | ✅ Pass |  368.00 ms |
|  5 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    3.00 ms |
|  6 | Kernel All Zero Test                               | ✅ Pass |  372.00 ms |
|  7 | Kernel All Zero Test Warm                          | ✅ Pass |    2.00 ms |
|  8 | Kernel Large Batch Test                            | ✅ Pass |  369.00 ms |
|  9 | Kernel Large Batch Test Warm                       | ✅ Pass |    2.00 ms |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  364.00 ms |
| 11 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    2.00 ms |
| 12 | Low Level Kernel Correctness Test                  | ✅ Pass |  368.00 ms |
| 13 | Low Level Kernel Correctness Test Warm             | ✅ Pass |    5.00 ms |
| 14 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   35.00 ms |
| 15 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |   59.00 ms |
| 16 | Precision Test Fp Edge Cases                       | ✅ Pass |  367.00 ms |
| 17 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    1.00 ms |
| 18 | Streaming Load Test                                | ✅ Pass |  617.00 ms |
| 19 | Streaming Load Test Warm                           | ✅ Pass |    6.00 ms |
| 20 | Temp Warm Harness Proof Of Concept                 | ✅ Pass |  717.00 ms |
| 21 | Basic GPU Buffer Operations                        | ✅ Pass |  387.00 ms |
| 22 | Bitlinear Layer Forward Pass                       | ✅ Pass |  815.00 ms |
| 23 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  348.00 ms |
| 24 | GPU Kernel Dimensions                              | ✅ Pass |  373.00 ms |
| 25 | GPU Kernel Dimensions Warm                         | ✅ Pass |    1.00 ms |
| 26 | Matmul Quantized Scalar                            | ✅ Pass |    0.00 ms |
| 27 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    5.00 ms |
| 28 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |
| 29 | Unit Test Pack Ternary Weights                     | ✅ Pass |    0.00 ms |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-23, 13:45:20.073] -> Running unit_test_pack_ternary_weights...
[2025-06-23, 13:45:20.073] -> Packed value check: Expected=0b00011000011000011000011000011000, Got=0b00011000011000011000011000011000
[2025-06-23, 13:45:20.073] -> unit_test_pack_ternary_weights passed.
[2025-06-23, 13:45:20.073] -> Running unit_test_calculate_weight_scales...
[2025-06-23, 13:45:20.074] -> Scales check: Expected=[1.0, 1.0, 1.0], Got=[1.0, 1.0, 1.0]
[2025-06-23, 13:45:20.074] -> unit_test_calculate_weight_scales passed.
[2025-06-23, 13:45:20.074] -> Starting test_matmul_quantized_scalar...
[2025-06-23, 13:45:20.074] -> Scalar matmul check: Expected=[8.0, 8.0], Got=[8.0, 8.0]
[2025-06-23, 13:45:20.076] -> Testing basic GPU operations...
[2025-06-23, 13:45:20.460] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-23, 13:45:20.463] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-23, 13:45:20.463] -> Basic GPU operations test passed!
[2025-06-23, 13:45:20.861] -> Running low_level_kernel_correctness_test...
[2025-06-23, 13:45:20.862] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-23, 13:45:20.863] -> [Profile] Buffer Setup: 1.32ms
[2025-06-23, 13:45:20.864] -> [Profile] Bind Group Setup: 178.20µs
[2025-06-23, 13:45:20.864] -> [Profile] Dispatch & Submit: 619.30µs
[2025-06-23, 13:45:20.865] -> [Profile] Readback (map/poll/copy): 163.60µs
[2025-06-23, 13:45:20.865] -> [Profile] Total launch_gpu_kernel Time: 2.98ms
[2025-06-23, 13:45:20.865] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-23, 13:45:20.865] -> low_level_kernel_correctness_test passed.
[2025-06-23, 13:45:27.218] -> [WARM] Running low_level_kernel_correctness_test...
[2025-06-23, 13:45:27.219] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-23, 13:45:27.221] -> [Profile] Buffer Setup: 2.38ms
[2025-06-23, 13:45:27.222] -> [Profile] Bind Group Setup: 127.20µs
[2025-06-23, 13:45:27.222] -> [Profile] Dispatch & Submit: 662.40µs
[2025-06-23, 13:45:27.223] -> [Profile] Readback (map/poll/copy): 267.50µs
[2025-06-23, 13:45:27.223] -> [Profile] Total launch_gpu_kernel Time: 4.12ms
[2025-06-23, 13:45:27.224] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-23, 13:45:27.224] -> [WARM] low_level_kernel_correctness_test passed.
[2025-06-23, 13:45:20.899] -> Running test_gpu_kernel_dimensions...
[2025-06-23, 13:45:20.899] -> Test dims: batch=1, in=16, out=2
[2025-06-23, 13:45:21.270] -> [Profile] Buffer Setup: 2.20ms
[2025-06-23, 13:45:21.271] -> [Profile] Bind Group Setup: 246.00µs
[2025-06-23, 13:45:21.271] -> [Profile] Dispatch & Submit: 609.70µs
[2025-06-23, 13:45:21.272] -> [Profile] Readback (map/poll/copy): 174.20µs
[2025-06-23, 13:45:21.272] -> [Profile] Total launch_gpu_kernel Time: 3.91ms
[2025-06-23, 13:45:21.272] -> GPU dimension test comparison: Expected[..2]=[8.0, 8.0], Got[..2]=[8.0, 8.0]
[2025-06-23, 13:45:21.272] -> test_gpu_kernel_dimensions passed.
[2025-06-23, 13:45:27.224] -> [WARM] Running test_gpu_kernel_dimensions...
[2025-06-23, 13:45:27.224] -> [Profile] Buffer Setup: 261.70µs
[2025-06-23, 13:45:27.225] -> [Profile] Bind Group Setup: 100.20µs
[2025-06-23, 13:45:27.225] -> [Profile] Dispatch & Submit: 408.60µs
[2025-06-23, 13:45:27.225] -> [Profile] Readback (map/poll/copy): 196.60µs
[2025-06-23, 13:45:27.226] -> [Profile] Total launch_gpu_kernel Time: 1.49ms
[2025-06-23, 13:45:27.226] -> [WARM] GPU dimension test comparison: Expected[..2]=[8.0, 8.0], Got[..2]=[8.0, 8.0]
[2025-06-23, 13:45:27.226] -> [WARM] test_gpu_kernel_dimensions passed.
[2025-06-23, 13:45:21.303] -> Running kernel_large_batch_test...
[2025-06-23, 13:45:21.303] -> Test dims: batch=64, in=32, out=16
[2025-06-23, 13:45:21.670] -> [Profile] Buffer Setup: 1.38ms
[2025-06-23, 13:45:21.671] -> [Profile] Bind Group Setup: 235.90µs
[2025-06-23, 13:45:21.672] -> [Profile] Dispatch & Submit: 843.80µs
[2025-06-23, 13:45:21.672] -> [Profile] Readback (map/poll/copy): 109.80µs
[2025-06-23, 13:45:21.672] -> [Profile] Total launch_gpu_kernel Time: 3.31ms
[2025-06-23, 13:45:21.672] -> Large batch test comparison: Expected[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784], Got[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784]
[2025-06-23, 13:45:21.673] -> kernel_large_batch_test passed.
[2025-06-23, 13:45:27.226] -> [WARM] Running kernel_large_batch_test...
[2025-06-23, 13:45:27.228] -> [Profile] Buffer Setup: 236.20µs
[2025-06-23, 13:45:27.228] -> [Profile] Bind Group Setup: 60.50µs
[2025-06-23, 13:45:27.228] -> [Profile] Dispatch & Submit: 286.70µs
[2025-06-23, 13:45:27.228] -> [Profile] Readback (map/poll/copy): 82.30µs
[2025-06-23, 13:45:27.228] -> [Profile] Total launch_gpu_kernel Time: 1.19ms
[2025-06-23, 13:45:27.229] -> [WARM] Large batch test comparison: Expected[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784], Got[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784]
[2025-06-23, 13:45:27.229] -> [WARM] kernel_large_batch_test passed.
[2025-06-23, 13:45:21.705] -> Running kernel_all_zero_test...
[2025-06-23, 13:45:21.705] -> Test dims: batch=32, in=32, out=16
[2025-06-23, 13:45:22.075] -> [Profile] Buffer Setup: 2.26ms
[2025-06-23, 13:45:22.076] -> [Profile] Bind Group Setup: 194.60µs
[2025-06-23, 13:45:22.076] -> [Profile] Dispatch & Submit: 628.70µs
[2025-06-23, 13:45:22.077] -> [Profile] Readback (map/poll/copy): 334.00µs
[2025-06-23, 13:45:22.077] -> [Profile] Total launch_gpu_kernel Time: 4.20ms
[2025-06-23, 13:45:22.077] -> All-zero test comparison: All outputs should be zero. Got[..4]=[1.601707, 1.601707, 1.601707, 1.601707]
[2025-06-23, 13:45:22.077] -> kernel_all_zero_test passed.
[2025-06-23, 13:45:27.229] -> [WARM] Running kernel_all_zero_test...
[2025-06-23, 13:45:27.230] -> [Profile] Buffer Setup: 244.30µs
[2025-06-23, 13:45:27.230] -> [Profile] Bind Group Setup: 57.20µs
[2025-06-23, 13:45:27.230] -> [Profile] Dispatch & Submit: 266.30µs
[2025-06-23, 13:45:27.231] -> [Profile] Readback (map/poll/copy): 71.60µs
[2025-06-23, 13:45:27.231] -> [Profile] Total launch_gpu_kernel Time: 1.17ms
[2025-06-23, 13:45:27.231] -> [WARM] kernel_all_zero_test passed.
[2025-06-23, 13:45:22.110] -> Running kernel_all_plus_one_weights_test...
[2025-06-23, 13:45:22.111] -> Test dims: batch=32, in=32, out=16
[2025-06-23, 13:45:22.476] -> [Profile] Buffer Setup: 2.57ms
[2025-06-23, 13:45:22.477] -> [Profile] Bind Group Setup: 169.10µs
[2025-06-23, 13:45:22.477] -> [Profile] Dispatch & Submit: 627.70µs
[2025-06-23, 13:45:22.478] -> [Profile] Readback (map/poll/copy): 105.60µs
[2025-06-23, 13:45:22.478] -> [Profile] Total launch_gpu_kernel Time: 4.32ms
[2025-06-23, 13:45:22.478] -> All-plus-one test comparison: Expected[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514], Got[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514]
[2025-06-23, 13:45:22.478] -> kernel_all_plus_one_weights_test passed.
[2025-06-23, 13:45:27.231] -> [WARM] Running kernel_all_plus_one_weights_test...
[2025-06-23, 13:45:27.232] -> [Profile] Buffer Setup: 278.90µs
[2025-06-23, 13:45:27.232] -> [Profile] Bind Group Setup: 145.20µs
[2025-06-23, 13:45:27.233] -> [Profile] Dispatch & Submit: 296.20µs
[2025-06-23, 13:45:27.233] -> [Profile] Readback (map/poll/copy): 134.00µs
[2025-06-23, 13:45:27.233] -> [Profile] Total launch_gpu_kernel Time: 1.58ms
[2025-06-23, 13:45:27.234] -> [WARM] kernel_all_plus_one_weights_test passed.
[2025-06-23, 13:45:22.511] -> Running kernel_all_minus_one_weights_test...
[2025-06-23, 13:45:22.511] -> Test dims: batch=32, in=32, out=16
[2025-06-23, 13:45:22.874] -> [Profile] Buffer Setup: 1.57ms
[2025-06-23, 13:45:22.874] -> [Profile] Bind Group Setup: 254.20µs
[2025-06-23, 13:45:22.875] -> [Profile] Dispatch & Submit: 782.70µs
[2025-06-23, 13:45:22.876] -> [Profile] Readback (map/poll/copy): 107.00µs
[2025-06-23, 13:45:22.876] -> [Profile] Total launch_gpu_kernel Time: 3.52ms
[2025-06-23, 13:45:22.876] -> All-minus-one test comparison: Expected[..4]=[0.0, 0.0, 0.0, 0.0], Got[..4]=[0.0, 0.0, 0.0, 0.0]
[2025-06-23, 13:45:22.876] -> kernel_all_minus_one_weights_test passed.
[2025-06-23, 13:45:27.234] -> [WARM] Running kernel_all_minus_one_weights_test...
[2025-06-23, 13:45:27.235] -> [Profile] Buffer Setup: 250.90µs
[2025-06-23, 13:45:27.236] -> [Profile] Bind Group Setup: 79.00µs
[2025-06-23, 13:45:27.236] -> [Profile] Dispatch & Submit: 337.60µs
[2025-06-23, 13:45:27.236] -> [Profile] Readback (map/poll/copy): 79.50µs
[2025-06-23, 13:45:27.236] -> [Profile] Total launch_gpu_kernel Time: 1.28ms
[2025-06-23, 13:45:27.236] -> [WARM] kernel_all_minus_one_weights_test passed.
[2025-06-23, 13:45:22.906] -> Running kernel_non_divisible_batch_test...
[2025-06-23, 13:45:22.906] -> Test dims: batch=33, in=32, out=16
[2025-06-23, 13:45:23.269] -> [Profile] Buffer Setup: 2.51ms
[2025-06-23, 13:45:23.269] -> [Profile] Bind Group Setup: 204.80µs
[2025-06-23, 13:45:23.270] -> [Profile] Dispatch & Submit: 652.50µs
[2025-06-23, 13:45:23.270] -> [Profile] Readback (map/poll/copy): 173.20µs
[2025-06-23, 13:45:23.270] -> [Profile] Total launch_gpu_kernel Time: 4.22ms
[2025-06-23, 13:45:23.271] -> Non-divisible batch test comparison: Expected[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603], Got[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603]
[2025-06-23, 13:45:23.271] -> kernel_non_divisible_batch_test passed.
[2025-06-23, 13:45:27.237] -> [WARM] Running kernel_non_divisible_batch_test...
[2025-06-23, 13:45:27.238] -> [Profile] Buffer Setup: 227.50µs
[2025-06-23, 13:45:27.238] -> [Profile] Bind Group Setup: 60.90µs
[2025-06-23, 13:45:27.238] -> [Profile] Dispatch & Submit: 280.20µs
[2025-06-23, 13:45:27.238] -> [Profile] Readback (map/poll/copy): 71.90µs
[2025-06-23, 13:45:27.239] -> [Profile] Total launch_gpu_kernel Time: 1.14ms
[2025-06-23, 13:45:27.239] -> [WARM] kernel_non_divisible_batch_test passed.
[2025-06-23, 13:45:23.304] -> Running test_bitlinear_layer_forward_pass...
[2025-06-23, 13:45:24.119] -> BitLinear forward pass output length: 32768
[2025-06-23, 13:45:24.119] -> test_bitlinear_layer_forward_pass passed.
[2025-06-23, 13:45:27.239] -> [WARM] Running test_bitlinear_layer_forward_pass...
[2025-06-23, 13:45:27.587] -> [WARM] BitLinear forward pass output length: 32768
[2025-06-23, 13:45:27.587] -> [WARM] test_bitlinear_layer_forward_pass passed.
[2025-06-23, 13:45:24.533] -> Timestamp query enabled with period: 1 ns/tick
[2025-06-23, 13:45:24.581] -> Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 359.313µs  | Total: 35.931ms  
  GPU (Kernel Time):  Avg: 21.520µs   | Total: 2.152ms   
  Scalar (CPU Time):  Avg: 114.183µs  | Total: 11.418ms  
Speedup (Wall vs Scalar):   0.32x
Speedup (Kernel vs Scalar): 5.31x
[2025-06-23, 13:45:27.648] -> [WARM] Performance Benchmark: Avg Wall Time: 594.198µs
[2025-06-23, 13:45:25.249] -> Found 5 adapters. Running consistency test.
[2025-06-23, 13:45:25.250] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-23, 13:45:25.332] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-23, 13:45:25.334] -> [Profile] Buffer Setup: 1.68ms
[2025-06-23, 13:45:25.334] -> [Profile] Bind Group Setup: 159.80µs
[2025-06-23, 13:45:25.335] -> [Profile] Dispatch & Submit: 710.60µs
[2025-06-23, 13:45:25.335] -> [Profile] Readback (map/poll/copy): 105.10µs
[2025-06-23, 13:45:25.335] -> [Profile] Total launch_gpu_kernel Time: 3.34ms
[2025-06-23, 13:45:25.336] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-23, 13:45:25.364] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-23, 13:45:25.364] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-23, 13:45:25.385] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-23, 13:45:25.385] -> WARNING: Skipping test on "Microsoft Basic Render Driver" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-23, 13:45:25.386] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-23, 13:45:25.386] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-23, 13:45:25.411] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-23, 13:45:25.429] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-23, 13:45:25.431] -> [Profile] Buffer Setup: 1.64ms
[2025-06-23, 13:45:25.432] -> [Profile] Bind Group Setup: 106.90µs
[2025-06-23, 13:45:25.434] -> [Profile] Dispatch & Submit: 1.90ms
[2025-06-23, 13:45:25.435] -> [Profile] Readback (map/poll/copy): 1.47ms
[2025-06-23, 13:45:25.435] -> [Profile] Total launch_gpu_kernel Time: 5.70ms
[2025-06-23, 13:45:25.436] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-23, 13:45:25.437] -> Kernel correctness passed on all available devices.
[2025-06-23, 13:45:26.060] -> Streaming Load Test (10 streams): Avg Latency: 1.344ms
[2025-06-23, 13:45:27.657] -> [WARM] Streaming Load Test (10 streams) passed.
[2025-06-23, 13:45:24.956] -> Original Activations for FP Edge Case Test: [1.0, -1.0, 0.0, 127.0, -127.0, 1e-6, 1000000.0, -1000000.0, 3.4028235e38, -3.4028235e38, 1.1920929e-7, NaN, inf, -inf, -10.0, 0.1]
[2025-06-23, 13:45:24.957] -> Quantized Activations from FP Edge Case Test: [0, 0, 0, 0, 0, 0, 0, 0, 127, -127, 0, 0, 127, -127, 0, 0]
[2025-06-23, 13:45:24.957] -> Activation Scale from FP Edge Case Test: 2679388700000000000000000000000000000
[2025-06-23, 13:45:24.978] -> [Profile] Buffer Setup: 2.13ms
[2025-06-23, 13:45:24.978] -> [Profile] Bind Group Setup: 141.80µs
[2025-06-23, 13:45:24.979] -> [Profile] Dispatch & Submit: 733.70µs
[2025-06-23, 13:45:24.980] -> [Profile] Readback (map/poll/copy): 110.10µs
[2025-06-23, 13:45:24.980] -> [Profile] Total launch_gpu_kernel Time: 3.92ms
[2025-06-23, 13:45:24.980] -> Precision test with FP edge cases (NaN, Infinity) passed.
[2025-06-23, 13:45:27.649] -> [Profile] Buffer Setup: 194.50µs
[2025-06-23, 13:45:27.649] -> [Profile] Bind Group Setup: 61.30µs
[2025-06-23, 13:45:27.650] -> [Profile] Dispatch & Submit: 588.40µs
[2025-06-23, 13:45:27.650] -> [Profile] Readback (map/poll/copy): 90.40µs
[2025-06-23, 13:45:27.650] -> [Profile] Total launch_gpu_kernel Time: 1.59ms
[2025-06-23, 13:45:27.650] -> [WARM] Precision test with FP edge cases passed.
[2025-06-23, 13:45:26.464] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-23, 13:45:26.464] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-23, 13:45:26.806] -> Memory safety test: Device max_buffer_size = 268435456. Calculated oversized batch size = 4194305.
[2025-06-23, 13:45:26.825] -> Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-23, 13:45:27.657] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-23, 13:45:26.431] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-23, 13:45:26.432] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-23, 13:45:26.432] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-23, 13:45:27.657] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-23, 13:45:19.350] -> Testing scalar packing-decoding symmetry...
[2025-06-23, 13:45:19.352] -> Original weights:  [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-23, 13:45:19.353] -> Decoded weights:   [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-23, 13:45:19.354] -> Scalar packing-decoding symmetry test passed.
[2025-06-23, 13:45:19.351] -> Running temp_warm_harness_proof_of_concept...
[2025-06-23, 13:45:19.352] -> Creating WarmGpuContext...
[2025-06-23, 13:45:20.061] -> WarmGpuContext created successfully.
[2025-06-23, 13:45:20.061] -> Running correctness logic using the warm context...
[2025-06-23, 13:45:20.061] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-23, 13:45:20.064] -> [Profile] Buffer Setup: 2.38ms
[2025-06-23, 13:45:20.065] -> [Profile] Bind Group Setup: 822.90µs
[2025-06-23, 13:45:20.066] -> [Profile] Dispatch & Submit: 990.50µs
[2025-06-23, 13:45:20.066] -> [Profile] Readback (map/poll/copy): 157.20µs
[2025-06-23, 13:45:20.066] -> [Profile] Total launch_gpu_kernel Time: 5.04ms
[2025-06-23, 13:45:20.068] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-23, 13:45:20.068] -> Temp warm harness proof-of-concept PASSED.
[2025-06-23, 13:45:20.072] -> STARTING KERNEL TEST SUITE
[2025-06-23, 13:45:20.072] -> --- STARTING COLD RUN (INDIVIDUAL TESTS) ---
[2025-06-23, 13:45:26.854] -> --- STARTING WARM RUN (SHARED CONTEXT) ---
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 29
- **Passed:** 29
- **Failed:** 0

### Timing Information

- **Total Time:** 6.39 sec
- **Average Time:** 220.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
