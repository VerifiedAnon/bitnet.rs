# KERNEL_TESTS Test Report

> Generated on: 2025-06-20 14:52:14

## Test Results

| No. | Test Name | Status | Time Taken |
|:---:|:----------|:------:|:----------:|
|  1 | Cross Device Consistency Test                      | ✅ Pass |  430.00 ms |
|  2 | Kernel All Minus One Weights Test                  | ✅ Pass |  363.00 ms |
|  3 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    2.00 ms |
|  4 | Kernel All Plus One Weights Test                   | ✅ Pass |  375.00 ms |
|  5 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    2.00 ms |
|  6 | Kernel All Zero Test                               | ✅ Pass |  384.00 ms |
|  7 | Kernel All Zero Test Warm                          | ✅ Pass |    2.00 ms |
|  8 | Kernel Large Batch Test                            | ✅ Pass |  378.00 ms |
|  9 | Kernel Large Batch Test Warm                       | ✅ Pass |    2.00 ms |
| 10 | Kernel Non Divisible Batch Test                    | ✅ Pass |  369.00 ms |
| 11 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    2.00 ms |
| 12 | Low Level Kernel Correctness Test                  | ✅ Pass |  380.00 ms |
| 13 | Low Level Kernel Correctness Test Warm             | ✅ Pass |    5.00 ms |
| 14 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   35.00 ms |
| 15 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |   57.00 ms |
| 16 | Precision Test Fp Edge Cases                       | ✅ Pass |  366.00 ms |
| 17 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    1.00 ms |
| 18 | Streaming Load Test                                | ✅ Pass |  640.00 ms |
| 19 | Streaming Load Test Warm                           | ✅ Pass |    7.00 ms |
| 20 | Stress Test Maximum Dimension Support Warm         | ✅ Pass |  847.00 ms |
| 21 | Basic GPU Buffer Operations                        | ✅ Pass |  544.00 ms |
| 22 | Bitlinear Layer Forward Pass                       | ✅ Pass |  685.00 ms |
| 23 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  345.00 ms |
| 24 | GPU Kernel Dimensions                              | ✅ Pass |  372.00 ms |
| 25 | GPU Kernel Dimensions Warm                         | ✅ Pass |    1.00 ms |
| 26 | Matmul Quantized Scalar                            | ✅ Pass |    1.00 ms |
| 27 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |
| 28 | Unit Test Pack Ternary Weights                     | ✅ Pass |    0.00 ms |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-20, 14:52:04.122] -> Running unit_test_pack_ternary_weights...
[2025-06-20, 14:52:04.122] -> Packed value check: Expected=0b00011000011000011000011000011000, Got=0b00011000011000011000011000011000
[2025-06-20, 14:52:04.122] -> unit_test_pack_ternary_weights passed.
[2025-06-20, 14:52:04.122] -> Running unit_test_calculate_weight_scales...
[2025-06-20, 14:52:04.122] -> Scales check: Expected=[1.0, 1.0, 1.0], Got=[1.0, 1.0, 1.0]
[2025-06-20, 14:52:04.123] -> unit_test_calculate_weight_scales passed.
[2025-06-20, 14:52:04.123] -> Starting test_matmul_quantized_scalar...
[2025-06-20, 14:52:04.124] -> Scalar matmul check: Expected=[8.0, 8.0], Got=[8.0, 8.0]
[2025-06-20, 14:52:04.125] -> Testing basic GPU operations...
[2025-06-20, 14:52:04.667] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-20, 14:52:04.670] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-20, 14:52:04.670] -> Basic GPU operations test passed!
[2025-06-20, 14:52:05.079] -> Running low_level_kernel_correctness_test...
[2025-06-20, 14:52:05.080] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 14:52:05.082] -> [Profile] Buffer Setup: 2.01ms
[2025-06-20, 14:52:05.083] -> [Profile] Bind Group Setup: 603.80µs
[2025-06-20, 14:52:05.083] -> [Profile] Dispatch & Submit: 607.40µs
[2025-06-20, 14:52:05.084] -> [Profile] Readback (map/poll/copy): 276.70µs
[2025-06-20, 14:52:05.084] -> [Profile] Total launch_gpu_kernel Time: 3.86ms
[2025-06-20, 14:52:05.084] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 14:52:05.084] -> low_level_kernel_correctness_test passed.
[2025-06-20, 14:52:11.414] -> [WARM] Running low_level_kernel_correctness_test...
[2025-06-20, 14:52:11.414] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 14:52:11.417] -> [Profile] Buffer Setup: 2.48ms
[2025-06-20, 14:52:11.418] -> [Profile] Bind Group Setup: 154.20µs
[2025-06-20, 14:52:11.418] -> [Profile] Dispatch & Submit: 642.70µs
[2025-06-20, 14:52:11.419] -> [Profile] Readback (map/poll/copy): 311.20µs
[2025-06-20, 14:52:11.419] -> [Profile] Total launch_gpu_kernel Time: 4.09ms
[2025-06-20, 14:52:11.419] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 14:52:11.419] -> [WARM] low_level_kernel_correctness_test passed.
[2025-06-20, 14:52:05.114] -> Running test_gpu_kernel_dimensions...
[2025-06-20, 14:52:05.115] -> Test dims: batch=1, in=16, out=2
[2025-06-20, 14:52:05.485] -> [Profile] Buffer Setup: 1.92ms
[2025-06-20, 14:52:05.486] -> [Profile] Bind Group Setup: 232.80µs
[2025-06-20, 14:52:05.486] -> [Profile] Dispatch & Submit: 560.80µs
[2025-06-20, 14:52:05.487] -> [Profile] Readback (map/poll/copy): 94.10µs
[2025-06-20, 14:52:05.487] -> [Profile] Total launch_gpu_kernel Time: 3.46ms
[2025-06-20, 14:52:05.487] -> GPU dimension test comparison: Expected[..2]=[8.0, 8.0], Got[..2]=[8.0, 8.0]
[2025-06-20, 14:52:05.487] -> test_gpu_kernel_dimensions passed.
[2025-06-20, 14:52:11.419] -> [WARM] Running test_gpu_kernel_dimensions...
[2025-06-20, 14:52:11.420] -> [Profile] Buffer Setup: 174.00µs
[2025-06-20, 14:52:11.420] -> [Profile] Bind Group Setup: 65.50µs
[2025-06-20, 14:52:11.421] -> [Profile] Dispatch & Submit: 458.00µs
[2025-06-20, 14:52:11.421] -> [Profile] Readback (map/poll/copy): 80.20µs
[2025-06-20, 14:52:11.421] -> [Profile] Total launch_gpu_kernel Time: 1.28ms
[2025-06-20, 14:52:11.421] -> [WARM] GPU dimension test comparison: Expected[..2]=[8.0, 8.0], Got[..2]=[8.0, 8.0]
[2025-06-20, 14:52:11.421] -> [WARM] test_gpu_kernel_dimensions passed.
[2025-06-20, 14:52:05.519] -> Running kernel_large_batch_test...
[2025-06-20, 14:52:05.520] -> Test dims: batch=64, in=32, out=16
[2025-06-20, 14:52:05.896] -> [Profile] Buffer Setup: 1.97ms
[2025-06-20, 14:52:05.897] -> [Profile] Bind Group Setup: 249.60µs
[2025-06-20, 14:52:05.897] -> [Profile] Dispatch & Submit: 594.20µs
[2025-06-20, 14:52:05.898] -> [Profile] Readback (map/poll/copy): 152.00µs
[2025-06-20, 14:52:05.898] -> [Profile] Total launch_gpu_kernel Time: 3.64ms
[2025-06-20, 14:52:05.898] -> Large batch test comparison: Expected[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784], Got[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784]
[2025-06-20, 14:52:05.898] -> kernel_large_batch_test passed.
[2025-06-20, 14:52:11.421] -> [WARM] Running kernel_large_batch_test...
[2025-06-20, 14:52:11.423] -> [Profile] Buffer Setup: 236.20µs
[2025-06-20, 14:52:11.423] -> [Profile] Bind Group Setup: 67.80µs
[2025-06-20, 14:52:11.424] -> [Profile] Dispatch & Submit: 273.90µs
[2025-06-20, 14:52:11.424] -> [Profile] Readback (map/poll/copy): 68.70µs
[2025-06-20, 14:52:11.424] -> [Profile] Total launch_gpu_kernel Time: 1.29ms
[2025-06-20, 14:52:11.424] -> [WARM] Large batch test comparison: Expected[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784], Got[..4]=[0.28729436, -0.7910434, -2.0858357, 1.4482784]
[2025-06-20, 14:52:11.424] -> [WARM] kernel_large_batch_test passed.
[2025-06-20, 14:52:05.928] -> Running kernel_all_zero_test...
[2025-06-20, 14:52:05.928] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 14:52:06.311] -> [Profile] Buffer Setup: 2.34ms
[2025-06-20, 14:52:06.311] -> [Profile] Bind Group Setup: 162.90µs
[2025-06-20, 14:52:06.312] -> [Profile] Dispatch & Submit: 605.60µs
[2025-06-20, 14:52:06.312] -> [Profile] Readback (map/poll/copy): 155.60µs
[2025-06-20, 14:52:06.312] -> [Profile] Total launch_gpu_kernel Time: 3.97ms
[2025-06-20, 14:52:06.312] -> All-zero test comparison: All outputs should be zero. Got[..4]=[1.601707, 1.601707, 1.601707, 1.601707]
[2025-06-20, 14:52:06.312] -> kernel_all_zero_test passed.
[2025-06-20, 14:52:11.424] -> [WARM] Running kernel_all_zero_test...
[2025-06-20, 14:52:11.425] -> [Profile] Buffer Setup: 209.70µs
[2025-06-20, 14:52:11.426] -> [Profile] Bind Group Setup: 60.90µs
[2025-06-20, 14:52:11.426] -> [Profile] Dispatch & Submit: 587.90µs
[2025-06-20, 14:52:11.427] -> [Profile] Readback (map/poll/copy): 217.60µs
[2025-06-20, 14:52:11.427] -> [Profile] Total launch_gpu_kernel Time: 1.80ms
[2025-06-20, 14:52:11.427] -> [WARM] kernel_all_zero_test passed.
[2025-06-20, 14:52:06.343] -> Running kernel_all_plus_one_weights_test...
[2025-06-20, 14:52:06.343] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 14:52:06.717] -> [Profile] Buffer Setup: 1.45ms
[2025-06-20, 14:52:06.717] -> [Profile] Bind Group Setup: 225.40µs
[2025-06-20, 14:52:06.718] -> [Profile] Dispatch & Submit: 554.30µs
[2025-06-20, 14:52:06.718] -> [Profile] Readback (map/poll/copy): 93.80µs
[2025-06-20, 14:52:06.718] -> [Profile] Total launch_gpu_kernel Time: 2.96ms
[2025-06-20, 14:52:06.719] -> All-plus-one test comparison: Expected[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514], Got[..4]=[2.4678514, 2.4678514, 2.4678514, 2.4678514]
[2025-06-20, 14:52:06.719] -> kernel_all_plus_one_weights_test passed.
[2025-06-20, 14:52:11.427] -> [WARM] Running kernel_all_plus_one_weights_test...
[2025-06-20, 14:52:11.428] -> [Profile] Buffer Setup: 231.60µs
[2025-06-20, 14:52:11.428] -> [Profile] Bind Group Setup: 89.10µs
[2025-06-20, 14:52:11.429] -> [Profile] Dispatch & Submit: 333.60µs
[2025-06-20, 14:52:11.429] -> [Profile] Readback (map/poll/copy): 85.10µs
[2025-06-20, 14:52:11.429] -> [Profile] Total launch_gpu_kernel Time: 1.30ms
[2025-06-20, 14:52:11.429] -> [WARM] kernel_all_plus_one_weights_test passed.
[2025-06-20, 14:52:06.749] -> Running kernel_all_minus_one_weights_test...
[2025-06-20, 14:52:06.750] -> Test dims: batch=32, in=32, out=16
[2025-06-20, 14:52:07.111] -> [Profile] Buffer Setup: 2.60ms
[2025-06-20, 14:52:07.111] -> [Profile] Bind Group Setup: 229.30µs
[2025-06-20, 14:52:07.112] -> [Profile] Dispatch & Submit: 629.70µs
[2025-06-20, 14:52:07.112] -> [Profile] Readback (map/poll/copy): 324.50µs
[2025-06-20, 14:52:07.113] -> [Profile] Total launch_gpu_kernel Time: 4.49ms
[2025-06-20, 14:52:07.113] -> All-minus-one test comparison: Expected[..4]=[0.0, 0.0, 0.0, 0.0], Got[..4]=[0.0, 0.0, 0.0, 0.0]
[2025-06-20, 14:52:07.113] -> kernel_all_minus_one_weights_test passed.
[2025-06-20, 14:52:11.430] -> [WARM] Running kernel_all_minus_one_weights_test...
[2025-06-20, 14:52:11.430] -> [Profile] Buffer Setup: 225.50µs
[2025-06-20, 14:52:11.431] -> [Profile] Bind Group Setup: 65.40µs
[2025-06-20, 14:52:11.431] -> [Profile] Dispatch & Submit: 315.30µs
[2025-06-20, 14:52:11.431] -> [Profile] Readback (map/poll/copy): 71.40µs
[2025-06-20, 14:52:11.431] -> [Profile] Total launch_gpu_kernel Time: 1.17ms
[2025-06-20, 14:52:11.432] -> [WARM] kernel_all_minus_one_weights_test passed.
[2025-06-20, 14:52:07.145] -> Running kernel_non_divisible_batch_test...
[2025-06-20, 14:52:07.145] -> Test dims: batch=33, in=32, out=16
[2025-06-20, 14:52:07.512] -> [Profile] Buffer Setup: 2.16ms
[2025-06-20, 14:52:07.513] -> [Profile] Bind Group Setup: 252.40µs
[2025-06-20, 14:52:07.514] -> [Profile] Dispatch & Submit: 802.10µs
[2025-06-20, 14:52:07.514] -> [Profile] Readback (map/poll/copy): 123.20µs
[2025-06-20, 14:52:07.514] -> [Profile] Total launch_gpu_kernel Time: 4.23ms
[2025-06-20, 14:52:07.514] -> Non-divisible batch test comparison: Expected[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603], Got[..4]=[2.9334989, 1.486411, 0.71174705, 0.12976603]
[2025-06-20, 14:52:07.514] -> kernel_non_divisible_batch_test passed.
[2025-06-20, 14:52:11.432] -> [WARM] Running kernel_non_divisible_batch_test...
[2025-06-20, 14:52:11.433] -> [Profile] Buffer Setup: 232.90µs
[2025-06-20, 14:52:11.433] -> [Profile] Bind Group Setup: 121.20µs
[2025-06-20, 14:52:11.434] -> [Profile] Dispatch & Submit: 457.40µs
[2025-06-20, 14:52:11.434] -> [Profile] Readback (map/poll/copy): 81.60µs
[2025-06-20, 14:52:11.434] -> [Profile] Total launch_gpu_kernel Time: 1.55ms
[2025-06-20, 14:52:11.434] -> [WARM] kernel_non_divisible_batch_test passed.
[2025-06-20, 14:52:07.545] -> Running test_bitlinear_layer_forward_pass...
[2025-06-20, 14:52:08.230] -> BitLinear forward pass output length: 32768
[2025-06-20, 14:52:08.231] -> test_bitlinear_layer_forward_pass passed.
[2025-06-20, 14:52:11.435] -> [WARM] Running test_bitlinear_layer_forward_pass...
[2025-06-20, 14:52:11.779] -> [WARM] BitLinear forward pass output length: 32768
[2025-06-20, 14:52:11.780] -> [WARM] test_bitlinear_layer_forward_pass passed.
[2025-06-20, 14:52:08.642] -> Timestamp query enabled with period: 1 ns/tick
[2025-06-20, 14:52:08.688] -> Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 352.366µs  | Total: 35.237ms  
  GPU (Kernel Time):  Avg: 21.380µs   | Total: 2.138ms   
  Scalar (CPU Time):  Avg: 107.611µs  | Total: 10.761ms  
Speedup (Wall vs Scalar):   0.31x
Speedup (Kernel vs Scalar): 5.03x
[2025-06-20, 14:52:11.838] -> [WARM] Performance Benchmark: Avg Wall Time: 571.717µs
[2025-06-20, 14:52:09.359] -> Found 5 adapters. Running consistency test.
[2025-06-20, 14:52:09.360] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-20, 14:52:09.435] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 14:52:09.437] -> [Profile] Buffer Setup: 2.35ms
[2025-06-20, 14:52:09.438] -> [Profile] Bind Group Setup: 239.50µs
[2025-06-20, 14:52:09.439] -> [Profile] Dispatch & Submit: 699.40µs
[2025-06-20, 14:52:09.439] -> [Profile] Readback (map/poll/copy): 244.10µs
[2025-06-20, 14:52:09.439] -> [Profile] Total launch_gpu_kernel Time: 4.28ms
[2025-06-20, 14:52:09.440] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 14:52:09.467] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 14:52:09.467] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 14:52:09.491] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-20, 14:52:09.491] -> WARNING: Skipping test on "Microsoft Basic Render Driver" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 14:52:09.493] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 14:52:09.493] -> WARNING: Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to a known WGSL compiler bug on the Dx12 backend. See source code for details.
[2025-06-20, 14:52:09.527] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-20, 14:52:09.542] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-20, 14:52:09.544] -> [Profile] Buffer Setup: 1.21ms
[2025-06-20, 14:52:09.544] -> [Profile] Bind Group Setup: 91.40µs
[2025-06-20, 14:52:09.547] -> [Profile] Dispatch & Submit: 2.75ms
[2025-06-20, 14:52:09.548] -> [Profile] Readback (map/poll/copy): 617.40µs
[2025-06-20, 14:52:09.548] -> [Profile] Total launch_gpu_kernel Time: 5.44ms
[2025-06-20, 14:52:09.548] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-20, 14:52:09.549] -> Kernel correctness passed on all available devices.
[2025-06-20, 14:52:10.195] -> Streaming Load Test (10 streams): Avg Latency: 1.376ms
[2025-06-20, 14:52:11.847] -> [WARM] Streaming Load Test (10 streams) passed.
[2025-06-20, 14:52:09.066] -> Original Activations for FP Edge Case Test: [1.0, -1.0, 0.0, 127.0, -127.0, 1e-6, 1000000.0, -1000000.0, 3.4028235e38, -3.4028235e38, 1.1920929e-7, NaN, inf, -inf, -10.0, 0.1]
[2025-06-20, 14:52:09.066] -> Quantized Activations from FP Edge Case Test: [0, 0, 0, 0, 0, 0, 0, 0, 127, -127, 0, 0, 127, -127, 0, 0]
[2025-06-20, 14:52:09.066] -> Activation Scale from FP Edge Case Test: 2679388700000000000000000000000000000
[2025-06-20, 14:52:09.083] -> [Profile] Buffer Setup: 1.83ms
[2025-06-20, 14:52:09.084] -> [Profile] Bind Group Setup: 142.60µs
[2025-06-20, 14:52:09.085] -> [Profile] Dispatch & Submit: 820.60µs
[2025-06-20, 14:52:09.085] -> [Profile] Readback (map/poll/copy): 157.60µs
[2025-06-20, 14:52:09.085] -> [Profile] Total launch_gpu_kernel Time: 3.80ms
[2025-06-20, 14:52:09.085] -> Precision test with FP edge cases (NaN, Infinity) passed.
[2025-06-20, 14:52:11.839] -> [Profile] Buffer Setup: 261.70µs
[2025-06-20, 14:52:11.839] -> [Profile] Bind Group Setup: 60.20µs
[2025-06-20, 14:52:11.839] -> [Profile] Dispatch & Submit: 302.70µs
[2025-06-20, 14:52:11.840] -> [Profile] Readback (map/poll/copy): 70.00µs
[2025-06-20, 14:52:11.840] -> [Profile] Total launch_gpu_kernel Time: 1.19ms
[2025-06-20, 14:52:11.840] -> [WARM] Precision test with FP edge cases passed.
[2025-06-20, 14:52:10.626] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-20, 14:52:10.626] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-20, 14:52:10.997] -> Memory safety test: Device max_buffer_size = 268435456. Calculated oversized batch size = 4194305.
[2025-06-20, 14:52:11.012] -> Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-20, 14:52:11.847] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-20, 14:52:10.592] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-20, 14:52:10.592] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-20, 14:52:10.593] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-20, 14:52:11.847] -> [WARM] Stress Test: Initializing with large dimensions (1024x1024x1024)...
[2025-06-20, 14:52:11.847] -> [WARM] Stress Test: Generating random data (this may take a moment)...
[2025-06-20, 14:52:12.505] -> [WARM] Stress Test: Starting scalar pre-computation...
[2025-06-20, 14:52:12.600] -> [WARM] Stress Test: Starting GPU execution...
[2025-06-20, 14:52:12.611] -> [Profile] Buffer Setup: 10.26ms
[2025-06-20, 14:52:12.611] -> [Profile] Bind Group Setup: 187.20µs
[2025-06-20, 14:52:12.612] -> [Profile] Dispatch & Submit: 633.40µs
[2025-06-20, 14:52:12.693] -> [Profile] Readback (map/poll/copy): 80.79ms
[2025-06-20, 14:52:12.693] -> [Profile] Total launch_gpu_kernel Time: 92.83ms
[2025-06-20, 14:52:12.694] -> [WARM] Stress Test: GPU execution complete. Time: 93.37ms
[2025-06-20, 14:52:12.694] -> [WARM] Stress Test: Skipping scalar execution for warm run.
[2025-06-20, 14:52:12.694] -> [WARM] Stress Test: Test passed! Total time: 847.17ms
[2025-06-20, 14:52:11.847] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-20, 14:52:04.121] -> STARTING KERNEL TEST SUITE
[2025-06-20, 14:52:04.122] -> --- STARTING COLD RUN (INDIVIDUAL TESTS) ---
[2025-06-20, 14:52:11.043] -> --- STARTING WARM RUN (SHARED CONTEXT) ---
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 28
- **Passed:** 28
- **Failed:** 0

### Timing Information

- **Total Time:** 6.61 sec
- **Average Time:** 236.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
