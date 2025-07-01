# KERNEL_TESTS Test Report

> Generated on: 2025-06-30 20:32:01

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Bitlinear Layer Forward Pass (OPTIMAL)             | ✅ Pass |  733.00 ms |             |
|  2 | Bitlinear Layer Forward Pass (SAFE)                | ✅ Pass |  770.00 ms |             |
|  3 | GPU Kernel Dimensions (OPTIMAL)                    | ✅ Pass |  409.00 ms |             |
|  4 | GPU Kernel Dimensions (SAFE)                       | ✅ Pass |  428.00 ms |             |
|  5 | Kernel All Minus One Weights Test (OPTIMAL)        | ✅ Pass |  402.00 ms |             |
|  6 | Kernel All Minus One Weights Test (SAFE)           | ✅ Pass |  432.00 ms |             |
|  7 | Kernel All Plus One Weights Test (OPTIMAL)         | ✅ Pass |  402.00 ms |             |
|  8 | Kernel All Plus One Weights Test (SAFE)            | ✅ Pass |  444.00 ms |             |
|  9 | Kernel All Zero Test (OPTIMAL)                     | ✅ Pass |  422.00 ms |             |
| 10 | Kernel All Zero Test (SAFE)                        | ✅ Pass |  440.00 ms |             |
| 11 | Kernel Large Batch Test (OPTIMAL)                  | ✅ Pass |  426.00 ms |             |
| 12 | Kernel Large Batch Test (SAFE)                     | ✅ Pass |  429.00 ms |             |
| 13 | Kernel Non Divisible Batch Test (OPTIMAL)          | ✅ Pass |  452.00 ms |             |
| 14 | Kernel Non Divisible Batch Test (SAFE)             | ✅ Pass |  383.00 ms |             |
| 15 | Low Level Kernel Correctness Test (OPTIMAL)        | ✅ Pass |  447.00 ms |             |
| 16 | Low Level Kernel Correctness Test (SAFE)           | ✅ Pass |  403.00 ms |             |
| 17 | Performance Benchmark GPU Vs Scalar (OPTIMAL)      | ✅ Pass |  558.00 ms |             |
| 18 | Performance Benchmark GPU Vs Scalar (SAFE)         | ✅ Pass |  557.00 ms |             |
| 19 | Precision Test Fp Edge Cases (OPTIMAL)             | ✅ Pass |  436.00 ms |             |
| 20 | Precision Test Fp Edge Cases (SAFE)                | ✅ Pass |  423.00 ms |             |
| 21 | Stress Test Maximum Dimension Support (OPTIMAL)    | ✅ Pass |   2.39 sec |             |
| 22 | Stress Test Maximum Dimension Support (SAFE)       | ✅ Pass |   2.40 sec |             |
| 23 | Cross Device Consistency Test                      | ✅ Pass |  24.55 sec |             |
| 24 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    3.00 ms |             |
| 25 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    3.00 ms |             |
| 26 | Kernel All Zero Test Warm                          | ✅ Pass |    3.00 ms |             |
| 27 | Kernel Large Batch Test Warm                       | ✅ Pass |    3.00 ms |             |
| 28 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    3.00 ms |             |
| 29 | Low Level Kernel Correctness Test Warm             | ✅ Pass |   12.00 ms |             |
| 30 | Memory Safety Buffer Overflow Test Warm            | ✅ Pass |    0.00 ms |             |
| 31 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  378.00 ms |             |
| 32 | Memory Safety Hardcoded Large Allocation Test Warm | ✅ Pass |    0.00 ms |             |
| 33 | Performance Benchmark GPU Vs Scalar Large Batch    | ✅ Pass |   4.34 sec |             |
| 34 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |  154.00 ms |             |
| 35 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    2.00 ms |             |
| 36 | Streaming Load Test                                | ✅ Pass |  419.00 ms |             |
| 37 | Streaming Load Test Warm                           | ✅ Pass |   16.00 ms |             |
| 38 | Stress Test Maximum Dimension Support Warm         | ✅ Pass |   2.06 sec |             |
| 39 | Basic GPU Buffer Operations                        | ✅ Pass |  614.00 ms |             |
| 40 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  359.00 ms |             |
| 41 | GPU Kernel Dimensions Warm                         | ✅ Pass |    2.00 ms |             |
| 42 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    0.00 ms |             |
| 43 | Unit Test Pack Ternary Weights                     | ✅ Pass |    2.00 ms |             |

## ⭐ Special Finding

**[Summary]**: `SAFE vs OPTIMAL Shader Comparison`  
This report shows each test run with both the SAFE (bitnet_kernel.wgsl) and OPTIMAL (bitnet_kernel_optimal.wgsl) kernels. Compare pass/fail status and timings to see which kernel is compatible and faster on your setup. If OPTIMAL fails on DX12 or is much faster elsewhere, this will be clear in the table above. Use this to guide further kernel development and DX12 workarounds.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-30, 20:26:03.468] -> Packed value check: Expected=0b10010010010010010010010010010010, Got=0b10010010010010010010010010010010
[2025-06-30, 20:26:03.470] -> unit_test_pack_ternary_weights passed.
[2025-06-30, 20:26:03.474] -> Testing basic GPU operations...
[2025-06-30, 20:26:04.065] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 20:26:04.088] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 20:26:04.088] -> Basic GPU operations test passed!
[2025-06-30, 20:26:04.985] -> Running correctness logic with dims: batch=4, in=16, out=8 (SAFE)
[2025-06-30, 20:26:05.337] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:26:05.358] -> [Profile] Buffer Setup: 21.39ms
[2025-06-30, 20:26:05.359] -> [Profile] Bind Group Setup: 231.50µs
[2025-06-30, 20:26:05.360] -> [Profile] Dispatch & Submit: 609.80µs
[2025-06-30, 20:26:05.360] -> [Profile] Readback (map/poll/copy): 218.60µs
[2025-06-30, 20:26:05.360] -> [Profile] Total launch_gpu_kernel Time: 23.11ms
[2025-06-30, 20:26:05.361] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:26:05.388] -> low_level_kernel_correctness_test passed.
[2025-06-30, 20:27:47.309] -> Running correctness logic with dims: batch=4, in=16, out=8 (OPTIMAL)
[2025-06-30, 20:27:47.719] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:27:47.728] -> [Profile] Buffer Setup: 8.83ms
[2025-06-30, 20:27:47.728] -> [Profile] Bind Group Setup: 177.10µs
[2025-06-30, 20:27:47.729] -> [Profile] Dispatch & Submit: 588.00µs
[2025-06-30, 20:27:47.730] -> [Profile] Readback (map/poll/copy): 131.90µs
[2025-06-30, 20:27:47.730] -> [Profile] Total launch_gpu_kernel Time: 10.46ms
[2025-06-30, 20:27:47.730] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:27:47.757] -> low_level_kernel_correctness_test passed.
[2025-06-30, 20:29:30.722] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:29:30.732] -> [Profile] Buffer Setup: 9.25ms
[2025-06-30, 20:29:30.732] -> [Profile] Bind Group Setup: 219.90µs
[2025-06-30, 20:29:30.733] -> [Profile] Dispatch & Submit: 673.90µs
[2025-06-30, 20:29:30.734] -> [Profile] Readback (map/poll/copy): 366.50µs
[2025-06-30, 20:29:30.734] -> [Profile] Total launch_gpu_kernel Time: 11.26ms
[2025-06-30, 20:29:30.735] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:30:44.987] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 20:30:44.996] -> [Profile] Buffer Setup: 9.36ms
[2025-06-30, 20:30:44.997] -> [Profile] Bind Group Setup: 216.20µs
[2025-06-30, 20:30:44.998] -> [Profile] Dispatch & Submit: 628.40µs
[2025-06-30, 20:30:44.998] -> [Profile] Readback (map/poll/copy): 140.30µs
[2025-06-30, 20:30:44.998] -> [Profile] Total launch_gpu_kernel Time: 11.22ms
[2025-06-30, 20:30:44.999] -> Correctness test comparison: GPU[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698], Scalar[..4]=[0.71785474, 2.2384717, 1.7676208, -0.8876698]
[2025-06-30, 20:26:05.391] -> Running test_gpu_kernel_dimensions... (SAFE)
[2025-06-30, 20:26:05.786] -> [Profile] Buffer Setup: 21.10ms
[2025-06-30, 20:26:05.787] -> [Profile] Bind Group Setup: 237.90µs
[2025-06-30, 20:26:05.788] -> [Profile] Dispatch & Submit: 560.30µs
[2025-06-30, 20:26:05.788] -> [Profile] Readback (map/poll/copy): 99.50µs
[2025-06-30, 20:26:05.788] -> [Profile] Total launch_gpu_kernel Time: 22.68ms
[2025-06-30, 20:26:05.819] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 20:27:47.760] -> Running test_gpu_kernel_dimensions... (OPTIMAL)
[2025-06-30, 20:27:48.140] -> [Profile] Buffer Setup: 10.23ms
[2025-06-30, 20:27:48.140] -> [Profile] Bind Group Setup: 237.20µs
[2025-06-30, 20:27:48.141] -> [Profile] Dispatch & Submit: 579.50µs
[2025-06-30, 20:27:48.141] -> [Profile] Readback (map/poll/copy): 85.20µs
[2025-06-30, 20:27:48.141] -> [Profile] Total launch_gpu_kernel Time: 11.81ms
[2025-06-30, 20:27:48.170] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 20:29:30.736] -> [Profile] Buffer Setup: 895.10µs
[2025-06-30, 20:29:30.736] -> [Profile] Bind Group Setup: 103.10µs
[2025-06-30, 20:29:30.737] -> [Profile] Dispatch & Submit: 258.20µs
[2025-06-30, 20:29:30.737] -> [Profile] Readback (map/poll/copy): 175.70µs
[2025-06-30, 20:29:30.737] -> [Profile] Total launch_gpu_kernel Time: 2.08ms
[2025-06-30, 20:30:45.000] -> [Profile] Buffer Setup: 872.50µs
[2025-06-30, 20:30:45.000] -> [Profile] Bind Group Setup: 77.30µs
[2025-06-30, 20:30:45.001] -> [Profile] Dispatch & Submit: 268.70µs
[2025-06-30, 20:30:45.001] -> [Profile] Readback (map/poll/copy): 182.00µs
[2025-06-30, 20:30:45.001] -> [Profile] Total launch_gpu_kernel Time: 1.92ms
[2025-06-30, 20:26:05.822] -> Running kernel_large_batch_test... (SAFE)
[2025-06-30, 20:26:06.221] -> [Profile] Buffer Setup: 8.82ms
[2025-06-30, 20:26:06.222] -> [Profile] Bind Group Setup: 167.20µs
[2025-06-30, 20:26:06.222] -> [Profile] Dispatch & Submit: 544.60µs
[2025-06-30, 20:26:06.222] -> [Profile] Readback (map/poll/copy): 101.50µs
[2025-06-30, 20:26:06.223] -> [Profile] Total launch_gpu_kernel Time: 10.32ms
[2025-06-30, 20:26:06.251] -> kernel_large_batch_test passed.
[2025-06-30, 20:27:48.171] -> Running kernel_large_batch_test... (OPTIMAL)
[2025-06-30, 20:27:48.566] -> [Profile] Buffer Setup: 10.35ms
[2025-06-30, 20:27:48.567] -> [Profile] Bind Group Setup: 167.10µs
[2025-06-30, 20:27:48.568] -> [Profile] Dispatch & Submit: 642.50µs
[2025-06-30, 20:27:48.568] -> [Profile] Readback (map/poll/copy): 91.90µs
[2025-06-30, 20:27:48.568] -> [Profile] Total launch_gpu_kernel Time: 11.95ms
[2025-06-30, 20:27:48.597] -> kernel_large_batch_test passed.
[2025-06-30, 20:29:30.740] -> [Profile] Buffer Setup: 888.40µs
[2025-06-30, 20:29:30.740] -> [Profile] Bind Group Setup: 81.30µs
[2025-06-30, 20:29:30.740] -> [Profile] Dispatch & Submit: 250.40µs
[2025-06-30, 20:29:30.741] -> [Profile] Readback (map/poll/copy): 73.00µs
[2025-06-30, 20:29:30.741] -> [Profile] Total launch_gpu_kernel Time: 1.80ms
[2025-06-30, 20:30:45.004] -> [Profile] Buffer Setup: 913.40µs
[2025-06-30, 20:30:45.004] -> [Profile] Bind Group Setup: 115.30µs
[2025-06-30, 20:30:45.005] -> [Profile] Dispatch & Submit: 248.80µs
[2025-06-30, 20:30:45.005] -> [Profile] Readback (map/poll/copy): 74.70µs
[2025-06-30, 20:30:45.005] -> [Profile] Total launch_gpu_kernel Time: 1.93ms
[2025-06-30, 20:26:06.254] -> Running kernel_all_zero_test... (SAFE)
[2025-06-30, 20:26:06.664] -> [Profile] Buffer Setup: 13.09ms
[2025-06-30, 20:26:06.664] -> [Profile] Bind Group Setup: 165.00µs
[2025-06-30, 20:26:06.665] -> [Profile] Dispatch & Submit: 567.00µs
[2025-06-30, 20:26:06.665] -> [Profile] Readback (map/poll/copy): 97.90µs
[2025-06-30, 20:26:06.665] -> [Profile] Total launch_gpu_kernel Time: 14.60ms
[2025-06-30, 20:26:06.695] -> kernel_all_zero_test passed.
[2025-06-30, 20:27:48.600] -> Running kernel_all_zero_test... (OPTIMAL)
[2025-06-30, 20:27:48.974] -> [Profile] Buffer Setup: 8.91ms
[2025-06-30, 20:27:48.975] -> [Profile] Bind Group Setup: 168.50µs
[2025-06-30, 20:27:48.976] -> [Profile] Dispatch & Submit: 674.80µs
[2025-06-30, 20:27:48.976] -> [Profile] Readback (map/poll/copy): 207.10µs
[2025-06-30, 20:27:48.976] -> [Profile] Total launch_gpu_kernel Time: 10.74ms
[2025-06-30, 20:27:49.022] -> kernel_all_zero_test passed.
[2025-06-30, 20:29:30.743] -> [Profile] Buffer Setup: 849.40µs
[2025-06-30, 20:29:30.743] -> [Profile] Bind Group Setup: 67.40µs
[2025-06-30, 20:29:30.743] -> [Profile] Dispatch & Submit: 318.30µs
[2025-06-30, 20:29:30.744] -> [Profile] Readback (map/poll/copy): 190.70µs
[2025-06-30, 20:29:30.744] -> [Profile] Total launch_gpu_kernel Time: 2.16ms
[2025-06-30, 20:30:45.007] -> [Profile] Buffer Setup: 822.70µs
[2025-06-30, 20:30:45.007] -> [Profile] Bind Group Setup: 81.00µs
[2025-06-30, 20:30:45.008] -> [Profile] Dispatch & Submit: 322.70µs
[2025-06-30, 20:30:45.008] -> [Profile] Readback (map/poll/copy): 72.80µs
[2025-06-30, 20:30:45.008] -> [Profile] Total launch_gpu_kernel Time: 1.93ms
[2025-06-30, 20:26:06.697] -> Running kernel_all_plus_one_weights_test... (SAFE)
[2025-06-30, 20:26:07.111] -> [Profile] Buffer Setup: 11.13ms
[2025-06-30, 20:26:07.111] -> [Profile] Bind Group Setup: 188.80µs
[2025-06-30, 20:26:07.112] -> [Profile] Dispatch & Submit: 548.20µs
[2025-06-30, 20:26:07.112] -> [Profile] Readback (map/poll/copy): 108.20µs
[2025-06-30, 20:26:07.112] -> [Profile] Total launch_gpu_kernel Time: 12.83ms
[2025-06-30, 20:26:07.141] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 20:27:49.024] -> Running kernel_all_plus_one_weights_test... (OPTIMAL)
[2025-06-30, 20:27:49.395] -> [Profile] Buffer Setup: 11.69ms
[2025-06-30, 20:27:49.395] -> [Profile] Bind Group Setup: 180.40µs
[2025-06-30, 20:27:49.396] -> [Profile] Dispatch & Submit: 871.80µs
[2025-06-30, 20:27:49.397] -> [Profile] Readback (map/poll/copy): 94.70µs
[2025-06-30, 20:27:49.397] -> [Profile] Total launch_gpu_kernel Time: 13.60ms
[2025-06-30, 20:27:49.426] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 20:29:30.746] -> [Profile] Buffer Setup: 881.30µs
[2025-06-30, 20:29:30.747] -> [Profile] Bind Group Setup: 99.30µs
[2025-06-30, 20:29:30.747] -> [Profile] Dispatch & Submit: 285.10µs
[2025-06-30, 20:29:30.748] -> [Profile] Readback (map/poll/copy): 228.80µs
[2025-06-30, 20:29:30.748] -> [Profile] Total launch_gpu_kernel Time: 2.13ms
[2025-06-30, 20:30:45.010] -> [Profile] Buffer Setup: 896.10µs
[2025-06-30, 20:30:45.011] -> [Profile] Bind Group Setup: 178.60µs
[2025-06-30, 20:30:45.011] -> [Profile] Dispatch & Submit: 247.40µs
[2025-06-30, 20:30:45.011] -> [Profile] Readback (map/poll/copy): 169.00µs
[2025-06-30, 20:30:45.012] -> [Profile] Total launch_gpu_kernel Time: 2.12ms
[2025-06-30, 20:26:07.144] -> Running kernel_all_minus_one_weights_test... (SAFE)
[2025-06-30, 20:26:07.521] -> [Profile] Buffer Setup: 12.50ms
[2025-06-30, 20:26:07.522] -> [Profile] Bind Group Setup: 164.20µs
[2025-06-30, 20:26:07.522] -> [Profile] Dispatch & Submit: 518.70µs
[2025-06-30, 20:26:07.523] -> [Profile] Readback (map/poll/copy): 95.90µs
[2025-06-30, 20:26:07.523] -> [Profile] Total launch_gpu_kernel Time: 13.97ms
[2025-06-30, 20:26:07.577] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 20:27:49.428] -> Running kernel_all_minus_one_weights_test... (OPTIMAL)
[2025-06-30, 20:27:49.801] -> [Profile] Buffer Setup: 8.89ms
[2025-06-30, 20:27:49.801] -> [Profile] Bind Group Setup: 169.70µs
[2025-06-30, 20:27:49.802] -> [Profile] Dispatch & Submit: 664.40µs
[2025-06-30, 20:27:49.802] -> [Profile] Readback (map/poll/copy): 92.00µs
[2025-06-30, 20:27:49.803] -> [Profile] Total launch_gpu_kernel Time: 10.49ms
[2025-06-30, 20:27:49.830] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 20:29:30.750] -> [Profile] Buffer Setup: 815.50µs
[2025-06-30, 20:29:30.750] -> [Profile] Bind Group Setup: 70.80µs
[2025-06-30, 20:29:30.751] -> [Profile] Dispatch & Submit: 427.50µs
[2025-06-30, 20:29:30.751] -> [Profile] Readback (map/poll/copy): 99.20µs
[2025-06-30, 20:29:30.751] -> [Profile] Total launch_gpu_kernel Time: 2.24ms
[2025-06-30, 20:30:45.014] -> [Profile] Buffer Setup: 829.10µs
[2025-06-30, 20:30:45.014] -> [Profile] Bind Group Setup: 205.10µs
[2025-06-30, 20:30:45.015] -> [Profile] Dispatch & Submit: 377.40µs
[2025-06-30, 20:30:45.015] -> [Profile] Readback (map/poll/copy): 95.20µs
[2025-06-30, 20:30:45.015] -> [Profile] Total launch_gpu_kernel Time: 2.34ms
[2025-06-30, 20:26:07.579] -> Running kernel_non_divisible_batch_test... (SAFE)
[2025-06-30, 20:26:07.932] -> [Profile] Buffer Setup: 8.62ms
[2025-06-30, 20:26:07.933] -> [Profile] Bind Group Setup: 187.70µs
[2025-06-30, 20:26:07.934] -> [Profile] Dispatch & Submit: 539.40µs
[2025-06-30, 20:26:07.934] -> [Profile] Readback (map/poll/copy): 177.60µs
[2025-06-30, 20:26:07.934] -> [Profile] Total launch_gpu_kernel Time: 10.30ms
[2025-06-30, 20:26:07.962] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 20:27:49.832] -> Running kernel_non_divisible_batch_test... (OPTIMAL)
[2025-06-30, 20:27:50.223] -> [Profile] Buffer Setup: 14.30ms
[2025-06-30, 20:27:50.224] -> [Profile] Bind Group Setup: 168.90µs
[2025-06-30, 20:27:50.225] -> [Profile] Dispatch & Submit: 698.20µs
[2025-06-30, 20:27:50.225] -> [Profile] Readback (map/poll/copy): 92.10µs
[2025-06-30, 20:27:50.225] -> [Profile] Total launch_gpu_kernel Time: 15.94ms
[2025-06-30, 20:27:50.284] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 20:29:30.754] -> [Profile] Buffer Setup: 855.30µs
[2025-06-30, 20:29:30.754] -> [Profile] Bind Group Setup: 78.70µs
[2025-06-30, 20:29:30.754] -> [Profile] Dispatch & Submit: 297.40µs
[2025-06-30, 20:29:30.755] -> [Profile] Readback (map/poll/copy): 192.10µs
[2025-06-30, 20:29:30.755] -> [Profile] Total launch_gpu_kernel Time: 1.96ms
[2025-06-30, 20:30:45.018] -> [Profile] Buffer Setup: 852.10µs
[2025-06-30, 20:30:45.018] -> [Profile] Bind Group Setup: 74.20µs
[2025-06-30, 20:30:45.019] -> [Profile] Dispatch & Submit: 387.60µs
[2025-06-30, 20:30:45.019] -> [Profile] Readback (map/poll/copy): 179.10µs
[2025-06-30, 20:30:45.019] -> [Profile] Total launch_gpu_kernel Time: 2.06ms
[2025-06-30, 20:26:07.965] -> Running test_bitlinear_layer_forward_pass... (SAFE)
[2025-06-30, 20:26:08.735] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 20:27:50.286] -> Running test_bitlinear_layer_forward_pass... (OPTIMAL)
[2025-06-30, 20:27:51.019] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 20:26:08.738] -> Running performance_benchmark_gpu_vs_scalar... (SAFE)
[2025-06-30, 20:26:09.296] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-30, 20:27:51.023] -> Running performance_benchmark_gpu_vs_scalar... (OPTIMAL)
[2025-06-30, 20:27:51.581] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-30, 20:26:09.298] -> Running precision_test_fp_edge_cases... (SAFE)
[2025-06-30, 20:26:09.722] -> precision_test_fp_edge_cases passed.
[2025-06-30, 20:26:09.726] -> Starting cross-device consistency test...
[2025-06-30, 20:26:09.726] -> Calculating scalar reference result...
[2025-06-30, 20:26:09.726] -> Scalar reference calculation complete.
[2025-06-30, 20:26:09.962] -> Found 5 adapters. Running per-device subtests.
[2025-06-30, 20:26:09.963] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 20:26:10.068] -> [Profile] Buffer Setup: 25.28ms
[2025-06-30, 20:26:10.069] -> [Profile] Bind Group Setup: 261.00µs
[2025-06-30, 20:26:10.070] -> [Profile] Dispatch & Submit: 579.40µs
[2025-06-30, 20:26:10.070] -> [Profile] Readback (map/poll/copy): 299.40µs
[2025-06-30, 20:26:10.070] -> [Profile] Total launch_gpu_kernel Time: 27.07ms
[2025-06-30, 20:26:10.128] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 20:26:10.128] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:26:21.972] -> [Profile] Buffer Setup: 9.39ms
[2025-06-30, 20:26:21.972] -> [Profile] Bind Group Setup: 227.10µs
[2025-06-30, 20:26:21.975] -> [Profile] Dispatch & Submit: 2.77ms
[2025-06-30, 20:26:21.977] -> [Profile] Readback (map/poll/copy): 2.29ms
[2025-06-30, 20:26:21.978] -> [Profile] Total launch_gpu_kernel Time: 15.38ms
[2025-06-30, 20:26:22.013] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:26:22.013] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:26:33.963] -> [Profile] Buffer Setup: 9.61ms
[2025-06-30, 20:26:33.963] -> [Profile] Bind Group Setup: 210.00µs
[2025-06-30, 20:26:33.966] -> [Profile] Dispatch & Submit: 2.82ms
[2025-06-30, 20:26:33.968] -> [Profile] Readback (map/poll/copy): 1.69ms
[2025-06-30, 20:26:33.968] -> [Profile] Total launch_gpu_kernel Time: 15.07ms
[2025-06-30, 20:26:34.002] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:26:34.002] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-30, 20:26:34.003] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-30, 20:26:34.003] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 20:26:34.027] -> [Profile] Buffer Setup: 1.61ms
[2025-06-30, 20:26:34.027] -> [Profile] Bind Group Setup: 154.20µs
[2025-06-30, 20:26:34.029] -> [Profile] Dispatch & Submit: 1.98ms
[2025-06-30, 20:26:34.034] -> [Profile] Readback (map/poll/copy): 4.35ms
[2025-06-30, 20:26:34.034] -> [Profile] Total launch_gpu_kernel Time: 8.88ms
[2025-06-30, 20:26:34.036] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 20:26:34.037] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-30, 20:27:51.582] -> Running precision_test_fp_edge_cases... (OPTIMAL)
[2025-06-30, 20:27:52.019] -> precision_test_fp_edge_cases passed.
[2025-06-30, 20:27:52.022] -> Starting cross-device consistency test...
[2025-06-30, 20:27:52.022] -> Calculating scalar reference result...
[2025-06-30, 20:27:52.023] -> Scalar reference calculation complete.
[2025-06-30, 20:27:52.245] -> Found 5 adapters. Running per-device subtests.
[2025-06-30, 20:27:52.245] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 20:27:52.331] -> [Profile] Buffer Setup: 8.37ms
[2025-06-30, 20:27:52.331] -> [Profile] Bind Group Setup: 239.00µs
[2025-06-30, 20:27:52.332] -> [Profile] Dispatch & Submit: 607.80µs
[2025-06-30, 20:27:52.332] -> [Profile] Readback (map/poll/copy): 116.20µs
[2025-06-30, 20:27:52.333] -> [Profile] Total launch_gpu_kernel Time: 10.02ms
[2025-06-30, 20:27:52.358] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 20:27:52.358] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:28:04.365] -> [Profile] Buffer Setup: 9.38ms
[2025-06-30, 20:28:04.366] -> [Profile] Bind Group Setup: 219.80µs
[2025-06-30, 20:28:04.369] -> [Profile] Dispatch & Submit: 2.80ms
[2025-06-30, 20:28:04.371] -> [Profile] Readback (map/poll/copy): 2.40ms
[2025-06-30, 20:28:04.371] -> [Profile] Total launch_gpu_kernel Time: 15.55ms
[2025-06-30, 20:28:04.405] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:28:04.406] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:28:16.493] -> [Profile] Buffer Setup: 9.67ms
[2025-06-30, 20:28:16.494] -> [Profile] Bind Group Setup: 210.40µs
[2025-06-30, 20:28:16.497] -> [Profile] Dispatch & Submit: 2.65ms
[2025-06-30, 20:28:16.499] -> [Profile] Readback (map/poll/copy): 2.35ms
[2025-06-30, 20:28:16.499] -> [Profile] Total launch_gpu_kernel Time: 15.53ms
[2025-06-30, 20:28:16.535] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 20:28:16.536] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-30, 20:28:16.536] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-30, 20:28:16.537] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 20:28:16.561] -> [Profile] Buffer Setup: 2.27ms
[2025-06-30, 20:28:16.562] -> [Profile] Bind Group Setup: 78.80µs
[2025-06-30, 20:28:16.564] -> [Profile] Dispatch & Submit: 1.66ms
[2025-06-30, 20:28:16.568] -> [Profile] Readback (map/poll/copy): 4.47ms
[2025-06-30, 20:28:16.568] -> [Profile] Total launch_gpu_kernel Time: 9.44ms
[2025-06-30, 20:28:16.571] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 20:28:16.571] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-30, 20:26:34.110] -> Running streaming_load_test...
[2025-06-30, 20:26:34.525] -> streaming_load_test passed.
[2025-06-30, 20:28:16.647] -> Running streaming_load_test...
[2025-06-30, 20:28:17.067] -> streaming_load_test passed.
[2025-06-30, 20:26:09.691] -> [Profile] Buffer Setup: 10.21ms
[2025-06-30, 20:26:09.691] -> [Profile] Bind Group Setup: 234.10µs
[2025-06-30, 20:26:09.692] -> [Profile] Dispatch & Submit: 603.00µs
[2025-06-30, 20:26:09.692] -> [Profile] Readback (map/poll/copy): 140.20µs
[2025-06-30, 20:26:09.692] -> [Profile] Total launch_gpu_kernel Time: 11.94ms
[2025-06-30, 20:27:51.968] -> [Profile] Buffer Setup: 8.74ms
[2025-06-30, 20:27:51.968] -> [Profile] Bind Group Setup: 233.80µs
[2025-06-30, 20:27:51.969] -> [Profile] Dispatch & Submit: 663.90µs
[2025-06-30, 20:27:51.970] -> [Profile] Readback (map/poll/copy): 91.80µs
[2025-06-30, 20:27:51.970] -> [Profile] Total launch_gpu_kernel Time: 10.39ms
[2025-06-30, 20:29:31.268] -> [Profile] Buffer Setup: 807.80µs
[2025-06-30, 20:29:31.268] -> [Profile] Bind Group Setup: 78.00µs
[2025-06-30, 20:29:31.268] -> [Profile] Dispatch & Submit: 227.30µs
[2025-06-30, 20:29:31.269] -> [Profile] Readback (map/poll/copy): 74.20µs
[2025-06-30, 20:29:31.269] -> [Profile] Total launch_gpu_kernel Time: 1.75ms
[2025-06-30, 20:30:45.535] -> [Profile] Buffer Setup: 728.10µs
[2025-06-30, 20:30:45.535] -> [Profile] Bind Group Setup: 85.80µs
[2025-06-30, 20:30:45.535] -> [Profile] Dispatch & Submit: 226.50µs
[2025-06-30, 20:30:45.536] -> [Profile] Readback (map/poll/copy): 69.90µs
[2025-06-30, 20:30:45.536] -> [Profile] Total launch_gpu_kernel Time: 1.68ms
[2025-06-30, 20:26:04.119] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-30, 20:26:04.119] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-30, 20:26:34.528] -> Running memory_safety_hardcoded_large_allocation_test... (SAFE)
[2025-06-30, 20:26:34.912] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-30, 20:28:17.069] -> Running memory_safety_hardcoded_large_allocation_test... (OPTIMAL)
[2025-06-30, 20:28:17.447] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-30, 20:29:31.285] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 20:30:45.553] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 20:26:34.915] -> Running stress_test_maximum_dimension_support... (SAFE)
[2025-06-30, 20:26:36.094] -> [Profile] Buffer Setup: 17.45ms
[2025-06-30, 20:26:36.094] -> [Profile] Bind Group Setup: 262.20µs
[2025-06-30, 20:26:36.095] -> [Profile] Dispatch & Submit: 655.80µs
[2025-06-30, 20:26:36.187] -> [Profile] Readback (map/poll/copy): 91.46ms
[2025-06-30, 20:26:36.187] -> [Profile] Total launch_gpu_kernel Time: 110.65ms
[2025-06-30, 20:26:37.317] -> stress_test_maximum_dimension_support passed.
[2025-06-30, 20:28:17.449] -> Running stress_test_maximum_dimension_support... (OPTIMAL)
[2025-06-30, 20:28:18.613] -> [Profile] Buffer Setup: 21.38ms
[2025-06-30, 20:28:18.613] -> [Profile] Bind Group Setup: 252.80µs
[2025-06-30, 20:28:18.614] -> [Profile] Dispatch & Submit: 601.30µs
[2025-06-30, 20:28:18.706] -> [Profile] Readback (map/poll/copy): 91.51ms
[2025-06-30, 20:28:18.706] -> [Profile] Total launch_gpu_kernel Time: 114.57ms
[2025-06-30, 20:28:19.839] -> stress_test_maximum_dimension_support passed.
[2025-06-30, 20:29:31.284] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 20:29:32.117] -> [Profile] Buffer Setup: 11.23ms
[2025-06-30, 20:29:32.117] -> [Profile] Bind Group Setup: 212.90µs
[2025-06-30, 20:29:32.118] -> [Profile] Dispatch & Submit: 439.00µs
[2025-06-30, 20:29:32.209] -> [Profile] Readback (map/poll/copy): 90.90ms
[2025-06-30, 20:29:32.209] -> [Profile] Total launch_gpu_kernel Time: 103.66ms
[2025-06-30, 20:30:45.552] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 20:30:46.392] -> [Profile] Buffer Setup: 12.08ms
[2025-06-30, 20:30:46.393] -> [Profile] Bind Group Setup: 212.30µs
[2025-06-30, 20:30:46.394] -> [Profile] Dispatch & Submit: 466.80µs
[2025-06-30, 20:30:46.485] -> [Profile] Readback (map/poll/copy): 91.26ms
[2025-06-30, 20:30:46.485] -> [Profile] Total launch_gpu_kernel Time: 104.88ms
[2025-06-30, 20:26:04.537] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-30, 20:26:04.538] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 20:26:04.538] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 20:26:04.983] -> Testing scalar packing-decoding symmetry...
[2025-06-30, 20:26:04.984] -> Original weights:  [-1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0]
[2025-06-30, 20:26:04.984] -> Decoded weights:   [-1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0]
[2025-06-30, 20:26:04.984] -> Scalar packing-decoding symmetry test passed.
[2025-06-30, 20:26:03.466] -> STARTING KERNEL TEST SUITE
[2025-06-30, 20:26:04.984] -> --- STARTING COLD RUN (SAFE) ---
[2025-06-30, 20:26:04.985] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-30, 20:27:47.309] -> --- STARTING COLD RUN (OPTIMAL) ---
[2025-06-30, 20:27:47.309] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-30, 20:29:30.327] -> --- STARTING WARM RUN (SAFE) ---
[2025-06-30, 20:29:30.327] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-30, 20:30:44.587] -> --- STARTING WARM RUN (OPTIMAL) ---
[2025-06-30, 20:30:44.587] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-30, 20:29:30.735] -> [WARM] low_level_kernel_correctness_test passed. (SAFE)
[2025-06-30, 20:30:44.999] -> [WARM] low_level_kernel_correctness_test passed. (OPTIMAL)
[2025-06-30, 20:29:30.737] -> [WARM] test_gpu_kernel_dimensions passed. (SAFE)
[2025-06-30, 20:30:45.002] -> [WARM] test_gpu_kernel_dimensions passed. (OPTIMAL)
[2025-06-30, 20:29:30.741] -> [WARM] kernel_large_batch_test passed. (SAFE)
[2025-06-30, 20:30:45.005] -> [WARM] kernel_large_batch_test passed. (OPTIMAL)
[2025-06-30, 20:26:06.666] -> kernel_all_zero_test passed.
[2025-06-30, 20:27:48.977] -> kernel_all_zero_test passed.
[2025-06-30, 20:29:30.745] -> [WARM] kernel_all_zero_test passed. (SAFE)
[2025-06-30, 20:30:45.009] -> [WARM] kernel_all_zero_test passed. (OPTIMAL)
[2025-06-30, 20:26:07.113] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 20:27:49.397] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 20:29:30.748] -> [WARM] kernel_all_plus_one_weights_test passed. (SAFE)
[2025-06-30, 20:30:45.012] -> [WARM] kernel_all_plus_one_weights_test passed. (OPTIMAL)
[2025-06-30, 20:26:07.523] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 20:27:49.803] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 20:29:30.752] -> [WARM] kernel_all_minus_one_weights_test passed. (SAFE)
[2025-06-30, 20:30:45.016] -> [WARM] kernel_all_minus_one_weights_test passed. (OPTIMAL)
[2025-06-30, 20:26:07.934] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 20:27:50.226] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 20:29:30.755] -> [WARM] kernel_non_divisible_batch_test passed. (SAFE)
[2025-06-30, 20:30:45.020] -> [WARM] kernel_non_divisible_batch_test passed. (OPTIMAL)
[2025-06-30, 20:29:31.112] -> [WARM] test_bitlinear_layer_forward_pass passed. (SAFE)
[2025-06-30, 20:30:45.379] -> [WARM] test_bitlinear_layer_forward_pass passed. (OPTIMAL)
[2025-06-30, 20:29:31.266] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.419ms    | Total: 141.921ms 
  Scalar (CPU Time):  Avg: 108.698µs  | Total: 10.870ms  
Speedup (Wall vs Scalar):   0.08x
[2025-06-30, 20:29:31.267] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (SAFE)
[2025-06-30, 20:30:45.533] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.415ms    | Total: 141.519ms 
  Scalar (CPU Time):  Avg: 110.214µs  | Total: 11.021ms  
Speedup (Wall vs Scalar):   0.08x
[2025-06-30, 20:30:45.534] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (OPTIMAL)
[2025-06-30, 20:29:31.269] -> [WARM] precision_test_fp_edge_cases passed. (SAFE)
[2025-06-30, 20:30:45.536] -> [WARM] precision_test_fp_edge_cases passed. (OPTIMAL)
[2025-06-30, 20:29:31.284] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.376ms
[2025-06-30, 20:30:45.552] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.495ms
[2025-06-30, 20:29:33.332] -> [WARM] stress_test_maximum_dimension_support passed. (SAFE)
[2025-06-30, 20:30:47.609] -> [WARM] stress_test_maximum_dimension_support passed. (OPTIMAL)
[2025-06-30, 20:26:37.320] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=SAFE
[2025-06-30, 20:27:47.267] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=SAFE):
  GPU (Wall Time):    Avg: 136.428ms  | Total: 4.093s    
  Scalar (CPU Time):  Avg: 2.147s     | Total: 64.423s   
Speedup (Wall vs Scalar):   15.74x
[2025-06-30, 20:28:19.840] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=OPTIMAL
[2025-06-30, 20:29:30.286] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=OPTIMAL):
  GPU (Wall Time):    Avg: 136.337ms  | Total: 4.090s    
  Scalar (CPU Time):  Avg: 2.164s     | Total: 64.923s   
Speedup (Wall vs Scalar):   15.87x
[2025-06-30, 20:29:33.333] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=SAFE
[2025-06-30, 20:30:44.548] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=SAFE):
  GPU (Wall Time):    Avg: 144.104ms  | Total: 4.323s    
  Scalar (CPU Time):  Avg: 2.194s     | Total: 65.819s   
Speedup (Wall vs Scalar):   15.22x
[2025-06-30, 20:30:47.611] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=OPTIMAL
[2025-06-30, 20:31:59.064] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=OPTIMAL):
  GPU (Wall Time):    Avg: 144.646ms  | Total: 4.339s    
  Scalar (CPU Time):  Avg: 2.201s     | Total: 66.037s   
Speedup (Wall vs Scalar):   15.22x
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 43
- **Passed:** 43
- **Failed:** 0

### Timing Information

- **Total Time:** 47.12 sec
- **Average Time:** 1095.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
