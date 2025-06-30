# KERNEL_TESTS Test Report

> Generated on: 2025-06-30 19:31:39

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Bitlinear Layer Forward Pass (OPTIMAL)             | ✅ Pass |  755.00 ms |             |
|  2 | Bitlinear Layer Forward Pass (SAFE)                | ✅ Pass |  771.00 ms |             |
|  3 | GPU Kernel Dimensions (OPTIMAL)                    | ✅ Pass |  415.00 ms |             |
|  4 | GPU Kernel Dimensions (SAFE)                       | ✅ Pass |  410.00 ms |             |
|  5 | Kernel All Minus One Weights Test (OPTIMAL)        | ✅ Pass |  406.00 ms |             |
|  6 | Kernel All Minus One Weights Test (SAFE)           | ✅ Pass |  407.00 ms |             |
|  7 | Kernel All Plus One Weights Test (OPTIMAL)         | ✅ Pass |  418.00 ms |             |
|  8 | Kernel All Plus One Weights Test (SAFE)            | ✅ Pass |  421.00 ms |             |
|  9 | Kernel All Zero Test (OPTIMAL)                     | ✅ Pass |  418.00 ms |             |
| 10 | Kernel All Zero Test (SAFE)                        | ✅ Pass |  405.00 ms |             |
| 11 | Kernel Large Batch Test (OPTIMAL)                  | ✅ Pass |  421.00 ms |             |
| 12 | Kernel Large Batch Test (SAFE)                     | ✅ Pass |  456.00 ms |             |
| 13 | Kernel Non Divisible Batch Test (OPTIMAL)          | ✅ Pass |  398.00 ms |             |
| 14 | Kernel Non Divisible Batch Test (SAFE)             | ✅ Pass |  390.00 ms |             |
| 15 | Low Level Kernel Correctness Test (OPTIMAL)        | ✅ Pass |  472.00 ms |             |
| 16 | Low Level Kernel Correctness Test (SAFE)           | ✅ Pass |  420.00 ms |             |
| 17 | Performance Benchmark GPU Vs Scalar (OPTIMAL)      | ✅ Pass |  574.00 ms |             |
| 18 | Performance Benchmark GPU Vs Scalar (SAFE)         | ✅ Pass |  538.00 ms |             |
| 19 | Precision Test Fp Edge Cases (OPTIMAL)             | ✅ Pass |  396.00 ms |             |
| 20 | Precision Test Fp Edge Cases (SAFE)                | ✅ Pass |  427.00 ms |             |
| 21 | Stress Test Maximum Dimension Support (OPTIMAL)    | ✅ Pass |   2.44 sec |             |
| 22 | Stress Test Maximum Dimension Support (SAFE)       | ✅ Pass |   2.42 sec |             |
| 23 | Cross Device Consistency Test                      | ✅ Pass |  24.68 sec |             |
| 24 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    3.00 ms |             |
| 25 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    3.00 ms |             |
| 26 | Kernel All Zero Test Warm                          | ✅ Pass |    2.00 ms |             |
| 27 | Kernel Large Batch Test Warm                       | ✅ Pass |    3.00 ms |             |
| 28 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    3.00 ms |             |
| 29 | Low Level Kernel Correctness Test Warm             | ✅ Pass |   12.00 ms |             |
| 30 | Memory Safety Buffer Overflow Test Warm            | ✅ Pass |    0.00 ms |             |
| 31 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  368.00 ms |             |
| 32 | Memory Safety Hardcoded Large Allocation Test Warm | ✅ Pass |    0.00 ms |             |
| 33 | Performance Benchmark GPU Vs Scalar Large Batch    | ✅ Pass |   4.28 sec |             |
| 34 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |  154.00 ms |             |
| 35 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    2.00 ms |             |
| 36 | Streaming Load Test                                | ✅ Pass |  417.00 ms |             |
| 37 | Streaming Load Test Warm                           | ✅ Pass |   14.00 ms |             |
| 38 | Stress Test Maximum Dimension Support Warm         | ✅ Pass |   2.06 sec |             |
| 39 | Basic GPU Buffer Operations                        | ✅ Pass |  618.00 ms |             |
| 40 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  359.00 ms |             |
| 41 | GPU Kernel Dimensions Warm                         | ✅ Pass |    2.00 ms |             |
| 42 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    0.00 ms |             |
| 43 | Unit Test Pack Ternary Weights                     | ✅ Pass |    0.00 ms |             |

## ⭐ Special Finding

**[Summary]**: `SAFE vs OPTIMAL Shader Comparison`  
This report shows each test run with both the SAFE (bitnet_kernel.wgsl) and OPTIMAL (bitnet_kernel_optimal.wgsl) kernels. Compare pass/fail status and timings to see which kernel is compatible and faster on your setup. If OPTIMAL fails on DX12 or is much faster elsewhere, this will be clear in the table above. Use this to guide further kernel development and DX12 workarounds.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-30, 19:25:41.701] -> Packed value check: Expected=0b00100100100100100100100100100100, Got=0b00100100100100100100100100100100
[2025-06-30, 19:25:41.701] -> unit_test_pack_ternary_weights passed.
[2025-06-30, 19:25:41.706] -> Testing basic GPU operations...
[2025-06-30, 19:25:42.299] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 19:25:42.325] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-30, 19:25:42.325] -> Basic GPU operations test passed!
[2025-06-30, 19:25:43.189] -> Running correctness logic with dims: batch=4, in=16, out=8 (SAFE)
[2025-06-30, 19:25:43.553] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 19:25:43.577] -> [Profile] Buffer Setup: 23.37ms
[2025-06-30, 19:25:43.577] -> [Profile] Bind Group Setup: 220.40µs
[2025-06-30, 19:25:43.578] -> [Profile] Dispatch & Submit: 619.60µs
[2025-06-30, 19:25:43.578] -> [Profile] Readback (map/poll/copy): 214.80µs
[2025-06-30, 19:25:43.578] -> [Profile] Total launch_gpu_kernel Time: 25.05ms
[2025-06-30, 19:25:43.579] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 19:25:43.609] -> low_level_kernel_correctness_test passed.
[2025-06-30, 19:27:26.059] -> Running correctness logic with dims: batch=4, in=16, out=8 (OPTIMAL)
[2025-06-30, 19:27:26.487] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 19:27:26.497] -> [Profile] Buffer Setup: 8.93ms
[2025-06-30, 19:27:26.497] -> [Profile] Bind Group Setup: 175.90µs
[2025-06-30, 19:27:26.498] -> [Profile] Dispatch & Submit: 693.80µs
[2025-06-30, 19:27:26.498] -> [Profile] Readback (map/poll/copy): 104.00µs
[2025-06-30, 19:27:26.498] -> [Profile] Total launch_gpu_kernel Time: 10.69ms
[2025-06-30, 19:27:26.499] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 19:27:26.531] -> low_level_kernel_correctness_test passed.
[2025-06-30, 19:29:09.532] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 19:29:09.542] -> [Profile] Buffer Setup: 9.00ms
[2025-06-30, 19:29:09.542] -> [Profile] Bind Group Setup: 194.30µs
[2025-06-30, 19:29:09.543] -> [Profile] Dispatch & Submit: 653.90µs
[2025-06-30, 19:29:09.543] -> [Profile] Readback (map/poll/copy): 136.60µs
[2025-06-30, 19:29:09.544] -> [Profile] Total launch_gpu_kernel Time: 10.85ms
[2025-06-30, 19:29:09.544] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 19:30:23.515] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-30, 19:30:23.524] -> [Profile] Buffer Setup: 8.97ms
[2025-06-30, 19:30:23.524] -> [Profile] Bind Group Setup: 206.60µs
[2025-06-30, 19:30:23.525] -> [Profile] Dispatch & Submit: 619.20µs
[2025-06-30, 19:30:23.526] -> [Profile] Readback (map/poll/copy): 198.30µs
[2025-06-30, 19:30:23.526] -> [Profile] Total launch_gpu_kernel Time: 10.86ms
[2025-06-30, 19:30:23.527] -> Correctness test comparison: GPU[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529], Scalar[..4]=[0.5789151, -0.6329472, -2.0069058, 1.8216529]
[2025-06-30, 19:25:43.611] -> Running test_gpu_kernel_dimensions... (SAFE)
[2025-06-30, 19:25:43.989] -> [Profile] Buffer Setup: 11.02ms
[2025-06-30, 19:25:43.990] -> [Profile] Bind Group Setup: 239.60µs
[2025-06-30, 19:25:43.991] -> [Profile] Dispatch & Submit: 549.60µs
[2025-06-30, 19:25:43.991] -> [Profile] Readback (map/poll/copy): 89.80µs
[2025-06-30, 19:25:43.991] -> [Profile] Total launch_gpu_kernel Time: 12.56ms
[2025-06-30, 19:25:44.021] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 19:27:26.533] -> Running test_gpu_kernel_dimensions... (OPTIMAL)
[2025-06-30, 19:27:26.918] -> [Profile] Buffer Setup: 10.23ms
[2025-06-30, 19:27:26.918] -> [Profile] Bind Group Setup: 229.10µs
[2025-06-30, 19:27:26.919] -> [Profile] Dispatch & Submit: 591.60µs
[2025-06-30, 19:27:26.919] -> [Profile] Readback (map/poll/copy): 142.70µs
[2025-06-30, 19:27:26.920] -> [Profile] Total launch_gpu_kernel Time: 11.94ms
[2025-06-30, 19:27:26.949] -> test_gpu_kernel_dimensions passed.
[2025-06-30, 19:29:09.546] -> [Profile] Buffer Setup: 973.60µs
[2025-06-30, 19:29:09.546] -> [Profile] Bind Group Setup: 126.60µs
[2025-06-30, 19:29:09.547] -> [Profile] Dispatch & Submit: 339.60µs
[2025-06-30, 19:29:09.547] -> [Profile] Readback (map/poll/copy): 72.50µs
[2025-06-30, 19:29:09.547] -> [Profile] Total launch_gpu_kernel Time: 2.16ms
[2025-06-30, 19:30:23.528] -> [Profile] Buffer Setup: 907.20µs
[2025-06-30, 19:30:23.528] -> [Profile] Bind Group Setup: 99.20µs
[2025-06-30, 19:30:23.529] -> [Profile] Dispatch & Submit: 311.70µs
[2025-06-30, 19:30:23.529] -> [Profile] Readback (map/poll/copy): 167.00µs
[2025-06-30, 19:30:23.529] -> [Profile] Total launch_gpu_kernel Time: 2.00ms
[2025-06-30, 19:25:44.024] -> Running kernel_large_batch_test... (SAFE)
[2025-06-30, 19:25:44.428] -> [Profile] Buffer Setup: 22.88ms
[2025-06-30, 19:25:44.429] -> [Profile] Bind Group Setup: 194.60µs
[2025-06-30, 19:25:44.429] -> [Profile] Dispatch & Submit: 560.80µs
[2025-06-30, 19:25:44.429] -> [Profile] Readback (map/poll/copy): 93.70µs
[2025-06-30, 19:25:44.430] -> [Profile] Total launch_gpu_kernel Time: 24.42ms
[2025-06-30, 19:25:44.480] -> kernel_large_batch_test passed.
[2025-06-30, 19:27:26.953] -> Running kernel_large_batch_test... (OPTIMAL)
[2025-06-30, 19:27:27.346] -> [Profile] Buffer Setup: 11.04ms
[2025-06-30, 19:27:27.346] -> [Profile] Bind Group Setup: 231.10µs
[2025-06-30, 19:27:27.347] -> [Profile] Dispatch & Submit: 591.60µs
[2025-06-30, 19:27:27.347] -> [Profile] Readback (map/poll/copy): 89.70µs
[2025-06-30, 19:27:27.347] -> [Profile] Total launch_gpu_kernel Time: 12.70ms
[2025-06-30, 19:27:27.374] -> kernel_large_batch_test passed.
[2025-06-30, 19:29:09.550] -> [Profile] Buffer Setup: 890.60µs
[2025-06-30, 19:29:09.550] -> [Profile] Bind Group Setup: 104.50µs
[2025-06-30, 19:29:09.550] -> [Profile] Dispatch & Submit: 213.80µs
[2025-06-30, 19:29:09.551] -> [Profile] Readback (map/poll/copy): 123.30µs
[2025-06-30, 19:29:09.551] -> [Profile] Total launch_gpu_kernel Time: 2.00ms
[2025-06-30, 19:30:23.532] -> [Profile] Buffer Setup: 873.90µs
[2025-06-30, 19:30:23.532] -> [Profile] Bind Group Setup: 79.60µs
[2025-06-30, 19:30:23.532] -> [Profile] Dispatch & Submit: 245.00µs
[2025-06-30, 19:30:23.532] -> [Profile] Readback (map/poll/copy): 106.40µs
[2025-06-30, 19:30:23.533] -> [Profile] Total launch_gpu_kernel Time: 1.91ms
[2025-06-30, 19:25:44.483] -> Running kernel_all_zero_test... (SAFE)
[2025-06-30, 19:25:44.854] -> [Profile] Buffer Setup: 8.62ms
[2025-06-30, 19:25:44.854] -> [Profile] Bind Group Setup: 179.50µs
[2025-06-30, 19:25:44.855] -> [Profile] Dispatch & Submit: 639.30µs
[2025-06-30, 19:25:44.855] -> [Profile] Readback (map/poll/copy): 121.60µs
[2025-06-30, 19:25:44.856] -> [Profile] Total launch_gpu_kernel Time: 10.23ms
[2025-06-30, 19:25:44.888] -> kernel_all_zero_test passed.
[2025-06-30, 19:27:27.375] -> Running kernel_all_zero_test... (OPTIMAL)
[2025-06-30, 19:27:27.740] -> [Profile] Buffer Setup: 8.56ms
[2025-06-30, 19:27:27.741] -> [Profile] Bind Group Setup: 244.50µs
[2025-06-30, 19:27:27.742] -> [Profile] Dispatch & Submit: 519.10µs
[2025-06-30, 19:27:27.742] -> [Profile] Readback (map/poll/copy): 90.70µs
[2025-06-30, 19:27:27.742] -> [Profile] Total launch_gpu_kernel Time: 10.15ms
[2025-06-30, 19:27:27.793] -> kernel_all_zero_test passed.
[2025-06-30, 19:29:09.553] -> [Profile] Buffer Setup: 885.70µs
[2025-06-30, 19:29:09.553] -> [Profile] Bind Group Setup: 129.80µs
[2025-06-30, 19:29:09.554] -> [Profile] Dispatch & Submit: 342.20µs
[2025-06-30, 19:29:09.554] -> [Profile] Readback (map/poll/copy): 172.30µs
[2025-06-30, 19:29:09.554] -> [Profile] Total launch_gpu_kernel Time: 2.07ms
[2025-06-30, 19:30:23.535] -> [Profile] Buffer Setup: 791.50µs
[2025-06-30, 19:30:23.535] -> [Profile] Bind Group Setup: 109.10µs
[2025-06-30, 19:30:23.535] -> [Profile] Dispatch & Submit: 255.00µs
[2025-06-30, 19:30:23.536] -> [Profile] Readback (map/poll/copy): 86.30µs
[2025-06-30, 19:30:23.536] -> [Profile] Total launch_gpu_kernel Time: 1.84ms
[2025-06-30, 19:25:44.892] -> Running kernel_all_plus_one_weights_test... (SAFE)
[2025-06-30, 19:25:45.281] -> [Profile] Buffer Setup: 8.97ms
[2025-06-30, 19:25:45.282] -> [Profile] Bind Group Setup: 252.60µs
[2025-06-30, 19:25:45.282] -> [Profile] Dispatch & Submit: 549.70µs
[2025-06-30, 19:25:45.283] -> [Profile] Readback (map/poll/copy): 102.20µs
[2025-06-30, 19:25:45.283] -> [Profile] Total launch_gpu_kernel Time: 10.52ms
[2025-06-30, 19:25:45.313] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 19:27:27.798] -> Running kernel_all_plus_one_weights_test... (OPTIMAL)
[2025-06-30, 19:27:28.188] -> [Profile] Buffer Setup: 8.85ms
[2025-06-30, 19:27:28.189] -> [Profile] Bind Group Setup: 246.40µs
[2025-06-30, 19:27:28.190] -> [Profile] Dispatch & Submit: 612.10µs
[2025-06-30, 19:27:28.190] -> [Profile] Readback (map/poll/copy): 87.00µs
[2025-06-30, 19:27:28.190] -> [Profile] Total launch_gpu_kernel Time: 10.50ms
[2025-06-30, 19:27:28.216] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 19:29:09.556] -> [Profile] Buffer Setup: 781.60µs
[2025-06-30, 19:29:09.557] -> [Profile] Bind Group Setup: 96.10µs
[2025-06-30, 19:29:09.557] -> [Profile] Dispatch & Submit: 217.90µs
[2025-06-30, 19:29:09.557] -> [Profile] Readback (map/poll/copy): 124.70µs
[2025-06-30, 19:29:09.557] -> [Profile] Total launch_gpu_kernel Time: 1.82ms
[2025-06-30, 19:30:23.538] -> [Profile] Buffer Setup: 800.90µs
[2025-06-30, 19:30:23.538] -> [Profile] Bind Group Setup: 77.30µs
[2025-06-30, 19:30:23.539] -> [Profile] Dispatch & Submit: 380.90µs
[2025-06-30, 19:30:23.539] -> [Profile] Readback (map/poll/copy): 90.60µs
[2025-06-30, 19:30:23.539] -> [Profile] Total launch_gpu_kernel Time: 2.08ms
[2025-06-30, 19:25:45.316] -> Running kernel_all_minus_one_weights_test... (SAFE)
[2025-06-30, 19:25:45.692] -> [Profile] Buffer Setup: 10.04ms
[2025-06-30, 19:25:45.692] -> [Profile] Bind Group Setup: 234.60µs
[2025-06-30, 19:25:45.693] -> [Profile] Dispatch & Submit: 485.60µs
[2025-06-30, 19:25:45.693] -> [Profile] Readback (map/poll/copy): 84.10µs
[2025-06-30, 19:25:45.693] -> [Profile] Total launch_gpu_kernel Time: 11.57ms
[2025-06-30, 19:25:45.723] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 19:27:28.218] -> Running kernel_all_minus_one_weights_test... (OPTIMAL)
[2025-06-30, 19:27:28.595] -> [Profile] Buffer Setup: 8.73ms
[2025-06-30, 19:27:28.595] -> [Profile] Bind Group Setup: 247.70µs
[2025-06-30, 19:27:28.596] -> [Profile] Dispatch & Submit: 627.30µs
[2025-06-30, 19:27:28.596] -> [Profile] Readback (map/poll/copy): 89.40µs
[2025-06-30, 19:27:28.596] -> [Profile] Total launch_gpu_kernel Time: 10.36ms
[2025-06-30, 19:27:28.625] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 19:29:09.560] -> [Profile] Buffer Setup: 906.00µs
[2025-06-30, 19:29:09.560] -> [Profile] Bind Group Setup: 121.60µs
[2025-06-30, 19:29:09.561] -> [Profile] Dispatch & Submit: 441.50µs
[2025-06-30, 19:29:09.561] -> [Profile] Readback (map/poll/copy): 174.60µs
[2025-06-30, 19:29:09.561] -> [Profile] Total launch_gpu_kernel Time: 2.52ms
[2025-06-30, 19:30:23.541] -> [Profile] Buffer Setup: 1.01ms
[2025-06-30, 19:30:23.542] -> [Profile] Bind Group Setup: 111.60µs
[2025-06-30, 19:30:23.542] -> [Profile] Dispatch & Submit: 266.40µs
[2025-06-30, 19:30:23.542] -> [Profile] Readback (map/poll/copy): 171.00µs
[2025-06-30, 19:30:23.543] -> [Profile] Total launch_gpu_kernel Time: 2.24ms
[2025-06-30, 19:25:45.727] -> Running kernel_non_divisible_batch_test... (SAFE)
[2025-06-30, 19:25:46.086] -> [Profile] Buffer Setup: 11.11ms
[2025-06-30, 19:25:46.087] -> [Profile] Bind Group Setup: 231.90µs
[2025-06-30, 19:25:46.087] -> [Profile] Dispatch & Submit: 493.80µs
[2025-06-30, 19:25:46.087] -> [Profile] Readback (map/poll/copy): 94.30µs
[2025-06-30, 19:25:46.088] -> [Profile] Total launch_gpu_kernel Time: 12.59ms
[2025-06-30, 19:25:46.116] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 19:27:28.626] -> Running kernel_non_divisible_batch_test... (OPTIMAL)
[2025-06-30, 19:27:28.995] -> [Profile] Buffer Setup: 10.44ms
[2025-06-30, 19:27:28.996] -> [Profile] Bind Group Setup: 234.50µs
[2025-06-30, 19:27:28.997] -> [Profile] Dispatch & Submit: 621.60µs
[2025-06-30, 19:27:28.997] -> [Profile] Readback (map/poll/copy): 87.40µs
[2025-06-30, 19:27:28.997] -> [Profile] Total launch_gpu_kernel Time: 12.16ms
[2025-06-30, 19:27:29.025] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 19:29:09.564] -> [Profile] Buffer Setup: 862.90µs
[2025-06-30, 19:29:09.564] -> [Profile] Bind Group Setup: 105.50µs
[2025-06-30, 19:29:09.564] -> [Profile] Dispatch & Submit: 324.30µs
[2025-06-30, 19:29:09.564] -> [Profile] Readback (map/poll/copy): 80.40µs
[2025-06-30, 19:29:09.565] -> [Profile] Total launch_gpu_kernel Time: 2.00ms
[2025-06-30, 19:30:23.545] -> [Profile] Buffer Setup: 861.90µs
[2025-06-30, 19:30:23.545] -> [Profile] Bind Group Setup: 121.80µs
[2025-06-30, 19:30:23.546] -> [Profile] Dispatch & Submit: 496.80µs
[2025-06-30, 19:30:23.546] -> [Profile] Readback (map/poll/copy): 103.30µs
[2025-06-30, 19:30:23.547] -> [Profile] Total launch_gpu_kernel Time: 2.40ms
[2025-06-30, 19:25:46.119] -> Running test_bitlinear_layer_forward_pass... (SAFE)
[2025-06-30, 19:25:46.891] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 19:27:29.027] -> Running test_bitlinear_layer_forward_pass... (OPTIMAL)
[2025-06-30, 19:27:29.782] -> test_bitlinear_layer_forward_pass passed.
[2025-06-30, 19:25:46.893] -> Running performance_benchmark_gpu_vs_scalar... (SAFE)
[2025-06-30, 19:25:47.432] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-30, 19:27:29.783] -> Running performance_benchmark_gpu_vs_scalar... (OPTIMAL)
[2025-06-30, 19:27:30.357] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-30, 19:25:47.435] -> Running precision_test_fp_edge_cases... (SAFE)
[2025-06-30, 19:25:47.862] -> precision_test_fp_edge_cases passed.
[2025-06-30, 19:25:47.867] -> Starting cross-device consistency test...
[2025-06-30, 19:25:47.867] -> Calculating scalar reference result...
[2025-06-30, 19:25:47.868] -> Scalar reference calculation complete.
[2025-06-30, 19:25:48.114] -> Found 5 adapters. Running per-device subtests.
[2025-06-30, 19:25:48.114] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 19:25:48.209] -> [Profile] Buffer Setup: 9.03ms
[2025-06-30, 19:25:48.210] -> [Profile] Bind Group Setup: 203.50µs
[2025-06-30, 19:25:48.211] -> [Profile] Dispatch & Submit: 571.10µs
[2025-06-30, 19:25:48.211] -> [Profile] Readback (map/poll/copy): 238.40µs
[2025-06-30, 19:25:48.211] -> [Profile] Total launch_gpu_kernel Time: 10.75ms
[2025-06-30, 19:25:48.238] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 19:25:48.238] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:26:00.164] -> [Profile] Buffer Setup: 9.50ms
[2025-06-30, 19:26:00.164] -> [Profile] Bind Group Setup: 233.80µs
[2025-06-30, 19:26:00.167] -> [Profile] Dispatch & Submit: 2.78ms
[2025-06-30, 19:26:00.170] -> [Profile] Readback (map/poll/copy): 2.32ms
[2025-06-30, 19:26:00.170] -> [Profile] Total launch_gpu_kernel Time: 15.48ms
[2025-06-30, 19:26:00.201] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:26:00.202] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:26:12.097] -> [Profile] Buffer Setup: 9.74ms
[2025-06-30, 19:26:12.098] -> [Profile] Bind Group Setup: 212.30µs
[2025-06-30, 19:26:12.101] -> [Profile] Dispatch & Submit: 2.83ms
[2025-06-30, 19:26:12.103] -> [Profile] Readback (map/poll/copy): 2.22ms
[2025-06-30, 19:26:12.103] -> [Profile] Total launch_gpu_kernel Time: 15.71ms
[2025-06-30, 19:26:12.134] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:26:12.134] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-30, 19:26:12.135] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-30, 19:26:12.135] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 19:26:12.158] -> [Profile] Buffer Setup: 1.41ms
[2025-06-30, 19:26:12.158] -> [Profile] Bind Group Setup: 89.60µs
[2025-06-30, 19:26:12.161] -> [Profile] Dispatch & Submit: 2.04ms
[2025-06-30, 19:26:12.165] -> [Profile] Readback (map/poll/copy): 3.73ms
[2025-06-30, 19:26:12.165] -> [Profile] Total launch_gpu_kernel Time: 8.05ms
[2025-06-30, 19:26:12.168] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 19:26:12.168] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-30, 19:27:30.362] -> Running precision_test_fp_edge_cases... (OPTIMAL)
[2025-06-30, 19:27:30.759] -> precision_test_fp_edge_cases passed.
[2025-06-30, 19:27:30.761] -> Starting cross-device consistency test...
[2025-06-30, 19:27:30.762] -> Calculating scalar reference result...
[2025-06-30, 19:27:30.762] -> Scalar reference calculation complete.
[2025-06-30, 19:27:31.016] -> Found 5 adapters. Running per-device subtests.
[2025-06-30, 19:27:31.016] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 19:27:31.125] -> [Profile] Buffer Setup: 10.61ms
[2025-06-30, 19:27:31.126] -> [Profile] Bind Group Setup: 249.80µs
[2025-06-30, 19:27:31.126] -> [Profile] Dispatch & Submit: 686.50µs
[2025-06-30, 19:27:31.127] -> [Profile] Readback (map/poll/copy): 176.70µs
[2025-06-30, 19:27:31.127] -> [Profile] Total launch_gpu_kernel Time: 12.56ms
[2025-06-30, 19:27:31.158] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-30, 19:27:31.159] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:27:43.265] -> [Profile] Buffer Setup: 10.00ms
[2025-06-30, 19:27:43.266] -> [Profile] Bind Group Setup: 146.60µs
[2025-06-30, 19:27:43.269] -> [Profile] Dispatch & Submit: 2.77ms
[2025-06-30, 19:27:43.271] -> [Profile] Readback (map/poll/copy): 2.28ms
[2025-06-30, 19:27:43.271] -> [Profile] Total launch_gpu_kernel Time: 15.88ms
[2025-06-30, 19:27:43.304] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:27:43.304] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:27:55.373] -> [Profile] Buffer Setup: 9.60ms
[2025-06-30, 19:27:55.373] -> [Profile] Bind Group Setup: 204.30µs
[2025-06-30, 19:27:55.376] -> [Profile] Dispatch & Submit: 2.62ms
[2025-06-30, 19:27:55.379] -> [Profile] Readback (map/poll/copy): 2.18ms
[2025-06-30, 19:27:55.379] -> [Profile] Total launch_gpu_kernel Time: 15.29ms
[2025-06-30, 19:27:55.415] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-30, 19:27:55.416] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-30, 19:27:55.416] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-30, 19:27:55.417] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 19:27:55.439] -> [Profile] Buffer Setup: 1.52ms
[2025-06-30, 19:27:55.439] -> [Profile] Bind Group Setup: 86.90µs
[2025-06-30, 19:27:55.441] -> [Profile] Dispatch & Submit: 1.44ms
[2025-06-30, 19:27:55.442] -> [Profile] Readback (map/poll/copy): 1.41ms
[2025-06-30, 19:27:55.443] -> [Profile] Total launch_gpu_kernel Time: 5.10ms
[2025-06-30, 19:27:55.444] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-30, 19:27:55.444] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-30, 19:26:12.243] -> Running streaming_load_test...
[2025-06-30, 19:26:12.660] -> streaming_load_test passed.
[2025-06-30, 19:27:55.526] -> Running streaming_load_test...
[2025-06-30, 19:27:55.943] -> streaming_load_test passed.
[2025-06-30, 19:25:47.832] -> [Profile] Buffer Setup: 9.34ms
[2025-06-30, 19:25:47.833] -> [Profile] Bind Group Setup: 238.20µs
[2025-06-30, 19:25:47.834] -> [Profile] Dispatch & Submit: 699.80µs
[2025-06-30, 19:25:47.834] -> [Profile] Readback (map/poll/copy): 89.50µs
[2025-06-30, 19:25:47.834] -> [Profile] Total launch_gpu_kernel Time: 11.01ms
[2025-06-30, 19:27:30.730] -> [Profile] Buffer Setup: 8.71ms
[2025-06-30, 19:27:30.731] -> [Profile] Bind Group Setup: 239.40µs
[2025-06-30, 19:27:30.732] -> [Profile] Dispatch & Submit: 636.20µs
[2025-06-30, 19:27:30.732] -> [Profile] Readback (map/poll/copy): 91.40µs
[2025-06-30, 19:27:30.732] -> [Profile] Total launch_gpu_kernel Time: 10.43ms
[2025-06-30, 19:29:10.074] -> [Profile] Buffer Setup: 834.30µs
[2025-06-30, 19:29:10.075] -> [Profile] Bind Group Setup: 74.60µs
[2025-06-30, 19:29:10.075] -> [Profile] Dispatch & Submit: 238.80µs
[2025-06-30, 19:29:10.075] -> [Profile] Readback (map/poll/copy): 93.50µs
[2025-06-30, 19:29:10.075] -> [Profile] Total launch_gpu_kernel Time: 1.87ms
[2025-06-30, 19:30:24.063] -> [Profile] Buffer Setup: 857.10µs
[2025-06-30, 19:30:24.063] -> [Profile] Bind Group Setup: 84.90µs
[2025-06-30, 19:30:24.064] -> [Profile] Dispatch & Submit: 238.80µs
[2025-06-30, 19:30:24.064] -> [Profile] Readback (map/poll/copy): 73.10µs
[2025-06-30, 19:30:24.064] -> [Profile] Total launch_gpu_kernel Time: 1.85ms
[2025-06-30, 19:25:42.364] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-30, 19:25:42.364] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-30, 19:26:12.664] -> Running memory_safety_hardcoded_large_allocation_test... (SAFE)
[2025-06-30, 19:26:13.035] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-30, 19:27:55.945] -> Running memory_safety_hardcoded_large_allocation_test... (OPTIMAL)
[2025-06-30, 19:27:56.314] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-30, 19:29:10.091] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 19:30:24.080] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-30, 19:26:13.038] -> Running stress_test_maximum_dimension_support... (SAFE)
[2025-06-30, 19:26:14.237] -> [Profile] Buffer Setup: 17.46ms
[2025-06-30, 19:26:14.237] -> [Profile] Bind Group Setup: 276.20µs
[2025-06-30, 19:26:14.238] -> [Profile] Dispatch & Submit: 680.20µs
[2025-06-30, 19:26:14.330] -> [Profile] Readback (map/poll/copy): 91.57ms
[2025-06-30, 19:26:14.330] -> [Profile] Total launch_gpu_kernel Time: 110.81ms
[2025-06-30, 19:26:15.454] -> stress_test_maximum_dimension_support passed.
[2025-06-30, 19:27:56.319] -> Running stress_test_maximum_dimension_support... (OPTIMAL)
[2025-06-30, 19:27:57.511] -> [Profile] Buffer Setup: 17.75ms
[2025-06-30, 19:27:57.511] -> [Profile] Bind Group Setup: 269.00µs
[2025-06-30, 19:27:57.512] -> [Profile] Dispatch & Submit: 610.80µs
[2025-06-30, 19:27:57.605] -> [Profile] Readback (map/poll/copy): 92.97ms
[2025-06-30, 19:27:57.605] -> [Profile] Total launch_gpu_kernel Time: 112.40ms
[2025-06-30, 19:27:58.760] -> stress_test_maximum_dimension_support passed.
[2025-06-30, 19:29:10.091] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 19:29:10.923] -> [Profile] Buffer Setup: 10.60ms
[2025-06-30, 19:29:10.924] -> [Profile] Bind Group Setup: 221.10µs
[2025-06-30, 19:29:10.924] -> [Profile] Dispatch & Submit: 495.30µs
[2025-06-30, 19:29:11.016] -> [Profile] Readback (map/poll/copy): 91.35ms
[2025-06-30, 19:29:11.016] -> [Profile] Total launch_gpu_kernel Time: 103.65ms
[2025-06-30, 19:30:24.080] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-30, 19:30:24.924] -> [Profile] Buffer Setup: 10.71ms
[2025-06-30, 19:30:24.924] -> [Profile] Bind Group Setup: 217.10µs
[2025-06-30, 19:30:24.925] -> [Profile] Dispatch & Submit: 469.00µs
[2025-06-30, 19:30:25.019] -> [Profile] Readback (map/poll/copy): 93.61ms
[2025-06-30, 19:30:25.019] -> [Profile] Total launch_gpu_kernel Time: 105.84ms
[2025-06-30, 19:25:42.754] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-30, 19:25:42.754] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 19:25:42.755] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-30, 19:25:43.187] -> Testing scalar packing-decoding symmetry...
[2025-06-30, 19:25:43.187] -> Original weights:  [-1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0]
[2025-06-30, 19:25:43.188] -> Decoded weights:   [-1, 0, 1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 0, -1, 1, 0]
[2025-06-30, 19:25:43.188] -> Scalar packing-decoding symmetry test passed.
[2025-06-30, 19:25:41.698] -> STARTING KERNEL TEST SUITE
[2025-06-30, 19:25:43.188] -> --- STARTING COLD RUN (SAFE) ---
[2025-06-30, 19:25:43.188] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-30, 19:27:26.058] -> --- STARTING COLD RUN (OPTIMAL) ---
[2025-06-30, 19:27:26.059] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-30, 19:29:09.134] -> --- STARTING WARM RUN (SAFE) ---
[2025-06-30, 19:29:09.134] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-30, 19:30:23.123] -> --- STARTING WARM RUN (OPTIMAL) ---
[2025-06-30, 19:30:23.123] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-30, 19:29:09.544] -> [WARM] low_level_kernel_correctness_test passed. (SAFE)
[2025-06-30, 19:30:23.527] -> [WARM] low_level_kernel_correctness_test passed. (OPTIMAL)
[2025-06-30, 19:29:09.547] -> [WARM] test_gpu_kernel_dimensions passed. (SAFE)
[2025-06-30, 19:30:23.529] -> [WARM] test_gpu_kernel_dimensions passed. (OPTIMAL)
[2025-06-30, 19:29:09.551] -> [WARM] kernel_large_batch_test passed. (SAFE)
[2025-06-30, 19:30:23.533] -> [WARM] kernel_large_batch_test passed. (OPTIMAL)
[2025-06-30, 19:25:44.856] -> kernel_all_zero_test passed.
[2025-06-30, 19:27:27.742] -> kernel_all_zero_test passed.
[2025-06-30, 19:29:09.555] -> [WARM] kernel_all_zero_test passed. (SAFE)
[2025-06-30, 19:30:23.536] -> [WARM] kernel_all_zero_test passed. (OPTIMAL)
[2025-06-30, 19:25:45.283] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 19:27:28.190] -> kernel_all_plus_one_weights_test passed.
[2025-06-30, 19:29:09.558] -> [WARM] kernel_all_plus_one_weights_test passed. (SAFE)
[2025-06-30, 19:30:23.540] -> [WARM] kernel_all_plus_one_weights_test passed. (OPTIMAL)
[2025-06-30, 19:25:45.694] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 19:27:28.597] -> kernel_all_minus_one_weights_test passed.
[2025-06-30, 19:29:09.562] -> [WARM] kernel_all_minus_one_weights_test passed. (SAFE)
[2025-06-30, 19:30:23.543] -> [WARM] kernel_all_minus_one_weights_test passed. (OPTIMAL)
[2025-06-30, 19:25:46.088] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 19:27:28.998] -> kernel_non_divisible_batch_test passed.
[2025-06-30, 19:29:09.565] -> [WARM] kernel_non_divisible_batch_test passed. (SAFE)
[2025-06-30, 19:30:23.547] -> [WARM] kernel_non_divisible_batch_test passed. (OPTIMAL)
[2025-06-30, 19:29:09.921] -> [WARM] test_bitlinear_layer_forward_pass passed. (SAFE)
[2025-06-30, 19:30:23.907] -> [WARM] test_bitlinear_layer_forward_pass passed. (OPTIMAL)
[2025-06-30, 19:29:10.073] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.398ms    | Total: 139.810ms 
  Scalar (CPU Time):  Avg: 107.814µs  | Total: 10.781ms  
Speedup (Wall vs Scalar):   0.08x
[2025-06-30, 19:29:10.073] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (SAFE)
[2025-06-30, 19:30:24.061] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.424ms    | Total: 142.432ms 
  Scalar (CPU Time):  Avg: 109.410µs  | Total: 10.941ms  
Speedup (Wall vs Scalar):   0.08x
[2025-06-30, 19:30:24.062] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (OPTIMAL)
[2025-06-30, 19:29:10.076] -> [WARM] precision_test_fp_edge_cases passed. (SAFE)
[2025-06-30, 19:30:24.064] -> [WARM] precision_test_fp_edge_cases passed. (OPTIMAL)
[2025-06-30, 19:29:10.091] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.368ms
[2025-06-30, 19:30:24.079] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.383ms
[2025-06-30, 19:29:12.139] -> [WARM] stress_test_maximum_dimension_support passed. (SAFE)
[2025-06-30, 19:30:26.144] -> [WARM] stress_test_maximum_dimension_support passed. (OPTIMAL)
[2025-06-30, 19:26:15.457] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=SAFE
[2025-06-30, 19:27:26.016] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=SAFE):
  GPU (Wall Time):    Avg: 137.855ms  | Total: 4.136s    
  Scalar (CPU Time):  Avg: 2.166s     | Total: 64.994s   
Speedup (Wall vs Scalar):   15.72x
[2025-06-30, 19:27:58.762] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=OPTIMAL
[2025-06-30, 19:29:09.092] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=OPTIMAL):
  GPU (Wall Time):    Avg: 136.454ms  | Total: 4.094s    
  Scalar (CPU Time):  Avg: 2.161s     | Total: 64.826s   
Speedup (Wall vs Scalar):   15.84x
[2025-06-30, 19:29:12.141] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=SAFE
[2025-06-30, 19:30:23.085] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=SAFE):
  GPU (Wall Time):    Avg: 144.884ms  | Total: 4.347s    
  Scalar (CPU Time):  Avg: 2.184s     | Total: 65.521s   
Speedup (Wall vs Scalar):   15.07x
[2025-06-30, 19:30:26.145] -> Running performance_benchmark_gpu_vs_scalar_large_batch: batch_size=2048, in_features=1024, out_features=1024, iterations=30, shader_label=OPTIMAL
[2025-06-30, 19:31:37.492] -> Large Batch Performance Benchmark (30 iterations, 2048 batch, 1024 in, 1024 out, shader_label=OPTIMAL):
  GPU (Wall Time):    Avg: 142.572ms  | Total: 4.277s    
  Scalar (CPU Time):  Avg: 2.199s     | Total: 65.979s   
Speedup (Wall vs Scalar):   15.43x
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 43
- **Passed:** 43
- **Failed:** 0

### Timing Information

- **Total Time:** 47.18 sec
- **Average Time:** 1097.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
