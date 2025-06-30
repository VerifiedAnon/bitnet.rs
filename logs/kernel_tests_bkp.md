# KERNEL_TESTS Test Report

> Generated on: 2025-06-29 19:00:29

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Bitlinear Layer Forward Pass (OPTIMAL)             | ✅ Pass |  740.00 ms |             |
|  2 | Bitlinear Layer Forward Pass (SAFE)                | ✅ Pass |  729.00 ms |             |
|  3 | GPU Kernel Dimensions (OPTIMAL)                    | ✅ Pass |  390.00 ms |             |
|  4 | GPU Kernel Dimensions (SAFE)                       | ✅ Pass |  391.00 ms |             |
|  5 | Kernel All Minus One Weights Test (OPTIMAL)        | ✅ Pass |  392.00 ms |             |
|  6 | Kernel All Minus One Weights Test (SAFE)           | ✅ Pass |  410.00 ms |             |
|  7 | Kernel All Plus One Weights Test (OPTIMAL)         | ✅ Pass |  391.00 ms |             |
|  8 | Kernel All Plus One Weights Test (SAFE)            | ✅ Pass |  408.00 ms |             |
|  9 | Kernel All Zero Test (OPTIMAL)                     | ✅ Pass |  361.00 ms |             |
| 10 | Kernel All Zero Test (SAFE)                        | ✅ Pass |  400.00 ms |             |
| 11 | Kernel Large Batch Test (OPTIMAL)                  | ✅ Pass |  400.00 ms |             |
| 12 | Kernel Large Batch Test (SAFE)                     | ✅ Pass |  395.00 ms |             |
| 13 | Kernel Non Divisible Batch Test (OPTIMAL)          | ✅ Pass |  383.00 ms |             |
| 14 | Kernel Non Divisible Batch Test (SAFE)             | ✅ Pass |  426.00 ms |             |
| 15 | Low Level Kernel Correctness Test (OPTIMAL)        | ✅ Pass |  417.00 ms |             |
| 16 | Low Level Kernel Correctness Test (SAFE)           | ✅ Pass |  396.00 ms |             |
| 17 | Performance Benchmark GPU Vs Scalar (OPTIMAL)      | ✅ Pass |  562.00 ms |             |
| 18 | Performance Benchmark GPU Vs Scalar (SAFE)         | ✅ Pass |  536.00 ms |             |
| 19 | Precision Test Fp Edge Cases (OPTIMAL)             | ✅ Pass |  384.00 ms |             |
| 20 | Precision Test Fp Edge Cases (SAFE)                | ✅ Pass |  404.00 ms |             |
| 21 | Stress Test Maximum Dimension Support (OPTIMAL)    | ✅ Pass |   2.29 sec |             |
| 22 | Stress Test Maximum Dimension Support (SAFE)       | ✅ Pass |   2.29 sec |             |
| 23 | Cross Device Consistency Test                      | ✅ Pass |  24.35 sec |             |
| 24 | Kernel All Minus One Weights Test Warm             | ✅ Pass |    2.00 ms |             |
| 25 | Kernel All Plus One Weights Test Warm              | ✅ Pass |    3.00 ms |             |
| 26 | Kernel All Zero Test Warm                          | ✅ Pass |    2.00 ms |             |
| 27 | Kernel Large Batch Test Warm                       | ✅ Pass |    3.00 ms |             |
| 28 | Kernel Non Divisible Batch Test Warm               | ✅ Pass |    2.00 ms |             |
| 29 | Low Level Kernel Correctness Test Warm             | ✅ Pass |   12.00 ms |             |
| 30 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  387.00 ms |             |
| 31 | Memory Safety Buffer Overflow Test Warm            | ✅ Pass |    0.00 ms |             |
| 32 | Memory Safety Hardcoded Large Allocation Test      | ✅ Pass |  374.00 ms |             |
| 33 | Memory Safety Hardcoded Large Allocation Test Warm | ✅ Pass |    0.00 ms |             |
| 34 | Performance Benchmark GPU Vs Scalar Warm           | ✅ Pass |  162.00 ms |             |
| 35 | Precision Test Fp Edge Cases Warm                  | ✅ Pass |    2.00 ms |             |
| 36 | Streaming Load Test                                | ✅ Pass |  413.00 ms |             |
| 37 | Streaming Load Test Warm                           | ✅ Pass |   16.00 ms |             |
| 38 | Stress Test Maximum Dimension Support Warm         | ✅ Pass |   1.89 sec |             |
| 39 | Basic GPU Buffer Operations                        | ✅ Pass |  589.00 ms |             |
| 40 | Bitlinear Layer Forward Pass Warm                  | ✅ Pass |  344.00 ms |             |
| 41 | GPU Kernel Dimensions Warm                         | ✅ Pass |    2.00 ms |             |
| 42 | Matmul Quantized Scalar (SAFE)                     | ✅ Pass |    2.00 ms |             |
| 43 | Scalar Packing Decoding Symmetry                   | ✅ Pass |    0.00 ms |             |
| 44 | Unit Test Calculate Weight Scales (SAFE)           | ✅ Pass |    2.00 ms |             |
| 45 | Unit Test Pack Ternary Weights (SAFE)              | ✅ Pass |    2.00 ms |             |

## ⭐ Special Finding

**[Summary]**: `SAFE vs OPTIMAL Shader Comparison`  
This report shows each test run with both the SAFE (bitnet_kernel.wgsl) and OPTIMAL (bitnet_kernel_optimal.wgsl) kernels. Compare pass/fail status and timings to see which kernel is compatible and faster on your setup. If OPTIMAL fails on DX12 or is much faster elsewhere, this will be clear in the table above. Use this to guide further kernel development and DX12 workarounds.


<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-29, 18:59:15.689] -> Running unit_test_pack_ternary_weights... (SAFE)
[2025-06-29, 18:59:15.691] -> unit_test_pack_ternary_weights passed.
[2025-06-29, 18:59:15.693] -> Running unit_test_calculate_weight_scales... (SAFE)
[2025-06-29, 18:59:15.695] -> unit_test_calculate_weight_scales passed.
[2025-06-29, 18:59:15.697] -> Starting test_matmul_quantized_scalar... (SAFE)
[2025-06-29, 18:59:15.699] -> test_matmul_quantized_scalar passed.
[2025-06-29, 18:59:15.702] -> Testing basic GPU operations...
[2025-06-29, 18:59:16.280] -> Test data: [1.0, 2.0, 3.0, 4.0]
[2025-06-29, 18:59:16.290] -> Read-back data: [1.0, 2.0, 3.0, 4.0]
[2025-06-29, 18:59:16.291] -> Basic GPU operations test passed!
[2025-06-29, 18:59:17.162] -> Running correctness logic with dims: batch=4, in=16, out=8 (SAFE)
[2025-06-29, 18:59:17.547] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-29, 18:59:17.556] -> [Profile] Buffer Setup: 8.86ms
[2025-06-29, 18:59:17.557] -> [Profile] Bind Group Setup: 244.10µs
[2025-06-29, 18:59:17.557] -> [Profile] Dispatch & Submit: 630.00µs
[2025-06-29, 18:59:17.558] -> [Profile] Readback (map/poll/copy): 227.40µs
[2025-06-29, 18:59:17.558] -> [Profile] Total launch_gpu_kernel Time: 10.66ms
[2025-06-29, 18:59:17.558] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 18:59:17.559] -> low_level_kernel_correctness_test passed.
[2025-06-29, 18:59:49.718] -> Running correctness logic with dims: batch=4, in=16, out=8 (OPTIMAL)
[2025-06-29, 18:59:50.122] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-29, 18:59:50.133] -> [Profile] Buffer Setup: 11.01ms
[2025-06-29, 18:59:50.133] -> [Profile] Bind Group Setup: 191.90µs
[2025-06-29, 18:59:50.134] -> [Profile] Dispatch & Submit: 537.90µs
[2025-06-29, 18:59:50.134] -> [Profile] Readback (map/poll/copy): 206.10µs
[2025-06-29, 18:59:50.135] -> [Profile] Total launch_gpu_kernel Time: 12.59ms
[2025-06-29, 18:59:50.135] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 18:59:50.135] -> low_level_kernel_correctness_test passed.
[2025-06-29, 19:00:22.387] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-29, 19:00:22.398] -> [Profile] Buffer Setup: 10.86ms
[2025-06-29, 19:00:22.399] -> [Profile] Bind Group Setup: 189.40µs
[2025-06-29, 19:00:22.399] -> [Profile] Dispatch & Submit: 516.30µs
[2025-06-29, 19:00:22.400] -> [Profile] Readback (map/poll/copy): 133.80µs
[2025-06-29, 19:00:22.400] -> [Profile] Total launch_gpu_kernel Time: 12.40ms
[2025-06-29, 19:00:22.400] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 19:00:25.236] -> Running correctness logic with dims: batch=4, in=16, out=8
[2025-06-29, 19:00:25.246] -> [Profile] Buffer Setup: 10.07ms
[2025-06-29, 19:00:25.247] -> [Profile] Bind Group Setup: 207.70µs
[2025-06-29, 19:00:25.247] -> [Profile] Dispatch & Submit: 539.80µs
[2025-06-29, 19:00:25.248] -> [Profile] Readback (map/poll/copy): 115.30µs
[2025-06-29, 19:00:25.248] -> [Profile] Total launch_gpu_kernel Time: 11.56ms
[2025-06-29, 19:00:25.248] -> Correctness test comparison: GPU[..4]=[2.855981, 1.343083, -0.007718868, -2.215315], Scalar[..4]=[2.855981, 1.343083, -0.007718868, -2.215315]
[2025-06-29, 18:59:17.590] -> Running test_gpu_kernel_dimensions... (SAFE)
[2025-06-29, 18:59:17.979] -> [Profile] Buffer Setup: 10.31ms
[2025-06-29, 18:59:17.980] -> [Profile] Bind Group Setup: 246.50µs
[2025-06-29, 18:59:17.980] -> [Profile] Dispatch & Submit: 646.10µs
[2025-06-29, 18:59:17.981] -> [Profile] Readback (map/poll/copy): 192.90µs
[2025-06-29, 18:59:17.981] -> [Profile] Total launch_gpu_kernel Time: 12.19ms
[2025-06-29, 18:59:17.981] -> test_gpu_kernel_dimensions passed.
[2025-06-29, 18:59:50.166] -> Running test_gpu_kernel_dimensions... (OPTIMAL)
[2025-06-29, 18:59:50.555] -> [Profile] Buffer Setup: 10.03ms
[2025-06-29, 18:59:50.555] -> [Profile] Bind Group Setup: 236.80µs
[2025-06-29, 18:59:50.556] -> [Profile] Dispatch & Submit: 454.90µs
[2025-06-29, 18:59:50.556] -> [Profile] Readback (map/poll/copy): 209.20µs
[2025-06-29, 18:59:50.556] -> [Profile] Total launch_gpu_kernel Time: 11.58ms
[2025-06-29, 18:59:50.557] -> test_gpu_kernel_dimensions passed.
[2025-06-29, 19:00:22.402] -> [Profile] Buffer Setup: 801.30µs
[2025-06-29, 19:00:22.402] -> [Profile] Bind Group Setup: 109.50µs
[2025-06-29, 19:00:22.402] -> [Profile] Dispatch & Submit: 211.80µs
[2025-06-29, 19:00:22.402] -> [Profile] Readback (map/poll/copy): 77.80µs
[2025-06-29, 19:00:22.403] -> [Profile] Total launch_gpu_kernel Time: 1.69ms
[2025-06-29, 19:00:25.250] -> [Profile] Buffer Setup: 840.00µs
[2025-06-29, 19:00:25.250] -> [Profile] Bind Group Setup: 70.10µs
[2025-06-29, 19:00:25.250] -> [Profile] Dispatch & Submit: 351.30µs
[2025-06-29, 19:00:25.251] -> [Profile] Readback (map/poll/copy): 77.00µs
[2025-06-29, 19:00:25.251] -> [Profile] Total launch_gpu_kernel Time: 1.96ms
[2025-06-29, 18:59:18.012] -> Running kernel_large_batch_test... (SAFE)
[2025-06-29, 18:59:18.405] -> [Profile] Buffer Setup: 10.19ms
[2025-06-29, 18:59:18.405] -> [Profile] Bind Group Setup: 247.00µs
[2025-06-29, 18:59:18.406] -> [Profile] Dispatch & Submit: 502.30µs
[2025-06-29, 18:59:18.406] -> [Profile] Readback (map/poll/copy): 195.30µs
[2025-06-29, 18:59:18.406] -> [Profile] Total launch_gpu_kernel Time: 11.81ms
[2025-06-29, 18:59:18.407] -> kernel_large_batch_test passed.
[2025-06-29, 18:59:50.588] -> Running kernel_large_batch_test... (OPTIMAL)
[2025-06-29, 18:59:50.986] -> [Profile] Buffer Setup: 10.71ms
[2025-06-29, 18:59:50.986] -> [Profile] Bind Group Setup: 243.10µs
[2025-06-29, 18:59:50.987] -> [Profile] Dispatch & Submit: 536.00µs
[2025-06-29, 18:59:50.988] -> [Profile] Readback (map/poll/copy): 233.20µs
[2025-06-29, 18:59:50.988] -> [Profile] Total launch_gpu_kernel Time: 12.49ms
[2025-06-29, 18:59:50.988] -> kernel_large_batch_test passed.
[2025-06-29, 19:00:22.405] -> [Profile] Buffer Setup: 835.90µs
[2025-06-29, 19:00:22.405] -> [Profile] Bind Group Setup: 73.10µs
[2025-06-29, 19:00:22.406] -> [Profile] Dispatch & Submit: 216.40µs
[2025-06-29, 19:00:22.406] -> [Profile] Readback (map/poll/copy): 86.00µs
[2025-06-29, 19:00:22.406] -> [Profile] Total launch_gpu_kernel Time: 1.77ms
[2025-06-29, 19:00:25.253] -> [Profile] Buffer Setup: 803.60µs
[2025-06-29, 19:00:25.253] -> [Profile] Bind Group Setup: 79.20µs
[2025-06-29, 19:00:25.254] -> [Profile] Dispatch & Submit: 325.50µs
[2025-06-29, 19:00:25.254] -> [Profile] Readback (map/poll/copy): 71.40µs
[2025-06-29, 19:00:25.254] -> [Profile] Total launch_gpu_kernel Time: 1.81ms
[2025-06-29, 18:59:18.439] -> Running kernel_all_zero_test... (SAFE)
[2025-06-29, 18:59:18.837] -> [Profile] Buffer Setup: 8.66ms
[2025-06-29, 18:59:18.838] -> [Profile] Bind Group Setup: 250.60µs
[2025-06-29, 18:59:18.839] -> [Profile] Dispatch & Submit: 555.30µs
[2025-06-29, 18:59:18.839] -> [Profile] Readback (map/poll/copy): 206.70µs
[2025-06-29, 18:59:18.839] -> [Profile] Total launch_gpu_kernel Time: 10.31ms
[2025-06-29, 18:59:18.840] -> kernel_all_zero_test passed.
[2025-06-29, 18:59:51.017] -> Running kernel_all_zero_test... (OPTIMAL)
[2025-06-29, 18:59:51.376] -> [Profile] Buffer Setup: 10.42ms
[2025-06-29, 18:59:51.377] -> [Profile] Bind Group Setup: 203.70µs
[2025-06-29, 18:59:51.378] -> [Profile] Dispatch & Submit: 517.10µs
[2025-06-29, 18:59:51.378] -> [Profile] Readback (map/poll/copy): 184.40µs
[2025-06-29, 18:59:51.378] -> [Profile] Total launch_gpu_kernel Time: 12.04ms
[2025-06-29, 18:59:51.379] -> kernel_all_zero_test passed.
[2025-06-29, 19:00:22.408] -> [Profile] Buffer Setup: 833.40µs
[2025-06-29, 19:00:22.408] -> [Profile] Bind Group Setup: 68.70µs
[2025-06-29, 19:00:22.409] -> [Profile] Dispatch & Submit: 362.70µs
[2025-06-29, 19:00:22.409] -> [Profile] Readback (map/poll/copy): 149.70µs
[2025-06-29, 19:00:22.409] -> [Profile] Total launch_gpu_kernel Time: 2.17ms
[2025-06-29, 19:00:25.256] -> [Profile] Buffer Setup: 768.80µs
[2025-06-29, 19:00:25.256] -> [Profile] Bind Group Setup: 99.30µs
[2025-06-29, 19:00:25.257] -> [Profile] Dispatch & Submit: 202.80µs
[2025-06-29, 19:00:25.257] -> [Profile] Readback (map/poll/copy): 63.70µs
[2025-06-29, 19:00:25.257] -> [Profile] Total launch_gpu_kernel Time: 1.65ms
[2025-06-29, 18:59:18.887] -> Running kernel_all_plus_one_weights_test... (SAFE)
[2025-06-29, 18:59:19.292] -> [Profile] Buffer Setup: 11.19ms
[2025-06-29, 18:59:19.293] -> [Profile] Bind Group Setup: 161.00µs
[2025-06-29, 18:59:19.294] -> [Profile] Dispatch & Submit: 689.40µs
[2025-06-29, 18:59:19.294] -> [Profile] Readback (map/poll/copy): 189.90µs
[2025-06-29, 18:59:19.294] -> [Profile] Total launch_gpu_kernel Time: 12.96ms
[2025-06-29, 18:59:19.295] -> kernel_all_plus_one_weights_test passed.
[2025-06-29, 18:59:51.409] -> Running kernel_all_plus_one_weights_test... (OPTIMAL)
[2025-06-29, 18:59:51.798] -> [Profile] Buffer Setup: 12.86ms
[2025-06-29, 18:59:51.798] -> [Profile] Bind Group Setup: 268.10µs
[2025-06-29, 18:59:51.799] -> [Profile] Dispatch & Submit: 639.10µs
[2025-06-29, 18:59:51.800] -> [Profile] Readback (map/poll/copy): 285.00µs
[2025-06-29, 18:59:51.800] -> [Profile] Total launch_gpu_kernel Time: 14.85ms
[2025-06-29, 18:59:51.800] -> kernel_all_plus_one_weights_test passed.
[2025-06-29, 19:00:22.411] -> [Profile] Buffer Setup: 849.30µs
[2025-06-29, 19:00:22.412] -> [Profile] Bind Group Setup: 192.20µs
[2025-06-29, 19:00:22.412] -> [Profile] Dispatch & Submit: 308.40µs
[2025-06-29, 19:00:22.413] -> [Profile] Readback (map/poll/copy): 191.30µs
[2025-06-29, 19:00:22.413] -> [Profile] Total launch_gpu_kernel Time: 2.20ms
[2025-06-29, 19:00:25.259] -> [Profile] Buffer Setup: 790.00µs
[2025-06-29, 19:00:25.259] -> [Profile] Bind Group Setup: 197.10µs
[2025-06-29, 19:00:25.260] -> [Profile] Dispatch & Submit: 341.00µs
[2025-06-29, 19:00:25.260] -> [Profile] Readback (map/poll/copy): 112.90µs
[2025-06-29, 19:00:25.260] -> [Profile] Total launch_gpu_kernel Time: 2.18ms
[2025-06-29, 18:59:19.326] -> Running kernel_all_minus_one_weights_test... (SAFE)
[2025-06-29, 18:59:19.735] -> [Profile] Buffer Setup: 8.94ms
[2025-06-29, 18:59:19.735] -> [Profile] Bind Group Setup: 234.80µs
[2025-06-29, 18:59:19.736] -> [Profile] Dispatch & Submit: 578.20µs
[2025-06-29, 18:59:19.736] -> [Profile] Readback (map/poll/copy): 208.60µs
[2025-06-29, 18:59:19.736] -> [Profile] Total launch_gpu_kernel Time: 10.63ms
[2025-06-29, 18:59:19.737] -> kernel_all_minus_one_weights_test passed.
[2025-06-29, 18:59:51.828] -> Running kernel_all_minus_one_weights_test... (OPTIMAL)
[2025-06-29, 18:59:52.218] -> [Profile] Buffer Setup: 9.55ms
[2025-06-29, 18:59:52.219] -> [Profile] Bind Group Setup: 245.00µs
[2025-06-29, 18:59:52.219] -> [Profile] Dispatch & Submit: 598.10µs
[2025-06-29, 18:59:52.220] -> [Profile] Readback (map/poll/copy): 127.40µs
[2025-06-29, 18:59:52.220] -> [Profile] Total launch_gpu_kernel Time: 11.23ms
[2025-06-29, 18:59:52.220] -> kernel_all_minus_one_weights_test passed.
[2025-06-29, 19:00:22.415] -> [Profile] Buffer Setup: 849.80µs
[2025-06-29, 19:00:22.415] -> [Profile] Bind Group Setup: 89.70µs
[2025-06-29, 19:00:22.415] -> [Profile] Dispatch & Submit: 212.30µs
[2025-06-29, 19:00:22.416] -> [Profile] Readback (map/poll/copy): 76.10µs
[2025-06-29, 19:00:22.416] -> [Profile] Total launch_gpu_kernel Time: 1.88ms
[2025-06-29, 19:00:25.262] -> [Profile] Buffer Setup: 785.40µs
[2025-06-29, 19:00:25.263] -> [Profile] Bind Group Setup: 76.40µs
[2025-06-29, 19:00:25.263] -> [Profile] Dispatch & Submit: 278.90µs
[2025-06-29, 19:00:25.263] -> [Profile] Readback (map/poll/copy): 92.20µs
[2025-06-29, 19:00:25.263] -> [Profile] Total launch_gpu_kernel Time: 1.79ms
[2025-06-29, 18:59:19.767] -> Running kernel_non_divisible_batch_test... (SAFE)
[2025-06-29, 18:59:20.190] -> [Profile] Buffer Setup: 12.03ms
[2025-06-29, 18:59:20.191] -> [Profile] Bind Group Setup: 292.50µs
[2025-06-29, 18:59:20.192] -> [Profile] Dispatch & Submit: 601.40µs
[2025-06-29, 18:59:20.192] -> [Profile] Readback (map/poll/copy): 173.70µs
[2025-06-29, 18:59:20.192] -> [Profile] Total launch_gpu_kernel Time: 14.02ms
[2025-06-29, 18:59:20.193] -> kernel_non_divisible_batch_test passed.
[2025-06-29, 18:59:52.248] -> Running kernel_non_divisible_batch_test... (OPTIMAL)
[2025-06-29, 18:59:52.629] -> [Profile] Buffer Setup: 8.66ms
[2025-06-29, 18:59:52.630] -> [Profile] Bind Group Setup: 241.80µs
[2025-06-29, 18:59:52.630] -> [Profile] Dispatch & Submit: 588.30µs
[2025-06-29, 18:59:52.631] -> [Profile] Readback (map/poll/copy): 125.00µs
[2025-06-29, 18:59:52.631] -> [Profile] Total launch_gpu_kernel Time: 10.34ms
[2025-06-29, 18:59:52.631] -> kernel_non_divisible_batch_test passed.
[2025-06-29, 19:00:22.418] -> [Profile] Buffer Setup: 891.20µs
[2025-06-29, 19:00:22.418] -> [Profile] Bind Group Setup: 70.30µs
[2025-06-29, 19:00:22.419] -> [Profile] Dispatch & Submit: 206.10µs
[2025-06-29, 19:00:22.419] -> [Profile] Readback (map/poll/copy): 66.30µs
[2025-06-29, 19:00:22.419] -> [Profile] Total launch_gpu_kernel Time: 1.94ms
[2025-06-29, 19:00:25.265] -> [Profile] Buffer Setup: 777.20µs
[2025-06-29, 19:00:25.266] -> [Profile] Bind Group Setup: 68.70µs
[2025-06-29, 19:00:25.266] -> [Profile] Dispatch & Submit: 224.20µs
[2025-06-29, 19:00:25.266] -> [Profile] Readback (map/poll/copy): 77.10µs
[2025-06-29, 19:00:25.266] -> [Profile] Total launch_gpu_kernel Time: 1.67ms
[2025-06-29, 18:59:20.230] -> Running test_bitlinear_layer_forward_pass... (SAFE)
[2025-06-29, 18:59:20.959] -> test_bitlinear_layer_forward_pass passed.
[2025-06-29, 18:59:52.660] -> Running test_bitlinear_layer_forward_pass... (OPTIMAL)
[2025-06-29, 18:59:53.400] -> test_bitlinear_layer_forward_pass passed.
[2025-06-29, 18:59:20.989] -> Running performance_benchmark_gpu_vs_scalar... (SAFE)
[2025-06-29, 18:59:21.525] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-29, 18:59:53.430] -> Running performance_benchmark_gpu_vs_scalar... (OPTIMAL)
[2025-06-29, 18:59:53.992] -> performance_benchmark_gpu_vs_scalar passed.
[2025-06-29, 18:59:21.558] -> Running precision_test_fp_edge_cases... (SAFE)
[2025-06-29, 18:59:21.962] -> precision_test_fp_edge_cases passed.
[2025-06-29, 18:59:22.016] -> Starting cross-device consistency test...
[2025-06-29, 18:59:22.016] -> Calculating scalar reference result...
[2025-06-29, 18:59:22.017] -> Scalar reference calculation complete.
[2025-06-29, 18:59:22.255] -> Found 5 adapters. Running per-device subtests.
[2025-06-29, 18:59:22.255] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-29, 18:59:22.344] -> [Profile] Buffer Setup: 8.21ms
[2025-06-29, 18:59:22.344] -> [Profile] Bind Group Setup: 212.60µs
[2025-06-29, 18:59:22.345] -> [Profile] Dispatch & Submit: 635.90µs
[2025-06-29, 18:59:22.345] -> [Profile] Readback (map/poll/copy): 134.50µs
[2025-06-29, 18:59:22.345] -> [Profile] Total launch_gpu_kernel Time: 10.02ms
[2025-06-29, 18:59:22.370] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-29, 18:59:22.371] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 18:59:34.352] -> [Profile] Buffer Setup: 11.40ms
[2025-06-29, 18:59:34.352] -> [Profile] Bind Group Setup: 240.00µs
[2025-06-29, 18:59:34.356] -> [Profile] Dispatch & Submit: 3.66ms
[2025-06-29, 18:59:34.357] -> [Profile] Readback (map/poll/copy): 949.50µs
[2025-06-29, 18:59:34.358] -> [Profile] Total launch_gpu_kernel Time: 17.20ms
[2025-06-29, 18:59:34.396] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 18:59:34.397] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 18:59:46.388] -> [Profile] Buffer Setup: 9.33ms
[2025-06-29, 18:59:46.389] -> [Profile] Bind Group Setup: 212.60µs
[2025-06-29, 18:59:46.391] -> [Profile] Dispatch & Submit: 2.70ms
[2025-06-29, 18:59:46.394] -> [Profile] Readback (map/poll/copy): 1.96ms
[2025-06-29, 18:59:46.394] -> [Profile] Total launch_gpu_kernel Time: 14.90ms
[2025-06-29, 18:59:46.424] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 18:59:46.424] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-29, 18:59:46.424] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-29, 18:59:46.425] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-29, 18:59:46.448] -> [Profile] Buffer Setup: 1.60ms
[2025-06-29, 18:59:46.448] -> [Profile] Bind Group Setup: 87.10µs
[2025-06-29, 18:59:46.451] -> [Profile] Dispatch & Submit: 2.34ms
[2025-06-29, 18:59:46.454] -> [Profile] Readback (map/poll/copy): 3.36ms
[2025-06-29, 18:59:46.454] -> [Profile] Total launch_gpu_kernel Time: 8.07ms
[2025-06-29, 18:59:46.456] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-29, 18:59:46.456] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-29, 18:59:54.020] -> Running precision_test_fp_edge_cases... (OPTIMAL)
[2025-06-29, 18:59:54.404] -> precision_test_fp_edge_cases passed.
[2025-06-29, 18:59:54.432] -> Starting cross-device consistency test...
[2025-06-29, 18:59:54.432] -> Calculating scalar reference result...
[2025-06-29, 18:59:54.433] -> Scalar reference calculation complete.
[2025-06-29, 18:59:54.661] -> Found 5 adapters. Running per-device subtests.
[2025-06-29, 18:59:54.661] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-29, 18:59:54.756] -> [Profile] Buffer Setup: 15.20ms
[2025-06-29, 18:59:54.756] -> [Profile] Bind Group Setup: 240.40µs
[2025-06-29, 18:59:54.757] -> [Profile] Dispatch & Submit: 578.90µs
[2025-06-29, 18:59:54.757] -> [Profile] Readback (map/poll/copy): 208.80µs
[2025-06-29, 18:59:54.757] -> [Profile] Total launch_gpu_kernel Time: 16.95ms
[2025-06-29, 18:59:54.812] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Vulkan")
[2025-06-29, 18:59:54.813] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 19:00:06.741] -> [Profile] Buffer Setup: 9.16ms
[2025-06-29, 19:00:06.741] -> [Profile] Bind Group Setup: 214.40µs
[2025-06-29, 19:00:06.746] -> [Profile] Dispatch & Submit: 4.47ms
[2025-06-29, 19:00:06.750] -> [Profile] Readback (map/poll/copy): 3.42ms
[2025-06-29, 19:00:06.750] -> [Profile] Total launch_gpu_kernel Time: 18.00ms
[2025-06-29, 19:00:06.779] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 19:00:06.779] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 19:00:18.714] -> [Profile] Buffer Setup: 9.97ms
[2025-06-29, 19:00:18.714] -> [Profile] Bind Group Setup: 255.50µs
[2025-06-29, 19:00:18.717] -> [Profile] Dispatch & Submit: 2.47ms
[2025-06-29, 19:00:18.719] -> [Profile] Readback (map/poll/copy): 1.67ms
[2025-06-29, 19:00:18.719] -> [Profile] Total launch_gpu_kernel Time: 15.01ms
[2025-06-29, 19:00:18.751] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER" ("Dx12")
[2025-06-29, 19:00:18.752] -> SUBTEST: Running on "Microsoft Basic Render Driver" ("Dx12")
[2025-06-29, 19:00:18.752] -> SKIPPING: Microsoft Basic Render Driver ("Dx12")
[2025-06-29, 19:00:18.753] -> SUBTEST: Running on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-29, 19:00:18.774] -> [Profile] Buffer Setup: 1.37ms
[2025-06-29, 19:00:18.774] -> [Profile] Bind Group Setup: 73.30µs
[2025-06-29, 19:00:18.776] -> [Profile] Dispatch & Submit: 1.57ms
[2025-06-29, 19:00:18.780] -> [Profile] Readback (map/poll/copy): 4.02ms
[2025-06-29, 19:00:18.780] -> [Profile] Total launch_gpu_kernel Time: 8.00ms
[2025-06-29, 19:00:18.782] -> PASS: Kernel correctness on "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" ("OpenGL")
[2025-06-29, 19:00:18.783] -> Cross-device consistency test PASSED on all tested devices.
[2025-06-29, 18:59:46.534] -> Running streaming_load_test...
[2025-06-29, 18:59:46.952] -> streaming_load_test passed.
[2025-06-29, 19:00:18.864] -> Running streaming_load_test...
[2025-06-29, 19:00:19.277] -> streaming_load_test passed.
[2025-06-29, 18:59:21.960] -> [Profile] Buffer Setup: 22.60ms
[2025-06-29, 18:59:21.960] -> [Profile] Bind Group Setup: 268.00µs
[2025-06-29, 18:59:21.961] -> [Profile] Dispatch & Submit: 630.40µs
[2025-06-29, 18:59:21.962] -> [Profile] Readback (map/poll/copy): 193.50µs
[2025-06-29, 18:59:21.962] -> [Profile] Total launch_gpu_kernel Time: 24.70ms
[2025-06-29, 18:59:54.402] -> [Profile] Buffer Setup: 10.11ms
[2025-06-29, 18:59:54.402] -> [Profile] Bind Group Setup: 244.60µs
[2025-06-29, 18:59:54.403] -> [Profile] Dispatch & Submit: 554.80µs
[2025-06-29, 18:59:54.403] -> [Profile] Readback (map/poll/copy): 119.40µs
[2025-06-29, 18:59:54.403] -> [Profile] Total launch_gpu_kernel Time: 11.75ms
[2025-06-29, 19:00:22.927] -> [Profile] Buffer Setup: 738.80µs
[2025-06-29, 19:00:22.927] -> [Profile] Bind Group Setup: 72.10µs
[2025-06-29, 19:00:22.928] -> [Profile] Dispatch & Submit: 206.80µs
[2025-06-29, 19:00:22.928] -> [Profile] Readback (map/poll/copy): 155.70µs
[2025-06-29, 19:00:22.928] -> [Profile] Total launch_gpu_kernel Time: 1.83ms
[2025-06-29, 19:00:25.775] -> [Profile] Buffer Setup: 813.00µs
[2025-06-29, 19:00:25.775] -> [Profile] Bind Group Setup: 81.40µs
[2025-06-29, 19:00:25.776] -> [Profile] Dispatch & Submit: 320.10µs
[2025-06-29, 19:00:25.776] -> [Profile] Readback (map/poll/copy): 156.40µs
[2025-06-29, 19:00:25.776] -> [Profile] Total launch_gpu_kernel Time: 2.04ms
[2025-06-29, 18:59:16.323] -> Successfully caught invalid weight value (2) as a Result::Err.
[2025-06-29, 18:59:16.323] -> Successfully caught invalid weight value (-2) as a Result::Err.
[2025-06-29, 18:59:46.980] -> Running memory_safety_hardcoded_large_allocation_test... (SAFE)
[2025-06-29, 18:59:47.370] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-29, 19:00:19.307] -> Running memory_safety_hardcoded_large_allocation_test... (OPTIMAL)
[2025-06-29, 19:00:19.681] -> memory_safety_hardcoded_large_allocation_test passed.
[2025-06-29, 19:00:22.944] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-29, 19:00:25.794] -> [WARM] Successfully caught expected error for 10GB allocation: Requested buffer size (10737418240 bytes) exceeds device limits.
[2025-06-29, 18:59:16.745] -> Running memory_safety_buffer_overflow_test...
[2025-06-29, 18:59:17.132] -> memory_safety_buffer_overflow_test passed.
[2025-06-29, 18:59:47.398] -> Running stress_test_maximum_dimension_support... (SAFE)
[2025-06-29, 18:59:48.520] -> [Profile] Buffer Setup: 17.39ms
[2025-06-29, 18:59:48.520] -> [Profile] Bind Group Setup: 256.20µs
[2025-06-29, 18:59:48.521] -> [Profile] Dispatch & Submit: 660.80µs
[2025-06-29, 18:59:48.615] -> [Profile] Readback (map/poll/copy): 94.30ms
[2025-06-29, 18:59:48.616] -> [Profile] Total launch_gpu_kernel Time: 113.41ms
[2025-06-29, 18:59:49.685] -> stress_test_maximum_dimension_support passed.
[2025-06-29, 19:00:19.707] -> Running stress_test_maximum_dimension_support... (OPTIMAL)
[2025-06-29, 19:00:20.830] -> [Profile] Buffer Setup: 17.76ms
[2025-06-29, 19:00:20.830] -> [Profile] Bind Group Setup: 264.50µs
[2025-06-29, 19:00:20.831] -> [Profile] Dispatch & Submit: 622.00µs
[2025-06-29, 19:00:20.923] -> [Profile] Readback (map/poll/copy): 91.50ms
[2025-06-29, 19:00:20.923] -> [Profile] Total launch_gpu_kernel Time: 111.00ms
[2025-06-29, 19:00:21.992] -> stress_test_maximum_dimension_support passed.
[2025-06-29, 19:00:22.944] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-29, 19:00:23.678] -> [Profile] Buffer Setup: 10.39ms
[2025-06-29, 19:00:23.678] -> [Profile] Bind Group Setup: 221.00µs
[2025-06-29, 19:00:23.679] -> [Profile] Dispatch & Submit: 452.20µs
[2025-06-29, 19:00:23.771] -> [Profile] Readback (map/poll/copy): 91.60ms
[2025-06-29, 19:00:23.771] -> [Profile] Total launch_gpu_kernel Time: 103.59ms
[2025-06-29, 19:00:25.794] -> [WARM] Successfully caught expected error: Requested buffer size (268435520 bytes) exceeds device limits.
[2025-06-29, 19:00:26.527] -> [Profile] Buffer Setup: 10.33ms
[2025-06-29, 19:00:26.527] -> [Profile] Bind Group Setup: 216.50µs
[2025-06-29, 19:00:26.528] -> [Profile] Dispatch & Submit: 441.80µs
[2025-06-29, 19:00:26.619] -> [Profile] Readback (map/poll/copy): 91.06ms
[2025-06-29, 19:00:26.620] -> [Profile] Total launch_gpu_kernel Time: 102.91ms
[2025-06-29, 18:59:16.715] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-29, 18:59:16.715] -> Requested limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-29, 18:59:16.716] -> Actual device limits returned: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 1, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-06-29, 18:59:17.161] -> Testing scalar packing-decoding symmetry...
[2025-06-29, 18:59:17.161] -> Original weights:  [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-29, 18:59:17.161] -> Decoded weights:   [-1, 0, 1, 0, 1, 1, 0, -1, -1, -1, 0, 0, 1, 1, 0, 1]
[2025-06-29, 18:59:17.162] -> Scalar packing-decoding symmetry test passed.
[2025-06-29, 18:59:15.687] -> STARTING KERNEL TEST SUITE
[2025-06-29, 18:59:17.162] -> --- STARTING COLD RUN (SAFE) ---
[2025-06-29, 18:59:17.162] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-29, 18:59:49.718] -> --- STARTING COLD RUN (OPTIMAL) ---
[2025-06-29, 18:59:49.718] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-29, 19:00:22.020] -> --- STARTING WARM RUN (SAFE) ---
[2025-06-29, 19:00:22.020] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel.wgsl
[2025-06-29, 19:00:24.869] -> --- STARTING WARM RUN (OPTIMAL) ---
[2025-06-29, 19:00:24.870] -> [KERNEL] Using shader: ../src/kernels/bitnet_kernel_optimal.wgsl
[2025-06-29, 19:00:22.401] -> [WARM] low_level_kernel_correctness_test passed. (SAFE)
[2025-06-29, 19:00:25.249] -> [WARM] low_level_kernel_correctness_test passed. (OPTIMAL)
[2025-06-29, 19:00:22.403] -> [WARM] test_gpu_kernel_dimensions passed. (SAFE)
[2025-06-29, 19:00:25.251] -> [WARM] test_gpu_kernel_dimensions passed. (OPTIMAL)
[2025-06-29, 19:00:22.406] -> [WARM] kernel_large_batch_test passed. (SAFE)
[2025-06-29, 19:00:25.255] -> [WARM] kernel_large_batch_test passed. (OPTIMAL)
[2025-06-29, 18:59:18.839] -> kernel_all_zero_test passed.
[2025-06-29, 18:59:51.379] -> kernel_all_zero_test passed.
[2025-06-29, 19:00:22.410] -> [WARM] kernel_all_zero_test passed. (SAFE)
[2025-06-29, 19:00:25.257] -> [WARM] kernel_all_zero_test passed. (OPTIMAL)
[2025-06-29, 18:59:19.295] -> kernel_all_plus_one_weights_test passed.
[2025-06-29, 18:59:51.800] -> kernel_all_plus_one_weights_test passed.
[2025-06-29, 19:00:22.413] -> [WARM] kernel_all_plus_one_weights_test passed. (SAFE)
[2025-06-29, 19:00:25.261] -> [WARM] kernel_all_plus_one_weights_test passed. (OPTIMAL)
[2025-06-29, 18:59:19.737] -> kernel_all_minus_one_weights_test passed.
[2025-06-29, 18:59:52.220] -> kernel_all_minus_one_weights_test passed.
[2025-06-29, 19:00:22.416] -> [WARM] kernel_all_minus_one_weights_test passed. (SAFE)
[2025-06-29, 19:00:25.264] -> [WARM] kernel_all_minus_one_weights_test passed. (OPTIMAL)
[2025-06-29, 18:59:20.193] -> kernel_non_divisible_batch_test passed.
[2025-06-29, 18:59:52.631] -> kernel_non_divisible_batch_test passed.
[2025-06-29, 19:00:22.420] -> [WARM] kernel_non_divisible_batch_test passed. (SAFE)
[2025-06-29, 19:00:25.267] -> [WARM] kernel_non_divisible_batch_test passed. (OPTIMAL)
[2025-06-29, 19:00:22.763] -> [WARM] test_bitlinear_layer_forward_pass passed. (SAFE)
[2025-06-29, 19:00:25.611] -> [WARM] test_bitlinear_layer_forward_pass passed. (OPTIMAL)
[2025-06-29, 19:00:22.926] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.503ms    | Total: 150.322ms 
  Scalar (CPU Time):  Avg: 107.410µs  | Total: 10.741ms  
Speedup (Wall vs Scalar):   0.07x
[2025-06-29, 19:00:22.926] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (SAFE)
[2025-06-29, 19:00:25.773] -> [WARM] Performance Benchmark (100 iterations, 64 batch, 32 in, 16 out):
  GPU (Wall Time):    Avg: 1.504ms    | Total: 150.420ms 
  Scalar (CPU Time):  Avg: 105.361µs  | Total: 10.536ms  
Speedup (Wall vs Scalar):   0.07x
[2025-06-29, 19:00:25.774] -> [WARM] performance_benchmark_gpu_vs_scalar passed. (OPTIMAL)
[2025-06-29, 19:00:22.929] -> [WARM] precision_test_fp_edge_cases passed. (SAFE)
[2025-06-29, 19:00:25.777] -> [WARM] precision_test_fp_edge_cases passed. (OPTIMAL)
[2025-06-29, 19:00:22.944] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.398ms
[2025-06-29, 19:00:25.793] -> [WARM] Streaming Load Test (10 streams): Avg Latency: 1.546ms
[2025-06-29, 19:00:24.839] -> [WARM] stress_test_maximum_dimension_support passed. (SAFE)
[2025-06-29, 19:00:27.684] -> [WARM] stress_test_maximum_dimension_support passed. (OPTIMAL)
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 45
- **Passed:** 45
- **Failed:** 0

### Timing Information

- **Total Time:** 42.07 sec
- **Average Time:** 934.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
