# KERNEL_TESTS Test Report

> Generated on: 2025-06-20 00:33:37

## Test Results

| No. | Test Name | Status | Time Taken |
|:---:|:----------|:------:|:----------:|
|  1 | Edge Case Invalid Input Weights                    | ✅ Pass |    0.00 ms |
|  2 | Error Handling GPU Unavailable                     | ✅ Pass |  382.00 ms |
|  3 | Memory Safety Buffer Overflow Test                 | ✅ Pass |  378.00 ms |
|  4 | Cross Device Consistency Test                      | ✅ Pass |  450.00 ms |
|  5 | Kernel All Minus One Weights Test                  | ✅ Pass |  394.00 ms |
|  6 | Kernel All Plus One Weights Test                   | ✅ Pass |  395.00 ms |
|  7 | Kernel All Zero Test                               | ✅ Pass |  392.00 ms |
|  8 | Kernel Large Batch Test                            | ✅ Pass |  412.00 ms |
|  9 | Kernel Non Divisible Batch Test                    | ✅ Pass |  387.00 ms |
| 10 | Low Level Kernel Correctness Test                  | ✅ Pass |  419.00 ms |
| 11 | Performance Benchmark GPU Vs Scalar                | ✅ Pass |   1.32 sec |
| 12 | Precision Test Fp Edge Cases                       | ✅ Pass |  388.00 ms |
| 13 | Streaming Load Test                                | ✅ Pass |  499.00 ms |
| 14 | Stress Test Maximum Dimension Support              | ✅ Pass |  18.03 sec |
| 15 | Basic GPU Buffer Operations                        | ✅ Pass |  565.00 ms |
| 16 | Bitlinear Layer Forward Pass                       | ✅ Pass |  693.00 ms |
| 17 | GPU Kernel Dimensions                              | ✅ Pass |  379.00 ms |
| 18 | Matmul Quantized Scalar                            | ✅ Pass |    0.00 ms |
| 19 | Unit Test Calculate Weight Scales                  | ✅ Pass |    0.00 ms |
| 20 | Unit Test Pack Ternary Weights                     | ✅ Pass |    1.00 ms |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-20, 00:33:09.457] -> Running unit_test_pack_ternary_weights...
[2025-06-20, 00:33:09.459] -> unit_test_pack_ternary_weights passed.
[2025-06-20, 00:33:09.460] -> Running unit_test_calculate_weight_scales...
[2025-06-20, 00:33:09.461] -> unit_test_calculate_weight_scales passed.
[2025-06-20, 00:33:09.461] -> Starting test_matmul_quantized_scalar...
[2025-06-20, 00:33:09.462] -> Output values: [8.0, 4.0]
[2025-06-20, 00:33:09.464] -> Testing basic GPU operations...
[2025-06-20, 00:33:10.029] -> Basic GPU operations test passed!
[2025-06-20, 00:33:10.518] -> Running test_gpu_kernel_dimensions...
[2025-06-20, 00:33:10.897] -> test_gpu_kernel_dimensions passed.
[2025-06-20, 00:33:10.931] -> Running kernel_large_batch_test...
[2025-06-20, 00:33:11.343] -> kernel_large_batch_test passed.
[2025-06-20, 00:33:11.347] -> Running kernel_all_zero_test...
[2025-06-20, 00:33:11.739] -> kernel_all_zero_test passed.
[2025-06-20, 00:33:11.743] -> Running kernel_all_plus_one_weights_test...
[2025-06-20, 00:33:12.138] -> kernel_all_plus_one_weights_test passed.
[2025-06-20, 00:33:12.142] -> Running kernel_all_minus_one_weights_test...
[2025-06-20, 00:33:12.536] -> kernel_all_minus_one_weights_test passed.
[2025-06-20, 00:33:12.540] -> Running kernel_non_divisible_batch_test...
[2025-06-20, 00:33:12.927] -> kernel_non_divisible_batch_test passed.
[2025-06-20, 00:33:12.932] -> Running test_bitlinear_layer_forward_pass...
[2025-06-20, 00:33:13.625] -> test_bitlinear_layer_forward_pass passed.
[2025-06-20, 00:33:15.393] -> Performance Benchmark (100 iterations):
GPU Avg Time: 13.204ms | Scalar Avg Time: 464.850µs
Speedup: 0.04x
[2025-06-20, 00:33:16.100] -> Found 5 adapters. Running consistency test.
[2025-06-20, 00:33:16.100] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Vulkan)
[2025-06-20, 00:33:16.204] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 00:33:16.204] -> Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to known DXC compiler bug.
[2025-06-20, 00:33:16.229] -> Testing on device: "Microsoft Basic Render Driver" (Dx12)
[2025-06-20, 00:33:16.229] -> Skipping test on "Microsoft Basic Render Driver" (Dx12) due to known DXC compiler bug.
[2025-06-20, 00:33:16.231] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER" (Dx12)
[2025-06-20, 00:33:16.231] -> Skipping test on "NVIDIA GeForce RTX 2070 SUPER" (Dx12) due to known DXC compiler bug.
[2025-06-20, 00:33:16.272] -> Testing on device: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2" (Gl)
[2025-06-20, 00:33:16.297] -> Kernel correctness passed on all available devices.
[2025-06-20, 00:33:16.803] -> Streaming Load Test (10 streams): Avg Latency: 13.387ms
[2025-06-20, 00:33:15.815] -> Precision test with FP edge cases passed.
[2025-06-20, 00:33:17.215] -> Running edge_case_invalid_input_weights...
[2025-06-20, 00:33:17.215] -> edge_case_invalid_input_weights passed (panicked as expected).
[2025-06-20, 00:33:17.216] -> Running memory_safety_buffer_overflow_test...
[2025-06-20, 00:33:17.594] -> memory_safety_buffer_overflow_test passed (panicked as expected).
[2025-06-20, 00:33:16.832] -> Running error_handling_gpu_unavailable...
[2025-06-20, 00:33:17.185] -> WGPU context creation succeeded unexpectedly with impossible limits.
[2025-06-20, 00:33:17.214] -> error_handling_gpu_unavailable passed.
[2025-06-20, 00:33:17.594] -> Running stress_test_maximum_dimension_support...
[2025-06-20, 00:33:17.596] -> Stress Test: Initializing with large dimensions (1024x1024x1024)...
[2025-06-20, 00:33:17.964] -> Stress Test: Generating random data (this may take a moment)...
[2025-06-20, 00:33:18.619] -> Stress Test: Data generation complete. Time: 1.02s
[2025-06-20, 00:33:18.619] -> Stress Test: Starting scalar pre-computation...
[2025-06-20, 00:33:18.696] -> Stress Test: Scalar pre-computation complete. Time: 76.62ms
[2025-06-20, 00:33:18.696] -> Stress Test: Starting GPU execution...
[2025-06-20, 00:33:18.804] -> Stress Test: GPU execution complete. Time: 106.93ms
[2025-06-20, 00:33:18.804] -> Stress Test: Starting scalar reference execution (this will be slow)...
[2025-06-20, 00:33:35.601] -> Stress Test: Scalar reference execution complete. Time: 16.80s
[2025-06-20, 00:33:35.601] -> Stress Test: Comparing results...
[2025-06-20, 00:33:35.626] -> Stress Test: Comparison complete. Test passed! Total time: 18.03s
[2025-06-20, 00:33:35.666] -> stress_test_maximum_dimension_support passed.
[2025-06-20, 00:33:09.456] -> STARTING KERNEL TEST SUITE
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 20
- **Passed:** 20
- **Failed:** 0

### Timing Information

- **Total Time:** 25.50 sec
- **Average Time:** 1274.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
