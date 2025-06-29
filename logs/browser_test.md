# BROWSER_TEST Test Report

> Generated on: 2025-06-29 17:41:08

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Run Test Inline Add Scalar Shader Compile          | ✅ Pass |   54.00 ms |             |
|  2 | Run Test Kernel Compile Default                    | ✅ Pass |   63.00 ms |             |
|  3 | Run Test Kernel Compile Optimal                    | ✅ Pass |   59.00 ms |             |
|  4 | Run Test Kernel Compile Wasm                       | ✅ Pass |   75.00 ms |             |
|  5 | Run Test Minimal Shader Compile                    | ✅ Pass |   61.00 ms |             |
|  6 | Ping GPU Info                                      | ✅ Pass |   54.00 ms |             |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-06-29, 17:41:08.264] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Switched to tab: test
[2025-06-29, 17:41:08.264] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] DOMContentLoaded event fired, calling setupTestButtons...
[2025-06-29, 17:41:08.264] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Initializing WASM via init()...
[2025-06-29, 17:41:08.264] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] WASM initialized. Exports:
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test list from WASM:
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: run_test_kernel_compile_optimal
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: run_test_kernel_compile_wasm
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: run_test_kernel_compile_default
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: run_test_inline_add_scalar_shader_compile
[2025-06-29, 17:41:08.265] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: test_ping_gpu_info
[2025-06-29, 17:41:08.266] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Test button generated: run_test_minimal_shader_compile
[2025-06-29, 17:41:08.266] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [BitNet] Generate Report button handler attached (robust).
[2025-06-29, 17:41:08.266] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [LogBridge] Connected to Rust WebSocket log server
[2025-06-29, 17:41:08.266] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Found 6 tests: run_test_kernel_compile_optimal, run_test_kernel_compile_wasm, run_test_kernel_compile_default, run_test_inline_add_scalar_shader_compile, test_ping_gpu_info, run_test_minimal_shader_compile
[2025-06-29, 17:41:08.266] -> [INFO] [INFO] [http://localhost:8080/main.js:193] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Starting test run for 6 tests...
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 1/6: run_test_kernel_compile_optimal
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 1/6: run_test_kernel_compile_optimal
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_optimal: ✅ PASS in 165.40 ms
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 2/6: run_test_kernel_compile_wasm
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 2/6: run_test_kernel_compile_wasm
[2025-06-29, 17:41:08.267] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_wasm: ✅ PASS in 35.20 ms
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 3/6: run_test_kernel_compile_default
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 3/6: run_test_kernel_compile_default
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_default: ✅ PASS in 36.00 ms
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 4/6: run_test_inline_add_scalar_shader_compile
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 4/6: run_test_inline_add_scalar_shader_compile
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_inline_add_scalar_shader_compile: ✅ PASS in 38.50 ms
[2025-06-29, 17:41:08.268] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 5/6: test_ping_gpu_info
[2025-06-29, 17:41:08.269] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 5/6: test_ping_gpu_info
[2025-06-29, 17:41:08.269] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test test_ping_gpu_info: ✅ PASS in 40.30 ms
[2025-06-29, 17:41:08.269] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 6/6: run_test_minimal_shader_compile
[2025-06-29, 17:41:08.269] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 6/6: run_test_minimal_shader_compile
[2025-06-29, 17:41:08.269] -> [INFO] [test_minimal_shader_compile] Creating wgpu device...
[2025-06-29, 17:41:08.269] -> [INFO] [test_minimal_shader_compile] Device created.
[2025-06-29, 17:41:08.270] -> [INFO] [test_minimal_shader_compile] Compiling shader...
[2025-06-29, 17:41:08.270] -> [INFO] [test_minimal_shader_compile] Shader compiled successfully.
[2025-06-29, 17:41:08.270] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_minimal_shader_compile: ✅ PASS in 42.40 ms
[2025-06-29, 17:41:08.270] -> [INFO] [INFO] [http://localhost:8080/main.js:204] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] All tests complete. Sending report to server...
[2025-06-29, 17:41:08.270] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [UI] Sent structured test report to server.
[2025-06-29, 17:41:08.270] -> [INFO] [INFO] [http://localhost:8080/main.js:214] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Report sent to server.
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Found 6 tests: run_test_kernel_compile_optimal, run_test_kernel_compile_wasm, run_test_kernel_compile_default, run_test_inline_add_scalar_shader_compile, test_ping_gpu_info, run_test_minimal_shader_compile
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:193] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Starting test run for 6 tests...
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 1/6: run_test_kernel_compile_optimal
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 1/6: run_test_kernel_compile_optimal
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_optimal: ✅ PASS in 59.20 ms
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 2/6: run_test_kernel_compile_wasm
[2025-06-29, 17:41:08.271] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 2/6: run_test_kernel_compile_wasm
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_wasm: ✅ PASS in 75.50 ms
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 3/6: run_test_kernel_compile_default
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 3/6: run_test_kernel_compile_default
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_kernel_compile_default: ✅ PASS in 63.80 ms
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 4/6: run_test_inline_add_scalar_shader_compile
[2025-06-29, 17:41:08.272] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 4/6: run_test_inline_add_scalar_shader_compile
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_inline_add_scalar_shader_compile: ✅ PASS in 54.70 ms
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 5/6: test_ping_gpu_info
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 5/6: test_ping_gpu_info
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test test_ping_gpu_info: ✅ PASS in 54.80 ms
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:98] [Object.info] [TestRunner] Running test 6/6: run_test_minimal_shader_compile
[2025-06-29, 17:41:08.273] -> [INFO] [INFO] [http://localhost:8080/main.js:197] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Running test 6/6: run_test_minimal_shader_compile
[2025-06-29, 17:41:08.274] -> [INFO] [test_minimal_shader_compile] Creating wgpu device...
[2025-06-29, 17:41:08.274] -> [INFO] [test_minimal_shader_compile] Device created.
[2025-06-29, 17:41:08.274] -> [INFO] [test_minimal_shader_compile] Compiling shader...
[2025-06-29, 17:41:08.274] -> [INFO] [test_minimal_shader_compile] Shader compiled successfully.
[2025-06-29, 17:41:08.274] -> [INFO] [INFO] [http://localhost:8080/main.js:202] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] Finished test run_test_minimal_shader_compile: ✅ PASS in 61.40 ms
[2025-06-29, 17:41:08.274] -> [INFO] [INFO] [http://localhost:8080/main.js:204] [HTMLButtonElement.runAllTestsAndSendReport] [TestRunner] All tests complete. Sending report to server...
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 6
- **Passed:** 6
- **Failed:** 0

### Timing Information

- **Total Time:** 0.37 sec
- **Average Time:** 61.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
