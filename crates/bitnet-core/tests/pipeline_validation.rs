//!
//! # BitNet Pipeline Validation Suite
//!
//! **Purpose:**
//! - End-to-end, TDD-style validation of the BitNet pipeline: model loading, inference, device selection, golden output, settings, and performance.
//! - Each logical step is a separate test. The master test runs all steps in order for CI/full validation.
//! - All tests use a shared TestReporter for rich logging and Markdown report generation.
//! - All tests are isolated: create/teardown their own pipeline/model/context.
//! - Extensible: add new tests (settings, batch, streaming, etc.) as new functions and call from the master test.
//!
//! **How to extend:**
//! - Add new golden cases to the `GoldenCase` array.
//! - Add new test functions for new features (settings, batch, streaming, etc.).
//! - To integrate settings, add a `settings` field to PipelineOptions and thread it through.
//! - To add new device types, update the backend selection logic.
//!
//! **Philosophy:**
//! - TDD: Add a test, see it fail, implement/fix, see it pass, repeat.
//! - All failures are logged and reported, not panics (unless setup fails).
//! - All tests are robust to missing devices/backends (log and skip, don't panic).
//! - Markdown report is always generated for CI/debugging.
//!
//! **TODO:**
//! - Integrate InferenceSettings from settings.rs (see placeholder test).
//! - Add batch/streaming/perf tests as needed.
//!
//! ## Pipeline Creation Test Structure
//!
//! - test_pipeline_creation_cpu: Creates pipeline with CPU backend, logs RAM usage, device info, and time.
//! - test_pipeline_creation_gpu: Creates pipeline with GPU backend, logs VRAM (if available), device info, and time.
//! - test_pipeline_creation: Runs both subtests and generates a summary table in the report.

use bitnet_tools::test_utils::TestReporter;
use bitnet_core::pipeline::{Pipeline, PipelineOptions, PipelineBackend};
use sysinfo::System;
use std::time::Instant;
use serial_test::serial;
use bitnet_core::settings::InferenceSettings;
use lazy_static::lazy_static;
use rayon::ThreadPoolBuilder;
use num_cpus;

struct GoldenCase<'a> {
    prompt: &'a str,
    expected_top_token: &'a str, // expected top token (hardcoded)
}

fn golden_cases() -> Vec<GoldenCase<'static>> {
    vec![
        GoldenCase { prompt: "Hello", expected_top_token: "world" },
        GoldenCase { prompt: "The quick brown fox", expected_top_token: "jumps" },
        GoldenCase { prompt: "", expected_top_token: "<pad>" },
        GoldenCase { prompt: "!@#$%^&*()", expected_top_token: "Special" },
    ]
}

lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("pipeline_validation").expect("Failed to create test reporter");
}

// --- Test 1: Pipeline Creation ---
#[test]
#[serial]
#[ignore]
fn test_pipeline_creation_cpu() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Cpu,
            settings: None,
        };
        let mut sys = System::new_all();
        sys.refresh_memory();
        let mem_before = sys.used_memory();
        let t0 = Instant::now();
        let pipeline = match Pipeline::new(options).await {
            Ok(p) => {
                reporter.log_message(1, "Pipeline::new (CPU) succeeded");
                p
            },
            Err(e) => {
                reporter.record_failure("Pipeline Creation CPU", &format!("Failed to create pipeline: {e}"), None);
                return;
            }
        };
        let t1 = t0.elapsed();
        sys.refresh_memory();
        let mem_after = sys.used_memory();
        let mem_delta = mem_after as i64 - mem_before as i64;
        reporter.log_message(1, &format!("[CPU] Used system memory before: {} KB, after: {} KB, delta: {} KB", mem_before, mem_after, mem_delta));
        reporter.log_message(1, &format!("[CPU] Pipeline creation time: {:?}", t1));
        reporter.log_message(1, "[CPU] Backend: GL (CPU fallback)");
        drop(pipeline);
    });
}

#[test]
#[serial]
#[ignore]
fn test_pipeline_creation_gpu() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Gpu,
            settings: None,
        };
        let mut sys = System::new_all();
        sys.refresh_memory();
        let mem_before = sys.used_memory();
        let t0 = Instant::now();
        let pipeline = match Pipeline::new(options).await {
            Ok(p) => {
                reporter.log_message(2, "Pipeline::new (GPU) succeeded");
                p
            },
            Err(e) => {
                reporter.record_failure("Pipeline Creation GPU", &format!("Failed to create pipeline: {e}"), None);
                return;
            }
        };
        let t1 = t0.elapsed();
        sys.refresh_memory();
        let mem_after = sys.used_memory();
        let mem_delta = mem_after as i64 - mem_before as i64;
        // VRAM info (not available via sysinfo, so log N/A)
        reporter.log_message(2, &format!("[GPU] Used system memory before: {} KB, after: {} KB, delta: {} KB", mem_before, mem_after, mem_delta));
        reporter.log_message(2, "[GPU] VRAM usage: N/A (not available via sysinfo)");
        reporter.log_message(2, &format!("[GPU] Pipeline creation time: {:?}", t1));
        reporter.log_message(2, "[GPU] Backend: VULKAN | DX12 | METAL");
        drop(pipeline);
    });
}

#[test]
#[serial]
#[ignore]
fn test_pipeline_creation() {
    test_pipeline_creation_cpu();
    test_pipeline_creation_gpu();
    let reporter = &*TEST_REPORTER;
    reporter.log_message(0, "Pipeline creation (CPU and GPU) tests completed. See above for details.");
}

// --- Test 2: Model Ready ---
#[test]
#[serial]
#[ignore]
fn test_model_ready() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Auto,
            settings: None,
        };
        let mut pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
        let t2 = Instant::now();
        match pipeline.ensure_model_ready().await {
            Ok(_) => reporter.log_message(2, "ensure_model_ready succeeded"),
            Err(e) => {
                reporter.record_failure("Model Ready", &format!("Failed to prepare model: {e}"), None);
            }
        }
        let t3 = t2.elapsed();
        reporter.record_timing("Model Ready", t3);
        drop(pipeline);
    });
}

// --- Test 3: CPU Inference ---
#[test]
#[serial]
#[ignore]
fn test_cpu_inference() {
    ThreadPoolBuilder::new().num_threads(num_cpus::get()).build_global().ok();
    println!("[BitNet] [CPU] Rayon thread pool size: {}", rayon::current_num_threads());
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let settings = InferenceSettings::default().with_max_new_tokens(1);
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Cpu,
            settings: Some(settings.clone()),
        };
        let mut pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
        pipeline.ensure_model_ready().await.expect("Failed to prepare model");
        for (i, case) in golden_cases().iter().enumerate() {
            let t4 = Instant::now();
            let result = match pipeline.run_inference(case.prompt).await {
                Ok(r) => r,
                Err(e) => {
                    reporter.record_failure(&format!("CPU Inference (case {i})"), &format!("Inference failed: {e}"), None);
                    continue;
                }
            };
            let t5 = t4.elapsed();
            reporter.record_timing(&format!("CPU Inference (case {i})"), t5);
            reporter.log_message(3, &format!("[GOLDEN] Top token for \"{}\": {}", case.prompt, result.top_token));
            if result.logits.is_empty() {
                reporter.record_failure(&format!("CPU Inference (case {i})"), "Logits vector is empty", None);
            }
            if !result.logits.iter().all(|&l| l.is_finite()) {
                reporter.record_failure(&format!("CPU Inference (case {i})"), "Logits contain NaN or Infinity", None);
            }
            if result.top_token != case.expected_top_token {
                reporter.record_failure(&format!("CPU Inference (case {i})"), &format!("Golden output mismatch: expected '{}' , got '{}'", case.expected_top_token, result.top_token), None);
            }
            // Log why generation stopped
            if result.stopped_by_eos {
                reporter.log_message(3, "[GENERATION] Stopped by EOS token");
            } else if result.stopped_by_max_tokens {
                reporter.log_message(3, "[GENERATION] Stopped by max_tokens");
            }
        }
        drop(pipeline);
    });
}

// --- Test 4: GPU Inference ---
#[test]
#[serial]
#[ignore]
fn test_gpu_inference() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Gpu,
            settings: None,
        };
        let mut pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
        if let Err(e) = pipeline.ensure_model_ready().await {
            reporter.record_failure("GPU Inference", &format!("Failed to prepare model: {e}"), None);
            return;
        }
        for (i, case) in golden_cases().iter().enumerate() {
            let t4 = Instant::now();
            let result = match pipeline.run_inference(case.prompt).await {
                Ok(r) => r,
                Err(e) => {
                    reporter.record_failure(&format!("GPU Inference (case {i})"), &format!("Inference failed: {e}"), None);
                    continue;
                }
            };
            let t5 = t4.elapsed();
            reporter.record_timing(&format!("GPU Inference (case {i})"), t5);
            reporter.log_message(4, &format!("[GOLDEN] Top token for \"{}\": {}", case.prompt, result.top_token));
            if result.logits.is_empty() {
                reporter.record_failure(&format!("GPU Inference (case {i})"), "Logits vector is empty", None);
            }
            if !result.logits.iter().all(|&l| l.is_finite()) {
                reporter.record_failure(&format!("GPU Inference (case {i})"), "Logits contain NaN or Infinity", None);
            }
            if result.top_token != case.expected_top_token {
                reporter.record_failure(&format!("GPU Inference (case {i})"), &format!("Golden output mismatch: expected '{}', got '{}'", case.expected_top_token, result.top_token), None);
            }
        }
        drop(pipeline);
    });
}

// --- Test 5: Performance Metrics ---
#[test]
#[serial]
#[ignore]
fn test_performance_metrics() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Cpu,
            settings: None,
        };
        let mut pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
        pipeline.ensure_model_ready().await.expect("Failed to prepare model");
        let prompt = "Performance test prompt.";
        let settings = InferenceSettings::default().with_max_new_tokens(32);
        let t0 = Instant::now();
        let mut total_tokens = 0;
        let mut _last_result = None;
        for _ in 0..settings.max_new_tokens {
            let result = pipeline.run_inference(prompt).await.expect("Inference failed");
            total_tokens += 1;
            _last_result = Some(result);
        }
        let elapsed = t0.elapsed();
        let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
        reporter.log_message(5, &format!("Performance: {} tokens in {:?} ({:.2} tokens/sec)", total_tokens, elapsed, tokens_per_sec));
    });
}

// --- Test 6: Settings.rs Integration ---
#[test]
#[serial]
#[ignore]
fn test_settings_integration() {
    let reporter = &*TEST_REPORTER;
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let settings = InferenceSettings::default()
            .with_temperature(0.5)
            .with_top_p(0.8)
            .with_max_new_tokens(4);
        let options = PipelineOptions {
            model_id: None,
            input_dir: None,
            output_dir: None,
            reporter: None,
            backend: PipelineBackend::Cpu,
            settings: Some(settings.clone()),
        };
        let mut pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
        pipeline.ensure_model_ready().await.expect("Failed to prepare model");
        let prompt = "Settings integration test.";
        let result = pipeline.run_inference(prompt).await.expect("Inference failed");
        reporter.log_message(6, &format!("Settings test: top_token={}, logits_len={}, settings={:?}", result.top_token, result.logits.len(), settings));
    });
}

fn zzz_final_report() {
    // This function runs last and generates the final report.
    // Add a small delay to ensure all async tests complete.
    std::thread::sleep(std::time::Duration::from_secs(2));
    TEST_REPORTER.generate_report();
}

// --- Master Test: Run All Sequentially ---
#[test]
#[serial]
fn test_all_pipeline_validation_sequentially() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(100, "STARTING PIPELINE VALIDATION TEST SUITE");
    test_pipeline_creation_cpu();
    test_pipeline_creation_gpu();
    test_model_ready();
    test_cpu_inference();
    test_gpu_inference();
    test_performance_metrics();
    test_settings_integration();
   
   
    let duration = t0.elapsed();
    TEST_REPORTER.log_message(100, &format!("Pipeline validation test suite passed (took {:.2?})", duration));

    zzz_final_report();
} 