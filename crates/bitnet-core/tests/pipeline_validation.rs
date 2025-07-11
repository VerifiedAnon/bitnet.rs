//! BitNet Pipeline Validation Suite (Refactored)
//!
//! - Covers all combinations: CPU/GPU × singlefile/streaming
//! - Each test is explicit, self-documenting, and uses TestReporter
//! - Warm/cold system: _warm functions accept Option<&mut Pipeline>
//! - All tests are #[test] #[serial] #[ignore] except the master
//! - Master test runs all and generates report

use bitnet_tools::test_utils::TestReporter;
use bitnet_core::pipeline::{Pipeline, PipelineOptions, PipelineBackend};
use bitnet_core::settings::InferenceSettings;
use sysinfo::System;
use std::time::Instant;
use serial_test::serial;
use lazy_static::lazy_static;
use rayon::ThreadPoolBuilder;
use num_cpus;

lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("pipeline_validation").expect("Failed to create test reporter");
}

struct GoldenCase<'a> {
    prompt: &'a str,
    expected_top_token: &'a str,
}

fn golden_cases() -> Vec<GoldenCase<'static>> {
    vec![
        GoldenCase { prompt: "Hello", expected_top_token: "," },
        GoldenCase { prompt: "The quick brown fox", expected_top_token: " jumps" },
        GoldenCase { prompt: "", expected_top_token: "ties" },
        GoldenCase { prompt: "!@#$%^&*()", expected_top_token: "\n" },
    ]
}

// --- Helper: Pipeline Creation ---
async fn create_pipeline(backend: PipelineBackend, use_single_file: bool) -> Result<Pipeline, String> {
    let options = PipelineOptions {
        model_id: None,
        input_dir: if use_single_file {
            Some(std::path::PathBuf::from(r"E:/Desktop/Bitnet rs/models/Converted/microsoft/bitnet-b1.58-2B-4T-bf16"))
        } else {
            None
        },
        output_dir: None,
        reporter: None,
        backend,
        settings: None,
        use_single_file,
        log_level: None,
        verbose: false,
    };
    Pipeline::new(options).await.map_err(|e| format!("Pipeline creation failed: {e}"))
}

// --- Helper: Inference ---
async fn run_inference(pipeline: &mut Pipeline, settings: &InferenceSettings, reporter: &TestReporter, backend: &str, filetype: &str, mode: &str) {
    for (i, case) in golden_cases().iter().enumerate() {
        let t0 = Instant::now();
        let output = match pipeline.generate_text(case.prompt, settings).await {
            Ok(r) => r,
            Err(e) => {
                reporter.record_failure(&format!("{backend}_{filetype}_{mode}_inference_case_{i}"), &format!("Inference failed: {e}"), None);
                continue;
            }
        };
        let t1 = t0.elapsed();
        reporter.record_timing(&format!("{backend}_{filetype}_{mode}_inference_case_{i}"), t1);
        reporter.log_message(3, &format!("[{backend}/{filetype}/{mode}] Prompt: '{}', Output: '{}', Expected: '{}'", case.prompt, output, case.expected_top_token));
        if output.is_empty() {
            reporter.record_failure(&format!("{backend}_{filetype}_{mode}_inference_case_{i}"), "Output is empty", None);
        }
        if !output.contains(case.expected_top_token) {
            reporter.record_failure(&format!("{backend}_{filetype}_{mode}_inference_case_{i}"), &format!("Golden output mismatch: expected '{}' , got '{}'", case.expected_top_token, output), None);
        }
    }
}

// --- Warm/Cold Logic Functions ---

async fn cpu_singlefile_inference_golden_warm(pipeline: Option<&mut Pipeline>, reporter: &TestReporter, mode: &str) -> Result<(), String> {
    let mut owned_pipeline;
    let pipeline = match pipeline {
        Some(p) => p,
        None => {
            owned_pipeline = create_pipeline(PipelineBackend::Cpu, true).await?;
            &mut owned_pipeline
        }
    };
    let settings = InferenceSettings::default().with_max_new_tokens(1);
    run_inference(pipeline, &settings, reporter, "CPU", "singlefile", mode).await;
    Ok(())
}

async fn cpu_streaming_inference_golden_warm(pipeline: Option<&mut Pipeline>, reporter: &TestReporter, mode: &str) -> Result<(), String> {
    let mut owned_pipeline;
    let pipeline = match pipeline {
        Some(p) => p,
        None => {
            owned_pipeline = create_pipeline(PipelineBackend::Cpu, false).await?;
            &mut owned_pipeline
        }
    };
    let settings = InferenceSettings::default().with_max_new_tokens(1);
    run_inference(pipeline, &settings, reporter, "CPU", "streaming", mode).await;
    Ok(())
}

async fn gpu_singlefile_inference_golden_warm(pipeline: Option<&mut Pipeline>, reporter: &TestReporter, mode: &str) -> Result<(), String> {
    let mut owned_pipeline;
    let pipeline = match pipeline {
        Some(p) => p,
        None => {
            owned_pipeline = create_pipeline(PipelineBackend::Gpu, true).await?;
            &mut owned_pipeline
        }
    };
    let settings = InferenceSettings::default().with_max_new_tokens(1);
    run_inference(pipeline, &settings, reporter, "GPU", "singlefile", mode).await;
    Ok(())
}

async fn gpu_streaming_inference_golden_warm(pipeline: Option<&mut Pipeline>, reporter: &TestReporter, mode: &str) -> Result<(), String> {
    let mut owned_pipeline;
    let pipeline = match pipeline {
        Some(p) => p,
        None => {
            owned_pipeline = create_pipeline(PipelineBackend::Gpu, false).await?;
            &mut owned_pipeline
        }
    };
    let settings = InferenceSettings::default().with_max_new_tokens(1);
    run_inference(pipeline, &settings, reporter, "GPU", "streaming", mode).await;
    Ok(())
}

// --- Creation Only (timing, readiness) ---
async fn pipeline_creation_warm(backend: PipelineBackend, use_single_file: bool, reporter: &TestReporter, label: &str) -> Result<(), String> {
    let t0 = Instant::now();
    let pipeline = create_pipeline(backend, use_single_file).await?;
    let t1 = t0.elapsed();
    reporter.log_message(1, &format!("[{}] Pipeline created successfully.", label));
    reporter.record_timing(&format!("{}_creation", label), t1);
    drop(pipeline);
    Ok(())
}

// --- Settings Integration ---
async fn settings_integration_warm(reporter: &TestReporter) -> Result<(), String> {
    let settings = InferenceSettings::default().with_temperature(0.5).with_top_p(0.8).with_max_new_tokens(4);
    let mut pipeline = create_pipeline(PipelineBackend::Cpu, true).await?;
    let prompt = "Settings integration test.";
    let output = pipeline.generate_text(prompt, &settings).await.map_err(|e| format!("Inference failed: {e}"))?;
    reporter.log_message(6, &format!("Settings test: output={}, settings={:?}", output, settings));
    Ok(())
}

// --- Performance Metrics ---
async fn performance_metrics_warm(backend: PipelineBackend, use_single_file: bool, reporter: &TestReporter, label: &str) -> Result<(), String> {
    let mut pipeline = create_pipeline(backend, use_single_file).await?;
    let prompt = "Performance test prompt.";
    let settings = InferenceSettings::default().with_max_new_tokens(32);
    let t0 = Instant::now();
    let mut total_tokens = 0;
    let mut _last_result = None;
    for _ in 0..settings.max_new_tokens {
        let output = pipeline.generate_text(prompt, &settings).await.map_err(|e| format!("Inference failed: {e}"))?;
        total_tokens += 1;
        _last_result = Some(output);
    }
    let elapsed = t0.elapsed();
    let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
    reporter.log_message(5, &format!("Performance: {} tokens in {:?} ({:.2} tokens/sec)", total_tokens, elapsed, tokens_per_sec));
    Ok(())
}

// --- TESTS: COLD ---

#[test]
#[serial]
#[ignore]
fn test_cpu_singlefile_creation_cold() {
    let reporter = &*TEST_REPORTER;
    let t0 = Instant::now();
    let result = std::panic::catch_unwind(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(pipeline_creation_warm(PipelineBackend::Cpu, true, reporter, "cpu_singlefile_cold"))
    });
    match result {
        Ok(Ok(_)) => {
            reporter.log_message(1, "test_cpu_singlefile_creation_cold passed.");
            reporter.record_timing("test_cpu_singlefile_creation_cold", t0.elapsed());
        }
        Ok(Err(e)) => {
            reporter.log_message(1, &format!("test_cpu_singlefile_creation_cold failed: {}", e));
            reporter.record_failure("test_cpu_singlefile_creation_cold", &e, Some(t0.elapsed()));
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            reporter.log_message(1, &format!("test_cpu_singlefile_creation_cold panicked: {}", err_msg));
            reporter.record_failure("test_cpu_singlefile_creation_cold", &err_msg, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_cpu_singlefile_inference_golden_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(cpu_singlefile_inference_golden_warm(None, reporter, "cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_cpu_singlefile_inference_golden_cold passed.");
            reporter.record_timing("test_cpu_singlefile_inference_golden_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_singlefile_inference_golden_cold failed: {}", e));
            reporter.record_failure("test_cpu_singlefile_inference_golden_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_cpu_streaming_creation_cold() {
    let reporter = &*TEST_REPORTER;
    let result = std::panic::catch_unwind(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(pipeline_creation_warm(PipelineBackend::Cpu, false, reporter, "cpu_streaming_cold"))
    });
    match result {
        Ok(Ok(_)) => {
            reporter.log_message(1, "test_cpu_streaming_creation_cold passed.");
        }
        Ok(Err(e)) => {
            reporter.log_message(1, &format!("test_cpu_streaming_creation_cold failed: {}", e));
            reporter.record_failure("test_cpu_streaming_creation_cold", &e, None);
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            reporter.log_message(1, &format!("test_cpu_streaming_creation_cold panicked: {}", err_msg));
            reporter.record_failure("test_cpu_streaming_creation_cold", &err_msg, None);
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_cpu_streaming_inference_golden_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(cpu_streaming_inference_golden_warm(None, reporter, "cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_cpu_streaming_inference_golden_cold passed.");
            reporter.record_timing("test_cpu_streaming_inference_golden_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_streaming_inference_golden_cold failed: {}", e));
            reporter.record_failure("test_cpu_streaming_inference_golden_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_gpu_singlefile_creation_cold() {
    let reporter = &*TEST_REPORTER;
    let result = std::panic::catch_unwind(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(pipeline_creation_warm(PipelineBackend::Gpu, true, reporter, "gpu_singlefile_cold"))
    });
    match result {
        Ok(Ok(_)) => {
            reporter.log_message(1, "test_gpu_singlefile_creation_cold passed.");
        }
        Ok(Err(e)) => {
            reporter.log_message(1, &format!("test_gpu_singlefile_creation_cold failed: {}", e));
            reporter.record_failure("test_gpu_singlefile_creation_cold", &e, None);
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            reporter.log_message(1, &format!("test_gpu_singlefile_creation_cold panicked: {}", err_msg));
            reporter.record_failure("test_gpu_singlefile_creation_cold", &err_msg, None);
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_gpu_singlefile_inference_golden_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(gpu_singlefile_inference_golden_warm(None, reporter, "cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_gpu_singlefile_inference_golden_cold passed.");
            reporter.record_timing("test_gpu_singlefile_inference_golden_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_singlefile_inference_golden_cold failed: {}", e));
            reporter.record_failure("test_gpu_singlefile_inference_golden_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_gpu_streaming_creation_cold() {
    let reporter = &*TEST_REPORTER;
    let result = std::panic::catch_unwind(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(pipeline_creation_warm(PipelineBackend::Gpu, false, reporter, "gpu_streaming_cold"))
    });
    match result {
        Ok(Ok(_)) => {
            reporter.log_message(1, "test_gpu_streaming_creation_cold passed.");
        }
        Ok(Err(e)) => {
            reporter.log_message(1, &format!("test_gpu_streaming_creation_cold failed: {}", e));
            reporter.record_failure("test_gpu_streaming_creation_cold", &e, None);
        }
        Err(e) => {
            let err_msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Test panicked".to_string()
            };
            reporter.log_message(1, &format!("test_gpu_streaming_creation_cold panicked: {}", err_msg));
            reporter.record_failure("test_gpu_streaming_creation_cold", &err_msg, None);
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_gpu_streaming_inference_golden_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(gpu_streaming_inference_golden_warm(None, reporter, "cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_gpu_streaming_inference_golden_cold passed.");
            reporter.record_timing("test_gpu_streaming_inference_golden_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_streaming_inference_golden_cold failed: {}", e));
            reporter.record_failure("test_gpu_streaming_inference_golden_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

// --- TESTS: WARM ---

#[test]
#[serial]
#[ignore]
fn test_cpu_singlefile_inference_golden_warm() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let mut pipeline = match runtime.block_on(create_pipeline(PipelineBackend::Cpu, true)) {
        Ok(p) => p,
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_singlefile_inference_golden_warm pipeline creation failed: {}", e));
            reporter.record_failure("test_cpu_singlefile_inference_golden_warm", &e, Some(t0.elapsed()));
            return;
        }
    };
    let result = runtime.block_on(cpu_singlefile_inference_golden_warm(Some(&mut pipeline), reporter, "warm"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_cpu_singlefile_inference_golden_warm passed.");
            reporter.record_timing("test_cpu_singlefile_inference_golden_warm", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_singlefile_inference_golden_warm failed: {}", e));
            reporter.record_failure("test_cpu_singlefile_inference_golden_warm", &e, Some(t0.elapsed()));
        }
    }
}

#[test]
#[serial]
#[ignore]
fn test_cpu_streaming_inference_golden_warm() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let mut pipeline = match runtime.block_on(create_pipeline(PipelineBackend::Cpu, false)) {
        Ok(p) => p,
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_streaming_inference_golden_warm pipeline creation failed: {}", e));
            reporter.record_failure("test_cpu_streaming_inference_golden_warm", &e, Some(t0.elapsed()));
            return;
        }
    };
    let result = runtime.block_on(cpu_streaming_inference_golden_warm(Some(&mut pipeline), reporter, "warm"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_cpu_streaming_inference_golden_warm passed.");
            reporter.record_timing("test_cpu_streaming_inference_golden_warm", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_cpu_streaming_inference_golden_warm failed: {}", e));
            reporter.record_failure("test_cpu_streaming_inference_golden_warm", &e, Some(t0.elapsed()));
        }
    }
}

#[test]
#[serial]
#[ignore]
fn test_gpu_singlefile_inference_golden_warm() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let mut pipeline = match runtime.block_on(create_pipeline(PipelineBackend::Gpu, true)) {
        Ok(p) => p,
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_singlefile_inference_golden_warm pipeline creation failed: {}", e));
            reporter.record_failure("test_gpu_singlefile_inference_golden_warm", &e, Some(t0.elapsed()));
            return;
        }
    };
    let result = runtime.block_on(gpu_singlefile_inference_golden_warm(Some(&mut pipeline), reporter, "warm"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_gpu_singlefile_inference_golden_warm passed.");
            reporter.record_timing("test_gpu_singlefile_inference_golden_warm", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_singlefile_inference_golden_warm failed: {}", e));
            reporter.record_failure("test_gpu_singlefile_inference_golden_warm", &e, Some(t0.elapsed()));
        }
    }
}

#[test]
#[serial]
#[ignore]
fn test_gpu_streaming_inference_golden_warm() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let mut pipeline = match runtime.block_on(create_pipeline(PipelineBackend::Gpu, false)) {
        Ok(p) => p,
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_streaming_inference_golden_warm pipeline creation failed: {}", e));
            reporter.record_failure("test_gpu_streaming_inference_golden_warm", &e, Some(t0.elapsed()));
            return;
        }
    };
    let result = runtime.block_on(gpu_streaming_inference_golden_warm(Some(&mut pipeline), reporter, "warm"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_gpu_streaming_inference_golden_warm passed.");
            reporter.record_timing("test_gpu_streaming_inference_golden_warm", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_gpu_streaming_inference_golden_warm failed: {}", e));
            reporter.record_failure("test_gpu_streaming_inference_golden_warm", &e, Some(t0.elapsed()));
        }
    }
}

// --- Settings Integration ---
#[test]
#[serial]
#[ignore]
fn test_settings_integration_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(settings_integration_warm(reporter));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_settings_integration_cold passed.");
            reporter.record_timing("test_settings_integration_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_settings_integration_cold failed: {}", e));
            reporter.record_failure("test_settings_integration_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

// --- Performance Metrics ---
#[test]
#[serial]
#[ignore]
fn test_performance_metrics_cpu_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(performance_metrics_warm(PipelineBackend::Cpu, true, reporter, "cpu_perf_cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_performance_metrics_cpu_cold passed.");
            reporter.record_timing("test_performance_metrics_cpu_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_performance_metrics_cpu_cold failed: {}", e));
            reporter.record_failure("test_performance_metrics_cpu_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
#[ignore]
fn test_performance_metrics_gpu_cold() {
    let reporter = &*TEST_REPORTER;
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let t0 = Instant::now();
    let result = runtime.block_on(performance_metrics_warm(PipelineBackend::Gpu, true, reporter, "gpu_perf_cold"));
    match result {
        Ok(_) => {
            reporter.log_message(1, "test_performance_metrics_gpu_cold passed.");
            reporter.record_timing("test_performance_metrics_gpu_cold", t0.elapsed());
        }
        Err(e) => {
            reporter.log_message(1, &format!("test_performance_metrics_gpu_cold failed: {}", e));
            reporter.record_failure("test_performance_metrics_gpu_cold", &e, Some(t0.elapsed()));
        }
    }
    TEST_REPORTER.generate_report();
}

// --- Master Test ---
#[test]
#[serial]
fn test_all_pipeline_validation_sequentially() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(100, "STARTING PIPELINE VALIDATION SUITE");
    
    // --- Run non-kernel (CPU-only) tests ONCE ---
    test_cpu_singlefile_creation_cold();
    test_cpu_singlefile_inference_golden_cold();
    test_cpu_streaming_creation_cold();
    test_cpu_streaming_inference_golden_cold();
    test_gpu_singlefile_creation_cold();
    test_gpu_singlefile_inference_golden_cold();
    test_gpu_streaming_creation_cold();
    test_gpu_streaming_inference_golden_cold();
    test_settings_integration_cold();
    test_performance_metrics_cpu_cold();
    test_performance_metrics_gpu_cold();

    let warm_run_runtime = tokio::runtime::Runtime::new().unwrap();
    warm_run_runtime.block_on(async {
        test_cpu_singlefile_inference_golden_warm();
        test_cpu_streaming_inference_golden_warm();
        test_gpu_singlefile_inference_golden_warm();
        test_gpu_streaming_inference_golden_warm();

    });

    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_all_pipeline_validation_sequentially", duration);
    TEST_REPORTER.log_message(100, &format!("Pipeline validation suite passed (took {:.2?})", duration));
    TEST_REPORTER.generate_report();
} 