use std::sync::Arc;
use bitnet_tools::test_utils::TestReporter;
use bitnet_core::pipeline::{Pipeline, PipelineOptions};

#[tokio::test]
async fn test_full_pipeline() {
    let reporter = Arc::new(TestReporter::new("pipeline_validation").unwrap());
    let progress_cb = {
        let reporter = reporter.clone();
        Arc::new(move |step: usize, msg: &str| reporter.log_message(step, msg))
    };
    let options = PipelineOptions {
        model_id: None,
        input_dir: None,
        output_dir: None,
        reporter: Some(progress_cb),
    };
    let pipeline = Pipeline::new(options).await.expect("Failed to create pipeline");
    pipeline.ensure_model_ready().await.expect("Failed to prepare model");
    let result = pipeline.run_inference("Hello").await.expect("Inference failed");
    assert!(result.logits.len() > 0, "Logits vector is empty");
    assert!(result.logits.iter().all(|&l| l.is_finite()), "Logits contain NaN or Infinity");
    println!("Top token: {} (ID: {}, Logit: {:.4})", result.top_token, result.top_token_id, result.top_logit);
    reporter.generate_report();
} 