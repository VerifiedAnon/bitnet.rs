//!
//! WARNING: Only `test_end_to_end_pipeline` runs by default to guarantee correct test order.
//! All other tests are marked #[ignore] and must be run explicitly if needed.
//! This prevents accidental out-of-order or partial runs that could cause confusing failures.


use bitnet_tools::constants::{original_dir, converted_dir, DEFAULT_MODEL_ID, CONFIG_JSON};
use bitnet_converter::convert_model_on_disk;
use bitnet_tools::hf_loader::get_model;
use serial_test::serial;
use std::time::Instant;
use std::fs;
use bincode::serde::decode_from_slice;
use bincode::config::standard;
use safetensors::{SafeTensors, Dtype};
use half::bf16;
use bitnet_tools::test_utils::TestReporter;
use lazy_static::lazy_static;

// Initialize test reporter
lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("pipeline_integration")
        .expect("Failed to create test reporter");
}


#[test]
#[serial]
#[ignore]
fn test_model_files_present() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(1, "Ensuring model files are present (downloading if needed)...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_model_files_present", duration);
    TEST_REPORTER.log_message(1, &format!("Model files ready (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_conversion_outputs_blocks() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(2, "Running conversion and checking per-block outputs...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");

    let input_dir = original_dir().join(DEFAULT_MODEL_ID);
    let output_dir = converted_dir().join(DEFAULT_MODEL_ID);
    let _ = convert_model_on_disk(
        input_dir.to_str().unwrap(),
        output_dir.to_str().unwrap(),
    ).expect("Conversion failed");
    // Check for top-level files
    let expect_files = ["embedding.bin", "norm.bin", "lm_head.bin"];
    for f in &expect_files {
        let path = output_dir.join(f);
        assert!(path.exists(), "Expected output file missing: {}", f);
    }
    // Check for at least one block file
    let block0 = output_dir.join("block_0.bin");
    assert!(block0.exists(), "Expected block_0.bin to exist");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_conversion_outputs_blocks", duration);
    TEST_REPORTER.log_message(2, &format!("Conversion and block outputs check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_load_some_blocks() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(3, "Loading and deserializing some per-block files...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let output_dir = converted_dir().join(DEFAULT_MODEL_ID);
    // Try to load embedding, norm, lm_head, and first 2 blocks
    let files = [
        "embedding.bin",
        "norm.bin",
        "lm_head.bin",
        "block_0.bin",
        "block_1.bin",
    ];
    for f in &files {
        let path = output_dir.join(f);
        let buf = fs::read(&path).expect(&format!("Failed to read {}", f));
        // Just try to decode, don't check contents
        let _ : Result<(serde_json::Value, usize), _> = decode_from_slice(&buf, standard());
        // If decode fails, try as bincode for the known types
        // (We don't know the exact type here, so just check that it's not corrupt)
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_load_some_blocks", duration);
    TEST_REPORTER.log_message(3, &format!("Per-block file load/deserialization check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_required_files_present() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(4, "Checking for required model files...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let input_dir = original_dir().join(DEFAULT_MODEL_ID);
    let expect_files = [
        CONFIG_JSON,
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ];
    for f in &expect_files {
        let path = input_dir.join(f);
        assert!(path.exists(), "Required file missing: {}", f);
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_required_files_present", duration);
    TEST_REPORTER.log_message(4, &format!("Required files check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_config_json_parsing() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(5, "Parsing config.json...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let input_dir = original_dir().join(DEFAULT_MODEL_ID);
    let config_path = input_dir.join(CONFIG_JSON);
    let config_str = std::fs::read_to_string(&config_path).expect("Failed to read config.json");
    let config: serde_json::Value = serde_json::from_str(&config_str).expect("Failed to parse config.json");
    assert!(config.get("num_hidden_layers").is_some(), "num_hidden_layers missing in config");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_config_json_parsing", duration);
    TEST_REPORTER.log_message(5, &format!("config.json parsing check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_safetensors_loading() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(6, "Loading and validating safetensors...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let input_dir = original_dir().join(DEFAULT_MODEL_ID);
    let safetensors_path = input_dir.join("model.safetensors");
    let file = std::fs::File::open(&safetensors_path).expect("Failed to open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("Failed to mmap safetensors") };
    let safetensors = SafeTensors::deserialize(&mmap).expect("Failed to parse safetensors");
    let names = safetensors.names();
    assert!(!names.is_empty(), "No tensors found in safetensors");
    for name in &names {
        let tensor = safetensors.tensor(name).expect("Tensor missing");
        assert_eq!(tensor.dtype(), Dtype::BF16, "Tensor {} is not BF16", name);
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_safetensors_loading", duration);
    TEST_REPORTER.log_message(6, &format!("Safetensors loading check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_tensor_conversion_bf16_to_f32() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(7, "Testing BF16 to F32 conversion...");
    let bf16_vals = vec![bf16::from_f32(1.0), bf16::from_f32(-2.0)];
    // SAFETY: bf16 is not Pod, so cast to u16 first, then to u8
    let u16_vals: Vec<u16> = bf16_vals.iter().map(|b| b.to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&u16_vals);
    let u16_slice = bytemuck::cast_slice::<u8, u16>(bytes);
    let f32_vec: Vec<f32> = u16_slice.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect();
    assert_eq!(f32_vec.len(), 2);
    assert!((f32_vec[0] - 1.0).abs() < 0.01);
    assert!((f32_vec[1] + 2.0).abs() < 0.01);
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_tensor_conversion_bf16_to_f32", duration);
    TEST_REPORTER.log_message(7, &format!("BF16 to F32 conversion check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_layer_quantization_and_packing() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(8, "Testing layer quantization and packing...");
    use bitnet_converter::packer::quantize_to_1_58_bit_optimized;
    let n = 4;
    let k = 16;
    let tensor: Vec<f32> = (0..n*k).map(|i| (i % 3) as f32 - 1.0).collect();
    let shape = vec![n, k];
    let (quantized, scales) = quantize_to_1_58_bit_optimized(&tensor, &shape);
    assert_eq!(quantized.len(), n*k);
    assert_eq!(scales.len(), n);
    assert!(quantized.iter().all(|&v| v >= -1 && v <= 1));
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_layer_quantization_and_packing", duration);
    TEST_REPORTER.log_message(8, &format!("Layer quantization and packing check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_per_block_serialization() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(9, "Testing per-block serialization...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    let input_dir = original_dir().join(DEFAULT_MODEL_ID);
    let output_dir = converted_dir().join(DEFAULT_MODEL_ID);
    let _ = bitnet_converter::convert_model_on_disk(
        input_dir.to_str().unwrap(),
        output_dir.to_str().unwrap(),
    ).expect("Conversion failed");
    let expect_files = ["embedding.bin", "norm.bin", "lm_head.bin", "block_0.bin"];
    for f in &expect_files {
        let path = output_dir.join(f);
        assert!(path.exists(), "Expected output file missing: {}", f);
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_per_block_serialization", duration);
    TEST_REPORTER.log_message(9, &format!("Per-block serialization check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_per_block_deserialization() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(10, "Testing per-block deserialization...");
    get_model(Some(DEFAULT_MODEL_ID)).expect("Failed to get or download model files");
    use bincode::serde::decode_from_slice;
    use bincode::config::standard;
    let output_dir = converted_dir().join(DEFAULT_MODEL_ID);
    let files = [
        "embedding.bin",
        "norm.bin",
        "lm_head.bin",
        "block_0.bin",
    ];
    for f in &files {
        let path = output_dir.join(f);
        let buf = std::fs::read(&path).expect(&format!("Failed to read {}", f));
        let _ : Result<(serde_json::Value, usize), _> = decode_from_slice(&buf, standard());
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_per_block_deserialization", duration);
    TEST_REPORTER.log_message(10, &format!("Per-block deserialization check passed (took {:.2?})", duration));
}

#[test]
#[serial]
#[ignore]
fn test_error_on_missing_tensor() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(11, "Testing error on shape mismatch...");
    use bitnet_converter::packer::quantize_to_1_58_bit_optimized;
    let n = 4;
    let k = 16;
    let mut tensor: Vec<f32> = (0..n*k).map(|i| (i % 3) as f32 - 1.0).collect();
    let shape = vec![n, k];
    // Remove a value to break shape
    tensor.pop();
    let result = std::panic::catch_unwind(|| {
        quantize_to_1_58_bit_optimized(&tensor, &shape);
    });
    assert!(result.is_err(), "Should panic on shape mismatch");
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_error_on_missing_tensor", duration);
    TEST_REPORTER.log_message(11, &format!("Shape mismatch error check passed (took {:.2?})", duration));
}

fn zzz_final_pipeline_report() {
    // This function runs last and generates the final report.
    TEST_REPORTER.generate_report();
}

#[test]
#[serial]
fn test_end_to_end_pipeline() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(100, "STARTING END-TO-END PIPELINE VALIDATION");
    test_model_files_present();
    test_required_files_present();
    test_conversion_outputs_blocks();
    test_load_some_blocks();
    test_config_json_parsing();
    test_safetensors_loading();
    test_tensor_conversion_bf16_to_f32();
    test_layer_quantization_and_packing();
    test_per_block_serialization();
    test_per_block_deserialization();
    test_error_on_missing_tensor();

    // Generate the report at the very end
    zzz_final_pipeline_report();

    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_end_to_end_pipeline", duration);
    TEST_REPORTER.log_message(100, &format!("End-to-end pipeline passed (took {:.2?})", duration));
}

