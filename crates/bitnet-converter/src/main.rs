// File: E:\Desktop\Bitnet rs\crates\bitnet-converter\src\main.rs

//! CLI entry point for the BitNet weight conversion tool (Burn-free, pure Rust).

use bitnet_converter::convert_model_on_disk;
use clap::Parser;
use bitnet_tools::constants::{models_root, ORIGINAL_DIR, CONVERTED_DIR};
use simplelog::*;
use std::fs::File;
use chrono::Local;

#[derive(Parser, Debug)]
#[command(author, version, about = "A tool to convert BitNet models to the custom packed format.")]
struct Args {
    /// Path to the directory containing the source model files (e.g., model.safetensors, config.json).
    #[arg(short, long)]
    input_dir: Option<String>,

    /// Path to the directory where the converted model files will be saved.
    #[arg(short, long)]
    output_dir: Option<String>,
}

fn main() {
    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs").ok();

    // Create a timestamped log file with .txt extension
    let log_file = format!("logs/bitnet-converter-{}.txt", Local::now().format("%Y%m%d-%H%M%S"));
    CombinedLogger::init(vec![
        TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
        WriteLogger::new(LevelFilter::Debug, Config::default(), File::create(&log_file).unwrap()),
    ]).unwrap();

    log::info!("BitNet Converter started. Log file: {}", log_file);

    let args = Args::parse();
    let default_model_subdir = "microsoft/bitnet-b1.58-2B-4T-bf16";
    let input_dir = args.input_dir.unwrap_or_else(|| {
        models_root().join(ORIGINAL_DIR).join(default_model_subdir).to_string_lossy().to_string()
    });
    let output_dir = args.output_dir.unwrap_or_else(|| {
        models_root().join(CONVERTED_DIR).join(default_model_subdir).to_string_lossy().to_string()
    });
    match convert_model_on_disk(&input_dir, &output_dir, false) {
        Ok(output_file) => {
            log::info!("\n✅ Conversion successful!");
            log::info!("Converted model saved in: {}", output_file.display());
        }
        Err(e) => {
            log::error!("Error during conversion: {}", e);
            eprintln!("\n❌ Error during conversion: {}", e);
            std::process::exit(1);
        }
    }
}