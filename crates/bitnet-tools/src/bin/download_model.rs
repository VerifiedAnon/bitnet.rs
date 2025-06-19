// CLI tool to download all BitNet model files using the hf_loader module.
// Usage: cargo run -p bitnet-tools --bin download_model

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bitnet_tools::hf_loader::get_model;
    match get_model(None) {
        Ok(model_files) => {
            println!("\nDownloaded model files to: {:?}", model_files.model_dir);
            Ok(())
        },
        Err(e) => {
            eprintln!("\nError: {}", e);
            Err(e)
        }
    }
} 