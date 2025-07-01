use std::path::{Path, PathBuf};
use hf_hub::{api::sync::Api, Repo, RepoType};
use crate::constants::{models_root, CONFIG_JSON, SAFETENSORS_EXT, REQUIRED_MODEL_FILES, TOKENIZER_MODEL, SAFETENSORS_FILE};
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{self, Read};
use reqwest::blocking::Client;
use reqwest::header::CONTENT_LENGTH;

pub use crate::constants::DEFAULT_MODEL_ID;

/// Returns the default model download directory (relative to project root).
pub fn default_model_dir() -> PathBuf {
    models_root().join(crate::constants::DEFAULT_MODEL_ID)
}

/// Returns the directory for original (downloaded) model files.
pub fn original_model_dir(base_dir: &Path, model_id: &str) -> PathBuf {
    base_dir.join("Original").join(model_id)
}

/// Returns the directory for converted model files.
pub fn converted_model_dir(base_dir: &Path, model_id: &str) -> PathBuf {
    base_dir.join("Converted").join(model_id)
}

/// Struct holding paths to key model files.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub safetensors_files: Vec<PathBuf>,
    // Add more fields as needed
}

/// Download model weights and tokenizer files from Hugging Face, with progress reporting.
pub fn download_model_and_tokenizer_with_progress(
    model_id: &str,
    revision: Option<&str>,
    destination: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, rev.to_string())),
        None => api.repo(Repo::new(model_id.to_string(), RepoType::Model)),
    };

    // List of files to try to download (except tokenizer.model)
    let files: Vec<&str> = REQUIRED_MODEL_FILES
        .iter()
        .filter(|f| **f != TOKENIZER_MODEL)
        .copied()
        .collect();

    let dest = destination.as_ref();
    std::fs::create_dir_all(dest)?;

    // Get repo info once to get all file sizes
    let repo_info = repo.info()?;
    let mut file_sizes: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut total_size: u64 = 0;
    let client = Client::new();
    let base_repo_url = format!("https://huggingface.co/{}/resolve/main/", model_id);
    for sibling in &repo_info.siblings {
        let name = &sibling.rfilename;
        // Only use HEAD request to get Content-Length for size
        let mut size = 0u64;
        let url = format!("{}{}", base_repo_url, name);
        if let Ok(resp) = client.head(&url).send() {
            if let Some(len) = resp.headers().get(CONTENT_LENGTH) {
                if let Ok(len) = len.to_str().unwrap_or("").parse::<u64>() {
                    size = len;
                }
            }
        }
        file_sizes.insert(name.clone(), size);
        // Checksum validation not possible here: sibling.sha256 not available from API
        if files.contains(&name.as_str()) {
            total_size += size;
        }
    }
    let overall_pb = ProgressBar::new(total_size);
    overall_pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})").unwrap());

    for file in &files {
        let out_path = dest.join(file);
        if out_path.exists() {
            println!("Already present: {}", file);
            // Checksum validation skipped: expected hash not available
            continue;
        }
        print!("Downloading: {}... ", file);
        io::stdout().flush().ok();
        match repo.get(file) {
            Ok(path) => {
                let mut src = File::open(&path)?;
                let mut dst = File::create(&out_path)?;
                let size = file_sizes.get(&file.to_string()).copied().unwrap_or(0);
                let pb = if size > 0 {
                    let pb = ProgressBar::new(size);
                    pb.set_style(ProgressStyle::with_template("{bar:40.cyan/blue} {bytes}/{total_bytes} ({percent}%)").unwrap());
                    Some(pb)
                } else {
                    None
                };
                let mut buf = [0u8; 8192];
                loop {
                    let n = src.read(&mut buf)?;
                    if n == 0 { break; }
                    dst.write_all(&buf[..n])?;
                    overall_pb.inc(n as u64);
                    if let Some(ref pb) = pb {
                        pb.inc(n as u64);
                    }
                }
                if let Some(pb) = pb {
                    pb.finish_and_clear();
                }
                // Checksum validation skipped: expected hash not available
                println!("done");
            },
            Err(_e) => {
                println!("not found in repo");
            }
        }
    }
    overall_pb.finish_and_clear();
    Ok(())
}

/// Find all .safetensors files in a directory.
pub fn find_safetensors_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flat_map(|it| it)
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().map_or(false, |ext| ext == &SAFETENSORS_EXT[1..]))
        .collect()
}

/// Attempts to get (and if needed, download) a model, using a custom base directory.
/// If `model_id` is None, uses the default model.
/// Returns paths to the needed files (config.json, all .safetensors files).
///
/// - Creates <base_dir>/Original/<model_id>/ if not present
/// - Downloads files if not present
/// - Validates by file presence (not checksum)
/// - Returns ModelFiles struct with key paths
pub fn get_model_with_base(base_dir: &Path, model_id: Option<&str>) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let id = model_id.unwrap_or(DEFAULT_MODEL_ID);
    let dir = original_model_dir(base_dir, id);
    std::fs::create_dir_all(&dir)?;

    // Download if any required file is missing (except tokenizer.model)
    let missing_files: Vec<&str> = REQUIRED_MODEL_FILES
        .iter()
        .filter(|file| **file != TOKENIZER_MODEL && !dir.join(file).exists())
        .map(|f| *f)
        .collect();

    if !missing_files.is_empty() {
        download_model_and_tokenizer_with_progress(id, None, &dir)?;
        // Re-check missing files after download
        let _still_missing: Vec<&str> = REQUIRED_MODEL_FILES
            .iter()
            .filter(|file| **file != TOKENIZER_MODEL && !dir.join(file).exists())
            .map(|f| *f)
            .collect();
    }

    // Special handling for tokenizer.model
    if !dir.join(TOKENIZER_MODEL).exists() {
        let url = "https://github.com/microsoft/BitNet/raw/main/gpu/tokenizer.model";
        let out_path = dir.join(TOKENIZER_MODEL);
        println!("Downloading: tokenizer.model from BitNet GitHub...");
        let resp = ureq::get(url).call();
        if let Some(resp) = resp.ok() {
            let mut file = std::fs::File::create(&out_path)?;
            std::io::copy(&mut resp.into_reader(), &mut file)?;
            println!("Downloaded tokenizer.model");
        } else {
            eprintln!("WARNING: tokenizer.model missing. BitNet uses the Llama tokenizer. Please download tokenizer.model from https://github.com/microsoft/BitNet/blob/main/gpu/tokenizer.model and place it in the model directory.");
        }
    }

    // Final check for required files (except safetensors logic)
    let config_path = dir.join(CONFIG_JSON);
    if !config_path.exists() {
        return Err(format!("Missing {} for model {}", CONFIG_JSON, id).into());
    }
    // Safetensors logic: accept either single or all shards
    let single = dir.join(SAFETENSORS_FILE);
    let shard1 = dir.join("model-00001-of-00002.safetensors");
    let shard2 = dir.join("model-00002-of-00002.safetensors");
    let has_single = single.exists();
    let has_shards = shard1.exists() && shard2.exists();
    if !has_single && !has_shards {
        return Err(format!("Missing model.safetensors or all sharded safetensors files for model {}", id).into());
    }
    // Find all safetensors files (may be sharded)
    let safetensors_files = find_safetensors_files(&dir);
    Ok(ModelFiles {
        model_dir: dir,
        config: config_path,
        safetensors_files,
    })
}

/// Production wrapper: uses MODELS_ROOT as base dir.
pub fn get_model(model_id: Option<&str>) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    get_model_with_base(models_root().as_path(), model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use crate::constants::{models_root, DEFAULT_MODEL_ID, CONFIG_JSON, REQUIRED_MODEL_FILES};

    #[test]
    fn test_default_model_dir_helpers() {
        let model_id = DEFAULT_MODEL_ID;
        let base_dir = models_root();
        let expected_default = base_dir.join(model_id);
        let expected_original = base_dir.join("Original").join(model_id);
        let expected_converted = base_dir.join("Converted").join(model_id);

        assert_eq!(default_model_dir(), expected_default);
        assert_eq!(original_model_dir(base_dir.as_path(), model_id), expected_original);
        assert_eq!(converted_model_dir(base_dir.as_path(), model_id), expected_converted);
    }

    #[test]
    fn test_find_safetensors_files() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir = tmp_dir.path();
        let st1 = dir.join("foo.safetensors");
        let st2 = dir.join("bar.safetensors");
        fs::write(&st1, b"test").unwrap();
        fs::write(&st2, b"test").unwrap();
        let found = find_safetensors_files(dir);
        assert!(found.contains(&st1));
        assert!(found.contains(&st2));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_get_model_mock() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let base_dir = tmp_dir.path();
        let model_id = "test-model";
        let orig_dir = base_dir.join("Original").join(model_id);
        std::fs::create_dir_all(&orig_dir).unwrap();
        // Create all required files with dummy content
        for file in REQUIRED_MODEL_FILES.iter() {
            let path = orig_dir.join(file);
            std::fs::write(&path, b"test").unwrap();
        }
        let config_path = orig_dir.join(CONFIG_JSON);
        let safetensors_path = orig_dir.join("foo.safetensors");
        std::fs::write(&safetensors_path, b"test").unwrap();
        assert!(config_path.exists(), "Config file does not exist");
        // At least one .safetensors file must exist
        assert!(safetensors_path.exists(), ".safetensors file does not exist");
        let result = get_model_with_base(base_dir, Some(model_id));
        assert!(result.is_ok(), "get_model_with_base failed: {:?}", result.err());
        let files = result.unwrap();
        assert_eq!(files.model_dir, orig_dir);
        assert_eq!(files.config, config_path);
        // Should find the extra .safetensors file too
        assert!(files.safetensors_files.contains(&safetensors_path));
    }
} 