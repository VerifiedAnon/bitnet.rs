use std::path::Path;
use std::process::Command;

const OFFICIAL_REPO_DIR: &str = "../../References/official";
const OFFICIAL_GPU_DIR: &str = "../../References/official/gpu";
const OFFICIAL_PRESET_KERNELS_DIR: &str = "../../References/official/preset_kernels";

#[test]
fn test_official_repo_presence() {
    if !Path::new(OFFICIAL_REPO_DIR).exists() {
        println!("Cloning official BitNet repo to {}...", OFFICIAL_REPO_DIR);
        let status = Command::new("git")
            .args(["clone", "https://github.com/microsoft/BitNet.git", OFFICIAL_REPO_DIR])
            .status()
            .expect("Failed to run git clone. Is git installed?");
        assert!(status.success(), "Failed to clone official BitNet repo");
    } else {
        println!("Official BitNet repo already present at {}.", OFFICIAL_REPO_DIR);
    }
    println!("Official BitNet repo is present and ready for use.");
    // Print and check GPU kernels path
    if Path::new(OFFICIAL_GPU_DIR).exists() {
        println!("GPU kernels path: {} (exists)", OFFICIAL_GPU_DIR);
    } else {
        println!("GPU kernels path: {} (NOT FOUND)", OFFICIAL_GPU_DIR);
    }
    // Print and check preset kernels path
    if Path::new(OFFICIAL_PRESET_KERNELS_DIR).exists() {
        println!("Preset kernels path: {} (exists)", OFFICIAL_PRESET_KERNELS_DIR);
    } else {
        println!("Preset kernels path: {} (NOT FOUND)", OFFICIAL_PRESET_KERNELS_DIR);
    }
} 
