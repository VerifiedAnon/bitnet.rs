use std::path::Path;
use std::process::Command;
use std::time::Instant;
use bitnet_tools::test_utils::TestReporter;
use lazy_static::lazy_static;
use serial_test::serial;

const OFFICIAL_REPO_DIR: &str = "../../References/official";
const OFFICIAL_GPU_DIR: &str = "../../References/official/gpu";
const OFFICIAL_PRESET_KERNELS_DIR: &str = "../../References/official/preset_kernels";

// Initialize test reporter
lazy_static! {
    static ref TEST_REPORTER: TestReporter = TestReporter::new("official_parity_test")
        .expect("Failed to create test reporter");
}

fn do_official_repo_presence_check() {
    let t0 = Instant::now();
    TEST_REPORTER.log_message(1, "Running test_official_repo_presence...");

    if !Path::new(OFFICIAL_REPO_DIR).exists() {
        TEST_REPORTER.log_message(1, &format!("Cloning official BitNet repo to {}...", OFFICIAL_REPO_DIR));
        let status = Command::new("git")
            .args(["clone", "https://github.com/microsoft/BitNet.git", OFFICIAL_REPO_DIR])
            .status()
            .expect("Failed to run git clone. Is git installed?");
        assert!(status.success(), "Failed to clone official BitNet repo");
    } else {
        TEST_REPORTER.log_message(1, &format!("Official BitNet repo already present at {}.", OFFICIAL_REPO_DIR));
    }
    TEST_REPORTER.log_message(1, "Official BitNet repo is present and ready for use.");
    // Print and check GPU kernels path
    if Path::new(OFFICIAL_GPU_DIR).exists() {
        TEST_REPORTER.log_message(1, &format!("GPU kernels path: {} (exists)", OFFICIAL_GPU_DIR));
    } else {
        TEST_REPORTER.log_message(1, &format!("GPU kernels path: {} (NOT FOUND)", OFFICIAL_GPU_DIR));
    }
    // Print and check preset kernels path
    if Path::new(OFFICIAL_PRESET_KERNELS_DIR).exists() {
        TEST_REPORTER.log_message(1, &format!("Preset kernels path: {} (exists)", OFFICIAL_PRESET_KERNELS_DIR));
    } else {
        TEST_REPORTER.log_message(1, &format!("Preset kernels path: {} (NOT FOUND)", OFFICIAL_PRESET_KERNELS_DIR));
    }
    let duration = t0.elapsed();
    TEST_REPORTER.record_timing("test_official_repo_presence", duration);
}

#[test]
#[serial]
fn test_parity_suite() {
    do_official_repo_presence_check();
    TEST_REPORTER.generate_report();
} 
