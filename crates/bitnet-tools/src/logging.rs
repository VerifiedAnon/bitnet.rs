//! Unified logging utility for BitNet projects.
//!
//! Provides a reusable logging initialization function for all binaries and libraries.
//! Log format includes timestamp, level, file, line, and function/module.
//! Safe to call multiple times (only initializes once).
//!
//! # Usage
//!
//! ```rust
//! bitnet_tools::logging::init_logging(Some("info"), false, None);
//! ```
//!
//! - `level`: Some("debug"), Some("info"), etc. If None, uses RUST_LOG or defaults to "info".
//! - `verbose`: If true, overrides level to "debug".
//! - `log_file`: Optionally, path to a log file for file logging (if Some).

use std::sync::Once;
use std::io::Write;
use env_logger;

/// Initialize logging for the application.
///
/// - `level`: Some("debug"), Some("info"), etc. If None, uses RUST_LOG or defaults to "info".
/// - `verbose`: If true, overrides level to "debug".
/// - `log_file`: Optionally, path to a log file for file logging (if Some).
///
/// Safe to call multiple times (only initializes once).
pub fn init_logging(level: Option<&str>, verbose: bool, log_file: Option<&str>) {
    static INIT: Once = Once::new();
    let level = if verbose {
        "debug"
    } else {
        level.unwrap_or("info")
    };
    INIT.call_once(|| {
        if let Some(path) = log_file {
            // File + terminal logging using env_logger and a custom writer
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("Failed to open log file");
            let file = std::sync::Mutex::new(file);
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
                .format(move |buf, record| {
                    let ts = buf.timestamp_millis();
                    let file_line = match (record.file(), record.line()) {
                        (Some(f), Some(l)) => format!("{}:{}", f, l),
                        _ => "?".to_string(),
                    };
                    let func = record.module_path().unwrap_or("");
                    let msg = format!("[{}][{}][{}][{}] {}\n", ts, record.level(), file_line, func, record.args());
                    // Write to file
                    let mut file = file.lock().unwrap();
                    let _ = file.write_all(msg.as_bytes());
                    // Write to terminal
                    write!(buf, "{}", msg)
                })
                .is_test(false)
                .try_init()
                .ok();
        } else {
            // Terminal only
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
                .format(|buf, record| {
                    let ts = buf.timestamp_millis();
                    let file_line = match (record.file(), record.line()) {
                        (Some(f), Some(l)) => format!("{}:{}", f, l),
                        _ => "?".to_string(),
                    };
                    let func = record.module_path().unwrap_or("");
                    writeln!(buf, "[{}][{}][{}][{}] {}", ts, record.level(), file_line, func, record.args())
                })
                .is_test(false)
                .try_init()
                .ok();
        }
    });
} 