use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use chrono::{self, DateTime, Local};
use crate::constants::workspace_root;

/// Add a struct to track test results
#[derive(Clone)]
struct TestResult {
    name: String,
    status: TestStatus,
    duration: Option<std::time::Duration>,
    error: Option<String>,
}

#[derive(Clone, PartialEq)]
enum TestStatus {
    Pass,
    Fail,
}

#[derive(Clone)]
pub struct SpecialFinding {
    pub test_name: String,
    pub label: String,
    pub description: String,
}

/// A struct to manage test reporting functionality
pub struct TestReporter {
    info_logs: Mutex<Vec<(usize, Instant, DateTime<Local>, String)>>,
    timings: Mutex<HashMap<String, std::time::Duration>>,
    log_dir: PathBuf,
    test_name: String,
    // Store all test results
    results: Mutex<Vec<TestResult>>,
    special_finding: Mutex<Option<SpecialFinding>>,
}

impl TestReporter {
    /// Creates a new TestReporter that stores logs in memory
    pub fn new(log_name: &str) -> io::Result<Self> {
        let log_dir = workspace_root().join("logs");
        fs::create_dir_all(&log_dir)?;

        Ok(Self {
            info_logs: Mutex::new(Vec::new()),
            timings: Mutex::new(HashMap::new()),
            log_dir,
            test_name: log_name.to_string(),
            results: Mutex::new(Vec::new()),
            special_finding: Mutex::new(None),
        })
    }

    /// Logs a general message with a test ID and timestamp to the in-memory log
    /// Also automatically detects and records test failures from log messages
    pub fn log_message(&self, test_id: usize, message: &str) {
        let now = Local::now();
        let timestamp_str = now.format("%Y-%m-%d, %H:%M:%S%.3f");
        println!("[{}][Test {}] {}", timestamp_str, test_id, message);
        
        // Auto-detect failures from log messages
        if message.starts_with("[FAIL]") {
            self.extract_and_record_failure(message);
        }
        
        self.info_logs
            .lock()
            .unwrap()
            .push((test_id, Instant::now(), now, message.to_string()));
    }

    /// Enhanced log_message that accepts message without test_id (for backward compatibility)
    pub fn log_message_simple(&self, message: &str) {
        self.log_message(0, message);
    }

    /// Extract failure information from log message and record it
    fn extract_and_record_failure(&self, message: &str) {
        // Parse failure message format: "[FAIL] test_name: error_message"
        if let Some(content) = message.strip_prefix("[FAIL] ") {
            if let Some(colon_pos) = content.find(':') {
                let test_name = content[..colon_pos].trim();
                let error_message = content[colon_pos + 1..].trim();
                
                // Check if we already recorded this failure to avoid duplicates
                let mut results = self.results.lock().unwrap();
                if !results.iter().any(|r| r.name == test_name && r.status == TestStatus::Fail) {
                    results.push(TestResult {
                        name: test_name.to_string(),
                        status: TestStatus::Fail,
                        duration: None, // Could be enhanced to extract timing if available
                        error: Some(error_message.to_string()),
                    });
                }
            }
        }
    }

    /// Records a test timing in memory (for passed tests)
    pub fn record_timing(&self, test_name: &str, duration: std::time::Duration) {
        self.timings
            .lock()
            .unwrap()
            .insert(test_name.to_string(), duration);
        
        // Record as a pass in results
        let mut results = self.results.lock().unwrap();
        // Remove any existing entry for this test (in case of retries)
        results.retain(|r| r.name != test_name);
        results.push(TestResult {
            name: test_name.to_string(),
            status: TestStatus::Pass,
            duration: Some(duration),
            error: None,
        });
    }

    /// Records a failed test result explicitly
    pub fn record_failure(&self, test_name: &str, error: &str, duration: Option<std::time::Duration>) {
        let mut results = self.results.lock().unwrap();
        // Remove any existing entry for this test (in case of retries)
        results.retain(|r| r.name != test_name);
        results.push(TestResult {
            name: test_name.to_string(),
            status: TestStatus::Fail,
            duration,
            error: Some(error.to_string()),
        });
    }

    /// Records a test status explicitly (can be used for both pass and fail)
    pub fn record_test_result(&self, test_name: &str, passed: bool, duration: Option<std::time::Duration>, error: Option<&str>) {
        if passed {
            if let Some(d) = duration {
                self.record_timing(test_name, d);
            } else {
                let mut results = self.results.lock().unwrap();
                results.retain(|r| r.name != test_name);
                results.push(TestResult {
                    name: test_name.to_string(),
                    status: TestStatus::Pass,
                    duration: None,
                    error: None,
                });
            }
        } else {
            self.record_failure(test_name, error.unwrap_or("Test failed"), duration);
        }
    }

    /// Record a special finding (e.g., workaround/fix) to be highlighted in the report
    pub fn record_special_finding(&self, test_name: &str, label: &str, description: &str) {
        let mut sf = self.special_finding.lock().unwrap();
        *sf = Some(SpecialFinding {
            test_name: test_name.to_string(),
            label: label.to_string(),
            description: description.to_string(),
        });
    }

    /// Generates a markdown report from in-memory logs
    pub fn generate_report(&self) {
        let timings = self.timings.lock().unwrap();
        let mut info_logs = self.info_logs.lock().unwrap();
        let results = self.results.lock().unwrap();
        let special_finding = self.special_finding.lock().unwrap();

        if results.is_empty() && timings.is_empty() && info_logs.is_empty() {
            println!("No test data found, skipping report generation.");
            return;
        }
        
        // Sort logs by test ID, then by timestamp to ensure sequential order
        info_logs.sort_by_key(|k| (k.0, k.1));

        let report_path = self.log_dir.join(format!("{}.md", self.test_name));
        let mut report_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(report_path)
            .expect("Failed to create report file");
        
        // Sort results by name for stable output
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| a.name.cmp(&b.name));

        let title = format!("{} Test Report", self.test_name.to_uppercase());
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        
        writeln!(report_file, "# {}", title).expect("Failed to write title");
        writeln!(report_file, "\n> Generated on: {}\n", timestamp).expect("Failed to write timestamp");
        
        writeln!(report_file, "## Test Results\n").expect("Failed to write section header");
        writeln!(report_file, "| No. | Test Name | Status | Time Taken | Error Message |").expect("Failed to write table header");
        writeln!(report_file, "|:---:|:----------|:------:|:----------:|:-------------|").expect("Failed to write table separator");

        // Only keep the latest result for each unique test name
        let mut latest_results: HashMap<String, TestResult> = HashMap::new();
        for result in sorted_results.into_iter() {
            latest_results.insert(result.name.clone(), result);
        }
        let mut final_results: Vec<_> = latest_results.into_values().collect();
        final_results.sort_by(|a, b| a.name.cmp(&b.name));

        let mut pass_count = 0;
        let mut fail_count = 0;
        let mut shown = 0;
        for result in final_results.iter() {
            let duration_str = match result.duration {
                Some(d) if d.as_secs() >= 1 => format!("{:.2} sec", d.as_secs_f64()),
                Some(d) => format!("{:.2} ms", d.as_millis() as f64),
                None => "-".to_string(),
            };
            let display_name = result.name
                .strip_prefix("test_").unwrap_or(&result.name)
                .replace("_", " ")
                .split_whitespace()
                .map(|word| {
                    if word.eq_ignore_ascii_case("gpu") || word.eq_ignore_ascii_case("cpu") {
                        word.to_uppercase()
                    } else {
                        word[0..1].to_uppercase() + &word[1..]
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            match result.status {
                TestStatus::Pass => {
                    pass_count += 1;
                    shown += 1;
                    writeln!(report_file, "| {:2} | {:<50} | ✅ Pass | {:>10} |             |",
                        shown,
                        display_name,
                        duration_str
                    ).expect("Failed to write test result");
                }
                TestStatus::Fail => {
                    fail_count += 1;
                    shown += 1;
                    let error_message = result.error.as_deref().unwrap_or("Unknown error");
                    // Truncate long error messages for table readability
                    let truncated_error = if error_message.len() > 50 {
                        format!("{}...", &error_message[..47])
                    } else {
                        error_message.to_string()
                    };
                    writeln!(report_file, "| {:2} | {:<50} | ❌ Fail | {:>10} | {} |",
                        shown,
                        display_name,
                        duration_str,
                        truncated_error
                    ).expect("Failed to write test result");
                }
            }
        }

        if let Some(sf) = &*special_finding {
            writeln!(report_file, "\n## ⭐ Special Finding\n").expect("Failed to write special finding header");
            writeln!(report_file, "**[{}]**: `{}`  ", sf.label, sf.test_name).expect("Failed to write special finding label");
            writeln!(report_file, "{}\n", sf.description).expect("Failed to write special finding description");
        }

        if !info_logs.is_empty() {
            writeln!(report_file, "\n<details>\n<summary>📝 View Full Log Dump</summary>\n").expect("Failed to write log dump details tag");
            writeln!(report_file, "```").expect("Failed to start log block");
            for (_, _, timestamp, log) in info_logs.iter() {
                let formatted_log = format!("[{}] -> {}", timestamp.format("%Y-%m-%d, %H:%M:%S%.3f"), log);
                writeln!(report_file, "{}", formatted_log).expect("Failed to write log line");
            }
            writeln!(report_file, "```").expect("Failed to end log block");
            writeln!(report_file, "\n</details>\n").expect("Failed to write log dump details closing tag");
        }

        let total_count = pass_count + fail_count;
        let total_duration: std::time::Duration = final_results.iter().filter_map(|r| r.duration).sum();
        let avg_duration = if total_count > 0 { total_duration / total_count as u32 } else { std::time::Duration::from_secs(0) };
        let total_time_str = format!("{:.2} sec", total_duration.as_secs_f64());
        let avg_time_str = format!("{:.2} ms", avg_duration.as_millis() as f64);
        
        writeln!(report_file, "\n## Summary\n").expect("Failed to write summary header");
        writeln!(report_file, "### Test Statistics\n").expect("Failed to write statistics header");
        writeln!(report_file, "- **Total Tests:** {}", total_count).expect("Failed to write total tests");
        writeln!(report_file, "- **Passed:** {}", pass_count).expect("Failed to write passed tests");
        writeln!(report_file, "- **Failed:** {}", fail_count).expect("Failed to write failed tests");
        
        if total_count > 0 {
            writeln!(report_file, "\n### Timing Information\n").expect("Failed to write timing header");
            writeln!(report_file, "- **Total Time:** {}", total_time_str).expect("Failed to write total time");
            writeln!(report_file, "- **Average Time:** {}", avg_time_str).expect("Failed to write average time");
        }
        
        writeln!(report_file, "\n### Status\n").expect("Failed to write status header");
        if fail_count == 0 {
            writeln!(report_file, "✅ All tests passed successfully!").expect("Failed to write status");
        } else {
            writeln!(report_file, "❌ {} test(s) failed. See above for details.", fail_count).expect("Failed to write status");
        }
        
        writeln!(report_file, "\n---\n").expect("Failed to write footer");
        writeln!(report_file, "_Report generated by BitNet Test Framework_").expect("Failed to write footer text");
        report_file.flush().expect("Failed to flush report file");
        
        println!("Report generated with {} passed and {} failed tests", pass_count, fail_count);
    }
}

/// Convenience macro for logging with timing
#[macro_export]
macro_rules! log_timed {
    ($reporter:expr, $test_name:expr, $duration:expr) => {
        let msg = format!("{:<40} took {:>10.3} ms", $test_name, $duration.as_secs_f64() * 1000.0);
        $reporter.log_message_simple(&msg);
        $reporter.record_timing($test_name, $duration);
    };
}

/// Convenience macro for logging generic messages
#[macro_export]
macro_rules! log_message {
    ($reporter:expr, $msg:expr) => {
        $reporter.log_message_simple($msg);
    };
}

/// Convenience macro for logging test failures
#[macro_export]
macro_rules! log_failure {
    ($reporter:expr, $test_name:expr, $error:expr) => {
        let msg = format!("[FAIL] {}: {}", $test_name, $error);
        $reporter.log_message_simple(&msg);
        $reporter.record_failure($test_name, $error, None);
    };
    ($reporter:expr, $test_name:expr, $error:expr, $duration:expr) => {
        let msg = format!("[FAIL] {}: {}", $test_name, $error);
        $reporter.log_message_simple(&msg);
        $reporter.record_failure($test_name, $error, Some($duration));
    };
}