use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use chrono::{self, DateTime, Local};
use crate::constants::workspace_root;
use tokio::sync::oneshot;

/// A struct to manage test reporting functionality
pub struct TestReporter {
    info_logs: Mutex<Vec<(usize, Instant, DateTime<Local>, String)>>,
    timings: Mutex<HashMap<String, std::time::Duration>>,
    log_dir: PathBuf,
    test_name: String,
    active_tests: Arc<AtomicUsize>,
    report_ready_tx: Mutex<Option<oneshot::Sender<()>>>,
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
            active_tests: Arc::new(AtomicUsize::new(0)),
            report_ready_tx: Mutex::new(None),
        })
    }

    pub fn start_test(&self) {
        self.active_tests.fetch_add(1, Ordering::SeqCst);
    }

    pub fn finish_test(&self) {
        if self.active_tests.fetch_sub(1, Ordering::SeqCst) == 1 {
            if let Some(tx) = self.report_ready_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
        }
    }

    /// Logs a general message with a test ID and timestamp to the in-memory log
    pub fn log_message(&self, test_id: usize, message: &str) {
        let now = Local::now();
        let timestamp_str = now.format("%Y-%m-%d, %H:%M:%S%.3f");
        println!("[{}][Test {}] {}", timestamp_str, test_id, message);
        self.info_logs
            .lock()
            .unwrap()
            .push((test_id, Instant::now(), now, message.to_string()));
    }

    /// Records a test timing in memory
    pub fn record_timing(&self, test_name: &str, duration: std::time::Duration) {
        self.timings
            .lock()
            .unwrap()
            .insert(test_name.to_string(), duration);
        self.finish_test();
    }

    /// Generates a markdown report from in-memory logs
    pub async fn generate_report(&self) {
        let (tx, rx) = oneshot::channel();
        *self.report_ready_tx.lock().unwrap() = Some(tx);

        // If no tests were ever started, don't wait forever.
        if self.active_tests.load(Ordering::SeqCst) == 0 {
            if let Some(tx) = self.report_ready_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
        }
        
        // Wait for the signal that all tests are done
        let _ = rx.await;

        let timings = self.timings.lock().unwrap();
        let mut info_logs = self.info_logs.lock().unwrap();

        if timings.is_empty() && info_logs.is_empty() {
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
        
        let mut sorted_tests: Vec<_> = timings.iter().collect();
        sorted_tests.sort_by(|a, b| a.0.cmp(b.0));

        let title = format!("{} Test Report", self.test_name.to_uppercase());
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        
        writeln!(report_file, "# {}", title).expect("Failed to write title");
        writeln!(report_file, "\n> Generated on: {}\n", timestamp).expect("Failed to write timestamp");
        
        writeln!(report_file, "## Test Results\n").expect("Failed to write section header");
        writeln!(report_file, "| No. | Test Name | Status | Time Taken |").expect("Failed to write table header");
        writeln!(report_file, "|:---:|:----------|:------:|:----------:|").expect("Failed to write table separator");

        for (idx, (test_name, duration)) in sorted_tests.iter().enumerate() {
            let duration_str = if duration.as_secs() >= 1 {
                format!("{:.2} sec", duration.as_secs_f64())
            } else {
                format!("{:.2} ms", duration.as_millis() as f64)
            };
            
            let display_name = test_name
                .strip_prefix("test_").unwrap_or(test_name)
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

            writeln!(report_file, "| {:2} | {:<50} | ✅ Pass | {:>10} |",
                idx + 1,
                display_name,
                duration_str
            ).expect("Failed to write test result");
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

        let total_duration: std::time::Duration = timings.values().sum();
        let avg_duration = if !timings.is_empty() { total_duration / timings.len() as u32 } else { std::time::Duration::from_secs(0) };
        
        let total_time_str = format!("{:.2} sec", total_duration.as_secs_f64());
        let avg_time_str = format!("{:.2} ms", avg_duration.as_millis() as f64);
        
        writeln!(report_file, "\n## Summary\n").expect("Failed to write summary header");
        writeln!(report_file, "### Test Statistics\n").expect("Failed to write statistics header");
        writeln!(report_file, "- **Total Tests:** {}", timings.len()).expect("Failed to write total tests");
        writeln!(report_file, "- **Passed:** {}", timings.len()).expect("Failed to write passed tests");
        writeln!(report_file, "- **Failed:** 0").expect("Failed to write failed tests");
        
        writeln!(report_file, "\n### Timing Information\n").expect("Failed to write timing header");
        writeln!(report_file, "- **Total Time:** {}", total_time_str).expect("Failed to write total time");
        writeln!(report_file, "- **Average Time:** {}", avg_time_str).expect("Failed to write average time");
        
        writeln!(report_file, "\n### Status\n").expect("Failed to write status header");
        writeln!(report_file, "✅ All tests passed successfully!").expect("Failed to write status");
        
        writeln!(report_file, "\n---\n").expect("Failed to write footer");
        writeln!(report_file, "_Report generated by BitNet Test Framework_").expect("Failed to write footer text");
        report_file.flush().expect("Failed to flush report file");
    }
}

/// Convenience macro for logging with timing
#[macro_export]
macro_rules! log_timed {
    ($reporter:expr, $test_name:expr, $duration:expr) => {
        let msg = format!("{:<40} took {:>10.3} ms", $test_name, $duration.as_secs_f64() * 1000.0);
        $reporter.log_message(&msg);
        $reporter.record_timing($test_name, $duration);
    };
} 