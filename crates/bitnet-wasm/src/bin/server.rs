#![cfg(not(target_arch = "wasm32"))]
//! BitNet WASM dev/demo server using warp (serves static files + WebSocket log bridge)

use std::convert::Infallible;
use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use warp::Filter;
use warp::ws::{Message, WebSocket};
use futures_util::{StreamExt, SinkExt};
use serde_json::Value;
use bitnet_tools::test_utils::TestReporter;
use serde::Deserialize;

#[derive(Deserialize)]
struct TestResult {
    test: String,
    status: String,
    time_ms: f64,
    error: Option<String>,
    logs: Vec<String>,
}

#[derive(Deserialize)]
struct TestReportMsg {
    #[serde(rename = "type")]
    msg_type: String,
    results: Vec<TestResult>,
    logs: Vec<String>,
}

fn handle_test_report(msg: &str) {
    let report: TestReportMsg = serde_json::from_str(msg).unwrap();
    // Use TestReporter for consistent reporting
    let reporter = match bitnet_tools::test_utils::TestReporter::new("browser_test") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[BrowserLog] Failed to create TestReporter: {}", e);
            return;
        }
    };
    for result in &report.results {
        // Log each test result
        let test_name = &result.test;
        let duration = std::time::Duration::from_millis(result.time_ms as u64);
        if result.status == "pass" {
            reporter.record_timing(test_name, duration);
        } else {
            let err = result.error.as_deref().unwrap_or("Test failed");
            reporter.record_failure(test_name, err, Some(duration));
        }
        // Log per-test logs if present
        for log in &result.logs {
            reporter.log_message_simple(log);
        }
    }
    // Log any global logs
    for log in &report.logs {
        reporter.log_message_simple(log);
    }
    reporter.generate_report();
    println!("[BrowserLog] Markdown report generated for browser tests (TestReporter). Report includes summary, table, and logs.");
}

#[tokio::main]
async fn main() {
    let port = 8080;
    let static_dir = PathBuf::from("static");
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("BitNet WASM dev server running at http://localhost:{}", port);
    println!("Serving files from: {:?}", static_dir);
    println!("WebSocket log bridge at ws://localhost:{}/ws", port);

    // Shared state for TestReporter (per connection)
    let reporter = Arc::new(Mutex::new(None));

    // WebSocket route
    let reporter_ws = reporter.clone();
    let ws_route = warp::path("ws")
        .and(warp::ws())
        .and_then(move |ws: warp::ws::Ws| {
            let reporter = reporter_ws.clone();
            async move {
                Ok::<_, Infallible>(ws.on_upgrade(move |socket| handle_ws(socket, reporter)))
            }
        });

    // Static file route (serves everything except /ws)
    let static_route = warp::any()
        .and(warp::path::full())
        .and_then(move |full_path: warp::path::FullPath| {
            let static_dir = static_dir.clone();
            async move {
                let rel_path = full_path.as_str().trim_start_matches('/');
                let mut file_path = static_dir.join(rel_path);
                if rel_path.is_empty() || !file_path.exists() || file_path.is_dir() {
                    file_path = static_dir.join("index.html");
                }
                match tokio::fs::read(&file_path).await {
                    Ok(contents) => {
                        let mime = mime_guess::from_path(&file_path).first_or_octet_stream();
                        Ok(warp::reply::with_header(contents, "Content-Type", mime.as_ref()))
                    }
                    Err(_) => {
                        // Fallback to index.html
                        let fallback = static_dir.join("index.html");
                        match tokio::fs::read(&fallback).await {
                            Ok(contents) => Ok(warp::reply::with_header(contents, "Content-Type", "text/html; charset=utf-8")),
                            Err(_) => Err(warp::reject::not_found()),
                        }
                    }
                }
            }
        });

    // Compose routes
    let routes = ws_route.or(static_route);

    warp::serve(routes).run(addr).await;
}

async fn handle_ws(ws: WebSocket, reporter: Arc<Mutex<Option<TestReporter>>>) {
    let (mut tx, mut rx) = ws.split();
    let mut test_name = String::from("browser_test");
    // Each connection gets its own TestReporter
    let mut local_reporter = TestReporter::new(&test_name).ok();
    println!("[WebSocket] New log client connected");
    while let Some(result) = rx.next().await {
        match result {
            Ok(msg) => {
                if msg.is_text() {
                    let txt = msg.to_str().unwrap_or("");
                    println!("[BrowserLog] {}", txt);
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(txt) {
                        // Handle new test_report message
                        if val.get("type").and_then(|t| t.as_str()) == Some("test_report") {
                            // Print summary to console
                            if let (Some(results), Some(logs)) = (val.get("results"), val.get("logs")) {
                                if let Some(arr) = results.as_array() {
                                    let test_names: Vec<_> = arr.iter().filter_map(|r| r.get("test")).filter_map(|t| t.as_str()).collect();
                                    println!("[BrowserLog] Received test_report: {} tests: {}", arr.len(), test_names.join(", "));
                                }
                            }
                            handle_test_report(txt);
                            println!("[BrowserLog] Markdown report generated for browser tests.");
                            continue;
                        }
                        // If this is a 'start' message, update test_name and reporter
                        if val.get("type").and_then(|t| t.as_str()) == Some("start") {
                            if let Some(tname) = val.get("test").and_then(|t| t.as_str()) {
                                test_name = tname.to_string();
                                local_reporter = TestReporter::new(&test_name).ok();
                            }
                        }
                        // Log messages
                        if let Some(msg) = val.get("message").and_then(|m| m.as_str()) {
                            if let Some(r) = local_reporter.as_mut() {
                                r.log_message_simple(msg);
                            }
                        }
                        // On 'done', generate report
                        if val.get("type").and_then(|t| t.as_str()) == Some("done") {
                            if let Some(r) = local_reporter.as_mut() {
                                r.generate_report();
                                println!("[BrowserLog] Markdown report generated for test '{}'.", test_name);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("[WebSocket] Error: {}", e);
                break;
            }
        }
    }
    println!("[WebSocket] Client disconnected");
}

fn mime_type_for_path(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript",
        Some("css") => "text/css",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
} 