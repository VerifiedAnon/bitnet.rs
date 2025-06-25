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
                    if let Ok(val) = serde_json::from_str::<Value>(txt) {
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