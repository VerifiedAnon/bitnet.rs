//! Static file server for bitnet-wasm demo (dev only).
//! Now with WebSocket log bridge at /ws.
// Minimal, dependency-light: uses tiny_http for static, tokio-tungstenite for WebSocket

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::net::SocketAddr;
use std::thread;

// --- WebSocket dependencies ---
use tokio::runtime::Runtime;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::protocol::Message;
use futures_util::{StreamExt, SinkExt};
use bitnet_tools::test_utils::TestReporter;
use std::sync::{Arc, Mutex};
use serde_json::Value;

fn main() {
    let port = 8080;
    let static_dir = Path::new("static");
    let addr = format!("0.0.0.0:{}", port);
    println!("BitNet WASM dev server running at http://localhost:{}", port);
    println!("Serving files from: {:?}", static_dir);
    println!("WebSocket log bridge at ws://localhost:{}/ws", port);

    // Start WebSocket server in a background thread
    let ws_addr = addr.clone();
    thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            let listener = TcpListener::bind(&ws_addr).await.expect("Failed to bind WebSocket port");
            loop {
                let (stream, addr) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(e) => {
                        eprintln!("[WebSocket] Accept error: {}", e);
                        continue;
                    }
                };
                // Peek HTTP request to check if it's a WebSocket upgrade for /ws
                let mut buf = [0u8; 1024];
                let n = match stream.try_read(&mut buf) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let req_str = String::from_utf8_lossy(&buf[..n]);
                if req_str.contains("Upgrade: websocket") && req_str.contains("GET /ws ") {
                    // Spawn a task to handle this WebSocket connection
                    let ws_stream = stream;
                    tokio::spawn(handle_ws(ws_stream, addr));
                }
                // Otherwise, ignore (static files handled by tiny_http below)
            }
        });
    });

    // --- Static file server (unchanged) ---
    let server = tiny_http::Server::http(&addr).expect("Failed to start server");
    for request in server.incoming_requests() {
        let url_path = request.url().trim_start_matches('/');
        let file_path = if url_path.is_empty() {
            static_dir.join("index.html")
        } else {
            static_dir.join(url_path)
        };
        if file_path.exists() && file_path.is_file() {
            let mut file = File::open(&file_path).unwrap();
            let mut content = Vec::new();
            file.read_to_end(&mut content).unwrap();
            let mime = mime_type_for_path(&file_path);
            let response = tiny_http::Response::from_data(content)
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()).unwrap());
            let _ = request.respond(response);
        } else {
            // Serve index.html for any not-found file (SPA fallback)
            let mut file = File::open(static_dir.join("index.html")).unwrap();
            let mut content = Vec::new();
            file.read_to_end(&mut content).unwrap();
            let response = tiny_http::Response::from_data(content)
                .with_header(tiny_http::Header::from_bytes(&b"Content-Type"[..], b"text/html; charset=utf-8").unwrap());
            let _ = request.respond(response);
        }
    }
}

// --- WebSocket handler ---
async fn handle_ws<S>(stream: S, addr: SocketAddr)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("[WebSocket] Handshake error: {}", e);
            return;
        }
    };
    // TestReporter: Use test name from 'start' message if present, else default
    let reporter = Arc::new(Mutex::new(
        TestReporter::new("browser_test").expect("Failed to create TestReporter"),
    ));
    let (_, mut ws_receiver) = ws_stream.split();
    let mut test_name = String::from("browser_test");
    println!("[WebSocket] New log client: {}", addr);
    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(txt)) => {
                // Print raw log
                println!("[BrowserLog] {}", txt);
                // Parse JSON and log to TestReporter
                if let Ok(val) = serde_json::from_str::<Value>(&txt) {
                    // If this is a 'start' message, update test_name and reporter
                    if val.get("type").and_then(|t| t.as_str()) == Some("start") {
                        if let Some(tname) = val.get("test").and_then(|t| t.as_str()) {
                            test_name = tname.to_string();
                            // Replace reporter with new one for this test
                            let new_reporter = TestReporter::new(&test_name).expect("Failed to create TestReporter");
                            *reporter.lock().unwrap() = new_reporter;
                        }
                    }
                    // Log messages
                    if let Some(msg) = val.get("message").and_then(|m| m.as_str()) {
                        reporter.lock().unwrap().log_message_simple(msg);
                    }
                    // On 'done', generate report
                    if val.get("type").and_then(|t| t.as_str()) == Some("done") {
                        reporter.lock().unwrap().generate_report();
                        println!("[BrowserLog] Markdown report generated for test '{}'.", test_name);
                    }
                }
            }
            Ok(Message::Binary(_)) => {
                // Ignore binary
            }
            Ok(Message::Close(_)) => {
                println!("[WebSocket] Client {} disconnected", addr);
                break;
            }
            _ => {}
        }
    }
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