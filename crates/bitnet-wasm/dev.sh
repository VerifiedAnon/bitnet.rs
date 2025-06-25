#!/usr/bin/env bash
# dev.sh - One-command build, serve, and open for BitNet WASM
set -e

WASM_OUT_DIR="./static/pkg"
WASM_OUT_NAME="bitnet_wasm"
SERVER_PORT=8080

# Check for cargo
if ! command -v cargo > /dev/null; then
    echo "[dev.sh] WARNING: cargo not found in PATH. Rust toolchain is required."
fi

# Check for wasm-pack
if ! command -v wasm-pack > /dev/null; then
    echo
    echo "[dev.sh] ERROR: wasm-pack not found in PATH."
    echo "[dev.sh] To install wasm-pack, copy and run one of the following commands in your terminal:" 
    echo
    echo "    cargo install wasm-pack"
    echo "    # or, if you have npm:"
    echo "    npm install -g wasm-pack"
    echo
    exit 127
fi

# Clean WASM output directory to prevent stale/cached files
if [ -d "$WASM_OUT_DIR" ]; then
    rm -rf "$WASM_OUT_DIR"
fi

# 1. Build WASM
printf '\n[dev.sh] Building WASM package...\n'
wasm-pack build --target web --out-dir "$WASM_OUT_DIR" --out-name "$WASM_OUT_NAME"

# Check if the server is already running on the desired port (cross-platform)
PORT_IN_USE=false
if command -v netstat > /dev/null; then
    if netstat -ano | grep LISTEN | grep ":$SERVER_PORT" > /dev/null; then
        PORT_IN_USE=true
    fi
elif command -v lsof > /dev/null; then
    if lsof -i :$SERVER_PORT | grep LISTEN > /dev/null; then
        PORT_IN_USE=true
    fi
elif command -v ss > /dev/null; then
    if ss -ltnp | grep ":$SERVER_PORT" > /dev/null; then
        PORT_IN_USE=true
    fi
fi

if [ "$PORT_IN_USE" = false ]; then
    # 2. Start the Rust server in the background
    printf '\n[dev.sh] Starting Rust static file server on port %s...\n' "$SERVER_PORT"
    cargo run --bin server &
    SERVER_PID=$!
    sleep 1
else
    printf '\n[dev.sh] Server already running on port %s.\n' "$SERVER_PORT"
    echo "[dev.sh] To stop it,  use: make wasm-stop"
fi

# 3. Open the browser (cross-platform, prefer Chrome on Windows)
URL="http://localhost:$SERVER_PORT"
printf '\n[dev.sh] Opening browser at %s...\n' "$URL"
CHROME_PATH1="/c/Program Files/Google/Chrome/Application/chrome.exe"
CHROME_PATH2="/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
CHROME_PATH3="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
CHROME_PATH4="/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
if command -v google-chrome > /dev/null; then
    google-chrome "$URL"
elif command -v chrome > /dev/null; then
    chrome "$URL"
elif [ -f "$CHROME_PATH1" ]; then
    "$CHROME_PATH1" "$URL"
elif [ -f "$CHROME_PATH2" ]; then
    "$CHROME_PATH2" "$URL"
elif [ -f "$CHROME_PATH3" ]; then
    "$CHROME_PATH3" "$URL"
elif [ -f "$CHROME_PATH4" ]; then
    "$CHROME_PATH4" "$URL"
elif command -v xdg-open > /dev/null; then
    xdg-open "$URL"
elif command -v open > /dev/null; then
    open "$URL"
elif command -v start > /dev/null; then
    start "$URL"
else
    echo "[dev.sh] Could not detect a command to open the browser. Please open $URL manually."
fi

# 4. Print instructions for stopping the server
printf '\n[dev.sh] Server running in background (PID %s). To stop: kill %s\n' "$SERVER_PID" "$SERVER_PID"

# 5. Wait for server process to exit (optional: comment out if you want script to exit immediately)
# wait $SERVER_PID 