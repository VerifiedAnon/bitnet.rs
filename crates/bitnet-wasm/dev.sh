#!/usr/bin/env bash
# dev.sh - One-command build, serve, and open for BitNet WASM
set -e

WASM_OUT_DIR="./static/pkg"
WASM_OUT_NAME="bitnet_wasm"
SERVER_PORT=8080

# --- HELP COMMAND ---
if [[ "$1" == "help" || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: ./dev.sh [stop|help]"
    echo
    echo "  (no args)   Build, serve, and open BitNet WASM dev server."
    echo "  stop        Stop the dev server running on port $SERVER_PORT."
    echo "  help        Show this help message."
    exit 0
fi

# --- STOP COMMAND ---
if [[ "$1" == "stop" ]]; then
    # Detect OS
    OS="$(uname -s 2>/dev/null || echo Unknown)"
    if [[ "$OS" == MINGW* || "$OS" == CYGWIN* || "$OS" == MSYS* || "$OS" == Windows* || "$OS" == Unknown ]]; then
        echo "[dev.sh] Detected Windows. Searching for process on port $SERVER_PORT..."
        PID=$(netstat -ano 2>/dev/null | grep :$SERVER_PORT | grep LISTEN | awk '{print $5}' | head -n1)
        if [ -z "$PID" ]; then
            echo "[dev.sh] No process found listening on port $SERVER_PORT."
            exit 0
        fi
        echo "[dev.sh] Killing process with PID $PID..."
        taskkill //PID $PID //F
        echo "[dev.sh] Done."
    else
        echo "[dev.sh] Detected Unix-like OS. Searching for process on port $SERVER_PORT..."
        PID=$(lsof -ti tcp:$SERVER_PORT)
        if [ -z "$PID" ]; then
            echo "[dev.sh] No process found listening on port $SERVER_PORT."
            exit 0
        fi
        echo "[dev.sh] Killing process with PID $PID..."
        kill -9 $PID
        echo "[dev.sh] Done."
    fi
    exit 0
fi

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
    cargo run --bin server --features dev-server &
    SERVER_PID=$!
    # Poll for server readiness
    printf '[dev.sh] Waiting for server to be ready'
    for i in {1..30}; do
        if command -v curl > /dev/null; then
            if curl --output /dev/null --silent --head --fail "http://localhost:$SERVER_PORT"; then
                printf '\n[dev.sh] Server is up!\n'
                break
            fi
        else
            # Fallback: just sleep a bit
            sleep 1
            break
        fi
        printf '.'
        sleep 1
    done
else
    printf '\n[dev.sh] Server already running on port %s.\n' "$SERVER_PORT"
    echo "[dev.sh] To stop it,  use: make wasm-stop"
    SERVER_PID=""
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
if [ -n "$SERVER_PID" ]; then
    printf '\n[dev.sh] Server running in background (PID %s). To stop: kill %s\n' "$SERVER_PID" "$SERVER_PID"
fi

# 5. Wait for server process to exit (optional: comment out if you want script to exit immediately)
# wait $SERVER_PID 