#!/usr/bin/env bash
# stop.sh - Stop the BitNet WASM dev server running on port 8080
PORT=8080

# Detect OS
OS="$(uname -s 2>/dev/null || echo Unknown)"

if [[ "$OS" == MINGW* || "$OS" == CYGWIN* || "$OS" == MSYS* || "$OS" == Windows* || "$OS" == Unknown ]]; then
    # Windows (Git Bash, MSYS, or PowerShell)
    echo "[stop.sh] Detected Windows. Searching for process on port $PORT..."
    PID=$(netstat -ano 2>/dev/null | grep :$PORT | grep LISTEN | awk '{print $5}' | head -n1)
    if [ -z "$PID" ]; then
        echo "[stop.sh] No process found listening on port $PORT."
        exit 0
    fi
    echo "[stop.sh] Killing process with PID $PID..."
    taskkill //PID $PID //F
    echo "[stop.sh] Done."
else
    # Unix/Linux/macOS
    echo "[stop.sh] Detected Unix-like OS. Searching for process on port $PORT..."
    PID=$(lsof -ti tcp:$PORT)
    if [ -z "$PID" ]; then
        echo "[stop.sh] No process found listening on port $PORT."
        exit 0
    fi
    echo "[stop.sh] Killing process with PID $PID..."
    kill -9 $PID
    echo "[stop.sh] Done."
fi 