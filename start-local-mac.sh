#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
PIP_EXE="$VENV_DIR/bin/pip"

echo "=== Music Transcriber — Local Test Mode (macOS) ==="
echo "Project: $PROJECT_DIR"

# --- Pre-flight: kill anything already on our ports ---
for port in 8000 5000 5173; do
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "[ports] Killing PID(s) $pids on port $port"
    kill -9 $pids 2>/dev/null || true
  fi
done

# --- 1. Homebrew + LilyPond (required for PDF export) ---
if ! command -v brew &>/dev/null; then
  echo "[error] Homebrew not found. Install from https://brew.sh then re-run."
  exit 1
fi

if ! command -v lilypond &>/dev/null; then
  echo "[setup] Installing LilyPond via Homebrew..."
  brew install lilypond
else
  echo "[setup] LilyPond OK: $(lilypond --version | head -1)"
fi

# --- 2. Require Python 3.11 or 3.10 (basic-pitch needs <=3.11) ---
PYTHON3X=""
for candidate in python3.11 python3.10; do
  if command -v "$candidate" &>/dev/null; then
    PYTHON3X="$candidate"
    break
  fi
done

if [ -z "$PYTHON3X" ]; then
  if command -v python3 &>/dev/null && python3 --version 2>&1 | grep -Eq "Python 3\.(10|11)\."; then
    PYTHON3X="python3"
  fi
fi

if [ -z "$PYTHON3X" ]; then
  echo "[error] Python 3.11 or 3.10 not found."
  echo "        Install with: brew install python@3.11"
  exit 1
fi

echo "[setup] Using $($PYTHON3X --version)"

# --- 3. Python venv ---
need_new_venv=true
if [ -x "$PYTHON_EXE" ]; then
  venv_ver=$("$PYTHON_EXE" --version 2>&1)
  if echo "$venv_ver" | grep -Eq "Python 3\.(10|11)\."; then
    need_new_venv=false
    echo "[setup] Venv OK ($venv_ver)"
  fi
fi

if $need_new_venv; then
  if [ -d "$VENV_DIR" ]; then
    echo "[setup] Existing venv is not Python 3.10/3.11 — recreating..."
    rm -rf "$VENV_DIR"
  else
    echo "[setup] Creating Python venv..."
  fi
  "$PYTHON3X" -m venv "$VENV_DIR"
fi

# --- 4. Python deps ---
echo "[setup] Installing Python dependencies..."
"$PIP_EXE" install --quiet 'setuptools<81' fastapi uvicorn websockets numpy basic-pitch

# --- 5. Frontend deps ---
echo "[setup] Syncing npm packages..."
cd "$PROJECT_DIR"
npm install
npm prune

# --- 6. Launch servers ---
echo ""
echo "[start] Launching servers..."

cd "$BACKEND_DIR"
"$PYTHON_EXE" audio.py &
WS_PID=$!
echo "[start] WebSocket server → port 8000 (PID $WS_PID)"

"$PYTHON_EXE" -m uvicorn api:app --host 0.0.0.0 --port 5000 &
API_PID=$!
echo "[start] REST API → port 5000 (PID $API_PID)"

cd "$PROJECT_DIR"
npm run dev &
VITE_PID=$!
echo "[start] Vite dev server → port 5173 (PID $VITE_PID)"

echo ""
echo "=== All services started ==="
echo "  Frontend:  http://localhost:5173/record"
echo "  REST API:  http://localhost:5000"
echo "  WebSocket: ws://localhost:8000"
echo ""
echo "Note: audio.py takes ~15s to load the AI model before recording works."
echo "Press Ctrl+C to stop all services."

# --- Cleanup on exit ---
cleanup() {
  echo ""
  echo "[stop] Shutting down..."
  kill $WS_PID $API_PID $VITE_PID 2>/dev/null || true
  wait $WS_PID $API_PID $VITE_PID 2>/dev/null || true
  echo "[stop] Done."
}
trap cleanup INT TERM

wait
