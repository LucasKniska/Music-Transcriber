#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

echo "=== Music Transcriber — Local Test Mode ==="
echo "Project: $PROJECT_DIR"

# --- 1. LilyPond (required for PDF export) ---
if ! command -v lilypond &>/dev/null; then
  echo "[setup] Installing LilyPond..."
  sudo apt-get install -y lilypond
else
  echo "[setup] LilyPond OK: $(lilypond --version | head -1)"
fi

# --- 2. Python venv ---
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating Python venv..."
  python3 -m venv "$VENV_DIR"
else
  echo "[setup] Using existing venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- 3. Python deps ---
echo "[setup] Installing Python dependencies..."
pip install --quiet fastapi uvicorn websockets numpy basic-pitch

# --- 4. Frontend deps ---
if [ ! -d "$PROJECT_DIR/node_modules" ]; then
  echo "[setup] Installing npm packages..."
  cd "$PROJECT_DIR" && npm install
else
  echo "[setup] node_modules already present"
fi

# --- 5. Start WebSocket backend (port 8000) ---
echo "[start] WebSocket server → port 8000"
cd "$BACKEND_DIR"
"$VENV_DIR/bin/python" audio.py &
WS_PID=$!

# --- 6. Start REST backend (port 5000) ---
echo "[start] REST API → port 5000"
"$VENV_DIR/bin/python" -m uvicorn api:app --host 0.0.0.0 --port 5000 &
API_PID=$!

# --- 7. Start Vite frontend (port 5173) ---
echo "[start] Vite dev server → port 5173"
cd "$PROJECT_DIR"
npm run dev &
VITE_PID=$!

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
