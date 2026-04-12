# Music Transcriber

A full-stack application that listens to your microphone in real-time and transcribes what you play into sheet music using AI. The frontend renders notation with VexFlow and the backend runs the Basic Pitch neural network model over a WebSocket stream.

## Tech Stack

- **Frontend:** React, TypeScript, VexFlow, Zustand, Vite
- **Backend:** Python, WebSockets (`audio.py`), FastAPI (`api.py`)
- **AI Model:** [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify) — ONNX, CPU inference

---

## Setup

### Backend

Two servers must run simultaneously.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate
pip install -r requirements.txt

# Terminal 1 — WebSocket AI server (port 8000)
python audio.py

# Terminal 2 — REST API (port 5000)
python -m uvicorn api:app --reload --port 5000
```

### Frontend

```bash
npm install
npm run dev   # http://localhost:5173
```

---

## How It Works

1. Click **Record** — the browser captures microphone audio via the Web Audio API
2. Float32 audio chunks stream to the WebSocket server on port 8000
3. `audio.py` buffers chunks into a rolling window and runs Basic Pitch on each hop (768 samples ≈ 35 ms)
4. Detected notes are emitted as `note_on` / `note_off` / `volume` / `silence_reset` events
5. The Zustand store receives events, quantizes note durations, and appends them to the note list
6. VexFlow re-renders the sheet music canvas on every update, grouping notes into 4-beat measures

## Note Detection

- Onset threshold: `0.6` for new notes, `0.85` to re-trigger a sustained note
- Retrigger cooldown: `0.12s` per note
- Overtone/ghost suppression: iterates HIGH→LOW and zeroes out harmonics (octaves + fifths above active notes)

## Quantization

Beat bucket thresholds (relative to current BPM):

| Duration | Symbol |
|----------|--------|
| < 0.29 beats | 16th |
| < 0.62 beats | 8th |
| < 1.30 beats | quarter |
| < 1.75 beats | dotted quarter |
| < 2.5 beats | half |
| < 3.5 beats | dotted half |
| ≥ 3.5 beats | whole |

## PDF Export

`GET /api/export` → `api.py` reads the current session → `lilypond.py` converts VexFlow duration codes to LilyPond syntax → compiles to PDF. Requires the `lilypond` binary installed on the backend system.
