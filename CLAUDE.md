# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Frontend
```bash
npm run dev       # Dev server on http://0.0.0.0:5173
npm run build     # Type-check + production build
npm run lint      # ESLint
npm run preview   # Preview production build
```

### Backend
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Two servers must run simultaneously:
python audio.py          # WebSocket server — port 8000 (real-time AI transcription)
python -m uvicorn api:app --reload --port 5000  # REST API — port 5000
```

## Architecture

### Two-Backend Design
The app runs **two separate Python servers**:
- **Port 8000** (`audio.py`) — WebSocket server that streams microphone audio through the Basic Pitch AI model and emits `note_on` / `note_off` / `volume` / `silence_reset` events
- **Port 5000** (`api.py`) — FastAPI REST server for saving sessions, fetching notes, and triggering PDF export via LilyPond

The frontend must connect to both. The REST API base URL is constructed dynamically from `window.location.hostname` (assumes a reverse proxy routes `/api/*` to port 5000 in production). WebSocket connects directly to port 8000.

### Real-Time Recording Flow
1. `RecordButton` captures mic audio via Web Audio API
2. Float32 audio chunks stream to WebSocket (port 8000)
3. `audio.py` buffers chunks into a rolling window and runs Basic Pitch on each hop
4. Detected notes are emitted as WebSocket events
5. `scoreStore` (Zustand) receives events, tracks `activeNotes` as a `Map<midi, {startTime}>`, and on `note_off` quantizes raw duration → `NoteDuration` and appends to `notes[]`
6. `SheetMusic.tsx` re-renders the VexFlow canvas on every store update, grouping notes into 4-beat measures in batches of 4 measures per system

### State & Persistence
- **Zustand** (`scoreStore.tsx`) is the single source of truth during a session. Persists `notes` and `bpm` to localStorage (key: `maestro-backup`); `activeNotes` and `isMetronomeOn` are ephemeral.
- **FastAPI** holds the current session in a global in-memory variable — it resets on server restart.
- **Supabase** (`sheets` table) is the durable store. Rows have `user_id`, `title`, `bpm`, `notes` (JSONB), and timestamps. RLS ensures users only see their own sheets.

### Note Detection Nuances (audio.py)
- Two onset thresholds: `ONSET_THRESHOLD=0.6` for new notes, `RETRIGGER_ONSET_THRESHOLD=0.85` for re-triggering an already-active note (prevents false re-triggers on sustained notes)
- `RETRIGGER_COOLDOWN=0.12s` per note
- Overtone/ghost suppression: iterates HIGH→LOW probability notes and suppresses harmonics (octaves above active notes)

### Quantization (musicMath.ts)
Beat bucket thresholds (at the current BPM):
```
< 0.29 beats → '16'  sixteenth
< 0.62 beats → '8'   eighth
< 1.30 beats → 'q'   quarter
< 1.75 beats → 'qd'  dotted quarter
< 2.5  beats → 'h'   half
< 3.5  beats → 'hd'  dotted half
≥ 3.5  beats → 'w'   whole
```
Buckets are intentionally asymmetric to favor common values.

### Auth
Supabase Auth via `AuthContext`. Public routes: `/` and `/login`. All other routes are wrapped in `<ProtectedRoute>`. Supabase credentials are in `.env` as `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_DEFAULT_KEY`.

### PDF Export
`GET /api/export` → `api.py` reads the in-memory session → `lilypond.py` converts VexFlow duration codes to LilyPond syntax → compiles to PDF → returned as blob. Requires the `lilypond` binary on the backend system.

## Key Files

| File | Role |
|------|------|
| `backend/audio.py` | WebSocket + Basic Pitch AI, note detection logic |
| `backend/api.py` | FastAPI REST endpoints |
| `backend/lilypond.py` | VexFlow → LilyPond → PDF |
| `src/store/scoreStore.tsx` | All note state, quantization on note_off, save/load |
| `src/components/Canvas/SheetMusic.tsx` | VexFlow rendering, measure grouping |
| `src/utils/musicMath.ts` | `quantizeDuration` and `getDurationValue` |
| `src/utils/VexMap.ts` | `convertToVexNotes` — RenderedNote → StaveNote |
| `src/api/api.ts` | Axios client for the REST backend |
| `src/context/AuthContext.tsx` | Supabase auth provider |
| `src/pages/SheetsListPage.tsx` | Dashboard |
| `src/pages/SheetDetailPage.tsx` | Read-only sheet viewer |
