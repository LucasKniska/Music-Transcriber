#!/usr/bin/env python3
import asyncio
import json
import sys
import time
import uuid
from typing import Dict, List, Optional, Set

import numpy as np
import websockets
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050
HOP_SIZE         = 2048
WINDOW_LENGTH    = 43844
NOTE_THRESHOLD   = 0.4
ONSET_THRESHOLD  = 0.5
MIN_VOLUME       = 0.001
RETRIGGER_GAP_MS = 100
FOCUS_FRAMES     = 5
MIDI_OFFSET      = 21

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── ANSI helpers ──────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
G   = "\033[92m"
Y   = "\033[93m"
C   = "\033[96m"
M   = "\033[95m"
RE  = "\033[91m"

# ── Music helpers ─────────────────────────────────────────────────────────────
def midi_to_note(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"

_PATTERNS = [
    (frozenset({0, 4, 7}),       "maj"),
    (frozenset({0, 3, 7}),       "min"),
    (frozenset({0, 3, 6}),       "dim"),
    (frozenset({0, 4, 8}),       "aug"),
    (frozenset({0, 7}),          "5"),
    (frozenset({0, 5}),          "5"),
    (frozenset({0, 4}),          "maj?"),
    (frozenset({0, 8}),          "maj?"),
    (frozenset({0, 3}),          "min?"),
    (frozenset({0, 9}),          "min?"),
    (frozenset({0, 5, 7}),       "sus4"),
    (frozenset({0, 2, 7}),       "sus2"),
    (frozenset({0, 4, 7, 11}),   "maj7"),
    (frozenset({0, 4, 7, 10}),   "7"),
    (frozenset({0, 3, 7, 10}),   "min7"),
    (frozenset({0, 3, 7, 11}),   "minmaj7"),
    (frozenset({0, 3, 6, 10}),   "m7b5"),
    (frozenset({0, 3, 6, 9}),    "dim7"),
    (frozenset({0, 4, 8, 10}),   "aug7"),
    (frozenset({0, 4, 7, 2}),    "add9"),
    (frozenset({0, 3, 7, 2}),    "madd9"),
    (frozenset({0, 4, 7, 9}),    "6"),
    (frozenset({0, 3, 7, 9}),    "min6"),
    (frozenset({0, 4, 10, 2}),   "9"),
    (frozenset({0, 3, 10, 2}),   "min9"),
    (frozenset({0, 4, 11, 2}),   "maj9"),
]
_TRIAD_SUBSETS = [
    (frozenset({0, 4, 7}), "maj"),
    (frozenset({0, 3, 7}), "min"),
    (frozenset({0, 3, 6}), "dim"),
    (frozenset({0, 4, 8}), "aug"),
    (frozenset({0, 5, 7}), "sus4"),
    (frozenset({0, 2, 7}), "sus2"),
]

def chord_label(midis: List[int]) -> Optional[str]:
    if len(midis) < 2:
        return None
    pcs = frozenset(m % 12 for m in midis)
    if len(pcs) < 2:
        return None
    best = None
    for root in sorted(pcs):
        ivs = frozenset((p - root) % 12 for p in pcs)
        for pat, quality in _PATTERNS:
            if ivs == pat:
                return f"{NOTE_NAMES[root]}{quality}"
        if best is None:
            for pat, quality in _TRIAD_SUBSETS:
                if pat.issubset(ivs):
                    best = f"{NOTE_NAMES[root]}{quality}(ext)"
    return best


# ── MessageBus ────────────────────────────────────────────────────────────────
class MessageBus:
    def __init__(self, session_id: str, start_unix: float, websocket):
        self.session_id = session_id
        self.start_unix = start_unix
        self.ws = websocket

    def _ms(self) -> int:
        return int((time.time() - self.start_unix) * 1000)

    async def emit(self, msg: dict) -> None:
        self._console_print(msg)
        await self.ws.send(json.dumps(msg))

    async def session_start(self) -> None:
        await self.emit({"type": "session_start", "session_id": self.session_id, "time_ms": 0})

    async def note_on(self, midi: int, velocity: float, active_midis: List[int]) -> None:
        notes = sorted(active_midis)
        label = chord_label(notes)
        is_ch = len(notes) > 1
        await self.emit({
            "type":        "note_on",
            "session_id":  self.session_id,
            "time_ms":     self._ms(),
            "midi":        midi,
            "note":        midi_to_note(midi),
            "velocity":    round(float(velocity), 3),
            "is_chord":    is_ch,
            "chord_notes": [midi_to_note(m) for m in notes],
            "chord_midis": notes,
            "chord_label": label,
        })

    async def note_off(self, midi: int, onset_time: float, was_chord: bool, chord_midis: List[int]) -> None:
        duration_ms = int((time.time() - onset_time) * 1000)
        label = chord_label(chord_midis) if was_chord else None
        await self.emit({
            "type":        "note_off",
            "session_id":  self.session_id,
            "time_ms":     self._ms(),
            "midi":        midi,
            "note":        midi_to_note(midi),
            "duration_ms": duration_ms,
            "is_chord":    was_chord,
            "chord_label": label,
        })

    async def retrigger(self, midi: int, velocity: float, gap_ms: int, active_midis: List[int]) -> None:
        notes = sorted(active_midis)
        label = chord_label(notes)
        await self.emit({
            "type":        "retrigger",
            "session_id":  self.session_id,
            "time_ms":     self._ms(),
            "midi":        midi,
            "note":        midi_to_note(midi),
            "velocity":    round(float(velocity), 3),
            "is_chord":    len(notes) > 1,
            "chord_notes": [midi_to_note(m) for m in notes],
            "chord_midis": notes,
            "chord_label": label,
            "gap_ms":      gap_ms,
        })

    async def silence(self, midis_ended: List[int]) -> None:
        await self.emit({
            "type":        "silence",
            "session_id":  self.session_id,
            "time_ms":     self._ms(),
            "notes_ended": [midi_to_note(m) for m in midis_ended],
            "midis_ended": sorted(midis_ended),
        })

    async def session_end(self, total_notes: int, total_chords: int) -> None:
        await self.emit({
            "type":         "session_end",
            "session_id":   self.session_id,
            "time_ms":      self._ms(),
            "total_notes":  total_notes,
            "total_chords": total_chords,
        })

    def _console_print(self, msg: dict) -> None:
        t     = msg.get("time_ms", 0)
        ts    = f"{DIM}[{t/1000:06.2f}s]{R}"
        mtype = msg["type"]

        if mtype == "session_start":
            print(f"\n{M}{B}{'═'*60}{R}")
            print(f"{M}{B}  ♪  Live Transcription  —  session {msg['session_id'][:8]}{R}")
            print(f"{M}{B}{'═'*60}{R}\n")

        elif mtype == "note_on":
            label_str = f"  {M}({msg['chord_label']}){R}" if msg["chord_label"] else ""
            if msg["is_chord"]:
                notes_str = " + ".join(f"{C}{B}{n}{R}" for n in msg["chord_notes"])
                print(f"  {ts} {G}{B}▶ ON {R}{Y}{B}CHORD{R} {notes_str}{label_str}"
                      f"  vel={msg['velocity']:.2f}")
            else:
                print(f"  {ts} {G}{B}▶ ON {R}{C}{B}{msg['note']:<5}{R}"
                      f"  midi={msg['midi']}"
                      f"  vel={msg['velocity']:.2f}")

        elif mtype == "note_off":
            label_str = f"  {DIM}[{msg['chord_label']}]{R}" if msg["chord_label"] else ""
            print(f"  {ts} {RE}{B}■ OFF{R} {C}{B}{msg['note']:<5}{R}{label_str}"
                  f"  dur={msg['duration_ms']}ms")

        elif mtype == "retrigger":
            label_str = f"  {M}({msg['chord_label']}){R}" if msg["chord_label"] else ""
            print(f"  {ts} {Y}{B}↺ RE  {R}{C}{B}{msg['note']:<5}{R}{label_str}"
                  f"  gap={msg['gap_ms']}ms  vel={msg['velocity']:.2f}")

        elif mtype == "silence":
            notes = ", ".join(msg["notes_ended"])
            print(f"  {ts} {DIM}~ silence  ({notes} ended){R}")

        elif mtype == "session_end":
            print(f"\n{DIM}{'─'*60}")
            print(f"  Session ended  |  notes={msg['total_notes']}"
                  f"  chords={msg['total_chords']}{R}\n")

        sys.stdout.flush()


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Basic Pitch Model...")
model = Model(ICASSP_2022_MODEL_PATH)
print("Model Loaded. Ready.")


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def audio_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    start_unix = time.time()
    bus = MessageBus(session_id, start_unix, websocket)

    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    input_accumulator = []
    active_notes: Dict[int, dict] = {}
    total_notes  = 0
    total_chords = 0

    await bus.session_start()

    try:
        async for message in websocket:
            try:
                chunk = np.frombuffer(message, dtype=np.float32)
            except Exception:
                continue
            if len(chunk) == 0:
                continue

            input_accumulator.extend(chunk)
            if len(input_accumulator) < HOP_SIZE:
                continue

            new_data = np.array(input_accumulator[:HOP_SIZE], dtype=np.float32)
            input_accumulator = input_accumulator[HOP_SIZE:]

            audio_buffer = np.roll(audio_buffer, -HOP_SIZE, axis=1)
            audio_buffer[0, -HOP_SIZE:, 0] = new_data

            volume = float(np.sqrt(np.mean(new_data ** 2)))
            await websocket.send(json.dumps({"type": "volume", "value": volume}))

            # ── Silence handling ──────────────────────────────────────────
            if volume < MIN_VOLUME:
                if active_notes:
                    await bus.silence(list(active_notes.keys()))
                    active_notes.clear()
                continue

            # ── AI processing ─────────────────────────────────────────────
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, lambda: model.predict(audio_buffer))

            note_probs  = output.get("note")
            onset_probs = output.get("onset")
            if note_probs is None or onset_probs is None:
                continue

            note_now  = np.max(note_probs[0,  -FOCUS_FRAMES:, :], axis=0)
            onset_now = np.max(onset_probs[0, -FOCUS_FRAMES:, :], axis=0)

            detected_this_frame: Set[int] = set()
            for i in range(88):
                midi = i + MIDI_OFFSET
                if note_now[i] > NOTE_THRESHOLD:
                    detected_this_frame.add(midi)

            active_list = sorted(detected_this_frame)

            # ── Note detection ────────────────────────────────────────────
            for i in range(88):
                midi      = i + MIDI_OFFSET
                is_on     = note_now[i]  > NOTE_THRESHOLD
                is_attack = onset_now[i] > ONSET_THRESHOLD

                if not is_on:
                    continue

                velocity = float(note_now[i])

                if midi in active_notes:
                    if is_attack:
                        gap_ms = int((time.time() - active_notes[midi]["onset_time"]) * 1000)
                        if gap_ms >= RETRIGGER_GAP_MS:
                            active_notes[midi]["onset_time"]  = time.time()
                            active_notes[midi]["chord_midis"] = active_list
                            await bus.retrigger(midi, velocity, gap_ms, active_list)
                else:
                    active_notes[midi] = {"onset_time": time.time(), "chord_midis": active_list}
                    await bus.note_on(midi, velocity, active_list)
                    total_notes += 1
                    if len(active_list) > 1:
                        total_chords += 1

            # ── Cleanup ended notes ───────────────────────────────────────
            for midi in list(active_notes.keys()):
                if midi not in detected_this_frame:
                    info      = active_notes.pop(midi)
                    was_chord = len(info["chord_midis"]) > 1
                    await bus.note_off(midi, info["onset_time"], was_chord, info["chord_midis"])

    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed. Notes: {total_notes}, Chords: {total_chords}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            await bus.session_end(total_notes, total_chords)
        except Exception:
            pass


async def main():
    print("Server running on localhost:8000")
    async with websockets.serve(audio_handler, "0.0.0.0", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
