import asyncio
import websockets
import json
import time
import numpy as np
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
HOP_SIZE = 768
WINDOW_LENGTH = 43844

# --- INFERENCE CADENCE ---
MODEL_INTERVAL_SEC = 0.25  # target cadence: 4 times per second

# --- HYSTERESIS THRESHOLDS ---
ONSET_THRESHOLD = 0.6
RETRIGGER_ONSET_THRESHOLD = 0.85
NOTE_START_THRESHOLD = 0.5
NOTE_KEEP_THRESHOLD = 0.25
MIN_VOLUME = 0.001

# --- COOLDOWN ---
RETRIGGER_COOLDOWN = 0.12

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

print("Loading Basic Pitch Model...")
model = Model(ICASSP_2022_MODEL_PATH)
print("Model Loaded. Ready.")

def midi_to_note_name(midi_number):
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    return f"{NOTE_NAMES[note_index]}{octave}"

def build_note_data(midi_num, start_time, now, session_start_time):
    rel_start = start_time - session_start_time
    dur = now - start_time
    return {
        "note": midi_to_note_name(midi_num),
        "midi": midi_num,
        "start_time": round(rel_start, 3),
        "duration": round(dur, 3),
    }

async def process_inference_frame(
    websocket,
    audio_snapshot,
    active_notes,
    recorded_song,
    session_start_time,
):
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(
        None,
        lambda: model.predict(audio_snapshot)
    )

    note_probs = output["note"]
    onset_probs = output["onset"]
    if note_probs is None:
        return session_start_time

    focus = 8
    current_notes_max = np.max(note_probs[0, -focus:, :], axis=0)
    current_onsets_max = np.max(onset_probs[0, -focus:, :], axis=0)

    # --- SUPPRESSION LOGIC (Iterate High -> Low) ---
    for i in range(87, 24, -1):
        prob = current_notes_max[i]
        if prob < 0.1:
            continue

        # CHECK 1: AM I AN OVERTONE?
        idx_below = i - 12
        if idx_below >= 0:
            prob_below = current_notes_max[idx_below]
            if prob_below > 0.5 and prob < prob_below:
                current_notes_max[i] = 0.0
                continue

        # CHECK 2: AM I CAUSING GHOSTS?
        if prob > 0.5:
            for offset in [12, 19]:
                low_idx = i - offset
                if low_idx >= 0:
                    prob_low = current_notes_max[low_idx]
                    if prob_low < (prob * 0.9):
                        current_notes_max[low_idx] = 0.0

    now = time.time()
    detected_this_frame = set()

    for i in range(88):
        midi_num = i + 21
        prob_note = current_notes_max[i]
        prob_onset = current_onsets_max[i]

        is_active = midi_num in active_notes
        thresh = NOTE_KEEP_THRESHOLD if is_active else NOTE_START_THRESHOLD
        is_sustaining = prob_note > thresh

        is_standard_attack = prob_onset > ONSET_THRESHOLD
        is_retrigger_attack = prob_onset > RETRIGGER_ONSET_THRESHOLD

        if is_sustaining:
            detected_this_frame.add(midi_num)

            if is_active:
                if is_retrigger_attack and (now - active_notes[midi_num]) > RETRIGGER_COOLDOWN:
                    old_start = active_notes[midi_num]
                    note_data = build_note_data(midi_num, old_start, now, session_start_time)

                    recorded_song.append(note_data)
                    await websocket.send(json.dumps({"type": "note_off", **note_data}))

                    active_notes[midi_num] = now
                    await websocket.send(json.dumps({
                        "type": "note_on",
                        "note": midi_to_note_name(midi_num),
                        "midi": midi_num,
                        "event": "re_trigger",
                        "start_time": round(now - session_start_time, 3)
                    }))
            else:
                if is_standard_attack:
                    if session_start_time is None:
                        session_start_time = now
                    active_notes[midi_num] = now
                    await websocket.send(json.dumps({
                        "type": "note_on",
                        "note": midi_to_note_name(midi_num),
                        "midi": midi_num,
                        "event": "new_attack",
                        "start_time": round(now - session_start_time, 3)
                    }))

    # --- CLEANUP ---
    if session_start_time is not None:
        for midi_num in list(active_notes.keys()):
            if midi_num not in detected_this_frame:
                start_time = active_notes[midi_num]
                note_info = build_note_data(midi_num, start_time, now, session_start_time)
                recorded_song.append(note_info)
                del active_notes[midi_num]
                await websocket.send(json.dumps({"type": "note_off", **note_info}))

    return session_start_time

async def inference_worker(websocket, state):
    try:
        while True:
            await state["run_event"].wait()
            state["run_event"].clear()

            # Only one inference in flight
            if state["inference_running"]:
                continue

            state["inference_running"] = True

            try:
                while True:
                    # Take the newest snapshot only
                    state["rerun_needed"] = False
                    audio_snapshot = state["audio_buffer"].copy()

                    state["session_start_time"] = await process_inference_frame(
                        websocket=websocket,
                        audio_snapshot=audio_snapshot,
                        active_notes=state["active_notes"],
                        recorded_song=state["recorded_song"],
                        session_start_time=state["session_start_time"],
                    )

                    # If fresh audio arrived while we were inferring, run once more immediately
                    if not state["rerun_needed"]:
                        break

            finally:
                state["inference_running"] = False

    except asyncio.CancelledError:
        raise

async def audio_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")

    state = {
        "audio_buffer": np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32),
        "input_accumulator": [],
        "active_notes": {},
        "recorded_song": [],
        "session_start_time": None,
        "last_inference_request": 0.0,
        "inference_running": False,
        "rerun_needed": False,
        "run_event": asyncio.Event(),
    }

    worker_task = asyncio.create_task(inference_worker(websocket, state))

    try:
        async for message in websocket:
            try:
                chunk = np.frombuffer(message, dtype=np.float32)
            except Exception:
                continue

            if len(chunk) == 0:
                continue

            state["input_accumulator"].extend(chunk.tolist())

            while len(state["input_accumulator"]) >= HOP_SIZE:
                new_data = np.array(state["input_accumulator"][:HOP_SIZE], dtype=np.float32)
                state["input_accumulator"] = state["input_accumulator"][HOP_SIZE:]

                state["audio_buffer"] = np.roll(state["audio_buffer"], -HOP_SIZE, axis=1)
                state["audio_buffer"][0, -HOP_SIZE:, 0] = new_data

                volume = float(np.sqrt(np.mean(new_data ** 2)))
                await websocket.send(json.dumps({"type": "volume", "value": volume}))

                # --- SILENCE HANDLING ---
                if volume < MIN_VOLUME:
                    if state["active_notes"]:
                        now = time.time()
                        if state["session_start_time"] is not None:
                            for midi_num, start in list(state["active_notes"].items()):
                                note_data = build_note_data(
                                    midi_num,
                                    start,
                                    now,
                                    state["session_start_time"]
                                )
                                state["recorded_song"].append(note_data)
                                await websocket.send(json.dumps({"type": "note_off", **note_data}))
                        state["active_notes"].clear()
                        await websocket.send(json.dumps({"type": "silence_reset"}))
                    continue

                now = time.time()

                # Ask for inference at most every MODEL_INTERVAL_SEC
                if (now - state["last_inference_request"]) >= MODEL_INTERVAL_SEC:
                    state["last_inference_request"] = now

                    if state["inference_running"]:
                        # Don't queue stale jobs, just remember that a fresh pass is needed
                        state["rerun_needed"] = True
                    else:
                        # No inference running; wake worker to process newest snapshot
                        state["run_event"].set()

    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed. Notes recorded: {len(state['recorded_song'])}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

async def main():
    print("Server running on localhost:8000")
    async with websockets.serve(audio_handler, "0.0.0.0", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
