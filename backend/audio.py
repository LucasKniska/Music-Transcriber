import asyncio
import logging
import traceback
import websockets
import json
import time
import numpy as np

# basic_pitch checks for optional backends (TF, ONNX, CoreML) at import time and
# warns if they're absent. Suppress noise, then try ONNX → fallback to default.
_root_level = logging.getLogger().level
logging.getLogger().setLevel(logging.ERROR)
try:
    from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH, InferenceType
    model = Model(ICASSP_2022_MODEL_PATH, inference_type=InferenceType.ONNX)
    logging.getLogger().setLevel(_root_level)
    print("Using ONNX inference backend.")
except (ImportError, AttributeError, Exception):
    from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH
    logging.getLogger().setLevel(_root_level)
    model = Model(ICASSP_2022_MODEL_PATH)
    print("Using default inference backend.")

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
HOP_SIZE = 768
WINDOW_LENGTH = 43844  # ~2s context window required by the compiled TFLite model

# --- HYSTERESIS THRESHOLDS ---
ONSET_THRESHOLD = 0.6          # Sensitivity for starting a NEW note from silence
RETRIGGER_ONSET_THRESHOLD = 0.85 # Higher sensitivity required to re-trigger an EXISTING note
NOTE_START_THRESHOLD = 0.5
NOTE_KEEP_THRESHOLD = 0.25
MIN_VOLUME = 0.001

# --- COOLDOWN ---
RETRIGGER_COOLDOWN = 0.12

# --- SILENCE GRACE PERIOD ---
SILENCE_GRACE_FRAMES = 3  # consecutive silent frames before cutting notes (~105ms)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

print("Model Loaded. Ready.")

def midi_to_note_name(midi_number):
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    return f"{NOTE_NAMES[note_index]}{octave}"

async def audio_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    
    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    input_accumulator = []
    active_notes = {}
    silence_grace_count = 0

    session_start_time = None
    recorded_song = []

    frame_count = 0
    try:
        async for message in websocket:
            try:
                chunk = np.frombuffer(message, dtype=np.float32)
            except Exception:
                continue
            if len(chunk) == 0: continue

            frame_count += 1
            if frame_count == 1:
                print(f"[audio] First frame received, chunk size={len(chunk)}")
            elif frame_count % 200 == 0:
                print(f"[audio] Frames processed: {frame_count}, accumulator={len(input_accumulator)}")

            input_accumulator.extend(chunk)
            if len(input_accumulator) < HOP_SIZE: continue

            try:
                new_data = np.array(input_accumulator[:HOP_SIZE], dtype=np.float32)
                input_accumulator = input_accumulator[HOP_SIZE:]

                audio_buffer = np.roll(audio_buffer, -HOP_SIZE, axis=1)
                audio_buffer[0, -HOP_SIZE:, 0] = new_data

                volume = float(np.sqrt(np.mean(new_data**2)))
                await websocket.send(json.dumps({"type": "volume", "value": volume}))

                # --- SILENCE HANDLING (with grace period) ---
                if volume < MIN_VOLUME:
                    silence_grace_count += 1
                    if silence_grace_count >= SILENCE_GRACE_FRAMES and active_notes:
                        now = time.time()
                        for midi_num, start in active_notes.items():
                            rel_start = start - session_start_time
                            dur = now - start
                            note_data = {"note": midi_to_note_name(midi_num), "midi": midi_num, "start_time": round(rel_start, 3), "duration": round(dur, 3)}
                            recorded_song.append(note_data)
                            await websocket.send(json.dumps({"type": "note_off", **note_data}))
                        active_notes = {}
                        await websocket.send(json.dumps({"type": "silence_reset"}))
                    continue
                else:
                    silence_grace_count = 0

                # --- AI PROCESSING ---
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(None, lambda: model.predict(audio_buffer))

                note_probs = output['note']
                onset_probs = output['onset']
                if note_probs is None: continue

                focus = 5
                current_notes_max = np.max(note_probs[0, -focus:, :], axis=0)
                current_onsets_max = np.max(onset_probs[0, -focus:, :], axis=0)

                # --- SUPPRESSION LOGIC (Iterate High -> Low) ---
                for i in range(87, 24, -1):
                    prob = current_notes_max[i]
                    if prob < 0.1: continue

                    idx_below = i - 12
                    if idx_below >= 0:
                        prob_below = current_notes_max[idx_below]
                        if prob_below > 0.5 and prob < prob_below:
                            current_notes_max[i] = 0.0
                            continue

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
                                duration = now - old_start
                                rel_start = old_start - session_start_time
                                note_data = {
                                    "note": midi_to_note_name(midi_num),
                                    "midi": midi_num,
                                    "start_time": round(rel_start, 3),
                                    "duration": round(duration, 3)
                                }
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
                                if session_start_time is None: session_start_time = now
                                active_notes[midi_num] = now
                                await websocket.send(json.dumps({
                                    "type": "note_on",
                                    "note": midi_to_note_name(midi_num),
                                    "midi": midi_num,
                                    "event": "new_attack",
                                    "start_time": round(now - session_start_time, 3)
                                }))

                # --- CLEANUP ---
                for midi_num in list(active_notes.keys()):
                    if midi_num not in detected_this_frame:
                        start_time = active_notes[midi_num]
                        duration = now - start_time
                        rel_start = start_time - session_start_time
                        note_info = {
                            "note": midi_to_note_name(midi_num),
                            "midi": midi_num,
                            "start_time": round(rel_start, 3),
                            "duration": round(duration, 3)
                        }
                        recorded_song.append(note_info)
                        del active_notes[midi_num]
                        await websocket.send(json.dumps({"type": "note_off", **note_info}))

            except websockets.exceptions.ConnectionClosed:
                raise
            except Exception as e:
                print(f"[audio] Frame processing error (frame {frame_count}): {e}")
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"[audio] Connection closed (code={e.code}). Frames processed: {frame_count}, notes recorded: {len(recorded_song)}")
    except Exception as e:
        print(f"[audio] FATAL ERROR after {frame_count} frames: {e}")
        traceback.print_exc()

async def main():
    print("Server running on localhost:8000")
    async with websockets.serve(audio_handler, "0.0.0.0", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
