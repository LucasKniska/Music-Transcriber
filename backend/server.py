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

# --- HYSTERESIS THRESHOLDS ---
ONSET_THRESHOLD = 0.6   
NOTE_START_THRESHOLD = 0.5
NOTE_KEEP_THRESHOLD = 0.25 
MIN_VOLUME = 0.02

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

async def audio_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    
    audio_buffer = np.zeros((1, WINDOW_LENGTH, 1), dtype=np.float32)
    input_accumulator = []
    active_notes = {} # midi_num -> abs_start_time
    
    session_start_time = None
    recorded_song = []

    try:
        async for message in websocket:
            try:
                chunk = np.frombuffer(message, dtype=np.float32)
            except Exception:
                continue
            if len(chunk) == 0: continue

            input_accumulator.extend(chunk)
            if len(input_accumulator) < HOP_SIZE: continue

            new_data = np.array(input_accumulator[:HOP_SIZE], dtype=np.float32)
            input_accumulator = input_accumulator[HOP_SIZE:]

            audio_buffer = np.roll(audio_buffer, -HOP_SIZE, axis=1)
            audio_buffer[0, -HOP_SIZE:, 0] = new_data

            volume = float(np.sqrt(np.mean(new_data**2)))
            await websocket.send(json.dumps({"type": "volume", "value": volume}))
            
            # --- SILENCE HANDLER ---
            if volume < MIN_VOLUME:
                if active_notes:
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

            # --- AI PROCESSING ---
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, lambda: model.predict(audio_buffer))
            
            note_probs = output['note']
            onset_probs = output['onset']
            if note_probs is None: continue

            focus = 8
            current_notes_max = np.max(note_probs[0, -focus:, :], axis=0)
            current_onsets_max = np.max(onset_probs[0, -focus:, :], axis=0)

            # --- BUG FIX 4: SUB-HARMONIC SUPPRESSION ---
            # We iterate high-to-low. If a high note is strong, we squelch its
            # specific lower octaves (sub-harmonics) unless they are also very strong.
            # This prevents a G4 from triggering a ghost G2.
            for i in range(87, 24, -1): # Check all notes down to C3
                prob_high = current_notes_max[i]

                if prob_high > 0.5: # If we definitely have a high note
                    # Check 1 Octave down (i-12) and 2 Octaves down (i-24)
                    for offset in [12, 24]:
                        low_idx = i - offset
                        if low_idx >= 0:
                            prob_low = current_notes_max[low_idx]
                            
                            # If the low note is just a "shadow" (weaker than the high note), kill it.
                            # We use a 0.8 factor: if real G2 was playing, it should be loud on its own.
                            if prob_low < (prob_high * 0.9): 
                                current_notes_max[low_idx] = 0.0

            
            now = time.time()
            detected_this_frame = set()

            for i in range(88):
                midi_num = i + 21
                prob_note = current_notes_max[i]
                prob_onset = current_onsets_max[i]
                
                # --- DYNAMIC LOW-FREQUENCY BIAS ---
                start_thresh = NOTE_START_THRESHOLD
                onset_thresh = ONSET_THRESHOLD
                
                # If note is below C3 (MIDI 48), lower the threshold by 30%
                if midi_num < 48:
                    start_thresh *= 0.7
                    onset_thresh *= 0.7

                # Check thresholds using Hysteresis
                is_active = midi_num in active_notes
                thresh = NOTE_KEEP_THRESHOLD if is_active else start_thresh
                
                is_sustaining = prob_note > thresh
                is_attack = prob_onset > onset_thresh

                if is_sustaining:
                    detected_this_frame.add(midi_num)
                    
                    if is_active:
                        if is_attack and (now - active_notes[midi_num]) > RETRIGGER_COOLDOWN:
                            old_start = active_notes[midi_num]
                            recorded_song.append({"note": midi_to_note_name(midi_num), "midi": midi_num, "start_time": round(old_start - session_start_time, 3), "duration": round(now - old_start, 3)})
                            active_notes[midi_num] = now
                            await websocket.send(json.dumps({
                                "type": "note_on", "note": midi_to_note_name(midi_num), "midi": midi_num,
                                "event": "re_trigger", "start_time": round(now - session_start_time, 3)
                            }))
                    else:
                        if session_start_time is None: session_start_time = now
                        active_notes[midi_num] = now
                        await websocket.send(json.dumps({
                            "type": "note_on", "note": midi_to_note_name(midi_num), "midi": midi_num,
                            "event": "new_attack", "start_time": round(now - session_start_time, 3)
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
        print(f"Connection closed. Notes recorded: {len(recorded_song)}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    print("Server running on localhost:8000")
    async with websockets.serve(audio_handler, "localhost", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())