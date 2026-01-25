# run this to test before commit


import asyncio
import websockets
import json
import time
import numpy as np
from basic_pitch.inference import Model, ICASSP_2022_MODEL_PATH

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
HOP_SIZE = 2048         
WINDOW_LENGTH = 43844   

NOTE_THRESHOLD = 0.4
ONSET_THRESHOLD = 0.5
MIN_VOLUME = 0.01

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
    active_notes = {} 
    
    # --- RECORDING STATE ---
    # We initialize as None so we can wait for the first note
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
            
            # --- 1. SILENCE HANDLER (Save notes before resetting) ---
            if volume < MIN_VOLUME:
                if active_notes:
                    print("Silence detected - closing active notes...")
                    current_time = time.time()
                    
                    for midi_num, abs_start_time in active_notes.items():
                        duration = current_time - abs_start_time
                        
                        # Calculate relative start time (0.0 if it was the first note)
                        rel_start = abs_start_time - session_start_time if session_start_time else 0.0
                        
                        note_name = midi_to_note_name(midi_num)
                        
                        completed_note = {
                            "note": note_name,
                            "midi": midi_num,
                            "start_time": round(rel_start, 3),
                            "duration": round(duration, 3)
                        }
                        
                        recorded_song.append(completed_note)
                        
                        await websocket.send(json.dumps({
                            "type": "note_off", 
                            **completed_note
                        }))
                    
                    active_notes = {}
                    await websocket.send(json.dumps({"type": "silence_reset"}))
                continue

            # --- AI PROCESSING ---
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, lambda: model.predict(audio_buffer))
            
            note_probs = output['note']
            onset_probs = output['onset']
            if note_probs is None: continue

            focus_window = 5
            current_notes_max = np.max(note_probs[0, -focus_window:, :], axis=0)
            current_onsets_max = np.max(onset_probs[0, -focus_window:, :], axis=0)
            
            detected_this_frame = set()

            for i in range(88):
                midi_num = i + 21
                prob_note = current_notes_max[i]
                prob_onset = current_onsets_max[i]
                
                is_sustaining = prob_note > NOTE_THRESHOLD
                is_attack = prob_onset > ONSET_THRESHOLD

                if is_sustaining:
                    detected_this_frame.add(midi_num)
                    note_name = midi_to_note_name(midi_num)
                    
                    # --- 2. START / RE-TRIGGER LOGIC ---
                    if midi_num in active_notes and is_attack:
                        if (time.time() - active_notes[midi_num]) > 0.1:
                            
                            # A. Close OLD note
                            old_abs_start = active_notes[midi_num]
                            old_duration = time.time() - old_abs_start
                            old_rel_start = old_abs_start - session_start_time # Safe bc active_notes exists
                            
                            recorded_song.append({
                                "note": note_name,
                                "midi": midi_num,
                                "start_time": round(old_rel_start, 3),
                                "duration": round(old_duration, 3)
                            })

                            # B. Start NEW note
                            active_notes[midi_num] = time.time()
                            new_rel_start = time.time() - session_start_time
                            
                            await websocket.send(json.dumps({
                                "type": "note_on", 
                                "note": note_name, 
                                "midi": midi_num, 
                                "event": "re_trigger",
                                "start_time": round(new_rel_start, 3)
                            }))
                    
                    elif midi_num not in active_notes:
                        # --- ANCHOR TIME TO FIRST NOTE ---
                        current_time = time.time()
                        if session_start_time is None:
                            print("First note detected! Setting T=0.")
                            session_start_time = current_time
                        
                        rel_start = current_time - session_start_time
                        # ---------------------------------

                        active_notes[midi_num] = current_time
                        
                        await websocket.send(json.dumps({
                            "type": "note_on", 
                            "note": note_name, 
                            "midi": midi_num, 
                            "event": "new_attack",
                            "start_time": round(rel_start, 3)
                        }))

            # --- 3. NATURAL NOTE OFF ---
            for midi_num in list(active_notes.keys()):
                if midi_num not in detected_this_frame:
                    abs_start_time = active_notes[midi_num]
                    duration = time.time() - abs_start_time
                    rel_start = abs_start_time - session_start_time
                    
                    note_name = midi_to_note_name(midi_num)
                    
                    completed_note = {
                        "note": note_name,
                        "midi": midi_num,
                        "start_time": round(rel_start, 3),
                        "duration": round(duration, 3)
                    }
                    
                    recorded_song.append(completed_note)
                    del active_notes[midi_num]
                    
                    await websocket.send(json.dumps({
                        "type": "note_off", 
                        **completed_note
                    }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected. Recorded {len(recorded_song)} notes.")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    print("Server listening on 8000...")
    async with websockets.serve(audio_handler, "localhost", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())