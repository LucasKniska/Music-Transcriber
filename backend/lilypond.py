import asyncio
import subprocess
import os
import uuid
import sys
import traceback
import re

def parse_vexflow_duration(duration_str):
    """
    Converts VexFlow duration codes (w, h, q, 8, 16) to LilyPond numbers (1, 2, 4, 8, 16).
    Handles dots (e.g., 'qd' -> '4.').
    """
    clean_dur = duration_str.lower().replace('r', '') # Remove 'r' rest indicator for calculation
    is_dotted = 'd' in clean_dur
    base = clean_dur.replace('d', '')
    
    mapping = {
        'w': '1',
        'h': '2',
        'q': '4',
        '8': '8',
        '16': '16',
        '32': '32'
    }
    
    lily_dur = mapping.get(base, '4') # Default to quarter if unknown
    if is_dotted:
        lily_dur += "."
        
    return lily_dur

def parse_vexflow_pitch(vex_key):
    """
    Converts VexFlow key "c#/4" to LilyPond "cis'".
    LilyPond Absolute Octaves:
    c   = C3 (Low C)
    c'  = C4 (Middle C)
    c'' = C5 (High C)
    """
    if '/' not in vex_key:
        return "c'" # Fallback

    note_part, octave_part = vex_key.split('/')
    octave = int(octave_part)

    # 1. Handle Accidental (f# -> fis, eb -> ees)
    pitch = note_part.lower()
    if '#' in pitch:
        pitch = pitch.replace('#', 'is')
    elif 'b' in pitch:
        pitch = pitch.replace('b', 'es')
    
    # 2. Handle Octave
    # Base 'c' in LilyPond is C3.
    # C4 (Middle C) needs one apostrophe (').
    # C5 needs two ('').
    suffix = ""
    if octave == 4:
        suffix = "'"
    elif octave == 5:
        suffix = "''"
    elif octave == 6:
        suffix = "'''"
    elif octave == 3:
        suffix = ""     # c is C3
    elif octave == 2:
        suffix = ","    # lower

    return f"{pitch}{suffix}"

def edit_notes(notes):
    """
    Parses list of Note objects into a LilyPond string.
    """
    lily_string = ""
    
    for note in notes:
        # Determine if it's a dict (raw json) or Pydantic model
        if isinstance(note, dict):
            keys = note.get('keys', [])
            duration = note.get('duration', 'q')
            is_rest = note.get('isRest', False)
        else:
            keys = note.keys
            duration = note.duration
            is_rest = note.isRest

        lily_dur = parse_vexflow_duration(duration)

        if is_rest or 'r' in duration:
            # REST: syntax is "r4", "r8"
            lily_string += f" r{lily_dur}"
        elif len(keys) > 1:
            # CHORD: syntax is <c' e' g'>4
            pitches = [parse_vexflow_pitch(k) for k in keys]
            chord_str = " ".join(pitches)
            lily_string += f" <{chord_str}>{lily_dur}"
        elif len(keys) == 1:
            # SINGLE NOTE: syntax is c'4
            pitch = parse_vexflow_pitch(keys[0])
            lily_string += f" {pitch}{lily_dur}"
            
    return lily_string.strip()

async def convert_to_lilypond(notes):
    unique_id = str(uuid.uuid4())
    base_filename = f"temp_{unique_id}"
    ly_filename = f"{base_filename}.ly"
    pdf_filename = f"{base_filename}.pdf"

    # Convert notes to string
    music_notes = edit_notes(notes)
    
    # Melodic Template (Treble Clef, C Major, 4/4)
    lilypond_content = f"""
\\version "2.24.0"
\\score {{
  \\new Staff {{
    \\clef treble
    \\time 4/4
    \\key c \\major
    \\absolute {{
        {music_notes}
    }}
  }}
  \\layout {{ }}
}}
"""

    try:
        # Write the .ly file
        with open(ly_filename, "w") as f:
            f.write(lilypond_content)

        # Execute LilyPond
        # Windows/Linux compatibility check
        cmd = ["lilypond", "--output", base_filename, ly_filename]
        
        if sys.platform == "win32":
            # Run synchronously on Windows to prevent event loop issues
            process = await asyncio.to_thread(
                subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            returncode = process.returncode
            stderr = process.stderr
        else:
            # Run asynchronously on Linux/Mac
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            returncode = process.returncode

        if returncode != 0:
            error_msg = stderr.decode()
            print(f"LILYPOND ERROR:\n{error_msg}")
            return None, f"LilyPond Error: {error_msg}"

        # Return PDF bytes
        if os.path.exists(pdf_filename):
            with open(pdf_filename, "rb") as f:
                pdf_bytes = f.read()
            return pdf_bytes, None
        else:
            return None, "PDF created but file not found."

    except Exception:
        full_error = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{full_error}")
        return None, f"Server Error: {full_error}"

    finally:
        # Cleanup
        if os.path.exists(ly_filename):
            os.remove(ly_filename)
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        # Lilypond often creates a .midi or .log file too, nice to clean those if they exist
        if os.path.exists(f"{base_filename}.log"):
            os.remove(f"{base_filename}.log")